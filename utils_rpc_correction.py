import numpy as np
import os
import sys
import glob
import json
import ast
import rpcm
import rasterio
import subprocess
import dsmr
import shutil

import config


# List of PAN images to ignore for each city, this images are badly localized
IMAGES_TO_IGNORE = {
    "JAX": ["6", "8", "9"],
    "UCSD": ["30", "31", "7"],
    "OMA": ["12"]
}

def aoi_from_json(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d["geojson"]

def rpc_from_json(json_path, return_dict=False):
    with open(json_path) as f:
        d = json.load(f)
    if return_dict:
        return d["rpc"]
    return rpcm.RPCModel(d["rpc"], dict_format="rpcm")

def run_ba(dataset_path, aoi_id, output_dir=None, save_tmp_files=False, overwrite=False, max_keypoints=None, dem_file=None):

    from bundle_adjust.cam_utils import SatelliteImage
    from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline
    from bundle_adjust import loader

    img_dir = os.path.join(dataset_path, "crops", aoi_id)
    in_rpc_dir = os.path.join(dataset_path, "root", aoi_id)
    if output_dir is None:
        out_rpc_dir = os.path.join(dataset_path, "root_dir_ba", aoi_id)
        ba_files_dir = os.path.join(dataset_path, "ba_files", aoi_id)
    else:
        out_rpc_dir = os.path.join(output_dir, "root_dir_ba", aoi_id)
        ba_files_dir = os.path.join(output_dir, "ba_files", aoi_id)


    # overwrite if needed
    if os.path.exists(out_rpc_dir) and not overwrite:
        print("Output_dir already exists! Skipping!\n")
        return None
    else:
        os.makedirs(out_rpc_dir, exist_ok=True)

    # load input data
    myimages = sorted(glob.glob(img_dir + "/*.tif"))
    # Ignore badly localized images
    myimages = [im for im in myimages if os.path.basename(im).split("_")[-2].split(".")[0] not in IMAGES_TO_IGNORE.get(aoi_id.split("_")[0], [])]
    myjsons = [os.path.join(in_rpc_dir, os.path.basename(p).replace(".tif", ".json")) for p in myimages]
    for p1, p2 in zip(myjsons, myimages):
        assert os.path.exists(p1) and os.path.exists(p2)
    #myrpcs = [rpcm.rpc_from_geotiff(p) for p in mypanimages]
    myrpcs = [rpc_from_json(p) for p in myjsons]
    input_images = [SatelliteImage(fn, rpc) for fn, rpc in zip(myimages, myrpcs)]
    ba_input_data = {}
    ba_input_data['in_dir'] = img_dir
    ba_input_data['out_dir'] = ba_files_dir
    ba_input_data['images'] = input_images

    print('Input data successfully set!\n')

    # redirect all prints to a bundle adjustment logfile inside the output directory
    os.makedirs(ba_input_data['out_dir'], exist_ok=True)
    path_to_log_file = "{}/bundle_adjust.log".format(ba_input_data['out_dir'])
    print("Running bundle adjustment for RPC model refinement ...")
    print(f"Path to log file: {path_to_log_file}")
    log_file = open(path_to_log_file, "w+")
    try:
        sys.stdout = log_file
        sys.stderr = log_file
        # run bundle adjustment
        #tracks_config = {'FT_reset': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based', "FT_K": 300}
        # tracks_config = {'FT_reset': False, 'FT_save': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based', "FT_kp_max": 20000, "FT_n_proc": 12}
        if max_keypoints is None:
            tracks_config = {'FT_reset': False, 'FT_save': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based', "FT_n_proc": 12}
        else:
            tracks_config = {'FT_reset': False, 'FT_save': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based', "FT_kp_max": max_keypoints, "FT_n_proc": 12}
        ba_extra = {"cam_model": "rpc", "dem_file": dem_file}
        ba_pipeline = BundleAdjustmentPipeline(ba_input_data, tracks_config=tracks_config, extra_ba_config=ba_extra)
        ba_pipeline.run()
    finally:
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__
        log_file.close()
    print("... done !")
    print(f"Path to bundle adjustment files: {ba_input_data['out_dir']}")


    # save all bundle adjustment parameters in a temporary directory if needed for debugging
    if save_tmp_files:
        ba_params_dir = os.path.join(ba_pipeline.out_dir, "ba_params")
        os.makedirs(ba_params_dir, exist_ok=True)
        np.save(os.path.join(ba_params_dir, "pts_ind.npy"), ba_pipeline.ba_params.pts_ind)
        np.save(os.path.join(ba_params_dir, "cam_ind.npy"), ba_pipeline.ba_params.cam_ind)
        np.save(os.path.join(ba_params_dir, "pts3d.npy"), ba_pipeline.ba_params.pts3d_ba - ba_pipeline.global_transform)
        np.save(os.path.join(ba_params_dir, "pts2d.npy"), ba_pipeline.ba_params.pts2d)
        fnames_in_use = [ba_pipeline.images[idx].geotiff_path for idx in ba_pipeline.ba_params.cam_prev_indices]
        loader.save_list_of_paths(os.path.join(ba_params_dir, "geotiff_paths.txt"), fnames_in_use)

    for in_json_path in myjsons:
        with open(in_json_path) as f:
            d = json.load(f)
        rpc_adj_path = os.path.join(ba_files_dir, f"rpcs_adj/{os.path.basename(in_json_path).replace('.json', '.rpc_adj')}")
        d["rpc"] = rpcm.rpc_from_rpc_file(rpc_adj_path).__dict__
        out_json_path = os.path.join(out_rpc_dir, os.path.basename(in_json_path))
        os.makedirs(out_rpc_dir, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(d, f, indent=2)
    print(f"All done! Path to bundle-adjusted root_dir: {out_rpc_dir}")


def run_s2p(img_path1, img_path2, json_path1, json_path2, s2p_out_dir, dsm_res):
    for p in [img_path1, img_path2, json_path1, json_path2]:
        assert os.path.exists(p)
    
    # define s2p config
    s2p_config = {
        "images": [
            {"img": img_path1, "rpc": rpc_from_json(json_path1, return_dict=True)},
            {"img": img_path2, "rpc": rpc_from_json(json_path2, return_dict=True)},
        ],
        "out_dir": s2p_out_dir,
        "dsm_resolution": dsm_res,
        "rectification_method": "sift",
        "matching_algorithm": "mgm_multi",
        "roi_geojson": aoi_from_json(json_path1),
        "disp_range_method": "exogenous",
    }
    # ROGER: I had trouble with the disparity range in old s2p versions, had to set "disp_range_method": "exogenous"
    
    # write s2p config to disk
    os.makedirs(s2p_out_dir, exist_ok=True)
    config_path = os.path.join(s2p_out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(s2p_config, f, indent=2)
    
    # run s2p and redirect output to log file
    log_file = os.path.join(s2p_out_dir, 'log.txt')
    out_dsm_path = os.path.join(s2p_out_dir, 'dsm.tif')
    if not os.path.exists(out_dsm_path):
        print(f"Running s2p for {img_path1} and {img_path2} ...")
        with open(log_file, 'w') as outfile:
            subprocess.run(['s2p', config_path], stdout=outfile, stderr=outfile, check=True)
    assert os.path.exists(out_dsm_path)
    print(f"... done! Output dsm: {out_dsm_path}")



def crop_dsm(ref_dsm_path, in_dsm_path, out_dsm_path):
    """
    Crop and reproject a DSM to match a reference DSM's extent and CRS.
    
    Args:
        ref_dsm_path (str): Path to reference DSM that defines the target extent and CRS
        in_dsm_path (str): Path to input DSM to be cropped/reprojected
        out_dsm_path (str): Path where the resulting DSM will be saved
    """
    from osgeo import gdal
    import rasterio
    import os

    # Create temporary file for reprojected DSM if needed
    temp_reprojected = None
    
    try:
        # Open reference and input DSMs to check CRS
        ref_ds = gdal.Open(ref_dsm_path)
        in_ds = gdal.Open(in_dsm_path)
        
        ref_crs = ref_ds.GetProjection()
        in_crs = in_ds.GetProjection()
        
        input_path = in_dsm_path
        
        # If CRS different, reproject input to match reference
        if ref_crs != in_crs:
            temp_reprojected = os.path.join(os.path.dirname(out_dsm_path), 'temp_reprojected.tif')
            gdal.Warp(temp_reprojected, in_ds, 
                     dstSRS=ref_crs,
                     resampleAlg=gdal.GRA_Bilinear)
            input_path = temp_reprojected
        
        # Get bounds from reference
        with rasterio.open(ref_dsm_path) as src:
            xoff, yoff = src.bounds.left, src.bounds.bottom
            xsize, ysize = src.width, src.height
            resolution = src.res[0]
        
        # Define projwin for gdal translate
        ulx = xoff
        uly = yoff + ysize * resolution
        lrx = xoff + xsize * resolution
        lry = yoff
        
        # Crop to reference extent
        ds = gdal.Translate(out_dsm_path, input_path, 
                          options=f"-projwin {ulx} {uly} {lrx} {lry} -tr {resolution} {resolution}")
        ds = None
        
        assert os.path.exists(out_dsm_path), "Output file was not created"
        
    finally:
        # Clean up temporary file if it was created
        if temp_reprojected and os.path.exists(temp_reprojected):
            os.remove(temp_reprojected)


def remove_nonvalid_vals_in_gt_dsm(in_dsm_path, out_dsm_path):
    with rasterio.open(in_dsm_path, "r") as src:
        profile = src.profile
        dsm = src.read()[0, :, :]
        dsm[dsm==-9999.0] = np.nan
    with rasterio.open(out_dsm_path, "w", **profile) as dst:
        dst.write(dsm, 1)
    assert(os.path.exists(out_dsm_path))

def save_rdsm_transform_txt(transform, transform_path):
    with open(transform_path, 'w') as f:
        f.write(str(transform))
    assert os.path.exists(transform_path)

def load_rdsm_transform_txt(transform_path):
    with open(transform_path, 'r') as f:
        transform = ast.literal_eval(f.read())
    return transform

def interpolate_s2p_dsm_imscript(input_s2p_dsm, out_dsm_path, remove_tmp_files=True):
    bin_dir = config.get_imscript_bin_dir()
    assert os.path.exists(input_s2p_dsm)
    assert os.path.exists(bin_dir), "imscript not found!"
    input_dir = os.path.dirname(input_s2p_dsm)
    # small hole interpolation by closing
    dsm = input_s2p_dsm
    cdsm = input_dir+'/cdsm.tif'  # dsm after closing
    os.system(f'{bin_dir}/morsi square closing {dsm} | {bin_dir}/plambda {dsm} - "x isfinite x y isfinite y nan if if" -o {cdsm}')
    assert os.path.exists(cdsm)
    # larger holes with min interpolation
    mcdsm = input_dir+'/mcdsm.tif'  # dsm after closing and min interpolation
    os.system(f'{bin_dir}/bdint5pc -a min {dsm} {mcdsm}')
    assert os.path.exists(mcdsm)
    os.makedirs(os.path.dirname(out_dsm_path), exist_ok=True)
    shutil.copyfile(mcdsm, out_dsm_path)
    os.remove(cdsm)
    os.remove(mcdsm)
    copy_dsm_rasterio_profile(input_s2p_dsm, out_dsm_path)
    assert os.path.exists(out_dsm_path)

    
def copy_dsm_rasterio_profile(src_dsm_path, dst_dsm_path):
    assert os.path.exists(src_dsm_path)
    assert os.path.exists(dst_dsm_path)
    with rasterio.open(src_dsm_path, "r") as src:
        profile = src.profile
    with rasterio.open(dst_dsm_path, "r") as src:
        dsm = src.read()[0, :, :]
    with rasterio.open(dst_dsm_path, "w", **profile) as dst:
        dst.write(dsm, 1)
    
def align_gt_dsm(s2p_dsm_path, gt_dsm_path, gt_rdsm_path, interpolate_s2p=False):
    assert os.path.exists(s2p_dsm_path) and os.path.exists(gt_dsm_path)
    # (1) remove non valid vals from gt dsm
    gt_dsm_tmppath = gt_rdsm_path.replace(".tif", "_tmp.tif")
    os.makedirs(os.path.dirname(gt_rdsm_path), exist_ok=True)
    remove_nonvalid_vals_in_gt_dsm(gt_dsm_path, gt_dsm_tmppath)
    # (2) ensure s2p dsm has the same exact geographic boundaries as the gt dsm
    s2p_dsm_tmppath = s2p_dsm_path.replace(".tif", "_tmp.tif")
    crop_dsm(gt_dsm_tmppath, s2p_dsm_path, s2p_dsm_tmppath)
    # (3) interpolate s2p dsm if necessary
    if interpolate_s2p:
        s2p_dsm_tmppath2 = s2p_dsm_path.replace(".tif", "_tmp2.tif")
        interpolate_s2p_dsm_imscript(s2p_dsm_tmppath, s2p_dsm_tmppath2)
        s2p_dsm_tmppath = s2p_dsm_tmppath2
    # (4) compute dsmr shift
    transform = dsmr.compute_shift(s2p_dsm_tmppath, gt_dsm_tmppath, scaling=False)
    dsmr.apply_shift(gt_dsm_tmppath, gt_rdsm_path, *transform)
    # save output files
    transform_path = gt_rdsm_path.replace(".tif", "_transform.txt")
    save_rdsm_transform_txt(transform, transform_path)
    copy_dsm_rasterio_profile(gt_dsm_path, gt_rdsm_path)
    print(f"Done! The aligned gt dsm can be found in {gt_rdsm_path}")


def align_dsm_using_precomputed_transform(s2p_dsm_path, gt_dsm_path, gt_rdsm_path, transform_path, interpolate_s2p=False):
    assert os.path.exists(s2p_dsm_path) and os.path.exists(gt_dsm_path) and os.path.exists(transform_path)
    # (1) remove non valid vals from gt dsm
    gt_dsm_tmppath = gt_rdsm_path.replace(".tif", "_tmp.tif")
    os.makedirs(os.path.dirname(gt_rdsm_path), exist_ok=True)
    remove_nonvalid_vals_in_gt_dsm(gt_dsm_path, gt_dsm_tmppath)
    # (2) ensure s2p dsm has the same exact geographic boundaries as the gt dsm
    s2p_dsm_tmppath = s2p_dsm_path.replace(".tif", "_tmp.tif")
    crop_dsm(gt_dsm_tmppath, s2p_dsm_path, s2p_dsm_tmppath)
    # (3) interpolate s2p dsm if necessary
    if interpolate_s2p:
        s2p_dsm_tmppath2 = s2p_dsm_path.replace(".tif", "_tmp2.tif")
        interpolate_s2p_dsm_imscript(s2p_dsm_tmppath, s2p_dsm_tmppath2)
        s2p_dsm_tmppath = s2p_dsm_tmppath2
    # (4) apply precomputed dsmr shift
    transform = load_rdsm_transform_txt(transform_path)
    dsmr.apply_shift(gt_dsm_tmppath, gt_rdsm_path, *transform)
    # save output files
    copy_dsm_rasterio_profile(gt_dsm_path, gt_rdsm_path)
    print(f"Done! The aligned gt dsm can be found in {gt_rdsm_path}")


def get_minmax_alt_from_dsm(dsm_path):
    with rasterio.open(dsm_path, "r") as src:
        dsm = src.read()[0, :, :]
        dsm[dsm<=-9000.0] = np.nan
    min_alt, max_alt = np.nanmin(dsm), np.nanmax(dsm)
    return min_alt, max_alt

def update_minmax_alt_in_root_dir(ref_dsm_path, root_dir):

    assert os.path.exists(ref_dsm_path) and os.path.exists(root_dir)
    with rasterio.open(ref_dsm_path, "r") as src:
        profile = src.profile
    min_alt, max_alt = get_minmax_alt_from_dsm(ref_dsm_path)

    myjsons = sorted(glob.glob(os.path.join(root_dir, "*.json")))
    print(f"Updating min_alt max_alt in root_dir {root_dir}...")
    for p in myjsons:
        with open(p) as f:
            d = json.load(f)
        d["min_alt"] = int(min_alt - 1)
        d["max_alt"] = int(max_alt + 1)
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
    print("... done!")

def err_between_two_aligned_dsms(dsm_path1, dsm_path2, err_path=None, min_alt=0, max_alt=55):
    with rasterio.open(dsm_path1, "r") as src:
        dsm1 = src.read()[0, :, :]
        dsm1[dsm1<=-9000.0] = np.nan
    with rasterio.open(dsm_path2, "r") as src:
        dsm2 = src.read()[0, :, :]
        dsm2[dsm2<=-9000.0] = np.nan
        profile = src.profile
    h = min(dsm1.shape[0], dsm2.shape[0])
    w = min(dsm1.shape[1], dsm2.shape[1])
    if min_alt is not None and max_alt is not None:
        dsm1 = np.clip(dsm1, min_alt, max_alt)
        dsm2 = np.clip(dsm2, min_alt, max_alt)
    pointwise_diff = dsm1[:h, :w] - dsm2[:h, :w]        
    if err_path is not None:
        if len(os.path.dirname(err_path)) > 1:
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
        profile["width"] = w
        profile["height"] = h
        with rasterio.open(err_path, "w", **profile) as dst:
            dst.write(pointwise_diff, 1)
        assert os.path.exists(err_path)
    return abs(pointwise_diff[~np.isnan(pointwise_diff)])

def fit_new_rpc_after_dsmr_transform(dsmr_transform_path, json_path, n_samples=10):
    from bundle_adjust import cam_utils, geo_utils, ba_rpcfit
    """
    dsmr_transform = (dx, dy, a, b)
        dx, dy, a, b: shift coefficients to register `dsm_sec` on `dsm_ref`
            `dx` and `dy` are the horizontal shift coefficients
            `a` and `b` are the coefficients of the affine mapping
            `z -> a*z + b` to be applied to the values of `dsm_sec`
    """
    import warnings
    warnings.filterwarnings("ignore")
    dsmr_transform = load_rdsm_transform_txt(dsmr_transform_path)
    dx, dy, a, b = dsmr_transform
    with open(json_path) as f:
        d = json.load(f)
    original_rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")
    
    # define altitude boundaries
    alt_scale = original_rpc.alt_scale
    alt_offset = original_rpc.alt_offset
    min_alt = -1.0 * alt_scale + alt_offset
    max_alt = +1.0 * alt_scale + alt_offset
    alt_range = [min_alt, max_alt, n_samples]

    # define x,y boundaries
    x0, y0, w, h = 0, 0, d["width"], d["height"]
    margin = 100  # margin in image space, in pixel units

    # create a 2D grid of image points covering the input image
    col_range = [x0 - margin, x0 + w + margin, n_samples]
    row_range = [y0 - margin, y0 + h + margin, n_samples]
    cols, lins, alts = cam_utils.generate_point_mesh(col_range, row_range, alt_range)

    # localize the 2D grid at the different altitude values of the altitude range
    lons, lats = original_rpc.localization(cols, lins, alts)
    easts, norths = geo_utils.utm_from_latlon(lats, lons)
    zs = geo_utils.zonestring_from_lonlat(lons[0], lats[0])
    """
    alts = a * alts + b
    easts, norths = easts + dx, norths + dy
    """
    # if dsmr computed the transform to align the GT to the BA rpcs
    # we need the opposite transform to align the BA rpc to the GT
    alts = (alts - b).astype(np.float32)/a   # ---> alts = a * alts + b becomes alts = (alts - b) / a
    easts, norths = easts + dx, norths + dy  # empirically verified that the sign must be positive here
    
    lons, lats = geo_utils.lonlat_from_utm(easts, norths, zs)
    target = np.vstack([cols, lins]).T
    input_locs = np.vstack([lons, lats, alts]).T

    # output RPC is fitted here
    rpc_calib = ba_rpcfit.weighted_lsq(target, input_locs)
    rmse_err = ba_rpcfit.check_errors(rpc_calib, input_locs, target)
    warnings.resetwarnings()
    return rpc_calib, rmse_err

def update_rpcs_after_dsmr(dsmr_transform_path, input_json_dir, output_json_dir):
    json_paths = sorted(glob.glob(input_json_dir + "/*.json"))
    for json_path in json_paths:
        new_rpc, rmse_err = fit_new_rpc_after_dsmr_transform(dsmr_transform_path, json_path, n_samples=10)
        with open(json_path) as f:
            d = json.load(f)
        new_rpc_dict = new_rpc.__dict__
        for key, val in new_rpc_dict.items():
            if isinstance(val, np.ndarray):
                new_rpc_dict[key] = val.tolist()
        d["rpc"] = new_rpc_dict
        out_json_path = os.path.join(output_json_dir, os.path.basename(json_path))
        os.makedirs(output_json_dir, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(d, f, indent=2)
        assert os.path.exists(out_json_path)
    print("everything was a success!")
