import os
from time import time
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

import config
import utils
import utils_rpc_correction


def _output_root(dataset_dir: str, output_dir: Optional[str] = None) -> str:
    return output_dir or dataset_dir


def load_sorted_pairs(file_path: str):
    """
    Loads the sorted pairs of ranked s2p indices 

    Args:
        file_path (str)

    Returns:
        list: A list of tuples representing sorted pairs of ranked s2p indices
    """
    with open(file_path, 'r') as f:
        pairs = [tuple(map(int, line.strip().split(','))) for line in f]
    return pairs


def run_ba_and_align_dsms(
    dataset_dir: str, aoi_csv: str, output_dir: str, max_keypoints: int = 10000
):
    aoi_df = pd.read_csv(aoi_csv)
    for _, row in tqdm(aoi_df.iterrows(), total=len(aoi_df)):
        aoi = row["aoi_name"]
        try:
            run_ba(dataset_dir, aoi, output_dir, max_keypoints)
            run_s2p(dataset_dir, aoi, output_dir)
            align_dsms(dataset_dir, aoi, output_dir)
        except Exception as e:
            print(f"Error in aoi {aoi}: {e}")
        else:
            print(f"Completed {aoi}")


def run_ba(dataset_dir: str, aoi: str, output_dir: str, max_keypoints: int = None):
    os.makedirs(output_dir, exist_ok=True)

    # Get the correct dem file path
    dem_file = os.path.join(dataset_dir, "tiles", f"{aoi.split('_')[0]}_dem.tif")

    utils_rpc_correction.run_ba(
        dataset_dir,
        aoi,
        output_dir,
        max_keypoints=max_keypoints,
        dem_file=dem_file,
    )


def run_s2p(dataset_dir: str, aoi: str, output_dir: Optional[str] = None):
    """
    Run S2P over the top 10 pairs for the given AOI, align all dsms and compute the median
    """
    t0 = time()
    city = aoi.split("_")[0]
    output_root = _output_root(dataset_dir, output_dir)
    s2p_pairs = load_sorted_pairs(f"{dataset_dir}/{city}_s2p_pairs_ranking.txt")
    s2p_pairs = [(f"{aoi}_{pair[0]}_pan.tif", f"{aoi}_{pair[1]}_pan.tif") for pair in s2p_pairs]
    img_path = os.path.join(dataset_dir, "crops", aoi)
    img_list = os.listdir(img_path)
    executed_pairs = 0
    # Step 1 run S2p for top 10 pairs
    while executed_pairs < 10 and len(s2p_pairs) != 0:
        candidate_pair = s2p_pairs.pop(0)
        if candidate_pair[0] in img_list and candidate_pair[1] in img_list:
            img_path1 = os.path.join(img_path, candidate_pair[0])
            img_path2 = os.path.join(img_path, candidate_pair[1])
            img_name1 = os.path.splitext(os.path.basename(img_path1))[0]
            img_name2 = os.path.splitext(os.path.basename(img_path2))[0]
            json_path1 = os.path.join(
                output_root, "root_dir_ba", aoi, f"{img_name1}.json"
            )
            json_path2 = os.path.join(
                output_root, "root_dir_ba", aoi, f"{img_name2}.json"
            )
            s2p_out_dir = (
                f"{output_root}/s2p/{aoi}/{img_name1}_{img_name2}"
            )
            print(f"Running S2P for {img_name1} and {img_name2}")
            try:
                utils_rpc_correction.run_s2p(
                    img_path1, img_path2, json_path1, json_path2, s2p_out_dir, dsm_res = config.DSM_GRID_RESOLUTION
                )
                executed_pairs += 1
            except Exception as e:
                print(f"Error running S2P for {img_name1} and {img_name2}: {e}")
    print(f"S2P took {time() - t0} seconds")
    # Step 2 find the biggest S2P to be used as a reference and resize all to it
    s2p_root = os.path.join(output_root, "s2p", aoi)
    if not os.path.isdir(s2p_root):
        raise RuntimeError(f"No S2P outputs found for {aoi} in {s2p_root}")

    s2p_dirs = os.listdir(s2p_root)
    biggest_s2p_file = None
    biggest_s2p_size = 0
    for s2p_file in s2p_dirs:
        s2p_path = os.path.join(
            s2p_root, s2p_file, "dsm.tif"
        )
        if not os.path.exists(s2p_path):
            continue
        try:
            with rasterio.open(s2p_path) as src:
                s2p_size = src.width * src.height
                if s2p_size > biggest_s2p_size:
                    biggest_s2p_size = s2p_size
                    biggest_s2p_file = s2p_file
        except Exception as e:
            print(f"Error reading s2p {s2p_path}: {e}")
    if biggest_s2p_file is None:
        raise RuntimeError(f"S2P did not produce any valid DSMs for {aoi}")
    ref_s2p_path = os.path.join(
        s2p_root, biggest_s2p_file, "dsm.tif"
    )
    # Iterate over all s2ps and resize them to the biggest one
    for s2p_file in s2p_dirs:
        s2p_path = os.path.join(
            s2p_root, s2p_file, "dsm.tif"
        )
        if not os.path.exists(s2p_path):
            continue
        s2p_out_path = s2p_path.replace("dsm.tif", "dsm_resized.tif")
        try:
            utils_rpc_correction.crop_dsm(
                ref_s2p_path, s2p_path, s2p_out_path
            )
        except Exception as e:
            print(f"Error resizing s2p {s2p_path}: {e}")
    # Step 3 compute the median S2P
    s2ps = []
    for s2p_file in s2p_dirs:
        s2p_path = os.path.join(
            s2p_root, s2p_file, "dsm_resized.tif"
        )
        if not os.path.exists(s2p_path):
            continue
        try:
            with rasterio.open(s2p_path) as src:
                s2ps.append(src.read(1))
        except Exception as e:
            print(f"Error reading s2p {s2p_path}: {e}")
    if not s2ps:
        raise RuntimeError(f"Unable to compute S2P median for {aoi}: no resized DSMs found")
    median_s2p = np.nanmedian(s2ps, axis=0)
    median_s2p_path = os.path.join(s2p_root, "dsm_median.tif")
    # Save it
    with rasterio.open(ref_s2p_path) as src:
        profile = src.profile
        with rasterio.open(median_s2p_path, "w", **profile) as dst:
            dst.write(median_s2p, 1)
    print(f"S2P median took {time() - t0} seconds")


def mask_outliers(dsm_in, dsm_1, dsm_2, outliers_th=5):
    dsm_in = dsm_in.copy()
    dsm_outliers = np.abs((dsm_1 - dsm_2)) > outliers_th
    # 1 values in dsm_outliers mask are outliers and filled with nan
    dsm_in[dsm_outliers] = np.nan
    return dsm_in, dsm_outliers


def align_dsms(dataset_dir: str, aoi: str, output_dir: str):
    t0 = time()
    output_root = _output_root(dataset_dir, output_dir)
    # Define the paths to the data
    dsm_path = utils.get_latest_dsm_path(dataset_dir, aoi, aoi.split("_")[0])
    print(dsm_path)
    rdsm_path = os.path.join(
        output_root, "rdsm", aoi, dsm_path.split("/")[-1].replace("dsm", "rdsm")
    )
    if os.path.exists(rdsm_path):
        print(f"DSM already aligned for {aoi}")
        return
    os.makedirs(os.path.dirname(rdsm_path), exist_ok=True)

    # Crop median S2P to have the same bounds as the original DSM
    median_s2p_path = os.path.join(
        output_root, "s2p", aoi, "dsm_median.tif"
    )
    if not os.path.exists(median_s2p_path):
        print(f"DSM median not found for {aoi}")
        print(f"Skipping alignment for {aoi}")
        return
    median_s2p_path_resized = median_s2p_path.replace("dsm_median.tif", "dsm_median_resized.tif")
    utils_rpc_correction.crop_dsm(
    dsm_path, median_s2p_path, median_s2p_path_resized
    )

    # Load dsm and median s2p
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
    with rasterio.open(median_s2p_path_resized) as src:
        s2p_dsm = src.read(1)

    # Mask both dsms in areas where the difference is too big
    dsm_masked, _ = mask_outliers(dsm, dsm, s2p_dsm)
    s2p_dsm_masked, _ = mask_outliers(s2p_dsm, dsm, s2p_dsm)
    dsm_masked_path = dsm_path.replace("dsm_min.tif", "dsm_min_masked.tif")
    s2p_dsm_masked_path = median_s2p_path_resized.replace("dsm_median_resized.tif", "dsm_median_masked.tif")
    with rasterio.open(dsm_path) as src:
        profile = src.profile
        profile.update(nodata=np.nan)
        with rasterio.open(dsm_masked_path, "w", **profile) as dst:
            dst.write(dsm_masked, 1)
    with rasterio.open(median_s2p_path_resized) as src:
        profile = src.profile
        profile.update(nodata=np.nan)
        with rasterio.open(s2p_dsm_masked_path, "w", **profile) as dst:
            dst.write(s2p_dsm_masked, 1)
    
    # Align masked DSM and re use the transform to align the original dsm
    rdsm_masked_path = rdsm_path.replace("rdsm", "rdsm_masked")
    utils_rpc_correction.align_gt_dsm(
        s2p_dsm_masked_path, dsm_masked_path, rdsm_masked_path
    )
    utils_rpc_correction.align_dsm_using_precomputed_transform(
        s2p_dsm_masked_path, dsm_path, rdsm_path, rdsm_masked_path.replace(".tif", "_transform.txt")
    )
    print(f"DSM alignment took {time() - t0} seconds")


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "run_ba": run_ba,
            "align_dsms": align_dsms,
            "run_ba_and_align_dsms": run_ba_and_align_dsms,
        }
    )
