import json
import os
import subprocess
from multiprocessing import Pool, cpu_count
from time import time

import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from skimage.morphology import remove_small_objects
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - optional GPU acceleration
    torch = None

try:
    from torch_scatter import scatter_max
except ImportError:  # pragma: no cover - optional GPU acceleration
    scatter_max = None

import config
import utils


def remove_small_shadows(binary_image, min_area=100):
    """
    Remove small shadows from a binary image using connected component analysis.

    Parameters:
    binary_image (numpy.ndarray): Binary image where 0 represents shadow (black)
                                 and 1 represents non-shadow (white)
    min_area (int): Minimum area threshold. Shadows smaller than this will be removed

    Returns:
    numpy.ndarray: Processed binary image with small shadows removed
    """
    original_dtype = binary_image.dtype

    if np.nanmin(binary_image) != 0.0 or np.nanmax(binary_image) != 1.0:
        raise ValueError("Shadow image must be binary before filtering small objects")

    inverted = np.logical_not(binary_image)
    cleaned = remove_small_objects(inverted, min_size=int(max(1, np.ceil(min_area))))
    result = np.logical_not(cleaned)
    return result.astype(original_dtype)


def _safe_remove(path):
    if path and os.path.exists(path):
        os.remove(path)


def _get_crop_numbers(dataset_dir: str, aoi: str):
    root_dir = os.path.join(dataset_dir, "root_dir_ba", aoi)
    crop_files = [f for f in os.listdir(root_dir) if f.endswith(".json")]
    return sorted({int(f.split("_")[-2]) for f in crop_files})


def _resolve_max_projection(width, ambiguous_x, ambiguous_y, ambiguous_dsm, ambiguous_shadow):
    if len(ambiguous_x) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    pixel_indices = ambiguous_y * width + ambiguous_x

    if torch is not None and scatter_max is not None and torch.cuda.is_available():
        tensor_index = torch.as_tensor(pixel_indices, device="cuda", dtype=torch.long)
        tensor_dsm = torch.as_tensor(ambiguous_dsm, device="cuda")
        z_buffer, z_buffer_index = scatter_max(tensor_dsm, tensor_index)
        valid = z_buffer_index >= 0
        best_pixels = valid.nonzero(as_tuple=False).flatten().cpu().numpy()
        best_shadow = ambiguous_shadow[z_buffer_index[valid].cpu().numpy()]
        best_dsm = z_buffer[valid].cpu().numpy()
    else:
        order = np.lexsort((-ambiguous_dsm, pixel_indices))
        sorted_pixels = pixel_indices[order]
        sorted_shadow = ambiguous_shadow[order]
        sorted_dsm = ambiguous_dsm[order]
        keep = np.concatenate(([True], sorted_pixels[1:] != sorted_pixels[:-1]))
        best_pixels = sorted_pixels[keep]
        best_shadow = sorted_shadow[keep]
        best_dsm = sorted_dsm[keep]

    best_y, best_x = np.divmod(best_pixels, width)
    return best_x.astype(int), best_y.astype(int), best_shadow, best_dsm


def _cast_shadows_for_crop(aoi: str, dataset_dir: str, crop_number: int):
    city = aoi.split("_")[0]
    shadow_output_dir = os.path.join(dataset_dir, "shadows", "shadow_masks", aoi)
    uncertainty_output_dir = os.path.join(
        dataset_dir, "shadows", "uncertainty_masks", aoi
    )
    output_fname = os.path.join(shadow_output_dir, f"{aoi}_{crop_number}_shadow.png")
    uncertainty_fname = os.path.join(
        uncertainty_output_dir, f"{aoi}_{crop_number}_uncertainty_mask.png"
    )
    pixels_fname = os.path.join(
        uncertainty_output_dir, f"{aoi}_{crop_number}_unseen_pixels_mask.png"
    )
    tree_fname = os.path.join(
        uncertainty_output_dir, f"{aoi}_{crop_number}_tree_mask.png"
    )

    if os.path.exists(output_fname):
        print(f"Shadow already exists for {aoi}_{crop_number}")
        return

    t0 = time()
    tmp_dir = config.get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)

    crop_path = os.path.join(dataset_dir, "crops", aoi, f"{aoi}_{crop_number}_pan.tif")
    msi_path = os.path.join(
        dataset_dir,
        "msi_radiometric_correction",
        aoi,
        f"{aoi}_{crop_number}_msi_aligned.tif",
    )
    root_path = os.path.join(
        dataset_dir, "root_dir_ba", aoi, f"{aoi}_{crop_number}_pan.json"
    )
    dsm_path = utils.get_latest_dsm_path(dataset_dir, aoi, city).replace("dsm", "rdsm")
    if not os.path.exists(dsm_path):
        raise FileNotFoundError(f"Aligned DSM not found for {aoi}")

    with open(root_path) as f:
        json_meta = json.load(f)
    rpc = utils.rpc_from_json(root_path)
    with rasterio.open(crop_path) as src:
        img = src.read().squeeze()

    dsm_upscaled, xy_utm_upscaled, dsm_meta_upscaled = utils.upsample_dsm(
        dsm_path, config.UPSCALE_DSM_FACTOR
    )
    dsm_upscaled = dsm_upscaled / dsm_meta_upscaled["transform"][0]

    upscaled_dsm_path = os.path.join(tmp_dir, f"{aoi}_{crop_number}_upscaled_dsm.tif")
    shadow_dsm_path = os.path.join(tmp_dir, f"{aoi}_{crop_number}_shadow_dsm.tif")

    try:
        with rasterio.open(upscaled_dsm_path, "w", **dsm_meta_upscaled) as dst:
            dst.write(dsm_upscaled, 1)

        original_nans = np.isnan(dsm_upscaled)
        p, q, a = utils.calculate_shadow_params(
            json_meta["sun_elevation"], json_meta["sun_azimuth"]
        )
        subprocess.run(
            [
                config.get_shadow_command(),
                str(p),
                str(q),
                str(a),
                upscaled_dsm_path,
                shadow_dsm_path,
            ],
            check=True,
        )

        dsm_upscaled *= dsm_meta_upscaled["transform"][0]
        with rasterio.open(shadow_dsm_path) as src:
            shadow_dsm = src.read(1)
        shadow_dsm[original_nans] = 1
        shadow_dsm = np.nan_to_num(shadow_dsm)
        shadow_dsm[shadow_dsm != 0] = 1

        transformer = Transformer.from_crs(
            f"epsg:326{config.UTM_ZONES[city]}", "epsg:4326", always_xy=True
        )
        lon, lat = transformer.transform(xy_utm_upscaled[0], xy_utm_upscaled[1])
        x_img, y_img = rpc.projection(
            lon.flatten(), lat.flatten(), dsm_upscaled.flatten()
        )
        x_img = np.round(x_img).flatten().astype(int)
        y_img = np.round(y_img).flatten().astype(int)
        dsm_flat = dsm_upscaled.flatten()
        shadow_dsm_flat = shadow_dsm.flatten()
        valid = (
            (x_img >= 0) & (x_img < img.shape[1]) & (y_img >= 0) & (y_img < img.shape[0])
        )
        x_img = x_img[valid]
        y_img = y_img[valid]
        dsm_flat = dsm_flat[valid]
        shadow_dsm_flat = shadow_dsm_flat[valid]

        index = y_img * img.shape[1] + x_img
        unique_index, unique_counts = np.unique(index, return_counts=True)
        unique_index = unique_index[unique_counts == 1]
        unique_mask = np.isin(index, unique_index)

        shadow_image = np.full_like(img, np.nan, dtype=np.float32)
        unambiguous_x = x_img[unique_mask]
        unambiguous_y = y_img[unique_mask]
        unambiguous_shadow = shadow_dsm_flat[unique_mask]
        shadow_image[unambiguous_y, unambiguous_x] = unambiguous_shadow

        ambiguous_x = x_img[~unique_mask]
        ambiguous_y = y_img[~unique_mask]
        ambiguous_shadow = shadow_dsm_flat[~unique_mask]
        ambiguous_dsm = dsm_flat[~unique_mask]
        best_x, best_y, best_shadow, _ = _resolve_max_projection(
            img.shape[1],
            ambiguous_x,
            ambiguous_y,
            ambiguous_dsm,
            ambiguous_shadow,
        )
        shadow_image[best_y, best_x] = best_shadow

        min_object_height = 3
        sun_elevation_deg = float(json_meta["sun_elevation"])
        meters_per_pixel = 0.31
        pixels_per_meter = 1 / meters_per_pixel
        sun_elevation_rad = np.deg2rad(sun_elevation_deg)
        shadow_length_meters = min_object_height / np.tan(sun_elevation_rad)
        shadow_length_pixels = shadow_length_meters * pixels_per_meter
        min_shadow_area = shadow_length_pixels * (min_object_height * pixels_per_meter)
        nan_mask = np.isnan(shadow_image)
        shadow_image = remove_small_shadows(shadow_image, min_shadow_area)
        shadow_image[nan_mask] = np.nan

        uncertainty_mask_pixels = np.isnan(shadow_image)
        with rasterio.open(msi_path) as src:
            R = src.read(3)
            NIR = src.read(4)
        R = (R / 255).astype(np.float32)
        NIR = (NIR / 255).astype(np.float32)
        NDVI = (NIR - R) / (NIR + R + 1e-6)
        tree_mask = NDVI > 0.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tree_mask = cv2.dilate(tree_mask.astype(np.uint8), kernel, iterations=1)

        shadow_image[uncertainty_mask_pixels] = 1.0
        os.makedirs(shadow_output_dir, exist_ok=True)
        shadow_image = (shadow_image * 255).astype(np.uint8)
        Image.fromarray(shadow_image).save(output_fname, format="PNG")

        os.makedirs(uncertainty_output_dir, exist_ok=True)
        uncertainty_mask_pixels = uncertainty_mask_pixels.astype(bool)
        tree_mask = tree_mask.astype(bool)
        uncertainty_mask = np.logical_or(uncertainty_mask_pixels, tree_mask)
        uncertainty_mask = (uncertainty_mask * 255).astype(np.uint8)
        Image.fromarray(uncertainty_mask).save(uncertainty_fname, format="PNG")
        uncertainty_mask_pixels = (uncertainty_mask_pixels * 255).astype(np.uint8)
        Image.fromarray(uncertainty_mask_pixels).save(pixels_fname, format="PNG")
        tree_mask = (tree_mask * 255).astype(np.uint8)
        Image.fromarray(tree_mask).save(tree_fname, format="PNG")
        print(f"Succesfully casted shadows in {time() - t0:.2f} seconds")
    finally:
        _safe_remove(upscaled_dsm_path)
        _safe_remove(upscaled_dsm_path + ".aux.xml")
        _safe_remove(shadow_dsm_path)


def safe_cast_shadows_single_crop(args):
    try:
        _cast_shadows_for_crop(*args)
    except Exception as e:
        print(f"Error casting shadows in crop {args}: {e}")


def cast_shadows_parallel(dataset_dir: str):
    args_list = []
    aoi_df = pd.read_csv(os.path.join(dataset_dir, "curated_aois_v3.csv"))
    for _, row in aoi_df.iterrows():
        aoi = row["aoi_name"]
        aoi_list = [(aoi, dataset_dir, crop) for crop in _get_crop_numbers(dataset_dir, aoi)]
        add_site = False
        if os.path.exists(os.path.join(dataset_dir, "rdsm", aoi)):
            for f in os.listdir(os.path.join(dataset_dir, "rdsm", aoi)):
                if f.endswith("rdsm.tif"):
                    add_site = True
        if add_site:
            args_list.extend(aoi_list)

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(
            pool.imap_unordered(safe_cast_shadows_single_crop, args_list),
            total=len(args_list),
        ):
            pass


def cast_shadows(aoi: str, dataset_dir: str, crop_number: int = None):
    crop_numbers = [crop_number] if crop_number is not None else _get_crop_numbers(dataset_dir, aoi)
    for current_crop in crop_numbers:
        _cast_shadows_for_crop(aoi, dataset_dir, current_crop)


def cast_all_shadows(dataset_dir: str):
    """
    Cast shadows for all the AOIs in the curated list.

    Args:
        dataset_dir (str): _description_
    """
    aoi_df = pd.read_csv(os.path.join(dataset_dir, "curated_aois_v3.csv"))
    for _, row in tqdm(aoi_df.iterrows(), total=len(aoi_df)):
        aoi = row["aoi_name"]
        try:
            cast_shadows(aoi, dataset_dir)
        except Exception as e:
            print(f"Error casting shadows in aoi {aoi}: {e}")


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "cast_shadows": cast_shadows,
            "cast_all_shadows": cast_all_shadows,
            "cast_shadows_parallel": cast_shadows_parallel,
        }
    )
