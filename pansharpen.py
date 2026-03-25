import os
import tarfile

import cv2
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rpcm
import xmltodict
from osgeo_utils import gdal_pansharpen
from tqdm import tqdm

MSI_PIXEL_SIZE = 1.24
PAN_PIXEL_SIZE = 0.31


def get_tar_path_for_msi_crop(dataset_dir: str, msi_filename: str) -> str:
    parts = msi_filename.replace(".tif", "").split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected MSI crop filename format: {msi_filename}")

    city = parts[0]
    source_index = parts[-2]
    tar_path = os.path.join(dataset_dir, "tar", f"{city}_{source_index}.tar")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(
            f"Missing MSI metadata tar file for {msi_filename}: expected {tar_path}"
        )
    return tar_path


def top_of_atmosphere_correction(msi_crop_path, tar_path, out_msi_crop_path):
    print(f"Processing {msi_crop_path}")
    if os.path.exists(out_msi_crop_path):
        return
    os.makedirs(os.path.dirname(out_msi_crop_path), exist_ok=True)
    # Read abscal factor and effective bandwidth from metadata
    with tarfile.open(tar_path) as tar:
        tar_members = tar.getmembers()
        metadata_xml = [
            p.name
            for p in tar_members
            if (".XML" in os.path.basename(p.name))
            and not ("README" in os.path.basename(p.name))
        ][0]
        extracted = tar.extractfile(metadata_xml)
        content = extracted.read()
    metadata = xmltodict.parse(content)

    # Read calibration data
    R_ABSCAL = float(metadata["isd"]["IMD"]["BAND_R"]["ABSCALFACTOR"])
    R_BW = float(metadata["isd"]["IMD"]["BAND_R"]["EFFECTIVEBANDWIDTH"])
    G_ABSCAL = float(metadata["isd"]["IMD"]["BAND_G"]["ABSCALFACTOR"])
    G_BW = float(metadata["isd"]["IMD"]["BAND_G"]["EFFECTIVEBANDWIDTH"])
    B_ABSCAL = float(metadata["isd"]["IMD"]["BAND_B"]["ABSCALFACTOR"])
    B_BW = float(metadata["isd"]["IMD"]["BAND_B"]["EFFECTIVEBANDWIDTH"])
    NIR_ABSCAL = float(metadata["isd"]["IMD"]["BAND_N"]["ABSCALFACTOR"])
    NIR_BW = float(metadata["isd"]["IMD"]["BAND_N"]["EFFECTIVEBANDWIDTH"])

    # Hardcoded gain and offset values
    R_GAIN, R_OFFSET = 0.945, -1.350
    G_GAIN, G_OFFSET = 0.907, -3.287
    B_GAIN, B_OFFSET = 0.905, -4.189
    NIR_GAIN, NIR_OFFSET = 0.982, -3.752  # Updated with provided NIR values

    # Read MSI crop
    with rasterio.open(msi_crop_path, "r") as src:
        B = src.read(1)
        G = src.read(2)
        R = src.read(3)
        NIR = src.read(4)
        profile = src.profile  # Extract profile information
        tags = src.tags()
        rpc_tags = src.tags(ns="RPC")

    # Apply the top of atmosphere correction
    R = R_GAIN * R * (R_ABSCAL / R_BW) + R_OFFSET
    G = G_GAIN * G * (G_ABSCAL / G_BW) + G_OFFSET
    B = B_GAIN * B * (B_ABSCAL / B_BW) + B_OFFSET
    NIR = NIR_GAIN * NIR * (NIR_ABSCAL / NIR_BW) + NIR_OFFSET

    # Stack corrected bands and clip
    corrected_image = np.stack([R, G, B, NIR], axis=0)
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)

    # Update profile for output
    profile.update(
        {
            "driver": "GTiff",
            "count": 4,
            "dtype": corrected_image.dtype,
            "compress": "lzw",
        }
    )

    with rasterio.open(out_msi_crop_path, "w", **profile) as dst:
        dst.write(corrected_image)
        dst.update_tags(**tags)
        if rpc_tags:
            dst.update_tags(ns="RPC", **rpc_tags)


def radiometric_correction(pan_crops_paths, out_pan_crop_path):
    top_percentile = 100
    bottom_percentile = 0
    # Get intensity boundaries from reference image
    reference_in = rasterio.open(pan_crops_paths[0]).read(1)
    max_ref = np.nanpercentile(reference_in, top_percentile)
    min_ref = np.nanpercentile(reference_in, bottom_percentile)
    # Iterate over all the pan images
    for img_path in tqdm(pan_crops_paths):
        out_path = os.path.join(
            out_pan_crop_path, img_path.split("/")[-2], os.path.basename(img_path)
        )
        if os.path.exists(out_path):
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Load image
        with rasterio.open(img_path) as src:
            img = src.read(1)
            profile = src.profile
            tags = src.tags()
            try:
                rpc = rpcm.rpc_from_geotiff(img_path)
            except:
                rpc = None
        # Compute and apply affine correction
        max_i = np.nanpercentile(img, top_percentile)
        min_i = np.nanpercentile(img, bottom_percentile)
        a_i = (max_ref - min_ref) / ((max_i - min_i) + 1e-12)
        b_i = max_ref - a_i * max_i
        img_out = a_i * img + b_i
        img_out = img_out.clip(min_ref, max_ref).astype(img.dtype)
        # Normalize into 0-1 and then convert to 8bit
        img_out = (img_out - min_ref) / (max_ref - min_ref)
        img_out = (img_out / img_out.max() * 255).astype(np.uint8)
        img_out = img_out.clip(0, 255)
        # Write output image
        profile["count"] = 1
        profile["dtype"] = img_out.dtype
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(img_out, 1)
            dst.update_tags(**tags)
            if rpc is not None:
                dst.update_tags(ns="RPC", **rpc.to_geotiff_dict())


# Step 1: Radiometric correction of all pan images
def radiometric_correction_all_pan(dataset_dir: str):
    white_list_df = pd.read_csv(os.path.join(dataset_dir, "curated_aois_v3.csv"))
    white_list = white_list_df["aoi_name"].tolist()
    crops_dir = os.path.join(dataset_dir, "crops")
    all_pan_dirs = os.listdir(crops_dir)
    for city in ["JAX", "OMA", "UCSD"]:
        print(f"Processing city {city}")
        city_pan_dirs = sorted(
            [os.path.join(crops_dir, d) for d in all_pan_dirs if d.startswith(city)]
        )
        city_pan_paths = []
        for pan_dir in city_pan_dirs:
            pan_paths = sorted(
                [
                    os.path.join(pan_dir, f)
                    for f in os.listdir(pan_dir)
                    if f.endswith(".tif") and pan_dir.split("/")[-1] in white_list
                ]
            )
            if len(pan_paths) > 0:
                print(f"AOI: {pan_dir.split('/')[-1]}")
                city_pan_paths.extend(pan_paths)
        radiometric_correction(
            city_pan_paths, os.path.join(dataset_dir, "crops_radiometric_correction")
        )


# Step 2: Top of atmosphere correction of all MSI images
def radiometric_correction_all_msi(dataset_dir: str):
    white_list_df = pd.read_csv(os.path.join(dataset_dir, "curated_aois_v3.csv"))
    white_list = white_list_df["aoi_name"].tolist()
    for subdir in tqdm(os.listdir(os.path.join(dataset_dir, "msi"))):
        if subdir.split(".")[0] not in white_list:
            continue
        for msi in os.listdir(os.path.join(dataset_dir, "msi", subdir)):
            if msi.endswith(".tif"):
                msi_crop_path = os.path.join(dataset_dir, "msi", subdir, msi)
                msi_output_path = os.path.join(
                    dataset_dir, "msi_radiometric_correction", subdir, msi
                )
                tar_path = get_tar_path_for_msi_crop(dataset_dir, msi)
                top_of_atmosphere_correction(msi_crop_path, tar_path, msi_output_path)


# Step 3: Pansharpening of all MSI images given the previously corrected pan images and corrected MSI images
def pansharpen_all(dataset_dir: str):
    white_list_df = pd.read_csv(os.path.join(dataset_dir, "curated_aois_v3.csv"))
    white_list = white_list_df["aoi_name"].tolist()
    for subdir in tqdm(
        os.listdir(os.path.join(dataset_dir, "msi_radiometric_correction"))
    ):
        if subdir.split(".")[0] not in white_list:
            continue
        for msi in os.listdir(
            os.path.join(dataset_dir, "msi_radiometric_correction", subdir)
        ):
            if msi.endswith(".tif"):
                msi_crop_path = os.path.join(
                    dataset_dir, "msi_radiometric_correction", subdir, msi
                )
                pan_crop_path = os.path.join(
                    dataset_dir,
                    "crops_radiometric_correction",
                    subdir,
                    msi.replace("msi", "pan"),
                )
                if os.path.exists(pan_crop_path):
                    out_path = os.path.join(
                        dataset_dir,
                        "pansharpened",
                        subdir,
                        msi.replace("msi.tif", "rgb.tif"),
                    )
                    if not os.path.exists(out_path):
                        try:
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            # First align the images using ORB
                            with rasterio.open(pan_crop_path) as pan_ds, rasterio.open(
                                msi_crop_path
                            ) as msi_ds:
                                pan_img = pan_ds.read(1).astype(np.uint8)
                                msi_img = msi_ds.read()
                                msi_img = np.moveaxis(msi_img, 0, -1)
                                msi_profile = msi_ds.profile.copy()
                            # Convert to grayscale and upscale before alignment
                            msi_gray = cv2.cvtColor(msi_img[:, : , :3], cv2.COLOR_RGB2GRAY)
                            high_res_msi_gray = cv2.resize(
                                msi_gray, pan_img.shape[::-1], interpolation=cv2.INTER_CUBIC
                            )
                            high_res_msi = cv2.resize(
                                msi_img, pan_img.shape[::-1], interpolation=cv2.INTER_CUBIC
                            )
                            # Keypoint matching
                            orb = cv2.ORB_create()
                            kp1, des1 = orb.detectAndCompute(pan_img, None)
                            kp2, des2 = orb.detectAndCompute(high_res_msi_gray, None)
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                            matches = bf.match(des1, des2)
                            dst_pts = np.float32(
                                [kp1[m.queryIdx].pt for m in matches]
                            ).reshape(
                                -1, 1, 2
                            )  # DST is PAN
                            src_pts = np.float32(
                                [kp2[m.trainIdx].pt for m in matches]
                            ).reshape(
                                -1, 1, 2
                            )  # SRC is MSI
                            # Compute homography
                            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                            # Warp MSI to PAN
                            aligned_msi = cv2.warpPerspective(
                                high_res_msi, H, (pan_img.shape[1], pan_img.shape[0])
                            )
                            aligned_msi = np.moveaxis(aligned_msi, -1, 0)
                            # Update profile with new dimensions
                            msi_profile["height"] = pan_img.shape[0]
                            msi_profile["width"] = pan_img.shape[1]
                            msi_profile["transform"] = pan_ds.transform
                            # Write the aligned MSI
                            aligned_msi_crop_path = msi_crop_path.replace(
                                "msi.tif", "msi_aligned.tif"
                            )
                            with rasterio.open(
                                aligned_msi_crop_path, "w", **msi_profile
                            ) as dst:
                                dst.write(aligned_msi)
                            # Do the pansharpening
                            gdal_pansharpen.gdal_pansharpen(
                                pan_name=pan_crop_path,
                                spectral_names=[aligned_msi_crop_path],
                                dst_filename=out_path,
                                resampling="cubic",
                                band_nums=[1, 2, 3],
                            )
                        except Exception as e:
                            print(f"Error in {msi_crop_path}: {e}")


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "step_1_radiometric_correction": radiometric_correction_all_pan,
            "step_2_radiometric_correction_all_msi": radiometric_correction_all_msi,
            "step_3_pansharpen_all": pansharpen_all,
        }
    )
