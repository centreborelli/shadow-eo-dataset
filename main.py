"""
Pipeline to download all needed files for the project: crops and DSMs
The pipeline is divided in two main steps:
1. Create the tiles for each city (removing the ones that are mostly water)
2. Process the tiles to create the crops and DSMs
"""

import json
import multiprocessing
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pdal
import rasterio
import requests
import scipy.ndimage
# TODO fix later: dem_sticher sometimes gives import error; if you are not using it just comment the following line
from dem_stitcher import stitch_dem
from shapely.geometry import box
from staticmap import Polygon, StaticMap
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Measured.*geometry types are not supported")


import config
import pdal_utils
import utils


def compute_tiles(mbr_gdf: gpd.GeoDataFrame, tile_size: int = 500) -> gpd.GeoDataFrame:
    # Ensure we're working in a projected CRS
    if mbr_gdf.crs.is_geographic:
        raise ValueError("MBR must be in a projected coordinate system")

    mbr = mbr_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = mbr.bounds
    tiles = []
    x_coords = np.arange(minx, maxx + tile_size, tile_size)
    y_coords = np.arange(miny, maxy + tile_size, tile_size)

    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            tile = box(x, y, x + tile_size, y + tile_size)
            tiles.append(tile)

    tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs=mbr_gdf.crs)
    return tiles_gdf


def create_city_tiles(city: str, size: int, output_dir: str):
    print(f"Creating {size}x{size} tiles for {city} in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # Compute the footprint of all the images available for the city considering all the points from which we have at least two images
    city_footprint = gpd.GeoDataFrame(
        geometry=[
            utils.create_footprint(
                config.SAT_FILES[city], config.FOOTPRINT_AGGREGATION_MODE
            )
        ],
        crs=config.DEFAULT_CRS,
    )
    # Create the DEM of the city (useful in other parts of the pipeline)
    dem_bounds = city_footprint.envelope.geometry.bounds.values.tolist()[0]
    # add margin in degrees (since it's in EPSG:4326)
    margin = 0.01  # roughly 1km at equator
    minx, miny, maxx, maxy = dem_bounds
    dem_bounds = [
        minx - margin,  # left
        miny - margin,  # bottom
        maxx + margin,  # right
        maxy + margin,  # top
    ]
    dem, meta = stitch_dem(
        dem_bounds,
        dem_name="glo_30",
        dst_area_or_point="Area",
        dst_ellipsoidal_height=True,
    )
    with rasterio.open(os.path.join(output_dir, f"{city}_dem.tif"), "w", **meta) as ds:
        ds.write(dem, 1)
        ds.update_tags(AREA_OR_POINT="Area")

    # Read the water body mask for the city
    waterbody_gdf = gpd.read_file(config.WATERBODY_FILES[city])
    waterbody_gdf.crs = config.DEFAULT_CRS
    # Convert to UTM because tiles are defined in meters
    city_footprint_utm = city_footprint.to_crs(f"epsg:326{config.UTM_ZONES[city]}")
    waterbody_gdf_utm = waterbody_gdf.to_crs(f"epsg:326{config.UTM_ZONES[city]}")
    # Compute the minimum bounding box of the city footprint
    mbr_utm = city_footprint_utm.envelope
    # Create the tiles
    tiles_utm = compute_tiles(mbr_utm, size)
    # Remove tiles that are mostly water
    # First compute the intersection between the tiles and the water body
    tiles_and_water = gpd.overlay(tiles_utm, waterbody_gdf_utm, how="intersection")
    tiles_and_water["tile_area"] = tiles_and_water.area
    # Water tiles are those that have more than threshold% of their area covered by water
    water_tiles = tiles_and_water[
        tiles_and_water.tile_area > size**2 * config.WATER_AREA_THRESHOLD
    ]
    non_water_tiles = gpd.overlay(tiles_utm, water_tiles, how="difference")
    # Recompute area and discard any tile that is not complete
    non_water_tiles["tile_area"] = non_water_tiles.area
    tiles_utm = non_water_tiles[non_water_tiles.tile_area > size**2 * 0.99]
    # Convert back to default CRS
    tiles = tiles_utm.to_crs(config.DEFAULT_CRS)
    # Remove Z coord from the geometry (tiles are 2D)
    tiles["geometry"] = tiles["geometry"].apply(utils.drop_z)

    # Create a map with the mbr and the tiles and save it
    m = StaticMap(800, 800)
    # Iterater over the tiles and remove the one that do not appear in at least one image
    n_crops = 0
    print(f"Checking which tiles have images for {city}")
    for tile in tqdm(tiles.itertuples(), total=len(tiles)):
        img_files_for_tile = []
        for img_file in config.SAT_FILES[city]:
            z = utils.get_z_from_tile(tile, dem, meta["transform"])
            if utils.tile_in_image(tile, img_file, z):
                img_files_for_tile.append(img_file)
        if len(img_files_for_tile) == 0:
            tiles = tiles.drop(index=tile.Index)
        else:
            tile_coords = list(tile.geometry.exterior.coords)
            m.add_polygon(Polygon(tile_coords, "#66FF0000", "#FF0000", False))
            n_crops += len(img_files_for_tile)

    # Reindex the tiles
    tiles = tiles.reset_index(drop=True)

    map_img = m.render()
    map_img.save(os.path.join(output_dir, f"{city}_tile_map.png"))
    # Save the tiles to disk
    tiles.to_file(os.path.join(output_dir, f"{city}_tiles.shp"))
    print(f"Created {len(tiles)} tiles for {city}")
    print(f"Number of crops: {n_crops}")


def create_all_city_tiles(size: int, output_dir: str):
    for city in config.SAT_FILES.keys():
        out_dir = os.path.join(output_dir, "tiles")
        os.makedirs(out_dir, exist_ok=True)
        create_city_tiles(city, size, out_dir)


def download_tar_files(output_dir: str, city: Optional[str] = None, overwrite: bool = False):
    """Download the MSI metadata tar files required for TOA correction."""

    tar_output_dir = os.path.join(output_dir, "tar")
    os.makedirs(tar_output_dir, exist_ok=True)

    cities = [city] if city is not None else list(config.SAT_FILES.keys())
    for current_city in cities:
        for i, img_file in enumerate(config.SAT_FILES_MSI[current_city]):
            tar_file = img_file.replace(".NTF.tif", ".tar")
            tar_local_path = os.path.join(tar_output_dir, f"{current_city}_{i}.tar")

            if os.path.exists(tar_local_path) and not overwrite:
                continue

            with requests.get(tar_file, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(tar_local_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)


def create_crops_and_dsms_parallel(
    city: str,
    output_dir: str,
    df_3DEP: Optional[gpd.GeoDataFrame] = None,
    max_workers: Optional[int] = None,
):
    """Parallel implementation of crops and DSMs creation with checkpointing"""

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    dsm_output_dir = os.path.join(output_dir, "dsm")
    pan_crops_output_dir = os.path.join(output_dir, "crops")
    msi_crops_output_dir = os.path.join(output_dir, "msi")
    root_output_dir = os.path.join(output_dir, "root")
    os.makedirs(dsm_output_dir, exist_ok=True)
    os.makedirs(pan_crops_output_dir, exist_ok=True)
    os.makedirs(msi_crops_output_dir, exist_ok=True)
    os.makedirs(root_output_dir, exist_ok=True)

    # Comment the following to run the pipeline for ALL the tiles; if not we only run it for the manually curated ones
    tiles_gdf = gpd.read_file(os.path.join(output_dir, "tiles", f"{city}_tiles.shp"))
    # aoi_df = pd.read_csv(os.path.join(output_dir, "curated_aois_v3.csv"))
    # aoi_df = aoi_df[aoi_df['aoi_name'].str.startswith(city)]
    # aoi_df['ix'] = aoi_df['aoi_name'].str.split('_').str[-1].astype(int)
    # tiles_gdf = tiles_gdf.iloc[aoi_df['ix'].values]

    # Read the DEM
    with rasterio.open(os.path.join(output_dir, "tiles", f"{city}_dem.tif")) as src:
        city_dem = {"raster": src.read(1), "transform": src.transform, "crs": src.crs}

    if df_3DEP is None:
        df_3DEP = pdal_utils.get_3dep_data()

    print(
        f"Processing {len(tiles_gdf)} tiles for {city} using {max_workers} workers..."
    )

    # Process DSMs in parallel, using min aggregation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {
            executor.submit(
                process_single_dsm,
                (tile, tile.Index),
                dsm_output_dir,
                df_3DEP,
                city,
                config,
                "min",
            ): tile.Index
            for tile in tiles_gdf.itertuples()
        }

        # Store results for crop processing
        tile_results = {}

        for future in tqdm(
            as_completed(future_to_tile),
            total=len(tiles_gdf),
            desc="Processing DSMs",
        ):
            tile_idx = future_to_tile[future]
            try:
                result = future.result()
                if result[1] is not None:  # If processing was successful
                    tile_results[tile_idx] = result
            except Exception as e:
                print(f"Tile {tile_idx} generated an exception: {str(e)}")

    # Process crops in parallel
    print("Processing PAN crops ...")
    crop_tasks = []

    print(f"Checking which tiles have images for {city}")
    for tile in tqdm(tiles_gdf.itertuples(), total=len(tiles_gdf)):
        if tile.Index not in tile_results:
            continue

        _, max_alt, min_alt = tile_results[tile.Index]
        z = utils.get_z_from_tile(tile, city_dem["raster"], city_dem["transform"])

        for i, img_file in enumerate(config.SAT_FILES[city]):
            if utils.tile_in_image(tile, img_file, z):
                crop_tasks.append(
                    (
                        tile,
                        img_file,
                        i,
                        z,
                        max_alt,
                        min_alt,
                        os.path.join(pan_crops_output_dir, f"{city}_{tile.Index}"),
                        os.path.join(root_output_dir, f"{city}_{tile.Index}"),
                    )
                )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_crop, task, city, "pan")
            for task in crop_tasks
        ]

        for future in tqdm(
            as_completed(futures), total=len(crop_tasks), desc="Processing crops"
        ):
            try:
                result = future.result()
                if result is None:
                    print(f"Failed to process crop")
            except Exception as e:
                print(f"Crop processing generated an exception: {str(e)}")

    print("Processing MSI crops ...")
    crop_tasks = []

    print(f"Checking which tiles have images for {city}")
    for tile in tqdm(tiles_gdf.itertuples(), total=len(tiles_gdf)):
        if tile.Index not in tile_results:
            continue

        _, max_alt, min_alt = tile_results[tile.Index]
        z = utils.get_z_from_tile(tile, city_dem["raster"], city_dem["transform"])

        for i, img_file in enumerate(config.SAT_FILES_MSI[city]):
            if utils.tile_in_image(tile, img_file, z):
                crop_tasks.append(
                    (
                        tile,
                        img_file,
                        i,
                        z,
                        max_alt,
                        min_alt,
                        os.path.join(msi_crops_output_dir, f"{city}_{tile.Index}"),
                        None,
                    )
                )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_crop,
                task,
                city,
                "msi",
            )
            for task in crop_tasks
        ]

        for future in tqdm(
            as_completed(futures), total=len(crop_tasks), desc="Processing MSI crops"
        ):
            try:
                result = future.result()
                if result is None:
                    print(f"Failed to process crop")
            except Exception as e:
                print(f"Crop processing generated an exception: {str(e)}")

def process_single_crop(
    args: Tuple,
    city: str,
    modality: str = "pan",
) -> Optional[str]:
    """Process a single crop with error handling"""
    tile, img_file, i, z, max_alt, min_alt, crops_dir, root_dir = args
    channels = None
    try:
        os.makedirs(crops_dir, exist_ok=True)

        crop_name = f"{city}_{tile.Index}_{i}_{modality}.tif"
        crop_path = os.path.join(crops_dir, crop_name)

        if not os.path.exists(crop_path):
            if modality != "pan":
                # Only download the R,G, B and NIR1 bands for the MSI
                channels = [2, 3, 5, 7]
            utils.crop_geotiff_lonlat_aoi(img_file, crop_path, tile, z, channels)

        if modality == "pan":
            os.makedirs(root_dir, exist_ok=True)
            meta = utils.generate_json_metadata(crop_path, z)
            meta["min_alt"] = min_alt
            meta["max_alt"] = max_alt
            meta_name = os.path.join(root_dir, meta["img"].replace(".tif", ".json"))
            with open(meta_name, "w") as f:
                json.dump(meta, f)

        return crop_path

    except Exception as e:
        print(f"Error processing crop {crop_name}: {str(e)}")
        return None


def fill_nan_cleanly(array, window_size=3, final_smooth_size=3):
    """Fill NaNs using only original values for local medians."""
    result = array.copy()
    nan_mask = np.isnan(array)

    # First pass: compute all local medians using only original values
    local_medians = np.full_like(array, np.nan)
    for i, j in zip(*np.where(nan_mask)):
        i_start = max(0, i - window_size // 2)
        i_end = min(array.shape[0], i + window_size // 2 + 1)
        j_start = max(0, j - window_size // 2)
        j_end = min(array.shape[1], j + window_size // 2 + 1)

        window = array[i_start:i_end, j_start:j_end]
        local_medians[i, j] = np.nanmedian(window)

    # Fill with local medians where available, global median elsewhere
    global_median = np.nanmedian(array)
    result[nan_mask] = np.where(
        np.isnan(local_medians[nan_mask]), global_median, local_medians[nan_mask]
    )

    result = scipy.ndimage.median_filter(result, size=final_smooth_size)
    return result


def process_single_dsm(
    tile_info: Tuple,
    output_dir: str,
    df_3DEP: gpd.GeoDataFrame,
    city: str,
    config: object,
    grid_method: str = "min",
) -> Tuple[int, float, float]:
    """Process a single DSM with error handling and retries, attempting all available data sources"""
    tile, tile_idx = tile_info
    max_alt = -np.inf
    min_alt = np.inf
    successful_process = False  # Tracks if at least one polygon was successful

    try:
        # Create directory if it doesn't exist
        tile_output_dir = os.path.join(output_dir, f"{city}_{tile_idx}")
        os.makedirs(tile_output_dir, exist_ok=True)

        # Check if this tile was already processed
        checkpoint_file = os.path.join(tile_output_dir, "processed.json")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                return tile_idx, checkpoint_data["max_alt"], checkpoint_data["min_alt"]

        # Transform tile to 3DEP CRS (EPSG:3857)
        transformed_tile = (
            gpd.GeoSeries([tile.geometry], crs=config.DEFAULT_CRS)
            .to_crs("epsg:3857")
            .iloc[0]
        )

        # Find intersecting polygons
        intersecting_polys = pdal_utils.find_intersecting_polys(
            df_3DEP, tile.geometry.__geo_interface__
        )

        # TODO: remove these hardcoded values
        # Control which DSM are we going to use for each city
        # By using this we make sure if available we are always using the same (and most recent) data
        GOLD_DSM = {
            "JAX": "FL_Peninsular_FDEM_Duval_2018",
            "OMA": "IA_FullState",
            "UCSD": "CA_SanDiegoQL2_2014",
        }
        # If GOLD_DSM is available, use it
        gold_dsm = GOLD_DSM.get(city)
        if gold_dsm in intersecting_polys.name.values:
            intersecting_polys = intersecting_polys[intersecting_polys.name == gold_dsm]
        elif len(intersecting_polys) > 1:
            intersecting_polys = intersecting_polys[
                -1:
            ]  # Only use the last one; only download the newest data (to avoid downloading old data)
        else:
            intersecting_polys = intersecting_polys

        errors = []  # List to collect errors

        # Try to process each intersecting polygon
        for data in intersecting_polys.itertuples():
            try:
                dsm_pipeline = pdal_utils.make_DEM_pipeline(
                    extent_epsg3857=transformed_tile,
                    usgs_3dep_dataset_name=[data.name],
                    pc_resolution=config.DSM_PC_RESOLUTION,
                    dem_resolution=config.DSM_GRID_RESOLUTION,
                    filterNoise=True,
                    reclassify=False,
                    savePointCloud=False,
                    outCRS=f"EPSG:326{config.UTM_ZONES[city]}+4326",
                    pc_outName=None,
                    demType="dsm",
                    gridMethod=grid_method,
                    dem_outName=os.path.join(
                        tile_output_dir,
                        f"{city}_{tile_idx}_{data.name}_dsm_{grid_method}",
                    ),
                )

                # Execute the pipeline
                pipeline = pdal.Pipeline(json.dumps(dsm_pipeline))
                pipeline.execute()

                # Process the DSM
                dsm_file = os.path.join(
                    tile_output_dir,
                    f"{city}_{tile_idx}_{data.name}_dsm_{grid_method}.tif",
                )
                with rasterio.open(dsm_file) as src:
                    dsm = src.read(1)
                    dsm_meta = src.meta
                    dsm[dsm == src.nodata] = np.nan
                    dsm = fill_nan_cleanly(dsm)
                    max_alt = max(max_alt, dsm.max())
                    min_alt = min(min_alt, dsm.min())

                # Overwrite the file after processing
                with rasterio.open(dsm_file, "w", **dsm_meta) as dst:
                    dst.write(dsm, 1)

                # Mark that at least one process was successful
                successful_process = True

            except Exception as e:
                error_msg = f"Failed to process DSM for tile {tile_idx}, data {data.name}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)  # Log the error
                continue  # Continue to the next polygon

        # Save checkpoint if any successful process was made
        if successful_process:
            with open(checkpoint_file, "w") as f:
                json.dump(
                    {
                        "tile_idx": tile_idx,
                        "max_alt": float(max_alt),
                        "min_alt": float(min_alt),
                        "status": "completed",
                        "errors": errors,  # Store any errors that occurred
                    },
                    f,
                )
            return tile_idx, max_alt, min_alt
        else:
            # If all attempts failed, raise an exception with all error messages
            raise Exception(
                f"Failed to process any data for tile {tile_idx}. Errors: {'; '.join(errors)}"
            )

    except Exception as e:
        print(f"Error processing tile {tile_idx}: {str(e)}")
        return tile_idx, None, None


def create_all_crops_and_dsms_parallel(
    output_dir: str, max_workers: Optional[int] = None
):
    """Process all cities in parallel with shared 3DEP data"""
    df_3DEP = pdal_utils.get_3dep_data()
    for city in config.SAT_FILES.keys():
        create_crops_and_dsms_parallel(city, output_dir, df_3DEP, max_workers)
    download_tar_files(output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "create_all_city_tiles": create_all_city_tiles,
            "create_all_crops_and_dsms": create_all_crops_and_dsms_parallel,
            "download_tar_files": download_tar_files,
        }
    )
