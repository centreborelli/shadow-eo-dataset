"""Configuration helpers for the dataset creation pipeline."""

from __future__ import annotations

import functools
import os
import shutil
import tempfile
from collections.abc import Mapping

import utils


DEFAULT_CRS = "epsg:4326"
CITY_FOLDERS = {"JAX": "Jacksonville", "OMA": "Omaha", "UCSD": "UCSD"}
DEFAULT_CORE3D_BASE_URL = "PRIVATE_URL_TO_CORE3D_DATA"
DEFAULT_SHADOW_COMMAND = "shadowcast"
DEFAULT_TMP_DIR = os.path.join(tempfile.gettempdir(), "shadow_dataset")

WATERBODY_FILES = {
    "JAX": "data/water_masks/JAX/w082n30n.shp",
    "OMA": "data/water_masks/OMA/w096n41n.shp",
    "UCSD": "data/water_masks/UCSD/w118n32n.shp",
}
UTM_ZONES = {"OMA": 14, "UCSD": 11, "JAX": 17}
FOOTPRINT_AGGREGATION_MODE = "pairwise_intersections"
WATER_AREA_THRESHOLD = (
    0.5  # 50% of the tile must be water to be considered a water tile
)
DSM_GRID_RESOLUTION = 0.5  # meters
DSM_PC_RESOLUTION = 0.1  # 100 points per square meter
UPSCALE_DSM_FACTOR = 4


def _resolve_executable(executable: str, env_var: str, description: str) -> str:
    configured = os.environ.get(env_var, executable)
    if os.path.sep in configured:
        if os.path.exists(configured):
            return configured
        raise RuntimeError(
            f"{description} not found at {configured!r}. "
            f"Set {env_var} to a valid executable path."
        )

    resolved = shutil.which(configured)
    if resolved:
        return resolved

    raise RuntimeError(
        f"{description} executable {configured!r} was not found on PATH. "
        f"Set {env_var} to the executable path."
    )


def get_shadow_command() -> str:
    """Return the configured shadow casting executable."""

    return _resolve_executable(
        DEFAULT_SHADOW_COMMAND,
        "SHADOW_COMMAND",
        "Shadow casting",
    )


def get_imscript_bin_dir() -> str:
    """Return the directory containing imscript binaries."""

    configured = os.environ.get("IMSCRIPT_BIN_DIR")
    if configured:
        if os.path.isdir(configured):
            return configured
        raise RuntimeError(
            f"IMSCRIPT_BIN_DIR points to {configured!r}, but that directory does not exist."
        )

    shadow_command = get_shadow_command()
    bin_dir = os.path.dirname(shadow_command)
    if bin_dir and os.path.isdir(bin_dir):
        return bin_dir

    raise RuntimeError(
        "Unable to infer the imscript bin directory. Set IMSCRIPT_BIN_DIR explicitly."
    )


def get_tmp_dir() -> str:
    """Return the temporary directory used by shadow casting."""

    return os.environ.get("SHADOW_DATA_TMP_DIR", DEFAULT_TMP_DIR)


def get_core3d_base_url() -> str:
    """Return the base URL for the remote CORE3D mirror."""

    return os.environ.get("SHADOW_EO_CORE3D_BASE_URL", DEFAULT_CORE3D_BASE_URL).rstrip(
        "/"
    )


def _core3d_city_url(city: str, modality: str) -> str:
    if city not in CITY_FOLDERS:
        raise KeyError(f"Unknown city {city!r}. Expected one of {tuple(CITY_FOLDERS)}.")
    return f"{get_core3d_base_url()}/{CITY_FOLDERS[city]}/WV3/{modality}/"


@functools.lru_cache(maxsize=None)
def list_satellite_files(city: str, modality: str) -> tuple[str, ...]:
    """Discover and sort remote image files for a given city and modality."""

    url = _core3d_city_url(city, modality)
    try:
        files = utils.find(url, ".NTF.tif")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to discover {modality} files for {city} from {url}. "
            "Set SHADOW_EO_CORE3D_BASE_URL to a valid mirror or verify network access."
        ) from exc

    return tuple(sorted(files, key=utils.acquisition_date))


class _LazySatelliteFiles(Mapping):
    def __init__(self, modality: str):
        self.modality = modality

    def __getitem__(self, city: str) -> list[str]:
        return list(list_satellite_files(city, self.modality))

    def __iter__(self):
        return iter(CITY_FOLDERS)

    def __len__(self) -> int:
        return len(CITY_FOLDERS)


SAT_FILES = _LazySatelliteFiles("PAN")
SAT_FILES_MSI = _LazySatelliteFiles("MSI")
