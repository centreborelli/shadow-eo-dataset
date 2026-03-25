# S-EO Dataset Creation Code

Research code for reproducing the dataset construction pipeline from **[S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications](https://ieeexplore.ieee.org/document/11147657)**, accepted at the **EarthVision Workshop at CVPR 2025**. The preprint is also available on [arXiv](https://arxiv.org/abs/2504.06920).

This repository is intended as a **reproducibility-first research release**. It contains the scripts used to assemble the dataset from public upstream sources, but it is not packaged as a turnkey production pipeline.

## Resources

The public dataset and model artifacts are available in the [Hugging Face Shadow EO collection](https://huggingface.co/collections/emasquil/shadow-eo), and the companion training and evaluation code is available in the [shadow-eo-detection repository](https://github.com/centreborelli/shadow-eo-detection).

```bash
git lfs clone git@hf.co:datasets/emasquil/shadow-eo
```

## Project Summary

> We introduce the S-EO dataset: a large-scale, high-resolution dataset, designed to advance geometry-aware shadow detection Collected from diverse public-domain sources, including challenge datasets and government providers such as USGS, our dataset comprises 702 georeferenced tiles across the USA, each covering 500x500 m. Each tile includes multi-date, multi-angle WorldView-3 pansharpened RGB images, panchromatic images, and a ground-truth DSM of the area obtained from LiDAR scans. For each image, we provide a shadow mask derived from geometry and sun position, a vegetation mask based on the NDVI index, and a bundle-adjusted RPC model. With approximately 20,000 images, the S-EO dataset establishes a new public resource for shadow detection in remote sensing imagery and its applications to 3D reconstruction. To demonstrate the dataset's impact, we train and evaluate a shadow detector, showcasing its ability to generalize, even to aerial images. Finally, we extend EO-NeRF - a state-of-the-art NeRF approach for satellite imagery - to leverage our shadow predictions for improved 3D reconstructions.

This repository focuses on the **data generation workflow** used to build those assets.

## What This Repo Does

The codebase covers the main stages of dataset construction:

1. Tile generation and crop extraction from remote imagery sources.
2. DSM generation from USGS 3DEP point clouds and DEM support data.
3. RPC bundle adjustment and DSM alignment.
4. MSI correction, alignment, and pansharpening.
5. Geometry-based shadow casting plus uncertainty and vegetation masks.

Key entrypoints:

- [main.py](main.py): tile creation, PAN/MSI crop generation, DSM generation, tar download for MSI metadata.
- [create_s2p_pairs_ranking.py](create_s2p_pairs_ranking.py): ranks image pairs used for stereo reconstruction.
- [run_ba.py](run_ba.py): bundle adjustment, S2P execution, and DSM alignment.
- [pansharpen.py](pansharpen.py): radiometric correction, MSI alignment, and pansharpening.
- [shadow_cast.py](shadow_cast.py): shadow mask generation plus uncertainty and vegetation masks.

## Prerequisites

Python dependencies are listed in [requirements.txt](requirements.txt), and a bootstrap script is provided in [setup_env.sh](setup_env.sh).

External tools are still required:

- [`imscript`](https://github.com/mnhrdt/imscript/tree/master), including `shadowcast`, `morsi`, `plambda`, and `bdint5pc`
- [`s2p-hd`](https://github.com/centreborelli/s2p-hd)
- GDAL / PDAL native dependencies

The setup script installs the Python environment and the Python-side [`s2p-hd`](https://github.com/centreborelli/s2p-hd) wrapper, but **does not install [`imscript`](https://github.com/mnhrdt/imscript/tree/master)**.

Environment variables supported by the public release:

- `SHADOW_EO_CORE3D_BASE_URL`: alternate mirror for remote CORE3D imagery listing
- `SHADOW_COMMAND`: path or command name for `shadowcast`
- `IMSCRIPT_BIN_DIR`: directory containing imscript binaries
- `SHADOW_DATA_TMP_DIR`: temporary directory for intermediate shadow-casting files
- `SHADOW_THUMBNAIL_SRC_DIR` and `SHADOW_THUMBNAIL_DST_DIR`: optional overrides for thumbnail generation

## High-Level Pipeline

The intended execution order is:

1. Create the environment with `./setup_env.sh`.
2. Generate tiles with `python main.py create_all_city_tiles --size <tile_size> --output_dir <dataset_dir>`.
3. Generate DSMs and crops with `python main.py create_all_crops_and_dsms --output_dir <dataset_dir>`.
4. Rank stereo pairs with `python create_s2p_pairs_ranking.py --output_dir <dataset_dir>`.
5. Run bundle adjustment and DSM alignment with [run_ba.py](run_ba.py).
6. Run MSI correction / pansharpening with [pansharpen.py](pansharpen.py).
7. Generate shadow masks with [shadow_cast.py](shadow_cast.py).

The scripts assume a dataset workspace containing subdirectories such as `tiles/`, `crops/`, `msi/`, `root/`, `root_dir_ba/`, `s2p/`, `rdsm/`, and `shadows/`.

## Expected Outputs

The generated workspace can contain:

- Tile shapefiles and preview maps
- City DEMs and AOI DSM folders
- PAN and MSI crops
- MSI metadata tar files
- Bundle-adjusted RPC JSON files
- S2P DSMs and aligned reference DSMs
- Radiometrically corrected and pansharpened imagery
- Shadow masks, unseen-pixel masks, and vegetation masks

The published dataset and model artifacts in the [Hugging Face collection](https://huggingface.co/collections/emasquil/shadow-eo) are the recommended starting point if you want to **use** S-EO rather than regenerate it.

## Limitations

- This is research code. It has been cleaned up for public release, but it still assumes familiarity with the remote sensing toolchain and upstream data sources.
- The pipeline depends on remote public resources and external binaries that may change independently of this repository.
- The downstream scripts in this repo are currently wired around the **Min DSM** processing path. The released dataset includes both Min and Max DSM-derived assets, but switching the full local pipeline to Max DSM still requires code changes.
- The notebooks in this repository are exploratory artifacts, not supported public entrypoints.

## Citation

If you use the S-EO dataset or this repository, cite **[S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications](https://ieeexplore.ieee.org/document/11147657)**. The preprint is also available on [arXiv](https://arxiv.org/abs/2504.06920).

```bibtex
@inproceedings{masquil2025shadoweo,
  title={S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications},
  author={Masquil, El{\'i}as and Mar{\'i}, Roger and Ehret, Thibaud and Meinhardt-Llopis, Enric and Mus{\'e}, Pablo and Facciolo, Gabriele},
  booktitle={Proceedings of the 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2025}
}
```
