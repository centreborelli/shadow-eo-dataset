#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-shadow-dataset}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" -c conda-forge python=3.9 -y
fi

conda activate "${ENV_NAME}"

echo "Installing GDAL and PDAL dependencies..."
conda install -y -c conda-forge \
  cxx-compiler \
  fftw \
  gdal=3.7.2 \
  libtiff \
  python-pdal

echo "Installing Python packages..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/emasquil/sat-bundleadjust
python -m pip install git+https://github.com/centreborelli/s2p-hd.git

cat <<'EOF'

External tools not installed by this script:
- imscript: https://github.com/mnhrdt/imscript
  Required binaries for this repo include `shadowcast`, `morsi`, `plambda`, and `bdint5pc`.
  After installation, either add them to PATH or set:
  - SHADOW_COMMAND=/path/to/shadowcast
  - IMSCRIPT_BIN_DIR=/path/to/imscript/bin

Optional acceleration:
- PyTorch and torch-scatter can be installed separately to accelerate shadow projection on CUDA GPUs.
  The public release keeps a CPU fallback, so they are not required for basic reproducibility.

EOF
