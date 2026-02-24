#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data4/dongmin/t-car"
VENV_DIR="${1:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] Python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[INFO] Creating venv: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[INFO] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing torch/torchvision (CUDA 12.1 wheels)"
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1

echo "[INFO] Installing stage2 requirements"
python -m pip install -r "$ROOT_DIR/tools/requirements.txt"

echo "[INFO] Verifying installation"
python - <<'PY'
import sys
import torch
import torchvision
import ultralytics
import cv2
import PIL
import numpy
import matplotlib
import timm

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("opencv", cv2.__version__)
print("pillow", PIL.__version__)
print("numpy", numpy.__version__)
print("matplotlib", matplotlib.__version__)
print("timm", timm.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY

echo "[DONE] Activate with: source $VENV_DIR/bin/activate"
