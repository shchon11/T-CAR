#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${ROOT_DIR:-}" ]]; then
  # Support both layouts:
  # 1) repo-root/setup_stage2_venv.sh
  # 2) script placed in a subdirectory (fallback to parent)
  if [[ -d "$SCRIPT_DIR/data" ]]; then
    ROOT_DIR="$SCRIPT_DIR"
  elif [[ -d "$SCRIPT_DIR/../data" ]]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  else
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  fi
fi

VENV_DIR="${1:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-}"
if [[ -z "$REQUIREMENTS_FILE" ]]; then
  REQUIREMENTS_CANDIDATES=(
    "$SCRIPT_DIR/requirements.txt"
    "$ROOT_DIR/requirements.txt"
  )
  for candidate in "${REQUIREMENTS_CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
      REQUIREMENTS_FILE="$candidate"
      break
    fi
  done
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] Python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "[ERR] requirements file not found: $REQUIREMENTS_FILE" >&2
  echo "[ERR] checked default candidates:" >&2
  echo "  - $SCRIPT_DIR/requirements.txt" >&2
  echo "  - $ROOT_DIR/requirements.txt" >&2
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
python -m pip install -r "$REQUIREMENTS_FILE"

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
