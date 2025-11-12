#!/usr/bin/env bash
set -euo pipefail

# PyCIL baseline: DER on CIFAR-100 -> CIFAR-10 (via CIFAR-110 if supported)
# Usage:
#   bash runner-pycil-der-c100to10.sh [CKPT_PATH]
# If CKPT_PATH is provided, it will be passed to eval script directly; otherwise use the default saved path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYCL_DIR="${SCRIPT_DIR}/third_party/PyCIL"

echo "[Stage] Prepare PyCIL at: ${PYCL_DIR}"
if [ ! -d "${PYCL_DIR}" ]; then
  mkdir -p "${PYCL_DIR%/*}"
  git clone --depth=1 https://github.com/G-U-N/PyCIL.git "${PYCL_DIR}"
  (
    cd "${PYCL_DIR}"
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi
  )
fi

# Use PyCIL json config (cifar100 90->10) with ResNet-34 backbone
CONFIG_PATH="${PYCL_DIR}/exps/der_c100_90to10_mem2000_resnet34.json"

# Optional arg1: explicit ckpt path (the model.pt saved by PyCIL)
USER_CKPT_PATH="${1:-}"

echo "[Run] PyCIL DER with --config=${CONFIG_PATH}"
(
  cd "${PYCL_DIR}"
  python main.py --config="${CONFIG_PATH}"
)

# Decide which ckpt to evaluate
DEFAULT_CKPT="${PYCL_DIR}/logs/der/cifar100/90/10/reproduce_1_resnet34/model.pt"
if [ -n "${USER_CKPT_PATH}" ]; then
  CKPT_PATH="${USER_CKPT_PATH}"
else
  CKPT_PATH="${DEFAULT_CKPT}"
fi

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[error] ckpt not found: ${CKPT_PATH}"
  echo "        You can pass the saved model path explicitly as: bash $(basename "$0") /abs/path/to/model.pt"
  exit 1
fi

echo "[Eval] Start OOD evaluation (SVHN, places365, LSUN, iSUN, dtd)"
bash "${SCRIPT_DIR}/runner-eval-pycil-der.sh" "${CKPT_PATH}"

echo "[Done] DER baseline finished and evaluated. Logs in PyCIL/logs/der/cifar100/90/10"


