#!/usr/bin/env bash
set -euo pipefail

# Evaluate PyCIL DER on PALM's OOD pipeline (feature_extract + eval_cifar)
# Usage:
#   bash runner-eval-pycil-der.sh /abs/path/to/model.pt

CKPT_PATH="${1:-}"
if [ -z "${CKPT_PATH}" ]; then
  echo "[error] missing ckpt path argument. Usage: bash $(basename "$0") /abs/path/to/model.pt"
  exit 1
fi
if [ ! -f "${CKPT_PATH}" ]; then
  echo "[error] ckpt not found: ${CKPT_PATH}"
  exit 1
fi

echo "[Eval] Using ckpt: ${CKPT_PATH}"

IN_DS="CIFAR-100"
OOD_LIST=(SVHN places365 LSUN iSUN dtd)
BACKBONE="resnet34"    # backbone flag is only used for cache dir naming; model is PyCILAdapter
METHOD="pycil-der"
SEED=1
BATCH=128
GPU=0

# 1) Feature extraction (ID + OOD)
python feature_extract.py \
  --in-dataset "${IN_DS}" \
  --out-datasets "${OOD_LIST[@]}" \
  --backbone "${BACKBONE}" \
  --method "${METHOD}" \
  --seed ${SEED} \
  --gpu ${GPU} \
  --batch-size ${BATCH} \
  --load-path "${CKPT_PATH}" 

# 2) Evaluate Mahalanobis on 5 OOD datasets
python eval_cifar.py \
  --in-dataset "${IN_DS}" \
  --out-datasets "${OOD_LIST[@]}" \
  --backbone "${BACKBONE}" \
  --method "${METHOD}" \
  --seed ${SEED} \
  --gpu ${GPU}

echo "[Done] Results saved under evaluation_results/*.csv and figs/*.png"
