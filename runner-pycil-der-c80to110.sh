#!/usr/bin/env bash
set -euo pipefail

# PyCIL DER: CIFAR-80 -> CIFAR-110 (increment 5) + per-stage OOD eval

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

CONFIG_PATH="${PYCL_DIR}/exps/der_c80to110_mem2000_resnet34.json"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[error] missing config: ${CONFIG_PATH}"
  echo "        请先确认已生成配置文件 exps/der_c80to110_mem2000_resnet34.json"
  exit 1
fi

echo "[Run] PyCIL DER with --config=${CONFIG_PATH}"
(
  cd "${PYCL_DIR}"
  python main.py --config="${CONFIG_PATH}"
)

# After-task OOD evaluation on 5 datasets for every incremental stage
LOG_DIR="${PYCL_DIR}/logs/der/cifar110/80/5/reproduce_1_resnet34"
IN_DS="CIFAR-110"
OOD_LIST=(SVHN places365 LSUN iSUN dtd)
BACKBONE="resnet34"
METHOD="pycil-der"
SEED=1
BATCH=128
GPU=0

if [ -d "${LOG_DIR}" ]; then
  echo "[Eval] Per-stage OOD evaluation in: ${LOG_DIR}"
  shopt -s nullglob
  CKPTS=("${LOG_DIR}"/model_task*.pt)
  # sort by task id
  IFS=$'\n' CKPTS_SORTED=($(printf '%s\n' "${CKPTS[@]}" | sort -V))
  unset IFS
  for CKPT in "${CKPTS_SORTED[@]}"; do
    BASENAME="$(basename "${CKPT}")"
    TASK_ID="${BASENAME#model_task}"
    TASK_ID="${TASK_ID%.pt}"
    LEARNED_CSV_FILE="${LOG_DIR}/learned_task${TASK_ID}.csv"
    LEARNED_CSV=""
    if [ -f "${LEARNED_CSV_FILE}" ]; then
      LEARNED_CSV="$(cat "${LEARNED_CSV_FILE}")"
    fi
    echo "[Eval][Task ${TASK_ID}] ckpt=${CKPT} learned=${LEARNED_CSV}"
    # 1) Feature extraction (ID + OOD), incremental filtering to已学类
    python "${SCRIPT_DIR}/feature_extract.py" \
      --in-dataset "${IN_DS}" \
      --out-datasets "${OOD_LIST[@]}" \
      --backbone "${BACKBONE}" \
      --method "${METHOD}" \
      --seed ${SEED} \
      --gpu ${GPU} \
      --batch-size ${BATCH} \
      --load-path "${CKPT}" \
      --incremental \
      --forget_classes_seen "${LEARNED_CSV}"
    # 2) Evaluate Mahalanobis on 5 OOD datasets
    python "${SCRIPT_DIR}/eval_cifar.py" \
      --in-dataset "${IN_DS}" \
      --out-datasets "${OOD_LIST[@]}" \
      --backbone "${BACKBONE}" \
      --method "${METHOD}" \
      --seed ${SEED} \
      --gpu ${GPU}
  done
else
  echo "[warn] 未找到日志目录: ${LOG_DIR}，跳过阶段性评估"
fi

DEFAULT_CKPT="${LOG_DIR}/model.pt"
echo "[Info] 训练完成，最终模型保存在: ${DEFAULT_CKPT}"


