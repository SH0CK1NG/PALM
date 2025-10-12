#!/usr/bin/env bash

# basic info
id=CIFAR-10
backbone=resnet34
method=top5-palm-cache6-ema0.999
pretrain=/home/shaokun/PALM/checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999-with-prototypes.pt

# grid settings
declare -a epochs_list=(5 10 20 50 100)
declare -a lr_list=(0.5 0.05 0.01)
batch=256

# devices (up to 4 GPUs); adjust as needed
GPU_LIST=(0 1 2 3)
MAX_JOBS=${#GPU_LIST[@]}

# eval config
ood="SVHN places365 LSUN iSUN dtd"
score=mahalanobis
cache=6
eval_epochs=0

mkdir -p logs

run_job() {
  local ep=$1
  local lr=$2
  local gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu python finetune.py \
    --in-dataset $id \
    --backbone $backbone \
    --method $method \
    --epochs $ep \
    --lr $lr \
    --batch-size $batch \
    --load-path $pretrain \
    --save-path checkpoints/${id}-${backbone}-${method}-finetune.pt \
    > logs/finetune_${id}_${backbone}_${method}_ep${ep}_lr${lr}.log 2>&1 &
}

# job_idx=0
# for ep in "${epochs_list[@]}"; do
#   for lr in "${lr_list[@]}"; do
#     gpu=${GPU_LIST[$((job_idx % MAX_JOBS))]}
#     run_job $ep $lr $gpu
#     job_idx=$((job_idx + 1))
#   done
# done

# echo "Launched $job_idx finetune jobs across ${MAX_JOBS} GPUs. Logs in logs/."

# # wait all finetune jobs to finish before evaluation
# wait || true
eval_id=CIFAR-110
# sequentially evaluate all checkpoints (one by one)
echo "Starting sequential evaluations..."
job_idx=0
for ep in "${epochs_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    lr_tag=${lr//./p}
    ckpt=checkpoints/${id}-${backbone}-${method}-finetune-ep${ep}-lr${lr_tag}.pt
    gpu=${GPU_LIST[$((job_idx % MAX_JOBS))]}
    CUDA_VISIBLE_DEVICES=$gpu bash eval.sh "$eval_id" "$ood" "$backbone" "$method" "$ckpt" "$score" "$cache" "$eval_epochs" \
      > logs/eval_${eval_id}_${backbone}_${method}_ep${ep}_lr${lr}.log 2>&1
    job_idx=$((job_idx + 1))
  done
done
echo "All evaluations completed. Logs in logs/."

# Optional: after all finish, evaluate checkpoints (uncomment to use)
# ood_list="SVHN places365 LSUN iSUN dtd"
# for ep in "${epochs_list[@]}"; do
#   for lr in "${lr_list[@]}"; do
#     lr_tag=${lr//./p}
#     ckpt=checkpoints/finetune/${id}-${backbone}-${method}-ft-ep${ep}-lr${lr_tag}.pt
#     bash eval.sh $id "$ood_list" $backbone $method $ckpt mahalanobis 6 0
#   done
# done