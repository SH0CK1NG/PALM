#!/usr/bin/env bash
set -e

# Usage:
#   bash eval.sh <in_dataset> "<out_datasets>" <backbone> <method> <ckpt_path> [score] [cache_size] [epochs]
# Example:
#   bash eval.sh CIFAR-10 "SVHN places365 LSUN iSUN dtd" resnet34 top5-palm-cache6-ema0.999 checkpoints/C10-...pt mahalanobis 6 0

id="$1"
ood_list="$2"      # pass as quoted string so it becomes multiple args when expanded below
backbone="$3"
method="$4"
ckpt="$5"
score="${6:-mahalanobis}"
cache="${7:-6}"
epochs="${8:-0}"

# 1) extract features for IN/OOD using the provided checkpoint
python feature_extract.py \
  --in-dataset "$id" \
  --out-datasets $ood_list \
  --backbone "$backbone" \
  --method "$method" \
  --epochs "$epochs" \
  --save-path "$ckpt" \
  --cache-size "$cache"

# 2) run evaluation using the same checkpoint and cached features
python eval_cifar.py \
  --in-dataset "$id" \
  --out-datasets $ood_list \
  --backbone "$backbone" \
  --method "$method" \
  --epochs "$epochs" \
  --save-path "$ckpt" \
  --score "$score" \
  --cache-size "$cache"



