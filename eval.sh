#!/usr/bin/env bash
set -e

# Usage:
#   bash eval.sh <in_dataset> "<out_datasets>" <backbone> <method> <ckpt_path> [score] [cache_size] [epochs]
# Example:
#   bash eval.sh CIFAR-10 "SVHN places365 LSUN iSUN dtd" resnet34 top5-palm-cache6-ema0.999 checkpoints/C10-...pt mahalanobis 6 0 [adapter_path] [forget_csv] [forget_list_path] [forget_center_set] [forget_lambda] [lora_r] [lora_alpha] [lora_dropout]

id="$1"
ood_list="$2"      # pass as quoted string so it becomes multiple args when expanded below
backbone="$3"
method="$4"
ckpt="$5"
score="${6:-mahalanobis}"
cache="${7:-6}"
epochs="${8:-0}"
adapter_path="${9:-}"
forget_csv="${10:-}"
forget_list_path="${11:-}"
forget_center_set="${12:-}"
forget_lambda="${13:-}"
lora_r="${14:-}"
lora_alpha="${15:-}"
lora_dropout="${16:-}"
lora_target="${17:-}"

# 1) extract features for IN/OOD using the provided checkpoint
python feature_extract.py \
  --in-dataset "$id" \
  --out-datasets $ood_list \
  --backbone "$backbone" \
  --method "$method" \
  --epochs "$epochs" \
  --save-path "$ckpt" \
  --cache-size "$cache" \
  $(if [ -n "$adapter_path" ]; then echo --use_lora --lora_impl peft --adapter_load_path "$adapter_path"; fi) \
  $(if [ -n "$lora_target" ]; then echo --lora_target "$lora_target"; fi)

# 2) run evaluation using the same checkpoint and cached features
python eval_cifar.py \
  --in-dataset "$id" \
  --out-datasets $ood_list \
  --backbone "$backbone" \
  --method "$method" \
  --epochs "$epochs" \
  --save-path "$ckpt" \
  --score "$score" \
  --cache-size "$cache" \
  $(if [ -n "$adapter_path" ]; then echo --use_lora --lora_impl peft --adapter_load_path "$adapter_path"; fi) \
  $(if [ -n "$lora_target" ]; then echo --lora_target "$lora_target"; fi) \
  $(if [ -n "$forget_csv" ]; then echo --forget_classes "$forget_csv"; fi) \
  $(if [ -n "$forget_list_path" ]; then echo --forget_list_path "$forget_list_path"; fi) \
  $(if [ -n "$forget_center_set" ]; then echo --forget_center_set "$forget_center_set"; fi) \
  $(if [ -n "$forget_lambda" ]; then echo --forget_lambda "$forget_lambda"; fi) \
  $(if [ -n "$lora_r" ]; then echo --lora_r "$lora_r"; fi) \
  $(if [ -n "$lora_alpha" ]; then echo --lora_alpha "$lora_alpha"; fi) \
  $(if [ -n "$lora_dropout" ]; then echo --lora_dropout "$lora_dropout"; fi)



