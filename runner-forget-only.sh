#!/usr/bin/env bash
set -e

# Basic info
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

# Training info
batch=128
epochs=50
lr=0.001
wd=1e-4

backbone=resnet34

# Disable PALM loss (only forgetting loss)
pcon=0
m=0.999

# PALM method tag base
k=5
cache=6
method=top$k-palm-cache$cache-ema$m

# Base pretrained checkpoint (CIFAR-100)
pretrain_ckpt=checkpoints/$id-$backbone-$method-with-prototypes.pt

# One-shot forget classes (Plan B)
forget_csv="0,8,11,40,51,66,67,88,94,57"
forget_lambda=0.2

# LoRA config
lora_r=8
lora_alpha=32
lora_dropout=0.05
lora_target=both # head|encoder|both|encoder_all|both_all

# Per-batch composition for forget/retain
batch_forget_mode=balanced # none|balanced|proportional|retain_only

temp=0.08

# Tag for outputs/adapters (mark nopalm)
method_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}-nopalm

# Where to save adapter only (PEFT uses a directory)
adapter_path=checkpoints/${id}-${backbone}-${method_tag}-forget_only_adapter

# Train with LoRA adapters only; PALM loss disabled via --lambda_pcon 0; keep forgetting loss only
python main.py --in-dataset $id --backbone $backbone --method $method \
  --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
  --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k --palm_enable false \
  --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
  --forget_classes $forget_csv --forget_lambda $forget_lambda \
  --batch_forget_mode $batch_forget_mode \
  --temp $temp \
  --adapter_save_path $adapter_path

# Evaluate with base ckpt + adapter (retain vs external OOD + forget-as-OOD)
score="mahalanobis"
bash eval.sh $id "$ood" $backbone $method_tag $pretrain_ckpt $score $cache 0 $adapter_path "$forget_csv" "" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only

# Usage:
# CUDA_VISIBLE_DEVICES=0 nohup bash runner-forget-only.sh > logs/runner_forget_only_$(date +%F_%H%M).log 2>&1 &


