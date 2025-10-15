#!/usr/bin/env bash
set -e

# LoRA 固定批次调试脚本：反复使用同一批样本，观察损失是否随步数下降

# basic info
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

# training info
batch=128
epochs=1
lr=0.001
wd=1e-4

backbone=resnet34
pcon=1.

m=0.999

# Runing PALM on supervised OOD detection
k=5
cache=6
method=top$k-palm-cache$cache-ema$m

# base pretrained checkpoint (CIFAR-100)
pretrain_ckpt=/home/shaokun/PALM/checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999-with-prototypes.pt

# centers/precision (from compute_centers.py for CIFAR-100)
center_dir=cache/resnet34-$method/$id
centers_path=$center_dir/class_centers.pt
precision_path=$center_dir/precision.pt

# LoRA config
lora_r=8
lora_alpha=32
lora_dropout=0.05
lora_target=both

# 固定批次调试参数
fixed_steps=200

# 遗忘学习：忘记类 0-9（训练时将 forget 推离中心，retain 作为原型学习对象）
forget_csv="0,1,2,3,4,5,6,7,8,9"
forget_center_set="retain"
forget_lambda=0.1
# 采用平衡采样，确保固定的首个 batch 同时包含 retain/forget 样本
batch_forget_mode=balanced
forget_margin=100

# 0) ensure centers/precision
python compute_centers.py \
  --in-dataset $id \
  --backbone $backbone \
  --method $method \
  --load-path $pretrain_ckpt

# 1) 仅LoRA训练 + 固定批次调试
python main.py --in-dataset $id --backbone $backbone --method $method \
  --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
  --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k \
  --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
  --centers_path $centers_path --precision_path $precision_path \
  --forget_classes $forget_csv --forget_center_set $forget_center_set --forget_lambda $forget_lambda --forget_margin $forget_margin \
  --batch_forget_mode $batch_forget_mode \
  --debug_fixed_batch --debug_fixed_batch_steps $fixed_steps --print_every 1

echo "Done fixed-batch LoRA debug run."


