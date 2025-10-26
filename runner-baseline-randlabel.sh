#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# 基本信息
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"
backbone=resnet34
k=5
cache=6
m=0.999
method=top$k-palm-cache$cache-ema$m

# 训练信息
batch=128
epochs=5
lr=0.001
wd=1e-4
temp=0.08

# 预训练基座（直接在其权重上遗忘，不使用 LoRA）
pretrain_ckpt=checkpoints/${id}-${backbone}-${method}-with-prototypes.pt

# 遗忘设置（一次性全部遗忘）
forget_csv="0,8,11,40,51,66,67,88,94,57"
forget_lambda=1
batch_forget_mode=balanced

# 保存路径（会覆盖完整模型权重）
save_ckpt=checkpoints/${id}-${backbone}-${method}-baseline_randlabel_forget.pt
score="mahalanobis"
method_tag=${method}-baseline-randlabel-b${batch}-e${epochs}-lr${lr}-wd${wd}-fl${forget_lambda}

python main.py --in-dataset $id --backbone $backbone --method $method \
  --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
  --cache-size $cache --lambda_pcon 0.0 --proto_m $m --k $k \
  --forget_classes "$forget_csv" --forget_lambda $forget_lambda \
  --batch_forget_mode $batch_forget_mode \
  --temp $temp \
  --save-path $save_ckpt \
  --forget_strategy randlabel

# 评估（不使用 LoRA 适配器，直接评估保存的完整权重）
bash eval.sh $id "$ood" $backbone $method_tag $save_ckpt $score $cache 0 "" "$forget_csv" "" "$forget_lambda" "" "" "" "" --umap_enable --umap_rf_only

# 用法：
# CUDA_VISIBLE_DEVICES=0 nohup bash runner-baseline-randlabel.sh > logs/runner_randlabel_$(date +%F_%H%M).log 2>&1 &


