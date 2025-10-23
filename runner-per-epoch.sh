#!/usr/bin/env bash
set -e

# 进入脚本所在目录，保证相对路径一致
cd "$(dirname "$0")"

# basic info
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

# training info
batch=128
epochs=5
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
pretrain_ckpt=checkpoints/$id-$backbone-$method-with-prototypes.pt

# forget classes / LoRA / forgetting config（与 runner.sh 保持一致）
# CIFAR100 Plan B
forget_csv="0,8,11,40,51,66,67,88,94,57"
forget_lambda=0.2
lora_r=8
lora_alpha=32
lora_dropout=0.05
# where to inject lora: head|encoder|both|encoder_all|both_all
lora_target=both
# per-batch forget/retain composition: none|balanced|proportional
batch_forget_mode=balanced

forget_attr_w=4
forget_proto_rep_w=4

temp=0.08

forget_avgproto_w=1.0

# tag method for evaluation to avoid overwriting outputs (embed key hparams)
method_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}-fpw${forget_avgproto_w}

# where to save adapter only (PEFT uses a directory) — 作为“基路径”，逐轮在其后附加 -epN
adapter_path=checkpoints/${id}-${backbone}-${method_tag}-forget_avgproto_enable-planB_adapter

# 按 epoch 训练与评估
for ep in $(seq 1 $epochs); do
  adapter_prev="${adapter_path}-ep$((ep-1))"
  adapter_curr="${adapter_path}-ep${ep}"
  method_tag_ep="${method_tag}-ep${ep}"

  echo "==== Epoch ${ep}/${epochs}: train 1 epoch, save adapter to ${adapter_curr}, then evaluate ===="

  # 训练 1 个 epoch，仅 LoRA 适配器（主干冻结），并在第 2 轮起加载上一轮的适配器继续
  python main.py --in-dataset $id --backbone $backbone --method $method \
    --epochs 1 --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
    --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k \
    --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
    --forget_classes $forget_csv --forget_lambda $forget_lambda \
    --batch_forget_mode $batch_forget_mode \
    --temp $temp \
    $(if [ "$ep" -gt 1 ] && [ -e "$adapter_prev" ]; then echo --adapter_load_path "$adapter_prev"; fi) \
    --adapter_save_path "$adapter_curr" 
    # --forget_avgproto_enable --forget_avgproto_w $forget_avgproto_w
    # 如需切换另一遗忘策略，注释上一行，启用下行：
    # --forget_proto_enable --forget_attr_w $forget_attr_w --forget_proto_rep_w $forget_proto_rep_w

  # 评估当前轮适配器（基座权重 + 当前适配器），避免覆盖：method_tag 使用 -epN
  score="mahalanobis"
  bash eval.sh $id "$ood" $backbone $method_tag_ep $pretrain_ckpt $score $cache 0 "$adapter_curr" "$forget_csv" "" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only
done

# 用法示例：
# CUDA_VISIBLE_DEVICES=0 nohup bash runner-per-epoch.sh > logs/runner_per_epoch_$(date +%F_%H%M).log 2>&1 &


