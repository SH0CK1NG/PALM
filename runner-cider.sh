#!/usr/bin/env bash
set -e

# basic info
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

# training info
batch=128
epochs=50
lr=0.001
wd=1e-4
seed=1
gpu=0

backbone=resnet34

# CIDER params
feat_dim=128
w=1
m=0.999            # proto_m (EMA)
temp=0.08

# Forgetting config
# 示例（可编辑）：忘记 10 个 CIFAR-100 类
# 遗忘设置（一次性全部遗忘）
forget_class_num=5
case $forget_class_num in
  5)
    forget_csv="0,8,11,40,51"
    ;;
  10)
    forget_csv="0,8,11,40,51,66,67,88,94,57"
    ;;
  15)
    forget_csv="0,8,11,40,51,66,67,88,94,57,59,58,44,93,10"
    ;;
  20)
    forget_csv="0,8,11,40,51,66,67,88,94,57,59,58,44,93,10,64,22,42,9,90"
    ;;
esac
forget_lambda=0.2
batch_forget_mode=balanced   # none|balanced|proportional|retain_only

# # pretrain ckpt (CIDER base checkpoint)
# pretrain_ckpt=/home/shaokun/cider-master/checkpoints/CIFAR-100/30_10_18:17_cider_resnet34_lr_0.5_cosine_True_bsz_512_cider_wd_1.0_500_128_trial_0_temp_0.1_CIFAR-100_pm_0.5/checkpoint_500.pth.tar
# # C90Upper pretrain ckpt
# pretrain_ckpt=/home/shaokun/cider-master/checkpoints/CIFAR-100/14_11_04:04_cider_resnet34_lr_0.5_cosine_True_bsz_512_cider_wd_1.0_500_128_trial_0_temp_0.1_CIFAR-100_pm_0.5/checkpoint_500.pth.tar
# C95Upper pretrain ckpt
pretrain_ckpt=/home/shaokun/cider-master/checkpoints/CIFAR-100/14_11_04:06_cider_resnet34_lr_0.5_cosine_True_bsz_512_cider_wd_1.0_300_128_trial_0_temp_0.1_CIFAR-100_pm_0.5/checkpoint_300.pth.tar
# LoRA config
lora_r=8
lora_alpha=32
lora_dropout=0.05
lora_target=both   # head|encoder|both|encoder_all|both_all

# tag for naming (让 eval 用 PALMResNet 以兼容 head 特征；包含 palm 关键字即可)
method_tag=test-palm-cider-b${batch}-e${epochs}-lr${lr}-wd${wd}-fd${feat_dim}-w${w}-pm${m}-temp${temp}-bfm${batch_forget_mode}-fl${forget_lambda}

# where to save checkpoint / adapters
ckpt=checkpoints/${id}/${id}-${backbone}-${method_tag}.pt
adapter_path=checkpoints/${id}-${backbone}-${method_tag}-adapter

# # train (CIDER + LoRA forgetting on base ckpt)
# python cider_main.py \
#   --in-dataset $id --backbone $backbone \
#   --epochs $epochs -b $batch --lr $lr --weight-decay $wd \
#   --seed $seed --gpu $gpu \
#   --temp $temp --proto_m $m \
#   --load-path "$pretrain_ckpt" \
#   --forget_classes "$forget_csv" --forget_lambda $forget_lambda \
#   --batch_forget_mode $batch_forget_mode \
#   --save-path "$ckpt" \
#   --print_every 50 \
#   --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
#   --adapter_save_path "$adapter_path"

# evaluate (retain vs OOD; 并在有遗忘集时，忽略被遗忘类中心)
score="mahalanobis"
cache=6
eval_epochs=0

# 遗忘学习的备份
bash eval.sh $id "$ood" $backbone $method_tag "$pretrain_ckpt" $score $cache 0 "$adapter_path" "$forget_csv" "" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only ""

# # 手动调用特征提取与评估，避免在 eval.sh 中触发增量路径导致遗忘集缓存缺失
# python feature_extract.py \
#   --in-dataset "$id" \
#   --out-datasets $ood \
#   --backbone "$backbone" \
#   --method "$method_tag" \
#   --save-path "$pretrain_ckpt" \
#   --load-path "$pretrain_ckpt" \
#   --epochs "$eval_epochs" \
#   -b "$batch" \
#   --cache-size "$cache" \
#   --seed "$seed" \
#   --gpu "$gpu" \
#   --temp "$temp" \
#   --proto_m "$m" \
#   --forget_csv "$forget_csv" \
#   --forget_lambda "$forget_lambda" \
#   --batch_forget_mode "$batch_forget_mode" \
#   --use_lora --lora_impl peft \
#   --adapter_load_path "$adapter_path" \
#   --lora_target "$lora_target" \
#   --lora_r "$lora_r" \
#   --lora_alpha "$lora_alpha" \
#   --lora_dropout "$lora_dropout"

# python eval_cifar.py \
#   --in-dataset "$id" \
#   --out-datasets $ood \
#   --backbone "$backbone" \
#   --method "$method_tag" \
#   --save-path "$pretrain_ckpt" \
#   --load-path "$pretrain_ckpt" \
#   --epochs "$eval_epochs" \
#   --cache-size "$cache" \
#   --seed "$seed" \
#   --gpu "$gpu" \
#   --score "$score" \
#   --forget_classes "$forget_csv" \
#   --forget_lambda "$forget_lambda" \
#   --batch_forget_mode "$batch_forget_mode" \
#   --use_lora --lora_impl peft \
#   --adapter_load_path "$adapter_path" \
#   --lora_target "$lora_target" \
#   --lora_r "$lora_r" \
#   --lora_alpha "$lora_alpha" \
#   --lora_dropout "$lora_dropout"

# 用法：
#   CUDA_VISIBLE_DEVICES=4 nohup bash runner-cider.sh > logs/runner_cider_$(date +%F_%H%M).log 2>&1 &


