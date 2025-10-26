#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# 基本信息
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

# 训练信息
batch=128
epochs=50
lr=0.001
wd=1e-4

backbone=resnet34
pcon=1.
m=0.999
k=5
cache=6
method=top$k-palm-cache$cache-ema$m

# 预训练基座
pretrain_ckpt=checkpoints/$id-$backbone-$method-with-prototypes.pt

# LoRA 设置（PEFT叠加）
lora_r=8
lora_alpha=32
lora_dropout=0.05
lora_target=both

# 正交开关与权重（对齐 O3 的 oLoRA/非 oLoRA 切换）
# orth_enable=true  => “热启动 + 正交约束”：加载并冻结所有历史适配器作为参照，
#                      同时训练当前阶段 LoRA，并启用正交正则（当前 A ⟂ 历史各阶段 A）。
# orth_enable=false => 非 oLoRA：仅从上一阶段 LoRA 热启动并继续训练，不加载更早阶段作参照且不施加正交项。
orth_enable=false
lora_orth_lambda=1

# 忘记设置
forget_classes_all="0,8,11,40,51,66,67,88,94,57"   # 全部要遗忘的类
# 将每个阶段要遗忘的类按数组定义，例如两阶段：
stages=(
  "0,8,11,40,51"
  "66,67,88,94,57"
)

batch_forget_mode=balanced
temp=0.08
forget_lambda=0.2
forget_avgproto_w=1.0

base_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}

if [ "$orth_enable" = "true" ]; then
  base_tag="${base_tag}-ol${lora_orth_lambda}"
fi

# 适配器保存基路径（每阶段单独子目录）
adapter_root=checkpoints/${id}-${backbone}-${base_tag}-stack
mkdir -p "$adapter_root"

# 运行阶段
seen=""  # 已经遗忘过的类的CSV

for idx in "${!stages[@]}"; do
  stage=$((idx+1))
  inc_csv="${stages[$idx]}"
  method_tag_stage="${base_tag}-stage${stage}"
  adapter_stage_dir="${adapter_root}/stage${stage}"
  mkdir -p "$adapter_stage_dir"

  # 聚合历史适配器路径，用于冻结组合
  load_csv=""
  if [ "$stage" -gt 1 ]; then
    # 加载之前所有阶段目录，以逗号拼接
    for j in $(seq 1 $((stage-1))); do
      p="${adapter_root}/stage${j}"
      if [ -d "$p" ]; then
        if [ -z "$load_csv" ]; then load_csv="$p"; else load_csv="$load_csv,$p"; fi
      fi
    done
  fi

  echo "==== Stage ${stage}: forget_inc={${inc_csv}}; forget_seen={${seen}}; all={${forget_classes_all}} ===="

  # python main.py --in-dataset $id --backbone $backbone --method $method \
  #   --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
  #   --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k \
  #   --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
  #   $(
  #     # oLoRA：仅加载上一阶段LoRA为可训练 + 从磁盘读取历史A作为正交参照
  #     if [ "$stage" -gt 1 ] && [ "$orth_enable" = "true" ]; then
  #       prev_dir="${adapter_root}/stage$((stage-1))"
  #       refs_csv=""
  #       # 历史参照为 1..(stage-1) 的所有阶段目录
  #       for j in $(seq 1 $((stage-1))); do
  #         p="${adapter_root}/stage${j}"
  #         if [ -d "$p" ]; then
  #           if [ -z "$refs_csv" ]; then refs_csv="$p"; else refs_csv="$refs_csv,$p"; fi
  #         fi
  #       done
  #       if [ -d "$prev_dir" ]; then echo --adapter_load_path "$prev_dir" --lora_orth_enable --lora_orth_lambda $lora_orth_lambda --lora_orth_ref_paths "$refs_csv"; fi
  #     fi
  #   ) \
  #   $(
  #     # 非 oLoRA：直接从上一阶段 LoRA 热启动继续训练（不加载更早阶段作参照）
  #     if [ "$stage" -gt 1 ] && [ "$orth_enable" != "true" ]; then
  #       prev_dir="${adapter_root}/stage$((stage-1))"
  #       if [ -d "$prev_dir" ]; then echo --adapter_load_path "$prev_dir"; fi
  #     fi
  #   ) \
  #   --forget_classes "$forget_classes_all" --forget_classes_inc "$inc_csv" $(if [ -n "$seen" ]; then echo --forget_classes_seen "$seen"; fi) --forget_lambda $forget_lambda \
  #   --batch_forget_mode $batch_forget_mode \
  #   --temp $temp \
  #   --adapter_save_path "$adapter_stage_dir" 
  #   # --forget_avgproto_enable --forget_avgproto_w $forget_avgproto_w
  # 评估：每个阶段的增量子集（逐个）+ 历史累计（包含当前阶段）
  score="mahalanobis"
  # 对齐 O3：评估仅使用“当前阶段”适配器（不与历史适配器组合）
  eval_paths_csv="$adapter_stage_dir"

  # 统一计算 cum_seen（seen ∪ inc_csv），并用于所有评估的保留排除集
  cum_seen="$seen"
  if [ -n "$cum_seen" ]; then cum_seen="$cum_seen,$inc_csv"; else cum_seen="$inc_csv"; fi

  # 1) 历史每个增量子集（逐个） - 保留集=全部类别−cum_seen；遗忘集=inc_j
  for j in $(seq 1 $stage); do
    inc_j="${stages[$((j-1))]}"
    retain_exclude_csv="$cum_seen"
    bash eval.sh $id "$ood" $backbone ${method_tag_stage}-inc${j} $pretrain_ckpt $score $cache 0 "$adapter_stage_dir" "$inc_j" "" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only "" "$retain_exclude_csv"
  done
  # 2) 历史累计 seen（包含当前阶段） - 保留集=全部类别−cum_seen；遗忘集=cum_seen
  bash eval.sh $id "$ood" $backbone ${method_tag_stage}-seen $pretrain_ckpt $score $cache 0 "$adapter_stage_dir" "$cum_seen" "" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only "" "$cum_seen"

  # 更新 seen（累计）
  if [ -z "$seen" ]; then
    seen="$inc_csv"
  else
    seen="$seen,$inc_csv"
  fi
done

# 用法：
# CUDA_VISIBLE_DEVICES=0 nohup bash runner-incremental-stack.sh > logs/runner_incremental_$(date +%F_%H%M).log 2>&1 &


