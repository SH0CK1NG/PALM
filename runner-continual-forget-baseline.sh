#!/usr/bin/env bash
set -e

# 持续遗忘学习（Continual Incremental Unlearning）
                    # if len(forget_all) > 0:
                    #     excl |= forget_all 持续遗忘要删除，非持续要加上

cd "$(dirname "$0")"

# 基本信息（可按需修改或通过环境变量覆盖）
# id: 增量学习的目标数据集（包含所有阶段的类）
# base_id: 预训练权重对应的数据集
id=${ID:-CIFAR-100}
base_id=${BASE_ID:-CIFAR-100}
ood=${OOD_LIST:-"SVHN places365 LSUN iSUN dtd"}

# 训练信息
batch=${BATCH_SIZE:-128}
epochs=${EPOCHS:-50}
lr=${LR:-0.001}
wd=${WEIGHT_DECAY:-1e-4}

backbone=${BACKBONE:-resnet34}
pcon=${PCON:-1.0}
m=${PROTO_M:-0.999}
k=${TOPK:-5}
cache=${CACHE_SIZE:-6}
temp=${TEMP:-0.08}
# method=top${k}-palm-cache${cache}-ema${m} 之前MLE有误
method=top${k}-palm-cache${cache}-ema${m}

#####
base_task=c90 # c80 or c90
#####
# 预训练基座（增量第一阶段的加载点）
# 从 CIFAR-90 开始（已在除了下列10类外的90类上训练）
# pretrain_ckpt=${PRETRAIN_CKPT:-checkpoints/${base_id}-${backbone}-${method}-upper-retain-${base_task}.pt}
pretrain_ckpt=checkpoints/$id-$backbone-$method-with-prototypes.pt

# randlabel, ga
# baseline_method=randlabel
baseline_method=ga


# LoRA 设置（PEFT）
lora_r=${LORA_R:-8}
lora_alpha=${LORA_ALPHA:-32}
lora_dropout=${LORA_DROPOUT:-0.05}
lora_target=${LORA_TARGET:-both}

# 正交（PEFT 下可选） #####
orth_enable=${ORTH_ENABLE:-false}
#####
lora_orth_lambda=${LORA_ORTH_LAMBDA:-1}
# 交错正交：可为 CIFAR-100 阶段与 CIFAR-10 阶段分别配置是否启用正交;orth_enable_before_c10/orth_enable_after_c10 优先级高于 orth_enable
# - ORTH_ENABLE_BEFORE_C10：在学习 CIFAR-100 类（id<100）阶段是否启用正交，默认沿用 ORTH_ENABLE
# - ORTH_ENABLE_AFTER_C10：在学习 CIFAR-10 类（id>=100）阶段是否启用正交，默认沿用 ORTH_ENABLE
orth_enable_before_c10=${ORTH_ENABLE_BEFORE_C10:-$orth_enable}
orth_enable_after_c10=${ORTH_ENABLE_AFTER_C10:-$orth_enable}
# orth_enable_before_c10=${ORTH_ENABLE_BEFORE_C10:-false}
# orth_enable_after_c10=${ORTH_ENABLE_AFTER_C10:-true}

# 标记增量
incremental_flag=false
# 忘记/增量设置
forget_lambda=${FORGET_LAMBDA:-0.2}
batch_forget_mode=${BATCH_FORGET_MODE:-none}
forget_enable=${FORGET_ENABLE:-true}

# pcon_inc 模式：'split' | 'new_only' | 'off'；若未设置，按默认规则由程序端决定
# pcon_inc=${PCON_INC:-split}
# MLE 计算模式：all | retain_only | old_mle_all_update（若未设置则由程序端默认）
# palm_mle_mode=${PALM_MLE_MODE:-all}
# 阶段来源：
# - 若提供 STAGES_FILE，则按行读取（忽略空行与 # 注释）
# - 否则使用内置 stages 数组
stages=()
if [ -n "${STAGES_FILE:-}" ] && [ -f "$STAGES_FILE" ]; then
  mapfile -t stages < <(sed -e 's/#.*//' -e '/^\s*$/d' "$STAGES_FILE")
else
  if [ "$incremental_flag" = "false" ]; then
    if [ "$base_task" = "c90" ]; then
      stages=(
        "0,8,11,40,51"
        "66,67,88,94,57"
      )
    elif [ "$base_task" = "c80" ]; then
      stages=(
        "0,8,11,40,51,66,67,88,94,57"        # Stage 1: CIFAR-100 的5个类
        "59,58,44,93,10,64,22,42,9,90"
      )
    fi
  else
    # 从 CIFAR-80 开始的增量学习，扩展到 CIFAR-110：
    # Stage 1-4：补齐 CIFAR-100（学习被排除的20个类）
    # Stage 5-6：扩展到 CIFAR-110（学习 CIFAR-10 的10个类，映射为类别 100-109）
    if [ "$base_task" = "c80" ]; then
      stages=(
        "0,8,11,40,51"        # Stage 1: CIFAR-100 的5个类
        "66,67,88,94,57"      # Stage 2: CIFAR-100 的另外5个类
        "59,58,44,93,10"
        "64,22,42,9,90"
        "100,101,102,103,104" # Stage 5: CIFAR-10 的前5个类（100-104）
        "105,106,107,108,109" # Stage 6: CIFAR-10 的后5个类（105-109）
      )
    elif [ "$base_task" = "c90" ]; then
      stages=(
        "0,8,11,40,51"
        "66,67,88,94,57"
        "100,101,102,103,104" # Stage 5: CIFAR-10 的前5个类（100-104）
        "105,106,107,108,109" # Stage 6: CIFAR-10 的后5个类（105-109）
      )
    fi
    # 如果只想到 CIFAR-100，可删除最后两行
  fi  
fi

if [ "${#stages[@]}" -eq 0 ]; then
  echo "[error] no stages provided (STAGES_FILE empty or built-in empty)" >&2
  exit 1
fi

# 根据阶段并集生成 forget_classes_all（去重排序）
_join=""
for _csv in "${stages[@]}"; do
  if [ -z "$_join" ]; then _join="${_csv}"; else _join="${_join},${_csv}"; fi
done
forget_classes_all=$(echo "${_join}" | tr ',' '\n' | awk 'NF' | sort -n | uniq | paste -sd, -)
unset _join _csv


if [ "$incremental_flag" = "true" ]; then
  forget_enable=false
fi
#  之前MLE有误，现在修正了
# base_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}
# base_tag=MLEFixed-${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}
base_tag=${method}-baseline-${baseline_method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}
# 如果任一阶段启用了正交，则标注 ol；并额外标注 before/after 的交错配置
if [ "$orth_enable_before_c10" = "true" ] || [ "$orth_enable_after_c10" = "true" ]; then
  base_tag="${base_tag}-ol${lora_orth_lambda}"
fi
base_tag="${base_tag}-continual-from-${base_task}-to-${id}"
if [ "$incremental_flag" = "false" ]; then
  base_tag="${base_tag}-forget-from-${id}-to-${base_task}"
elif [ "$incremental_flag" = "true" ]; then
  base_tag="${base_tag}-continual-from-${base_task}-to-${id}"
fi


if [ -n "$palm_mle_mode" ]; then
  base_tag="${base_tag}-palm_mle_${palm_mle_mode}"
fi
# 若显式指定 pcon_inc，则加入标注（split/new_only/off）
if [ -n "$pcon_inc" ]; then
  base_tag="${base_tag}-pcon_${pcon_inc}"
fi
# 标注交错正交配置（便于区分实验）
if [ "$orth_enable_before_c10" != "$orth_enable_after_c10" ]; then
  base_tag="${base_tag}-ol_c10_${orth_enable_before_c10}_${orth_enable_after_c10}"
fi
# 适配器保存根目录
adapter_root=checkpoints/${id}-${backbone}-${base_tag}-stack
mkdir -p "$adapter_root"

# # 断点续跑：检测已完成阶段（以 stage 目录存在且非空为准）
# last_done=0
# seen=""
# for idx in "${!stages[@]}"; do
#   stage=$((idx+1))
#   d="${adapter_root}/stage${stage}"
#   if [ -d "$d" ] && [ "$(find "$d" -type f | head -n 1 | wc -l)" -gt 0 ]; then
#     inc_csv="${stages[$idx]}"
#     if [ -z "$seen" ]; then seen="$inc_csv"; else seen="$seen,$inc_csv"; fi
#     last_done=$stage
#   else
#     break
#   fi
# done

# if [ "$last_done" -gt 0 ]; then
#   echo "[resume] detected last completed stage = ${last_done}; seen={${seen}}"
# fi

# 从 next_stage 开始持续增量
# start_idx=$last_done
start_idx=0
seen=""

for idx in $(seq $start_idx $((${#stages[@]} - 1))); do
  stage=$((idx+1))
  inc_csv="${stages[$idx]}"
  method_tag_stage="${base_tag}-stage${stage}"
  adapter_stage_dir="${adapter_root}/stage${stage}"
  mkdir -p "$adapter_stage_dir"
  
  save_ckpt="$adapter_stage_dir/${id}-${backbone}-${method_tag_stage}.pt"

  echo "==== Stage ${stage}: inc={${inc_csv}}; seen={${seen}}; all(union)={${forget_classes_all}} ===="

  # 判断该阶段是否为 CIFAR-10（包含 id>=100）；据此决定是否启用正交
  c10_flag=$(echo "$inc_csv" | tr ',' '\n' | awk '($1+0)>=100{print 1; exit}')
  if [ "$c10_flag" = "1" ]; then
    is_c10_stage=true
  else
    is_c10_stage=false
  fi
  stage_orth_enable=$orth_enable_before_c10
  if [ "$is_c10_stage" = "true" ]; then
    stage_orth_enable=$orth_enable_after_c10
  fi

  python main.py --in-dataset $id --backbone $backbone --method $method \
    --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
    --cache-size $cache --lambda_pcon 0.0 --proto_m $m --k $k --temp $temp \
    --forget_classes "$forget_classes_all" --forget_classes_inc "$inc_csv" $(if [ -n "$seen" ]; then echo --forget_classes_seen "$seen"; fi) \
    --batch_forget_mode $batch_forget_mode \
    $(
      if [ "$forget_enable" = "true" ]; then
        echo --forget_lambda $forget_lambda
      else
        echo --forget_lambda 0
      fi
    ) \
    --forget_strategy ${baseline_method} \
    --save-path "$save_ckpt"

  # 评估：每个阶段的增量子集（逐个）+ 历史累计（包含当前阶段）
  score="mahalanobis"
  cum_seen="$seen"
  if [ -n "$cum_seen" ]; then cum_seen="$cum_seen,$inc_csv"; else cum_seen="$inc_csv"; fi

  if [ "$forget_enable" = "true" ]; then
    # 历史每个增量子集（逐个）
    for j in $(seq 1 $stage); do
      inc_j="${stages[$((j-1))]}"
      # 1a) 增量指标（Old/New/Overall）：old=seen, new=inc_j
      # bash eval.sh $id "$ood" $backbone ${method_tag_stage}-inc${j}-inc $pretrain_ckpt $score $cache 0 "$adapter_stage_dir" "" "" "" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" "" "" "" "$inc_j" "$seen"
      # 1b) 遗忘指标（Retain-Acc + Forget-as-OOD）：forget=inc_j
      bash eval.sh $id "$ood" $backbone ${method_tag_stage}-inc${j}-forget $save_ckpt $score $cache 0 "" "$inc_j" "" "$forget_lambda" "" "" "" "" --umap_enable --umap_rf_only ""
    done
    # 历史累计 seen（包含当前阶段）
    # 2a) 增量指标（Overall + Old/New：old=seen, new=当前 inc）
    # bash eval.sh $id "$ood" $backbone ${method_tag_stage}-seen-inc $pretrain_ckpt $score $cache 0 "$adapter_stage_dir" "" "" "" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" "" "" "" "$inc_csv" "$seen"
    # 2b) 遗忘指标（forget=cum_seen）
    bash eval.sh $id "$ood" $backbone ${method_tag_stage}-seen-forget $save_ckpt $score $cache 0 "" "$cum_seen" "" "$forget_lambda" "$lora_r" "" "" "" --umap_enable --umap_rf_only ""
  else
    # 纯增量评测：仅输出增量指标
    bash eval.sh $id "$ood" $backbone ${method_tag_stage}-seen-inc $pretrain_ckpt $score $cache 0 "$adapter_stage_dir" "" "" "" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" "" "" "" "$inc_csv" "$seen"
  fi

  # 更新 seen（累计）
  if [ -z "$seen" ]; then
    seen="$inc_csv"
  else
    seen="$seen,$inc_csv"
  fi
done

echo "[done] continual incremental run finished. Adapters at: ${adapter_root}"


