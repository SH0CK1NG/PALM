#!/usr/bin/env bash
set -euo pipefail

# 持续学习网格搜索脚本
# 结合 runner-incremental-continual.sh 的持续学习逻辑和 grid_search-up.sh 的网格搜索功能
# 
# 功能：
# - 对多个超参数组合（lambda_pcon, lr, epochs, lora_r）进行网格搜索
# - 每个组合执行完整的持续增量学习流程（多阶段训练）
# - 评估每个组合的最终性能，计算综合评分
#
# Score计算：
# - AVG-AUROC, AVG-FPR, Final(Top-1), Average 四个指标权重相等
# - Score = (AVG-AUROC + (100-AVG-FPR) + Final-Top1*100 + Average*100) / 4
# - 所有指标都转换为"越大越好"的形式后取平均
#
# 使用方法：
#   1. 直接运行（使用默认搜索空间）：
#      bash grid_search_continual.sh
#
#   2. 通过环境变量设置搜索空间：
#      GRID_LAMBDA="0.1,0.2,0.3" GRID_LR="0.001,0.01" GRID_EPOCHS="50,100" GRID_LORA_R="8,16" bash grid_search_continual.sh
#      注意：GRID_LAMBDA 用于搜索 lambda_pcon（不是 forget_lambda）
#
#   3. 通过环境变量覆盖其他配置：
#      ID=CIFAR-110 BASE_TASK=c90 ORTH_ENABLE=true bash grid_search_continual.sh
#
# 输出：
#   - 结果保存在 evaluation_results/${id}-${backbone}-${method}-continual-grid_runs.csv
#   - 每个超参数组合的适配器保存在 checkpoints/${id}-${backbone}-${base_tag}-stack/

cd "$(dirname "$0")"

# 基本信息（可按需修改或通过环境变量覆盖）
id=${ID:-CIFAR-110}
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
method=top${k}-palm-cache${cache}-ema${m}


# 网格搜索空间（lambda_list 用于搜索 lambda_pcon）
# lambda_list=($(seq 0.5 0.1 1.5))
lambda_list=(0.1)
lr_list=(0.001)
# epoch_list=($(seq 5 5 50))
epoch_list=(1)
lora_r_list=(8)

tag_suffix="-continual-grid_runs"

#####
base_task=c80 # c80 or c90
#####
# 预训练基座（增量第一阶段的加载点）
pretrain_ckpt=${PRETRAIN_CKPT:-checkpoints/${base_id}-${backbone}-${method}-upper-retain-${base_task}.pt}

# LoRA 设置（PEFT）
lora_alpha=${LORA_ALPHA:-32}
lora_dropout=${LORA_DROPOUT:-0.05}
lora_target=${LORA_TARGET:-both}

# 正交（PEFT 下可选）
orth_enable=${ORTH_ENABLE:-true}
lora_orth_lambda=${LORA_ORTH_LAMBDA:-1}
orth_enable_before_c10=${ORTH_ENABLE_BEFORE_C10:-$orth_enable}
orth_enable_after_c10=${ORTH_ENABLE_AFTER_C10:-$orth_enable}

# 标记增量
incremental_flag=true
# 忘记/增量设置
forget_lambda=${FORGET_LAMBDA:-0.2}
batch_forget_mode=${BATCH_FORGET_MODE:-none}
forget_enable=${FORGET_ENABLE:-false}

# pcon_inc 模式
pcon_inc=${PCON_INC:-split}
# MLE 计算模式
palm_mle_mode=${PALM_MLE_MODE:-all}

# 阶段来源
stages=()
if [ -n "${STAGES_FILE:-}" ] && [ -f "$STAGES_FILE" ]; then
  mapfile -t stages < <(sed -e 's/#.*//' -e '/^\s*$/d' "$STAGES_FILE")
else
  if [ "$incremental_flag" = "false" ]; then
    if [ "$base_task" = "c80" ]; then
      stages=(
        "0,8,11,40,51"
        "66,67,88,94,57"
      )
    elif [ "$base_task" = "c90" ]; then
      stages=(
        "0,8,11,40,51"
        "66,67,88,94,57"
        "59,58,44,93,10"
        "64,22,42,9,90"
      )
    fi
  else
    if [ "$base_task" = "c80" ]; then
      stages=(
        "0,8,11,40,51"
        "66,67,88,94,57"
        "59,58,44,93,10"
        "64,22,42,9,90"
        # "100,101,102,103,104"
        # "105,106,107,108,109"
      )
    elif [ "$base_task" = "c90" ]; then
      stages=(
        "0,8,11,40,51"
        "66,67,88,94,57"
        "100,101,102,103,104"
        "105,106,107,108,109"
      )
    fi
  fi
fi

if [ "${#stages[@]}" -eq 0 ]; then
  echo "[error] no stages provided (STAGES_FILE empty or built-in empty)" >&2
  exit 1
fi

# 根据阶段并集生成 forget_classes_all
_join=""
for _csv in "${stages[@]}"; do
  if [ -z "$_join" ]; then _join="${_csv}"; else _join="${_join},${_csv}"; fi
done
forget_classes_all=$(echo "${_join}" | tr ',' '\n' | awk 'NF' | sort -n | uniq | paste -sd, -)
unset _join _csv

if [ "$incremental_flag" = "true" ]; then
  forget_enable=false
fi

# 网格搜索空间
# lambda_list=(${LAMBDA_LIST:-0.2})
# lr_list=(${LR_LIST:-0.001})
# epoch_list=(${EPOCH_LIST:-50})
# lora_r_list=(${LORA_R_LIST:-8})



# 如果通过环境变量设置了搜索空间，使用环境变量
# GRID_LAMBDA 用于搜索 lambda_pcon
if [ -n "${GRID_LAMBDA:-}" ]; then
  IFS=',' read -ra lambda_list <<< "$GRID_LAMBDA"
fi
if [ -n "${GRID_LR:-}" ]; then
  IFS=',' read -ra lr_list <<< "$GRID_LR"
fi
if [ -n "${GRID_EPOCHS:-}" ]; then
  IFS=',' read -ra epoch_list <<< "$GRID_EPOCHS"
fi
if [ -n "${GRID_LORA_R:-}" ]; then
  IFS=',' read -ra lora_r_list <<< "$GRID_LORA_R"
fi

# 结果CSV文件
# tag_suffix=${TAG_SUFFIX:-"-continual-grid_runs"}
results_csv="evaluation_results/${id}-${backbone}-${method}-continual${tag_suffix}.csv"
mkdir -p "evaluation_results"
expected_header="Timestamp,InDataset,Backbone,MethodTag,Epochs,LR,LambdaPcon,LoRA_r,AdapterRoot,FinalStage,AVG-AUROC,AVG-FPR,Final-Top1,Average,Score"
if [[ ! -f "$results_csv" ]]; then
  echo "$expected_header" > "$results_csv"
else
  first_line=$(head -n 1 "$results_csv")
  if [[ "$first_line" != "$expected_header" ]]; then
    mv "$results_csv" "${results_csv}.backup"
    echo "$expected_header" > "$results_csv"
    echo "[Warning] Updated CSV header. Old file backed up to ${results_csv}.backup"
  fi
fi

# Helper函数：解析评估指标
parse_eval_metrics() {
  local eval_output="$1"
  local -n avg_fpr_ref="$2"
  local -n avg_auroc_ref="$3"
  local -n final_top1_ref="$4"
  local -n average_ref="$5"
  
  # 初始化
  avg_fpr_ref=""
  avg_auroc_ref=""
  final_top1_ref=""
  average_ref=""
  
  # 解析 AVG 行: "AVG           39.84  89.96  90.26" (FPR AUROC AUIN)
  if [[ "$eval_output" =~ AVG[[:space:]]+([0-9.]+)[[:space:]]+([0-9.]+)[[:space:]]+([0-9.]+) ]]; then
    avg_fpr_ref="${BASH_REMATCH[1]}"
    avg_auroc_ref="${BASH_REMATCH[2]}"
  fi
  
  # 解析增量指标: "[incremental] Final(Top-1): 0.9220  Average: 0.9220"
  # 使用更灵活的正则表达式，允许空格和制表符
  if [[ "$eval_output" =~ \[incremental\][[:space:]]+Final\(Top-1\):[[:space:]]+([0-9.]+)[[:space:]]+Average:[[:space:]]+([0-9.]+) ]]; then
    final_top1_ref="${BASH_REMATCH[1]}"
    average_ref="${BASH_REMATCH[2]}"
  else
    # 尝试更宽松的匹配（可能在不同行）
    if echo "$eval_output" | grep -q "\[incremental\]"; then
      # 尝试从多行中提取
      local incremental_line=$(echo "$eval_output" | grep "\[incremental\].*Final(Top-1)" | head -n 1)
      if [[ -n "$incremental_line" ]]; then
        if [[ "$incremental_line" =~ Final\(Top-1\):[[:space:]]+([0-9.]+) ]]; then
          final_top1_ref="${BASH_REMATCH[1]}"
        fi
        if [[ "$incremental_line" =~ Average:[[:space:]]+([0-9.]+) ]]; then
          average_ref="${BASH_REMATCH[1]}"
        fi
      fi
    fi
  fi
}

# Helper函数：执行持续学习并评估
run_continual_learning() {
  local epochs_val="$1"
  local lr_val="$2"
  local pcon_val="$3"
  local lora_r_val="$4"
  
  # 构建base_tag
  local base_tag="${method}-b${batch}-e${epochs_val}-lr${lr_val}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-pcon${pcon_val}-lora_r${lora_r_val}a${lora_alpha}d${lora_dropout}-temp${temp}"
  
  if [ "$orth_enable_before_c10" = "true" ] || [ "$orth_enable_after_c10" = "true" ]; then
    base_tag="${base_tag}-ol${lora_orth_lambda}"
  fi
  
  if [ "$incremental_flag" = "false" ]; then
    base_tag="${base_tag}-forget-from-${id}-to-${base_task}"
  elif [ "$incremental_flag" = "true" ]; then
    base_tag="${base_tag}-continual-from-${base_task}-to-${id}"
  fi
  
  if [ -n "$palm_mle_mode" ]; then
    base_tag="${base_tag}-palm_mle_${palm_mle_mode}"
  fi
  
  if [ -n "$pcon_inc" ]; then
    base_tag="${base_tag}-pcon_${pcon_inc}"
  fi
  
  if [ "$orth_enable_before_c10" != "$orth_enable_after_c10" ]; then
    base_tag="${base_tag}-ol_c10_${orth_enable_before_c10}_${orth_enable_after_c10}"
  fi
  
  # 适配器保存根目录
  local adapter_root="checkpoints/${id}-${backbone}-${base_tag}-stack"
  mkdir -p "$adapter_root"
  
  # 清理旧的增量评估曲线文件（如果存在），确保从头开始累积
  local curve_file_pattern="evaluation_results/inc_curve_${id}_${backbone}_${base_tag}.json"
  if [ -f "$curve_file_pattern" ]; then
    echo "[info] Removing old curve file: $curve_file_pattern"
    rm -f "$curve_file_pattern"
  fi
  
  # 断点续跑：检测已完成阶段
  local last_done=0
  local seen=""
  # for idx in "${!stages[@]}"; do
  #   local stage=$((idx+1))
  #   local d="${adapter_root}/stage${stage}"
  #   if [ -d "$d" ] && [ "$(find "$d" -type f | head -n 1 | wc -l)" -gt 0 ]; then
  #     local inc_csv="${stages[$idx]}"
  #     if [ -z "$seen" ]; then seen="$inc_csv"; else seen="$seen,$inc_csv"; fi
  #     last_done=$stage
  #   else
  #     break
  #   fi
  # done
  
  if [ "$last_done" -gt 0 ]; then
    echo "[resume] detected last completed stage = ${last_done}; seen={${seen}}"
  fi
  
  # 从 next_stage 开始持续增量
  local start_idx=$last_done
  for idx in $(seq $start_idx $((${#stages[@]} - 1))); do
    local stage=$((idx+1))
    local inc_csv="${stages[$idx]}"
    local method_tag_stage="${base_tag}-stage${stage}"
    local adapter_stage_dir="${adapter_root}/stage${stage}"
    mkdir -p "$adapter_stage_dir"
    
    echo "==== Stage ${stage}: inc={${inc_csv}}; seen={${seen}}; all(union)={${forget_classes_all}} ===="
    
    # 判断该阶段是否为 CIFAR-10
    local c10_flag=$(echo "$inc_csv" | tr ',' '\n' | awk '($1+0)>=100{print 1; exit}')
    local is_c10_stage=false
    if [ "$c10_flag" = "1" ]; then
      is_c10_stage=true
    fi
    local stage_orth_enable=$orth_enable_before_c10
    if [ "$is_c10_stage" = "true" ]; then
      stage_orth_enable=$orth_enable_after_c10
    fi
    
    python main.py --in-dataset $id --backbone $backbone --method $method \
      --epochs $epochs_val --load-path $pretrain_ckpt -b $batch --lr $lr_val --wd $wd \
      --cache-size $cache --lambda_pcon $pcon_val --proto_m $m --k $k --temp $temp \
      --use_lora --lora_impl peft --lora_r $lora_r_val --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
      $(
        # oLoRA（按阶段）
        if [ "$stage" -gt 1 ] && [ "$stage_orth_enable" = "true" ]; then
          local prev_dir="${adapter_root}/stage$((stage-1))"
          local refs_csv=""
          for j in $(seq 1 $((stage-1))); do
            local p="${adapter_root}/stage${j}"
            if [ -d "$p" ]; then
              if [ -z "$refs_csv" ]; then refs_csv="$p"; else refs_csv="$refs_csv,$p"; fi
            fi
          done
          if [ -d "$prev_dir" ]; then echo --adapter_load_path "$prev_dir" --lora_orth_enable --lora_orth_lambda $lora_orth_lambda --lora_orth_ref_paths "$refs_csv"; fi
        fi
      ) \
      $(
        # 非 oLoRA（按阶段）
        if [ "$stage" -gt 1 ] && [ "$stage_orth_enable" != "true" ]; then
          local prev_dir="${adapter_root}/stage$((stage-1))"
          if [ -d "$prev_dir" ]; then echo --adapter_load_path "$prev_dir"; fi
        fi
      ) \
      $(
        if [ "$incremental_flag" = "true" ]; then
          echo --incremental
        fi
      )\
      --forget_classes "$forget_classes_all" --forget_classes_inc "$inc_csv" $(if [ -n "$seen" ]; then echo --forget_classes_seen "$seen"; fi) \
      --batch_forget_mode $batch_forget_mode \
      $(
        if [ "$forget_enable" = "true" ]; then
          echo --forget_lambda $forget_lambda
        else
          echo --forget_lambda 0
        fi
      ) \
      --adapter_save_path "$adapter_stage_dir" $(if [ -n "$pcon_inc" ]; then echo --pcon_inc "$pcon_inc"; fi) $(if [ -n "$palm_mle_mode" ]; then echo --palm_mle_mode "$palm_mle_mode"; fi)
    
    # 评估：每个阶段训练后进行评估，更新增量评估曲线文件
    # 注意：必须包含 "-seen" 才能更新曲线文件
    local score="mahalanobis"
    if [ "$forget_enable" = "true" ]; then
      # 如果启用遗忘，进行完整评估
      local cum_seen="$seen"
      if [ -n "$cum_seen" ]; then cum_seen="$cum_seen,$inc_csv"; else cum_seen="$inc_csv"; fi
      # 增量指标（Overall + Old/New：old=seen, new=当前 inc）
      bash eval.sh $id "$ood" $backbone ${method_tag_stage}-seen-inc $pretrain_ckpt $score $cache 0 "$adapter_stage_dir" "" "" "" "$lora_r_val" "$lora_alpha" "$lora_dropout" "$lora_target" "" "" "" "$inc_csv" "$seen"
    else
      # 纯增量评测：仅输出增量指标（更新曲线文件）
      bash eval.sh $id "$ood" $backbone ${method_tag_stage}-seen-inc $pretrain_ckpt $score $cache 0 "$adapter_stage_dir" "" "" "" "$lora_r_val" "$lora_alpha" "$lora_dropout" "$lora_target" "" "" "" "$inc_csv" "$seen"
    fi
    
    # 更新 seen（累计）
    if [ -z "$seen" ]; then
      seen="$inc_csv"
    else
      seen="$seen,$inc_csv"
    fi
  done
  
  # 评估最后一个阶段（使用所有已学类）
  local final_stage=${#stages[@]}
  local adapter_stage_dir="${adapter_root}/stage${final_stage}"
  # 注意：method_tag 必须包含 "-seen" 才能输出增量指标（Final-Top1 和 Average）
  local method_tag_final="${base_tag}-stage${final_stage}-seen-final"
  local score="mahalanobis"
  
  # 获取最后一个阶段的增量类和之前所有阶段的seen
  local final_inc="${stages[$((final_stage-1))]}"
  local final_seen=""
  for idx in $(seq 0 $((final_stage-2))); do
    local inc_csv="${stages[$idx]}"
    if [ -z "$final_seen" ]; then final_seen="$inc_csv"; else final_seen="$final_seen,$inc_csv"; fi
  done
  
  # 执行评估并捕获输出
  # eval.sh参数: in_dataset out_datasets backbone method ckpt score cache epochs adapter_path forget_csv forget_list_path forget_lambda lora_r lora_alpha lora_dropout lora_target umap_enable umap_rf_only retain_exclude_csv forget_classes_inc forget_classes_seen
  local eval_output
  eval_output=$(bash eval.sh $id "$ood" $backbone ${method_tag_final} $pretrain_ckpt $score $cache 0 "$adapter_stage_dir" "" "" "" "$lora_r_val" "$lora_alpha" "$lora_dropout" "$lora_target" "" "" "" "$final_inc" "$final_seen" 2>&1)
  
  # 调试：输出评估结果的前几行和后几行（用于调试）
  echo "[DEBUG] Eval output (last 30 lines):" >&2
  echo "$eval_output" | tail -n 30 >&2
  
  # 解析指标
  local avg_fpr avg_auroc final_top1 average
  parse_eval_metrics "$eval_output" avg_fpr avg_auroc final_top1 average
  
  # 调试：输出解析结果
  echo "[DEBUG] Parsed: avg_fpr=$avg_fpr avg_auroc=$avg_auroc final_top1=$final_top1 average=$average" >&2
  
  # 计算score：四个指标权重相等
  # AVG-AUROC: 越大越好（已经是百分比，0-100）
  # AVG-FPR: 越小越好（已经是百分比，0-100），所以用 (100 - FPR) 来转换为越大越好
  # Final(Top-1): 越大越好（0-1之间），转换为百分比
  # Average: 越大越好（0-1之间），转换为百分比
  local score_val=""
  if [[ -n "$avg_auroc" && -n "$avg_fpr" && -n "$final_top1" && -n "$average" ]]; then
    # 将Final-Top1和Average转换为百分比
    local final_top1_pct=$(awk "BEGIN {printf \"%.2f\", $final_top1 * 100}")
    local average_pct=$(awk "BEGIN {printf \"%.2f\", $average * 100}")
    # 将FPR转换为"越大越好"的形式：100 - FPR
    local fpr_inverted=$(awk "BEGIN {printf \"%.2f\", 100 - $avg_fpr}")
    # Score = (AVG-AUROC + (100-AVG-FPR) + Final-Top1(%) + Average(%)) / 4
    # 四个指标权重相等，每个占25%
    score_val=$(awk "BEGIN {printf \"%.2f\", ($avg_auroc + $fpr_inverted + $final_top1_pct + $average_pct) / 4}")
  fi
  
  # 返回结果（通过全局变量）
  eval_avg_fpr="$avg_fpr"
  eval_avg_auroc="$avg_auroc"
  eval_final_top1="$final_top1"
  eval_average="$average"
  eval_score="$score_val"
  eval_adapter_root="$adapter_root"
  eval_final_stage="$final_stage"
}

# 主网格搜索循环
echo "[Grid Search Continual] Searching over: lambda_pcon(${lambda_list[*]}) × lrs(${lr_list[*]}) × epochs(${epoch_list[*]}) × lora_r(${lora_r_list[*]})"
echo "[Grid Search Continual] Stages: ${stages[*]}"

for pcon_val in "${lambda_list[@]}"; do
  for lr_val in "${lr_list[@]}"; do
    for ep in "${epoch_list[@]}"; do
      for lora_r_val in "${lora_r_list[@]}"; do
        echo ""
        echo "=========================================="
        echo "[Run] lambda_pcon=${pcon_val} lr=${lr_val} epochs=${ep} lora_r=${lora_r_val}"
        echo "=========================================="
        
        # 初始化评估指标
        eval_avg_fpr=""
        eval_avg_auroc=""
        eval_final_top1=""
        eval_average=""
        eval_score=""
        eval_adapter_root=""
        eval_final_stage=""
        
        # 执行持续学习
        run_continual_learning "$ep" "$lr_val" "$pcon_val" "$lora_r_val"
        
        # 记录结果
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        method_tag="${method}-b${batch}-e${ep}-lr${lr_val}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-pcon${pcon_val}-lora_r${lora_r_val}a${lora_alpha}d${lora_dropout}-temp${temp}"
        if [ "$orth_enable_before_c10" = "true" ] || [ "$orth_enable_after_c10" = "true" ]; then
          method_tag="${method_tag}-ol${lora_orth_lambda}"
        fi
        if [ "$incremental_flag" = "true" ]; then
          method_tag="${method_tag}-continual-from-${base_task}-to-${id}"
        fi
        if [ -n "$palm_mle_mode" ]; then
          method_tag="${method_tag}-palm_mle_${palm_mle_mode}"
        fi
        if [ -n "$pcon_inc" ]; then
          method_tag="${method_tag}-pcon_${pcon_inc}"
        fi
        if [ "$orth_enable_before_c10" != "$orth_enable_after_c10" ]; then
          method_tag="${method_tag}-ol_c10_${orth_enable_before_c10}_${orth_enable_after_c10}"
        fi
        method_tag="${method_tag}-stack"
        
        echo "$ts,$id,$backbone,$method_tag,$ep,$lr_val,$pcon_val,$lora_r_val,${eval_adapter_root},${eval_final_stage},${eval_avg_auroc:-},${eval_avg_fpr:-},${eval_final_top1:-},${eval_average:-},${eval_score:-}" >> "$results_csv"
        
        echo "[Result] AVG-AUROC=${eval_avg_auroc:-N/A} AVG-FPR=${eval_avg_fpr:-N/A} Final-Top1=${eval_final_top1:-N/A} Average=${eval_average:-N/A} Score=${eval_score:-N/A}"
      done
    done
  done
done

echo ""
echo "[Grid Search Continual] Completed. Results saved to: $results_csv"

