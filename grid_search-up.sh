#!/usr/bin/env bash
set -euo pipefail

# Full combinational grid search:
# iterate over all (lambda × lr × epochs × lora_r) combinations and save all results

# Required environment (edit as needed)
id="CIFAR-100"
ood_datasets=(SVHN places365 LSUN iSUN dtd)
backbone="resnet34"
k=5
cache=6
m=0.999
method="top${k}-palm-cache${cache}-ema${m}"

# base pretrained checkpoint
pretrain_ckpt="checkpoints/${id}-${backbone}-${method}-with-prototypes.pt"

# LoRA config
use_peft=1
# lora_r=8  # Now in search space
lora_alpha=32
lora_dropout=0.05
lora_target="both"

# Forget config
forget_csv="0,8,11,40,51,66,67,88,94,57"    # edit to your forget set
batch_forget_mode="balanced"                 # none|balanced|proportional|retain_only
temp=0.08

# Fixed defaults
batch=128
wd=1e-4

# Search spaces
lambda_list=($(seq 0.1 0.1 1.0))
# lambda_list=(0.2)
lr_list=(0.001)
# 起始5，步长5，终止50
# epoch_list=($(seq 5 5 50))
epoch_list=(50)
lora_r_list=(8)
# Helper to join OOD list
join_ood() { local IFS=" "; echo "$*"; }

tag_suffix="-epoch50-lambdagrid_runs"
# tag_suffix="-lambda0.2-epochgrid_runs"
# tag_suffix="-lora_r-grid_runs"
# Results manifest CSV (append-only)
# results_csv="evaluation_results/${id}-${backbone}-${method}-fullgrid_runs.csv"
results_csv="evaluation_results/${id}-${backbone}-${method}-${tag_suffix}.csv"
mkdir -p "evaluation_results"
expected_header="Timestamp,InDataset,Backbone,MethodTag,Epochs,LR,ForgetLambda,Batch,WD,Cache,K,ProtoM,UseLoRA,LoRA_r,LoRA_alpha,LoRA_dropout,LoRA_target,BatchForgetMode,Temp,AdapterPath,PretrainCkpt,OODs,AVG-FPR,AVG-AUROC,Forget-FPR,Forget-AUROC,Forget-AUIN,Retain-Acc,Score"
if [[ ! -f "$results_csv" ]]; then
  echo "$expected_header" > "$results_csv"
else
  # Check if header needs to be updated (if old format without metrics)
  first_line=$(head -n 1 "$results_csv")
  if [[ "$first_line" != "$expected_header" ]]; then
    # Backup old file and create new one with updated header
    mv "$results_csv" "${results_csv}.backup"
    echo "$expected_header" > "$results_csv"
    echo "[Warning] Updated CSV header. Old file backed up to ${results_csv}.backup"
  fi
fi

# Helper function to parse evaluation metrics from eval.sh output
parse_eval_metrics() {
  local eval_output="$1"
  local -n avg_fpr_ref="$2"
  local -n avg_auroc_ref="$3"
  local -n forget_fpr_ref="$4"
  local -n forget_auroc_ref="$5"
  local -n forget_auin_ref="$6"
  local -n retain_acc_ref="$7"
  
  # Initialize to empty
  avg_fpr_ref=""
  avg_auroc_ref=""
  forget_fpr_ref=""
  forget_auroc_ref=""
  forget_auin_ref=""
  retain_acc_ref=""
  
  # Parse AVG line: "AVG           39.84  89.96  90.26" (FPR AUROC AUIN)
  if [[ "$eval_output" =~ AVG[[:space:]]+([0-9.]+)[[:space:]]+([0-9.]+)[[:space:]]+([0-9.]+) ]]; then
    avg_fpr_ref="${BASH_REMATCH[1]}"
    avg_auroc_ref="${BASH_REMATCH[2]}"
    # AUIN is the third value, but we don't use it for AVG-AUIN
  fi
  
  # Parse Retain-Acc: "Retain-Acc: 0.7482"
  if [[ "$eval_output" =~ Retain-Acc:[[:space:]]*([0-9.]+) ]]; then
    retain_acc_ref="${BASH_REMATCH[1]}"
    # Convert to percentage (multiply by 100)
    retain_acc_ref=$(awk "BEGIN {printf \"%.2f\", $retain_acc_ref * 100}")
  fi
  
  # Parse Forget-as-OOD block: "FPR: XX.XX AUROC: XX.XX AUIN: XX.XX"
  # The output may span multiple lines, so we use a more flexible pattern
  if [[ "$eval_output" =~ FPR:[[:space:]]*([0-9.]+).*AUROC:[[:space:]]*([0-9.]+).*AUIN:[[:space:]]*([0-9.]+) ]]; then
    forget_fpr_ref="${BASH_REMATCH[1]}"
    forget_auroc_ref="${BASH_REMATCH[2]}"
    forget_auin_ref="${BASH_REMATCH[3]}"
  fi
}

train_and_eval() {
  local epochs="$1"; local lr="$2"; local flambda="$3"; local lora_r_val="$4"; local tag_suffix="$5"
  local method_tag="${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${flambda}-lora_r${lora_r_val}a${lora_alpha}d${lora_dropout}-temp${temp}${tag_suffix}"
  local adapter_path="checkpoints/${id}-${backbone}-${method_tag}-planB_adapter"

  # train LoRA adapters only
  # guard empty flambda
  local fl_for_train="$flambda"
  if [[ -z "$fl_for_train" ]]; then fl_for_train="0.1"; fi
  python main.py --in-dataset "$id" --backbone "$backbone" --method "$method" \
    --epochs "$epochs" --load-path "$pretrain_ckpt" -b "$batch" --lr "$lr" --wd "$wd" \
    --cache-size "$cache" --lambda_pcon 1.0 --proto_m "$m" --k "$k" \
    $( [[ "$use_peft" -eq 1 ]] && echo --use_lora --lora_impl peft ) --lora_r "$lora_r_val" --lora_alpha "$lora_alpha" --lora_dropout "$lora_dropout" --lora_target "$lora_target" \
    --forget_classes_inc "$forget_csv" --forget_lambda "$fl_for_train" --batch_forget_mode "$batch_forget_mode" --temp "$temp" \
    --adapter_save_path "$adapter_path"

  # evaluate with base ckpt + adapter and capture output
  local ood_joined
  ood_joined=$(join_ood "${ood_datasets[@]}")
  local eval_output
  eval_output=$(bash eval.sh "$id" "$ood_joined" "$backbone" "$method_tag" "$pretrain_ckpt" mahalanobis "$cache" 0 "$adapter_path" "$forget_csv" "" "$flambda" "$lora_r_val" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only "" 2>&1)
  
  # Parse metrics from eval output
  local avg_fpr avg_auroc forget_fpr forget_auroc forget_auin retain_acc
  parse_eval_metrics "$eval_output" avg_fpr avg_auroc forget_fpr forget_auroc forget_auin retain_acc
  
  # Calculate score: AVG-AUROC - 0.5*AVG-FPR - 0.5*Forget-FPR + 0.2*Retain-Acc
  local score=""
  if [[ -n "$avg_auroc" && -n "$avg_fpr" ]]; then
    local ffpr_val="${forget_fpr:-0.0}"
    local ra_val="${retain_acc:-0.0}"
    score=$(awk "BEGIN {printf \"%.2f\", $avg_auroc - 0.5 * $avg_fpr - 0.5 * $ffpr_val + 0.2 * $ra_val}")
  fi
  
  # Store metrics in global variables for later use
  eval_avg_fpr="$avg_fpr"
  eval_avg_auroc="$avg_auroc"
  eval_forget_fpr="$forget_fpr"
  eval_forget_auroc="$forget_auroc"
  eval_forget_auin="$forget_auin"
  eval_retain_acc="$retain_acc"
  eval_score="$score"
}

best_lambda=""; best_lambda_score="-1"
fixed_epochs=10
fixed_lr=0.001

echo "[Full Grid] Searching over: lambdas(${lambda_list[*]}) × lrs(${lr_list[*]}) × epochs(${epoch_list[*]}) × lora_r(${lora_r_list[*]})"
ood_joined=$(join_ood "${ood_datasets[@]}")

# tag_suffix="-fullgrid"

for fl in "${lambda_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    for ep in "${epoch_list[@]}"; do
      for lora_r_val in "${lora_r_list[@]}"; do
        method_tag="${method}-b${batch}-e${ep}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${fl}-lora_r${lora_r_val}a${lora_alpha}d${lora_dropout}-temp${temp}${tag_suffix}"
        adapter_path="checkpoints/${id}-${backbone}-${method_tag}-planB_adapter"

        echo "[Run] lambda=${fl} lr=${lr} epochs=${ep} lora_r=${lora_r_val}"
        # Initialize eval metrics to empty before each run
        eval_avg_fpr=""
        eval_avg_auroc=""
        eval_forget_fpr=""
        eval_forget_auroc=""
        eval_forget_auin=""
        eval_retain_acc=""
        eval_score=""
        
        train_and_eval "$ep" "$lr" "$fl" "$lora_r_val" "$tag_suffix"

        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        # Write CSV row with evaluation metrics
        echo "$ts,$id,$backbone,$method_tag,$ep,$lr,$fl,$batch,$wd,$cache,$k,$m,$use_peft,$lora_r_val,$lora_alpha,$lora_dropout,$lora_target,$batch_forget_mode,$temp,$adapter_path,$pretrain_ckpt,\"$ood_joined\",${eval_avg_fpr:-},${eval_avg_auroc:-},${eval_forget_fpr:-},${eval_forget_auroc:-},${eval_forget_auin:-},${eval_retain_acc:-},${eval_score:-}" >> "$results_csv"
      done
    done
  done
done

echo "[Full Grid] Completed. Manifest saved to: $results_csv"


