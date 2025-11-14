# 旧版网格搜索脚本，不能自动填结果，需要re_eval
#!/usr/bin/env bash
set -euo pipefail

# Full combinational grid search:
# iterate over all (lambda × lr × epochs) combinations and save all results

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
lora_r=8
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
lr_list=(0.001)
# 起始5，步长5，终止50
epoch_list=($(seq 5 5 50))

# Helper to join OOD list
join_ood() { local IFS=" "; echo "$*"; }

# Results manifest CSV (append-only)
# results_csv="evaluation_results/${id}-${backbone}-${method}-fullgrid_runs.csv"
results_csv="evaluation_results/${id}-${backbone}-${method}-fullgrid_reruns.csv"
mkdir -p "evaluation_results"
if [[ ! -f "$results_csv" ]]; then
  echo "Timestamp,InDataset,Backbone,MethodTag,Epochs,LR,ForgetLambda,Batch,WD,Cache,K,ProtoM,UseLoRA,LoRA_r,LoRA_alpha,LoRA_dropout,LoRA_target,BatchForgetMode,Temp,AdapterPath,PretrainCkpt,OODs" > "$results_csv"
fi

train_and_eval() {
  local epochs="$1"; local lr="$2"; local flambda="$3"; local tag_suffix="$4"
  local method_tag="${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${flambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}${tag_suffix}"
  local adapter_path="checkpoints/${id}-${backbone}-${method_tag}-planB_adapter"

  # train LoRA adapters only
  # guard empty flambda
  local fl_for_train="$flambda"
  if [[ -z "$fl_for_train" ]]; then fl_for_train="0.1"; fi
  python main.py --in-dataset "$id" --backbone "$backbone" --method "$method" \
    --epochs "$epochs" --load-path "$pretrain_ckpt" -b "$batch" --lr "$lr" --wd "$wd" \
    --cache-size "$cache" --lambda_pcon 1.0 --proto_m "$m" --k "$k" \
    $( [[ "$use_peft" -eq 1 ]] && echo --use_lora --lora_impl peft ) --lora_r "$lora_r" --lora_alpha "$lora_alpha" --lora_dropout "$lora_dropout" --lora_target "$lora_target" \
    --forget_classes "$forget_csv" --forget_lambda "$fl_for_train" --batch_forget_mode "$batch_forget_mode" --temp "$temp" \
    --adapter_save_path "$adapter_path"

  # evaluate with base ckpt + adapter
  local ood_joined
  ood_joined=$(join_ood "${ood_datasets[@]}")
  bash eval.sh "$id" "$ood_joined" "$backbone" "$method_tag" "$pretrain_ckpt" mahalanobis "$cache" 0 "$adapter_path" "$forget_csv" "" "$flambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only ""
}

best_lambda=""; best_lambda_score="-1"
fixed_epochs=10
fixed_lr=0.001

echo "[Full Grid] Searching over: lambdas(${lambda_list[*]}) × lrs(${lr_list[*]}) × epochs(${epoch_list[*]})"
ood_joined=$(join_ood "${ood_datasets[@]}")

# tag_suffix="-fullgrid"
for fl in "${lambda_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    for ep in "${epoch_list[@]}"; do
      tag_suffix="-fullgrid_reruns"
      method_tag="${method}-b${batch}-e${ep}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${fl}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}${tag_suffix}"
      adapter_path="checkpoints/${id}-${backbone}-${method_tag}-planB_adapter"

      echo "[Run] lambda=${fl} lr=${lr} epochs=${ep}"
      train_and_eval "$ep" "$lr" "$fl" "$tag_suffix"

      ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      echo "$ts,$id,$backbone,$method_tag,$ep,$lr,$fl,$batch,$wd,$cache,$k,$m,$use_peft,$lora_r,$lora_alpha,$lora_dropout,$lora_target,$batch_forget_mode,$temp,$adapter_path,$pretrain_ckpt,\"$ood_joined\"" >> "$results_csv"
    done
  done
done

echo "[Full Grid] Completed. Manifest saved to: $results_csv"


