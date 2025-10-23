#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate all combinations recorded in a fullgrid manifest CSV.
# Usage:
#   bash re_eval_fullgrid.sh [manifest_csv] [forget_csv] [score] [umap_flag] [umap_rf_only]
# Defaults:
#   manifest_csv: latest evaluation_results/*-fullgrid_runs.csv if not provided
#   forget_csv:   empty (不传则不计算 Retain-Acc/Forget-as-OOD)
#   score:        mahalanobis
#   umap_flag:    empty (非空则开启 --umap_enable)
#   umap_rf_only: empty (非空则开启 --umap_rf_only)

manifest_csv="${1:-}"
forget_csv="${2:-}"
score_name="${3:-mahalanobis}"
umap_flag="${4:-}"
umap_rf_only="${5:-}"

if [[ -z "$manifest_csv" ]]; then
  # pick the latest manifest if not specified
  manifest_csv=$(ls -t evaluation_results/*-fullgrid_runs.csv 2>/dev/null | head -n 1 || true)
  if [[ -z "$manifest_csv" ]]; then
    echo "[error] manifest_csv not provided and no *-fullgrid_runs.csv found under evaluation_results/" >&2
    exit 1
  fi
fi

if [[ ! -f "$manifest_csv" ]]; then
  echo "[error] manifest file not found: $manifest_csv" >&2
  exit 1
fi

mkdir -p logs
ts=$(date -u +%Y-%m-%d_%H%M%S)
base=$(basename "$manifest_csv" .csv)
log_file="logs/re_eval_fullgrid_${base}_${ts}.log"

echo "[re-eval] manifest: $manifest_csv" | tee -a "$log_file"
echo "[re-eval] forget_csv: ${forget_csv:-<empty>} | score: $score_name" | tee -a "$log_file"

# Read header to know column indices (robust against column order changes)
header=$(head -n 1 "$manifest_csv")

# helper: extract a column index by exact name (1-based); empty if not found
col_idx() {
  local name="$1"
  awk -v FS="," -v target="$name" 'NR==1{for(i=1;i<=NF;i++){if($i==target){print i; exit}}}' "$manifest_csv"
}

idx_InDataset=$(col_idx "InDataset")
idx_Backbone=$(col_idx "Backbone")
idx_MethodTag=$(col_idx "MethodTag")
idx_Epochs=$(col_idx "Epochs")
idx_Cache=$(col_idx "Cache")
idx_ForgetLambda=$(col_idx "ForgetLambda")
idx_LoRA_r=$(col_idx "LoRA_r")
idx_LoRA_alpha=$(col_idx "LoRA_alpha")
idx_LoRA_dropout=$(col_idx "LoRA_dropout")
idx_LoRA_target=$(col_idx "LoRA_target")
idx_AdapterPath=$(col_idx "AdapterPath")
idx_PretrainCkpt=$(col_idx "PretrainCkpt")
idx_OODs=$(col_idx "OODs")

# Verify required columns
for v in idx_InDataset idx_Backbone idx_MethodTag idx_Epochs idx_Cache idx_AdapterPath idx_PretrainCkpt idx_OODs; do
  if [[ -z "${!v}" ]]; then
    echo "[error] missing column index: $v" >&2
    echo "header: $header" >&2
    exit 1
  fi
done

# Iterate rows (skip header)
tail -n +2 "$manifest_csv" | while IFS= read -r line; do
  # Use gawk FPAT to handle quoted OODs field correctly
  parsed=$(awk 'BEGIN{FPAT="([^,]*)|(\"[^\"]+\")"} {print $0}' <<< "$line")
  # Extract needed fields by index and print as TSV
  tsv=$(awk -v FPAT='([^,]*)|(\"[^\"]+\")' \
    -v i1="$idx_InDataset" -v i2="$idx_Backbone" -v i3="$idx_MethodTag" \
    -v i4="$idx_Epochs" -v i5="$idx_Cache" -v i6="$idx_ForgetLambda" \
    -v i7="$idx_LoRA_r" -v i8="$idx_LoRA_alpha" -v i9="$idx_LoRA_dropout" \
    -v i10="$idx_LoRA_target" -v i11="$idx_AdapterPath" -v i12="$idx_PretrainCkpt" \
    -v i13="$idx_OODs" '
      {
        id=$i1; backbone=$i2; method_tag=$i3; epochs=$i4; cache=$i5; fl=$i6; lr_r=$i7; lr_a=$i8; lr_d=$i9; lt=$i10; ap=$i11; ckpt=$i12; ood=$i13;
        gsub(/^\"|\"$/, "", ood); # strip surrounding quotes if any
        # strip optional quotes around AdapterPath/MethodTag too (defensive)
        gsub(/^\"|\"$/, "", ap); gsub(/^\"|\"$/, "", method_tag);
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", id, ood, backbone, method_tag, ckpt, cache, epochs, ap, fl, lr_r, lr_a, lr_d, lt;
      }
    ' <<< "$line")

  # read TSV into variables
  IFS=$'\t' read -r id oods backbone method_tag ckpt cache epochs adapter_path flambda lora_r lora_alpha lora_dropout lora_target <<< "$tsv"
  # default LoRA target to 'both' if missing
  if [[ -z "$lora_target" ]]; then lora_target="both"; fi

  # Sanity checks
  if [[ -z "$id" || -z "$backbone" || -z "$method_tag" || -z "$ckpt" || -z "$adapter_path" ]]; then
    echo "[skip] malformed row: $line" | tee -a "$log_file"
    continue
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[warn] ckpt not found: $ckpt (id=$id method=$method_tag)" | tee -a "$log_file"
  fi
  if [[ ! -d "$adapter_path" && ! -f "$adapter_path" ]]; then
    echo "[warn] adapter path not found: $adapter_path (id=$id method=$method_tag)" | tee -a "$log_file"
  fi

  echo "========== RE-EVAL ==========" | tee -a "$log_file"
  echo "id=$id | backbone=$backbone | method=$method_tag" | tee -a "$log_file"
  echo "epochs=$epochs | cache=$cache | flambda=${flambda:-} | oods=[$oods]" | tee -a "$log_file"
  echo "adapter=$adapter_path | ckpt=$ckpt" | tee -a "$log_file"

  # Build optional flags
  extra_flags=()
  if [[ -n "$umap_flag" ]]; then extra_flags+=(--umap_enable); fi
  if [[ -n "$umap_rf_only" ]]; then extra_flags+=(--umap_rf_only); fi

  # Invoke eval.sh
  bash eval.sh "$id" "$oods" "$backbone" "$method_tag" "$ckpt" "$score_name" "$cache" "$epochs" \
    "$adapter_path" "$forget_csv" "" "${flambda:-}" "${lora_r:-}" "${lora_alpha:-}" "${lora_dropout:-}" "${lora_target:-}" \
    ${extra_flags[@]:-} 2>&1 | tee -a "$log_file"

done

echo "[re-eval] completed. Log: $log_file"


