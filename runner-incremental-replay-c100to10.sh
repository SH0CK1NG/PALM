#!/usr/bin/env bash
set -e

# Incremental LoRA + Replay: CIFAR-100 (old) -> CIFAR-10 (new) on CIFAR-110

# base settings
old_id=CIFAR-100
new_id=CIFAR-10
mix_id=CIFAR-110   # replay via mixed dataset (C100 + subsampled C10)

backbone=resnet34
batch=128
epochs=50
lr=0.001
wd=1e-4

# PALM proto settings
k=5
cache=6
m=0.999
temp=0.08
pcon=1.0

# LoRA (PEFT)
lora_r=8
lora_alpha=32
lora_dropout=0.05
lora_target=both

temp=0.08

# Balanced replay within batch for CIFAR-110 (only effective when forgetting enabled)
batch_forget_mode=balanced
forget_lambda=0.2
forget_enable=${FORGET_ENABLE:-true}

# paths
method=top${k}-palm-cache${cache}-ema${m}
base_ckpt=checkpoints/${old_id}-${backbone}-${method}-with-prototypes.pt
# Derived sets on CIFAR-110: new (C10) map to 100..109; old (C100) are 0..99
inc_csv=$(seq -s, 100 109)
seen_csv=$(seq -s, 0 99)
forget_classes_all="$inc_csv"
tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}-replay
adapter_path=checkpoints/${mix_id}-${backbone}-${tag}-inc-replay_adapter

echo "[Stage] Incremental LoRA + replay: ${old_id} -> ${new_id} on ${mix_id}"

# Train on mixed CIFAR-110 as replay (retain old + learn new) with LoRA adapters only
python main.py \
  --in-dataset ${mix_id} \
  --backbone ${backbone} \
  --method ${method} \
  --epochs ${epochs} -b ${batch} --lr ${lr} --wd ${wd} \
  --cache-size ${cache} --lambda_pcon ${pcon} --proto_m ${m} --k ${k} --temp ${temp} \
  --incremental \
  --load-path ${base_ckpt} \
  --use_lora --lora_impl peft --lora_r ${lora_r} --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} --lora_target ${lora_target} \
  --forget_classes "$forget_classes_all" --forget_classes_inc "$inc_csv" --forget_classes_seen "$seen_csv" \
  $(
    if [ "$forget_enable" = "true" ]; then
      echo --forget_lambda ${forget_lambda} --batch_forget_mode ${batch_forget_mode}
    else
      echo --forget_lambda 0
    fi
  ) \
  --adapter_save_path ${adapter_path}

echo "[Eval] Evaluate with LoRA adapter on ${mix_id}"
ood="SVHN places365 LSUN iSUN dtd"
# 1) Incremental metrics (Old/New/Overall) for stage j=1 (new=C10, old=C100)
bash eval.sh ${mix_id} "$ood" ${backbone} ${tag}-inc ${base_ckpt} mahalanobis ${cache} 0 ${adapter_path} "" "" "" ${lora_r} ${lora_alpha} ${lora_dropout} ${lora_target} "" "" "" "$inc_csv" "$seen_csv"
# 2) Optional forgetting metrics (Retain-Acc + Forget-as-OOD) treating C10 as forget set
if [ "$forget_enable" = "true" ]; then
  bash eval.sh ${mix_id} "$ood" ${backbone} ${tag}-forget ${base_ckpt} mahalanobis ${cache} 0 ${adapter_path} "$inc_csv" "" "${forget_lambda}" ${lora_r} ${lora_alpha} ${lora_dropout} ${lora_target}
fi

echo "Done. Adapter saved at: ${adapter_path}"


