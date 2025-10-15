# Evaluate only: load trained CIFAR-100 weights and evaluate ID=retain vs standard OOD + forget(OOD)

# basic info
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

backbone=resnet34

# PALM method tag base
k=5
cache=6
m=0.999
method=top$k-palm-cache$cache-ema$m

# trained checkpoint to evaluate (change if needed)
ckpt=/home/shaokun/PALM/checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999-with-prototypes.pt

# forget classes (treated as OOD in eval), keep retain as ID
forget_csv="0,1,2,3,4,5,6,7,8,9"
forget_center_set="retain"
forget_lambda=0

# optional: adapter directory for PEFT/LoRA (leave empty to disable)
adapter_path=""
lora_r=""
lora_alpha=""
lora_dropout=""
lora_target=""

# distinct tag for evaluation outputs/caches
method_tag=${method}-eval-retainforget

score="mahalanobis"

# Run evaluation only (feature extraction + metrics)
bash eval.sh "$id" "$ood" "$backbone" "$method_tag" "$ckpt" "$score" "$cache" "0" \
  "$adapter_path" "$forget_csv" "" "$forget_center_set" "$forget_lambda" \
  "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target"


