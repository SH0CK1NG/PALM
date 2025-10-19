# Upper-bound experiment: train PALM from scratch on retain classes only (no pretrained, no LoRA);
# evaluation uses retain as ID and includes forget as OOD along with standard OOD sets.

# basic info
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

# training info
batch=512
epochs=500
lr=0.5
wd=1e-6

backbone=resnet34
pcon=1.

m=0.999

# Runing PALM on supervised OOD detection
k=5
cache=6
method=top$k-palm-cache$cache-ema$m

# Save path for full model checkpoint (train from scratch, no pretrained)
save_path=checkpoints/${id}-${backbone}-${method}-upper-retain.pt

# Forget list (example: first 10 classes) used ONLY for evaluation splitting
forget_csv="0,1,2,3,4,5,6,7,8,9"
forget_center_set="retain"
forget_lambda=0

# Ensure batches only contain retain samples during training
batch_forget_mode=retain_only

# tag for outputs
method_tag=${method}-upper-retain

# 1) Train from scratch on retain-only batches (no LoRA, no pretrained ckpt)
python main.py --in-dataset $id --backbone $backbone --method $method \
  --epochs $epochs -b $batch --lr $lr --wd $wd \
  --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k \
  --batch_forget_mode $batch_forget_mode \
  --forget_classes $forget_csv --forget_center_set $forget_center_set \
  --save-path $save_path

# 2) Evaluate: use trained ckpt; retain(ID) vs standard OOD + forget(OOD)
score="mahalanobis"
bash eval.sh "$id" "$ood" "$backbone" "$method_tag" "$save_path" "$score" "$cache" "0" "" "$forget_csv" "" "$forget_center_set" "$forget_lambda"


