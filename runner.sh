# basic info
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

# training info
batch=512
epochs=5
lr=0.5
wd=1e-6

backbone=resnet34
pcon=1.

m=0.999

# Runing PALM on supervised OOD detection
k=5
cache=6
method=top$k-palm-cache$cache-ema$m

# base pretrained checkpoint (CIFAR-100)
pretrain_ckpt=/home/shaokun/PALM/checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999-with-prototypes.pt

# centers/precision (from compute_centers.py for CIFAR-100)
center_dir=cache/resnet34-$method/$id
centers_path=$center_dir/class_centers.pt
precision_path=$center_dir/precision.pt

# forget first 10 classes (0-9) as example; editable
forget_csv="0,1,2,3,4,5,6,7,8,9"
forget_center_set="retain"
forget_lambda=0.1
lora_r=8
lora_alpha=32
lora_dropout=0.05
# where to inject lora: head|encoder|both
lora_target=both
# per-batch forget/retain composition: none|balanced|proportional
batch_forget_mode=balanced
# where to save adapter only (PEFT uses a directory)
adapter_path=checkpoints/${id}-${backbone}-${method}-${batch_forget_mode}-${forget_center_set}-${forget_lambda}-0to9-${lora_target}_adapter



# tag method for evaluation to avoid overwriting outputs (embed key hparams)
method_tag=${method}-lt${lora_target}-bfm${batch_forget_mode}-fcs${forget_center_set}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}

# # 0) recompute class centers/precision every run to ensure availability/consistency
# python compute_centers.py \
#   --in-dataset $id \
#   --backbone $backbone \
#   --method $method \
#   --load-path $pretrain_ckpt

# # train with LoRA adapters only (base frozen), push forget samples away from centers of all classes
# python main.py --in-dataset $id --backbone $backbone --method $method \
#   --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
#   --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k \
#   --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
#   --centers_path $centers_path --precision_path $precision_path \
#   --forget_classes $forget_csv --forget_center_set $forget_center_set --forget_lambda $forget_lambda \
#   --batch_forget_mode $batch_forget_mode \
#   --adapter_save_path $adapter_path --forget_margin 100

# evaluate with base ckpt + adapter
score="mahalanobis"
bash eval.sh $id "$ood" $backbone $method_tag $pretrain_ckpt $score $cache 0 $adapter_path "$forget_csv" "" "$forget_center_set" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable  --umap_rf_only

# # CUDA_VISIBLE_DEVICES=0 nohup bash runner.sh > logs/runner_$(date +%F_%H%M).log 2>&1 &
