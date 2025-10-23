# Sequential two-phase forgetting with PALM + LoRA (avg-prototype forgetting enabled)

# basic info
id=CIFAR-100
ood="SVHN places365 LSUN iSUN dtd"

# training info
batch=128
epochs=50
lr=0.001
wd=1e-4

backbone=resnet34
pcon=1.

m=0.999

# Runing PALM on supervised OOD detection
k=5
cache=6
method=top$k-palm-cache$cache-ema$m

# base pretrained checkpoint (CIFAR-100)
pretrain_ckpt=checkpoints/$id-$backbone-$method-with-prototypes.pt

# sequential forgetting plan: split 10 classes -> 5 then cumulative 10
# full set (CIFAR100 Plan B)
forget_csv_all="0,8,11,40,51,66,67,88,94,57"
forget_csv_phase1="0,8,11,40,51"
forget_csv_phase2_new5="66,67,88,94,57"

# forgetting/lora settings
forget_lambda=0.2
lora_r=8
lora_alpha=32
lora_dropout=0.05
# where to inject lora: head|encoder|both|encoder_all|both_all
lora_target=both
# per-batch forget/retain composition: none|balanced|proportional
batch_forget_mode=balanced

temp=0.08
forget_avgproto_w=1.0

# tag method for evaluation to avoid overwriting outputs (embed key hparams)
method_tag_base=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfmb${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}-fpw${forget_avgproto_w}
method_tag_phase1=${method_tag_base}-seq_p1
method_tag_phase2=${method_tag_base}-seq_p2

# where to save/load adapters (PEFT uses directories)
adapter_path_phase1=checkpoints/${id}-${backbone}-${method_tag_phase1}-forget_avgproto_enable_adapter
adapter_path_phase2=checkpoints/${id}-${backbone}-${method_tag_phase2}-forget_avgproto_enable_adapter

echo "[seq] Phase 1: forgetting first 5 classes: ${forget_csv_phase1}"
python main.py --in-dataset $id --backbone $backbone --method $method \
  --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
  --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k \
  --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
  --forget_classes $forget_csv_phase1 --forget_lambda $forget_lambda \
  --batch_forget_mode $batch_forget_mode \
  --temp $temp \
  --adapter_save_path $adapter_path_phase1 
  # --forget_avgproto_enable --forget_avgproto_w $forget_avgproto_w

echo "[seq] Evaluate Phase 1 (forget=first5)"
score="mahalanobis"
bash eval.sh $id "$ood" $backbone $method_tag_phase1 $pretrain_ckpt $score $cache 0 $adapter_path_phase1 "$forget_csv_phase1" "" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only

echo "[seq] Phase 2: cumulative forgetting (first5 + new5): ${forget_csv_all} (loading phase1 adapter)"
python main.py --in-dataset $id --backbone $backbone --method $method \
  --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
  --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k \
  --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
  --forget_classes $forget_csv_phase2_new5  --forget_lambda $forget_lambda \
  --batch_forget_mode $batch_forget_mode \
  --temp $temp \
  --adapter_load_path $adapter_path_phase1 \
  --adapter_save_path $adapter_path_phase2 
  # --forget_avgproto_enable --forget_avgproto_w $forget_avgproto_w

echo "[seq] Evaluate Phase 2 (forget=all10)"
bash eval.sh $id "$ood" $backbone $method_tag_phase2 $pretrain_ckpt $score $cache 0 $adapter_path_phase2 "$forget_csv_all" "" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable --umap_rf_only

# Usage (example):
#   CUDA_VISIBLE_DEVICES=0 nohup bash sequential_forget.sh > logs/sequential_forget_$(date +%F_%H%M).log 2>&1 &


