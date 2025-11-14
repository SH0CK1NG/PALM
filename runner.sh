#!/usr/bin/env bash
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

# # centers/precision (from compute_centers.py for CIFAR-100)
# center_dir=cache/resnet34-$method/$id
# centers_path=$center_dir/class_centers.pt
# precision_path=$center_dir/precision.pt

# forget first 10 classes (0-9) as example; editable
# #Plan B
# forget_csv="0,8,11,40,51,66,67,88,94,100" 
#CIFAR100 Plan B
# forget_csv="0,8,11,40,51,66,67,88,94,57" 

forget_class_num=10
case $forget_class_num in
  5)
    forget_csv="0,8,11,40,51"
    ;;
  10)
    forget_csv="0,8,11,40,51,66,67,88,94,57"
    ;;
  15)
    forget_csv="0,8,11,40,51,66,67,88,94,57,59,58,44,93,10"
    ;;
  20)
    forget_csv="0,8,11,40,51,66,67,88,94,57,59,58,44,93,10,64,22,42,9,90"
    ;;
esac

# forget_center_set="retain"
forget_lambda=0.2
lora_r=8
lora_alpha=32
lora_dropout=0.05
# where to inject lora: head|encoder|both|encoder_all|both_all
lora_target=head
# per-batch forget/retain composition: none|balanced|proportional
batch_forget_mode=balanced

forget_attr_w=4
forget_proto_rep_w=4

temp=0.08

forget_avgproto_w=1.0

forget_mode=0
# base_method_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}
base_method_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-${id}forget${forget_class_num}
case $forget_mode in
  1)
    method_tag=${base_method_tag}-temp${temp}-fpw${forget_avgproto_w}
    adapter_path=checkpoints/${id}-${backbone}-${method_tag}-forget_avgproto_enable-planB_adapter
    ;;
  2)
    method_tag=${base_method_tag}-fa${forget_attr_w}-fpr${forget_proto_rep_w}-temp${temp}
    adapter_path=checkpoints/${id}-${backbone}-${method_tag}-forget_proto_enable-planB_adapter
    ;;
  0|*)
    [ "$forget_mode" != "0" ] && echo "Invalid forget mode, using default forget mode (0)"
    method_tag=${base_method_tag}-temp${temp}
    adapter_path=checkpoints/${id}-${backbone}-${method_tag}-planB_adapter
    ;;
esac

# tag method for evaluation to avoid overwriting outputs (embed key hparams)
# method_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}
# method_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-temp${temp}-fpw${forget_avgproto_w}
# method_tag=${method}-b${batch}-e${epochs}-lr${lr}-wd${wd}-lt${lora_target}-bfm${batch_forget_mode}-fl${forget_lambda}-lora_r${lora_r}a${lora_alpha}d${lora_dropout}-fa${forget_attr_w}-fpr${forget_proto_rep_w}-temp${temp}

# where to save adapter only (PEFT uses a directory)
# adapter_path=checkpoints/${id}-${backbone}-${method_tag}-planB_adapter
# adapter_path=checkpoints/${id}-${backbone}-${method_tag}-forget_avgproto_enable-planB_adapter
# adapter_path=checkpoints/${id}-${backbone}-${method_tag}-forget_proto_enable-planB_adapter


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

# train with LoRA adapters only (base frozen), prototype-based forgetting on hypersphere (no centers/precision)
python main.py --in-dataset $id --backbone $backbone --method $method \
  --epochs $epochs --load-path $pretrain_ckpt -b $batch --lr $lr --wd $wd \
  --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k \
  --use_lora --lora_impl peft --lora_r $lora_r --lora_alpha $lora_alpha --lora_dropout $lora_dropout --lora_target $lora_target \
  --forget_classes_inc $forget_csv --forget_lambda $forget_lambda \
  --batch_forget_mode $batch_forget_mode \
  --temp $temp \
  --adapter_save_path $adapter_path \
  $(
    case $forget_mode in
      1)
        echo --forget_avgproto_enable --forget_avgproto_w $forget_avgproto_w
        ;;
      2)
        echo --forget_proto_enable --forget_attr_w $forget_attr_w --forget_proto_rep_w $forget_proto_rep_w
        ;;
      *)
        echo ""
        ;;
    esac
  )
  # --forget_avgproto_enable --forget_avgproto_w $forget_avgproto_w
  # --forget_proto_enable --forget_attr_w $forget_attr_w --forget_proto_rep_w $forget_proto_rep_w

# evaluate with base ckpt + adapter
score="mahalanobis"
bash eval.sh $id "$ood" $backbone $method_tag $pretrain_ckpt $score $cache 0 $adapter_path "$forget_csv" "" "$forget_lambda" "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target" --umap_enable  --umap_rf_only ""
# # 直接调用 feature_extract.py 和 eval_cifar.py，避免在 eval.sh 中触发增量路径导致遗忘集缓存缺失
# python feature_extract.py \
#   --in-dataset "$id" \
#   --out-datasets $ood \
#   --backbone "$backbone" \
#   --method "$method_tag" \
#   --save-path "$pretrain_ckpt" \
#   --load-path "$pretrain_ckpt" \
#   --epochs 0 \
#   -b "$batch" \
#   --cache-size "$cache" \
#   --temp "$temp" \
#   --proto_m "$m" \
#   --forget_csv "$forget_csv" \
#   --forget_lambda "$forget_lambda" \
#   --batch_forget_mode "$batch_forget_mode" \
#   --use_lora --lora_impl peft \
#   --adapter_load_path "$adapter_path" \
#   --lora_target "$lora_target" \
#   --lora_r "$lora_r" \
#   --lora_alpha "$lora_alpha" \
#   --lora_dropout "$lora_dropout"

# python eval_cifar.py \
#   --in-dataset "$id" \
#   --out-datasets $ood \
#   --backbone "$backbone" \
#   --method "$method_tag" \
#   --save-path "$pretrain_ckpt" \
#   --load-path "$pretrain_ckpt" \
#   --epochs 0 \
#   --cache-size "$cache" \
#   --score "$score" \
#   --forget_classes "$forget_csv" \
#   --forget_lambda "$forget_lambda" \
#   --batch_forget_mode "$batch_forget_mode" \
#   --use_lora --lora_impl peft \
#   --adapter_load_path "$adapter_path" \
#   --lora_target "$lora_target" \
#   --lora_r "$lora_r" \
#   --lora_alpha "$lora_alpha" \
#   --lora_dropout "$lora_dropout" \
#   --umap_enable \
#   --umap_rf_only

# # CUDA_VISIBLE_DEVICES=0 nohup bash runner.sh > logs/runner_$(date +%F_%H%M).log 2>&1 &



# # 手动调用特征提取与评估，避免在 eval.sh 中触发增量路径导致遗忘集缓存缺失
# python feature_extract.py \
#   --in-dataset "$id" \
#   --out-datasets $ood \
#   --backbone "$backbone" \
#   --method "$method_tag" \
#   --save-path "$pretrain_ckpt" \
#   --load-path "$pretrain_ckpt" \
#   --epochs "$epochs" \
#   -b "$batch" \
#   --cache-size "$cache" \
#   --temp "$temp" \
#   --proto_m "$m" \
#   --forget_csv "$forget_csv" \
#   --forget_lambda "$forget_lambda" \
#   --batch_forget_mode "$batch_forget_mode" \
#   --use_lora --lora_impl peft \
#   --adapter_load_path "$adapter_path" \
#   --lora_target "$lora_target" \
#   --lora_r "$lora_r" \
#   --lora_alpha "$lora_alpha" \
#   --lora_dropout "$lora_dropout"

# python eval_cifar.py \
#   --in-dataset "$id" \
#   --out-datasets $ood \
#   --backbone "$backbone" \
#   --method "$method_tag" \
#   --save-path "$pretrain_ckpt" \
#   --load-path "$pretrain_ckpt" \
#   --epochs "$epochs" \
#   --cache-size "$cache" \
#   --score "$score" \
#   --forget_classes "$forget_csv" \
#   --forget_lambda "$forget_lambda" \
#   --batch_forget_mode "$batch_forget_mode" \
#   --use_lora --lora_impl peft \
#   --adapter_load_path "$adapter_path" \
#   --lora_target "$lora_target" \
#   --lora_r "$lora_r" \
#   --lora_alpha "$lora_alpha" \
#   --lora_dropout "$lora_dropout"