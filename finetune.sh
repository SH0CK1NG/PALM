'''
测试下限
'''
# basic info
#id=CIFAR-100
id=CIFAR-10
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
save_path=checkpoints/$id-$backbone-$method.pt

cd ~/PALM
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --in-dataset $id \
  --backbone $backbone \
  --method $method \
  --epochs 100 \
  --lr 0.01 \
  --batch-size 256 \
  --save-path checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999.pt

# python main.py --in-dataset $id --backbone $backbone --method $method --epochs $epochs --save-path $save_path -b $batch --lr $lr --wd $wd --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k
# python feature_extract.py --in-dataset $id  --out-datasets $ood --backbone $backbone --method $method --epochs $epochs --save-path $save_path --cache-size $cache
# score="mahalanobis"
# python eval_cifar.py --in-dataset $id --out-datasets $ood --backbone $backbone --method $method --epochs $epochs --save-path $save_path --score $score --cache-size $cache

# CUDA_VISIBLE_DEVICES=1 nohup bash finetune.sh > logs/finetune_$(date +%F_%H%M).log 2>&1 &