# basic info
#id=CIFAR-100
id=CIFAR-110
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
#save_path=checkpoints/$id-$backbone-$method.pt
save_path=checkpoints/CIFAR-10-resnet34-top5-palm-cache6-ema0.999-finetuned-110.pt

#python main.py --in-dataset $id --backbone $backbone --method $method --epochs $epochs --save-path $save_path -b $batch --lr $lr --wd $wd --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k
python feature_extract.py --in-dataset $id  --out-datasets $ood --backbone $backbone --method $method --epochs $epochs --save-path $save_path --cache-size $cache
score="mahalanobis"
python eval_cifar.py --in-dataset $id --out-datasets $ood --backbone $backbone --method $method --epochs $epochs --save-path $save_path --score $score --cache-size $cache
# CUDA_VISIBLE_DEVICES=1 nohup bash runner.sh > logs/runner_$(date +%F_%H%M).log 2>&1 &