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
#save_path=checkpoints/$id-$backbone-$method-with-prototypes.pt
save_path=checkpoints/CIFAR-10-resnet34-top5-palm-cache6-ema0.999-with-prototypes.pt
#save_path=checkpoints/残差空间位置错误CIFAR-10-resnet34-top5-palm-cache6-ema0.999-with-prototypes.pt

# 指定已预计算的残差空间（来自 CIFAR-100）
res_dir=cache/resnet34-top5-palm-cache6-ema0.999/CIFAR-100
proj_path=$res_dir/residual_projector.pt   # 或 basis_path=$res_dir/residual_basis.pt

#python main.py --in-dataset $id --backbone $backbone --method $method --epochs $epochs --save-path $save_path -b $batch --lr $lr --wd $wd --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k 
# python main.py --in-dataset $id --backbone $backbone --method $method --epochs $epochs --save-path $save_path -b $batch --lr $lr --wd $wd --cache-size $cache --lambda_pcon $pcon --proto_m $m --k $k --incremental --residual_space --residual_at encoder  --residual_projector_path $proj_path
python feature_extract.py --in-dataset $id  --out-datasets $ood --backbone $backbone --method $method --epochs $epochs --save-path $save_path --cache-size $cache
score="mahalanobis"
python eval_cifar.py --in-dataset $id --out-datasets $ood --backbone $backbone --method $method --epochs $epochs --save-path $save_path --score $score --cache-size $cache
# # CUDA_VISIBLE_DEVICES=0 nohup bash runner.sh > logs/runner_$(date +%F_%H%M).log 2>&1 &
