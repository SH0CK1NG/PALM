  python scripts/umap_custom_visualization.py \
    --in-dataset CIFAR-100 \
    --backbone resnet34 \
    --method-tag top5-palm-cache6-ema0.999-b128-e50-lr0.001-wd1e-4-ltboth-bfmbalanced-fl0.2-lora_r8a32d0.05-temp0.08-fullgrid \
    --pretrain-ckpt checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999-with-prototypes.pt \
    --adapter-path checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999-b128-e50-lr0.001-wd1e-4-ltboth-bfmbalanced-fl0.2-lora_r8a32d0.05-temp0.08-fullgrid-planB_adapter \
    --forget-classes 0,8,11,40,51,57,66,67,88,94 \
    --ood-datasets "SVHN places365 LSUN iSUN dtd" \
    --output-dir figs/umap_custom \
    --stochastic-dropout-scale 0 --outlier-quantile 0.2