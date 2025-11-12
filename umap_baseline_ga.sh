#!/usr/bin/env bash

python scripts/umap_custom_visualization.py \
  --in-dataset CIFAR-100 \
  --backbone resnet34 \
  --method-tag top5-palm-cache6-ema0.999-baseline-ga-b128-e5-lr0.001-wd1e-4-fl1.0 \
  --pretrain-ckpt checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999-baseline_ga_forget.pt \
  --forget-classes 0,8,11,40,51,66,67,88,94,57 \
  --ood-datasets "SVHN places365 LSUN iSUN dtd" \
  --output-dir figs/umap_custom_baseline_ga \
  --outlier-quantile 0.2


