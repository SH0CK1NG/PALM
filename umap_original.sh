#!/usr/bin/env bash

python scripts/umap_palm_original.py \
  --checkpoint "checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999.pt" \
  --method-tag top5-palm-cache6-ema0.999 \
  --output-dir figs/umap_original


