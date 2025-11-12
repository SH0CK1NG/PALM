#!/usr/bin/env python3
"""UMAP visualisation for the original PALM checkpoint (retain vs OOD only).

This script reuses the utilities from ``umap_custom_visualization.py`` but
assumes there is no explicit forget subset. It extracts features when missing
and then renders 2×N figures (ID class subsets × OOD datasets).

Example usage
-------------

    python scripts/umap_palm_original.py \
        --checkpoint "checkpoints/original CIFAR-100-resnet34-top5-palm-cache6-ema0.999.pt" \
        --method-tag top5-palm-cache6-ema0.999 \
        --output-dir figs/umap_original
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.umap_custom_visualization import (
    build_embeddings,
    ensure_feature_caches,
    parse_int_list,
    parse_str_list,
    plot_embeddings,
    run_umap,
    sanitize_filename,
    stable_int,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP for original PALM (retain vs OOD)")
    parser.add_argument("--in-dataset", default="CIFAR-100", help="In-distribution dataset name")
    parser.add_argument("--backbone", default="resnet34", help="Backbone name")
    parser.add_argument("--method-tag", default="top5-palm-cache6-ema0.999", help="Method tag used in cache directory")
    parser.add_argument("--checkpoint", required=True, help="Path to the pretrained PALM checkpoint")
    parser.add_argument("--ood-datasets", default="SVHN places365 LSUN iSUN dtd", help="Space/comma separated OOD dataset names")
    parser.add_argument("--cache-root", default="cache", help="Root directory of cached features")
    parser.add_argument("--output-dir", default="figs/umap_original", help="Directory to save UMAP figures")
    parser.add_argument("--id-class-counts", default="20,90", help="Comma separated ID class counts to sample")
    parser.add_argument("--id-per-class", type=int, default=60, help="Max samples per ID class (0 = all)")
    parser.add_argument("--ood-limit", type=int, default=1200, help="Max OOD samples to plot")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--cache-size", type=int, default=6, help="Cache size argument forwarded to feature extraction")
    parser.add_argument("--gpu", default="0", help="CUDA visible device for feature extraction")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.05, help="UMAP min_dist parameter")
    parser.add_argument("--metric", default="cosine", help="UMAP metric")
    parser.add_argument("--outlier-quantile", type=float, default=1.0, help="Quantile for distance-based trimming (0-1]")
    parser.add_argument("--stochastic-dropout-scale", type=float, default=0.0, help="Scale controlling probabilistic removal outside ellipse")
    parser.add_argument("--no-ellipses", action="store_true", help="Disable covariance ellipses")

    args = parser.parse_args()

    ood_names = parse_str_list(args.ood_datasets)
    if not ood_names:
        raise ValueError("At least one OOD dataset must be provided")
    id_counts = parse_int_list(args.id_class_counts)
    if not id_counts:
        raise ValueError("At least one ID class count must be provided")

    ensure_feature_caches(
        cache_root=args.cache_root,
        backbone=args.backbone,
        method_tag=args.method_tag,
        in_dataset=args.in_dataset,
        ood_names=ood_names,
        forget_csv=None,
        pretrain_ckpt=args.checkpoint,
        adapter_path=None,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        seed=args.seed,
        gpu=args.gpu,
        lora_target="head",
        lora_r=0,
        lora_alpha=0,
        lora_dropout=0.0,
        lora_impl="native",
    )

    base_filename = sanitize_filename(
        f"{args.in_dataset}_{args.backbone}_{args.method_tag}_original"
    )

    for id_count in id_counts:
        for ood_dataset in ood_names:
            compound_seed = args.seed + id_count * 31 + stable_int(ood_dataset) % 10007
            embeddings = build_embeddings(
                cache_root=args.cache_root,
                backbone=args.backbone,
                method_tag=args.method_tag,
                in_dataset=args.in_dataset,
                forget_classes=[],
                id_class_count=id_count,
                ood_dataset=ood_dataset,
                seed=compound_seed,
                id_per_class=args.id_per_class,
                ood_limit=args.ood_limit,
                forget_limit=0,
            )

            stack_inputs = [embeddings.id_embeddings, embeddings.ood_embeddings]
            counts = [embeddings.id_embeddings.shape[0], embeddings.ood_embeddings.shape[0], None]

            umap_coords = run_umap(
                stack_inputs,
                metric=args.metric,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                seed=compound_seed,
            )

            id_label = f"ID ({id_count} classes)"
            ood_label = f"OOD ({ood_dataset})"
            forget_label = "Forget"

            filename = f"umap_original_{base_filename}_id{id_count}_{sanitize_filename(ood_dataset)}.png"
            output_path = os.path.join(args.output_dir, filename)
            title = (
                f"UMAP {args.in_dataset} · {args.backbone}\n"
                f"{args.method_tag} (original)\nID {id_count} cls vs OOD {ood_dataset}"
            )

            rng = np.random.default_rng(compound_seed + 7919)

            plot_embeddings(
                umap_coords,
                counts=(counts[0], counts[1], counts[2]),
                id_label=id_label,
                ood_label=ood_label,
                forget_label=forget_label,
                title=title,
                output_path=output_path,
                ellipse=(not args.no_ellipses),
                outlier_quantile=float(args.outlier_quantile),
                stochastic_dropout_scale=float(args.stochastic_dropout_scale),
                rng=rng,
            )

            print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()


