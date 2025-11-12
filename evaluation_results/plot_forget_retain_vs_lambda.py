import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """
    读取CSV并按ForgetLambda聚合，输出包含均值与标准差的数据框。
    需要列：ForgetLambda, Forget-FPR, Retain-Acc
    """
    df = pd.read_csv(csv_path)
    required_cols = ["ForgetLambda", "Forget-FPR", "Retain-Acc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少必要列: {missing}")

    # 按lambda聚合，计算均值与标准差
    grouped = (
        df.groupby("ForgetLambda", as_index=False)
        .agg(
            ForgetFPR_mean=("Forget-FPR", "mean"),
            ForgetFPR_std=("Forget-FPR", "std"),
            RetainAcc_mean=("Retain-Acc", "mean"),
            RetainAcc_std=("Retain-Acc", "std"),
            count=("Forget-FPR", "count"),
        )
        .sort_values("ForgetLambda")
    )
    return grouped


def plot_forget_retain_vs_lambda(
    agg_df: pd.DataFrame,
    title: str,
    out_prefix: str,
    style: str = "whitegrid",
) -> None:
    """
    使用seaborn绘制 Forget FPR95 (↓) 与 Retain-Acc (↑) 随 λunlearn 变化的曲线。
    - 左Y轴：Forget FPR95（即CSV中的 Forget-FPR）
    - 右Y轴：Retain-Acc
    保存为PNG与SVG。
    """
    sns.set_style(style)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    lambda_vals = agg_df["ForgetLambda"].values
    fpr_mean = agg_df["ForgetFPR_mean"].values
    fpr_std = agg_df["ForgetFPR_std"].values
    acc_mean = agg_df["RetainAcc_mean"].values
    acc_std = agg_df["RetainAcc_std"].values

    # 左轴：Forget FPR95
    color_fpr = sns.color_palette("deep")[0]
    ax1.plot(lambda_vals, fpr_mean, color=color_fpr, marker="o", label="Forget FPR95 (↓)")
    if not np.isnan(fpr_std).all():
        ax1.fill_between(
            lambda_vals,
            fpr_mean - fpr_std,
            fpr_mean + fpr_std,
            color=color_fpr,
            alpha=0.2,
            linewidth=0,
        )
    ax1.set_xlabel("λ_unlearn")
    ax1.set_ylabel("Forget FPR95 (↓)", color=color_fpr)
    ax1.tick_params(axis="y", labelcolor=color_fpr)

    # 右轴：Retain-Acc
    ax2 = ax1.twinx()
    color_acc = sns.color_palette("deep")[2]
    ax2.plot(lambda_vals, acc_mean, color=color_acc, marker="s", label="Retain-Acc (↑)")
    if not np.isnan(acc_std).all():
        ax2.fill_between(
            lambda_vals,
            acc_mean - acc_std,
            acc_mean + acc_std,
            color=color_acc,
            alpha=0.2,
            linewidth=0,
        )
    ax2.set_ylabel("Retain-Acc (↑)", color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    # 网格与标题
    ax1.grid(True, linestyle="--", alpha=0.3)
    plt.title(title)
    plt.tight_layout()

    # 输出
    png_path = f"{out_prefix}.png"
    svg_path = f"{out_prefix}.svg"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, dpi=300, bbox_inches="tight", format="svg")
    print(f"保存成功: {png_path}")
    print(f"保存成功: {svg_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Forget FPR95 and Retain-Acc vs λ_unlearn")
    parser.add_argument(
        "--csv",
        type=str,
        default="PALM/evaluation_results/re_eval_fullgrid_CIFAR-100-resnet34-top5-palm-cache6-ema0.999-fullgrid_runs_2025-10-21_060120_summary.csv",
        help="CSV文件路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="PALM/evaluation_results/forget_retain_vs_lambda",
        help="输出文件前缀（不带扩展名）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Forget FPR95 (↓) and Retain-Acc (↑) vs λ_unlearn",
        help="图标题",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"找不到CSV文件: {args.csv}")

    agg_df = load_and_aggregate(args.csv)
    if agg_df.empty:
        raise ValueError("聚合结果为空，请检查CSV内容与筛选条件。")

    plot_forget_retain_vs_lambda(
        agg_df=agg_df,
        title=args.title,
        out_prefix=args.out,
    )


if __name__ == "__main__":
    main()


