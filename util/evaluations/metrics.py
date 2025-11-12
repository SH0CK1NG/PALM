from __future__ import print_function
import argparse
import torch

from scipy import misc
import numpy as np
import sklearn.metrics as sk
from util.loaders.model_loader import get_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Sequence, Optional, Union


def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

    results = dict()

    # FPR
    mtype = 'FPR'
    # fpr95=np.sum(novel < np.percentile(known, 95)) / len(novel)
    fpr95=np.sum(novel > np.percentile(known, 5)) / len(novel)
    # results[mtype] = fpr_at_tpr95
    results[mtype] = fpr95

    # AUROC
    # mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    # results[mtype] = -np.trapz(1.-fpr, tpr)
    mtype = 'AUROC'
    # labels = [0] * len(known) + [1] * len(novel)
    labels = [1] * len(known) + [0] * len(novel)
    data = np.concatenate((known, novel))
    auroc = sk.roc_auc_score(labels, data)
    results[mtype] = auroc
    

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results


def split_acc_by_classes(labels: np.ndarray, preds: np.ndarray, forget_classes: Sequence[int]):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    fset = set(int(x) for x in forget_classes)
    fmask = np.isin(labels, list(fset))
    rmask = ~fmask
    def _acc(mask):
        if mask.sum() == 0:
            return float('nan')
        return float((preds[mask] == labels[mask]).mean())
    return _acc(fmask), _acc(rmask)


def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + in_dataset)
    print('out_distribution: ' + out_dataset)
    print('Model Name: ' + name)
    print('')

    print(' OOD detection method: ' + method)
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print('')

def print_all_results_tab(results, datasets, method):
    [print('{:6.2f}\t{:6.2f}\t{:6.2f}\t'.format(100.*result['FPR'], 100. *
           result['AUROC'], 100.*result['AUIN']), end='') for result in results]


def print_all_results(results, datasets, method):
    # ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    mtypes = ['FPR', 'AUROC', 'AUIN']
    avg_results = compute_average_results(results)
    print(' OOD detection method: ' + method)
    print('             ', end='')
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    for result, dataset in zip(results, datasets):
        print('\n{dataset:12s}'.format(dataset=dataset), end='')
        print(' {val:6.2f}'.format(val=100.*result['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*result['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*result['AUIN']), end='')

    print('\nAVG         ', end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['AUIN']), end='')
    print()


def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results

def compute_in(base_dir, in_dataset, method, name):

    known_nat = np.load(
        f'{base_dir}/{method}/{in_dataset}/{name}_in_scores.npy')
    known_nat_sorted = np.sort(known_nat)
    num_k = known_nat.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_nat_sorted[round(0.05 * num_k)]

    known_nat_label = np.load(
        f'{base_dir}/{method}/{in_dataset}/{name}_in_labels.npy')

    nat_in_cond = (known_nat > threshold).astype(np.float32)
    nat_correct = (known_nat_label[:, 0] ==
                   known_nat_label[:, 1]).astype(np.float32)
    nat_conf = np.mean(known_nat_label[:, 2])
    known_nat_cond_acc = np.sum(
        nat_correct * nat_in_cond) / max(np.sum(nat_in_cond), 1)
    known_nat_acc = np.mean(nat_correct)
    known_nat_cond_fnr = np.sum(
        nat_correct * (1.0 - nat_in_cond)) / max(np.sum(nat_correct), 1)
    known_nat_fnr = np.mean((1.0 - nat_in_cond))
    known_nat_eteacc = np.mean(nat_correct * nat_in_cond)

    print('FNR: {fnr:6.2f}, Acc: {acc:6.2f}, End-to-end Acc: {eteacc:6.2f}'.format(
        fnr=known_nat_fnr*100, acc=known_nat_acc*100, eteacc=known_nat_eteacc*100))

    return


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out



def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[
        fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def plot_kde(
    id_scores: Union[np.ndarray, Sequence[float]],
    ood_scores: Union[np.ndarray, Sequence[float]],
    id_label: str = 'ID',
    ood_label: str = 'OOD',
    title: str = 'KDE of Scores',
    save_path: Optional[str] = None,
    bandwidth: Optional[float] = None,
) -> None:
    """
    绘制 ID 与 OOD 分数的核密度估计曲线（直方图的平滑版）。
    - 横轴：Score（越大越像 ID）
    - 纵轴：Density（单条曲线下面积约为 1）
    - 两条曲线分别代表 ID 与 OOD 分布趋势
    额外：在图中标注 FPR@95% 对应的阈值（ID 分数第 5 百分位）。
    """
    id_scores = np.asarray(id_scores).ravel().astype(np.float64)
    ood_scores = np.asarray(ood_scores).ravel().astype(np.float64)

    # 计算 FPR@95% 的阈值（ID 分数第 5 百分位）
    thr = float(np.percentile(id_scores, 5))

    # 使用 seaborn 绘制 KDE，确保每条曲线各自归一化
    plt.figure(figsize=(6, 4))
    sns.set_palette('Set2')
    kde_kwargs = {}
    if bandwidth is not None:
        kde_kwargs['bw_adjust'] = float(bandwidth)
    ax = sns.kdeplot(id_scores, common_norm=False, fill=True, linewidth=2, label=id_label, **kde_kwargs)
    sns.kdeplot(ood_scores, common_norm=False, fill=True, linewidth=2, label=ood_label, **kde_kwargs)

    # 标注阈值线与箭头说明
    plt.axvline(thr, color='gray', linestyle='--', linewidth=1)
    plt.annotate(
        'FPR@95%',
        xy=(thr, ax.get_ylim()[1] * 0.6),
        xytext=(thr, ax.get_ylim()[1] * 0.9),
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center', va='bottom', color='gray', fontsize=9
    )

    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="CIFAR-10",
                        type=str, help='in-distribution dataset')
    parser.add_argument('--name', default="resnet18", type=str,
                        help='neural network name and training set')
    parser.add_argument('--method', default='energy',
                        type=str, help='odin mahalanobis')
    parser.add_argument('--base-dir', default='output/ood_scores',
                        type=str, help='result directory')
    parser.add_argument('--epsilon', default=8, type=int, help='epsilon')

    parser.set_defaults(argument=True)

    args = parser.parse_args()

    np.random.seed(1)

    out_datasets = ['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']
