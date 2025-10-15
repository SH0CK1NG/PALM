import time
import os
import json
from util.evaluations import metrics
import torch
import numpy as np
from util.evaluations.write_to_csv import write_csv
from util.evaluations.metrics import plot_kde

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_maha(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    class_num = 10 if args.in_dataset == "CIFAR-10" else 100
    
    feat_log = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/train_{args.backbone}-{args.method}_features.npy", allow_pickle=True)
    label_log = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/train_{args.backbone}-{args.method}_labels.npy", allow_pickle=True)
    feat_log = feat_log.astype(np.float32)

    feat_log_val = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/val_{args.backbone}-{args.method}_features.npy", allow_pickle=True)
    label_log_val = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/val_{args.backbone}-{args.method}_labels.npy", allow_pickle=True)
    feat_log_val = feat_log_val.astype(np.float32)

    ood_feat_log_all = {}
    for ood_dataset in args.out_datasets:
        ood_feat_log = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/{ood_dataset}/{args.backbone}-{args.method}_features.npy", allow_pickle=True)
        ood_label_log = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/{ood_dataset}/{args.backbone}-{args.method}_labels.npy", allow_pickle=True)
        ood_feat_log = ood_feat_log.astype(np.float32)
        ood_feat_log_all[ood_dataset] = ood_feat_log

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only

    ftrain = prepos_feat(feat_log)
    ftest = prepos_feat(feat_log_val)
    food_all = {}
    for ood_dataset in args.out_datasets:
        food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])


# #################### SSD+ score OOD detection #################
    begin = time.time()
    mean_feat = ftrain.mean(0)
    std_feat = ftrain.std(0)
    prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
    ftrain_ssd = prepos_feat_ssd(ftrain)
    ftest_ssd = prepos_feat_ssd(ftest)
    food_ssd_all = {}
    for ood_dataset in args.out_datasets:
        food_ssd_all[ood_dataset] = prepos_feat_ssd(food_all[ood_dataset])
    
    cov = lambda x: np.cov(x.T, bias=True)
    # 在标准化空间（_ssd）中计算共享协方差逆
    inv_sigma = np.linalg.pinv(cov(ftrain_ssd))

    # === 类条件中心（标准化空间） ===
    C = int(label_log.max()) + 1
    D = ftrain_ssd.shape[1]
    centers_ssd = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        idx = (label_log == c)
        if np.any(idx):
            centers_ssd[c] = ftrain_ssd[idx].mean(axis=0)

    # 只使用保留类中心或全部类中心
    use_center_set = str(getattr(args, 'forget_center_set', 'all'))
    forget_csv = getattr(args, 'forget_classes', None)
    forget_list_path = getattr(args, 'forget_list_path', None)
    forget_classes = []
    if forget_csv:
        forget_classes = [int(x) for x in str(forget_csv).split(',') if x!='']
    elif forget_list_path and os.path.exists(forget_list_path):
        try:
            with open(forget_list_path) as f:
                data = f.read().strip()
                try:
                    forget_classes = list(map(int, json.loads(data)))
                except Exception:
                    forget_classes = [int(line) for line in data.splitlines() if line.strip()!='']
        except Exception:
            forget_classes = []
    retain_mask = np.ones((C,), dtype=bool)
    if use_center_set == 'retain' and len(forget_classes) > 0:
        forget_classes = [x for x in forget_classes if 0 <= x < C]
        if len(forget_classes) > 0:
            retain_mask[np.array(forget_classes, dtype=int)] = False
    center_mat = centers_ssd[retain_mask]

    def maha_score_min_class(X):
        # X 为标准化后的特征（*_ssd），计算到每个类中心的马氏距离并取最小值
        # 公式: (x-μ)^T Σ^{-1} (x-μ)
        if center_mat.shape[0] == 0:
            # 退化保护：无中心时退回到全局中心（均值为0）
            z = X
            return -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
        delta = X[:, None, :] - center_mat[None, :, :]  # [N, K, D]
        md2 = np.einsum('nkd,dd,nkd->nk', delta, inv_sigma, delta)
        dmin = md2.min(axis=1)
        return -dmin

    # ==== 构造忘记类掩码，改为以 Retain 作为 ID 分布，Forget 作为额外 OOD ==== 
    forget_csv = getattr(args, 'forget_classes', None)
    forget_list_path = getattr(args, 'forget_list_path', None)
    forget_classes = []
    if forget_csv:
        forget_classes = [int(x) for x in str(forget_csv).split(',') if x!='']
    elif forget_list_path and os.path.exists(forget_list_path):
        try:
            with open(forget_list_path) as f:
                data = f.read().strip()
                try:
                    forget_classes = list(map(int, json.loads(data)))
                except Exception:
                    forget_classes = [int(line) for line in data.splitlines() if line.strip()!='']
        except Exception:
            forget_classes = []

    if len(forget_classes) > 0:
        fmask = np.isin(label_log_val, np.array(forget_classes, dtype=int))
        ftest_retain_ssd = ftest_ssd[~fmask]
        dtest = maha_score_min_class(ftest_retain_ssd)
    else:
        dtest = maha_score_min_class(ftest_ssd)

    # 评估 OOD：包含原 args.out_datasets 以及（若有）forget 作为额外 OOD
    eval_names = list(args.out_datasets)
    all_results = []
    for name, food in food_ssd_all.items():
        print(f"Evaluating {name}")
        dood = maha_score_min_class(food)
        results = metrics.cal_metric(dtest, dood)
        all_results.append(results)
        plot_kde(dtest, dood, id_label=f"{args.in_dataset}-retain" if len(forget_classes)>0 else args.in_dataset, ood_label=name,
            title='KDE of OOD Score', save_path='evaluation_results/'+name+'-'+args.in_dataset+'-'+args.backbone+'-'+args.method+'-'+'kde_scores.png')

    # 将 forget 作为一个 OOD 源加入评估
    if len(forget_classes) > 0:
        name = 'forget'
        print(f"Evaluating {name}")
        dood = maha_score_min_class(ftest_ssd[fmask])
        results = metrics.cal_metric(dtest, dood)
        all_results.append(results)
        eval_names.append(name)
        plot_kde(dtest, dood, id_label=f"{args.in_dataset}-retain", ood_label=name,
            title='KDE of OOD Score', save_path='evaluation_results/'+name+'-'+args.in_dataset+'-'+args.backbone+'-'+args.method+'-'+'kde_scores.png')

    metrics.print_all_results(all_results, eval_names, 'SSD+')
    # === Forget/Retain split Acc 与 Forget-as-OOD 指标（同时输出） ===
    # 若提供 forget_classes，则在 val 上拆分并输出 Forget-Acc/Retain-Acc，且把 Forget 当作 OOD、Retain 当作 ID 评估 OOD 指标
    if len(forget_classes) > 0:
        # 最近类中心分类（基于训练特征按类均值中心）
        C = int(label_log.max()) + 1
        D = ftrain.shape[1]
        centers = np.zeros((C, D), dtype=np.float32)
        for c in range(C):
            idx = (label_log == c)
            if np.any(idx):
                centers[c] = ftrain[idx].mean(axis=0)
        # 余弦距离最近中心
        ftest_n = ftest / (np.linalg.norm(ftest, axis=1, keepdims=True) + 1e-10)
        centers_n = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
        sims = ftest_n @ centers_n.T
        preds = np.argmax(sims, axis=1)
        forget_acc, retain_acc = metrics.split_acc_by_classes(label_log_val, preds, forget_classes)
        print(f"Forget-Acc: {forget_acc:.4f} | Retain-Acc: {retain_acc:.4f}")
        # Forget-as-OOD: retain 作为 known，forget 作为 novel（与上面 forget OOD 结果一致）
        # 这里复用 retain 的 ID 分布分数 dtest，以及 forget 的 dood
        forget_scores = maha_score_min_class(ftest_ssd[fmask])
        retain_scores = dtest
        fr_results = metrics.cal_metric(retain_scores, forget_scores)
        print('Forget-as-OOD (retain known vs forget novel):')
        print(f"  FPR: {100.*fr_results['FPR']:.2f} AUROC: {100.*fr_results['AUROC']:.2f} AUIN: {100.*fr_results['AUIN']:.2f}")
    args.score = "mahalanobis"
    write_csv(args, all_results)
    print(time.time() - begin)
    
    