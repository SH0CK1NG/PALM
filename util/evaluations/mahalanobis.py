import time
import os
import json
from util.evaluations import metrics
import torch
import numpy as np
from util.evaluations.write_to_csv import write_csv
from util.evaluations.metrics import plot_kde
from typing import List

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

    # 当存在遗忘类时，一律仅使用保留类中心
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
    if len(forget_classes) > 0:
        # 始终丢弃遗忘类中心，仅保留 retain 类中心
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

    # 评估 OOD：仅包含 args.out_datasets（不再加入 forget）
    eval_names = list(args.out_datasets)
    all_results = []
    for name, food in food_ssd_all.items():
        print(f"Evaluating {name}")
        dood = maha_score_min_class(food)
        results = metrics.cal_metric(dtest, dood)
        all_results.append(results)
        plot_kde(dtest, dood, id_label=f"{args.in_dataset}-retain" if len(forget_classes)>0 else args.in_dataset, ood_label=name,
            title='KDE of OOD Score', save_path='evaluation_results/'+name+'-'+args.in_dataset+'-'+args.backbone+'-'+args.method+'-'+'kde_scores.png')

    # 不再将 forget 作为额外 OOD 源加入评估/CSV（保留下方单独打印 Forget-as-OOD 指标）

    metrics.print_all_results(all_results, eval_names, 'SSD+')
    # === Retain-Acc 与 Forget-as-OOD 指标（不再评估 Forget-Acc） ===
    # 若提供 forget_classes，则在 val 上仅计算并输出 Retain-Acc，且把 Forget 当作 OOD、Retain 当作 ID 评估 OOD 指标
    if len(forget_classes) > 0:
        # 最近类中心分类（基于训练特征按类均值中心）
        C = int(label_log.max()) + 1
        D = ftrain.shape[1]
        centers = np.zeros((C, D), dtype=np.float32)
        for c in range(C):
            idx = (label_log == c)
            if np.any(idx):
                centers[c] = ftrain[idx].mean(axis=0)
        # 余弦距离最近中心（仅统计保留类精度）
        ftest_n = ftest / (np.linalg.norm(ftest, axis=1, keepdims=True) + 1e-10)
        centers_n = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
        sims = ftest_n @ centers_n.T
        preds = np.argmax(sims, axis=1)
        retain_idx = ~fmask
        if retain_idx.sum() > 0:
            retain_acc = np.mean(preds[retain_idx] == label_log_val[retain_idx])
            print(f"Retain-Acc: {retain_acc:.4f}")
            try:
                setattr(args, 'retain_acc', float(retain_acc))
            except Exception:
                pass
        # Forget-as-OOD: retain 作为 known，forget 作为 novel（与上面 forget OOD 结果一致）
        # 这里复用 retain 的 ID 分布分数 dtest，以及 forget 的 dood
        forget_scores = maha_score_min_class(ftest_ssd[fmask])
        retain_scores = dtest
        fr_results = metrics.cal_metric(retain_scores, forget_scores)
        print('Forget-as-OOD (retain known vs forget novel):')
        print(f"  FPR: {100.*fr_results['FPR']:.2f} AUROC: {100.*fr_results['AUROC']:.2f} AUIN: {100.*fr_results['AUIN']:.2f}")
        # expose forget metrics to CSV writer (we do not append to results anymore)
        try:
            setattr(args, 'forget_fpr', float(fr_results['FPR']))
            setattr(args, 'forget_auroc', float(fr_results['AUROC']))
        except Exception:
            pass
    args.score = "mahalanobis"
    write_csv(args, all_results)
    print(time.time() - begin)
    
    # === Optional: UMAP visualization ===
    if getattr(args, 'umap_enable', False):
        try:
            import umap.umap_ as umap
            import matplotlib.pyplot as plt
            # gather features and labels
            base = os.path.join("cache", f"{args.backbone}-{args.method}", args.in_dataset)
            X_list: List[np.ndarray] = []
            y_list: List[str] = []
            # ID retain (val) as domain retain
            fx = ftest
            X_list.append(fx)
            y_list += ["retain"] * fx.shape[0]
            # OOD including forget
            for name in list(args.out_datasets):
                food = food_all.get(name, None)
                if food is None:
                    continue
                X_list.append(food)
                y_list += [name] * food.shape[0]
            if len(forget_classes) > 0:
                # fmask defined earlier for val set
                X_list.append(ftest[fmask])
                y_list += ["forget"] * int(fmask.sum())
            X = np.concatenate(X_list, axis=0)
            # subsample
            max_points = int(getattr(args, 'umap_max_points', 20000))
            if X.shape[0] > max_points:
                rng = np.random.RandomState(0)
                idx = rng.choice(X.shape[0], max_points, replace=False)
                X = X[idx]
                y_list = [y_list[i] for i in idx]
            reducer = umap.UMAP(n_neighbors=int(getattr(args,'umap_neighbors',15)),
                                min_dist=float(getattr(args,'umap_min_dist',0.05)),
                                metric=str(getattr(args,'umap_metric','cosine')),
                                random_state=0)
            emb = reducer.fit_transform(X)
            # draw
            plt.figure(figsize=(8,6), dpi=120)
            uniq = sorted(set(y_list))
            colors = plt.cm.tab20(np.linspace(0,1,len(uniq)))
            color_map = {k:c for k,c in zip(uniq, colors)}
            for k in uniq:
                m = [i for i,t in enumerate(y_list) if t==k]
                if len(m)==0: continue
                plt.scatter(emb[m,0], emb[m,1], s=4, c=[color_map[k]], label=k, alpha=0.7, linewidths=0)
            plt.legend(markerscale=2, frameon=False, ncol=3)
            plt.title(f"UMAP - {args.in_dataset} ({args.backbone}-{args.method})")
            os.makedirs("figs", exist_ok=True)
            save_path = getattr(args,'umap_save_path', None)
            if not save_path:
                save_path = os.path.join("figs", f"umap_{args.in_dataset}_{args.backbone}_{args.method}_domain.png")
            plt.tight_layout()
            plt.savefig(save_path)
            print("[umap] saved to", save_path)

            # Optional extra figure: retain vs forget only
            if bool(getattr(args, 'umap_rf_only', False)) and (len(forget_classes) > 0):
                try:
                    X2_list: List[np.ndarray] = []
                    y2_list: List[str] = []
                    # retain vs forget from val set split
                    if (~fmask).sum() > 0:
                        X2_list.append(ftest[~fmask])
                        y2_list += ["retain"] * int((~fmask).sum())
                    if fmask.sum() > 0:
                        X2_list.append(ftest[fmask])
                        y2_list += ["forget"] * int(fmask.sum())
                    if len(X2_list) >= 1:
                        X2 = np.concatenate(X2_list, axis=0)
                        max_points2 = int(getattr(args, 'umap_max_points', 20000))
                        if X2.shape[0] > max_points2:
                            rng2 = np.random.RandomState(1)
                            idx2 = rng2.choice(X2.shape[0], max_points2, replace=False)
                            X2 = X2[idx2]
                            y2_list = [y2_list[i] for i in idx2]
                        reducer2 = umap.UMAP(n_neighbors=int(getattr(args,'umap_neighbors',15)),
                                             min_dist=float(getattr(args,'umap_min_dist',0.05)),
                                             metric=str(getattr(args,'umap_metric','cosine')),
                                             random_state=1)
                        emb2 = reducer2.fit_transform(X2)
                        plt.figure(figsize=(8,6), dpi=120)
                        uniq2 = sorted(set(y2_list))
                        colors2 = plt.cm.Set1(np.linspace(0,1,len(uniq2)))
                        cmap2 = {k:c for k,c in zip(uniq2, colors2)}
                        for k in uniq2:
                            m2 = [i for i,t in enumerate(y2_list) if t==k]
                            if len(m2)==0: continue
                            plt.scatter(emb2[m2,0], emb2[m2,1], s=6, c=[cmap2[k]], label=k, alpha=0.8, linewidths=0)
                        plt.legend(markerscale=2, frameon=False)
                        plt.title(f"UMAP RF - {args.in_dataset} ({args.backbone}-{args.method})")
                        os.makedirs("figs", exist_ok=True)
                        save_path2 = os.path.join("figs", f"umap_{args.in_dataset}_{args.backbone}_{args.method}_rf.png")
                        plt.tight_layout()
                        plt.savefig(save_path2)
                        print("[umap] saved to", save_path2)
                except Exception as e:
                    print("[umap] rf-only visualization failed:", e)
        except Exception as e:
            print("[umap] visualization failed:", e)
    
    