#! /usr/bin/env python3
import torch
import os
from util.loaders.args_loader import get_args
from util.loaders.data_loader import get_loader_in, get_loader_out
from util.loaders.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


trainloaderIn, num_classes = get_loader_in(args, split='train', mode='eval')
testloaderIn, _ = get_loader_in(args, split='val', mode='eval')
model = get_model(args, num_classes, load_ckpt=True)
model.to(device)
model.eval()

batch_size = args.batch_size

FORCE_RUN = True

dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
features = model.encoder(dummy_input)
featdims = features.shape[1]

begin = time.time()

for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn), ]:
    in_save_dir = os.path.join("cache", f"{args.backbone}-{args.method}", args.in_dataset)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)
    cache_name = os.path.join(
        in_save_dir, f"{split}_{args.backbone}-{args.method}_features.npy")
    label_cache_name = os.path.join(
        in_save_dir, f"{split}_{args.backbone}-{args.method}_labels.npy")
    if FORCE_RUN or not os.path.exists(cache_name):
        print(f"Processing in-distribution {args.in_dataset} images")
        t0 = time.time()
        ########################################
        features, labels = [], []
        # 增量/遗忘过滤控制
        # 在增量模式下：仅保留“已学类=seen∪inc”作为 ID；未学类不进入 ID 缓存
        # 在普通模式下：沿用 retain_exclude/forget_csv 逻辑
        retain_exclude_csv = getattr(args, 'retain_exclude_csv', None)
        forget_csv = getattr(args, 'forget_csv', None)
        def parse_csv(s):
            return [int(x) for x in str(s).split(',') if x!='']

        # 增量开关：显式 --incremental，或提供 retain_exclude_csv / forget_classes_inc / forget_classes_seen 也视为增量评估
        inc_enable = bool(getattr(args, 'incremental', False)) or bool(retain_exclude_csv) or bool(getattr(args, 'forget_classes_inc', None)) or bool(getattr(args, 'forget_classes_seen', None))
        if inc_enable:
            all_csv = getattr(args, 'forget_classes', None)
            inc_csv = getattr(args, 'forget_classes_inc', None)
            seen_csv = getattr(args, 'forget_classes_seen', None)
            all_set = set(parse_csv(all_csv)) if all_csv else set(range(num_classes))
            new_set = set(parse_csv(inc_csv)) if inc_csv else set()
            seen_set = set(parse_csv(seen_csv)) if seen_csv else set()
            learned_set = (new_set | seen_set)
            # 若未提供 inc/seen，但给了 retain_exclude_csv，则以其补集作为已学类集合
            if len(learned_set) == 0 and retain_exclude_csv:
                ex_set = set(parse_csv(retain_exclude_csv))
                learned_set = set([c for c in range(num_classes) if c not in ex_set])
            # 约束在 all_set 内（若提供 all_set）
            learned_set = set([c for c in learned_set if c in all_set and 0 <= c < num_classes])
            # eval 时不再构建 forget 子缓存
            retain_exclude = []
            forget_classes = []
        else:
            # 若未显式给 retain_exclude_csv，但给了 forget_csv，则在评估阶段退化为 retain_exclude=forget_csv
            retain_exclude = parse_csv(retain_exclude_csv) if retain_exclude_csv else (parse_csv(forget_csv) if forget_csv else [])
            forget_classes = parse_csv(forget_csv) if (split == 'val' and forget_csv) else []
        retain_features, retain_labels = [], []
        forget_features, forget_labels = [], []
        total = 0

        model.eval()

        for index, (img, label) in enumerate(tqdm(in_loader)):

            img = img.cuda()

            batch_feats = model.encoder(img).data.cpu().numpy()
            batch_labels = label.data.cpu().numpy()
            # 总是累加全量，后续统一按模式过滤
            features += list(batch_feats)
            labels += list(batch_labels)
            # 同时在 val 场景下，单独构建 forget 子集缓存（仅用于 OOD 评估）
            if split == 'val' and len(forget_classes) > 0:
                mask_forget = np.isin(batch_labels, np.array(forget_classes, dtype=int))
                if np.any(~mask_forget):
                    retain_features.extend(list(batch_feats[~mask_forget]))
                    retain_labels.extend(list(batch_labels[~mask_forget]))
                if np.any(mask_forget):
                    forget_features.extend(list(batch_feats[mask_forget]))
                    forget_labels.extend(list(batch_labels[mask_forget]))

            total += len(img)

        # 构建主缓存
        if inc_enable:
            all_feats = np.array(features)
            all_labels = np.array(labels)
            if len(learned_set) > 0:
                mask_keep = np.isin(all_labels, np.array(sorted(list(learned_set)), dtype=int))
                feat_log = all_feats[mask_keep]
                label_log = all_labels[mask_keep]
            else:
                feat_log = np.empty((0, all_feats.shape[1] if all_feats.ndim==2 else 0), dtype=np.float32)
                label_log = np.empty((0,), dtype=np.int64)
        else:
            # 普通模式：retain_exclude 优先；否则保留原逻辑
            if len(retain_exclude) > 0:
                all_feats = np.array(features)
                all_labels = np.array(labels)
                mask_ex = np.isin(all_labels, np.array(retain_exclude, dtype=int))
                feat_log = all_feats[~mask_ex]
                label_log = all_labels[~mask_ex]
            else:
                if split == 'val' and len(forget_classes) > 0:
                    feat_log = np.array(retain_features)
                    label_log = np.array(retain_labels)
                else:
                    feat_log, label_log = np.array(features), np.array(labels)
        ########################################
        np.save(cache_name, feat_log)
        np.save(label_cache_name, label_log)
        print(
            f"{total} images processed, {time.time()-t0} seconds used\n")

        # 普通模式下依旧写入 forget 子缓存（增量模式不写）
        if (not inc_enable) and (split == 'val') and len(forget_classes) > 0 and len(forget_features) > 0:
            out_save_dir = os.path.join(in_save_dir, 'forget')
            if not os.path.exists(out_save_dir):
                os.makedirs(out_save_dir)
            f_cache = os.path.join(out_save_dir, f"{args.backbone}-{args.method}_features.npy")
            l_cache = os.path.join(out_save_dir, f"{args.backbone}-{args.method}_labels.npy")
            np.save(f_cache, np.array(forget_features))
            np.save(l_cache, np.array(forget_labels))
            print(f"Saved forget OOD features to {out_save_dir}")

for ood_dataset in args.out_datasets:
    # print(f"OOD Dataset: {ood_dataset}")
    out_loader = get_loader_out(args, dataset=ood_dataset, split=('val'), mode='eval')

    out_save_dir = os.path.join(in_save_dir, ood_dataset)
    if not os.path.exists(out_save_dir):
        os.makedirs(out_save_dir)
    cache_name = os.path.join(out_save_dir, f"{args.backbone}-{args.method}_features.npy")
    label_cache_name = os.path.join(out_save_dir, f"{args.backbone}-{args.method}_labels.npy")

    if FORCE_RUN or not os.path.exists(cache_name):
        t0 = time.time()
        print(f"Processing out-of-distribution {ood_dataset} images")

        ########################################
        features, labels = [], []
        total = 0

        model.eval()

        for index, (img, label) in enumerate(tqdm(out_loader)):

            img, label = img.cuda(), label.cuda()

            features += list(model.encoder(img).data.cpu().numpy())
            labels += list(label.data.cpu().numpy())

            total += len(img)

        feat_log, label_log = np.array(features), np.array(labels)
        ########################################
        np.save(cache_name, feat_log)
        np.save(label_cache_name, label_log)
        print(f"{total} images processed, {time.time()-t0} seconds used\n")
        t0 = time.time()

print(time.time() - begin)
