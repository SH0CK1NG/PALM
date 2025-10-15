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
        # optional: split val set into retain(ID) and forget(OOD)
        forget_csv = getattr(args, 'forget_classes', None)
        forget_list_path = getattr(args, 'forget_list_path', None)
        forget_classes = []
        if split == 'val':
            if forget_csv:
                forget_classes = [int(x) for x in str(forget_csv).split(',') if x!='']
            elif forget_list_path and os.path.exists(forget_list_path):
                try:
                    with open(forget_list_path) as f:
                        data = f.read().strip()
                        try:
                            import json
                            forget_classes = list(map(int, json.loads(data)))
                        except Exception:
                            forget_classes = [int(line) for line in data.splitlines() if line.strip()!='']
                except Exception:
                    forget_classes = []
        retain_features, retain_labels = [], []
        forget_features, forget_labels = [], []
        total = 0

        model.eval()

        for index, (img, label) in enumerate(tqdm(in_loader)):

            img = img.cuda()

            batch_feats = model.encoder(img).data.cpu().numpy()
            batch_labels = label.data.cpu().numpy()
            if split == 'val' and len(forget_classes) > 0:
                # route to retain/forget
                mask_forget = np.isin(batch_labels, np.array(forget_classes, dtype=int))
                if np.any(~mask_forget):
                    retain_features.extend(list(batch_feats[~mask_forget]))
                    retain_labels.extend(list(batch_labels[~mask_forget]))
                if np.any(mask_forget):
                    forget_features.extend(list(batch_feats[mask_forget]))
                    forget_labels.extend(list(batch_labels[mask_forget]))
            else:
                features += list(batch_feats)
                labels += list(batch_labels)

            total += len(img)

        if split == 'val' and len(forget_classes) > 0:
            # save retain(ID) as val cache
            feat_log = np.array(retain_features)
            label_log = np.array(retain_labels)
        else:
            feat_log, label_log = np.array(features), np.array(labels)
        ########################################
        np.save(cache_name, feat_log)
        np.save(label_cache_name, label_log)
        print(
            f"{total} images processed, {time.time()-t0} seconds used\n")

        # additionally save forget subset as an OOD dataset "forget"
        if split == 'val' and len(forget_classes) > 0 and len(forget_features) > 0:
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
