import os
import math
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from util.loaders.args_loader import get_args
from util.train_utils import AverageMeter, adjust_learning_rate
from util.cider_adapter import CIDERCriterion
from models.resnet import SSLResNet
from util.lora import (
    apply_lora_to_resnet_head,
    apply_lora_to_resnet_layer4,
    apply_lora_to_resnet_layers,
    extract_lora_state_dict,
    load_lora_state_dict,
)
from util.peft_utils import (
    is_peft_available,
    apply_peft_lora_to_model,
    save_peft_adapter,
    load_peft_adapter,
)


def get_normalizer(dataset: str):
    if dataset == 'CIFAR-10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'CIFAR-100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return transforms.Normalize(mean=mean, std=std)


class TwoCropTransform:
    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def simclr_transform_train(dataset: str) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        get_normalizer(dataset)
    ])


def get_transform_eval(dataset: str) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        get_normalizer(dataset),
    ])


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, targets, forget_set, batch_size: int, num_batches: int, seed: int = 0):
        import numpy as np
        self.targets = np.array(list(targets), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.half_f = self.batch_size // 2
        self.half_r = self.batch_size - self.half_f
        self.num_batches = int(num_batches)
        self.rng = np.random.RandomState(seed)
        forget_mask = np.isin(self.targets, np.array(sorted(list(forget_set)), dtype=np.int64))
        self.forget_idx = np.where(forget_mask)[0]
        self.retain_idx = np.where(~forget_mask)[0]
        if len(self.forget_idx) == 0 or len(self.retain_idx) == 0:
            self.forget_idx = np.arange(len(self.targets))
            self.retain_idx = np.arange(len(self.targets))
    def __iter__(self):
        import numpy as np
        for _ in range(self.num_batches):
            f_take = self.rng.choice(self.forget_idx, size=min(self.half_f, len(self.forget_idx)), replace=(len(self.forget_idx) < self.half_f))
            r_take = self.rng.choice(self.retain_idx, size=min(self.half_r, len(self.retain_idx)), replace=(len(self.retain_idx) < self.half_r))
            need = self.batch_size - (len(f_take) + len(r_take))
            if need > 0:
                extra = self.rng.choice(self.retain_idx, size=need, replace=True)
                batch = np.concatenate([f_take, r_take, extra])
            else:
                batch = np.concatenate([f_take, r_take])
            self.rng.shuffle(batch)
            yield batch.tolist()
    def __len__(self):
        return self.num_batches


class ProportionalBatchSampler(torch.utils.data.Sampler):
    def __init__(self, targets, forget_set, batch_size: int, num_batches: int, seed: int = 0):
        import numpy as np
        self.targets = np.array(list(targets), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.rng = np.random.RandomState(seed)
        forget_mask = np.isin(self.targets, np.array(sorted(list(forget_set)), dtype=np.int64))
        self.forget_idx = np.where(forget_mask)[0]
        self.retain_idx = np.where(~forget_mask)[0]
        self.f_count = int(len(self.forget_idx))
        self.r_count = int(len(self.retain_idx))
        self.total = max(1, self.f_count + self.r_count)
        self.p_forget = self.f_count / float(self.total)
        if self.f_count == 0 and self.r_count == 0:
            self.all_idx = np.arange(len(self.targets))
        else:
            self.all_idx = None
    def __iter__(self):
        import numpy as np
        for _ in range(self.num_batches):
            if self.all_idx is not None:
                batch = self.rng.choice(self.all_idx, size=self.batch_size, replace=(len(self.all_idx) < self.batch_size))
                yield batch.tolist()
                continue
            f_take_n = int(round(self.p_forget * self.batch_size))
            r_take_n = self.batch_size - f_take_n
            f_take = self.rng.choice(self.forget_idx, size=min(f_take_n, len(self.forget_idx)), replace=(len(self.forget_idx) < f_take_n)) if self.f_count > 0 else np.empty((0,), dtype=np.int64)
            r_take = self.rng.choice(self.retain_idx, size=min(r_take_n, len(self.retain_idx)), replace=(len(self.retain_idx) < r_take_n)) if self.r_count > 0 else np.empty((0,), dtype=np.int64)
            need = self.batch_size - (len(f_take) + len(r_take))
            if need > 0:
                pool = self.retain_idx if self.r_count > 0 else (self.forget_idx if self.f_count > 0 else None)
                if pool is None or len(pool) == 0:
                    extra = np.array([], dtype=np.int64)
                else:
                    extra = self.rng.choice(pool, size=need, replace=(len(pool) < need))
                batch = np.concatenate([f_take, r_take, extra])
            else:
                batch = np.concatenate([f_take, r_take])
            self.rng.shuffle(batch)
            yield batch.tolist()
    def __len__(self):
        return self.num_batches


def build_loaders(args):
    base_dir = './data'
    kwargs = {'num_workers': max(2, min((os.cpu_count() or 1) - 2, 16)), 'pin_memory': True, 'persistent_workers': True}
    # train: two-crop simclr style; aux/eval: single-crop
    transform_train = TwoCropTransform(simclr_transform_train(args.in_dataset))
    transform_eval = get_transform_eval(args.in_dataset)

    if args.in_dataset == 'CIFAR-10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(base_dir, 'cifarpy'), train=True, download=True, transform=transform_train)
        auxset = torchvision.datasets.CIFAR10(root=os.path.join(base_dir, 'cifarpy'), train=True, download=True, transform=transform_eval)
        num_classes = 10
    elif args.in_dataset == 'CIFAR-100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(base_dir, 'cifarpy'), train=True, download=True, transform=transform_train)
        auxset = torchvision.datasets.CIFAR100(root=os.path.join(base_dir, 'cifarpy'), train=True, download=True, transform=transform_eval)
        num_classes = 100
    else:
        raise ValueError(f"unsupported dataset: {args.in_dataset}")

    # optional per-batch mixing by forget/retain composition
    forget_set = set()
    if getattr(args, 'forget_classes', None):
        forget_set |= set(int(x) for x in str(args.forget_classes).split(',') if x != '')
    mode = str(getattr(args, 'batch_forget_mode', 'none'))
    if mode in ('balanced', 'proportional', 'retain_only', 'forget_only') and len(forget_set) > 0:
        import numpy as np
        targets = getattr(trainset, 'targets', None)
        if targets is None:
            targets = getattr(trainset, 'train_labels', None)
        if targets is None:
            raise RuntimeError('cannot find targets for sampler')
        num_batches = len(trainset) // args.batch_size
        if mode == 'balanced':
            sampler = BalancedBatchSampler(targets, forget_set, args.batch_size, num_batches, seed=args.seed)
            train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=sampler, **kwargs)
        elif mode == 'proportional':
            sampler = ProportionalBatchSampler(targets, forget_set, args.batch_size, num_batches, seed=args.seed)
            train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=sampler, **kwargs)
        else:  # retain_only / forget_only
            targets = np.array(list(targets), dtype=np.int64)
            mask_forget = np.isin(targets, np.array(sorted(list(forget_set)), dtype=np.int64))
            if mode == 'retain_only':
                sel_idx = np.where(~mask_forget)[0].tolist()
            else:
                sel_idx = np.where(mask_forget)[0].tolist()
            subset = torch.utils.data.Subset(trainset, sel_idx)
            train_loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

    aux_loader = torch.utils.data.DataLoader(auxset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, aux_loader, num_classes


def parse_forget_sets(args, labels: torch.Tensor):
    # parse CSV lists on CPU side first
    forget_all = set()
    if getattr(args, 'forget_classes', None):
        forget_all |= set(int(x) for x in str(args.forget_classes).split(',') if x != '')
    # masks on GPU
    if len(forget_all) > 0:
        f_all_gpu = torch.tensor(sorted(list(forget_all)), device=labels.device)
        forget_mask = (labels.unsqueeze(1) == f_all_gpu.unsqueeze(0)).any(dim=1)
        retain_mask = ~forget_mask
    else:
        forget_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        retain_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
    return forget_all, forget_mask, retain_mask


def train_one_epoch(args, epoch: int, model: nn.Module, criterion: CIDERCriterion, optimizer: torch.optim.Optimizer,
                    train_loader, print_every: int = 50):
    model.train()
    losses = AverageMeter()
    sub = {"comp": 0.0, "dis": 0.0, "forget": 0.0}

    for step, (images, labels) in enumerate(train_loader, start=epoch * len(train_loader)):
        # two-crop inputs
        twocrop = isinstance(images, (list, tuple)) and len(images) == 2
        if twocrop:
            images = torch.cat([images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)], dim=0)
        else:
            images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        if twocrop:
            labels = labels.repeat(2)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=False):
            # forward: encoder + head => normalized features
            features = model(images)
            # CIDER loss
            loss_cider, d = criterion(features, labels)
            loss = loss_cider

            # forgetting term (push forget samples away from retained prototypes)
            forget_all, forget_mask, retain_mask = parse_forget_sets(args, labels)
            if float(getattr(args, 'forget_lambda', 0.0)) > 0.0 and forget_mask.any() and len(forget_all) > 0:
                C = int(criterion.num_classes)
                # retained classes exclude forget_all
                retain_classes = torch.tensor([c for c in range(C) if c not in forget_all], device=labels.device, dtype=torch.long)
                if retain_classes.numel() > 0:
                    P = F.normalize(criterion.protos.detach(), dim=1).index_select(0, retain_classes)
                    feats_mean = F.normalize(features, dim=-1)
                    feats_f = feats_mean[forget_mask]
                    if feats_f.numel() > 0:
                        sim = torch.matmul(feats_f, P.t())
                        temp_f = float(getattr(args, 'temp', 0.1))
                        forget_term = float(getattr(args, 'forget_lambda', 1.0)) * torch.logsumexp(sim / temp_f, dim=1).mean()
                        loss = loss + forget_term
                        d["forget"] = float(forget_term.detach().item())
                    else:
                        d["forget"] = 0.0
                else:
                    d["forget"] = 0.0
            else:
                d["forget"] = 0.0

        loss.backward()
        optimizer.step()

        # adjust lr (cosine with warmup)
        adjust_learning_rate(args, optimizer, train_loader, step)

        losses.update(loss.item(), bsz)
        for k in sub.keys():
            sub[k] = 0.9 * sub[k] + 0.1 * float(d.get(k, 0.0))

        if print_every and (step % print_every == 0):
            print(f"[cider] ep {epoch} it {step % len(train_loader)} total={loss.item():.4f} comp={d.get('comp', 0.0):.4f} dis={d.get('dis', 0.0):.4f} forget={d.get('forget', 0.0):.4f}")

    return losses.avg


def main():
    args = get_args()
    # augment missing args for CIDER
    if not hasattr(args, 'feat_dim'):
        args.feat_dim = 128
    if not hasattr(args, 'head'):
        args.head = 'mlp'
    if not hasattr(args, 'w'):
        args.w = 1.0
    if not hasattr(args, 'print_every'):
        args.print_every = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # loaders
    train_loader, aux_loader, num_classes = build_loaders(args)

    # model
    model = SSLResNet(name=args.backbone, head=args.head, feat_dim=args.feat_dim)
    model = model.to(device)

    # load base pretrained CIDER checkpoint (encoder+head)
    base_ckpt = getattr(args, 'load_path', None)
    if base_ckpt and os.path.exists(base_ckpt):
        try:
            ckpt = torch.load(base_ckpt, map_location=device)
            state_dict = None
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            # strip possible DataParallel prefix
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[ckpt] missing keys: {list(missing)[:8]}{' ...' if len(missing)>8 else ''}")
            if unexpected:
                print(f"[ckpt] unexpected keys: {list(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")
            print(f"[cider] base weights loaded from {base_ckpt}")
        except Exception as e:
            print(f"[cider] failed to load base ckpt: {e}")

    # apply LoRA (PEFT preferred if requested)
    lora_applied = False
    if bool(getattr(args, 'use_lora', False)):
        target = getattr(args, 'lora_target', 'head')
        r = int(getattr(args, 'lora_r', 8))
        alpha = int(getattr(args, 'lora_alpha', 32))
        dropout = float(getattr(args, 'lora_dropout', 0.05))
        prefer_peft = (str(getattr(args, 'lora_impl', 'native')) == 'peft') and is_peft_available()
        if prefer_peft:
            try:
                model, targets = apply_peft_lora_to_model(model, target=target, r=r, alpha=alpha, dropout=dropout)
                lora_applied = len(targets) > 0
                # warm start from single adapter if provided
                if getattr(args, 'adapter_load_path', None):
                    try:
                        model = load_peft_adapter(model, args.adapter_load_path)
                        print(f"[peft] adapter loaded from {args.adapter_load_path}")
                    except Exception as e:
                        print(f"[peft] failed to load adapter: {e}")
            except Exception as e:
                print(f"[peft] failed to apply lora: {e}")
                lora_applied = False
        if not lora_applied:
            # fallback to native lightweight LoRA
            applied_head = False
            applied_enc = False
            if target in ['head', 'both', 'both_all']:
                applied_head = apply_lora_to_resnet_head(model, r=r, alpha=alpha, dropout=dropout)
            if target in ['encoder', 'both']:
                applied_enc = apply_lora_to_resnet_layer4(model, r=r, alpha=alpha, dropout=dropout)
            if target in ['encoder_all', 'both_all']:
                applied_enc = apply_lora_to_resnet_layers(model, layers=[1,2,3,4], r=r, alpha=alpha, dropout=dropout) or applied_enc
            lora_applied = applied_head or applied_enc
            if lora_applied:
                # freeze all non-LoRA params
                for n, p in model.named_parameters():
                    p.requires_grad = False
                for m in model.modules():
                    if hasattr(m, 'A') and hasattr(m, 'B'):
                        if m.A is not None:
                            m.A.requires_grad_(True)
                        if m.B is not None:
                            m.B.requires_grad_(True)
                # optional warm start for native
                if getattr(args, 'adapter_load_path', None):
                    try:
                        ad = torch.load(args.adapter_load_path, map_location=device)
                        lora_sd = ad['lora'] if isinstance(ad, dict) and 'lora' in ad else ad
                        loaded = load_lora_state_dict(model, lora_sd)
                        print(f"[lora] loaded {loaded} LoRA tensors from {args.adapter_load_path}")
                    except Exception as e:
                        print(f"[lora] failed to load adapter: {e}")
        if lora_applied:
            model.train()

    # criterion: CIDER
    criterion = CIDERCriterion(num_classes=num_classes, feat_dim=args.feat_dim, temperature=args.temp, proto_m=args.proto_m, w=args.w).to(device)
    # try initialize prototypes from aux set; also try copy from base ckpt if available
    if base_ckpt and os.path.exists(base_ckpt):
        try:
            ckpt = torch.load(base_ckpt, map_location=device)
            if isinstance(ckpt, dict) and 'dis_state_dict' in ckpt and 'prototypes' in ckpt['dis_state_dict']:
                with torch.no_grad():
                    P = ckpt['dis_state_dict']['prototypes']
                    criterion.dis.prototypes = F.normalize(P.to(device), dim=1)
                print("[cider] prototypes loaded from base ckpt")
            else:
                criterion.init_prototypes(model, aux_loader)
        except Exception:
            criterion.init_prototypes(model, aux_loader)
    else:
        criterion.init_prototypes(model, aux_loader)

    # optimizer
    optimizer = torch.optim.SGD((p for p in model.parameters() if getattr(p, 'requires_grad', False)),
                                lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

    # naming / save
    t = datetime.now().strftime("%Y%m%d-%H%M%S")
    method_tag = f"cider-b{args.batch_size}-e{args.epochs}-lr{args.lr}-wd{args.weight_decay}-fd{args.feat_dim}-w{args.w}-pm{args.proto_m}-temp{args.temp}-bfm{getattr(args, 'batch_forget_mode', 'none')}-fl{getattr(args, 'forget_lambda', 0)}"
    save_dir = os.path.join('checkpoints', args.in_dataset)
    os.makedirs(save_dir, exist_ok=True)
    args.save_path = os.path.join(save_dir, f"{args.in_dataset}-{args.backbone}-{method_tag}.pt")
    print(f"[cider] save to {args.save_path}")

    best = math.inf
    for epoch in range(args.epochs):
        loss = train_one_epoch(args, epoch, model, criterion, optimizer, train_loader, print_every=args.print_every)
        if loss < best:
            best = loss
            # save adapter-only if path provided and LoRA is applied
            adapter_path = getattr(args, 'adapter_save_path', None)
            if adapter_path and lora_applied:
                os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
                if is_peft_available() and hasattr(model, 'save_pretrained'):
                    try:
                        save_peft_adapter(model, adapter_path)
                        print(f"[peft] adapter saved to {adapter_path}")
                    except Exception as e:
                        print(f"[peft] save failed, fallback to native: {e}")
                        lora_sd = extract_lora_state_dict(model)
                        torch.save({'lora': lora_sd,
                                    'meta': {'in_dataset': args.in_dataset,
                                             'backbone': args.backbone,
                                             'method': method_tag,
                                             'num_classes': num_classes}},
                                  adapter_path)
                else:
                    lora_sd = extract_lora_state_dict(model)
                    torch.save({'lora': lora_sd,
                                'meta': {'in_dataset': args.in_dataset,
                                         'backbone': args.backbone,
                                         'method': method_tag,
                                         'num_classes': num_classes}},
                              adapter_path)
            else:
                ckpt = {
                    'model': model.state_dict(),
                    'criterion': criterion.state_dict(),
                    'meta': {
                        'in_dataset': args.in_dataset,
                        'backbone': args.backbone,
                        'method': method_tag,
                        'num_classes': num_classes,
                        'feat_dim': args.feat_dim,
                    }
                }
                torch.save(ckpt, args.save_path)
                print(f"[cider] ckpt saved: {args.save_path}")


if __name__ == '__main__':
    main()


