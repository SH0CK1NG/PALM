import os
import argparse
import torch
import torchvision
from torchvision import transforms
from PIL import ImageOps, ImageFilter, Image
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple, List, Optional, Union, Sequence
from torchvision.transforms import InterpolationMode


imagesize = 32


def get_transform_test(in_set: str) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((imagesize, imagesize)),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        get_normalizer(in_set)
    ])


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=imagesize, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

def get_transform_eval(in_set: str) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((imagesize, imagesize)), 
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        get_normalizer(in_set)
    ])


def get_normalizer(dataset):#？什么作用
    if dataset == 'CIFAR-10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'CIFAR-100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "CIFAR-110":
        # Use util/preprocess/compute_stats.py
        mean = (0.5056, 0.4861, 0.4415)
        std = (0.2657, 0.2554, 0.2750)
    else:
        raise ValueError('dataset not supported: {}'.format(dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize


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


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform

    def __call__(self, x: Image.Image) -> List[torch.Tensor]:
        return [self.transform(x), self.transform(x)]


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Vicreg_transform_train(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    32, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )

    def __call__(self, sample: Image.Image) -> List[torch.Tensor]:
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return [x1, x2]

class CIFAR110Dataset(torch.utils.data.Dataset):
    """
    Mixed dataset of CIFAR-100 and CIFAR-10 with remapped labels.
    - CIFAR-100 classes are mapped to 0-99 (unchanged)
    - CIFAR-10 classes are mapped to 100-109 (by adding 100)
    """
    def __init__(self, root: str, train: bool, transform: Union[transforms.Compose, TwoCropTransform]):
        self.root = root
        self.transform = transform
        self.train = train
        cifar_root = os.path.join(root, 'cifarpy')
        self.cifar100 = torchvision.datasets.CIFAR100(
            root=cifar_root, train=train, download=True, transform=self.transform
        )
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=cifar_root, train=train, download=True, transform=self.transform
        )
        # CIFAR-100 原生就符合每類 500/100 的數量
        self.len_cifar100 = len(self.cifar100)

        # 對 CIFAR-10 進行每類定額抽樣：train=500/類；val/test=100/類
        # 兼容不同 torchvision 版本的標籤屬性名稱
        labels10 = getattr(self.cifar10, 'targets', None)
        if labels10 is None:
            labels10 = getattr(self.cifar10, 'labels', None)
        if labels10 is None:
            labels10 = getattr(self.cifar10, 'train_labels' if train else 'test_labels', None)
        labels10 = np.array(labels10)

        per_class = 500 if train else 100
        rng = np.random.RandomState(1)
        selected_indices = []
        for c in range(10):
            cls_idx = np.where(labels10 == c)[0]
            rng.shuffle(cls_idx)
            selected_indices.extend(cls_idx[:per_class].tolist())
        self.c10_subset_indices = np.array(selected_indices, dtype=np.int64)

        # build combined targets aligned with dataset indexing
        labels100 = getattr(self.cifar100, 'targets', None)
        if labels100 is None:
            labels100 = getattr(self.cifar100, 'labels', None)
        if labels100 is None:
            labels100 = getattr(self.cifar100, 'train_labels' if train else 'test_labels', None)
        labels100 = np.array(labels100, dtype=np.int64)
        selected_targets10 = labels10[self.c10_subset_indices] + 100
        self.targets = np.concatenate([labels100, selected_targets10], axis=0).tolist()

        self.len_cifar10 = len(self.c10_subset_indices)
        self.total_len = self.len_cifar100 + self.len_cifar10

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index < self.len_cifar100:
            image, label = self.cifar100[index]
            return image, int(label)
        else:
            base_idx = index - self.len_cifar100
            c10_idx = int(self.c10_subset_indices[base_idx])
            image, label = self.cifar10[c10_idx]
            return image, int(label) + 100


def get_transform_train(dataset: str, arch: str) -> Union[TwoCropTransform, transforms.Compose]:
    if 'palm' in arch:
        transform = TwoCropTransform(simclr_transform_train(dataset))
    else:
        transform = transform_train
    return transform


kwargs = {'num_workers': max(2, min((os.cpu_count() or 1) - 2, 16)), 'pin_memory': True, 'persistent_workers': True}
num_classes_dict = {'CIFAR-100': 100, 'CIFAR-10': 10, 'CIFAR-110': 110, 'imagenet': 1000}

val_shuffle = False


def get_loader_in(args: argparse.Namespace, split: Tuple[str, ...] = ('train', 'val', 'csid'), mode: str = 'train') -> Tuple[torch.utils.data.DataLoader, int]:
    base_dir = './data'
    
    kwargs = {'num_workers': max(2, min((os.cpu_count() or 1) - 2, 16)), 'pin_memory': True, 'persistent_workers': True}
    num_classes_dict = {'CIFAR-100': 100, 'CIFAR-10': 10, 'CIFAR-110': 110, 'imagenet': 1000}
    if mode == 'train':
        transform_train = get_transform_train(args.in_dataset, args.method)
        transform_test = get_transform_test(args.in_dataset)
    elif mode == "eval":
        transform_train = get_transform_eval(args.in_dataset)
        transform_test = get_transform_eval(args.in_dataset)

    # helper: build balanced batch sampler for CIFAR10/100
    class BalancedBatchSampler(torch.utils.data.Sampler[List[int]]):
        def __init__(self, targets: List[int], forget_set: set, batch_size: int, num_batches: int, seed: int = 0):
            self.targets = np.array(list(targets), dtype=np.int64)
            self.batch_size = int(batch_size)
            self.half_f = self.batch_size // 2
            self.half_r = self.batch_size - self.half_f
            self.num_batches = int(num_batches)
            self.rng = np.random.RandomState(seed)
            forget_mask = np.isin(self.targets, np.array(sorted(list(forget_set)), dtype=np.int64))
            self.forget_idx = np.where(forget_mask)[0]
            self.retain_idx = np.where(~forget_mask)[0]
            # fallback: if one side empty, use all indices to avoid crash
            if len(self.forget_idx) == 0 or len(self.retain_idx) == 0:
                self.forget_idx = np.arange(len(self.targets))
                self.retain_idx = np.arange(len(self.targets))
        def __iter__(self):
            for _ in range(self.num_batches):
                f_take = self.rng.choice(self.forget_idx, size=min(self.half_f, len(self.forget_idx)), replace=(len(self.forget_idx) < self.half_f))
                r_take = self.rng.choice(self.retain_idx, size=min(self.half_r, len(self.retain_idx)), replace=(len(self.retain_idx) < self.half_r))
                # if池子过小，补齐到batch_size
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

    # helper: build proportional batch sampler matching dataset-level forget ratio
    class ProportionalBatchSampler(torch.utils.data.Sampler[List[int]]):
        def __init__(self, targets: List[int], forget_set: set, batch_size: int, num_batches: int, seed: int = 0):
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
            # fallback: if one side empty, allow sampling from available set only
            if self.f_count == 0 and self.r_count == 0:
                self.all_idx = np.arange(len(self.targets))
            else:
                self.all_idx = None
        def __iter__(self):
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
                    # fill the rest from retain set if available; otherwise from forget set
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

    if args.in_dataset == "CIFAR-10":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(
                root=os.path.join(base_dir, 'cifarpy'), train=True, download=True, transform=transform_train)
            # enable balanced/proportional/retain-only sampling if requested and forget classes specified
            forget_set = set()
            # 优先使用当前阶段增量遗忘子集；若未提供则回退到全局遗忘集
            if getattr(args, 'forget_classes_inc', None):
                forget_set |= set(int(x) for x in str(args.forget_classes_inc).split(',') if x!='')
            elif getattr(args, 'forget_classes', None):
                forget_set |= set(int(x) for x in str(args.forget_classes).split(',') if x!='')
            mode = str(getattr(args, 'batch_forget_mode', 'none'))
            if mode in ('balanced', 'proportional', 'retain_only', 'forget_only') and len(forget_set) > 0:
                if mode == 'balanced':
                    num_batches = len(trainset) // args.batch_size
                    batch_sampler = BalancedBatchSampler(trainset.targets, forget_set, args.batch_size, num_batches, seed=args.seed)
                    train_loader = torch.utils.data.DataLoader(
                        trainset, batch_sampler=batch_sampler, **kwargs)
                elif mode == 'proportional':
                    num_batches = len(trainset) // args.batch_size
                    batch_sampler = ProportionalBatchSampler(trainset.targets, forget_set, args.batch_size, num_batches, seed=args.seed)
                    train_loader = torch.utils.data.DataLoader(
                        trainset, batch_sampler=batch_sampler, **kwargs)
                elif mode == 'retain_only':
                    targets = np.array(list(trainset.targets), dtype=np.int64)
                    mask_forget = np.isin(targets, np.array(sorted(list(forget_set)), dtype=np.int64))
                    retain_idx = np.where(~mask_forget)[0].tolist()
                    subset = torch.utils.data.Subset(trainset, retain_idx)
                    train_loader = torch.utils.data.DataLoader(
                        subset, batch_size=args.batch_size, shuffle=True, **kwargs)
                elif mode == 'forget_only':
                    targets = np.array(list(trainset.targets), dtype=np.int64)
                    mask_forget = np.isin(targets, np.array(sorted(list(forget_set)), dtype=np.int64))
                    forget_idx = np.where(mask_forget)[0].tolist()
                    subset = torch.utils.data.Subset(trainset, forget_idx)
                    train_loader = torch.utils.data.DataLoader(
                        subset, batch_size=args.batch_size, shuffle=True, **kwargs)
            else:
                train_loader = torch.utils.data.DataLoader(
                    trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(
                root=os.path.join(base_dir, 'cifarpy'), train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(
                valset, batch_size=args.batch_size, shuffle=val_shuffle, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR100(
                root=os.path.join(base_dir, 'cifarpy'), train=True, download=True, transform=transform_train)
            forget_set = set()
            # 优先使用当前阶段增量遗忘子集；若未提供则回退到全局遗忘集
            if getattr(args, 'forget_classes_inc', None):
                forget_set |= set(int(x) for x in str(args.forget_classes_inc).split(',') if x!='')
            elif getattr(args, 'forget_classes', None):
                forget_set |= set(int(x) for x in str(args.forget_classes).split(',') if x!='')
            mode = str(getattr(args, 'batch_forget_mode', 'none'))
            if mode in ('balanced', 'proportional', 'retain_only', 'forget_only') and len(forget_set) > 0:
                if mode == 'balanced':
                    num_batches = len(trainset) // args.batch_size
                    batch_sampler = BalancedBatchSampler(trainset.targets, forget_set, args.batch_size, num_batches, seed=args.seed)
                    train_loader = torch.utils.data.DataLoader(
                        trainset, batch_sampler=batch_sampler, **kwargs)
                elif mode == 'proportional':
                    num_batches = len(trainset) // args.batch_size
                    batch_sampler = ProportionalBatchSampler(trainset.targets, forget_set, args.batch_size, num_batches, seed=args.seed)
                    train_loader = torch.utils.data.DataLoader(
                        trainset, batch_sampler=batch_sampler, **kwargs)
                elif mode == 'retain_only':
                    targets = np.array(list(trainset.targets), dtype=np.int64)
                    mask_forget = np.isin(targets, np.array(sorted(list(forget_set)), dtype=np.int64))
                    retain_idx = np.where(~mask_forget)[0].tolist()
                    subset = torch.utils.data.Subset(trainset, retain_idx)
                    train_loader = torch.utils.data.DataLoader(
                        subset, batch_size=args.batch_size, shuffle=True, **kwargs)
                elif mode == 'forget_only':
                    targets = np.array(list(trainset.targets), dtype=np.int64)
                    mask_forget = np.isin(targets, np.array(sorted(list(forget_set)), dtype=np.int64))
                    forget_idx = np.where(mask_forget)[0].tolist()
                    subset = torch.utils.data.Subset(trainset, forget_idx)
                    train_loader = torch.utils.data.DataLoader(
                        subset, batch_size=args.batch_size, shuffle=True, **kwargs)
            else:
                train_loader = torch.utils.data.DataLoader(
                    trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(
                root=os.path.join(base_dir, 'cifarpy'), train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(
                valset, batch_size=args.batch_size, shuffle=val_shuffle, **kwargs)
    elif args.in_dataset == "CIFAR-110":
        # Mixed CIFAR-100 (labels 0-99) + CIFAR-10 (labels 100-109)
        if 'train' in split:
            trainset = CIFAR110Dataset(
                root=base_dir, train=True, transform=transform_train
            )
            # enable balanced/proportional/retain-only sampling if requested and forget classes specified
            forget_set = set()
            # 优先使用当前阶段增量遗忘子集；若未提供则回退到全局遗忘集
            if getattr(args, 'forget_classes_inc', None):
                forget_set |= set(int(x) for x in str(args.forget_classes_inc).split(',') if x!='')
            elif getattr(args, 'forget_classes', None):
                forget_set |= set(int(x) for x in str(args.forget_classes).split(',') if x!='')
            mode = str(getattr(args, 'batch_forget_mode', 'none'))
            if mode in ('balanced', 'proportional', 'retain_only', 'forget_only') and len(forget_set) > 0:
                if mode == 'balanced':
                    num_batches = len(trainset) // args.batch_size
                    batch_sampler = BalancedBatchSampler(trainset.targets, forget_set, args.batch_size, num_batches, seed=args.seed)
                    train_loader = torch.utils.data.DataLoader(
                        trainset, batch_sampler=batch_sampler, **kwargs)
                elif mode == 'proportional':
                    num_batches = len(trainset) // args.batch_size
                    batch_sampler = ProportionalBatchSampler(trainset.targets, forget_set, args.batch_size, num_batches, seed=args.seed)
                    train_loader = torch.utils.data.DataLoader(
                        trainset, batch_sampler=batch_sampler, **kwargs)
                elif mode == 'retain_only':
                    targets = np.array(list(trainset.targets), dtype=np.int64)
                    mask_forget = np.isin(targets, np.array(sorted(list(forget_set)), dtype=np.int64))
                    retain_idx = np.where(~mask_forget)[0].tolist()
                    subset = torch.utils.data.Subset(trainset, retain_idx)
                    train_loader = torch.utils.data.DataLoader(
                        subset, batch_size=args.batch_size, shuffle=True, **kwargs)
                elif mode == 'forget_only':
                    targets = np.array(list(trainset.targets), dtype=np.int64)
                    mask_forget = np.isin(targets, np.array(sorted(list(forget_set)), dtype=np.int64))
                    forget_idx = np.where(mask_forget)[0].tolist()
                    subset = torch.utils.data.Subset(trainset, forget_idx)
                    train_loader = torch.utils.data.DataLoader(
                        subset, batch_size=args.batch_size, shuffle=True, **kwargs)
            else:
                train_loader = torch.utils.data.DataLoader(
                    trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = CIFAR110Dataset(
                root=base_dir, train=False, transform=transform_test
            )
            val_loader = torch.utils.data.DataLoader(
                valset, batch_size=args.batch_size, shuffle=val_shuffle, **kwargs)

    if 'train' in split:
        return train_loader, num_classes_dict[args.in_dataset]
    else:
        return val_loader, num_classes_dict[args.in_dataset]


def get_loader_out(args: argparse.Namespace, dataset: str, split: Sequence[str] = ('train', 'val'), mode: str = 'eval') -> Optional[torch.utils.data.DataLoader]:
    base_dir = './data'
        
    kwargs = {'num_workers': max(2, min((os.cpu_count() or 1) - 2, 16)), 'pin_memory': True, 'persistent_workers': True}
    if mode == 'train':
        transform_test = get_transform_test(args.in_dataset)
    elif mode == "eval":
        transform_test = get_transform_eval(args.in_dataset)
    val_ood_loader = None

    if 'val' in split:
        val_dataset = dataset
        batch_size = args.batch_size
        if val_dataset == 'SVHN':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(os.path.join(base_dir, 'svhn'), split='test', transform=transform_test, download=False),
                                                         batch_size=batch_size, shuffle=val_shuffle, **kwargs)
        elif val_dataset == 'dtd':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=os.path.join(base_dir, 'dtd/images'), transform=transform_test),
                                                         batch_size=batch_size, shuffle=val_shuffle, **kwargs)
        elif val_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root=os.path.join(base_dir, "Places365/test_subset/"), transform=transform_test)
            if len(testsetout) > 10000: 
                testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
            val_ood_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=val_shuffle, **kwargs)
        elif val_dataset == 'CIFAR-100':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root=os.path.join(base_dir, 'cifarpy'), train=False, download=True, transform=transform_test),
                                                         batch_size=batch_size, shuffle=val_shuffle, **kwargs)
        elif val_dataset == 'CIFAR-10':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=os.path.join(base_dir, 'cifarpy'), train=False, download=True, transform=transform_test),
                                                         batch_size=batch_size, shuffle=val_shuffle, **kwargs)
        else:
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=os.path.join(base_dir, val_dataset),transform=transform_test), 
                                                         batch_size=batch_size, shuffle=val_shuffle, **kwargs)
    return val_ood_loader