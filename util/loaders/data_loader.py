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


kwargs = {'num_workers': 2, 'pin_memory': True}
num_classes_dict = {'CIFAR-100': 100, 'CIFAR-10': 10, 'CIFAR-110': 110, 'imagenet': 1000}

val_shuffle = False


def get_loader_in(args: argparse.Namespace, split: Tuple[str, ...] = ('train', 'val', 'csid'), mode: str = 'train') -> Tuple[torch.utils.data.DataLoader, int]:
    base_dir = './data'
    
    kwargs = {'num_workers': 2, 'pin_memory': True}
    num_classes_dict = {'CIFAR-100': 100, 'CIFAR-10': 10, 'CIFAR-110': 110, 'imagenet': 1000}
    if mode == 'train':
        transform_train = get_transform_train(args.in_dataset, args.method)
        transform_test = get_transform_test(args.in_dataset)
    elif mode == "eval":
        transform_train = get_transform_eval(args.in_dataset)
        transform_test = get_transform_eval(args.in_dataset)

    if args.in_dataset == "CIFAR-10":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(
                root=os.path.join(base_dir, 'cifarpy'), train=True, download=True, transform=transform_train)
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
        
    kwargs = {'num_workers': 4, 'pin_memory': True}
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