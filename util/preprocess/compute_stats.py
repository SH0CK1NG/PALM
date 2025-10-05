import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

import os, torch, numpy as np
from torchvision import transforms
from util.loaders.data_loader import CIFAR110Dataset

def compute_mean_std():
    ds = CIFAR110Dataset(root=base_dir/'data', train=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
    n = 0
    mean = torch.zeros(3)
    M2 = torch.zeros(3)  # Welford 累积二阶矩
    for x, _ in loader:
        b = x.size(0)
        x = x.view(b, 3, -1)  # [B,3,H*W]
        batch_mean = x.mean(dim=(0,2))
        batch_var = x.var(dim=(0,2), unbiased=False)
        # 合并批次统计
        total = n + b
        delta = batch_mean - mean
        mean = mean + delta * (b/total)
        M2 = M2 + batch_var*b + (delta**2) * (n*b/total)
        n = total
    var = M2 / n
    std = torch.sqrt(var + 1e-12)
    print('CIFAR-110 mean =', mean.tolist())
    print('CIFAR-110 std  =', std.tolist())

if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    compute_mean_std()