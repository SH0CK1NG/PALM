import os
import json
import torch
import numpy as np
from util.loaders.args_loader import get_args
from util.loaders.data_loader import get_loader_in
from util.loaders.model_loader import get_model
from sklearn.covariance import EmpiricalCovariance


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    loader, num_classes = get_loader_in(args, split='train', mode='eval')
    model = get_model(args, num_classes, load_ckpt=True)
    model.to(device)
    model.eval()

    feats = []
    labels = []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            f = model.encoder(images) if hasattr(model, 'encoder') else model(images)
            f = f.detach().cpu()
            feats.append(f)
            labels.append(y)
    feats = torch.cat(feats, dim=0).float()
    labels = torch.cat(labels, dim=0).long()

    # compute class centers
    C = num_classes
    D = feats.shape[1]
    centers = torch.zeros((C, D), dtype=torch.float32)
    for c in range(C):
        idx = (labels == c)
        if torch.any(idx):
            centers[c] = feats[idx].mean(dim=0)

    # shared precision from centered residuals wrt class centers
    center_samples = centers[labels]
    r = feats - center_samples
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(r.numpy())
    precision = torch.tensor(ec.precision_, dtype=torch.float32)

    out_dir = os.path.join('cache', f'{args.backbone}-{args.method}', args.in_dataset)
    os.makedirs(out_dir, exist_ok=True)
    centers_path = os.path.join(out_dir, 'class_centers.pt')
    precision_path = os.path.join(out_dir, 'precision.pt')
    torch.save(centers, centers_path)
    torch.save(precision, precision_path)
    print(f'Saved centers -> {centers_path}')
    print(f'Saved precision -> {precision_path}')


if __name__ == '__main__':
    main()


