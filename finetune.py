import os
import torch
import numpy as np
from tqdm import tqdm

from util.loaders.args_loader import get_args
from util.loaders.data_loader import get_transform_train, get_transform_eval
from util.loaders.model_loader import get_model
from util.loss_functions import PALM
from util.train_utils import get_optimizer, AverageMeter, adjust_learning_rate


def build_cifar10_loader_as_new_classes(args, split="train"):
    import torchvision
    base_dir = './data'
    transform = get_transform_eval(args.in_dataset) if split == 'val' else get_transform_train(args.in_dataset, args.method)
    dataset = torchvision.datasets.CIFAR10(root=os.path.join(base_dir, 'cifarpy'), train=(split=='train'), download=True, transform=transform)

    # wrap to offset labels by +100 -> new classes 100-109
    class OffsetC10(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return img, int(label) + 100

    wrapped = OffsetC10(dataset)
    loader = torch.utils.data.DataLoader(wrapped, batch_size=args.batch_size, shuffle=(split=='train'), num_workers=2, pin_memory=True)
    return loader


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # override requested hyperparams for finetune if provided by user
    # expected: --lr 0.01 --epochs 100 --in-dataset CIFAR-110 --method top5-palm-cache6-ema0.999

    # Build dataloaders: only CIFAR-10 part as new classes (100-109)
    train_loader = build_cifar10_loader_as_new_classes(args, split='train')
    val_loader = build_cifar10_loader_as_new_classes(args, split='val')

    # Create PALM model and criterion with 110 classes, load previous checkpoint weights
    num_classes_old = 100
    num_classes_new = 110

    # load model backbone + head from checkpoint (trained with 100 classes)
    model = get_model(args, num_classes=num_classes_old, load_ckpt=True)
    model = model.to(device)
    model.train()

    # Build PALM criterion with extended prototypes (110 classes)
    criterion = PALM(args, temp=args.temp, num_classes=num_classes_new, proto_m=args.proto_m, n_protos=num_classes_new*args.cache_size, k=args.k, lambda_pcon=args.lambda_pcon).to(device)

    # If you have stored old prototypes, you could load and expand here. In this code, we initialize new protos while allowing EMA updates to adapt.

    optimizer = get_optimizer(args, model, criterion)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    losses = AverageMeter()

    # fine-tune loop on new classes only, updating prototypes; use fine_tune forward if set
    for epoch in tqdm(range(args.epochs)):
        model.train()
        for step, (images, labels) in enumerate(train_loader, start=epoch * len(train_loader)):
            # 支援 TwoCropTransform：images 可能是 [x1, x2]
            is_twocrop = isinstance(images, list) and len(images) == 2

            if torch.cuda.is_available():
                if is_twocrop:
                    images = [img.cuda(non_blocking=True) for img in images]
                else:
                    images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    if is_twocrop:
                        imgs_cat = torch.cat(images, dim=0)
                        feats_cat = model(imgs_cat)
                        f1, f2 = torch.split(feats_cat, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    else:
                        f = model(images)
                        features = f.unsqueeze(1)
                    loss, _ = criterion(features, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scale = scaler.get_scale()
                scaler.update()
                new_scale = scaler.get_scale()
            else:
                if is_twocrop:
                    imgs_cat = torch.cat(images, dim=0)
                    feats_cat = model(imgs_cat)
                    f1, f2 = torch.split(feats_cat, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                else:
                    f = model(images)
                    features = f.unsqueeze(1)
                loss, _ = criterion(features, labels)
                loss.backward()
                optimizer.step()
                old_scale = new_scale = 1.0

            losses.update(loss.item(), bsz)
            if (not scaler) or (new_scale >= old_scale):
                adjust_learning_rate(args, optimizer, train_loader, step)

    # save finetuned checkpoint
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.in_dataset}-{args.backbone}-{args.method}-finetuned-110.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Finetuned checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()


