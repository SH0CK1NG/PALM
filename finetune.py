import os
import torch
import numpy as np
from tqdm import tqdm

from util.loaders.args_loader import get_args
from util.loaders.data_loader import get_transform_train, get_transform_eval
from util.loaders.model_loader import set_model
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

    # load model+criterion，内部自动兼容旧权重并在需要时扩展原型
    # 先构造一个假的 args 给 set_model 使用新的 num_classes_new
    class _ArgsProxy:
        def __init__(self, base, in_dataset_override):
            self.__dict__.update(base.__dict__)
            self.in_dataset = in_dataset_override
    args_new = _ArgsProxy(args, 'CIFAR-110')
    model, criterion = set_model(args_new, num_classes_new, load_ckpt=True)
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()

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

    # save finetuned checkpoint - filename must include epochs & lr
    lr_str = str(args.lr).replace('.', 'p')
    # if user provided save-path:
    #  - if it endswith .pt: append suffix with ep/lr to basename
    #  - else: treat as directory
    if getattr(args, 'save_path', None):
        base_path = args.save_path
        if base_path.endswith('.pt'):
            base_dir = os.path.dirname(base_path)
            base_name = os.path.splitext(os.path.basename(base_path))[0]
            os.makedirs(base_dir or 'checkpoints/finetune', exist_ok=True)
            save_path = os.path.join(base_dir or 'checkpoints/finetune', f"{base_name}-ep{args.epochs}-lr{lr_str}.pt")
        else:
            save_dir = base_path
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{args.in_dataset}-{args.backbone}-{args.method}-ft-ep{args.epochs}-lr{lr_str}.pt")
    else:
        save_dir = 'checkpoints/finetune'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{args.in_dataset}-{args.backbone}-{args.method}-ft-ep{args.epochs}-lr{lr_str}.pt")
    ckpt = {
        'model': model.state_dict(),
        'criterion': criterion.state_dict(),
        'meta': {
            'in_dataset': args.in_dataset,
            'backbone': args.backbone,
            'method': args.method,
            'num_classes': num_classes_new,
            'cache_size': getattr(args, 'cache_size', None),
            'epochs': args.epochs,
            'lr': args.lr,
        }
    }
    torch.save(ckpt, save_path)
    print(f"Finetuned checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()


