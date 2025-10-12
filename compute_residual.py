import os
import torch
import numpy as np
from util.loaders.args_loader import get_args
from util.loaders.data_loader import get_loader_in
from util.loaders.model_loader import set_model
from util.residual_space import compute_dcc_basis_from_loader


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # build data loader on ID training set with eval transforms
    # reuse get_loader_in in eval mode via alias in main.py pattern
    from util.loaders.data_loader import get_loader_in as _get_loader_in_eval
    train_eval_loader, num_classes = _get_loader_in_eval(args, split='train', mode='eval')

    # model (no training) - load ckpt if present to align features
    model, _ = set_model(args, num_classes, load_ckpt=True)
    model.to(device)
    model.eval()

    # compute residual basis/projector
    basis, projector = compute_dcc_basis_from_loader(args, model, train_eval_loader, device=torch.device(device))

    # output paths
    out_basis = getattr(args, 'save_residual_basis_path', None) or os.path.join('cache', f'{args.backbone}-{args.method}', args.in_dataset, 'residual_basis.pt')
    out_proj = getattr(args, 'save_residual_projector_path', None) or os.path.join('cache', f'{args.backbone}-{args.method}', args.in_dataset, 'residual_projector.pt')
    os.makedirs(os.path.dirname(out_basis), exist_ok=True)
    os.makedirs(os.path.dirname(out_proj), exist_ok=True)
    torch.save(basis, out_basis)
    torch.save(projector, out_proj)
    print(f'Saved basis to {out_basis}')
    print(f'Saved projector to {out_proj}')


if __name__ == '__main__':
    main()


'''
冷启动预处理并保存：
保存到默认路径：
python compute_residual.py --in-dataset CIFAR-100 --backbone resnet34 --method top5-palm-cache6-ema0.999 --save-path checkpoints/CIFAR-100-resnet34-top5-palm-cache6-ema0.999-with-prototypes.pt --residual_at encoder --residual_dim 128
指定保存路径：
python compute_residual.py --in-dataset CIFAR-10 --backbone resnet34 --method palm --save_residual_projector_path cache/C10/P.pt --save_residual_basis_path cache/C10/B.pt
训练时加载：
python main.py --method palm --residual_space --residual_projector_path cache/C10/P.pt
或 --residual_basis_path cache/C10/B.pt（会自动生成 P = B^T B）
'''