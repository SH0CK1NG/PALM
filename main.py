import torch
import os
import torch.backends.cudnn as cudnn
from util.loaders.args_loader import get_args
from util.loaders.data_loader import get_loader_in
from util.loaders.model_loader import set_model
from util.train_utils import get_optimizer
from trainer import get_trainer
import numpy as np
from tqdm import tqdm
from util.residual_space import compute_dcc_basis_from_loader
from util.loaders.data_loader import get_loader_in as _get_loader_in_eval

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def main():

    train_loader, num_classes = get_loader_in(args, split='train')

    model, criterion = set_model(args, num_classes, load_ckpt=False)
    model.to(device)
    model.encoder.to(device)
    criterion.to(device)
    # placeholders; will compute after possible resume
    args.residual_basis = None
    args.residual_projector = None


    # build optimizer
    optimizer = get_optimizer(args, model, criterion)
    loss_min = np.Inf

    # tensorboard
    t = datetime.now().strftime("%d-%B-%Y-%H:%M:%S")
    logger = SummaryWriter(log_dir=f"runs/{args.backbone}-{args.method}/{t}")

    # get trainer and scaler
    trainer = get_trainer(args)
    scaler = torch.cuda.amp.GradScaler()
                
    # load checkpoint if incremental and load_path/save_path exists
    load_candidate = getattr(args, 'load_path', None) or args.save_path
    if getattr(args, 'incremental', False) and load_candidate and os.path.exists(load_candidate):
        try:
            ckpt = torch.load(load_candidate, map_location=device)
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
            if 'criterion' in ckpt:
                try:
                    criterion.load_state_dict(ckpt['criterion'], strict=False)
                except Exception:
                    pass
            print(f"[incremental] resumed from {load_candidate}")
        except Exception as e:
            print(f"[incremental] failed to load ckpt: {e}")

    # compute/load residual basis if enabled (after possible resume)
    if getattr(args, 'residual_space', False):
        loaded = False
        # prefer loading from provided paths
        try:
            if getattr(args, 'residual_projector_path', None):
                path = args.residual_projector_path
                if path.endswith('.npy'):
                    proj_np = np.load(path)
                    projector = torch.as_tensor(proj_np, dtype=torch.float32, device=device)
                else:
                    projector = torch.load(path, map_location=device)
                args.residual_projector = projector
                # if basis not provided, leave None
                loaded = True
            if not loaded and getattr(args, 'residual_basis_path', None):
                path = args.residual_basis_path
                if path.endswith('.npy'):
                    basis_np = np.load(path)
                    basis = torch.as_tensor(basis_np, dtype=torch.float32, device=device)
                else:
                    basis = torch.load(path, map_location=device)
                args.residual_basis = basis
                args.residual_projector = basis.T @ basis
                loaded = True
        except Exception as e:
            print(f"[residual] failed to load provided basis/projector: {e}")

        if not loaded:
            # build residual basis on ID training set (with eval transforms)
            train_eval_loader, _ = _get_loader_in_eval(args, split='train', mode='eval')
            basis, projector = compute_dcc_basis_from_loader(args, model, train_eval_loader, device=torch.device(device))
            args.residual_basis = basis
            args.residual_projector = projector
            # optional save
            try:
                if getattr(args, 'save_residual_basis_path', None):
                    os.makedirs(os.path.dirname(args.save_residual_basis_path), exist_ok=True)
                    torch.save(basis, args.save_residual_basis_path)
                if getattr(args, 'save_residual_projector_path', None):
                    os.makedirs(os.path.dirname(args.save_residual_projector_path), exist_ok=True)
                    torch.save(projector, args.save_residual_projector_path)
            except Exception as e:
                print(f"[residual] failed to save basis/projector: {e}")

    for epoch in tqdm(range(args.epochs)):
        loss = trainer(args, train_loader, model, criterion, optimizer, epoch, scaler=scaler)
        
        if type(loss)==tuple:
            loss, l_dict = loss
            logger.add_scalar('Loss/train', loss, epoch)
            for k in l_dict.keys():
                logger.add_scalar(f'Loss/{k}', l_dict[k], epoch)
        else:
            logger.add_scalar('Loss/train', loss, epoch)
        logger.add_scalar('Lr/train', optimizer.param_groups[0]['lr'], epoch)

        if loss < loss_min:
            loss_min = loss
            # ensure directory exists
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            # save model and prototypes (criterion) together for incremental learning
            ckpt = {
                'model': model.state_dict(),
                'meta': {
                    'in_dataset': args.in_dataset,
                    'backbone': args.backbone,
                    'method': args.method,
                    'num_classes': num_classes,
                    'cache_size': getattr(args, 'cache_size', None),
                }
            }
            try:
                ckpt['criterion'] = criterion.state_dict()
            except Exception:
                pass
            torch.save(ckpt, args.save_path)

if __name__ == "__main__":

    FORCE_RUN = True
    # FORCE_RUN=False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)
    args.save_epoch = 50

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # check if the model is trained
    if os.path.exists(args.save_path) and not FORCE_RUN:
        exit()

    main()
