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
from util.lora import extract_lora_state_dict, load_lora_state_dict
from util.peft_utils import is_peft_available, save_peft_adapter
 

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def main():

    train_loader, num_classes = get_loader_in(args, split='train')

    # load pretrained model/criterion (including prototypes) when load-path is provided
    load_pretrained = bool(getattr(args, 'load_path', None))
    model, criterion = set_model(args, num_classes, load_ckpt=load_pretrained)
    model.to(device)
    model.encoder.to(device)
    criterion.to(device)

    # debug: quick self-check of trainable params (should be only LoRA A/B)
    try:
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"[debug] trainable_count = {len(trainable)}")
        head_list = trainable[:50]
        for n in head_list:
            print(f"[debug] trainable: {n}")
        others = [n for n in trainable if ('.A' not in n and '.B' not in n and not n.endswith('A.weight') and not n.endswith('B.weight'))]
        if len(others) > 0:
            print(f"[debug][warn] non-LoRA trainables detected: {others[:10]}")
    except Exception as e:
        print(f"[debug] trainable check failed: {e}")
    


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

    # load class centers / precision if provided (for forgetting loss)
    args.centers = None
    args.precision = None
    try:
        if getattr(args, 'centers_path', None) and os.path.exists(args.centers_path):
            args.centers = torch.load(args.centers_path, map_location=device)
        if getattr(args, 'precision_path', None) and os.path.exists(args.precision_path):
            args.precision = torch.load(args.precision_path, map_location=device)
    except Exception as e:
        print(f"[centers] failed to load centers/precision: {e}")

    

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
            # strictly save adapters only if requested
            if getattr(args, 'adapter_save_path', None):
                save_dir = os.path.dirname(args.adapter_save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                # If PEFT is available and model is PEFT-wrapped, save via PEFT format; else save custom lightweight
                if is_peft_available():
                    try:
                        save_peft_adapter(model, args.adapter_save_path)
                        print(f"[peft] adapter saved to {args.adapter_save_path}")
                    except Exception as e:
                        print(f"[peft] save failed, fallback to custom: {e}")
                        lora_sd = extract_lora_state_dict(model)
                        torch.save({'lora': lora_sd,
                                    'meta': {'in_dataset': args.in_dataset,
                                             'backbone': args.backbone,
                                             'method': args.method,
                                             'num_classes': num_classes}},
                                  args.adapter_save_path)
                else:
                    lora_sd = extract_lora_state_dict(model)
                    torch.save({'lora': lora_sd,
                                'meta': {'in_dataset': args.in_dataset,
                                         'backbone': args.backbone,
                                         'method': args.method,
                                         'num_classes': num_classes}},
                              args.adapter_save_path)
            else:
                # fallback: save full model as before
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
