import torch
import torch.nn as nn
from util.loss_functions import *
from torch.optim.lr_scheduler import MultiStepLR
from models.resnet import  SupCEResNet, PALMResNet
from util.lora import apply_lora_to_resnet_head, apply_lora_to_resnet_layer4, load_lora_state_dict
from util.peft_utils import is_peft_available, apply_peft_lora_to_model, load_peft_adapter

def get_model(args, num_classes, load_ckpt=True):
    method = args.method
    if 'palm' in method:
        model = PALMResNet(name=args.backbone)
    else:
        model = SupCEResNet(name=args.backbone, num_classes=num_classes)
        
    if load_ckpt:
        ckpt_path = getattr(args, 'load_path', None) or getattr(args, 'save_path', None)
        checkpoint = torch.load(ckpt_path, map_location="cuda:0")
        # support both old pure state_dict and new dict with keys
        state_dict = None
        if isinstance(checkpoint, dict):
            for k in ['model', 'state_dict', 'model_state']:
                if k in checkpoint and isinstance(checkpoint[k], dict):
                    state_dict = checkpoint[k]
                    break
        if state_dict is None:
            state_dict = checkpoint
        # strip DataParallel prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[ckpt] missing keys: {list(missing)[:5]}{' ...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"[ckpt] unexpected keys: {list(unexpected)[:5]}{' ...' if len(unexpected)>5 else ''}")
        print(f"ckpt loaded from {ckpt_path}")

    # apply lora if requested (for PALM models)
    if getattr(args, 'use_lora', False) and 'palm' in method:
        target = getattr(args, 'lora_target', 'head')
        r = getattr(args, 'lora_r', 8)
        alpha = getattr(args, 'lora_alpha', 32)
        dropout = getattr(args, 'lora_dropout', 0.05)
        applied = False
        # Prefer PEFT when available and user selects peft
        prefer_peft = (getattr(args, 'lora_impl', 'native') == 'peft')
        if is_peft_available() and prefer_peft:
            try:
                model, target_modules = apply_peft_lora_to_model(model, target=target, r=r, alpha=alpha, dropout=dropout)
                applied = len(target_modules) > 0
                # optionally load peft adapter directory
                if getattr(args, 'adapter_load_path', None):
                    try:
                        model = load_peft_adapter(model, args.adapter_load_path)
                        print(f"[peft] adapter loaded from {args.adapter_load_path}")
                    except Exception as e:
                        print(f"[peft] failed to load adapter: {e}")
            except Exception as e:
                print(f"[peft] failed to apply lora, fallback to custom: {e}")
                applied = False
        # Fallback to custom lightweight LoRA
        if not applied:
            applied_head = False
            applied_layer4 = False
            if target in ['head', 'both']:
                applied_head = apply_lora_to_resnet_head(model, r=r, alpha=alpha, dropout=dropout)
            if target in ['encoder', 'both']:
                applied_layer4 = apply_lora_to_resnet_layer4(model, r=r, alpha=alpha, dropout=dropout)
            applied = applied_head or applied_layer4
            if applied:
                model.train()
                # freeze all non-LoRA params by default when using LoRA
                for n, p in model.named_parameters():
                    p.requires_grad = False
                for m in model.modules():
                    if hasattr(m, 'A') and hasattr(m, 'B'):
                        if m.A is not None:
                            m.A.requires_grad_(True)
                        if m.B is not None:
                            m.B.requires_grad_(True)
                # optionally load adapter-only weights
                if getattr(args, 'adapter_load_path', None):
                    try:
                        ad = torch.load(args.adapter_load_path, map_location="cuda:0")
                        lora_sd = ad['lora'] if isinstance(ad, dict) and 'lora' in ad else ad
                        loaded = load_lora_state_dict(model, lora_sd)
                        print(f"[lora] loaded {loaded} LoRA tensors from {args.adapter_load_path}")
                    except Exception as e:
                        print(f"[lora] failed to load adapter: {e}")
            else:
                model.eval()
    else:
        model.eval()
    
    # get the number of model parameters
    print(f'{args.backbone}-{args.method}: Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model


def get_criterion(args, num_classes, model):
    arch = args.method
    if 'palm' in arch:
        cri = PALM(args, temp=args.temp, num_classes=num_classes, proto_m=args.proto_m, n_protos=num_classes*args.cache_size,  k=args.k, lambda_pcon=args.lambda_pcon)
    else:
        cri = nn.CrossEntropyLoss()

    return cri
    
def get_encoder_dim(model):
    dummy_input = torch.zeros((1, 3, 32, 32))
    features = model.encoder(dummy_input)
    featdims = features.shape[1]
    return featdims
    
def set_model(args, num_classes, load_ckpt=True, load_epoch=None):
    model = get_model(args, num_classes, load_ckpt)
    criterion = get_criterion(args, num_classes, model)
    # try load criterion/prototypes if present; also handle class expansion
    try:
        ckpt_path = getattr(args, 'load_path', None) or getattr(args, 'save_path', None)
        checkpoint = torch.load(ckpt_path, map_location="cuda:0") if load_ckpt else None
    except Exception:
        checkpoint = None
    if checkpoint and isinstance(checkpoint, dict) and 'criterion' in checkpoint:
        old_state = checkpoint['criterion']
        # if class count / n_protos changed, expand; else direct load
        if 'protos' in old_state:
            try:
                new_state = criterion.state_dict()
                old_protos = old_state['protos']
                new_protos = new_state['protos']
                if old_protos.shape != new_protos.shape:
                    # attempt expansion along class axis: layout is [cache_size*classes, feat]
                    cache_size = getattr(args, 'cache_size', 1)
                    # infer class counts
                    feat_dim = old_protos.shape[1]
                    old_C = old_protos.shape[0] // cache_size
                    new_C = new_protos.shape[0] // cache_size
                    if old_C < new_C and old_C*cache_size == old_protos.shape[0] and new_C*cache_size == new_protos.shape[0]:
                        # copy old into new for each cache slot
                        for r in range(cache_size):
                            new_protos[r*new_C : r*new_C + old_C] = old_protos[r*old_C : (r+1)*old_C]
                        new_state['protos'] = torch.nn.functional.normalize(new_protos, dim=1)
                        criterion.load_state_dict(new_state, strict=False)
                    else:
                        criterion.load_state_dict(old_state, strict=False)
                else:
                    criterion.load_state_dict(old_state, strict=False)
            except Exception:
                pass
        else:
            try:
                criterion.load_state_dict(old_state, strict=False)
            except Exception:
                pass
    return model, criterion

def set_schedular(args, optimizer):
    scheduler = MultiStepLR(optimizer, milestones=[50,75,90], gamma=0.1)
    return scheduler