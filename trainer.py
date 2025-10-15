import torch
import numpy as np
import os
from tqdm import tqdm

from util.train_utils import adjust_learning_rate, AverageMeter


def train_palm(args, train_loader, model, criterion, optimizer, epoch, scaler=None):
    model.train()

    losses = AverageMeter()
    sub_loss = {}
    # optional: cache the very first batch for fixed-batch debugging
    use_fixed = bool(getattr(args, 'debug_fixed_batch', False))
    fixed_batch = None
    steps_limit = int(getattr(args, 'debug_fixed_batch_steps', 50)) if use_fixed else None

    step_iterable = enumerate(train_loader, start=epoch * len(train_loader)) if not use_fixed else None
    step_count = 0
    while True:
        if use_fixed:
            if fixed_batch is None:
                try:
                    # pull one real batch from loader to cache it
                    fixed_batch = next(iter(train_loader))
                except Exception:
                    break
            images, labels = fixed_batch
        else:
            try:
                step, (images, labels) = next(step_iter := (step_iterable if 'step_iter' in locals() else iter(train_loader)))
                step_iterable = step_iter
            except StopIteration:
                break
        step = (epoch * len(train_loader)) + step_count
        step_count += 1
        if use_fixed and step_count > steps_limit:
            break
        # detect two-crop before any concatenation
        twocrop = isinstance(images, (list, tuple)) and len(images) == 2

        # parse forget classes set (CPU tensors here)
        forget_classes = set()
        if getattr(args, 'forget_classes', None):
            forget_classes |= set(int(x) for x in str(args.forget_classes).split(',') if x!='')
        elif getattr(args, 'forget_list_path', None) and os.path.exists(args.forget_list_path):
            try:
                import json
                with open(args.forget_list_path) as f:
                    data = f.read().strip()
                    try:
                        arr = json.loads(data)
                    except Exception:
                        arr = [int(line) for line in data.splitlines() if line.strip()!='']
                    forget_classes |= set(int(x) for x in arr)
            except Exception:
                pass
        if len(forget_classes) > 0:
            fcls_cpu = torch.tensor(sorted(list(forget_classes)))
            forget_mask_cpu = (labels.view(-1, 1) == fcls_cpu.view(1, -1)).any(dim=1)
            retain_mask_cpu = ~forget_mask_cpu
        else:
            forget_mask_cpu = torch.zeros(labels.shape[0], dtype=torch.bool)
            retain_mask_cpu = torch.ones(labels.shape[0], dtype=torch.bool)

        # pre-forward sub-selection according to batch_forget_mode
        mode = str(getattr(args, 'batch_forget_mode', 'none'))
        keep_idx = None
        # precompute indices
        f_idx = torch.nonzero(forget_mask_cpu, as_tuple=False).squeeze(1)
        r_idx = torch.nonzero(retain_mask_cpu, as_tuple=False).squeeze(1)
        if mode == 'retain_only':
            keep_idx = r_idx
        elif mode in ('balanced', 'proportional') and (forget_mask_cpu.any() or retain_mask_cpu.any()):
            if mode == 'balanced':
                k = min(len(f_idx), len(r_idx))
                if k > 0:
                    f_perm = f_idx[torch.randperm(len(f_idx))[:k]]
                    r_perm = r_idx[torch.randperm(len(r_idx))[:k]]
                    keep_idx = torch.cat([f_perm, r_perm], dim=0)
            elif mode == 'proportional':
                # downsample majority side to approximate within-batch ratio, keep total size as current
                f_count = int(forget_mask_cpu.sum().item())
                r_count = int(retain_mask_cpu.sum().item())
                total = f_count + r_count
                if total > 0:
                    pf = f_count / total
                    target_f = int(round(pf * total))
                    target_r = total - target_f
                    if f_count > 0 and r_count > 0:
                        f_take = min(f_count, target_f)
                        r_take = min(r_count, target_r)
                        f_perm = f_idx[torch.randperm(len(f_idx))[:f_take]]
                        r_perm = r_idx[torch.randperm(len(r_idx))[:r_take]]
                        keep_idx = torch.cat([f_perm, r_perm], dim=0)

        # apply selection if any
        if keep_idx is not None and keep_idx.numel() > 0:
            labels = labels[keep_idx]
            if twocrop:
                # images is a list of two views
                images = [images[0][keep_idx], images[1][keep_idx]]
            else:
                images = images[keep_idx]

        # finally, move to device and (if two-crop) concatenate for model forward
        if torch.cuda.is_available():
            if twocrop:
                images = torch.cat([images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)], dim=0)
            else:
                images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        else:
            if twocrop:
                images = torch.cat([images[0], images[1]], dim=0)
        bsz = labels.shape[0]

        # compute loss
        optimizer.zero_grad()
        # debug vars for printing
        dmin_value = None
        dmin_norm_value = None
        nf_out = None
        nr_out = None
        centers_shape = None
        p_shape = None
        if scaler:
            with torch.cuda.amp.autocast():
                if args.fine_tune:
                    features = model.fine_tune_forward(images)
                else:
                    features = model(images)
                if twocrop:
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                else:
                    features = features.unsqueeze(1)

                # compute retain/forget masks on selected labels (for restricting L_PALM to retained samples)
                if len(forget_classes) > 0:
                    fcls = torch.tensor(sorted(list(forget_classes)), device=labels.device)
                    forget_mask = (labels.unsqueeze(1) == fcls.unsqueeze(0)).any(dim=1)
                    retain_mask = ~forget_mask
                else:
                    forget_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
                    retain_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
                # cache counts for debug
                try:
                    nf_out = int(forget_mask.sum().item())
                    nr_out = int(retain_mask.sum().item())
                except Exception:
                    nf_out = nr_out = None

                # note: final retain_mask/forget_mask are on the selected subset only

                # L_PALM only on retained samples; if none retained, use zero loss tied to graph
                if retain_mask.any():
                    loss, l_dict = criterion(features[retain_mask], labels[retain_mask])
                else:
                    # no retained samples: zero PALM loss but keep keys consistent
                    loss = features.sum() * 0.0
                    l_dict = {'mle': 0.0, 'proto_contra': 0.0}
                # add Mahalanobis forgetting loss if configured
                if getattr(args, 'forget_lambda', 0) > 0 and (getattr(args, 'centers', None) is not None) and (getattr(args, 'precision', None) is not None):
                    # parse forget classes set
                    forget_classes = set()
                    if getattr(args, 'forget_classes', None):
                        forget_classes |= set(int(x) for x in str(args.forget_classes).split(',') if x!='')
                    elif getattr(args, 'forget_list_path', None) and os.path.exists(args.forget_list_path):
                        try:
                            import json
                            with open(args.forget_list_path) as f:
                                data = f.read().strip()
                                try:
                                    arr = json.loads(data)
                                except Exception:
                                    arr = [int(line) for line in data.splitlines() if line.strip()!='']
                                forget_classes |= set(int(x) for x in arr)
                        except Exception:
                            pass
                    # initialize forget component value
                    forget_term = 0.0
                    if len(forget_classes) > 0:
                        # select center set
                        centers = args.centers
                        if str(getattr(args, 'forget_center_set','all')) == 'retain':
                            retain_idx = [i for i in range(centers.shape[0]) if i not in forget_classes]
                            centers = centers[retain_idx]
                        try:
                            centers_shape = f"{centers.shape[0]}x{centers.shape[1]}"
                        except Exception:
                            centers_shape = None
                        # Always use encoder space (512-D) for forgetting term to match centers
                        enc_feats = model.encoder(images)
                        if twocrop:
                            f1e, _ = torch.split(enc_feats, [bsz, bsz], dim=0)
                            feats = f1e
                        else:
                            feats = enc_feats
                        # use forget subset for forgetting term
                        if forget_mask.any():
                            feats_f = feats[forget_mask]
                            # compute Mahalanobis distance (float32, stabilized)
                            with torch.cuda.amp.autocast(False):
                                feats32 = feats_f.to(dtype=torch.float32)
                                centers32 = centers.to(device=feats32.device, dtype=torch.float32)
                                P32 = args.precision.to(device=feats32.device, dtype=torch.float32)
                                # symmetrize and regularize precision
                                P32 = 0.5 * (P32 + P32.t())
                                eps = 1e-3
                                I = torch.eye(P32.shape[0], device=P32.device, dtype=P32.dtype)
                                P32_reg = P32 + eps * I
                                try:
                                    L = torch.linalg.cholesky(P32_reg)
                                except Exception:
                                    # escalate regularization then fall back to eigen if needed
                                    try:
                                        L = torch.linalg.cholesky(P32 + 1e-1 * I)
                                    except Exception:
                                        evals, evecs = torch.linalg.eigh(P32_reg)
                                        evals = torch.clamp(evals, min=1e-6)
                                        L = (evecs @ torch.diag(torch.sqrt(evals)) @ evecs.t())
                                try:
                                    p_shape = f"{P32.shape[0]}x{P32.shape[1]}"
                                except Exception:
                                    p_shape = None
                                # whitened diff: z = (x-μ) @ L^T (since P = L L^T)
                                diff32 = feats32[:, None, :] - centers32[None, :, :]
                                z = torch.einsum('bnd,fd->bnf', diff32, L.t())
                                md2 = (z * z).sum(dim=-1)
                                md2 = torch.nan_to_num(md2, posinf=1e6, neginf=1e6).clamp_max(1e6)
                                dmin = torch.min(md2, dim=1)[0].mean()
                                # normalize by feature dim to keep scale comparable
                                dmin = dmin / float(centers32.shape[1])
                                try:
                                    dmin_norm_value = float(dmin.detach().item())
                                except Exception:
                                    dmin_norm_value = None
                            try:
                                dmin_value = float(dmin.detach().item())
                            except Exception:
                                dmin_value = None
                            # hinge: L_forget = lambda_f * ReLU(margin - dmin_norm)
                            margin = float(getattr(args, 'forget_margin', 3.0))
                            forget_hinge = torch.relu(torch.as_tensor(margin, device=dmin.device, dtype=dmin.dtype) - dmin)
                            forget_term = args.forget_lambda * forget_hinge
                            loss = loss + forget_term
                    # always report forget term (0 if none)
                    try:
                        l_dict['forget'] = float(forget_term.detach().item() if hasattr(forget_term, 'detach') else forget_term)
                    except Exception:
                        l_dict['forget'] = 0.0
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            old_scale = scaler.get_scale()
            scaler.update()    
            new_scale = scaler.get_scale()   

        losses.update(loss.item(), bsz)
        
        if new_scale >= old_scale:
            adjust_learning_rate(args, optimizer, train_loader, step)
            
        if step%len(train_loader) == 0:
            for k in l_dict.keys():
                sub_loss[k] = []
                
        for k in l_dict.keys():
            sub_loss[k].append(l_dict[k])

        # optional console print for per-part losses
        if getattr(args, 'print_every', 0) and (step % int(args.print_every) == 0):
            it = step % len(train_loader)
            mle_v = l_dict.get('mle', 0.0)
            pcon_v = l_dict.get('proto_contra', 0.0)
            f_v = l_dict.get('forget', 0.0)
            extra = f" nr={nr_out} nf={nf_out} centers={centers_shape} P={p_shape} dmin_norm={dmin_norm_value if dmin_norm_value is not None else 'NA'}"
            print(f"[loss] ep {epoch} it {it} total={loss.item():.4f} mle={float(mle_v):.4f} pcon={float(pcon_v):.4f} forget={float(f_v):.4f}{extra}")
            
    for k in sub_loss.keys():
        sub_loss[k] = np.mean(sub_loss[k])
    

    return losses.avg, sub_loss


def train_supervised(args, train_loader, model, criterion, optimizer, epoch, warmup_schedular=None, schedular=None, scaler=None, index=None, index_map=None, k=1):
    model.train()

    losses = AverageMeter()
    sub_loss = {}

    for step, (images, labels) in enumerate(train_loader, start=epoch * len(train_loader)):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # warm-up learning rate
        # warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        bsz = labels.shape[0]
        # compute loss
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                features = model(images)
                loss = criterion(features, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()    
        else:
            features = model(images)
            loss = criterion(features, labels)
            # SGD
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), bsz)

        # adjust_learning_rate(args, optimizer, train_loader, step, warmup_schedular=warmup_schedular, schedular=schedular)
            
        if step%len(train_loader) == 0:
            sub_loss['train'] = []
                
        for k in sub_loss.keys():
            sub_loss[k].append(loss.item())
            
    for k in sub_loss.keys():
        sub_loss[k] = np.mean(sub_loss[k])
    

    return losses.avg, sub_loss

def get_trainer(args):
    arch = args.method
    
    if "palm" in arch:
            trainer = train_palm
    else:
        trainer = train_supervised
    return trainer 