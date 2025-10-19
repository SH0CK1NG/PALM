import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

from util.train_utils import adjust_learning_rate, AverageMeter


def train_palm(args, train_loader, model, criterion, optimizer, epoch, scaler=None):
    model.train()

    losses = AverageMeter()
    sub_loss = {}

    for step, (images, labels) in enumerate((train_loader), start=epoch * len(train_loader)):
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
        proto_count = None
        fproto_sim = None
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
                # Prototype-based forgetting loss on 128-D hypersphere (push away from retained-class prototypes)
                if getattr(args, 'forget_lambda', 0) > 0 and forget_mask.any():
                    # Determine class count and cache size from criterion
                    C = int(getattr(criterion, 'num_classes', 0) or labels.max().item() + 1)
                    R = int(getattr(criterion, 'cache_size', getattr(args, 'cache_size', 1)))
                    # Build retained class list
                    if len(forget_classes) > 0:
                        retain_classes = torch.tensor([c for c in range(C) if c not in forget_classes], device=labels.device, dtype=torch.long)
                    else:
                        retain_classes = torch.arange(C, device=labels.device, dtype=torch.long)
                    if retain_classes.numel() > 0:
                        # Map (replicas x classes) into flat prototype indices: idx = r*C + c
                        r_offsets = (torch.arange(R, device=labels.device, dtype=torch.long).view(-1, 1) * C)
                        proto_idx = (r_offsets + retain_classes.view(1, -1)).reshape(-1)
                        P = getattr(criterion, 'protos', None)
                        if P is not None:
                            P = F.normalize(P.detach(), dim=1)
                            P = P.index_select(0, proto_idx)
                            try:
                                proto_count = int(P.shape[0])
                            except Exception:
                                proto_count = None
                            # aggregate multi-views then renormalize to unit sphere
                            feats_mean = F.normalize(features.mean(dim=1), dim=-1)
                            feats_f = feats_mean[forget_mask]
                            if feats_f.numel() > 0:
                                # cosine similarities to retained prototypes
                                sim = torch.matmul(feats_f, P.t())
                                temp_f = float(getattr(args, 'temp', 0.1))
                                forget_term = args.forget_lambda * torch.logsumexp(sim / temp_f, dim=1).mean()
                                loss = loss + forget_term
                                try:
                                    l_dict['forget'] = float(forget_term.detach().item())
                                except Exception:
                                    l_dict['forget'] = 0.0
                            else:
                                l_dict['forget'] = 0.0
                        else:
                            l_dict['forget'] = 0.0
                    else:
                        l_dict['forget'] = 0.0

                # Optional: batch-level forget prototype (non-persistent)
                if bool(getattr(args, 'forget_proto_enable', False)) and forget_mask.any():
                    # compute batch forget prototype on sphere
                    feats_mean = F.normalize(features.mean(dim=1), dim=-1)
                    f_feats = feats_mean[forget_mask]
                    if f_feats.numel() > 0:
                        f_proto = F.normalize(f_feats.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
                        # attraction: bring forget samples closer to f_proto
                        # use negative cosine (1 - cos) minimized => maximize cos
                        cos_to_f = torch.matmul(f_feats, f_proto.t()).squeeze(1)  # [N_f]
                        f_attr_w = float(getattr(args, 'forget_attr_w', 1.0))
                        f_attr = args.forget_lambda * f_attr_w * (-cos_to_f.mean())
                        loss = loss + f_attr
                        # repulsion: push f_proto away from retained prototypes
                        C = int(getattr(criterion, 'num_classes', 0) or labels.max().item() + 1)
                        R = int(getattr(criterion, 'cache_size', getattr(args, 'cache_size', 1)))
                        if C > 0 and R > 0 and hasattr(criterion, 'protos') and criterion.protos is not None:
                            retain_classes = torch.tensor([c for c in range(C) if c not in set(map(int, fcls.tolist()))] if len(forget_classes)>0 else list(range(C)), device=labels.device, dtype=torch.long)
                            if retain_classes.numel() > 0:
                                r_offsets = (torch.arange(R, device=labels.device, dtype=torch.long).view(-1, 1) * C)
                                proto_idx = (r_offsets + retain_classes.view(1, -1)).reshape(-1)
                                P = F.normalize(criterion.protos.detach(), dim=1).index_select(0, proto_idx)
                                # want f_proto far from retain protos -> minimize logsumexp(cos/ T) with negative sign (i.e., maximize negative cos)
                                temp_f = float(getattr(args, 'temp', 0.1))
                                sim_fp = torch.matmul(f_proto, P.t()).squeeze(0)  # [K_r]
                                f_rep_w = float(getattr(args, 'forget_proto_rep_w', 1.0))
                                # repulsion term: +lambda * logsumexp(sim/T) to push sims down; but here we treat f_proto as variable-less, so we use negative to reduce cos
                                # use simple mean(sim) as surrogate since f_proto not a parameter; we backprop through features via f_proto dependency
                                f_rep = args.forget_lambda * f_rep_w * torch.logsumexp(sim_fp / temp_f, dim=0)
                                loss = loss + f_rep
                                try:
                                    fproto_sim = float(sim_fp.mean().detach().item())
                                except Exception:
                                    fproto_sim = None
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
            extra = f" nr={nr_out} nf={nf_out} protos={proto_count} fproto_sim={fproto_sim if fproto_sim is not None else 'NA'}"
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