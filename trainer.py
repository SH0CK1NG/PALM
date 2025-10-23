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

        # parse forget classes sets (CPU tensors here)
        # - forget_classes: full set to forget across all stages
        # - forget_classes_inc: stage-local forget set
        # - forget_classes_seen: already forgotten in previous stages (should be excluded from retain set)
        forget_all: set = set()
        forget_inc: set = set()
        forget_seen: set = set()
        if getattr(args, 'forget_classes', None):
            forget_all |= set(int(x) for x in str(args.forget_classes).split(',') if x!='')
        elif getattr(args, 'forget_list_path', None) and os.path.exists(args.forget_list_path):
            try:
                import json
                with open(args.forget_list_path) as f:
                    data = f.read().strip()
                    try:
                        arr = json.loads(data)
                    except Exception:
                        arr = [int(line) for line in data.splitlines() if line.strip()!='']
                    forget_all |= set(int(x) for x in arr)
            except Exception:
                pass
        if getattr(args, 'forget_classes_inc', None):
            forget_inc |= set(int(x) for x in str(args.forget_classes_inc).split(',') if x!='')
        if getattr(args, 'forget_classes_seen', None):
            forget_seen |= set(int(x) for x in str(args.forget_classes_seen).split(',') if x!='')

        # Build masks
        if len(forget_all) > 0 or len(forget_inc) > 0 or len(forget_seen) > 0:
            f_all = torch.tensor(sorted(list(forget_all)), dtype=torch.long) if len(forget_all)>0 else None
            f_inc = torch.tensor(sorted(list(forget_inc)), dtype=torch.long) if len(forget_inc)>0 else None
            f_seen = torch.tensor(sorted(list(forget_seen)), dtype=torch.long) if len(forget_seen)>0 else None
            # Current stage forget mask matches forget_inc; if empty, fall back to intersection with forget_all
            if f_inc is not None and f_inc.numel()>0:
                forget_mask_cpu = (labels.view(-1,1) == f_inc.view(1,-1)).any(dim=1)
            elif f_all is not None and f_all.numel()>0:
                forget_mask_cpu = (labels.view(-1,1) == f_all.view(1,-1)).any(dim=1)
            else:
                forget_mask_cpu = torch.zeros(labels.shape[0], dtype=torch.bool)
            # Retain excludes all seen (past) and current incremental forget classes
            # 保留集需要包含未来阶段要遗忘的类，因此不应排除 forget_all 中尚未进入的类别
            excl = set()
            excl |= forget_seen
            excl |= forget_inc if len(forget_inc)>0 else set()
            if len(excl) > 0:
                excl_t = torch.tensor(sorted(list(excl)), dtype=torch.long)
                exclude_mask = (labels.view(-1,1) == excl_t.view(1,-1)).any(dim=1)
                retain_mask_cpu = ~exclude_mask
            else:
                retain_mask_cpu = torch.ones(labels.shape[0], dtype=torch.bool)
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

                # compute retain/forget masks on selected labels (GPU tensors)
                if len(forget_all) > 0 or len(forget_inc) > 0 or len(forget_seen) > 0:
                    f_inc_gpu = torch.tensor(sorted(list(forget_inc)), device=labels.device) if len(forget_inc)>0 else None
                    f_all_gpu = torch.tensor(sorted(list(forget_all)), device=labels.device) if len(forget_all)>0 else None
                    f_seen_gpu = torch.tensor(sorted(list(forget_seen)), device=labels.device) if len(forget_seen)>0 else None
                    if f_inc_gpu is not None and f_inc_gpu.numel()>0:
                        forget_mask = (labels.unsqueeze(1) == f_inc_gpu.unsqueeze(0)).any(dim=1)
                    elif f_all_gpu is not None and f_all_gpu.numel()>0:
                        forget_mask = (labels.unsqueeze(1) == f_all_gpu.unsqueeze(0)).any(dim=1)
                    else:
                        forget_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
                    excl = set()
                    excl |= forget_seen
                    if len(forget_inc)>0:
                        excl |= forget_inc
                    if len(excl) > 0:
                        excl_gpu = torch.tensor(sorted(list(excl)), device=labels.device)
                        exclude_mask = (labels.unsqueeze(1) == excl_gpu.unsqueeze(0)).any(dim=1)
                        retain_mask = ~exclude_mask
                    else:
                        retain_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
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
                    # Build retained class list: exclude seen + current incremental forget; include future-to-forget classes
                    excl2 = set()
                    excl2 |= forget_seen
                    if len(forget_inc) > 0:
                        excl2 |= forget_inc
                    retain_classes = torch.tensor([c for c in range(C) if c not in excl2], device=labels.device, dtype=torch.long)
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
                                # Optional: push forget samples away from the average prototype of retained classes
                                if bool(getattr(args, 'forget_avgproto_enable', False)):
                                    # compute average of retained-class prototypes (across replicas and classes)
                                    P_avg = P.mean(dim=0, keepdim=True)  # [1, D]
                                    P_avg = F.normalize(P_avg, dim=1)
                                    # cosine sim between forget features and average proto
                                    sim_avg = torch.matmul(feats_f, P_avg.t()).squeeze(1)  # [N_f]
                                    w_avg = float(getattr(args, 'forget_avgproto_w', 1.0))
                                    # use sample-wise mean instead of logsumexp over samples
                                    avg_rep_term = args.forget_lambda * w_avg * (sim_avg / temp_f).mean()
                                    loss = loss + avg_rep_term
                                    try:
                                        l_dict['forget_avg'] = float(avg_rep_term.detach().item())
                                    except Exception:
                                        l_dict['forget_avg'] = 0.0
                            else:
                                l_dict['forget'] = 0.0
                                # keep key for consistent logging even when no forget samples in batch
                                l_dict['forget_avg'] = 0.0
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
                            excl2 = set()
                            excl2 |= forget_seen
                            if len(forget_inc) > 0:
                                excl2 |= forget_inc
                            retain_classes = torch.tensor([c for c in range(C) if c not in excl2], device=labels.device, dtype=torch.long)
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
            favg_v = l_dict.get('forget_avg', 0.0)
            extra = f" nr={nr_out} nf={nf_out} protos={proto_count} fproto_sim={fproto_sim if fproto_sim is not None else 'NA'}"
            print(f"[loss] ep {epoch} it {it} total={loss.item():.4f} mle={float(mle_v):.4f} pcon={float(pcon_v):.4f} forget={float(f_v):.4f} favg={float(favg_v):.4f}{extra}")
            
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