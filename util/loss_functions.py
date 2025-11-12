import torch.nn as nn
import torch
import torch.nn.functional as F
   

class PALM(nn.Module):
    def __init__(self, args, num_classes=10, n_protos=1000, proto_m=0.99, temp=0.1, lambda_pcon=1, k=0,feat_dim=128, epsilon=0.05):
        super(PALM, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  # temperature scaling
        self.nviews = args.nviews
        self.cache_size = args.cache_size
        
        self.lambda_pcon = lambda_pcon
        
        self.feat_dim = feat_dim
        
        self.epsilon = epsilon
        self.sinkhorn_iterations = 3
        self.k = min(k, self.cache_size)
        
        self.n_protos = n_protos
        self.proto_m = proto_m
        self.register_buffer("protos", torch.rand(self.n_protos,feat_dim))
        self.protos = F.normalize(self.protos, dim=-1)
        # global enable switch for PALM loss
        self.palm_enable = bool(getattr(args, 'palm_enable', True))
        # keep args for incremental split logic
        self.args = args
        
    def sinkhorn(self, features):
        out = torch.matmul(features, self.protos.detach().T)

        # 数值稳定：逐样本减最大值，再指数
        logits = (out.detach() / self.epsilon)
        logits = logits - logits.max(dim=1, keepdim=True).values
        Q = torch.exp(logits).t()  # K-by-B
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1 with eps 防止除零
        sum_Q = torch.sum(Q)
        if not torch.isfinite(sum_Q):
            self.protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, self.protos.detach().T)
            logits = (out.detach() / self.epsilon)
            logits = logits - logits.max(dim=1, keepdim=True).values
            Q = torch.exp(logits).t()
            sum_Q = torch.sum(Q)
        Q = Q / (sum_Q + 1e-6)

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q = F.normalize(Q, dim=1, p=1)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q = F.normalize(Q, dim=0, p=1)
            Q /= B

        Q *= B
        return Q.t()
        
    def mle_loss(self, features, targets):
        # update prototypes by EMA
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_labels = targets.contiguous().repeat(self.nviews).view(-1, 1)
        contrast_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda()
                
        Q = self.sinkhorn(features)
        # topk
        if self.k > 0:
            update_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            update_mask = F.normalize(F.normalize(topk_mask*update_mask, dim=1, p=1),dim=0, p=1)
        # original
        else:
            update_mask = F.normalize(F.normalize(mask * Q, dim=1, p=1),dim=0, p=1)
        update_features = torch.matmul(update_mask.T, features)
        
        protos = self.protos
        protos = self.proto_m * protos + (1-self.proto_m) * update_features

        self.protos = F.normalize(protos, dim=1, p=2)
        
        Q = self.sinkhorn(features)
        
        proto_dis = torch.matmul(features, self.protos.detach().T)
        anchor_dot_contrast = torch.div(proto_dis, self.temp)
        logits = anchor_dot_contrast
       
        if self.k > 0:
            loss_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            loss_mask = F.normalize(topk_mask*loss_mask, dim=1, p=1)
            masked_logits = loss_mask * logits 
        else:  
            masked_logits = F.normalize(Q*mask, dim=1, p=1) * logits
    
        pos=torch.sum(masked_logits, dim=1)
        neg=torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
        log_prob=pos-neg

        # Optional: restrict MLE reduction to a subset (e.g., old only) while keeping prototype update on all
        subset = getattr(self.args, 'mle_subset', None)
        if subset in ('old', 'retain'):
            # build old/new from args (same as proto_contra)
            try:
                def parse_csv(s):
                    return [int(x) for x in str(s).split(',') if x!='']
                C = int(self.num_classes)
                inc_csv = getattr(self.args, 'forget_classes_inc', None)
                seen_csv = getattr(self.args, 'forget_classes_seen', None)
                new_set = set(parse_csv(inc_csv)) if inc_csv else set()
                old_set = set(parse_csv(seen_csv)) if seen_csv else set(range(C))
            except Exception:
                C = int(self.num_classes)
                old_set = set(range(C))
            anchor_labels = targets.contiguous().repeat(self.nviews).view(-1)
            dev = anchor_labels.device
            if len(old_set) > 0:
                old_t = torch.tensor(sorted(list(old_set)), device=dev, dtype=anchor_labels.dtype)
                sel_mask = (anchor_labels.unsqueeze(1) == old_t.unsqueeze(0)).any(dim=1)
            else:
                sel_mask = torch.zeros_like(anchor_labels, dtype=torch.bool)
            if torch.any(sel_mask):
                loss = -torch.mean(log_prob[sel_mask])
            else:
                loss = logits.sum() * 0.0
            return loss

        loss = -torch.mean(log_prob)
        return loss   
    
    def proto_contra(self):
        
        protos = F.normalize(self.protos, dim=1)
        batch_size = self.num_classes
        device = protos.device
        
        proto_labels = torch.arange(self.num_classes, device=device).repeat(self.cache_size).view(-1,1)
        mask = torch.eq(proto_labels, proto_labels.T).float()    

        contrast_count = self.cache_size
        contrast_feature = protos

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            0.5)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # Decide proto_contra mode based on pcon_inc (split | new_only | off)
        try:
            inc_flag = bool(getattr(self.args, 'incremental', False))
            fl = float(getattr(self.args, 'forget_lambda', 0.0))
            strat = str(getattr(self.args, 'forget_strategy', 'proto'))
            forget_enabled = (fl > 0.0) or (strat in ('randlabel', 'ga'))
            pcon_inc_arg = getattr(self.args, 'pcon_inc', None)
            # Normalize to string mode
            mode = None
            if pcon_inc_arg is None:
                # defaults: incremental -> split; forgetting -> off; else off
                mode = 'split' if inc_flag and (not forget_enabled) else 'off'
            else:
                if isinstance(pcon_inc_arg, bool):
                    mode = 'split' if pcon_inc_arg else 'off'
                else:
                    mode = str(pcon_inc_arg).strip().lower()
            # guard invalid
            if mode not in ('split','new_only','off'):
                mode = 'off'
        except Exception:
            mode = 'off'

        if mode == 'off':
            pos = torch.sum(F.normalize(mask, dim=1, p=1) * logits, dim=1)
            neg = torch.log(torch.sum(logits_mask * torch.exp(logits), dim=1))
            log_prob = pos - neg
            loss = - torch.mean(log_prob)
            return loss

        # Build old/new sets from args in incremental mode
        args = getattr(self, 'args', None)
        def parse_csv(s):
            return [int(x) for x in str(s).split(',') if x!='']
        C = int(self.num_classes)
        fc = getattr(args, 'forget_classes', None)
        inc_csv = getattr(args, 'forget_classes_inc', None)
        seen_csv = getattr(args, 'forget_classes_seen', None)
        new_set = set(parse_csv(inc_csv)) if inc_csv else set()
        old_set = set(parse_csv(seen_csv)) if seen_csv else set()
        # fallback: only forget_classes provided => new=forget_classes, old=all\new
        if (len(new_set) == 0 and len(old_set) == 0) and fc is not None:
            fc_set = set(parse_csv(fc))
            new_set = set([c for c in fc_set if 0 <= c < C])
            old_set = set([c for c in range(C) if c not in new_set])
        # default fallback when nothing provided
        if (len(new_set) == 0 and len(old_set) == 0):
            old_set = set(range(C))
            new_set = set()

        # Split anchors by old/new classes according to class ids
        class_ids = proto_labels.view(-1)
        dev = class_ids.device
        is_old_proto = torch.zeros_like(class_ids, dtype=torch.float32, device=dev)
        is_new_proto = torch.zeros_like(class_ids, dtype=torch.float32, device=dev)
        if len(old_set) > 0:
            old_t = torch.tensor(sorted(list(old_set)), device=dev, dtype=class_ids.dtype)
            is_old_proto = (class_ids.unsqueeze(1) == old_t.unsqueeze(0)).any(dim=1).float()
        if len(new_set) > 0:
            new_t = torch.tensor(sorted(list(new_set)), device=dev, dtype=class_ids.dtype)
            is_new_proto = (class_ids.unsqueeze(1) == new_t.unsqueeze(0)).any(dim=1).float()

        # Column selection masks
        col_old = is_old_proto.view(1, -1)  # [1, N]
        den_mask_old_cols = logits_mask * col_old  # restrict denominator to old columns
        den_mask_all = logits_mask               # new denom: all (old+new)

        # Row (anchor) masks
        row_old = is_old_proto.view(-1, 1)  # [N, 1]
        row_new = is_new_proto.view(-1, 1)

        # Positive masks (same-class, exclude self), restricted to anchor subsets via row mask
        pos_mask_old = mask * row_old
        pos_mask_new = mask * row_new

        # Compute old anchors loss: denom only old
        pos_old = torch.sum(F.normalize(pos_mask_old, dim=1, p=1) * logits, dim=1)
        neg_old = torch.log(torch.sum(den_mask_old_cols * torch.exp(logits), dim=1) + 1e-12)
        log_prob_old = pos_old - neg_old
        sel_old = is_old_proto.bool()
        if torch.any(sel_old):
            loss_old = - torch.mean(log_prob_old[sel_old])
        else:
            loss_old = logits.sum() * 0.0

        # Compute new anchors loss: denom includes old+new; numerator only same-class (new)
        pos_new = torch.sum(F.normalize(pos_mask_new, dim=1, p=1) * logits, dim=1)
        neg_new = torch.log(torch.sum(den_mask_all * torch.exp(logits), dim=1) + 1e-12)
        log_prob_new = pos_new - neg_new
        sel_new = is_new_proto.bool()
        if torch.any(sel_new):
            loss_new = - torch.mean(log_prob_new[sel_new])
        else:
            loss_new = logits.sum() * 0.0
        if mode == 'new_only':
            # 仅使用新锚点；若无新锚点，退回标准 PALM 形态
            if torch.any(sel_new):
                return loss_new
            pos = torch.sum(F.normalize(mask, dim=1, p=1) * logits, dim=1)
            neg = torch.log(torch.sum(logits_mask * torch.exp(logits), dim=1))
            return - torch.mean(pos - neg)
        # split: 使用 old+new
        loss = loss_old + loss_new
        return loss
    
           
    def forward(self, features, targets):
        loss = 0
        loss_dict = {}

        if not self.palm_enable:
            # PALM disabled: return zero loss and skip prototype updates
            self.protos = self.protos.detach()
            loss = (features.sum() * 0.0)
            loss_dict['mle'] = 0.0
            loss_dict['proto_contra'] = 0.0
            return loss, loss_dict

        g_con = self.mle_loss(features, targets)
        loss += g_con
        loss_dict['mle'] = g_con.cpu().item()
                    
        if self.lambda_pcon > 0:            
            g_dis = self.lambda_pcon * self.proto_contra()
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()
                                
        self.protos = self.protos.detach()
                
        return loss, loss_dict
    
