import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompLoss(nn.Module):
    """
    Compactness Loss with class-conditional prototypes (CIDER component)
    """
    def __init__(self, num_classes: int, temperature: float = 0.1, base_temperature: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [N, D] (assumed L2-normalized)
        # prototypes: [C, D] (assumed L2-normalized)
        # labels: [N]
        device = features.device
        C = int(prototypes.shape[0])
        assert C == self.num_classes, "prototype class count mismatch"

        # mask[i, c] = 1 if sample i belongs to class c
        labels = labels.contiguous().view(-1, 1)
        proxy_labels = torch.arange(0, C, device=device).view(1, -1)
        mask = torch.eq(labels, proxy_labels).float()

        # logits = <x, p_c>/T
        prototypes = F.normalize(prototypes, dim=1)
        logits = torch.matmul(features, prototypes.T) / self.temperature

        # numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # log_prob over classes
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # positive class mean log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss


class DisLoss(nn.Module):
    """
    Dispersion Loss with EMA prototypes (CIDER component)
    Maintains class prototypes updated per batch with EMA, and pushes prototypes apart.
    """
    def __init__(self, num_classes: int, feat_dim: int, temperature: float = 0.1, base_temperature: float = 0.1,
                 proto_m: float = 0.5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.proto_m = float(proto_m)
        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.feat_dim))

    @torch.no_grad()
    def init_class_prototypes(self, model: nn.Module, loader) -> None:
        model.eval()
        start = time.time()
        proto = torch.zeros(self.num_classes, self.feat_dim, device=self.prototypes.device)
        counts = torch.zeros(self.num_classes, device=self.prototypes.device)
        for images, targets in loader:
            if isinstance(images, (list, tuple)) and len(images) == 2:
                images = images[0]
            images = images.to(self.prototypes.device, non_blocking=True)
            targets = targets.to(self.prototypes.device, non_blocking=True)
            with torch.no_grad():
                feats = model(images)  # assume L2-normalized
            for j, f in enumerate(feats):
                c = int(targets[j].item())
                proto[c] += f
                counts[c] += 1
        # avoid div-by-zero
        counts = torch.clamp(counts, min=1.0)
        proto = proto / counts.view(-1, 1)
        self.prototypes = F.normalize(proto, dim=1)
        dur = time.time() - start
        print(f"[cider] init prototypes in {dur:.3f}s")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        proto = self.prototypes.to(device)
        # EMA update per sample
        for j in range(features.shape[0]):
            c = int(labels[j].item())
            updated = F.normalize(proto[c] * self.proto_m + features[j] * (1.0 - self.proto_m), dim=0)
            proto[c] = updated
        self.prototypes = proto.detach()

        # push prototypes apart (exclude self-similarity)
        logits = torch.matmul(proto, proto.T) / self.temperature
        # mask out diagonal
        mask = torch.ones_like(logits, device=device) - torch.eye(self.num_classes, device=device)
        # mean over non-diagonal entries via logsumexp over each row
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = (self.temperature / self.base_temperature) * mean_prob_neg.mean()
        return loss


class CIDERCriterion(nn.Module):
    """
    Combined CIDER loss: L = w * CompLoss + DisLoss
    Exposes `protos` for external regularizers (e.g., forgetting terms).
    """
    def __init__(self, num_classes: int, feat_dim: int, temperature: float = 0.1, proto_m: float = 0.5, w: float = 1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.comp = CompLoss(num_classes=num_classes, temperature=temperature, base_temperature=temperature)
        self.dis = DisLoss(num_classes=num_classes, feat_dim=feat_dim, temperature=temperature,
                           base_temperature=temperature, proto_m=proto_m)
        self.w = float(w)

    @property
    def protos(self) -> torch.Tensor:
        return self.dis.prototypes

    @torch.no_grad()
    def init_prototypes(self, model: nn.Module, loader) -> None:
        self.dis.init_class_prototypes(model, loader)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        # features expected l2-normalized
        loss_dis = self.dis(features, labels)
        loss_comp = self.comp(features, self.dis.prototypes, labels)
        total = self.w * loss_comp + loss_dis
        return total, {"comp": float(loss_comp.detach().item()), "dis": float(loss_dis.detach().item())}


