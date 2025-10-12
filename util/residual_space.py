import os
import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance


def _to_tensor(x: np.ndarray, device: torch.device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class ResidualProjector:
    """
    Build and apply residual-space projection bases.

    Modes:
      - 'dcc': eigenvectors of class-centered residual covariance with smallest eigenvalues.
      - 'wdisc': orthogonal complement of discriminant span (requires discriminants input).
    """

    def __init__(self, device: torch.device = torch.device('cuda')) -> None:
        self.device = device
        self.basis = None  # (R, D)
        self.mode = None
        self.cov = None

    @staticmethod
    def l2_normalize(feats: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12
        return feats / norms

    def fit_dcc(self, feats: np.ndarray, labels: np.ndarray, residual_dim: int = None,
                normalize: bool = True, arch_hint: str = None, id_dset: str = None):
        """
        feats: (N, D) numpy array
        labels: (N,) numpy array [0..C-1]
        residual_dim: if None, choose by arch/dataset hint
        """
        if normalize:
            feats = self.l2_normalize(feats)
        N, D = feats.shape
        C = int(labels.max()) + 1
        # class means
        cls_means = np.zeros((C, D), dtype=np.float32)
        for c in range(C):
            idx = (labels == c)
            cls_means[c] = feats[idx].mean(axis=0)
        center_samples = cls_means[labels]
        r_feat = feats - center_samples

        # empirical covariance on residuals
        estimator = EmpiricalCovariance(assume_centered=True)
        estimator.fit(r_feat)
        cov = estimator.covariance_.astype(np.float32)
        eigvals, eigvecs = np.linalg.eig(cov)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        sort_idx = np.argsort(eigvals)  # ascending: smallest first

        if residual_dim is None:
            # simple heuristics matching ooddcc dynamic
            if arch_hint is not None and arch_hint.lower() == 'densenet':
                residual_dim = 300
            elif arch_hint is not None and arch_hint.lower() in ['wrn', 'wide_resnet', 'wresnet']:
                residual_dim = 50
            elif id_dset is not None and id_dset.lower() in ['imagenet', 'imagenet1k']:
                residual_dim = 512
            else:
                residual_dim = min(128, D)

        pick = sort_idx[:residual_dim]
        basis = eigvecs[:, pick].T  # (R, D)

        self.basis = _to_tensor(basis, device=self.device)
        self.cov = _to_tensor(cov, device=self.device)
        self.mode = 'dcc'
        return self.basis

    def fit_wdisc_from_discriminants(self, discriminants: torch.Tensor, D: int):
        """
        discriminants: (K, D) torch tensor
        Build H = I - V^T V where V are right-singular vectors spanning discriminant row-space.
        Return an orthogonal projector matrix H (D, D) and optionally an orthonormal basis by eigendecomp.
        """
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(discriminants)
            Vh = torch.real(Vh)
            proj_res = torch.eye(D, device=discriminants.device) - Vh.T @ Vh
        self.mode = 'wdisc'
        self.basis = None
        return proj_res

    def project(self, feats: torch.Tensor) -> torch.Tensor:
        """Project to residual space.
        If basis is (R, D), returns (N, R) = feats @ basis^T.
        If wdisc mode with full projector is provided externally, apply that separately.
        """
        assert self.basis is not None, "Residual basis not fitted."
        return feats @ self.basis.T


@torch.no_grad()
def compute_dcc_basis_from_loader(args, model, loader_eval, device=None):
    """Compute DCC residual basis using model features and labels from a data loader.

    Returns:
        basis (torch.Tensor): (R, D)
        projector (torch.Tensor): (D, D) symmetric idempotent
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model_device = next(model.parameters()).device
    # collect features and labels
    feats_list, labels_list = [], []
    for images, labels in loader_eval:
        if isinstance(images, (list, tuple)) and len(images) == 2:
            images = images[0]
        images = images.to(model_device, non_blocking=True)
        labels = labels.to(model_device, non_blocking=True)
        # choose feature space
        if getattr(args, 'residual_at', 'encoder') == 'encoder':
            # encoder output before projection head
            if hasattr(model, 'encoder'):
                feats_enc = model.encoder(images)
                feats = feats_enc.detach()
            else:
                feats = model(images)
        else:
            feats = model(images)  # embedding
        feats_list.append(feats.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
    feats = np.concatenate(feats_list, axis=0).astype(np.float32)
    labels = np.concatenate(labels_list, axis=0).astype(np.int64)

    rp = ResidualProjector(device=device)
    basis = rp.fit_dcc(
        feats, labels,
        residual_dim=getattr(args, 'residual_dim', None),
        normalize=(getattr(args, 'residual_norm', 'l2') == 'l2'),
        arch_hint=getattr(args, 'backbone', None),
        id_dset=getattr(args, 'in_dataset', None),
    )
    # build projector P = B^T B
    projector = basis.T @ basis
    return basis, projector


