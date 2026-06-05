"""
SIGReg — Sketched Isotropic Gaussian Regularisation
=====================================================
Covariance-penalty loss that pushes encoder embeddings toward an isotropic
distribution (all dimensions carry comparable variance).

AIA paper §SIGReg theory (Sajeev 2026):

    L_SIGReg = L_inv + λ_cov · L_cov^VIC

where L_cov^VIC is the VICReg-style off-diagonal covariance penalty:

    L_cov = (1/D) · Σ_{i≠j} [Cov(Z)]_{ij}²

WHY THIS MATTERS
----------------
The L1 divergence score Δ = ||softmax(z_A) - softmax(z_B)||₁ / √G routes
every inference decision.  When the encoder's latent space is anisotropic
(a few dimensions carry most variance) the softmax probability vectors are
dominated by those dimensions — the route signal is cheap to fool and easy
to collapse. SIGReg forces all D dimensions to carry comparable variance,
making Δ a reliable signal across the full embedding space.

AIA Experiment 3 target: ρ = AUROC_SIGReg / AUROC_baseline > 1.15.

STATUS
------
Ready for Experiment 3 (GPU encoder fine-tuning). Wired into the overnight
LoRA training cycle with lambda_iso=0.0 by default — completely inert until
an experiment sets lambda_iso ∈ {0.01, 0.1, 1.0} from the AIA sweep table.

USAGE
-----
    from dmn.sigreg import SIGRegLoss

    sigreg = SIGRegLoss(lambda_iso=0.1)

    # Inside a training step:
    z_adapted = encoder(x) + scale * lora_delta(x)   # (N, D)
    loss = task_loss + sigreg(z_adapted)

    # Diagnostic (not differentiable):
    ratio = sigreg.isotropy_ratio(z_adapted)  # 1.0 = perfect isotropy
"""

import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """
    Off-diagonal covariance penalty for isotropic embedding training.

    Plugs into any LoRA or encoder training loop as an additive regulariser.
    Default lambda_iso=0.0 makes it a no-op — safe to wire in production
    before experiments calibrate the optimal weight.

    Args:
        lambda_iso: Regularisation weight λ_cov.
                    Default 0.0 (disabled).
                    AIA Exp 3 sweep values: {0.01, 0.1, 1.0}.
    """

    def __init__(self, lambda_iso: float = 0.0):
        super().__init__()
        self.lambda_iso = lambda_iso

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the off-diagonal covariance penalty.

        Args:
            z: (N, D) embedding matrix — batch of encoder outputs.
        Returns:
            Scalar penalty tensor. Zero when lambda_iso == 0.0 or N < 2.
        """
        if self.lambda_iso == 0.0 or z.shape[0] < 2:
            return torch.zeros(1, device=z.device, dtype=z.dtype).squeeze()

        N, D = z.shape
        # Centre the batch
        z_c = z - z.mean(dim=0, keepdim=True)
        # Covariance matrix (D × D), unbiased estimator
        cov = (z_c.T @ z_c) / (N - 1)
        # Off-diagonal squared sum, normalised by D (VICReg formula)
        off_diag = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
        return self.lambda_iso * off_diag

    def isotropy_ratio(self, z: torch.Tensor) -> float:
        """
        Diagnostic: min/max eigenvalue ratio of the covariance matrix.

        1.0 = perfectly isotropic (uniform variance across all dimensions).
        ~0.0 = collapsed / dominated by a few dimensions.

        Used to compute ρ = AUROC_SIGReg / AUROC_baseline in AIA Experiment 3.
        Not differentiable — call outside the training loop for monitoring only.

        Args:
            z: (N, D) embedding matrix.
        Returns:
            Float ratio, or nan if N < 2.
        """
        with torch.no_grad():
            if z.shape[0] < 2:
                return float("nan")
            N, _ = z.shape
            z_c = z - z.mean(dim=0, keepdim=True)
            cov = (z_c.T @ z_c) / (N - 1)
            eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-8)
            return float(eigvals.min() / eigvals.max())

    def __repr__(self) -> str:
        status = "active" if self.lambda_iso > 0.0 else "disabled (lambda_iso=0.0)"
        return f"SIGRegLoss(lambda_iso={self.lambda_iso}, status={status})"
