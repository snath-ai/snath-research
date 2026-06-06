"""
Snath Research — JEPA Predictor (world model loop closure).
============================================================
Predicts z_reviews (evidence representation) from z_claims (claim
representation) using concept-space embeddings from the CLIP encoders.

This closes the annotation loop:
    abstract  → enc_claims  → z_claims
                                  ↓
                              predictor  →  ẑ_reviews
                                  ↓
    methods   → enc_reviews → z_reviews
                                  ↓
              D_pred = 1 - cos(ẑ_reviews, z_reviews)   [stop-gradient]

Training signal = prediction error. No labels. The paper itself provides
supervision: if the abstract's claims are consistent with the methodology,
the predictor should be able to predict z_reviews from z_claims. When it
can't, the paper is internally inconsistent.

Derivative Works note
---------------------
This file extends AbstractDivergenceRouter (V1–V6), JEPA_DMN_Consolidation_Node,
and the JEPA framing from Assran et al. (2023) / LeCun (2022).
Apache 2.0, github.com/snath-ai/Lar-JEPA.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class JEPAPredictor(nn.Module):
    """
    Concept-space predictor: z_claims → ẑ_reviews.

    Architecture: residual MLP with LayerNorm (stable for high-dim embeddings).
    Loss: cosine similarity to stop-gradient target — prevents representation
    collapse without a momentum encoder.

    The stop-gradient on z_reviews is the JEPA key:
        L = 1 - E[cos(predictor(z_claims), sg(z_reviews))]

    Papers where prediction error is high are internally inconsistent —
    the abstract claims something the methodology/evidence does not support.

    Args:
        embed_dim:   Concept space dimension. Must match encoder embed_dim.
        hidden_mult: Hidden layer size multiplier. Default 4 = embed_dim * 4.
    """

    def __init__(self, embed_dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = embed_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.skip = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, z_claims: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_claims: (B, embed_dim) claim representations.
        Returns:
            ẑ_reviews: (B, embed_dim) predicted evidence representations.
        """
        return self.net(z_claims) + self.skip(z_claims)

    def prediction_loss(
        self, z_claims: torch.Tensor, z_reviews: torch.Tensor
    ) -> torch.Tensor:
        """
        JEPA cosine loss to stop-gradient target.

        1.0 = perfectly wrong, 0.0 = perfectly predicted.
        """
        z_hat   = self.forward(z_claims)
        target  = z_reviews.detach()                   # stop-gradient — the key
        z_hat_n = F.normalize(z_hat,   dim=-1)
        tgt_n   = F.normalize(target,  dim=-1)
        return 1.0 - (z_hat_n * tgt_n).sum(dim=-1).mean()

    def prediction_error(
        self, z_claims: torch.Tensor, z_reviews: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-sample prediction error ∈ [0, 2] (no grad).

        High error → claim and evidence representations are inconsistent.
        Use as label-free quality signal for paper routing — papers with
        high prediction error are flagged as internally inconsistent before
        any human reads them.
        """
        with torch.no_grad():
            z_hat   = self.forward(z_claims)
            z_hat_n = F.normalize(z_hat,      dim=-1)
            z_rev_n = F.normalize(z_reviews,  dim=-1)
            return 1.0 - (z_hat_n * z_rev_n).sum(dim=-1)

    def train_on_batch(
        self,
        z_claims:  torch.Tensor,
        z_reviews: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single gradient step. Returns scalar loss value."""
        optimizer.zero_grad()
        loss = self.prediction_loss(z_claims, z_reviews)
        loss.backward()
        optimizer.step()
        return float(loss.item())


def train_predictor(
    predictor:  JEPAPredictor,
    z_claims:   torch.Tensor,
    z_reviews:  torch.Tensor,
    n_epochs:   int   = 200,
    lr:         float = 1e-3,
    batch_size: int   = 128,
) -> dict:
    """
    Train JEPA predictor on precomputed concept-space embedding pairs.

    No labels. The supervision comes from within-pair consistency:
    a claims representation should predict its paired evidence representation.
    Papers where the predictor fails (high error) are internally inconsistent.

    Args:
        predictor:  JEPAPredictor instance (on the correct device).
        z_claims:   (N, embed_dim) claim concept vectors (e.g. abstract, image).
        z_reviews:  (N, embed_dim) evidence concept vectors (e.g. methods, caption).
        n_epochs:   Training epochs over the full dataset.
        lr:         AdamW learning rate.
        batch_size: Mini-batch size.

    Returns:
        dict: error_before, error_after, loss_final.
    """
    device    = next(predictor.parameters()).device
    z_claims  = z_claims.to(device).detach()
    z_reviews = z_reviews.to(device).detach()
    N         = z_claims.size(0)

    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=lr, weight_decay=1e-4
    )

    with torch.no_grad():
        err_before = float(predictor.prediction_error(z_claims, z_reviews).mean())

    predictor.train()
    loss_val = 0.0
    for epoch in range(n_epochs):
        idx = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            b        = idx[start : start + batch_size]
            loss_val = predictor.train_on_batch(z_claims[b], z_reviews[b], optimizer)
        if (epoch + 1) % 50 == 0:
            log.info(f"  predictor epoch {epoch+1:3d}/{n_epochs}  loss={loss_val:.4f}")

    predictor.eval()
    with torch.no_grad():
        err_after = float(predictor.prediction_error(z_claims, z_reviews).mean())

    log.info(f"  Predictor: error {err_before:.4f} → {err_after:.4f}  "
             f"(Δ={err_after - err_before:+.4f})")
    return {
        "error_before": err_before,
        "error_after":  err_after,
        "loss_final":   loss_val,
    }
