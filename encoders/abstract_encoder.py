"""
Snath Research — Stream A: AbstractClaimsEncoder.
==================================================
Encodes the abstract + author summary of a scientific paper into a latent
vector z_claims ∈ R^768. Stream A in the V1–V6 divergence routing contract.

Architectural role
------------------
The claims stream answers: "what does this paper assert it has proved?"
It captures the author's strongest forward-facing statements — the abstract,
the contribution bullets, the conclusion — the parts written to impress.

Stream independence (M1–M3)
---------------------------
This encoder never reads the EvidenceEncoder's output. It is applied
independently to paper.abstract + paper.summary. The LoRA injection path
(load_lora) is separate from the EvidenceEncoder's LoRA path.

Ground truth signal
-------------------
When a paper is rejected because "claims outpace evidence", this stream
was overclaiming. The D_hard event records the divergence. The winner is
"reviews" (the evidence stream was right to be cautious). The LoRA adapter
trained on those events adjusts this encoder's projection to be more
conservative — pulling future overclaiming abstracts toward realistic
positions in concept space.

LoRA injection
--------------
load_lora() applies a signed A·B delta to the projection layer.
Called by AdapterRouter.resolve() only when W >= min_trust.
The delta is perishable — abstracts from 2022 conferences follow
different claim patterns than 2026. Temporal gate λ=0.50 for
scope_overclaim (fast-aging) or λ=0.02 for methodology_gap (persistent).

Derivative Works note
---------------------
This file is a Derivative Work of AbstractModalEncoder (M1–M3),
AbstractDivergenceRouter (V1–V6), and JEPA_DMN_Consolidation_Node,
Apache 2.0, github.com/snath-ai/Lar-JEPA.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # prevent deadlock on macOS fork

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.interfaces import AbstractModalEncoder


class AbstractClaimsEncoder(AbstractModalEncoder, nn.Module):
    """
    Scientific paper claims encoder. Stream A of the dual-stream routing contract.

    Encodes the abstract and author-written summary into a shared latent
    space R^768. Represents what the paper *claims* to have proved.

    Args:
        model_name: HuggingFace model ID. Default: allenai/scibert_scivocab_uncased
        embed_dim:  Shared latent space dimension D. Must match EvidenceEncoder.
                    Default 768.
        max_length: Max tokenizer length. Default 512.
        device:     Torch device. Defaults to MPS on M1, else CPU.
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        embed_dim: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.max_length = max_length

        if device is None:
            # Default to CPU — MPS can deadlock when multiple SciBERT
            # instances share the Metal command queue on M1.
            # Set device="mps" explicitly if running single-encoder inference.
            device = "cpu"
        self.device = torch.device(device)

        # Load SciBERT backbone (lazy — only on first encode call if needed)
        self._model_name = model_name
        self._backbone = None
        self._tokenizer = None

        # Concept projection: 768 → concept_dim (default 8).
        #
        # AIA §2 routes on v = softmax(φ(x) · W_c / τ) where W_c projects
        # to a LOW-DIMENSIONAL concept space (AIA uses C=80 COCO classes).
        # Routing softmax(z) over 768 dims produces near-uniform distributions
        # (values ≈ 1/768) — the routing signal is dead. Projecting to C=8
        # gives peaked, discriminative distributions.
        #
        # The 8 dimensions are learned (not named), but they represent the
        # encoder's internal "concept axes" for claim quality. LoRA adapters
        # then shift the projection for known failure patterns.
        # Raw linear projection — no LayerNorm.
        # LayerNorm inside proj forces each sample to unit within-sample variance,
        # which trivially zeroes the batch off-diagonal covariance SIGReg needs to
        # optimise. std=0.1 init gives z_i ~ N(0, ~0.8) so softmax is peaked enough
        # for routing confidence > tau_low while leaving real variance for SIGReg.
        self.proj = nn.Linear(768, embed_dim, bias=True)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.proj.bias)
        self.proj.to(self.device)

    def _load_backbone(self):
        """Lazy load SciBERT — only when encode() is first called."""
        if self._backbone is None:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._backbone = AutoModel.from_pretrained(self._model_name)
            self._backbone.to(self.device)
            self._backbone.eval()

    def _bert_encode(self, text: str) -> torch.Tensor:
        """
        Run SciBERT on text, return mean-pooled CLS embedding (768,).
        """
        self._load_backbone()
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._backbone(**inputs)
        # Mean pool over token dimension (exclude padding)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state          # (1, seq, 768)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (1, seq, 1)
        pooled = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        return pooled.squeeze(0)   # (768,)

    # ------------------------------------------------------------------
    # AbstractModalEncoder interface
    # ------------------------------------------------------------------

    def encode(self, x: str) -> np.ndarray:
        """
        M1–M3: encode paper abstract/summary → z_claims ∈ R^embed_dim.

        Args:
            x: Paper abstract + author summary as a single string.
        Returns:
            z: (embed_dim,) numpy array, L2-normalised.
        """
        raw = self._bert_encode(x)                          # (768,)
        with torch.no_grad():
            z = self.proj(raw.unsqueeze(0)).squeeze(0)      # (embed_dim,)
        return z.cpu().numpy()

    def get_confidence(self, z: np.ndarray) -> float:
        """
        Confidence = peakedness of the softmax distribution over embed_dim.

        A flat (uniform) distribution → confidence ≈ 0.
        A peaked distribution → confidence → 1.

        Identical formula to Snath Robotics / Aviation for cross-domain consistency.
        """
        z_t = torch.tensor(z, dtype=torch.float32)
        p = torch.softmax(z_t, dim=0)
        n = len(p)
        peak = (float(p.max()) - 1.0 / n) / (1.0 - 1.0 / n)
        return max(0.0, peak)

    # ------------------------------------------------------------------
    # LoRA injection (System 2)
    # ------------------------------------------------------------------

    def load_lora(self, pt_path: str) -> None:
        """
        Apply a signed LoRA delta to the projection layer.

        The delta encodes a learned correction for a specific claims-failure
        mode (e.g., scope_overclaim at NeurIPS-style venues). Perishable —
        gated by temporal trust W >= min_trust before injection.

        Args:
            pt_path: Path to the signed .pt adapter file.
        """
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        A = payload["A"].to(self.device)   # (embed_dim, rank)
        B = payload["B"].to(self.device)   # (rank, embed_dim)
        with torch.no_grad():
            self.proj.weight.data += (A @ B)

    # ------------------------------------------------------------------
    # SIGReg projection fine-tuning (AIA Experiment 3)
    # ------------------------------------------------------------------

    def finetune_projection(
        self,
        bert_embeddings: torch.Tensor,
        lambda_iso: float = 0.1,
        n_epochs: int = 300,
        lr: float = 5e-4,
    ) -> dict:
        """
        Fine-tune the concept projection for isotropy using SIGReg.

        Objective (VICReg-style):
            L = L_var + lambda_iso · L_cov
        where:
            L_var = mean(relu(0.1 - std(z, dim=0)))   collapse prevention
            L_cov = SIGRegLoss(z)                      off-diagonal covariance

        This is the "signal should be accurate in the first place" step —
        making the 8 concept dimensions equally informative before any
        LoRA correction is applied. Applied to pre-computed frozen BERT
        outputs so no second backbone pass is needed.

        Args:
            bert_embeddings: (N, 768) pre-computed SciBERT mean-pool tensors.
            lambda_iso: SIGReg weight. AIA Exp 3 sweep: {0.01, 0.1, 1.0}.
            n_epochs: Optimisation steps over the full batch.
            lr: AdamW learning rate.

        Returns:
            dict with isotropy_before, isotropy_after, final_loss.
        """
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dmn.sigreg import SIGRegLoss

        sigreg = SIGRegLoss(lambda_iso=lambda_iso)
        optimizer = torch.optim.AdamW(self.proj.parameters(), lr=lr)

        x = bert_embeddings.to(self.device).detach()

        with torch.no_grad():
            isotropy_before = sigreg.isotropy_ratio(self.forward(x))

        loss = torch.tensor(0.0)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            z = self.forward(x)                                    # (N, D)
            l_var = torch.relu(0.1 - z.std(dim=0)).mean()         # collapse prevention
            l_cov = sigreg(z)                                      # isotropy
            loss = l_var + l_cov
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            isotropy_after = sigreg.isotropy_ratio(self.forward(x))

        return {
            "isotropy_before": isotropy_before,
            "isotropy_after":  isotropy_after,
            "final_loss":      float(loss.item()),
        }

    # ------------------------------------------------------------------
    # nn.Module forward (for batch training)
    # ------------------------------------------------------------------

    def forward(self, raw_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_embedding: (B, 768) pre-computed SciBERT mean-pool output.
        Returns:
            z: (B, embed_dim) normalised projection.
        """
        return self.proj(raw_embedding)
