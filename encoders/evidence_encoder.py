"""
Snath Research — Stream B: EvidenceEncoder.
============================================
Encodes the aggregated peer review text of a scientific paper into a latent
vector z_evidence ∈ R^768. Stream B in the V1–V6 divergence routing contract.

Architectural role
------------------
The evidence stream answers: "what do independent reviewers say this paper
actually demonstrates?"

Peer reviews are written by domain experts who have read the full paper —
the methods, the experiments, the statistics. They represent external,
independent assessment of what the paper actually proved, not what it claims.

This is structurally independent from the claims stream (M1): reviewer text
is written by different people, with different intent, with access to the full
paper rather than just the abstract.

Why peer reviews, not the methods section
-----------------------------------------
The methods section is written by the authors — it shares authorial bias with
the abstract. Peer reviews are written by independent experts specifically
tasked with assessing the gap between claims and evidence. They are the
natural Stream B: the independent, external, adversarial assessment.

Ground truth signal
-------------------
When a paper is accepted with high reviewer scores, the evidence stream
(reviews) validated the claims. Winner = "claims". When a paper is rejected
because reviewers found the evidence insufficient, the evidence stream was
right to diverge from the abstract's confidence. Winner = "reviews".

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
from typing import Optional, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.interfaces import AbstractModalEncoder


class EvidenceEncoder(AbstractModalEncoder, nn.Module):
    """
    Peer review evidence encoder. Stream B of the dual-stream routing contract.

    Encodes aggregated reviewer text into a shared latent space R^768.
    Represents what independent experts say the paper actually proved.

    Args:
        model_name: HuggingFace model ID. Default: allenai/scibert_scivocab_uncased
        embed_dim:  Shared latent space dimension D. Must match AbstractClaimsEncoder.
        max_length: Max tokenizer length per review. Default 512.
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
            device = "cpu"   # CPU default — see abstract_encoder.py note
        self.device = torch.device(device)

        self._model_name = model_name
        self._backbone = None
        self._tokenizer = None

        # Concept projection: 768 → concept_dim (default 8).
        # See AbstractClaimsEncoder for full rationale.
        # Separate instance from AbstractClaimsEncoder — V1 independence.
        # LoRA for the evidence stream is injected here only.
        self.proj = nn.Linear(768, embed_dim, bias=True)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.proj.bias)
        self.proj.to(self.device)

        self._lora_A: Optional[torch.Tensor] = None
        self._lora_B: Optional[torch.Tensor] = None

    def _load_backbone(self):
        if self._backbone is None:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._backbone = AutoModel.from_pretrained(self._model_name)
            self._backbone.to(self.device)
            self._backbone.eval()

    def _bert_encode(self, text: str) -> torch.Tensor:
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
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        return pooled.squeeze(0)   # (768,)

    def _aggregate_reviews(self, reviews: List[str]) -> str:
        """
        Concatenate review texts into a single string for encoding.

        Multiple reviews are separated by [SEP] tokens so the model
        can attend across reviewer perspectives. Truncated to max_length.
        """
        return " [SEP] ".join(r.strip() for r in reviews if r.strip())

    # ------------------------------------------------------------------
    # AbstractModalEncoder interface
    # ------------------------------------------------------------------

    def encode(self, x) -> np.ndarray:
        """
        M1–M3: encode peer review text → z_evidence ∈ R^embed_dim.

        Args:
            x: Either a single review string or a list of review strings.
               Lists are aggregated with [SEP] separation before encoding.
        Returns:
            z: (embed_dim,) numpy array, L2-normalised.
        """
        if isinstance(x, list):
            text = self._aggregate_reviews(x)
        else:
            text = x

        raw = self._bert_encode(text)
        with torch.no_grad():
            z = self.proj(raw.unsqueeze(0)).squeeze(0)
        return z.cpu().numpy()

    def get_confidence(self, z: np.ndarray) -> float:
        """
        Confidence = peakedness of softmax distribution.
        Identical formula to AbstractClaimsEncoder for cross-domain consistency.
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
        Load a signed LoRA adapter for concept-space correction.

        Mirrors AbstractClaimsEncoder.load_lora — stores A,B for application
        in forward() as  z + (z @ A) @ B, matching the DMN training objective.
        """
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            self._lora_A = payload["A"].to(self.device)
            self._lora_B = payload["B"].to(self.device)

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

        Identical objective to AbstractClaimsEncoder.finetune_projection:
            L = L_var + lambda_iso · L_cov

        Applied to pre-computed frozen BERT outputs of review texts.
        See AbstractClaimsEncoder for full rationale.
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
            z = self.forward(x)
            l_var = torch.relu(0.1 - z.std(dim=0)).mean()
            l_cov = sigreg(z)
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
        z = self.proj(F.normalize(raw_embedding, dim=-1))
        if self._lora_A is not None:
            z = z + torch.matmul(torch.matmul(z, self._lora_A), self._lora_B)
        return z
