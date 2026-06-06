"""
Snath Research — Stream B: CLIPTextEncoder.
============================================
Encodes captions via CLIP ViT-B/32 text tower into a shared concept space.
Stream B in the V1–V6 divergence routing contract, visual-language domain.

Architectural role
------------------
The text stream answers: "what does this caption claim the image depicts?"
It is the linguistic claim against which the image evidence is tested.

Paired with CLIPImageEncoder (Stream A = image) in the COCO routing
experiment (AIA §Experiment 3). High routing score D signals that the
caption claims concepts the image doesn't support — the compositionality
failure the companion paper identified as invisible to global CLS fusion.

V1 compliance
-------------
CLIPTextEncoder never reads CLIPImageEncoder's output. The two projection
heads are separate nn.Linear instances with separate LoRA adapter slots.
Both encoders share the underlying CLIP backbone (loaded once via
_clip_backbone.get_clip()), but their concept projection parameters are
fully independent — sharing the backbone does not violate V1 because V1
prohibits cross-stream *output* reads, not shared pre-trained weights.

Concept projection initialisation
----------------------------------
Same two options as CLIPImageEncoder:

  init_concept_vocabulary()  — COCO 80-class CLIP embeddings × τ=100.
    Requires embed_dim=80. Use for Experiment 3.

  init_pca()  — PCA directions of caption embedding distribution.
    Works for any embed_dim. Mirror CLIPImageEncoder.init_pca() on
    caption embeddings for cross-stream alignment.

Derivative Works note
---------------------
This file is a Derivative Work of AbstractModalEncoder (M1–M3),
AbstractDivergenceRouter (V1–V6), and JEPA_DMN_Consolidation_Node,
Apache 2.0, github.com/snath-ai/Lar-JEPA.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.interfaces import AbstractModalEncoder
from encoders._clip_backbone import get_clip
from encoders.clip_image_encoder import COCO_CLASSES, BACKBONE_DIM


class CLIPTextEncoder(AbstractModalEncoder, nn.Module):
    """
    CLIP ViT-B/32 text encoder. Stream B of the visual-language routing contract.

    Encodes captions into a shared concept space R^embed_dim via:
        1. CLIP text tower   →  512-dim embedding  (backbone frozen at inference)
        2. Concept projection:  512 → embed_dim     (fine-tuned by SIGReg)

    The concept projection is a separate nn.Linear from CLIPImageEncoder's
    projection — same dimensionality (M2), independent parameters (V1).

    Args:
        embed_dim:   Concept space dimension. Must match CLIPImageEncoder.embed_dim.
        device:      Torch device. Auto-detects CUDA > MPS > CPU.
        max_length:  Max token length for CLIP tokeniser. CLIP hard cap is 77.
        temperature: Scale applied to proj.weight in init_concept_vocabulary().
    """

    def __init__(
        self,
        embed_dim:   int   = 80,
        device:      Optional[str] = None,
        max_length:  int   = 77,
        temperature: float = 100.0,
    ):
        nn.Module.__init__(self)
        self.embed_dim   = embed_dim
        self.max_length  = max_length
        self.temperature = temperature

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "cpu"   # MPS multi-instance deadlock — see abstract_encoder.py
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.proj = nn.Linear(BACKBONE_DIM, embed_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        self.proj.to(self.device)

        self._lora_A: Optional[torch.Tensor] = None
        self._lora_B: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Backbone encoding
    # ------------------------------------------------------------------

    def _clip_text_embed(self, text: str) -> torch.Tensor:
        """Run CLIP text tower. Returns L2-normalised (512,) tensor on self.device."""
        model, proc = get_clip(self.device)
        inputs = proc(
            text=[text], return_tensors="pt",
            padding=True, truncation=True, max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)   # (1, 512)
        return F.normalize(feats.squeeze(0), dim=-1)     # (512,)

    # ------------------------------------------------------------------
    # Concept projection initialisation
    # ------------------------------------------------------------------

    def init_concept_vocabulary(
        self,
        class_names: List[str] = COCO_CLASSES,
        freeze: bool = False,
    ) -> None:
        """
        Initialise proj.weight with CLIP text embeddings of class_names × temperature.

        Mirrors CLIPImageEncoder.init_concept_vocabulary — both streams share the
        same concept vocabulary so their z vectors live in a comparable concept
        space and D = ||softmax(z_img) - softmax(z_cap)||₁ / √G is meaningful.

        Requires embed_dim == len(class_names).
        """
        if self.embed_dim != len(class_names):
            raise ValueError(
                f"embed_dim={self.embed_dim} must equal len(class_names)="
                f"{len(class_names)} for vocabulary init."
            )
        model, proc = get_clip(self.device)
        with torch.no_grad():
            inputs = proc(
                text=class_names, return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_feats = model.get_text_features(**inputs)   # (C, 512)
            text_feats = F.normalize(text_feats, dim=-1)
        self.proj.weight.data = text_feats * self.temperature
        if freeze:
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def init_pca(
        self,
        clip_embeddings: torch.Tensor,
        routing_scale: float = 20.0,
    ) -> float:
        """
        Initialise proj.weight with PCA directions of clip_embeddings.

        Call this on caption embeddings, mirroring CLIPImageEncoder.init_pca()
        on image embeddings, to ensure both streams start with comparable
        concept-space geometry.

        Returns explained variance ratio sum.
        """
        from sklearn.decomposition import PCA
        X   = F.normalize(clip_embeddings, dim=-1).cpu().numpy()
        pca = PCA(n_components=self.embed_dim)
        pca.fit(X)
        W   = torch.tensor(pca.components_, dtype=torch.float32)
        self.proj.weight.data = (W * routing_scale).to(self.device)
        return float(pca.explained_variance_ratio_.sum())

    # ------------------------------------------------------------------
    # AbstractModalEncoder interface
    # ------------------------------------------------------------------

    def encode(self, x: str) -> np.ndarray:
        """
        M1–M3: encode caption → z ∈ R^embed_dim.

        Args:
            x: Caption string.
        Returns:
            z: (embed_dim,) numpy array.
        """
        raw = self._clip_text_embed(x)
        with torch.no_grad():
            z = self.forward(raw.unsqueeze(0)).squeeze(0)
        return z.cpu().numpy()

    def get_confidence(self, z: np.ndarray) -> float:
        """Confidence = peakedness of softmax(z). Identical to CLIPImageEncoder."""
        z_t  = torch.tensor(z, dtype=torch.float32)
        p    = torch.softmax(z_t, dim=0)
        n    = len(p)
        peak = (float(p.max()) - 1.0 / n) / (1.0 - 1.0 / n)
        return max(0.0, peak)

    # ------------------------------------------------------------------
    # LoRA injection
    # ------------------------------------------------------------------

    def load_lora(self, pt_path: str) -> None:
        """Load signed LoRA adapter. Identical interface to CLIPImageEncoder."""
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            self._lora_A = payload["A"].to(self.device)
            self._lora_B = payload["B"].to(self.device)

    # ------------------------------------------------------------------
    # SIGReg projection fine-tuning
    # ------------------------------------------------------------------

    def finetune_projection(
        self,
        clip_embeddings: torch.Tensor,
        lambda_iso: float = 0.1,
        n_epochs:   int   = 300,
        lr:         float = 5e-4,
    ) -> dict:
        """
        Fine-tune concept projection for isotropy using SIGReg.

        Apply to pre-computed frozen CLIP text embeddings of COCO captions.
        Objective: L = L_var + lambda_iso · L_cov (same as CLIPImageEncoder).

        For full InfoNCE + SIGReg continued training use run_coco_experiment.py.
        """
        from dmn.sigreg import SIGRegLoss
        sigreg    = SIGRegLoss(lambda_iso=lambda_iso)
        optimizer = torch.optim.AdamW(self.proj.parameters(), lr=lr)
        x         = clip_embeddings.to(self.device).detach()

        with torch.no_grad():
            isotropy_before = sigreg.isotropy_ratio(self.forward(x))

        loss = torch.tensor(0.0)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            z     = self.forward(x)
            l_var = torch.relu(0.1 - z.std(dim=0)).mean()
            l_cov = sigreg(z)
            loss  = l_var + l_cov
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
    # nn.Module forward
    # ------------------------------------------------------------------

    def forward(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embedding: (B, 512) pre-computed L2-normalised CLIP embeddings.
        Returns:
            z: (B, embed_dim) concept projection, with LoRA correction if loaded.
        """
        z = self.proj(F.normalize(clip_embedding, dim=-1))
        if self._lora_A is not None:
            z = z + torch.matmul(torch.matmul(z, self._lora_A), self._lora_B)
        return z
