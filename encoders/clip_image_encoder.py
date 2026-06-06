"""
Snath Research — Stream A: CLIPImageEncoder.
============================================
Encodes images via CLIP ViT-B/32 into a shared concept space R^embed_dim.
Stream A in the V1–V6 divergence routing contract, visual-language domain.

Architectural role
------------------
The image stream answers: "what does this image actually depict?"
It is the visual evidence against which the caption's claims are tested.

Paired with CLIPTextEncoder (Stream B = caption) in the COCO routing
experiment (AIA §Experiment 3). The routing score
D = ||softmax(z_img) - softmax(z_cap)||₁ / √G detects compositionality
failures where global CLS cosine agrees but concept-level alignment diverges.

Concept projection initialisation
----------------------------------
Two options, both produce a Linear(512, embed_dim) layer:

  init_concept_vocabulary()  — rows of proj.weight are L2-normalised CLIP
    text embeddings of COCO class names, scaled by temperature (τ=100).
    Requires embed_dim == len(class_names) == 80. Matches the companion
    paper exactly. Use this for Experiment 3.

  init_pca()  — rows are PCA directions of the image embedding distribution.
    Works for any embed_dim. Use for ablations or when running without COCO.

V1 compliance
-------------
CLIPImageEncoder never reads CLIPTextEncoder's output. The two projection
heads are separate nn.Linear instances. LoRA adapters are injected only into
this encoder's projection path.

Derivative Works note
---------------------
This file is a Derivative Work of AbstractModalEncoder (M1–M3),
AbstractDivergenceRouter (V1–V6), and JEPA_DMN_Consolidation_Node,
Apache 2.0, github.com/snath-ai/Lar-JEPA.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.interfaces import AbstractModalEncoder
from encoders._clip_backbone import get_clip

# COCO 80-class vocabulary — same order as COCO detection label IDs.
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

BACKBONE_DIM = 512  # CLIP ViT-B/32 image / text embedding dimension


class CLIPImageEncoder(AbstractModalEncoder, nn.Module):
    """
    CLIP ViT-B/32 image encoder. Stream A of the visual-language routing contract.

    Encodes images into a shared concept space R^embed_dim via:
        1. CLIP image tower  →  512-dim embedding  (backbone frozen at inference)
        2. Concept projection:  512 → embed_dim     (fine-tuned by SIGReg)

    Args:
        embed_dim:   Concept space dimension.
                     Use 80 for COCO vocabulary init (paper's Experiment 3).
                     Use any value for PCA init (ablations / pilots).
        device:      Torch device. Auto-detects CUDA > MPS > CPU.
        temperature: Scale applied to proj.weight in init_concept_vocabulary().
                     Paper uses τ=100. Has no effect when using PCA init
                     (routing_scale plays the same role there).
    """

    def __init__(
        self,
        embed_dim:   int   = 80,
        device:      Optional[str] = None,
        temperature: float = 100.0,
    ):
        nn.Module.__init__(self)
        self.embed_dim   = embed_dim
        self.temperature = temperature

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "cpu"   # MPS multi-instance deadlock — see abstract_encoder.py note
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Concept projection: 512 → embed_dim.
        # bias=False mirrors the dot-product / cosine structure of vocabulary init.
        self.proj = nn.Linear(BACKBONE_DIM, embed_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        self.proj.to(self.device)

        # LoRA adapter — stored as separate A, B tensors, applied in forward().
        self._lora_A: Optional[torch.Tensor] = None
        self._lora_B: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Backbone encoding
    # ------------------------------------------------------------------

    def _clip_image_embed(self, image) -> torch.Tensor:
        """Run CLIP image tower. Returns L2-normalised (512,) tensor on self.device."""
        model, proc = get_clip(self.device)
        inputs = proc(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)   # (1, 512)
        return F.normalize(feats.squeeze(0), dim=-1)      # (512,)

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

        This builds W_c from the companion paper: each row is the CLIP concept
        direction for one class, scaled by τ=100 so softmax(z) is peaked enough
        for routing confidence above tau_low.

        Requires embed_dim == len(class_names). For COCO Experiment 3 use the
        default COCO_CLASSES (80 classes) with embed_dim=80.

        Args:
            class_names: Vocabulary. Default: COCO_CLASSES (80 entries).
            freeze:      Freeze proj after init (companion paper / inference-only).
                         Set False for SIGReg fine-tuning (Experiment 3).
        """
        if self.embed_dim != len(class_names):
            raise ValueError(
                f"embed_dim={self.embed_dim} must equal len(class_names)="
                f"{len(class_names)} for vocabulary init. "
                f"Either set embed_dim={len(class_names)} or use init_pca()."
            )
        model, proc = get_clip(self.device)
        with torch.no_grad():
            inputs = proc(
                text=class_names, return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_feats = model.get_text_features(**inputs)   # (C, 512)
            text_feats = F.normalize(text_feats, dim=-1)      # unit vectors
        # Scale by temperature so softmax peaks above tau_low=0.25
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

        Works for any embed_dim. Returns explained variance ratio sum.

        Args:
            clip_embeddings: (N, 512) pre-computed CLIP image embeddings.
            routing_scale:   Multiplier applied to PCA components.
        """
        from sklearn.decomposition import PCA
        X = F.normalize(clip_embeddings, dim=-1).cpu().numpy()
        pca = PCA(n_components=self.embed_dim)
        pca.fit(X)
        W = torch.tensor(pca.components_, dtype=torch.float32)  # (embed_dim, 512)
        self.proj.weight.data = (W * routing_scale).to(self.device)
        return float(pca.explained_variance_ratio_.sum())

    # ------------------------------------------------------------------
    # AbstractModalEncoder interface
    # ------------------------------------------------------------------

    def encode(self, x) -> np.ndarray:
        """
        M1–M3: encode image → z ∈ R^embed_dim.

        Args:
            x: PIL Image, numpy array (H,W,3), or path string / Path to image file.
        Returns:
            z: (embed_dim,) numpy array.
        """
        if isinstance(x, (str, Path)):
            from PIL import Image as _PIL
            x = _PIL.open(x).convert("RGB")
        raw = self._clip_image_embed(x)             # (512,) on device
        with torch.no_grad():
            z = self.forward(raw.unsqueeze(0)).squeeze(0)
        return z.cpu().numpy()

    def get_confidence(self, z: np.ndarray) -> float:
        """
        Confidence = peakedness of softmax(z).

        z is already temperature-scaled (baked into proj.weight), so no
        additional scaling is applied here. Identical formula to SciBERT
        encoders for cross-domain consistency.
        """
        z_t = torch.tensor(z, dtype=torch.float32)
        p   = torch.softmax(z_t, dim=0)
        n   = len(p)
        peak = (float(p.max()) - 1.0 / n) / (1.0 - 1.0 / n)
        return max(0.0, peak)

    # ------------------------------------------------------------------
    # LoRA injection
    # ------------------------------------------------------------------

    def load_lora(self, pt_path: str) -> None:
        """
        Load a signed LoRA adapter for concept-space correction.

        Stores A:(embed_dim,1) and B:(1,embed_dim) for application in forward()
        as z + (z @ A) @ B. Matches the DMN training objective exactly.
        """
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

        Objective (VICReg-style, projection-only):
            L = L_var + lambda_iso · L_cov

        Applied to pre-computed frozen CLIP image embeddings.
        For full InfoNCE + SIGReg continued training on paired image-text
        batches, use run_coco_experiment.py instead.

        Args:
            clip_embeddings: (N, 512) pre-computed CLIP image embeddings.
            lambda_iso:      SIGReg weight. AIA Exp 3 sweep: {0.01, 0.1, 1.0}.
            n_epochs:        Optimisation steps over the full batch.
            lr:              AdamW learning rate.

        Returns:
            dict with isotropy_before, isotropy_after, final_loss.
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
