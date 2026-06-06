"""
Shared CLIP ViT-B/32 backbone singleton for CLIPImageEncoder / CLIPTextEncoder.

Loading CLIP twice causes Metal command queue contention on MPS and wastes
~700MB of VRAM on CUDA. Both encoder classes call get_clip() which loads
once and caches for the process lifetime.
"""
from __future__ import annotations

from typing import Optional, Tuple
import torch

_model  = None
_proc   = None
_device: Optional[torch.device] = None


def get_clip(device: torch.device) -> Tuple:
    """
    Return (CLIPModel, CLIPProcessor), loaded lazily and pinned to `device`.

    If called a second time with a different device the cached instance is
    returned unchanged — move it yourself if you need it elsewhere.
    """
    global _model, _proc, _device
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()
        _model.to(device)
        _device = device
    return _model, _proc
