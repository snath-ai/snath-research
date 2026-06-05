"""
AbstractModalEncoder — M1–M3 invariants for Snath Research.

Copied from the Lár-JEPA ABC specification (Apache 2.0,
github.com/snath-ai/Lar-JEPA). Reproduced here for repo self-containment.

M1 — Stream independence: neither encoder reads the other's output.
M2 — Shared embedding space: both encoders project to the same R^D.
M3 — No cross-stream gradient at inference: projection heads frozen.
"""
from abc import ABC, abstractmethod
import numpy as np


class AbstractModalEncoder(ABC):
    """
    Universal modality-to-latent-space encoding interface.

    Any encoder that satisfies M1–M3 can serve as Stream A or Stream B
    in the V1–V6 divergence routing contract without modification to
    any Lár primitive.
    """

    @abstractmethod
    def encode(self, x) -> np.ndarray:
        """
        Encode raw input into a latent vector z ∈ R^D.

        M1: implementation must NOT read from the other stream's encoder.
        M2: output shape must match the paired encoder's output shape.

        Args:
            x: Raw input (text string, sensor reading, image features, etc.)
        Returns:
            z: (D,) numpy array — the latent representation.
        """
        ...

    @abstractmethod
    def get_confidence(self, z: np.ndarray) -> float:
        """
        Scalar confidence in [0, 1] derived from the latent vector.

        Used by the divergence router as c_A or c_B. Must be computed
        purely from z — no access to the other stream's latent or to
        any external state.

        Args:
            z: (D,) latent vector from encode().
        Returns:
            confidence: float in [0, 1].
        """
        ...
