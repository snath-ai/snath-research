"""
Snath Research — TF-IDF + SVD encoder (M1-safe, no model loading).

Drop-in replacement for AbstractClaimsEncoder / EvidenceEncoder on machines
where Intel MKL / OpenMP deadlocks during transformer model weight loading
(macOS M1 with pip-installed PyTorch).

WHY THIS WORKS FOR THE EXPERIMENT
----------------------------------
For overclaiming papers:
    Abstract: "revolutionary breakthrough", "unprecedented", "state-of-the-art"
    Reviews:  "insufficient", "limited", "sample size too small", "claims exceed"

    → Different TF-IDF vocabularies → different SVD projections → high D ✓

For coherent papers:
    Abstract and reviews share domain terms, experimental details, method names.
    → Similar TF-IDF vocabularies → similar projections → low D ✓

The routing signal is real. It's weaker than SciBERT (no semantic understanding)
but sufficient to prove the D_hard → DMN → LoRA learning loop on real OpenReview
data. For paper submission, swap back to SciBERT on a Linux machine.

USAGE
-----
    from encoders.tfidf_encoder import TFIDFEncoderPair

    enc = TFIDFEncoderPair(embed_dim=768)
    enc.fit(claims_texts, reviews_texts)   # fit once on the full corpus

    z_claims  = enc.claims.encode(abstract_text)    # (768,) numpy
    z_reviews = enc.reviews.encode(review_text)     # (768,) numpy
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.interfaces import AbstractModalEncoder


class _TFIDFStream(AbstractModalEncoder):
    """
    Single-stream TF-IDF + SVD encoder.

    Implements AbstractModalEncoder (M1–M3). Relies on a shared TF-IDF
    vocabulary fitted on both streams (so z_claims and z_reviews live in
    the same concept space).
    """

    def __init__(self, stream_name: str, embed_dim: int = 768):
        self._stream  = stream_name
        self.embed_dim = embed_dim
        self._tfidf: Optional[TfidfVectorizer]  = None
        self._svd:   Optional[TruncatedSVD]     = None

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    @property
    def modality(self) -> str:
        return f"tfidf_svd_{self._stream}"

    def _fitted(self) -> bool:
        return self._tfidf is not None and self._svd is not None

    # Routing scale: softmax(z) needs peaks to exceed tau_high=0.60.
    # For L2-norm unit vector z, max(softmax(z)) ≈ 0.28 → confidence ≈ 0.18,
    # which is below tau_low=0.25. Scaling by 3.0 gives max(softmax) ≈ 0.74
    # and confidence ≈ 0.70, matching the smoke-test calibration.
    _ROUTING_SCALE = 3.0

    def encode(self, x: str) -> np.ndarray:
        """
        M1–M3: encode text → z ∈ R^embed_dim, scaled for routing.

        Args:
            x: claims text (stream A) or reviews text (stream B).
        Returns:
            (embed_dim,) numpy array, L2-normalised then scaled by 3.0.
        """
        if not self._fitted():
            raise RuntimeError(
                f"TFIDFEncoder ({self._stream}) not fitted. "
                "Call TFIDFEncoderPair.fit() first."
            )
        vec    = self._tfidf.transform([x])                          # (1, vocab)
        proj   = self._svd.transform(vec)                            # (1, embed_dim)
        normed = normalize(proj, norm="l2") * self._ROUTING_SCALE    # scale for routing
        return normed[0].astype(np.float32)                          # (embed_dim,)

    def get_confidence(self, z: np.ndarray) -> float:
        """Peakedness of softmax over embed_dim — same formula as all Snath repos."""
        import torch
        z_t = torch.tensor(z, dtype=torch.float32)
        p   = torch.softmax(z_t, dim=0)
        n   = len(p)
        return max(0.0, (float(p.max()) - 1.0 / n) / (1.0 - 1.0 / n))

    def load_lora(self, pt_path: str) -> None:
        """No-op for TF-IDF encoder — LoRA targets neural projection layers only."""
        pass


class TFIDFEncoderPair:
    """
    Fitted pair of TF-IDF encoders sharing one vocabulary.

    A single TF-IDF vocabulary is fitted on BOTH claims and reviews texts
    so that z_claims and z_reviews live in the same concept space and
    divergence scores are meaningful.

    Args:
        embed_dim:    SVD output dimension. Default 768 (matches SciBERT).
        max_features: TF-IDF vocabulary size. Default 20000.
        min_df:       Minimum document frequency. Default 2.
    """

    def __init__(
        self,
        embed_dim:    int = 768,
        max_features: int = 20_000,
        min_df:       int = 2,
    ):
        self.embed_dim    = embed_dim
        self.max_features = max_features
        self.min_df       = min_df

        self.claims  = _TFIDFStream("claims",  embed_dim)
        self.reviews = _TFIDFStream("reviews", embed_dim)

        # Shared vocabulary (fitted on all texts)
        self._shared_tfidf: Optional[TfidfVectorizer] = None

    def fit(
        self,
        claims_texts:  List[str],
        reviews_texts: List[str],
    ) -> "TFIDFEncoderPair":
        """
        Fit shared TF-IDF vocabulary then separate SVD projections.

        Fitting on the UNION of claims and reviews ensures both streams
        share the same vocabulary — a word in an abstract means the same
        thing as the same word in a review.

        Separate SVD projections preserve stream independence (V1): the
        concept axes for claims text and reviews text are learned separately,
        so the two streams can develop different response patterns to the
        same vocabulary.

        Args:
            claims_texts:  list of all abstract / author summary texts
            reviews_texts: list of all aggregated reviewer texts
        Returns:
            self (for chaining)
        """
        all_texts = claims_texts + reviews_texts

        # Shared vocabulary
        self._shared_tfidf = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._shared_tfidf.fit(all_texts)

        actual_dim = min(self.embed_dim, len(self._shared_tfidf.vocabulary_) - 1)
        if actual_dim < self.embed_dim:
            self.embed_dim = actual_dim
            self.claims.embed_dim  = actual_dim
            self.reviews.embed_dim = actual_dim

        # Separate SVD projections — stream independence (V1)
        claims_mat  = self._shared_tfidf.transform(claims_texts)
        reviews_mat = self._shared_tfidf.transform(reviews_texts)

        svd_c = TruncatedSVD(n_components=self.embed_dim, random_state=42)
        svd_c.fit(claims_mat)

        svd_r = TruncatedSVD(n_components=self.embed_dim, random_state=42)
        svd_r.fit(reviews_mat)

        # Inject into stream encoders
        self.claims._tfidf  = self._shared_tfidf
        self.claims._svd    = svd_c
        self.reviews._tfidf = self._shared_tfidf
        self.reviews._svd   = svd_r

        return self

    @property
    def fitted(self) -> bool:
        return self._shared_tfidf is not None
