"""
ResearchAdapterRouter — System 1 + System 2 two-pass inference.
===============================================================
Identical trust asymmetry pattern as Snath Robotics / Aviation / Basis:

  System 1 — Identification (trust-invariant)
  Cosine similarity matching on the divergence vector fingerprint.
  The geometric signature of "scope_overclaim" in concept space is
  durable across conference years. Fires regardless of adapter age.

  System 2 — Correction (perishable)
  LoRA weights encode a learned correction for a specific failure class
  at a specific venue epoch. Gated by W = exp(-λ · Δt) ≥ min_trust.

  Degradation path
  When System 2 is refused, System 1 still identifies the failure class
  and returns COMMIT_TRAJECTORY with the identification recorded in the
  audit note. Identify correctly, correct conservatively.

Temporal decay constants:
  scope_overclaim      λ=0.50  — conference-specific, venue patterns shift fast
  statistical_weakness λ=0.20  — medium decay, methodology trends evolve
  methodology_gap      λ=0.02  — persistent, fundamental flaws don't change

Derivative Works note
---------------------
This file is a Derivative Work of AbstractDivergenceRouter (V1–V6) and
JEPA_DMN_Consolidation_Node, Apache 2.0, github.com/snath-ai/Lar-JEPA.
"""
from __future__ import annotations

import json
import math
import datetime
import glob
from pathlib import Path
from typing import Optional

import hmac as _hmac
import hashlib
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.types import RouteDecision

try:
    from brain.abstract_adapter_router import AbstractAdapterRouter
except ImportError:
    from abc import ABC as AbstractAdapterRouter

_ADAPTER_KEY = b"snath_research_adapter_sovereignty_2026"


def _verify_json(adapter: dict) -> bool:
    """Verify HMAC of JSON centroid adapter. Rejects unsigned adapters."""
    stored_sig = adapter.get("sig", "")
    if not stored_sig:
        return False
    immutable_keys = ["failure_class", "centroid_claims", "centroid_reviews",
                      "winner", "win_rate", "n_events"]
    payload = json.dumps(
        {k: adapter[k] for k in immutable_keys if k in adapter},
        sort_keys=True,
    ).encode()
    expected = _hmac.new(_ADAPTER_KEY, payload, hashlib.sha256).hexdigest()
    return _hmac.compare_digest(stored_sig, expected)


def _verify_pt(meta: dict) -> bool:
    """Verify HMAC of .pt adapter before injection."""
    stored_sig = meta.get("hmac_hex", "")
    if not stored_sig:
        return False
    try:
        A = meta["A"]
        B = meta["B"]
        fc = meta.get("failure_class", "")
        te = meta.get("target_encoder", "")
        a_hash = hashlib.sha256(A.numpy().tobytes()).hexdigest()[:16]
        b_hash = hashlib.sha256(B.numpy().tobytes()).hexdigest()[:16]
        payload_str = f"{fc}|{te}|{a_hash}|{b_hash}"
        expected = _hmac.new(
            _ADAPTER_KEY, payload_str.encode(), hashlib.sha256
        ).hexdigest()
        return _hmac.compare_digest(stored_sig, expected)
    except Exception:
        return False


def _cos(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ── Temporal decay — identical formula across all Snath repos ─────────────────
_LAMBDA: dict = {
    "scope_overclaim":      0.50,
    "statistical_weakness": 0.20,
    "methodology_gap":      0.02,
    "default":              0.10,
}


def _decay_weight(created_at_iso: str | None, failure_class: str = "default") -> float:
    """W = exp(-λ · Δt). Returns 1.0 if no timestamp."""
    if not created_at_iso:
        return 1.0
    try:
        created     = datetime.datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        now         = datetime.datetime.now(datetime.timezone.utc)
        delta_years = (now - created).total_seconds() / (365.25 * 24 * 3600)
        lam         = _LAMBDA.get(failure_class, _LAMBDA["default"])
        return math.exp(-lam * max(0.0, delta_years))
    except Exception:
        return 1.0


class ResearchAdapterRouter(AbstractAdapterRouter):
    """
    Two-pass adapter router for Snath Research.

    Args:
        adapter_dir: Directory with .json centroid caches and .pt LoRA files.
        tau_sim:     Cosine similarity threshold for System 1 centroid match.
        min_trust:   Temporal trust floor W for System 2 LoRA injection.
    """

    def __init__(
        self,
        adapter_dir: str   = "models/adapters",
        tau_sim:     float = 0.90,
        min_trust:   float = 0.40,
        verbose:     bool  = False,
    ):
        self.adapter_dir = Path(adapter_dir)
        self.tau_sim     = tau_sim
        self.min_trust   = min_trust
        self.verbose     = verbose
        self._centroids: list[dict] = []
        self._load_all()

    def _load_all(self) -> None:
        self._centroids = []
        for fp in glob.glob(str(self.adapter_dir / "*.json")):
            try:
                data = json.loads(Path(fp).read_text())
                if _verify_json(data):
                    self._centroids.append(data)
                elif self.verbose:
                    print(f"[ResearchAdapterRouter] HMAC FAIL — skipped {fp}")
            except Exception as e:
                if self.verbose:
                    print(f"[ResearchAdapterRouter] load error {fp}: {e}")

    def refresh(self) -> None:
        self._load_all()

    def available(self) -> list[str]:
        return [c.get("failure_class", "?") for c in self._centroids]

    def _nearest(self, delta: np.ndarray) -> Optional[dict]:
        """
        System 1 — trust-invariant centroid match.

        No temporal gate. The geometric fingerprint of a failure class
        in claims–evidence Δ-space is durable. Identification fires
        regardless of how old the paired .pt adapter is.
        """
        best, best_s = None, self.tau_sim
        for c in self._centroids:
            c_claims  = c.get("centroid_claims",  [])
            c_reviews = c.get("centroid_reviews", [])
            if not c_claims or not c_reviews:
                continue
            centroid_delta = np.array(c_claims) - np.array(c_reviews)
            # Truncate to min length (fingerprint stored at 64 dims)
            n = min(len(delta), len(centroid_delta))
            s = _cos(delta[:n], centroid_delta[:n])
            if s >= best_s:
                best, best_s = c, s
        return best

    def resolve(
        self,
        z_claims:       np.ndarray,
        z_reviews:      np.ndarray,
        base_decision:  RouteDecision,
        conf_claims:    float,
        conf_reviews:   float,
        enc_claims:     object = None,
        enc_reviews:    object = None,
    ) -> tuple[RouteDecision, str]:
        """
        System 1 + System 2 combined inference.

        Args:
            z_claims / z_reviews:  Latent vectors from the two encoders.
            base_decision:         Decision from DivergenceRouter.
            conf_claims / conf_reviews: Encoder confidence scalars.
            enc_claims / enc_reviews:   Live encoder objects (optional).
                Pass to enable System 2 LoRA injection inside resolve().
        Returns:
            (decision, audit_note)
        """
        if base_decision != RouteDecision.TRIGGER_REPLAN:
            return base_decision, "base decision accepted"

        z_a   = np.asarray(z_claims,  float)
        z_b   = np.asarray(z_reviews, float)
        delta = z_a - z_b

        # ── SYSTEM 1: Fast centroid match (trust-invariant) ───────────────
        match = self._nearest(delta)
        if match is None:
            # Cache miss — unknown failure pattern, flag for investigation
            # This should escalate to STRUCTURAL_IMPASSE but is currently
            # TRIGGER_REPLAN — known gap, mirrors Snath Aviation/Robotics.
            return base_decision, "no matching pattern — flag for expert review"

        failure_class = match.get("failure_class", "unknown")
        winner        = match.get("winner",    "unknown")
        win_rate      = match.get("win_rate",  0.0)
        n_events      = match.get("n_events",  0)

        decision = RouteDecision.COMMIT_TRAJECTORY
        note = (f"[System 1] pattern [{failure_class}] n={n_events} "
                f"winner={winner} win_rate={win_rate:.0%}")

        # ── SYSTEM 2: LoRA injection (perishable, trust-gated) ───────────
        # target_encoder = the FAULTY encoder (loser). Read from .pt metadata.
        pt_path = self.adapter_dir / f"{failure_class}.pt"

        if pt_path.exists():
            try:
                import torch as _torch
                meta = _torch.load(str(pt_path), map_location="cpu",
                                   weights_only=False)
                if not _verify_pt(meta):
                    note += " | [System 2] .pt HMAC FAIL — refused"
                else:
                    W = _decay_weight(meta.get("created_at"),
                                      meta.get("failure_class", "default"))
                    target_enc_name = meta.get("target_encoder", "")
                    target_enc_obj  = enc_claims if target_enc_name == "claims" else enc_reviews
                    if W >= self.min_trust:
                        if target_enc_obj is not None and hasattr(target_enc_obj, "load_lora"):
                            target_enc_obj.load_lora(str(pt_path))
                            note += (f" | [System 2] LoRA → '{target_enc_name}' encoder "
                                     f"W={W:.2f} (injected)")
                        else:
                            note += f" | [System 2] W={W:.2f} — System 2 ready"
                    else:
                        note += (f" | [System 2] STALE adapter refused "
                                 f"(W={W:.2f} < {self.min_trust}) — System 1 only")
            except Exception as exc:
                note += f" | [System 2] load error: {exc}"

        return decision, note
