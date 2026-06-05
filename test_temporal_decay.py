"""
Temporal decay regression tests — Snath Research.

Tests the W = exp(-λ · Δt) gate across all three failure classes and
confirms the identification/correction trust asymmetry:
  - System 1 centroid match fires regardless of adapter age.
  - System 2 LoRA injection is refused when W < min_trust.

Identical test structure to Snath Robotics / Aviation for cross-domain
consistency verification.
"""
import math
import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmn.adapter_router import _decay_weight, _LAMBDA, ResearchAdapterRouter
from core.types import RouteDecision


def _iso(years_ago: float) -> str:
    dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=years_ago * 365.25
    )
    return dt.isoformat()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_scope_overclaim_fast_decay():
    """Conference-specific patterns age quickly: W < 0.40 after ~1.8 years."""
    W_fresh = _decay_weight(_iso(0.0), "scope_overclaim")
    W_stale = _decay_weight(_iso(2.0), "scope_overclaim")
    assert W_fresh > 0.99, f"fresh adapter should be trusted: {W_fresh}"
    assert W_stale < 0.40, f"2-year-old scope_overclaim adapter should be stale: {W_stale}"


def test_methodology_gap_slow_decay():
    """Fundamental methodology gaps are durable: W > 0.90 after 5 years."""
    W_5yr = _decay_weight(_iso(5.0), "methodology_gap")
    assert W_5yr > 0.90, f"methodology_gap adapter should still be trusted after 5yr: {W_5yr}"


def test_statistical_weakness_medium_decay():
    """Statistical weakness adapters: intermediate decay (λ=0.20)."""
    W_1yr    = _decay_weight(_iso(1.0), "statistical_weakness")
    expected = math.exp(-0.20 * 1.0)
    assert abs(W_1yr - expected) < 1e-4, f"W mismatch: {W_1yr} vs {expected}"


def test_missing_timestamp_returns_one():
    """No created_at → W = 1.0 (treat as freshly minted)."""
    assert _decay_weight(None) == 1.0
    assert _decay_weight("")   == 1.0


def test_min_trust_floor():
    """min_trust=0.40 is the injection gate — same as all Snath repos."""
    W_just_above = _decay_weight(_iso(1.3), "scope_overclaim")
    W_just_below = _decay_weight(_iso(1.9), "scope_overclaim")
    assert W_just_above > 0.40, "adapter just above threshold should be injectable"
    assert W_just_below < 0.40, "adapter just below threshold should be refused"


def test_system1_trust_invariant():
    """
    System 1 centroid match fires regardless of adapter age.
    ResearchAdapterRouter._nearest() carries NO temporal gate.
    Only resolve() checks W before System 2 injection.
    """
    import inspect
    src = inspect.getsource(ResearchAdapterRouter._nearest)
    assert "_decay_weight" not in src, (
        "System 1 must be trust-invariant — _decay_weight must NOT appear in "
        "_nearest(). The temporal gate belongs in resolve() only."
    )


def test_system2_refuses_stale():
    """
    resolve() returns a STALE note when W < min_trust and does NOT call
    load_lora() on the encoder.
    """
    import json, tempfile, pathlib

    class _MockEnc:
        def __init__(self): self.lora_loaded = False
        def load_lora(self, _): self.lora_loaded = True

    with tempfile.TemporaryDirectory() as td:
        import hmac as _hmac, hashlib
        stale_ts = _iso(2.0)
        centroid = [0.1, 0.2, 0.3]
        _KEY = b"snath_research_adapter_sovereignty_2026"
        immutable = {
            "failure_class":    "scope_overclaim",
            "centroid_claims":  centroid,
            "centroid_reviews": [0.0, 0.0, 0.0],
            "winner":           "reviews",
            "win_rate":         0.85,
            "n_events":         12,
        }
        sig = _hmac.new(_KEY, json.dumps(immutable, sort_keys=True).encode(),
                        hashlib.sha256).hexdigest()
        cjson = {**immutable, "created_at": stale_ts, "sig": sig}
        pathlib.Path(td, "scope_overclaim.json").write_text(json.dumps(cjson))

        import torch
        A = torch.zeros(3, 1)
        B = torch.zeros(1, 3)
        a_hash = hashlib.sha256(A.numpy().tobytes()).hexdigest()[:16]
        b_hash = hashlib.sha256(B.numpy().tobytes()).hexdigest()[:16]
        pt_sig = _hmac.new(_KEY,
            f"scope_overclaim|reviews|{a_hash}|{b_hash}".encode(),
            hashlib.sha256).hexdigest()
        pt_payload = {
            "A": A, "B": B,
            "target_encoder": "reviews",
            "failure_class":  "scope_overclaim",
            "created_at":     stale_ts,
            "hmac_hex":       pt_sig,
        }
        torch.save(pt_payload, os.path.join(td, "scope_overclaim.pt"))

        enc_c = _MockEnc()
        enc_r = _MockEnc()
        ar    = ResearchAdapterRouter(adapter_dir=td, tau_sim=0.0, min_trust=0.40)

        import numpy as np
        _, note = ar.resolve(
            z_claims=np.array(centroid),
            z_reviews=np.array([0.0, 0.0, 0.0]),
            base_decision=RouteDecision.TRIGGER_REPLAN,
            conf_claims=0.8,
            conf_reviews=0.8,
            enc_claims=enc_c,
            enc_reviews=enc_r,
        )
        assert "STALE" in note or "System 1 only" in note, f"Expected stale note: {note}"
        assert not enc_c.lora_loaded, "load_lora must NOT be called for stale adapter"
        assert not enc_r.lora_loaded, "load_lora must NOT be called for stale adapter"


def test_sigreg_reduces_covariance():
    """
    SIGRegLoss with lambda_iso > 0 should improve isotropy when used as a
    training objective on an anisotropic embedding batch.

    This verifies that finetune_projection() has the right effect:
    starting from a heavily skewed distribution (one dim dominates),
    gradient descent via SIGReg + variance penalty should raise the
    isotropy_ratio (min/max eigenvalue of the covariance matrix).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from dmn.sigreg import SIGRegLoss

    # Construct a heavily anisotropic embedding: dim 0 has 30× the variance
    # of all other dims (exactly the failure mode SIGReg is designed to fix).
    torch.manual_seed(0)
    N, D = 30, 8
    z_aniso = torch.zeros(N, D)
    z_aniso[:, 0] = torch.randn(N) * 3.0
    z_aniso[:, 1:] = torch.randn(N, D - 1) * 0.1

    sigreg = SIGRegLoss(lambda_iso=1.0)
    iso_before = sigreg.isotropy_ratio(z_aniso)

    # Fine-tune a linear projection to reduce the covariance.
    proj = nn.Linear(D, D, bias=False)
    nn.init.eye_(proj.weight)
    opt  = torch.optim.AdamW(proj.parameters(), lr=1e-2)

    for _ in range(300):
        opt.zero_grad()
        z = F.normalize(proj(z_aniso), dim=-1)
        loss = sigreg(z) + torch.relu(0.1 - z.std(dim=0)).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        iso_after = sigreg.isotropy_ratio(F.normalize(proj(z_aniso), dim=-1))

    assert iso_after > iso_before, (
        f"SIGReg must improve isotropy: before={iso_before:.4f} after={iso_after:.4f}"
    )
    # Relative improvement check — L2 normalization constrains outputs to the unit
    # sphere, limiting the absolute isotropy ratio achievable from extreme anisotropy.
    # The important invariant is that SIGReg meaningfully improves the ratio, not
    # that it reaches a specific absolute value.
    assert iso_after >= iso_before * 2.0, (
        f"SIGReg should at least double the isotropy ratio: "
        f"before={iso_before:.6f} after={iso_after:.6f}"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_scope_overclaim_fast_decay,
        test_methodology_gap_slow_decay,
        test_statistical_weakness_medium_decay,
        test_missing_timestamp_returns_one,
        test_min_trust_floor,
        test_system1_trust_invariant,
        test_system2_refuses_stale,
        test_sigreg_reduces_covariance,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
