"""
DivergenceRouter — V1–V6 routing for Snath Research.
=====================================================
Measures total-variation distance between the claims latent vector
(what a paper asserts) and the evidence latent vector (what reviewers
say it demonstrates) and routes to one of four decisions:

    COMMIT_TRAJECTORY   — claims and evidence agree; paper is coherent
    TRIGGER_REPLAN      — recoverable gap; load adapter and re-assess
    STRUCTURAL_IMPASSE  — irreconcilable gap; flag for expert review
    DEFER               — one stream uncertain; lean on the confident one

V1–V6 invariants (AbstractDivergenceRouter, github.com/snath-ai/Lar-JEPA):
  V1  Both streams present at every call.
  V2  Divergence computed from normalised probability vectors only.
  V3  Decision is a pure function of (D, conf_claims, conf_reviews).
  V4  Content-blind: route() never reads z_claims or z_reviews directly.
      It operates on scalar D and confidence values only.
  V5  STRUCTURAL_IMPASSE is always reachable regardless of content.
  V6  COMMIT_TRAJECTORY only returned when D < τ_low AND both conf ≥ τ_low.

Divergence metric
-----------------
    p_a = softmax(z_claims)    # (G,) probability vector over concept dims
    p_b = softmax(z_reviews)   # (G,) probability vector over concept dims
    D   = ||p_a - p_b||₁ / √G

The same formula, identical to Snath Basis / Aviation / Robotics.
Division by √G normalises across embedding dimensions.

Domain semantics
----------------
High D + both confident = the abstract is making strong claims AND the
reviews are telling a different story. This is the overclaiming signal.
Not noise. Not to be averaged. To be flagged and investigated.

Derivative Works note
---------------------
This file is a Derivative Work of AbstractDivergenceRouter (V1–V6),
Apache 2.0, github.com/snath-ai/Lar-JEPA.
"""
import math
import datetime
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.types import RouteDecision
from dhard import DHardQueue, ResearchDHardEvent


@dataclass
class RoutingResult:
    decision:       RouteDecision
    divergence:     float
    delta:          torch.Tensor   # (G,) probability-space delta
    conf_claims:    float
    conf_reviews:   float
    failure_class:  str
    note:           str


class DivergenceRouter:
    """
    V1–V6 routing contract for scientific paper claims vs evidence.

    Args:
        tau_high: Divergence above which STRUCTURAL_IMPASSE fires.
        tau_low:  Divergence below which COMMIT_TRAJECTORY is safe.
        delta:    Minimum divergence for a D_hard logging event.
        dhard:    Optional DHardQueue. When provided, qualifying events
                  are written on every TRIGGER_REPLAN / STRUCTURAL_IMPASSE.
    """

    def __init__(
        self,
        tau_high: float = 0.60,
        tau_low:  float = 0.25,
        delta:    float = 0.35,
        dhard:    Optional[DHardQueue] = None,
    ):
        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.delta    = delta
        self.dhard    = dhard

    def route(
        self,
        z_claims:  torch.Tensor,    # (G,) or (1, G)
        z_reviews: torch.Tensor,    # (G,) or (1, G)
        paper_id:  str = "",
        venue:     str = "",
    ) -> RoutingResult:
        """
        Compute divergence and return a routing decision.

        V4 — content-blind: this method never branches on the values
        inside z_claims or z_reviews. Only scalar D and confidence values
        determine the routing outcome.
        """
        z_a = z_claims.flatten()
        z_b = z_reviews.flatten()
        G   = z_a.shape[0]

        # Probability vectors (V2)
        p_a = F.softmax(z_a, dim=0)
        p_b = F.softmax(z_b, dim=0)

        # Total variation distance, normalised by √G
        delta_vec = p_a - p_b
        D = float(delta_vec.abs().sum() / math.sqrt(G))

        # Confidence: peakedness of the softmax distribution.
        # (max(p) - 1/G) / (1 - 1/G) — 0 when uniform, 1 when fully peaked.
        # Correct for concept distributions; sigmoid-mean is always constant
        # for softmax outputs since mean(p) = 1/G regardless of peakedness.
        conf_a = float(max(0.0, (float(p_a.max()) - 1.0/G) / (1.0 - 1.0/G)))
        conf_b = float(max(0.0, (float(p_b.max()) - 1.0/G) / (1.0 - 1.0/G)))

        decision, note = self._decide(D, conf_a, conf_b)

        # Infer failure class for D_hard logging
        failure_class = self._infer_failure_class(conf_a, conf_b, D)

        # Write to D_hard queue if curriculum-worthy
        if (
            self.dhard is not None
            and decision in (RouteDecision.TRIGGER_REPLAN, RouteDecision.STRUCTURAL_IMPASSE)
            and D >= self.delta
        ):
            self.dhard.log(
                paper_id=paper_id,
                venue=venue,
                routing_decision=decision,
                decision_basis=D,
                conf_claims=conf_a,
                conf_reviews=conf_b,
                v_claims=z_a[:64].tolist(),    # store first 64 dims as fingerprint
                v_reviews=z_b[:64].tolist(),
                failure_class=failure_class,
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            )

        return RoutingResult(
            decision=decision,
            divergence=D,
            delta=delta_vec,
            conf_claims=conf_a,
            conf_reviews=conf_b,
            failure_class=failure_class,
            note=note,
        )

    def _decide(self, D: float, conf_a: float, conf_b: float) -> Tuple[RouteDecision, str]:
        """Pure routing function — V3: function of scalars only."""
        both_confident = (conf_a >= self.tau_low) and (conf_b >= self.tau_low)

        if D < self.tau_low and both_confident:
            return (
                RouteDecision.COMMIT_TRAJECTORY,
                f"claims and evidence agree (D={D:.3f} < τ_low={self.tau_low})",
            )

        if not both_confident:
            weak = "claims" if conf_a < conf_b else "reviews"
            return (
                RouteDecision.DEFER,
                f"stream '{weak}' uncertain (conf_claims={conf_a:.2f}, "
                f"conf_reviews={conf_b:.2f}) — deferring to confident stream",
            )

        if D >= self.tau_high:
            return (
                RouteDecision.STRUCTURAL_IMPASSE,
                f"irreconcilable gap — abstract vs reviewers "
                f"(D={D:.3f} ≥ τ_high={self.tau_high}) — flag for expert review",
            )

        return (
            RouteDecision.TRIGGER_REPLAN,
            f"recoverable gap — load adapter and re-assess "
            f"(τ_low={self.tau_low} ≤ D={D:.3f} < τ_high={self.tau_high})",
        )

    def _infer_failure_class(self, conf_a: float, conf_b: float, D: float) -> str:
        """
        Heuristic failure class for D_hard logging.

        High claims confidence + moderate reviews = overclaiming abstract.
        Low reviews confidence = ambiguous/inconsistent reviewer signals.
        Both confident + very high D = deep methodology gap.
        """
        if conf_a >= 0.70 and conf_b < 0.50:
            return "scope_overclaim"
        if D >= self.tau_high:
            return "methodology_gap"
        return "statistical_weakness"
