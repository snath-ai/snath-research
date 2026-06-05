"""
Snath Research — the D_hard curriculum.

Logs each routing-hard paper (TRIGGER_REPLAN / STRUCTURAL_IMPASSE) to a
signed JSONL queue. When OpenReview decisions are known, verdicts are
attached via attach_verdicts(). The DMN consolidation cycle reads labelled
events and trains LoRA adapters.

Failure classes
---------------
scope_overclaim      — abstract claims exceed what experiments demonstrate.
                       Fast decay λ=0.50 (conference-specific, venue shifts)
statistical_weakness — sample sizes / statistical power insufficient for claims.
                       Medium decay λ=0.20
methodology_gap      — fundamental mismatch between experimental design and
                       claimed conclusions. Slow decay λ=0.02 (persists)

Winner field
------------
"claims"  — the abstract correctly represented the paper's strength;
            reviewers validated the claims. Claims stream was trustworthy.
"reviews" — reviewers correctly identified that the evidence was insufficient;
            the abstract was overclaiming. Evidence stream was trustworthy.
None      — verdict ambiguous (borderline accept/reject, off-topic rejection).

HMAC
----
Immutable fields are signed at log time. Verdict fields (winner, decision,
avg_score) are added later and are NOT covered by the HMAC — only the
original routing snapshot is authenticated.
"""
from dataclasses import dataclass, asdict, field
import json
import hmac
import hashlib
from pathlib import Path
from typing import Optional, List

_DHARD_KEY = b"snath_research_dhard_2026"

FAILURE_CLASSES = (
    "scope_overclaim",
    "statistical_weakness",
    "methodology_gap",
)


@dataclass
class ResearchDHardEvent:
    """One routing-hard paper event."""
    # Immutable — covered by HMAC
    paper_id:       str
    venue:          str               # e.g. "ICLR2024", "NeurIPS2023"
    decision_basis: float             # divergence D at routing time
    conf_claims:    float             # confidence of claims stream
    conf_reviews:   float             # confidence of evidence/review stream
    v_claims:       List[float]       # z_claims latent snapshot (sampled dims)
    v_reviews:      List[float]       # z_reviews latent snapshot (sampled dims)
    routing_decision: str             # TRIGGER_REPLAN or STRUCTURAL_IMPASSE
    failure_class:  str               # scope_overclaim / statistical_weakness / methodology_gap
    timestamp:      str

    # Mutable — filled by attach_verdicts(), NOT covered by HMAC
    or_decision:    Optional[str]   = None   # OpenReview: "Accept" / "Reject" / "Withdrawn"
    avg_score:      Optional[float] = None   # mean reviewer numerical rating
    winner:         Optional[str]   = None   # "claims" / "reviews" / None

    sig: str = ""

    _IMMUTABLE = (
        "paper_id", "venue", "decision_basis", "conf_claims", "conf_reviews",
        "v_claims", "v_reviews", "routing_decision", "failure_class", "timestamp",
    )

    def _payload(self) -> bytes:
        return json.dumps(
            {k: getattr(self, k) for k in self._IMMUTABLE}, sort_keys=True
        ).encode()

    def sign(self) -> "ResearchDHardEvent":
        self.sig = hmac.new(_DHARD_KEY, self._payload(), hashlib.sha256).hexdigest()
        return self

    def verify(self) -> bool:
        expected = hmac.new(_DHARD_KEY, self._payload(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.sig, expected)


class DHardQueue:
    """
    Append-only HMAC-signed JSONL queue of routing-hard paper events.

    Usage:
        q = DHardQueue("d_hard.jsonl")
        ev = q.log(paper_id="abc123", venue="ICLR2024", ...)
        # ... later, after OpenReview decision is known ...
        q.attach_verdicts({"abc123": {"decision": "Reject", "avg_score": 3.5}})
        labelled = q.resolved()
    """

    def __init__(self, path: str = "d_hard.jsonl"):
        self.path = Path(path)

    def log(
        self,
        paper_id: str,
        venue: str,
        routing_decision,
        decision_basis: float,
        conf_claims: float,
        conf_reviews: float,
        v_claims: List[float],
        v_reviews: List[float],
        failure_class: str,
        timestamp: str,
    ) -> Optional[ResearchDHardEvent]:
        """
        Log a routing-hard event. Only TRIGGER_REPLAN and STRUCTURAL_IMPASSE
        events are curriculum-worthy — COMMIT and DEFER are not logged.
        """
        dec = routing_decision.value if hasattr(routing_decision, "value") else str(routing_decision)
        if dec not in ("TRIGGER_REPLAN", "STRUCTURAL_IMPASSE"):
            return None

        ev = ResearchDHardEvent(
            paper_id=paper_id,
            venue=venue,
            decision_basis=round(float(decision_basis), 6),
            conf_claims=round(float(conf_claims), 4),
            conf_reviews=round(float(conf_reviews), 4),
            v_claims=[round(float(x), 6) for x in v_claims],
            v_reviews=[round(float(x), 6) for x in v_reviews],
            routing_decision=dec,
            failure_class=failure_class,
            timestamp=timestamp,
        ).sign()

        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(ev)) + "\n")
        return ev

    def all(self) -> List[ResearchDHardEvent]:
        """Read all events from the queue."""
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                events.append(ResearchDHardEvent(**json.loads(line)))
        return events

    def resolved(self) -> List[ResearchDHardEvent]:
        """Events with a winner label — ready for DMN consolidation."""
        return [e for e in self.all() if e.winner is not None]

    def attach_verdicts(self, verdicts: dict) -> int:
        """
        Fill in OpenReview outcomes for logged events.

        Args:
            verdicts: dict mapping paper_id → {
                "decision":  "Accept" | "Reject" | "Withdrawn",
                "avg_score": float (mean reviewer numerical rating),
                "winner":    "claims" | "reviews" | None
            }
            Winner can be provided directly, or is inferred automatically:
              - Reject + avg_score <= 4.0  → winner = "reviews"
              - Accept + avg_score >= 6.0  → winner = "claims"
              - Otherwise                  → winner = None (ambiguous)

        Returns:
            Number of events updated.
        """
        events = self.all()
        updated = 0
        for ev in events:
            if ev.paper_id in verdicts:
                v = verdicts[ev.paper_id]
                ev.or_decision = v.get("decision")
                ev.avg_score   = v.get("avg_score")
                # Use provided winner or infer from decision + score
                if "winner" in v and v["winner"] is not None:
                    ev.winner = v["winner"]
                elif ev.or_decision == "Reject" and ev.avg_score is not None and ev.avg_score <= 4.0:
                    ev.winner = "reviews"
                elif ev.or_decision == "Accept" and ev.avg_score is not None and ev.avg_score >= 6.0:
                    ev.winner = "claims"
                else:
                    ev.winner = None
                updated += 1

        # Rewrite the whole file with updated events
        with open(self.path, "w") as f:
            for ev in events:
                f.write(json.dumps(asdict(ev)) + "\n")
        return updated

    def stats(self) -> dict:
        events = self.all()
        resolved = self.resolved()
        return {
            "total":    len(events),
            "resolved": len(resolved),
            "pending":  len(events) - len(resolved),
            "by_class": {
                fc: sum(1 for e in events if e.failure_class == fc)
                for fc in FAILURE_CLASSES
            },
            "by_winner": {
                "claims":  sum(1 for e in resolved if e.winner == "claims"),
                "reviews": sum(1 for e in resolved if e.winner == "reviews"),
            },
        }

    def clear(self):
        self.path.unlink(missing_ok=True)
