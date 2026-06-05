"""
ResearchDMN — overnight consolidation cycle for Snath Research.
===============================================================
Reads the D_hard queue of routing-hard paper events, clusters them by
failure class, and generates signed LoRA adapters stored in models/adapters/.

System 1 (fast JSON centroid cache) — identifies which failure class a new
paper's divergence vector matches. Trust-invariant: the geometric fingerprint
of "scope_overclaim" in latent space is durable across conference years.

System 2 (LoRA .pt) — corrects the encoder that was systematically wrong
in each cluster. Perishable: review patterns at NeurIPS 2022 may differ from
NeurIPS 2026. Gated by W = exp(-λ · Δt), λ from config.json.

SIGReg — lambda_iso=0.0 by default (inert until AIA Experiment 3).

Usage:
    python -m dmn.research_dmn --run-cycle
"""
from __future__ import annotations

import os
import json
import hmac as _hmac
import hashlib
import datetime
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhard import DHardQueue, ResearchDHardEvent
from dmn.sigreg import SIGRegLoss

_ADAPTER_KEY = b"snath_research_adapter_sovereignty_2026"
_MIN_EVENTS  = 4


class ResearchDMN:
    """
    Default Mode Network for Snath Research.

    Nightly consolidation: D_hard events → System 1 JSON centroids
    + System 2 LoRA adapters.
    """

    def __init__(
        self,
        queue_path:  str = "d_hard.jsonl",
        adapter_dir: str = "models/adapters",
    ):
        self.queue       = DHardQueue(queue_path)
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

    def consolidate(
        self,
        min_events:  int   = _MIN_EVENTS,
        n_epochs:    int   = 150,
        lr:          float = 0.1,
        lambda_iso:  float = 0.0,
        verbose:     bool  = True,
    ) -> List[dict]:
        """
        Cluster resolved D_hard events → generate signed adapters.

        Returns list of adapter metadata dicts.
        """
        events   = self.queue.resolved()
        by_class = defaultdict(list)
        for e in events:
            by_class[e.failure_class].append(e)

        built  = []
        sigreg = SIGRegLoss(lambda_iso=lambda_iso)

        for failure_class, group in sorted(by_class.items()):
            if len(group) < min_events:
                if verbose:
                    print(f"  · {failure_class:<28} {len(group)} events "
                          f"— too few (need {min_events}), skipped")
                continue

            dim = len(group[0].v_claims)

            # ── SYSTEM 1: JSON centroid ───────────────────────────────────
            centroid_claims  = [
                round(sum(e.v_claims[i]  for e in group) / len(group), 6)
                for i in range(dim)
            ]
            centroid_reviews = [
                round(sum(e.v_reviews[i] for e in group) / len(group), 6)
                for i in range(dim)
            ]
            winner_counts: dict = {}
            for e in group:
                if e.winner:
                    winner_counts[e.winner] = winner_counts.get(e.winner, 0) + 1
            winner   = max(winner_counts, key=winner_counts.get) if winner_counts else "unknown"
            win_rate = round(winner_counts.get(winner, 0) / len(group), 3)

            json_immutable = {
                "failure_class":    failure_class,
                "centroid_claims":  centroid_claims,
                "centroid_reviews": centroid_reviews,
                "winner":           winner,
                "win_rate":         win_rate,
                "n_events":         len(group),
            }
            json_sig = _hmac.new(
                _ADAPTER_KEY,
                json.dumps(json_immutable, sort_keys=True).encode(),
                hashlib.sha256,
            ).hexdigest()
            json_payload = {
                **json_immutable,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "sig": json_sig,
            }
            json_path = self.adapter_dir / f"{failure_class}.json"
            json_path.write_text(json.dumps(json_payload, indent=2))

            # ── SYSTEM 2: LoRA .pt ───────────────────────────────────────
            # Faulty stream = loser. Target = winner.
            if winner == "claims":
                target_vecs = [e.v_claims  for e in group]
                faulty_vecs = [e.v_reviews for e in group]
                target_enc  = "reviews"
            else:
                target_vecs = [e.v_reviews for e in group]
                faulty_vecs = [e.v_claims  for e in group]
                target_enc  = "claims"

            target_t = torch.tensor(target_vecs, dtype=torch.float32)
            faulty_t = torch.tensor(faulty_vecs, dtype=torch.float32)

            A   = nn.Parameter(torch.randn(dim, 1) * 0.01)
            B   = nn.Parameter(torch.randn(1, dim) * 0.01)
            opt = optim.AdamW([A, B], lr=lr)

            for _ in range(n_epochs):
                opt.zero_grad()
                adapted = faulty_t + torch.matmul(torch.matmul(faulty_t, A), B)
                loss    = torch.nn.functional.l1_loss(adapted, target_t)
                loss    = loss + sigreg(adapted)
                loss.backward()
                opt.step()

            # HMAC sign
            a_hash = hashlib.sha256(A.detach().numpy().tobytes()).hexdigest()[:16]
            b_hash = hashlib.sha256(B.detach().numpy().tobytes()).hexdigest()[:16]
            sig = _hmac.new(
                _ADAPTER_KEY,
                f"{failure_class}|{target_enc}|{a_hash}|{b_hash}".encode(),
                hashlib.sha256,
            ).hexdigest()

            pt_payload = {
                "A":              A.detach(),
                "B":              B.detach(),
                "target_encoder": target_enc,
                "failure_class":  failure_class,
                "created_at":     datetime.datetime.utcnow().isoformat() + "Z",
                "n_events":       len(group),
                "win_rate":       win_rate,
                "final_loss":     round(float(loss.item()), 6),
                "hmac_hex":       sig,
            }
            pt_path = self.adapter_dir / f"{failure_class}.pt"
            torch.save(pt_payload, str(pt_path))

            meta = {**json_payload, "pt_path": str(pt_path),
                    "final_loss": pt_payload["final_loss"]}
            built.append(meta)

            if verbose:
                print(f"  ✓ {failure_class:<28} n={len(group):<3} "
                      f"winner={winner:<8} win_rate={win_rate:<5} "
                      f"loss={loss.item():.4f}")

        return built

    def stats(self) -> dict:
        return self.queue.stats()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ResearchDMN — D_hard → LoRA consolidation"
    )
    parser.add_argument("--run-cycle",   action="store_true")
    parser.add_argument("--queue-path",  default="d_hard.jsonl")
    parser.add_argument("--adapter-dir", default="models/adapters")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--lambda-iso",  type=float, default=0.0)
    args = parser.parse_args()

    if args.run_cycle:
        dmn = ResearchDMN(queue_path=args.queue_path, adapter_dir=args.adapter_dir)
        s   = dmn.stats()
        print(f"[ResearchDMN] D_hard: {s['total']} total, {s['resolved']} resolved")
        built = dmn.consolidate(n_epochs=args.epochs, lambda_iso=args.lambda_iso)
        print(f"[ResearchDMN] Built {len(built)} adapter(s).")
