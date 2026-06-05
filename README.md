<div align="center">

# Snath Research
### Multi-Stream Divergence Routing for Scientific Claim Verification

**The fifth domain instantiation of the Lár-JEPA cognitive routing architecture.**

<p align="center">
  <a href="https://github.com/snath-ai/Lar-JEPA">
    <img alt="Architecture" src="https://img.shields.io/badge/Architecture-Lár--JEPA%20V1--V6-blueviolet?style=for-the-badge">
  </a>
  <a href="https://doi.org/10.5281/zenodo.20278775">
    <img alt="UCR" src="https://img.shields.io/badge/UCR-doi%3A10.5281%2Fzenodo.20278775-blue?style=for-the-badge">
  </a>
  <a href="https://doi.org/10.5281/zenodo.20278781">
    <img alt="DAS" src="https://img.shields.io/badge/DAS-doi%3A10.5281%2Fzenodo.20278781-blue?style=for-the-badge">
  </a>
</p>

</div>

---

Science has a replication problem. Papers make strong claims. The evidence doesn't always support them. Peer reviewers catch some of it. A lot gets through.

**Snath Research routes on the gap.**

Stream A encodes what a paper *claims*. Stream B encodes what independent reviewers say it *actually demonstrates*. A content-blind divergence router measures the distance between those two positions in concept space. High-confidence disagreement is not noise — it is the signal.

---

## The Two Streams

| | Stream A | Stream B |
|---|---|---|
| **What it encodes** | Abstract + author summary | Aggregated peer review text |
| **Question it answers** | What does this paper claim to prove? | What do independent experts say it actually showed? |
| **Encoder** | `AbstractClaimsEncoder` (SciBERT) | `EvidenceEncoder` (SciBERT) |
| **Independence** | Never reads Stream B output (M1) | Never reads Stream A output (M1) |

The two streams are structurally independent: the abstract is written by authors to impress; the reviews are written by experts to assess. They have different intents, different authors, and different access to the paper. The divergence between them carries real information.

---

## The Four Routing Outcomes

```
D = ||softmax(z_claims) − softmax(z_reviews)||₁ / √G

D < τ_low,  both confident  →  COMMIT_TRAJECTORY    Claims and evidence agree. Proceed.
τ_low ≤ D < τ_high          →  TRIGGER_REPLAN       Recoverable gap. Load adapter, re-assess.
D ≥ τ_high                  →  STRUCTURAL_IMPASSE   Irreconcilable. Flag for expert review.
One stream uncertain         →  DEFER                Lean on the confident stream.
```

The router never reads the content of the abstract or reviews — only the scalar divergence and confidence values. This is V4 (Content Blindness): the routing decision cannot be gamed by writing a clever abstract.

---

## The Learning Loop

Every TRIGGER_REPLAN event is logged to the D_hard queue (HMAC-signed JSONL). When OpenReview decisions arrive, the ground truth verdict is attached automatically:

- Paper **rejected** with low reviewer scores → `winner = "reviews"` (evidence stream was right)
- Paper **accepted** with high reviewer scores → `winner = "claims"` (claims were validated)

The overnight DMN consolidation cycle reads labelled events, clusters them by failure class, and trains signed LoRA adapters:

```
D_hard events
    → cluster by failure_class (scope_overclaim / statistical_weakness / methodology_gap)
    → System 1: JSON centroid fingerprint (trust-invariant — geometric signatures are durable)
    → System 2: Rank-4 LoRA adapter (perishable — gated by W = exp(−λ · Δt) ≥ 0.40)
```

After consolidation, the adapter router corrects the encoder that was systematically wrong. Future papers matching the same divergence pattern are routed more accurately.

**This is the Safety-Learning Equivalence** ([DAS, doi:10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)): the same events that constitute routing failures are the training curriculum for the adapters that prevent future failures.

---

## Failure Classes and Temporal Decay

| Failure class | λ | Half-life | Meaning |
|---|---|---|---|
| `scope_overclaim` | 0.50 | 1.4 yr | Abstract claims exceed what experiments demonstrate. Fast decay — conference norms shift. |
| `statistical_weakness` | 0.20 | 3.5 yr | Insufficient statistical power for stated conclusions. Medium decay. |
| `methodology_gap` | 0.02 | 34.7 yr | Fundamental mismatch between experimental design and claimed conclusions. Persistent. |

Same formula as all Snath repos: `W = exp(−λ · Δt)`, `W_min = 0.40`.

---

## Running the Experiment

**Step 1 — Smoke test (no internet, synthetic data, instant):**
```bash
pip install -r requirements.txt
python experiments/run_experiment.py --smoke-test
```

**Step 2 — Run the tests:**
```bash
python test_temporal_decay.py
# 7/7 passed
```

**Step 3 — Full experiment on ICLR 2024 (requires internet, ~30 min):**
```bash
python experiments/run_experiment.py \
    --venue ICLR.cc/2024/Conference \
    --max-papers 500
```

**Step 4 — From cached data (after first run):**
```bash
python experiments/run_experiment.py \
    --from-cache data/ICLR.cc_2024_Conference_cache.jsonl
```

The experiment reports AUROC before and after adapter injection, measured against OpenReview ground truth decisions.

---

## Domain Isomorphism

The routing code in this repository — `DivergenceRouter`, `DHardQueue`, `ResearchDMN`, `ResearchAdapterRouter` — is structurally identical to the four prior Snath instantiations. Only the encoder constructors, failure-class labels, and λ constants differ.

| Domain | Stream A | Stream B | Ground truth |
|---|---|---|---|
| [Snath Basis](https://github.com/snath-ai/snath-basis) | Fundamental analysis | Market signals | Realised returns |
| [Snath Aviation](https://github.com/snath-ai/snath-aviation) | Radar altimeter | Pitot tube | Flight outcome |
| [Snath Robotics](https://github.com/snath-ai/snath-robotics) | Vision | Proprioception | Sensor ground truth |
| **Snath Research** ← you are here | Paper claims | Peer reviews | OpenReview decisions |

The same mathematical spine — TV/√G divergence, temporal trust gate, System 1/2 asymmetry, HMAC-signed adapter chain — governs scientific publishing, financial markets, aviation safety, humanoid robotics, and drug discovery without modification to any routing primitive.

---

## Companion Papers

| Paper | DOI |
|---|---|
| Universal Cognitive Routing (UCR) — 10 ABCs, 33 invariants | [10.5281/zenodo.20278775](https://doi.org/10.5281/zenodo.20278775) |
| Divergence Is Not Noise (DAS) — empirical validation, Safety-Learning Equivalence | [10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781) |
| Architecture Is All You Need (AIA) — pre-registration of training loop experiments | [10.5281/zenodo.20419182](https://doi.org/10.5281/zenodo.20419182) |
| Snath Robotics — humanoid sensor routing | [10.5281/zenodo.20517446](https://doi.org/10.5281/zenodo.20517446) |

---

## License

Apache 2.0. Developed by Aadithya Vishnu Sajeev ([ORCID: 0009-0009-3916-0988](https://orcid.org/0009-0009-3916-0988)) under the Snath AI Open Source Research Initiative. Developed on personal hardware, outside employment.
