"""
Snath Research — Main Experiment.
==================================
Proves the D_hard → DMN → LoRA learning loop on scientific papers.

The experiment:
  1. Fetch papers + reviews from OpenReview (or load from cache).
  2. Encode both streams with SciBERT (AbstractClaimsEncoder + EvidenceEncoder).
  3. Run divergence router on all papers — collect D_hard events.
  4. Attach OpenReview verdicts (winner labels).
  5. Run DMN consolidation → train LoRA adapters.
  6. Re-run router with adapters injected.
  7. Measure AUROC before vs after. Report.

This is AIA Experiment 2 run on a public domain with an automatic
external oracle. If AUROC improves after adapter injection, the learning
loop works.

Usage:
    # Full run (fetches live data — requires internet, ~30 min):
    python experiments/run_experiment.py --venue ICLR.cc/2024/Conference --max-papers 500

    # From cached data:
    python experiments/run_experiment.py --from-cache data/iclr2024_cache.jsonl

    # Quick smoke test (no internet, synthetic data):
    python experiments/run_experiment.py --smoke-test
"""
from __future__ import annotations

import os
# Prevent Intel OpenMP / NumExpr thread-pool deadlock on macOS M1/M2.
# These must be set BEFORE any torch/numpy/transformers import — the
# dynamic linker reads them at library load time.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]   = "TRUE"
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
os.environ["NUMEXPR_MAX_THREADS"]     = "1"

import sys
import json
import datetime
import argparse
import logging

import numpy as np
import torch
# Limit PyTorch's internal thread pool to 1. This is separate from the
# OMP/MKL env vars above — torch.set_num_threads() controls the ATen
# thread pool which is what actually triggers the mutex.cc race on M1.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.abstract_encoder import AbstractClaimsEncoder
from encoders.evidence_encoder  import EvidenceEncoder
from encoders.tfidf_encoder     import TFIDFEncoderPair
from divergence_router          import DivergenceRouter
from dhard                      import DHardQueue
from dmn.research_dmn           import ResearchDMN
from dmn.adapter_router         import ResearchAdapterRouter
from core.types                 import RouteDecision
from data.openreview_pipeline   import OpenReviewPipeline, PaperRecord, infer_winner

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── AUROC helper ──────────────────────────────────────────────────────────────

def compute_auroc(scores: list[float], labels: list[int]) -> float:
    """
    Simple AUROC from (score, binary_label) pairs.
    label=1 → routing correctly identified the paper as problematic (high D)
    label=0 → paper was coherent (low D expected)
    """
    from itertools import combinations
    if not scores or len(set(labels)) < 2:
        return float("nan")
    pairs = list(zip(scores, labels))
    pos   = [s for s, l in pairs if l == 1]
    neg   = [s for s, l in pairs if l == 0]
    correct = sum(p > n for p in pos for n in neg)
    total   = len(pos) * len(neg)
    return correct / total if total > 0 else float("nan")


# ── Synthetic smoke test data ─────────────────────────────────────────────────

def make_smoke_data(n: int = 40) -> list[PaperRecord]:
    """
    Generate synthetic papers for a smoke test (no internet needed).

    Half are 'overclaiming' (abstract strong, review weak → Reject).
    Half are 'coherent' (abstract + review aligned → Accept).
    """
    rng = np.random.default_rng(42)
    records = []
    for i in range(n):
        overclaiming = i < n // 2
        if overclaiming:
            claims  = ("We demonstrate a revolutionary breakthrough achieving state-of-the-art "
                       "results on all benchmarks with unprecedented improvements across every "
                       f"metric. Paper {i}.")
            reviews = (f"The experimental evaluation is limited. The sample size is insufficient. "
                       f"The comparison baselines are outdated. The claims significantly exceed "
                       f"what the experiments support. Paper {i}.")
            decision  = "Reject"
            avg_score = rng.uniform(2.0, 4.0)
        else:
            claims  = (f"We present an incremental improvement to existing methods with "
                       f"careful empirical evaluation on standard benchmarks. Paper {i}.")
            reviews = (f"The paper is well-written with thorough experiments. The claims are "
                       f"appropriately scoped and supported by the evidence. Paper {i}.")
            decision  = "Accept"
            avg_score = rng.uniform(6.0, 9.0)

        records.append(PaperRecord(
            paper_id=f"smoke_{i:03d}",
            venue="SmokeTest/2024",
            title=f"Synthetic Paper {i}",
            claims_text=claims,
            reviews_text=reviews,
            decision=decision,
            avg_score=float(avg_score),
            num_reviews=3,
        ))
    return records


# ── Main experiment ───────────────────────────────────────────────────────────

def run(args):
    os.makedirs("models/adapters", exist_ok=True)
    queue_path = "d_hard.jsonl"

    # ── 1. Load / fetch papers ────────────────────────────────────────────
    if args.smoke_test:
        log.info("Smoke test mode — synthetic data, no internet needed.")
        papers = make_smoke_data(n=40)
    elif args.from_cache:
        log.info(f"Loading from cache: {args.from_cache}")
        papers = OpenReviewPipeline.load(args.from_cache)
    else:
        pipe   = OpenReviewPipeline(venue=args.venue, sleep_sec=0.3)
        papers = pipe.fetch(max_papers=args.max_papers)
        cache  = f"data/{args.venue.replace('/', '_')}_cache.jsonl"
        os.makedirs("data", exist_ok=True)
        pipe.save(papers, cache)
        log.info(f"Cached to {cache}")

    log.info(f"\nLoaded {len(papers)} papers.")

    # ── 2. Initialise encoders + pre-compute embeddings ──────────────────
    enc_claims  = AbstractClaimsEncoder()
    enc_reviews = EvidenceEncoder()

    bert_cache_claims  = {}   # {paper_id: tensor(embed_dim,)}
    bert_cache_reviews = {}

    if args.smoke_test:
        # Smoke test: generate direct 8-dim concept vectors — no SciBERT,
        # no projection. This proves the routing → D_hard → DMN → LoRA
        # pipeline end-to-end with a guaranteed routing signal.
        #
        # Why 8-dim directly (not 768→8 projection): a random Xavier
        # projection of any 768-dim vector produces near-uniform 8-dim
        # outputs (CLT effect), making softmax(z) ≈ uniform and D ≈ 0
        # everywhere. An untrained projection kills the routing signal
        # regardless of how structured the 768-dim inputs are.
        # The smoke test must prove the pipeline, not that Xavier init works.
        #
        # Overclaiming papers: z_claims peaked at dim (i%4),
        #                       z_reviews peaked at dim (i%4 + 4) — opposite
        #                       half of the concept space → D ≈ 0.50 >> delta
        # Coherent papers:    z_claims ≈ z_reviews peaked at same dim → D ≈ 0
        #
        # Verification: softmax((3,0,0,0,0,0,0,0)) ≈ (0.741, 0.037×7)
        #   confidence = (0.741 − 1/8) / (1 − 1/8) = 0.704 > tau_high=0.60 ✓
        #   D(peak_dim0, peak_dim4) ≈ 0.50 > delta=0.35 ✓  → TRIGGER_REPLAN
        log.info("\nGenerating synthetic 8-dim concept vectors (smoke test — no model needed)...")
        D8 = enc_claims.embed_dim   # 8
        rng = np.random.default_rng(42)
        for i, p in enumerate(papers):
            noise_c = torch.tensor(rng.standard_normal(D8).astype(np.float32) * 0.05)
            noise_r = torch.tensor(rng.standard_normal(D8).astype(np.float32) * 0.05)
            if p.avg_score <= 4.0:              # overclaiming → high D
                c_vec = noise_c.clone()
                c_vec[i % (D8 // 2)] = 3.0
                r_vec = noise_r.clone()
                r_vec[(i % (D8 // 2)) + D8 // 2] = 3.0  # opposite half
            else:                               # coherent → low D
                c_vec = noise_c.clone()
                c_vec[i % D8] = 3.0
                r_vec = noise_r.clone()
                r_vec[i % D8] = 3.0             # same dim
            bert_cache_claims[p.paper_id]  = c_vec
            bert_cache_reviews[p.paper_id] = r_vec
        log.info(f"  {len(bert_cache_claims)} synthetic concept vectors generated ({D8}-dim).")

    elif args.embedding_cache:
        # Pre-computed embeddings from precompute_embeddings.py — no model loading.
        # This is the recommended path on macOS M1 (avoids Intel OpenMP deadlock).
        log.info(f"\nLoading pre-computed embeddings from {args.embedding_cache} ...")
        raw_cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=False)
        for pid, vecs in raw_cache.items():
            bert_cache_claims[pid]  = vecs["claims"]
            bert_cache_reviews[pid] = vecs["reviews"]
        log.info(f"  {len(bert_cache_claims)} embeddings loaded "
                 f"(dim={next(iter(bert_cache_claims.values())).shape[-1]}).")

    elif args.encoder == "tfidf":
        # TF-IDF + SVD encoder — no model loading, no Intel MKL conflict.
        # Works on macOS M1. Proves the routing → D_hard → DMN → LoRA pipeline
        # on real OpenReview data with a vocabulary-based semantic signal.
        log.info("\nFitting TF-IDF + SVD encoders (no model loading)...")
        claims_texts  = [p.claims_text  for p in papers]
        reviews_texts = [p.reviews_text for p in papers]

        embed_dim = min(8, len(papers) - 1)   # 8 concept dims = same as neural encoders
        tfidf_pair = TFIDFEncoderPair(embed_dim=embed_dim)
        tfidf_pair.fit(claims_texts, reviews_texts)

        # Inject fitted TF-IDF encoders so the rest of the pipeline is unchanged
        enc_claims  = tfidf_pair.claims
        enc_reviews = tfidf_pair.reviews
        log.info(f"  TF-IDF fitted. vocab={len(tfidf_pair._shared_tfidf.vocabulary_)} "
                 f"embed_dim={tfidf_pair.embed_dim}")

        log.info("  Encoding all papers with TF-IDF...")
        for p in papers:
            try:
                bert_cache_claims[p.paper_id]  = torch.tensor(
                    enc_claims.encode(p.claims_text),  dtype=torch.float32)
                bert_cache_reviews[p.paper_id] = torch.tensor(
                    enc_reviews.encode(p.reviews_text), dtype=torch.float32)
            except Exception as e:
                log.debug(f"  TF-IDF error {p.paper_id}: {e}")
        log.info(f"  {len(bert_cache_claims)} papers encoded.")

    else:
        # Real run: load SciBERT once, share backbone, pre-compute all embeddings.
        # Sharing one backbone between both encoders:
        #   • V1 (Stream Independence) satisfied — projection heads are separate
        #     nn.Module instances with disjoint LoRA parameter spaces.
        #   • No macOS M1 mutex deadlock — one model load, no concurrent init.
        log.info("\nInitialising SciBERT encoders (shared backbone, loads once)...")
        from transformers import AutoTokenizer, AutoModel
        log.info("  Loading SciBERT backbone...")
        shared_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        shared_backbone  = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        shared_backbone.eval()
        log.info("  SciBERT loaded.")

        enc_claims._tokenizer  = enc_reviews._tokenizer = shared_tokenizer
        enc_claims._backbone   = enc_reviews._backbone  = shared_backbone

        log.info("\nPre-computing BERT embeddings (one backbone pass)...")
        for i, p in enumerate(papers):
            try:
                bert_cache_claims[p.paper_id]  = enc_claims._bert_encode(p.claims_text).detach().cpu()
                bert_cache_reviews[p.paper_id] = enc_reviews._bert_encode(p.reviews_text).detach().cpu()
            except Exception as e:
                log.debug(f"  BERT error {p.paper_id}: {e}")
            if (i + 1) % 20 == 0:
                log.info(f"  BERT encoded {i+1}/{len(papers)} papers...")
        log.info(f"  Done. {len(bert_cache_claims)} papers cached.")

    # ── 2b. PCA initialisation of concept projection ──────────────────────
    #
    # Random N(0,0.1) init projects claims and reviews to arbitrary directions
    # in 8-dim space — the resulting D is anti-correlated with rejection
    # (AUROC < 0.5) because it captures stylistic differences (abstract style
    # vs. review style) rather than semantic claim/evidence gaps.
    #
    # Fix: fit PCA on the (claims_bert − reviews_bert) difference vectors.
    # These 8 components span the natural disagreement manifold — the axes
    # along which abstracts and reviews diverge in SciBERT space. Initialising
    # both projection heads with these components means:
    #   • D ≈ 0  for papers where abstract and reviews discuss similar content
    #   • D > 0  for papers where claims diverge from evidence
    # This gives AUROC > 0.5 without label supervision, providing a real signal
    # for SIGReg to improve upon (AIA Experiment 3: ρ = AUROC_SIGReg / AUROC_baseline).
    _first_cached_check = next(iter(bert_cache_claims.values()), None)
    _is_bert_dim = (_first_cached_check is not None and
                    _first_cached_check.shape[-1] != enc_claims.embed_dim)

    if _is_bert_dim and bert_cache_claims:
        from sklearn.decomposition import PCA as _PCA
        _valid_pca = [p.paper_id for p in papers
                      if p.paper_id in bert_cache_claims and p.paper_id in bert_cache_reviews]
        _diffs = torch.stack(
            [bert_cache_claims[pid] - bert_cache_reviews[pid] for pid in _valid_pca]
        ).numpy()                                          # (N, 768)
        _n_comp = min(enc_claims.embed_dim, len(_valid_pca) - 1)
        _pca = _PCA(n_components=_n_comp)
        _pca.fit(_diffs)
        _components = torch.tensor(_pca.components_, dtype=torch.float32)  # (8, 768)
        with torch.no_grad():
            enc_claims.proj.weight.data  = _components.clone()
            enc_claims.proj.bias.data.zero_()
            enc_reviews.proj.weight.data = _components.clone()
            enc_reviews.proj.bias.data.zero_()
        log.info(f"\nPCA projection init ({_n_comp} components, "
                 f"explained_var={_pca.explained_variance_ratio_.sum():.3f})")

    # ── 2c. Projection fine-tuning with SIGReg (--train-projection) ──────
    #
    # "Signal should be accurate in the first place." — applying SIGReg to
    # the base 768→8 concept projection forces the 8 concept dimensions to
    # carry equal variance before any LoRA correction is applied.
    # This is AIA Experiment 3: ρ = AUROC_SIGReg / AUROC_baseline.
    # Only meaningful for real SciBERT embeddings (smoke test uses direct
    # 8-dim vectors, so there is no 768→8 projection to fine-tune).
    _first_cached = next(iter(bert_cache_claims.values()), None)
    _cache_is_concept_dim = (_first_cached is not None and
                             _first_cached.shape[-1] == enc_claims.embed_dim)

    if args.train_projection and args.lambda_iso > 0.0 and not _cache_is_concept_dim:
        valid_ids = [p.paper_id for p in papers if p.paper_id in bert_cache_claims]
        claims_t  = torch.stack([bert_cache_claims[pid]  for pid in valid_ids])  # (N, 768)
        reviews_t = torch.stack([bert_cache_reviews[pid] for pid in valid_ids])  # (N, 768)

        log.info(f"\nFine-tuning claims projection  (lambda_iso={args.lambda_iso}, 300 epochs)...")
        c_res = enc_claims.finetune_projection(claims_t, lambda_iso=args.lambda_iso)
        log.info(f"  Isotropy: {c_res['isotropy_before']:.4f} → {c_res['isotropy_after']:.4f}  "
                 f"  loss={c_res['final_loss']:.6f}")

        log.info(f"Fine-tuning reviews projection (lambda_iso={args.lambda_iso}, 300 epochs)...")
        r_res = enc_reviews.finetune_projection(reviews_t, lambda_iso=args.lambda_iso)
        log.info(f"  Isotropy: {r_res['isotropy_before']:.4f} → {r_res['isotropy_after']:.4f}  "
                 f"  loss={r_res['final_loss']:.6f}")
    elif args.train_projection and _cache_is_concept_dim:
        log.info("\n--train-projection skipped: smoke test uses direct concept vectors.")
    elif args.train_projection:
        log.info("\n--train-projection requires --lambda-iso > 0.0 — skipping.")

    # ── 3. Encode all papers ──────────────────────────────────────────────
    log.info("\nEncoding papers (concept projection → 8-dim routing vectors)...")
    dhard   = DHardQueue(queue_path)
    dhard.clear()
    router  = DivergenceRouter(dhard=dhard, delta=args.delta)

    paper_results = []
    for i, p in enumerate(papers):
        try:
            if p.paper_id in bert_cache_claims:
                c_raw = bert_cache_claims[p.paper_id]
                r_raw = bert_cache_reviews[p.paper_id]
                if c_raw.shape[-1] == enc_claims.embed_dim:
                    # Already concept-dim (smoke test direct vectors) — use as-is
                    z_claims  = c_raw.float()
                    z_reviews = r_raw.float()
                else:
                    # 768-dim BERT output — run through concept projection
                    with torch.no_grad():
                        z_claims  = enc_claims.forward(c_raw.unsqueeze(0)).squeeze(0)
                        z_reviews = enc_reviews.forward(r_raw.unsqueeze(0)).squeeze(0)
            else:
                z_claims  = torch.tensor(enc_claims.encode(p.claims_text),  dtype=torch.float32)
                z_reviews = torch.tensor(enc_reviews.encode(p.reviews_text), dtype=torch.float32)
        except Exception as e:
            log.debug(f"  Encode error {p.paper_id}: {e}")
            continue

        result = router.route(z_claims, z_reviews, paper_id=p.paper_id, venue=p.venue)

        # True label: 1 = paper has a claims/evidence gap (Reject, low score)
        winner  = infer_winner(p.decision, p.avg_score)
        label   = 1 if winner == "reviews" else 0

        paper_results.append({
            "paper_id":    p.paper_id,
            "decision":    result.decision.value,
            "divergence":  result.divergence,
            "label":       label,
            "winner":      winner,
            "or_decision": p.decision,
            "avg_score":   p.avg_score,
        })
        if (i + 1) % 50 == 0:
            log.info(f"  Encoded {i+1}/{len(papers)} papers...")

    log.info(f"Encoded {len(paper_results)} papers.")

    # ── 4. BEFORE metrics ─────────────────────────────────────────────────
    d_scores = [r["divergence"] for r in paper_results]
    labels   = [r["label"]      for r in paper_results]
    auroc_before = compute_auroc(d_scores, labels)

    replan_rate = sum(1 for r in paper_results
                      if r["decision"] == "TRIGGER_REPLAN") / max(len(paper_results), 1)

    log.info(f"\n{'='*55}")
    log.info(f"BEFORE adapter training:")
    log.info(f"  Papers routed:    {len(paper_results)}")
    log.info(f"  TRIGGER_REPLAN:   {replan_rate:.1%}")
    log.info(f"  AUROC (D vs label): {auroc_before:.4f}")
    log.info(f"{'='*55}")

    # ── 5. Attach verdicts to D_hard queue ────────────────────────────────
    verdict_map = {}
    for r in paper_results:
        if r["winner"] is not None:
            verdict_map[r["paper_id"]] = {
                "decision":  r["or_decision"],
                "avg_score": r["avg_score"],
                "winner":    r["winner"],
            }

    n_labelled = dhard.attach_verdicts(verdict_map)
    stats      = dhard.stats()
    log.info(f"\nD_hard queue: {stats['total']} total, "
             f"{stats['resolved']} resolved ({n_labelled} just labelled)")
    log.info(f"  By class: {stats['by_class']}")
    log.info(f"  By winner: {stats['by_winner']}")

    # ── 6. DMN consolidation ──────────────────────────────────────────────
    log.info(f"\nRunning DMN consolidation (SIGReg lambda_iso={args.lambda_iso})...")
    dmn   = ResearchDMN(queue_path=queue_path, adapter_dir="models/adapters")
    built = dmn.consolidate(
        min_events=args.min_events,
        n_epochs=150,
        lambda_iso=args.lambda_iso,
        verbose=True,
    )
    log.info(f"Built {len(built)} adapter(s).")

    if not built:
        log.info("\nNo adapters built (not enough resolved events per class).")
        log.info("Need more papers with clear verdicts. Try --max-papers 1000.")
        return

    # ── 7. Re-run with adapters injected (AFTER) ──────────────────────────
    log.info(f"\nRe-running router with adapters...")
    adapter_router = ResearchAdapterRouter(adapter_dir="models/adapters")
    log.info(f"Adapters available: {adapter_router.available()}")

    # Re-run with adapters.
    # Correct order: (1) apply adapter router → may inject LoRA into an encoder,
    # (2) RE-ENCODE using the now-modified encoder, (3) compute new D.
    # Using pre-computed z vectors for step (3) would ignore the LoRA change.
    d_scores_after = []
    for r in paper_results:
        paper = next((p for p in papers if p.paper_id == r["paper_id"]), None)
        if paper is None:
            d_scores_after.append(r["divergence"])
            continue

        # Use original z for System 1 centroid matching (no re-encode needed)
        c_raw = bert_cache_claims.get(paper.paper_id)
        rv_raw = bert_cache_reviews.get(paper.paper_id)
        if c_raw is None:
            d_scores_after.append(r["divergence"])
            continue

        cache_is_8dim = c_raw.shape[-1] == enc_claims.embed_dim
        try:
            if cache_is_8dim:
                z_claims_orig  = c_raw.float()
                z_reviews_orig = rv_raw.float()
            else:
                with torch.no_grad():
                    z_claims_orig  = enc_claims.forward(c_raw.unsqueeze(0)).squeeze(0)
                    z_reviews_orig = enc_reviews.forward(rv_raw.unsqueeze(0)).squeeze(0)

            # Apply adapter router — may inject LoRA into the faulty encoder
            base_dec = RouteDecision(r["decision"])
            _final_dec, _note = adapter_router.resolve(
                z_claims=z_claims_orig.numpy(),
                z_reviews=z_reviews_orig.numpy(),
                base_decision=base_dec,
                conf_claims=0.7,
                conf_reviews=0.7,
                enc_claims=enc_claims,
                enc_reviews=enc_reviews,
            )

            # Re-encode AFTER potential LoRA injection to see the actual effect
            if cache_is_8dim:
                # 8-dim smoke test: adapters shift projection weights, but since
                # concept vectors bypass the projection, re-route with originals
                result_after = router.route(z_claims_orig, z_reviews_orig,
                                            paper_id=r["paper_id"], venue="")
            else:
                with torch.no_grad():
                    z_claims_new  = enc_claims.forward(c_raw.unsqueeze(0)).squeeze(0)
                    z_reviews_new = enc_reviews.forward(rv_raw.unsqueeze(0)).squeeze(0)
                result_after = router.route(z_claims_new, z_reviews_new,
                                            paper_id=r["paper_id"], venue="")

            d_scores_after.append(result_after.divergence)
        except Exception:
            d_scores_after.append(r["divergence"])
            continue

    auroc_after = compute_auroc(d_scores_after, labels)

    # ── 8. Report ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info(f"RESULTS — Snath Research Experiment")
    log.info(f"{'='*55}")
    log.info(f"  Papers:         {len(paper_results)}")
    log.info(f"  D_hard events:  {stats['total']} ({stats['resolved']} resolved)")
    log.info(f"  Adapters built: {len(built)}")
    log.info(f"")
    log.info(f"  AUROC before:   {auroc_before:.4f}")
    log.info(f"  AUROC after:    {auroc_after:.4f}")
    log.info(f"  Delta AUROC:    {auroc_after - auroc_before:+.4f}")
    log.info(f"")

    if auroc_after > auroc_before:
        log.info(f"  ✓ Learning loop IMPROVED routing accuracy.")
        log.info(f"    D_hard curriculum + LoRA adapters work.")
    else:
        log.info(f"  ✗ No improvement. More data or tuning needed.")
        log.info(f"    Check: min_events, adapter coverage, encoder quality.")

    log.info(f"{'='*55}")

    # Save results
    results_path = f"experiments/results_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump({
            "venue":              args.venue if not args.smoke_test else "SmokeTest",
            "n_papers":           len(paper_results),
            "n_dhard":            stats["total"],
            "n_resolved":         stats["resolved"],
            "adapters_built":     len(built),
            "auroc_before":       auroc_before,
            "auroc_after":        auroc_after,
            "delta_auroc":        auroc_after - auroc_before,
            "lambda_iso":         args.lambda_iso,
            "train_projection":   args.train_projection,
            "timestamp":          datetime.datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)
    log.info(f"\nResults saved: {results_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snath Research Experiment")
    parser.add_argument("--venue",            default="ICLR.cc/2024/Conference")
    parser.add_argument("--max-papers",       type=int,   default=500)
    parser.add_argument("--min-events",       type=int,   default=4)
    parser.add_argument("--delta",            type=float, default=0.25,
                        help="D_hard logging threshold. Default 0.25 (= tau_low, "
                             "logs all TRIGGER_REPLAN events). AIA uses 0.35.")
    parser.add_argument("--from-cache",       type=str,   default=None)
    parser.add_argument("--smoke-test",       action="store_true",
                        help="Run with synthetic data — no internet, instant.")
    parser.add_argument("--encoder",          default="scibert",
                        choices=["scibert", "tfidf"],
                        help="scibert: SciBERT (best, requires working transformers). "
                             "tfidf: TF-IDF+SVD (no model loading, works on macOS M1).")
    parser.add_argument("--embedding-cache",  type=str,   default=None,
                        help="Path to .pt file from precompute_embeddings.py. "
                             "Skips model loading entirely (recommended on macOS M1).")
    parser.add_argument("--lambda-iso",       type=float, default=0.0,
                        help="SIGReg weight for projection fine-tuning and LoRA training. "
                             "AIA Exp 3 sweep: {0.01, 0.1, 1.0}. Default 0.0 (disabled).")
    parser.add_argument("--train-projection", action="store_true",
                        help="Fine-tune the 768→8 concept projection for isotropy before "
                             "routing. Requires --lambda-iso > 0. AIA Experiment 3.")
    args = parser.parse_args()
    run(args)
