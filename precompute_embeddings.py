"""
precompute_embeddings.py — Pre-compute BERT embeddings for Snath Research.

Run this ONCE before the main experiment to cache 768-dim encoder embeddings.
The main experiment loads from cache — no model loading during routing.

WHY THIS EXISTS
---------------
macOS M1 has an Intel OpenMP mutex race that deadlocks when PyTorch and
transformers both initialise their thread pools in the same process as the
experiment pipeline code. This script is a single-purpose process: it only
loads the encoder, encodes, saves, exits. No experiment code imported.

MODELS
------
    scibert  (default) — allenai/scibert_scivocab_uncased (440MB)
                         Pre-trained on CS + biomedical papers (Semantic Scholar +
                         PubMed). Best domain match for scientific claims routing.
    specter             — allenai/specter via sentence-transformers (440MB)
                         Trained on citation graphs — better at paper-level
                         similarity than SciBERT. Try this if scibert deadlocks.
    minilm              — all-MiniLM-L6-v2 via sentence-transformers (80MB)
                         Fast, reliable, general purpose. Use to prove the
                         mechanism quickly; swap for scibert for paper results.

USAGE
-----
    # Fetch from OpenReview + encode in one step:
    python precompute_embeddings.py \\
        --venue ICLR.cc/2024/Conference --max-papers 200

    # Encode from existing text cache:
    python precompute_embeddings.py \\
        --from-cache data/ICLR.cc_2024_Conference_cache.jsonl

    # Use SPECTER if SciBERT deadlocks:
    python precompute_embeddings.py \\
        --from-cache data/ICLR.cc_2024_Conference_cache.jsonl --model specter

OUTPUT
------
    data/<venue>_embeddings.pt  — loaded by run_experiment.py --embedding-cache
    Format: {paper_id: {"claims": Tensor(768,), "reviews": Tensor(768,)}}
"""
from __future__ import annotations

# ── Thread isolation — MUST be before any torch / numpy / transformers import ──
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]   = "TRUE"
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
os.environ["NUMEXPR_MAX_THREADS"]     = "1"

import sys
import json
import time
import argparse
import logging

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


# ── Encoders ──────────────────────────────────────────────────────────────────

def load_scibert():
    """Load SciBERT backbone. Returns (tokenizer, model)."""
    from transformers import AutoTokenizer, AutoModel
    log.info("  Loading allenai/scibert_scivocab_uncased ...")
    tok   = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    model.eval()
    log.info("  SciBERT loaded.")
    return tok, model


def encode_scibert(text: str, tok, model, max_length: int = 512) -> torch.Tensor:
    """Mean-pool SciBERT over non-padding tokens → (768,)."""
    inputs = tok(text, return_tensors="pt", max_length=max_length,
                 truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled.squeeze(0)   # (768,)


def load_sentence_transformer(model_name: str):
    """Load a sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)
    log.info(f"  Loading {model_name} via sentence-transformers ...")
    model = SentenceTransformer(model_name)
    log.info("  Model loaded.")
    return model


MODEL_MAP = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "specter": "allenai/specter",
    "minilm":  "all-MiniLM-L6-v2",
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs("data", exist_ok=True)

    # ── 1. Load papers ────────────────────────────────────────────────────
    if args.from_cache:
        log.info(f"Loading papers from cache: {args.from_cache}")
        from data.openreview_pipeline import PaperRecord
        papers = []
        with open(args.from_cache) as f:
            for line in f:
                d = json.loads(line)
                papers.append(PaperRecord(**d))
        venue_slug = os.path.splitext(os.path.basename(args.from_cache))[0].replace("_cache", "")
    else:
        log.info(f"Fetching from OpenReview: {args.venue} (max {args.max_papers} papers)...")
        from data.openreview_pipeline import OpenReviewPipeline
        pipe   = OpenReviewPipeline(venue=args.venue, sleep_sec=0.3)
        papers = pipe.fetch(max_papers=args.max_papers)
        venue_slug = args.venue.replace("/", "_")
        cache_path = f"data/{venue_slug}_cache.jsonl"
        pipe.save(papers, cache_path)
        log.info(f"  Text cache saved → {cache_path}")

    log.info(f"  {len(papers)} papers to encode.\n")

    # ── 2. Load encoder ───────────────────────────────────────────────────
    model_key = args.model.lower()
    if model_key not in MODEL_MAP:
        log.error(f"Unknown model '{args.model}'. Choose from: {list(MODEL_MAP)}")
        sys.exit(1)

    use_st = model_key in ("specter", "minilm")   # sentence-transformers path

    if use_st:
        st_model = load_sentence_transformer(MODEL_MAP[model_key])
        tok, bert = None, None
        embed_dim = st_model.get_sentence_embedding_dimension()
    else:
        tok, bert = load_scibert()
        st_model  = None
        embed_dim = 768

    log.info(f"  Embedding dim: {embed_dim}\n")

    # ── 3. Encode all papers ──────────────────────────────────────────────
    cache = {}   # {paper_id: {"claims": Tensor, "reviews": Tensor}}
    errors = 0

    for i, p in enumerate(papers):
        try:
            if use_st:
                c_vec = torch.tensor(
                    st_model.encode(p.claims_text,  show_progress_bar=False),
                    dtype=torch.float32,
                )
                r_vec = torch.tensor(
                    st_model.encode(p.reviews_text, show_progress_bar=False),
                    dtype=torch.float32,
                )
            else:
                c_vec = encode_scibert(p.claims_text,  tok, bert).float()
                r_vec = encode_scibert(p.reviews_text, tok, bert).float()

            cache[p.paper_id] = {"claims": c_vec, "reviews": r_vec}

        except Exception as e:
            log.warning(f"  Error on {p.paper_id}: {e}")
            errors += 1
            continue

        if (i + 1) % 10 == 0:
            log.info(f"  Encoded {i+1}/{len(papers)} papers ...")

    log.info(f"\n  Encoded {len(cache)} / {len(papers)} papers ({errors} errors).")

    # ── 4. Save ───────────────────────────────────────────────────────────
    out_path = args.output or f"data/{venue_slug}_embeddings_{model_key}.pt"
    torch.save(cache, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    log.info(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")
    log.info(f"\nRun the experiment:")
    log.info(f"  python experiments/run_experiment.py \\")
    if args.from_cache:
        log.info(f"    --from-cache {args.from_cache} \\")
    else:
        log.info(f"    --from-cache data/{venue_slug}_cache.jsonl \\")
    log.info(f"    --embedding-cache {out_path} \\")
    log.info(f"    --lambda-iso 0.1 --train-projection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute BERT embeddings for Snath Research")
    parser.add_argument("--venue",      default="ICLR.cc/2024/Conference",
                        help="OpenReview venue ID (ignored if --from-cache)")
    parser.add_argument("--max-papers", type=int, default=200)
    parser.add_argument("--from-cache", type=str, default=None,
                        help="Load from existing text cache (JSONL) instead of fetching")
    parser.add_argument("--model",      default="scibert",
                        choices=list(MODEL_MAP),
                        help="Encoder: scibert (best accuracy) | specter | minilm (fastest)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output .pt path (auto-named if omitted)")
    args = parser.parse_args()
    main(args)
