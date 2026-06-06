"""
Snath Research — AIA Experiment 3: SIGReg Routing Amplification (COCO / CLIP).
===============================================================================
This is the primary experiment for the AIA paper pre-registration.

Domain
------
  Stream A  : CLIP ViT-B/32 image embeddings  (CLIPImageEncoder)
  Stream B  : CLIP ViT-B/32 caption embeddings (CLIPTextEncoder)
  Oracle    : matched (label=1) vs. mismatched (label=0) image-caption pairs
  Eval      : Winoground (N=400), ARO-Relation (N~50k), COCO hard negatives

Pipeline
--------
  Cell 1  Download & cache COCO 118k + Winoground
  Cell 2  Pre-compute CLIP embeddings (backbone pass, ~45 min on T4)
  Cell 3  Build oracle pairs + verify label balance
  Cell 4  Baseline routing: AUROC on Winoground before any training
  Cell 5  SIGReg fine-tuning (projection-only, λ sweep {0.01, 0.1, 1.0})
  Cell 6  Full InfoNCE + SIGReg continued training (2 epochs on COCO 118k)
  Cell 7  D_hard mining + DMN consolidation (LoRA adapters)
  Cell 8  Re-route with adapters, compute AUROC after
  Cell 9  Compute ρ = AUROC_SIGReg / AUROC_baseline, report results

Success criterion (pre-registered)
-----------------------------------
  Pearson r > 0.7 between isotropy I and ρ, p < 0.05.
  Best SIGReg variant: ρ > 1.15 (15% routing amplification).

Usage
-----
  # Colab: run cells top-to-bottom
  # Local (smoke test on N=200):
  #   python -m experiments.run_coco_experiment --smoke-test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.clip_image_encoder import CLIPImageEncoder, COCO_CLASSES
from encoders.clip_text_encoder  import CLIPTextEncoder
from divergence_router           import DivergenceRouter
from dhard                       import DHardQueue, ResearchDHardEvent
from dmn.research_dmn            import ResearchDMN
from dmn.sigreg                  import SIGRegLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/coco")
CACHE_DIR    = Path("data/coco_clip_cache")
RESULTS_DIR  = Path("experiments/coco_results")
ADAPTER_DIR  = Path("models/coco_adapters")
DHARD_PATH   = Path("coco_d_hard.jsonl")

# ── Routing thresholds (same as ICLR pilot) ───────────────────────────────────
TAU_LOW   = 0.25
TAU_HIGH  = 0.60
DELTA     = 0.25


# ==============================================================================
# CELL 1 — Download COCO + Winoground
# ==============================================================================

def download_coco(data_dir: Path = DATA_DIR, split: str = "train2017") -> None:
    """
    Download MS-COCO 2017 annotations. Images are NOT downloaded here —
    we only need the annotation JSON to build (image_id, caption) pairs.
    Images are fetched on-demand during embedding pre-computation.

    For Colab: mount Google Drive and set data_dir to a Drive path to avoid
    re-downloading on runtime restart.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    ann_url  = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    img_url  = f"http://images.cocodataset.org/zips/{split}.zip"
    ann_path = data_dir / "annotations_trainval2017.zip"
    img_path = data_dir / f"{split}.zip"

    import urllib.request, zipfile
    if not (data_dir / "annotations").exists():
        log.info("Downloading COCO annotations (~241MB)...")
        urllib.request.urlretrieve(ann_url, ann_path)
        with zipfile.ZipFile(ann_path) as z:
            z.extractall(data_dir)
        ann_path.unlink()

    img_dir = data_dir / split
    if not img_dir.exists():
        log.info(f"Downloading COCO {split} images (~18GB)...")
        urllib.request.urlretrieve(img_url, img_path)
        with zipfile.ZipFile(img_path) as z:
            z.extractall(data_dir)
        img_path.unlink()

    log.info(f"COCO {split}: {len(list(img_dir.glob('*.jpg')))} images in {img_dir}")


def download_winoground(data_dir: Path = DATA_DIR) -> None:
    """
    Download Winoground from HuggingFace datasets.
    Requires HF_TOKEN with dataset access (Winoground is gated).

    Cache is written to data_dir/winoground/.
    """
    wg_dir = data_dir / "winoground"
    if wg_dir.exists():
        log.info("Winoground already cached.")
        return
    try:
        from datasets import load_dataset
        ds = load_dataset("facebook/winoground", split="test")
        wg_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(wg_dir))
        log.info(f"Winoground: {len(ds)} examples saved to {wg_dir}")
    except Exception as e:
        log.warning(f"Winoground download failed: {e}")
        log.warning("Set HF_TOKEN and accept the dataset license at "
                    "https://huggingface.co/datasets/facebook/winoground")


# ==============================================================================
# CELL 2 — Pre-compute CLIP embeddings
# ==============================================================================

def precompute_coco_embeddings(
    data_dir:   Path = DATA_DIR,
    cache_dir:  Path = CACHE_DIR,
    split:      str  = "train2017",
    max_pairs:  Optional[int] = None,
    batch_size: int  = 64,
    device:     str  = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """
    Pre-compute CLIP image + caption embeddings for COCO pairs.

    Uses one caption per image (first of the 5). Returns:
        img_embs:  (N, 512) image embeddings
        cap_embs:  (N, 512) caption embeddings
        metadata:  list of {"image_id", "caption", "image_path"}

    Caches to cache_dir/coco_{split}_img.pt and coco_{split}_cap.pt.
    Reloads from cache on subsequent calls.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_cache = cache_dir / f"coco_{split}_img.pt"
    cap_cache = cache_dir / f"coco_{split}_cap.pt"
    meta_cache = cache_dir / f"coco_{split}_meta.json"

    if img_cache.exists() and cap_cache.exists():
        log.info(f"Loading CLIP embeddings from cache ({img_cache})...")
        img_embs = torch.load(img_cache, map_location="cpu", weights_only=True)
        cap_embs = torch.load(cap_cache, map_location="cpu", weights_only=True)
        metadata = json.loads(meta_cache.read_text())
        if max_pairs:
            img_embs = img_embs[:max_pairs]
            cap_embs = cap_embs[:max_pairs]
            metadata = metadata[:max_pairs]
        log.info(f"Loaded {len(metadata)} pairs from cache.")
        return img_embs, cap_embs, metadata

    # ── Load COCO annotations ─────────────────────────────────────────────────
    ann_file = data_dir / "annotations" / f"captions_{split}.json"
    assert ann_file.exists(), f"Missing {ann_file}. Run download_coco() first."
    with open(ann_file) as f:
        coco_ann = json.load(f)

    # Build image_id → first caption mapping
    id_to_caption: Dict[int, str] = {}
    for ann in coco_ann["annotations"]:
        iid = ann["image_id"]
        if iid not in id_to_caption:
            id_to_caption[iid] = ann["caption"]

    # Build image_id → file name mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco_ann["images"]}

    pairs = [
        {
            "image_id":   iid,
            "caption":    id_to_caption[iid],
            "image_path": str(data_dir / split / id_to_file[iid]),
        }
        for iid in id_to_caption
        if iid in id_to_file
    ]
    if max_pairs:
        pairs = pairs[:max_pairs]

    # ── Encode ────────────────────────────────────────────────────────────────
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    from encoders._clip_backbone import get_clip

    dev = torch.device(device)
    model, proc = get_clip(dev)

    img_embs_list, cap_embs_list = [], []
    valid_pairs = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        try:
            images   = [Image.open(p["image_path"]).convert("RGB") for p in batch]
            captions = [p["caption"] for p in batch]
            img_in   = proc(images=images,   return_tensors="pt", padding=True)
            cap_in   = proc(text=captions,   return_tensors="pt", padding=True,
                            truncation=True, max_length=77)
            img_in   = {k: v.to(dev) for k, v in img_in.items()}
            cap_in   = {k: v.to(dev) for k, v in cap_in.items()}
            with torch.no_grad():
                ie = F.normalize(model.get_image_features(**img_in), dim=-1)
                ce = F.normalize(model.get_text_features(**cap_in),  dim=-1)
            img_embs_list.append(ie.cpu())
            cap_embs_list.append(ce.cpu())
            valid_pairs.extend(batch)
        except Exception as exc:
            log.warning(f"Batch {i}–{i+batch_size} failed: {exc}")
            continue

        if (i // batch_size + 1) % 50 == 0:
            log.info(f"  Encoded {len(valid_pairs)}/{len(pairs)} pairs...")

    img_embs = torch.cat(img_embs_list, dim=0)
    cap_embs = torch.cat(cap_embs_list, dim=0)
    torch.save(img_embs, img_cache)
    torch.save(cap_embs, cap_cache)
    meta_cache.write_text(json.dumps(valid_pairs, indent=2))
    log.info(f"Cached {len(valid_pairs)} COCO pairs → {cache_dir}")
    return img_embs, cap_embs, valid_pairs


# ==============================================================================
# CELL 3 — Oracle: matched vs. mismatched pairs
# ==============================================================================

def build_oracle_pairs(
    img_embs: torch.Tensor,
    cap_embs: torch.Tensor,
    metadata: List[dict],
    n_pairs:  int = 5000,
    seed:     int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build a balanced oracle evaluation set.

    Matched   (label=1): (image_i, caption_i) — correct pairing
    Mismatched (label=0): (image_i, caption_j) where j ≠ i,
                          selected to be HARD: highest cross-image cosine
                          similarity (caption sounds plausible for image).

    Returns:
        oracle_img:  (2*n_pairs, 512)
        oracle_cap:  (2*n_pairs, 512)
        labels:      (2*n_pairs,)  int array, 1=matched 0=mismatched
    """
    rng   = np.random.default_rng(seed)
    N     = len(metadata)
    n_use = min(n_pairs, N)

    idx = rng.choice(N, size=n_use, replace=False)
    img_sel = img_embs[idx]
    cap_sel = cap_embs[idx]

    # Hard mismatches: for each image, find the caption (from different image)
    # with highest cosine similarity — the hardest negative.
    sim_matrix = (img_sel @ cap_sel.T).numpy()  # (n_use, n_use)
    np.fill_diagonal(sim_matrix, -1.0)           # mask matched
    mismatch_idx = sim_matrix.argmax(axis=1)     # hardest negative per image

    oracle_img = torch.cat([img_sel, img_sel], dim=0)
    oracle_cap = torch.cat([cap_sel, cap_sel[mismatch_idx]], dim=0)
    labels     = np.array([1] * n_use + [0] * n_use, dtype=int)

    # Shuffle
    perm = rng.permutation(len(labels))
    oracle_img = oracle_img[perm]
    oracle_cap = oracle_cap[perm]
    labels     = labels[perm]

    label_balance = labels.mean()
    log.info(f"Oracle: {len(labels)} pairs, label mean={label_balance:.3f} "
             f"(1.0=all matched, 0.5=balanced)")
    assert 0.45 <= label_balance <= 0.55, (
        f"Oracle is imbalanced (mean={label_balance:.3f}). "
        "Verify n_pairs <= N and seed."
    )
    return oracle_img, oracle_cap, labels


# ==============================================================================
# CELL 4 — Baseline routing AUROC
# ==============================================================================

def route_and_auroc(
    enc_img:    CLIPImageEncoder,
    enc_cap:    CLIPTextEncoder,
    oracle_img: torch.Tensor,
    oracle_cap: torch.Tensor,
    labels:     np.ndarray,
    router:     DivergenceRouter,
    batch_size: int = 256,
) -> Tuple[float, float, np.ndarray]:
    """
    Run routing on all oracle pairs. Returns (AUROC, trigger_rate, d_scores).

    d_scores[i] is the routing divergence D for pair i.
    AUROC measures how well D distinguishes mismatched (label=0, expect high D)
    from matched (label=1, expect low D) — so we compute AUROC(1-labels, d_scores)
    or equivalently flip: AUC of D predicting label=0.

    Returns AUROC for D vs (label==0), i.e. higher D → more likely mismatched.
    """
    from sklearn.metrics import roc_auc_score

    d_scores    = []
    n_replan    = 0

    enc_img.eval()
    enc_cap.eval()

    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            img_b = oracle_img[i : i + batch_size]
            cap_b = oracle_cap[i : i + batch_size]
            z_img = enc_img(img_b)
            z_cap = enc_cap(cap_b)
            for j in range(len(img_b)):
                result = router.route(z_img[j], z_cap[j])
                d_scores.append(result.divergence)
                if result.decision.name == "TRIGGER_REPLAN":
                    n_replan += 1

    d_arr        = np.array(d_scores)
    trigger_rate = n_replan / len(labels)
    # D should be HIGH for mismatched (label=0) → predict (1 - label)
    auroc = roc_auc_score(1 - labels, d_arr)
    return auroc, trigger_rate, d_arr


# ==============================================================================
# CELL 5 — SIGReg projection-only fine-tuning
# ==============================================================================

def run_sigreg_projection(
    enc_img:    CLIPImageEncoder,
    enc_cap:    CLIPTextEncoder,
    img_embs:   torch.Tensor,
    cap_embs:   torch.Tensor,
    lambda_iso: float = 0.1,
    n_epochs:   int   = 300,
) -> Tuple[dict, dict]:
    """Fine-tune both projection heads independently with SIGReg."""
    log.info(f"SIGReg fine-tuning projections (lambda_iso={lambda_iso}, "
             f"{n_epochs} epochs)...")
    res_img = enc_img.finetune_projection(img_embs, lambda_iso=lambda_iso,
                                          n_epochs=n_epochs)
    log.info(f"  Image:   isotropy {res_img['isotropy_before']:.4f} → "
             f"{res_img['isotropy_after']:.4f}  loss={res_img['final_loss']:.4f}")
    res_cap = enc_cap.finetune_projection(cap_embs, lambda_iso=lambda_iso,
                                          n_epochs=n_epochs)
    log.info(f"  Caption: isotropy {res_cap['isotropy_before']:.4f} → "
             f"{res_cap['isotropy_after']:.4f}  loss={res_cap['final_loss']:.4f}")
    return res_img, res_cap


# ==============================================================================
# CELL 6 — InfoNCE + SIGReg continued training
# ==============================================================================

def run_infonce_sigreg(
    enc_img:    CLIPImageEncoder,
    enc_cap:    CLIPTextEncoder,
    img_embs:   torch.Tensor,
    cap_embs:   torch.Tensor,
    lambda_iso: float = 0.1,
    n_epochs:   int   = 2,
    batch_size: int   = 256,
    lr:         float = 1e-5,
    temperature: float = 0.07,
) -> List[float]:
    """
    Continued training: InfoNCE + SIGReg on pre-computed COCO embeddings.

    This is the AIA Experiment 3 training step.
    Both projection heads are optimised jointly; backbones remain frozen.

    NOTE: We train on concept-space representations (512→embed_dim), not
    on raw pixels/tokens. The InfoNCE loss operates over the batch's
    concept vectors, treating same-pair as positive and all others as negatives.

    Returns list of per-epoch losses.
    """
    sigreg    = SIGRegLoss(lambda_iso=lambda_iso)
    optimizer = torch.optim.AdamW(
        list(enc_img.proj.parameters()) + list(enc_cap.proj.parameters()),
        lr=lr, weight_decay=0.01,
    )
    N          = len(img_embs)
    epoch_losses = []

    for epoch in range(n_epochs):
        perm  = torch.randperm(N)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, N, batch_size):
            idx   = perm[i : i + batch_size]
            x_img = img_embs[idx].to(enc_img.device)
            x_cap = cap_embs[idx].to(enc_cap.device)

            z_img = enc_img(x_img)   # (B, embed_dim)
            z_cap = enc_cap(x_cap)   # (B, embed_dim)

            # InfoNCE: diagonal = positive pairs
            z_img_n = F.normalize(z_img, dim=-1)
            z_cap_n = F.normalize(z_cap, dim=-1)
            logits  = (z_img_n @ z_cap_n.T) / temperature
            B       = logits.shape[0]
            targets = torch.arange(B, device=logits.device)
            l_nce   = (F.cross_entropy(logits, targets) +
                       F.cross_entropy(logits.T, targets)) / 2

            # SIGReg on both streams
            l_iso   = sigreg(z_img) + sigreg(z_cap)

            loss = l_nce + l_iso
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc_img.proj.parameters()) + list(enc_cap.proj.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        log.info(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return epoch_losses


# ==============================================================================
# CELL 7 — D_hard mining + DMN consolidation
# ==============================================================================

def mine_dhard_and_consolidate(
    enc_img:    CLIPImageEncoder,
    enc_cap:    CLIPTextEncoder,
    img_embs:   torch.Tensor,
    cap_embs:   torch.Tensor,
    metadata:   List[dict],
    oracle_labels: np.ndarray,
    router:     DivergenceRouter,
    dhard_path: Path = DHARD_PATH,
    adapter_dir: Path = ADAPTER_DIR,
    lambda_iso: float = 0.0,
) -> List[dict]:
    """
    Route all COCO pairs, log D_hard events, run DMN consolidation.

    D_hard condition: TRIGGER_REPLAN or STRUCTURAL_IMPASSE with D ≥ delta.
    Winner is determined by oracle label (1=matched→caption wins, 0=mismatched→image wins).

    Returns list of adapter metadata dicts (same as ResearchDMN.consolidate()).
    """
    from dhard import DHardQueue, ResearchDHardEvent

    adapter_dir.mkdir(parents=True, exist_ok=True)
    queue = DHardQueue(str(dhard_path))

    enc_img.eval()
    enc_cap.eval()

    n_dhard = 0
    with torch.no_grad():
        for i, (label) in enumerate(oracle_labels):
            x_img = img_embs[i].unsqueeze(0).to(enc_img.device)
            x_cap = cap_embs[i].unsqueeze(0).to(enc_cap.device)
            z_img = enc_img(x_img).squeeze(0)
            z_cap = enc_cap(x_cap).squeeze(0)
            result = router.route(z_img, z_cap)

            is_hard = (result.decision.name in ("TRIGGER_REPLAN", "STRUCTURAL_IMPASSE")
                       and result.divergence >= router.delta)
            if not is_hard:
                continue

            # Winner: label=1 (matched) → caption claim was correct → winner="caption"
            #         label=0 (mismatched) → image content was right → winner="image"
            winner = "caption" if label == 1 else "image"

            event = ResearchDHardEvent(
                paper_id     = metadata[i].get("image_id", str(i)),
                failure_class= "compositionality_gap",
                v_claims     = z_img.cpu().numpy().tolist(),
                v_reviews    = z_cap.cpu().numpy().tolist(),
                divergence   = result.divergence,
                winner       = winner,
                resolved     = True,
            )
            queue.append(event)
            n_dhard += 1

    log.info(f"D_hard: {n_dhard} events logged → {dhard_path}")

    dmn   = ResearchDMN(queue_path=str(dhard_path), adapter_dir=str(adapter_dir))
    built = dmn.consolidate(lambda_iso=lambda_iso, verbose=True)
    log.info(f"DMN: built {len(built)} adapter(s)")
    return built


# ==============================================================================
# CELL 8 — Winoground evaluation
# ==============================================================================

def evaluate_winoground(
    enc_img:  CLIPImageEncoder,
    enc_cap:  CLIPTextEncoder,
    router:   DivergenceRouter,
    data_dir: Path = DATA_DIR,
) -> dict:
    """
    Evaluate routing AUROC on Winoground test set (N=400 examples, 800 pairs).

    Each Winoground example has:
        image0, image1, caption0, caption1
    Correct pairs: (image0, caption0) and (image1, caption1)  → label=1
    Swapped pairs: (image0, caption1) and (image1, caption0)  → label=0

    Returns dict with routing_auroc, fusion_auroc, delta_auroc, trigger_rate.
    """
    from sklearn.metrics import roc_auc_score

    wg_dir = data_dir / "winoground"
    if not wg_dir.exists():
        log.warning("Winoground not found. Run download_winoground() first.")
        return {}

    from datasets import load_from_disk
    ds = load_from_disk(str(wg_dir))

    d_scores, fusion_scores, labels = [], [], []
    n_replan = 0

    enc_img.eval()
    enc_cap.eval()

    for ex in ds:
        for img_key, cap_key, label in [
            ("image_0", "caption_0", 1),
            ("image_1", "caption_1", 1),
            ("image_0", "caption_1", 0),
            ("image_1", "caption_0", 0),
        ]:
            try:
                img_emb = torch.tensor(
                    enc_img._clip_image_embed(ex[img_key]).cpu().numpy()
                ).unsqueeze(0).to(enc_img.device)
                cap_emb = torch.tensor(
                    enc_cap._clip_text_embed(ex[cap_key]).cpu().numpy()
                ).unsqueeze(0).to(enc_cap.device)

                with torch.no_grad():
                    z_img = enc_img(img_emb).squeeze(0)
                    z_cap = enc_cap(cap_emb).squeeze(0)

                result = router.route(z_img, z_cap)
                d_scores.append(result.divergence)
                if result.decision.name == "TRIGGER_REPLAN":
                    n_replan += 1

                # Fusion score: cosine similarity of raw CLIP embeddings
                fusion = float(F.cosine_similarity(
                    img_emb.squeeze(0), cap_emb.squeeze(0), dim=0
                ).item())
                fusion_scores.append(fusion)
                labels.append(label)
            except Exception as e:
                log.warning(f"Winoground pair failed: {e}")
                continue

    d_arr = np.array(d_scores)
    f_arr = np.array(fusion_scores)
    l_arr = np.array(labels)

    # High D → mismatched (label=0) → AUROC over (1-label)
    routing_auroc = roc_auc_score(1 - l_arr, d_arr)
    # High fusion → matched (label=1) → AUROC over label directly
    fusion_auroc  = roc_auc_score(l_arr, f_arr)

    results = {
        "routing_auroc":  routing_auroc,
        "fusion_auroc":   fusion_auroc,
        "delta_auroc":    routing_auroc - fusion_auroc,
        "trigger_rate":   n_replan / max(len(labels), 1),
        "n_pairs":        len(labels),
    }
    log.info(f"Winoground: routing={routing_auroc:.4f}  fusion={fusion_auroc:.4f}  "
             f"Δ={results['delta_auroc']:+.4f}  trigger={results['trigger_rate']:.1%}")
    return results


# ==============================================================================
# CELL 9 — Full experiment runner
# ==============================================================================

def _resolve_device(requested: Optional[str]) -> str:
    """Auto-detect the best available device, with a clear error if CUDA is requested but absent."""
    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "cpu"   # MPS multi-instance deadlock
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning(
            "CUDA requested but not available. Falling back to CPU.\n"
            "  In Colab: Runtime → Change runtime type → T4 GPU, then reconnect."
        )
        return "cpu"
    return requested


def run_full_experiment(
    smoke_test:     bool  = False,
    lambda_iso:     float = 0.1,
    embed_dim:      int   = 80,
    use_vocab_init: bool  = True,
    device:         Optional[str] = None,
    sigreg_epochs:  int   = 300,
    infonce_epochs: int   = 2,
) -> dict:
    """
    Run the complete AIA Experiment 3 pipeline.

    Args:
        smoke_test:  Run on N=200 pairs only (fast validation).
        lambda_iso:  SIGReg weight.
        embed_dim:   Concept space dimension.
        use_vocab_init: Use COCO vocabulary init (requires embed_dim=80).
                        Otherwise uses PCA init (any embed_dim).
        device:      "cuda", "mps", "cpu", or None (auto-detect).
        sigreg_epochs:  Epochs for projection-only SIGReg fine-tuning.
        infonce_epochs: Epochs for InfoNCE + SIGReg continued training.
    """
    device = _resolve_device(device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Init encoders ─────────────────────────────────────────────────────────
    log.info(f"Initialising CLIP encoders (embed_dim={embed_dim}, device={device})...")
    enc_img = CLIPImageEncoder(embed_dim=embed_dim, device=device)
    enc_cap = CLIPTextEncoder(embed_dim=embed_dim,  device=device)
    router  = DivergenceRouter(
        tau_low=TAU_LOW, tau_high=TAU_HIGH, delta=DELTA,
    )

    # ── Load or synthesise embeddings ─────────────────────────────────────────
    ann_file = DATA_DIR / "annotations" / "captions_train2017.json"
    if smoke_test and not ann_file.exists():
        # No COCO download needed — generate random 512-dim embeddings that
        # mimic the CLIP distribution (L2-normalised Gaussian).
        N = 200
        log.info(f"Smoke test: generating {N} synthetic CLIP pairs (no COCO required)...")
        torch.manual_seed(42)
        img_embs = F.normalize(torch.randn(N, 512), dim=-1)
        # Captions: matched half ≈ image + small noise, mismatched half = random
        matched    = F.normalize(img_embs[:N//2] + 0.3 * torch.randn(N//2, 512), dim=-1)
        mismatched = F.normalize(torch.randn(N//2, 512), dim=-1)
        cap_embs   = torch.cat([matched, mismatched], dim=0)
        metadata   = [{"image_id": i, "caption": f"synthetic_{i}",
                        "image_path": ""} for i in range(N)]
    else:
        max_pairs = 200 if smoke_test else None
        img_embs, cap_embs, metadata = precompute_coco_embeddings(
            max_pairs=max_pairs, device=device,
        )
    log.info(f"Loaded {len(metadata)} pairs.")

    # ── Concept projection init ───────────────────────────────────────────────
    if use_vocab_init and embed_dim == 80 and not smoke_test:
        log.info("Initialising concept projection with COCO vocabulary (τ=100)...")
        enc_img.init_concept_vocabulary(freeze=False)
        enc_cap.init_concept_vocabulary(freeze=False)
    else:
        log.info(f"Initialising concept projection with PCA (embed_dim={embed_dim})...")
        ev_img = enc_img.init_pca(img_embs)
        ev_cap = enc_cap.init_pca(cap_embs)
        log.info(f"  PCA explained var: image={ev_img:.3f}  caption={ev_cap:.3f}")

    # ── Oracle ────────────────────────────────────────────────────────────────
    oracle_img, oracle_cap, oracle_labels = build_oracle_pairs(
        img_embs, cap_embs, metadata,
        n_pairs=min(2500, len(metadata) // 2),
    )

    # ── Baseline AUROC ────────────────────────────────────────────────────────
    log.info("=== BASELINE ===")
    auroc_before, trig_before, d_before = route_and_auroc(
        enc_img, enc_cap, oracle_img, oracle_cap, oracle_labels, router,
    )
    log.info(f"  AUROC before: {auroc_before:.4f}  TRIGGER_REPLAN: {trig_before:.1%}")

    # ── SIGReg fine-tuning ────────────────────────────────────────────────────
    log.info(f"=== SIGReg projection fine-tuning (λ={lambda_iso}) ===")
    iso_img, iso_cap = run_sigreg_projection(
        enc_img, enc_cap, img_embs, cap_embs,
        lambda_iso=lambda_iso, n_epochs=sigreg_epochs,
    )

    # ── InfoNCE + SIGReg continued training ──────────────────────────────────
    if not smoke_test and infonce_epochs > 0:
        log.info(f"=== InfoNCE + SIGReg continued training ({infonce_epochs} epochs) ===")
        run_infonce_sigreg(
            enc_img, enc_cap, img_embs, cap_embs,
            lambda_iso=lambda_iso, n_epochs=infonce_epochs,
        )

    # ── Post-SIGReg AUROC ─────────────────────────────────────────────────────
    log.info("=== POST-SIGReg ROUTING ===")
    auroc_sigreg, trig_sigreg, d_sigreg = route_and_auroc(
        enc_img, enc_cap, oracle_img, oracle_cap, oracle_labels, router,
    )
    log.info(f"  AUROC SIGReg: {auroc_sigreg:.4f}  TRIGGER_REPLAN: {trig_sigreg:.1%}")

    # ── D_hard mining + DMN ───────────────────────────────────────────────────
    log.info("=== D_hard mining + DMN consolidation ===")
    built = mine_dhard_and_consolidate(
        enc_img, enc_cap, img_embs, cap_embs,
        metadata, oracle_labels, router,
        lambda_iso=lambda_iso,
    )

    # ── Apply adapters + final AUROC ──────────────────────────────────────────
    from pathlib import Path as _Path
    for meta in built:
        pt_path = meta.get("pt_path", "")
        if not pt_path or not _Path(pt_path).exists():
            continue
        target = meta.get("target_encoder", "")
        if target == "image":
            enc_img.load_lora(pt_path)
            log.info(f"  LoRA → enc_img  ({meta['failure_class']}, n={meta['n_events']})")
        elif target == "caption":
            enc_cap.load_lora(pt_path)
            log.info(f"  LoRA → enc_cap  ({meta['failure_class']}, n={meta['n_events']})")

    log.info("=== POST-LoRA ROUTING ===")
    auroc_after, trig_after, _ = route_and_auroc(
        enc_img, enc_cap, oracle_img, oracle_cap, oracle_labels, router,
    )
    log.info(f"  AUROC after:  {auroc_after:.4f}  TRIGGER_REPLAN: {trig_after:.1%}")

    # ── Winoground eval ───────────────────────────────────────────────────────
    wg_results = {}
    if not smoke_test:
        log.info("=== Winoground evaluation ===")
        wg_results = evaluate_winoground(enc_img, enc_cap, router)

    # ── rho ───────────────────────────────────────────────────────────────────
    rho     = auroc_sigreg / auroc_before if auroc_before > 0 else float("nan")
    rho_lora = auroc_after  / auroc_before if auroc_before > 0 else float("nan")

    print("\n" + "=" * 60)
    print("RESULTS — AIA Experiment 3 (COCO / CLIP ViT-B/32)")
    print("=" * 60)
    print(f"  N pairs:          {len(metadata)}")
    print(f"  embed_dim:        {embed_dim}")
    print(f"  lambda_iso:       {lambda_iso}")
    print(f"  Isotropy image:   {iso_img['isotropy_before']:.4f} → {iso_img['isotropy_after']:.4f}")
    print(f"  Isotropy caption: {iso_cap['isotropy_before']:.4f} → {iso_cap['isotropy_after']:.4f}")
    print(f"  AUROC before:     {auroc_before:.4f}")
    print(f"  AUROC SIGReg:     {auroc_sigreg:.4f}   ρ = {rho:.4f}")
    print(f"  AUROC after LoRA: {auroc_after:.4f}   ρ_LoRA = {rho_lora:.4f}")
    print(f"  TRIGGER_REPLAN:   {trig_sigreg:.1%}")
    print(f"  Adapters built:   {len(built)}")
    if wg_results:
        print(f"  Winoground:")
        print(f"    Routing AUROC:  {wg_results.get('routing_auroc', float('nan')):.4f}")
        print(f"    Fusion  AUROC:  {wg_results.get('fusion_auroc',  float('nan')):.4f}")
        print(f"    Δ AUROC:        {wg_results.get('delta_auroc',   float('nan')):+.4f}")
    print(f"  SUCCESS (ρ > 1.15): {'✓' if rho > 1.15 else '✗'} (ρ={rho:.4f})")
    print("=" * 60)

    results = {
        "n_pairs":           len(metadata),
        "embed_dim":         embed_dim,
        "lambda_iso":        lambda_iso,
        "isotropy_img":      {"before": iso_img["isotropy_before"],
                              "after":  iso_img["isotropy_after"]},
        "isotropy_cap":      {"before": iso_cap["isotropy_before"],
                              "after":  iso_cap["isotropy_after"]},
        "auroc_before":      auroc_before,
        "auroc_sigreg":      auroc_sigreg,
        "auroc_after_lora":  auroc_after,
        "rho":               rho,
        "rho_lora":          rho_lora,
        "trigger_rate":      trig_sigreg,
        "n_adapters":        len(built),
        "winoground":        wg_results,
    }

    import datetime
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"coco_exp3_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved: {out_path}")
    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AIA Experiment 3 — SIGReg on COCO / CLIP ViT-B/32"
    )
    parser.add_argument("--smoke-test",     action="store_true",
                        help="Run on N=200 pairs (fast validation)")
    parser.add_argument("--lambda-iso",     type=float, default=0.1)
    parser.add_argument("--embed-dim",      type=int,   default=80)
    parser.add_argument("--no-vocab-init",  action="store_true",
                        help="Use PCA init instead of COCO vocabulary")
    parser.add_argument("--device",         default=None,
                        help="cuda / mps / cpu (default: auto-detect)")
    parser.add_argument("--sigreg-epochs",  type=int,   default=300)
    parser.add_argument("--infonce-epochs", type=int,   default=2)
    parser.add_argument("--download",       action="store_true",
                        help="Download COCO + Winoground first")
    args = parser.parse_args()

    if args.download:
        download_coco()
        download_winoground()

    run_full_experiment(
        smoke_test      = args.smoke_test,
        lambda_iso      = args.lambda_iso,
        embed_dim       = args.embed_dim,
        use_vocab_init  = not args.no_vocab_init,
        device          = args.device,
        sigreg_epochs   = args.sigreg_epochs,
        infonce_epochs  = args.infonce_epochs,
    )
