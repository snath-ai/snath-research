# ============================================================
# Snath Research — Full Experiment (Google Colab)
# Run each cell in order. ~20 min on Colab free tier (CPU).
# GPU runtime (~5 min): Runtime → Change runtime type → T4 GPU
# ============================================================

# ── Cell 1: Install & clone ──────────────────────────────────
# !pip install -q torch transformers scikit-learn sentence-transformers
# !git clone https://github.com/snath-ai/snath-research.git
# %cd snath-research

# ── Cell 2: Baseline — SciBERT, no SIGReg ────────────────────
# !python experiments/run_experiment.py \
#     --venue ICLR.cc/2024/Conference \
#     --max-papers 500 \
#     --encoder scibert \
#     --min-events 3 \
#     --delta 0.25

# ── Cell 3: SIGReg — lambda_iso 0.1 + projection fine-tuning ─
# !python experiments/run_experiment.py \
#     --venue ICLR.cc/2024/Conference \
#     --max-papers 500 \
#     --encoder scibert \
#     --lambda-iso 0.1 \
#     --train-projection \
#     --min-events 3 \
#     --delta 0.25

# ── Cell 4: Compare results ───────────────────────────────────
import json, glob, os

results = []
for path in sorted(glob.glob("experiments/results_*.json")):
    with open(path) as f:
        r = json.load(f)
    results.append(r)

if not results:
    print("No results yet — run cells 2 and 3 first.")
else:
    print(f"{'Run':<4} {'lambda_iso':<12} {'proj':<6} {'papers':<8} "
          f"{'D_hard':<8} {'adapters':<10} {'AUROC_before':<14} "
          f"{'AUROC_after':<13} {'delta'}")
    print("-" * 90)
    for i, r in enumerate(results):
        liso  = r.get("lambda_iso", 0.0)
        proj  = "yes" if r.get("train_projection") else "no"
        delta = r.get("delta_auroc", 0.0)
        sign  = "✓" if delta > 0 else ("✗" if delta < 0 else "—")
        print(f"{i+1:<4} {liso:<12} {proj:<6} {r['n_papers']:<8} "
              f"{r['n_dhard']:<8} {r['adapters_built']:<10} "
              f"{r['auroc_before']:<14.4f} {r['auroc_after']:<13.4f} "
              f"{delta:+.4f} {sign}")

    # AIA Experiment 3 ratio: rho = AUROC_SIGReg / AUROC_baseline
    if len(results) >= 2:
        baseline = next((r for r in results if r.get("lambda_iso", 0) == 0), None)
        sigreg   = next((r for r in results if r.get("lambda_iso", 0) > 0), None)
        if baseline and sigreg:
            rho = sigreg["auroc_after"] / max(baseline["auroc_after"], 1e-6)
            print(f"\nAIA Experiment 3 — ρ = AUROC_SIGReg / AUROC_baseline = {rho:.4f}")
            print(f"Target: ρ > 1.15  →  {'✓ PASSED' if rho > 1.15 else '✗ not yet'}")
