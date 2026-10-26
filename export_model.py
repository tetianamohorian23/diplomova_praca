"""Train and save the deployable heads (5 seeds, frozen WavLM features).

  python export_model.py          -> models/wavlm_head_seed*.pt           (standard)
  python export_model.py --norm   -> models/norm/wavlm_head_seed*.pt      (per-source feature normalisation,
                                      used by `predict_emotion.py --batch-normalize` for batches from one source)
"""
import argparse
from pathlib import Path

from common_splits import load_splits
from train_frozen_heads import load_features, run_one

ap = argparse.ArgumentParser()
ap.add_argument("--norm", action="store_true")
args = ap.parse_args()
out = Path(__file__).resolve().parent / "models" / ("norm" if args.norm else "")
out.mkdir(parents=True, exist_ok=True)
data, X = load_splits(), load_features("wavlm_base_plus")
if args.norm:
    for c in data.dataset.unique():
        m = (data.dataset == c).to_numpy()
        X[m] = (X[m] - X[m].mean(0, keepdims=True)) / (X[m].std(0, keepdims=True) + 1e-5)
for seed in range(5):
    res, *_ = run_one("wavlm_base_plus", X, data, "none", 0.0, seed, save_path=out / f"wavlm_head_seed{seed}.pt")
    print(f"seed {seed}: val_f1={res['val_macro_f1']:.3f} test_f1={res['test']['macro_f1']:.3f}", flush=True)
