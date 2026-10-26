"""Copy the fine-tuned checkpoints used by `predict_emotion.py --finetuned` into models/finetuned/.

Default selection (best compromise on the test split, see FINAL_RESULTS.md): the three noise-augmented runs plus the clean control run.
  python export_finetuned.py                      # default selection
  python export_finetuned.py --runs RUN [RUN...]  # run directory names under experiments/v2/finetune/
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "experiments" / "v2" / "finetune"
DST = ROOT / "models" / "finetuned"
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=None)
a = ap.parse_args()
runs = a.runs or sorted(p.parent.name for pat in ("ftnoise__*", "ftclean__*") for p in SRC.glob(f"{pat}/best_model.pt"))
if not runs:
    raise SystemExit("No best_model.pt found: train with `train_wavlm_v2.py ... --save-model` first")
DST.mkdir(parents=True, exist_ok=True)
for r in runs:
    shutil.copyfile(SRC / r / "best_model.pt", DST / f"{r}.pt")
    print("copied", r)
