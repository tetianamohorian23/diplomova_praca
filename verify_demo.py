"""Quick check of the deployed model on the 66 test clips in samples/ (no datasets needed).
  python verify_demo.py              # default model: noise-augmented heads (models/noise_robust/)
  python verify_demo.py --standard   # heads trained on clean audio only (models/)
The clips are a small hand-picked slice of the test split (speakers unseen in training); labels are in samples/labels.csv.
66 clips only give a rough sanity check: the reported numbers in experiments/v2/FINAL_RESULTS.md use all 3,511 test clips."""
import argparse
from pathlib import Path

import pandas as pd

from predict_emotion import MODELS, ROBUST, Predictor

ROOT = Path(__file__).resolve().parent
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

ap = argparse.ArgumentParser()
ap.add_argument("--standard", action="store_true", help="use the clean-trained heads instead of the noise-augmented ones")
a = ap.parse_args()

df = pd.read_csv(ROOT / "samples" / "labels.csv")
pred = Predictor("cpu", MODELS if a.standard else ROBUST)
rows = []
for r in df.itertuples():
    out = pred.predict(ROOT / "samples" / r.file)
    rows.append((r.file, r.dataset, r.emotion, out["emotion"], out["confidence"]))
res = pd.DataFrame(rows, columns=["file", "dataset", "true", "pred", "confidence"])
res["ok"] = res.true == res.pred

print(res.to_string(index=False, formatters={"confidence": "{:.2f}".format}))
print(f"\naccuracy: {res.ok.mean():.3f}  ({res.ok.sum()}/{len(res)})")
print("\nper emotion (recall):")
print(res.groupby("true").ok.mean().reindex(LABELS).round(2).to_string())
print("\nper corpus (accuracy):")
print(res.groupby("dataset").ok.mean().round(2).to_string())
print("\nconfusion matrix (rows = true, columns = predicted):")
print(pd.crosstab(res.true, res.pred).reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())
res.to_csv(ROOT / "samples" / "demo_predictions.csv", index=False)
print("\nsaved samples/demo_predictions.csv")
