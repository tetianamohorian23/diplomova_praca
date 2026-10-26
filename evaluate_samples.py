"""Quick check of the prototype on the labelled example clips in samples/ (28 test-split clips, 4 per emotion, mixed corpora).

  python evaluate_samples.py               # default noise-robust heads
  python evaluate_samples.py --standard    # heads trained on clean audio only

Prints the predicted vs the true emotion for every clip, then accuracy and macro-F1. Only 28 clips: this shows that the whole chain
(audio -> WavLM -> heads) works; the reported numbers on the full 3,511-clip test split are in experiments/v2/FINAL_RESULTS.md.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from predict_emotion import MODELS, ROBUST, Predictor

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standard", action="store_true", help="use the heads trained on clean audio only (models/) instead of models/noise_robust")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(SAMPLES / "labels.csv", encoding="utf-8")))
    pred = Predictor("cpu", MODELS if a.standard or not ROBUST.exists() else ROBUST)

    true, guess = [], []
    print(f"{'file':<46} {'true':<9} {'predicted':<9} {'conf':>5}")
    for r in rows:
        res = pred.predict(SAMPLES / r["file"])
        true.append(r["emotion"]); guess.append(res["emotion"])
        mark = "" if res["emotion"] == r["emotion"] else "  <-- miss"
        print(f"{r['file']:<46} {r['emotion']:<9} {res['emotion']:<9} {res['confidence']:>5.0%}{mark}")
    print(f"\n{len(rows)} clips: accuracy {accuracy_score(true, guess):.1%}, macro-F1 {f1_score(true, guess, average='macro'):.3f}")


if __name__ == "__main__":
    main()
