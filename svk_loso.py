"""SVK-DB (30 clips, 4 speakers) evaluated on all clips instead of the 2 clips of the default test split.

zero_shot : train on the other corpora only, predict all 30 Slovak clips.
loso      : leave-one-speaker-out; the other Slovak speakers are added to training.
Frozen WavLM + head (mode none), 5 seeds, probabilities averaged over seeds, bootstrap CI over the 30 clips.
"""
import json

import numpy as np

from analyze_results import bootstrap_ci, mf1
from common_splits import CACHE, LABELS, load_splits
from train_frozen_heads import load_features, run_one

data = load_splits()
X = load_features("wavlm_base_plus")
svk = data.index[data.dataset == "SVK-DB"].to_numpy()
y = data.loc[svk, "label_id"].to_numpy()
pos = {i: k for k, i in enumerate(svk)}
speakers = data.loc[svk, "speaker_id"].to_numpy()
out = {}

P = np.zeros((len(svk), len(LABELS)))
for seed in range(5):
    _, _, probs, idx = run_one("wavlm_base_plus", X, data, "none", 0.0, seed, holdout="SVK-DB")
    for i, p in zip(idx, probs):
        P[pos[i]] += p / 5
out["zero_shot"] = P

P = np.zeros((len(svk), len(LABELS)))
for sp in sorted(set(speakers)):
    d = data.copy()
    d.loc[svk[speakers == sp], "dataset"] = "SVKHOLD"
    for seed in range(5):
        _, _, probs, idx = run_one("wavlm_base_plus", X, d, "none", 0.0, seed, holdout="SVKHOLD")
        for i, p in zip(idx, probs):
            P[pos[i]] += p / 5
    print("speaker", sp, "done", flush=True)
out["loso"] = P

res = {}
for name, P in out.items():
    pred = P.argmax(1)
    lo, hi = bootstrap_ci(y, pred)
    res[name] = {"n_clips": len(y), "macro_f1": mf1(y, pred), "accuracy": float((pred == y).mean()),
                 "ci95": [float(lo), float(hi)],
                 "per_class_recall": {LABELS[c]: float((pred[y == c] == c).mean()) for c in sorted(set(y))}}
    print(name, json.dumps(res[name]))
(CACHE.parent / "svk_loso.json").write_text(json.dumps(res, indent=1))
