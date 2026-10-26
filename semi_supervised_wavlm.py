"""Self-training vs supervised on frozen WavLM embeddings, same labeled subsets (fractions x seeds)."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from common_splits import CACHE, load_splits

d = load_splits()
z = np.load(CACHE.parent / "emb_wavlm_base_plus.npz")
X = np.concatenate([z["mean"], z["std"]], 2).astype(np.float32).mean(1)
sp, y = d.experiment_split.to_numpy(), d.label_id.to_numpy()
sc = StandardScaler().fit(X[sp == "train"]); X = sc.transform(X)
tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
clf = lambda: LogisticRegression(max_iter=2000, C=0.05, class_weight="balanced")
rows = []
for frac in (0.05, 0.1, 0.2, 0.5):
    for seed in range(5):
        rng = np.random.default_rng(seed)
        lab = rng.choice(tr, int(len(tr) * frac), replace=False)
        unl = np.setdiff1d(tr, lab)
        sup = f1_score(y[te], clf().fit(X[lab], y[lab]).predict(X[te]), average="macro")
        L, yl = lab, y[lab]
        for _ in range(5):
            m = clf().fit(X[L], yl)
            p = m.predict_proba(X[unl]); keep = p.max(1) >= 0.9
            if not keep.any(): break
            L = np.concatenate([L, unl[keep]]); yl = np.concatenate([yl, m.classes_[p[keep].argmax(1)]])
            unl = unl[~keep]
        semi = f1_score(y[te], clf().fit(X[L], yl).predict(X[te]), average="macro")
        rows.append({"labeled_fraction": frac, "seed": seed, "supervised": sup, "self_training": semi,
                     "pseudo_added": len(L) - len(lab)})
df = pd.DataFrame(rows)
(CACHE.parent / "semi_supervised_wavlm.json").write_text(df.to_json(orient="records"))
print(df.groupby("labeled_fraction")[["supervised", "self_training", "pseudo_added"]].agg(["mean", "std"]).round(3).to_string())
