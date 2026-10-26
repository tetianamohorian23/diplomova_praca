"""Validity checks: speaker-grouped CV, corpus-normalised LOCO, duplicate / leakage scan.
Usage: python extra_eval.py grouped_cv | loco_norm | duplicates
"""
import json
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from common_splits import CACHE, LABELS, load_splits
from train_frozen_heads import load_features, run_one

V2 = CACHE.parent
data = load_splits()
stage = sys.argv[1]


def mf1(y, p):
    return float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0))


if stage == "grouped_cv":
    X = load_features("wavlm_base_plus")
    spk = data.speaker_id.to_numpy()
    out = {"folds": [], "pooled_probs": np.zeros((len(data), len(LABELS)))}
    rng = np.random.default_rng(0)
    for k, (tr_val, te) in enumerate(GroupKFold(5).split(X[:, 0, 0], groups=spk)):
        speakers = np.unique(spk[tr_val]); rng.shuffle(speakers)
        val_sp = set(speakers[: max(1, int(0.15 * len(speakers)))])
        d = data.copy()
        d["experiment_split"] = "train"
        d.loc[d.index[te], "experiment_split"] = "test"
        d.loc[[i for i in tr_val if spk[i] in val_sp], "experiment_split"] = "validation"
        P = 0
        for seed in range(3):
            res, _, probs, idx = run_one("wavlm_base_plus", X, d, "none", 0.0, seed)
            P = P + probs / 3
        y = data.label_id.to_numpy()[idx]
        out["pooled_probs"][idx] = P
        fold = {"fold": k, "n": len(idx), "macro_f1": mf1(y, P.argmax(1)), "accuracy": float(accuracy_score(y, P.argmax(1))),
                "corpora": data.dataset.iloc[idx].value_counts().to_dict()}
        out["folds"].append(fold); print(fold, flush=True)
    y = data.label_id.to_numpy(); pred = out["pooled_probs"].argmax(1)
    per_corpus = {c: mf1(y[(data.dataset == c).to_numpy()], pred[(data.dataset == c).to_numpy()]) for c in sorted(data.dataset.unique())}
    summary = {"folds": out["folds"], "pooled_macro_f1": mf1(y, pred), "pooled_accuracy": float(accuracy_score(y, pred)),
               "fold_f1_mean": float(np.mean([f["macro_f1"] for f in out["folds"]])),
               "fold_f1_std": float(np.std([f["macro_f1"] for f in out["folds"]])), "per_corpus_f1": per_corpus,
               "macro_f1_without_tess": mf1(y[(data.dataset != "TESS").to_numpy()], pred[(data.dataset != "TESS").to_numpy()])}
    (V2 / "grouped_cv.json").write_text(json.dumps(summary, indent=1)); np.save(V2 / "grouped_cv_probs.npy", out["pooled_probs"])
    print({k: v for k, v in summary.items() if k != "folds"})

elif stage == "loco_norm":
    X = load_features("wavlm_base_plus")
    Xn = X.copy()
    for c in data.dataset.unique():
        m = (data.dataset == c).to_numpy()
        Xn[m] = (X[m] - X[m].mean(0, keepdims=True)) / (X[m].std(0, keepdims=True) + 1e-5)
    rows = []
    for hold in [h for h in sorted(data.dataset.unique()) if h != "SVK-DB"]:
        for seed in range(3):
            r, *_ = run_one("wavlm_base_plus", Xn, data, "none", 0.0, seed, holdout=hold)
            rows.append({"holdout": hold, "seed": seed, "test_f1": r["test"]["macro_f1"]}); print(rows[-1], flush=True)
    (V2 / "loco_corpus_norm.json").write_text(json.dumps(rows))

elif stage == "duplicates":
    waves = np.load(V2 / "cnn_ssl" / "waves_int16.npy", mmap_mode="r")
    import hashlib
    hashes = [hashlib.md5(np.asarray(waves[i]).tobytes()).hexdigest() for i in range(len(data))]
    data["h"] = hashes
    exact = data[data.duplicated("h", keep=False)]
    z = np.load(V2 / "emb_wavlm_base_plus.npz")
    F = np.concatenate([z["mean"], z["std"]], 2).astype(np.float32).mean(1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6); F /= np.linalg.norm(F, axis=1, keepdims=True)
    sp = data.experiment_split.to_numpy(); tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
    S = F[te] @ F[tr].T
    best, arg = S.max(1), S.argmax(1)
    near = best > 0.995
    same_label = (data.label_id.to_numpy()[te][near] == data.label_id.to_numpy()[tr][arg[near]]).mean() if near.any() else None
    res = {"exact_duplicate_rows": int(len(exact)), "exact_duplicate_groups_crossing_splits": int(exact.groupby("h").experiment_split.nunique().gt(1).sum()),
           "near_duplicates_test_vs_train(cos>0.995)": int(near.sum()), "near_dup_same_label_rate": None if same_label is None else float(same_label),
           "max_cosine_test_to_train_percentiles": {str(q): float(np.percentile(best, q)) for q in (50, 90, 99, 100)}}
    (V2 / "duplicates.json").write_text(json.dumps(res, indent=1)); print(res)
