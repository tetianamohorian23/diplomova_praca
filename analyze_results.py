"""Aggregate every v2 experiment into tables, plots and experiments/v2/REPORT.md."""
from __future__ import annotations

import argparse
import json
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from common_splits import CACHE, LABELS, load_splits

V2 = CACHE.parent
PLOTS = V2 / "plots"
PLOTS.mkdir(exist_ok=True)


def mf1(y, p) -> float:
    return f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".3f") if len(df) else "_no results yet_"


def frozen_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in (V2 / "frozen").glob("*.json"):
        r = json.loads(f.read_text())
        rows.append({"embedding": r["embedding"], "mode": r["mode"], "lambda": r["lambda"], "seed": r["seed"],
                     "holdout": r["holdout"], "val_f1": r["val_macro_f1"], "test_f1": r["test"]["macro_f1"],
                     "test_acc": r["test"]["accuracy"], "bal_acc": r["test"]["balanced_accuracy"]})
    df = pd.DataFrame(rows)
    return df[df.holdout.isna()] if len(df) else df, df[df.holdout.notna()] if len(df) else df


def agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by).agg(n=("seed", "count"), val_f1=("val_f1", "mean"), test_f1_mean=("test_f1", "mean"),
                           test_f1_std=("test_f1", "std"), acc=("test_acc", "mean"), bal_acc=("bal_acc", "mean"))
    return g.reset_index()


def finetune_table() -> pd.DataFrame:
    rows = []
    for f in (V2 / "finetune").glob("*/result.json"):
        r = json.loads(f.read_text())
        a = r["args"]
        rows.append({"run": f.parent.name, "mode": a["mode"], "lambda": a["lam"], "seed": a["seed"],
                     "holdout": a.get("holdout"), "best_epoch": r["best_epoch"], "val_f1": r["best_val_macro_f1"],
                     "test_f1": r["test"]["macro_f1"], "test_acc": r["test"]["accuracy"],
                     "bal_acc": r["test"]["balanced_accuracy"], "hours": r["minutes"] / 60})
    return pd.DataFrame(rows)


def bootstrap_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [mf1(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def paired_bootstrap(y, pa, pb, n=1000, seed=0):
    """P(model A is better than B) under resampling of the test set."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(mf1(y[i], pa[i]) - mf1(y[i], pb[i]))
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])


def load_probs(kind: str, name: str) -> np.ndarray | None:
    path = V2 / kind / (name + ("/probs.npz" if kind == "finetune" else ".npz"))
    return np.load(path)["test"] if path.exists() else None


def layer_probe(tags: list[str], data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    split = data["experiment_split"].to_numpy()
    y = data["label_id"].to_numpy()
    rows = []
    for tag in tags:
        z = np.load(V2 / f"emb_{tag}.npz")["mean"].astype(np.float32)
        for layer in range(z.shape[1]):
            sc = StandardScaler().fit(z[split == "train", layer])
            clf = LogisticRegression(max_iter=300, C=0.5, class_weight="balanced")
            clf.fit(sc.transform(z[split == "train", layer]), y[split == "train"])
            pred = clf.predict(sc.transform(z[split == "test", layer]))
            rows.append({"embedding": tag, "layer": layer, "test_f1": mf1(y[split == "test"], pred),
                         "test_acc": accuracy_score(y[split == "test"], pred)})
            print("probe", tag, layer, rows[-1]["test_f1"], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run (slow) layer-wise linear probing")
    args = ap.parse_args()
    data = load_splits()
    test_mask = (data.experiment_split == "test").to_numpy()
    y_test = data.loc[test_mask, "label_id"].to_numpy()
    test_idx = np.where(test_mask)[0]
    out = ["# Experiment report (v2)\n",
           f"Test set: {test_mask.sum()} clips, speaker-disjoint; same split for every model.\n"]

    base, loco = frozen_table()
    if len(base):
        a = agg(base, ["embedding", "mode", "lambda"])
        a["test_f1"] = [f"{m:.3f} ± {0 if np.isnan(s) else s:.3f}" for m, s in zip(a.test_f1_mean, a.test_f1_std)]
        out += ["\n## 1. Frozen SSL features + light head: corpus-task ablation (mean ± std over seeds)\n",
                md(a[["embedding", "mode", "lambda", "n", "val_f1", "test_f1", "acc", "bal_acc"]].set_index("embedding")), "\n"]
        fig, ax = plt.subplots(figsize=(8, 4))
        for tag, g in a.groupby("embedding"):
            lab = [f"{m}\n{l:g}" for m, l in zip(g["mode"], g["lambda"])]
            ax.errorbar(range(len(g)), g.test_f1_mean, yerr=g.test_f1_std.fillna(0), marker="o", capsize=3, label=tag)
        ax.set_xticks(range(len(lab)), lab); ax.set_ylabel("test macro-F1"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("none / aux (multi-task) / grl (adversarial) by λ")
        fig.tight_layout(); fig.savefig(PLOTS / "ablation_lambda.png", dpi=150); plt.close(fig)

    if len(loco):
        a = loco.groupby(["embedding", "mode", "lambda", "holdout"]).test_f1.agg(["mean", "std", "count"]).reset_index()
        piv = a.pivot_table(index=["embedding", "mode", "lambda"], columns="holdout", values="mean")
        piv["average"] = piv.mean(1)
        out += ["\n## 2. Leave-one-corpus-out (train on the other corpora, test on the held-out one; macro-F1)\n", md(piv), "\n"]
        fig, ax = plt.subplots(figsize=(8, max(2.5, .6 * len(piv))))
        im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30); ax.set_yticks(range(len(piv)), [" ".join(map(str, i)) for i in piv.index])
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(PLOTS / "loco_heatmap.png", dpi=150); plt.close(fig)

    ft = finetune_table()
    if len(ft):
        out += ["\n## 3. Fine-tuned models (train_wavlm_v2.py)\n", md(ft.drop(columns=["run"]).assign(run=ft.run).set_index("run")), "\n"]

    members: dict[str, np.ndarray] = {}
    for tag in ("wavlm_base_plus", "wav2vec2_base"):
        ps = [load_probs("frozen", f"{tag}__none_0__seed{s}") for s in range(20)]
        ps = [p for p in ps if p is not None]
        if ps:
            members[f"{tag}_frozen(mean of {len(ps)} seeds)"] = np.mean(ps, 0)
    for f in sorted((V2 / "finetune").glob("*/probs.npz")):
        z = np.load(f)
        if len(z["test"]) == test_mask.sum() and "hold-" not in f.parent.name:
            members[f"finetune:{f.parent.name}"] = z["test"]
    if members:
        out += ["\n## 4. Single models vs ensembles (test macro-F1 with 95% bootstrap CI)\n"]
        rows = []
        combos = {k: v for k, v in members.items()}
        if len(members) > 1:
            combos["ENSEMBLE(all members, prob. average)"] = np.mean(list(members.values()), 0)
        for name, p in combos.items():
            pred = p.argmax(1)
            lo, hi = bootstrap_ci(y_test, pred)
            rows.append({"model": name, "macro_f1": mf1(y_test, pred), "ci95": f"[{lo:.3f}, {hi:.3f}]",
                         "acc": accuracy_score(y_test, pred)})
        out += [md(pd.DataFrame(rows).set_index("model")), "\n"]
        names = list(combos)
        best = max(rows, key=lambda r: r["macro_f1"])["model"]
        for n in names:
            if n != best:
                m, (lo, hi) = paired_bootstrap(y_test, combos[best].argmax(1), combos[n].argmax(1))
                out.append(f"- {best} − {n}: ΔF1 = {m:+.3f} (95% CI {lo:+.3f}…{hi:+.3f})\n")
        pred = combos[best].argmax(1)
        ds = data.loc[test_mask, "dataset"].to_numpy()
        pc = pd.DataFrame([{"corpus": d, "language": data.loc[test_mask & (data.dataset == d).to_numpy(), "language"].iloc[0],
                            "n": int((ds == d).sum()), "macro_f1": mf1(y_test[ds == d], pred[ds == d]),
                            "acc": accuracy_score(y_test[ds == d], pred[ds == d])} for d in sorted(set(ds))])
        out += [f"\n### Per-corpus results of `{best}`\n", md(pc.set_index("corpus")), "\n"]
        cm = confusion_matrix(y_test, pred, labels=range(len(LABELS)))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues")
        ax.set_xticks(range(7), LABELS, rotation=45); ax.set_yticks(range(7), LABELS)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=8)
        ax.set_title("Confusion matrix (best ensemble)"); fig.tight_layout()
        fig.savefig(PLOTS / "confusion_best.png", dpi=150); plt.close(fig)
        off = [(cm[i, j], LABELS[i], LABELS[j]) for i, j in product(range(7), range(7)) if i != j]
        off.sort(reverse=True)
        out += ["\n### Most frequent confusions (true → predicted)\n"] + [f"- {a} → {b}: {n}\n" for n, a, b in off[:6]]
        sur = y_test == LABELS.index("surprise")
        out.append("\n`surprise` exists only in RAVDESS, SAVEE and TESS; test recall by corpus: " +
                   ", ".join(f"{d}: {(pred[sur & (ds == d)] == LABELS.index('surprise')).mean():.2f} (n={int((sur & (ds == d)).sum())})"
                             for d in sorted(set(ds[sur]))) + "\n")

    if args.probe:
        pr = layer_probe([t for t in ("wavlm_base_plus", "wav2vec2_base") if (V2 / f"emb_{t}.npz").exists()], data)
        pr.to_csv(V2 / "layer_probe.csv", index=False)
        out += ["\n## 5. Layer-wise linear probing (frozen, mean pooled)\n", md(pr.pivot(index="layer", columns="embedding", values="test_f1")), "\n"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for t, g in pr.groupby("embedding"):
            ax.plot(g.layer, g.test_f1, marker="o", label=t)
        ax.set_xlabel("layer"); ax.set_ylabel("test macro-F1"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(PLOTS / "layer_probe.png", dpi=150); plt.close(fig)

    out += ["\n## Data caveats\n",
            "- TESS has only 2 speakers, so the speaker-disjoint split puts none in train and 1400 clips each in validation and test "
            "(≈44% of val and 40% of test). Validation-based model selection and test scores are therefore TESS-heavy.\n",
            "- CREMA-D has no `surprise`; SVK-DB has 30 clips, EMO-DB/UA-SER lack several classes.\n"]
    (V2 / "REPORT.md").write_text("".join(out), encoding="utf-8")
    print("".join(out).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
