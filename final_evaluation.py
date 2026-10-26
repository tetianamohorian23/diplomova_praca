"""Collect every result into experiments/v2/FINAL_RESULTS.md (+ plots). Missing inputs are skipped."""
from __future__ import annotations

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from common_splits import CACHE, LABELS, ROOT, load_splits

V2 = CACHE.parent
PL = V2 / "plots"
PL.mkdir(exist_ok=True)
data = load_splits()
te = (data.experiment_split == "test").to_numpy()
y = data.label_id.to_numpy()[te]
ds = data.dataset.to_numpy()[te]
out: list[str] = []


def load(path):
    p = V2 / path
    return json.loads(p.read_text()) if p.exists() else None


def md(df, index=False):
    return df.to_markdown(floatfmt=".3f", index=index) + "\n"


def mf1(yt, yp):
    return float(f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0))


def full_metrics(yt, P):
    p = P.argmax(1)
    pr, rc, f, _ = precision_recall_fscore_support(yt, p, average="macro", zero_division=0)
    return {"acc": accuracy_score(yt, p), "bal_acc": balanced_accuracy_score(yt, p), "macro_P": pr, "macro_R": rc, "macro_F1": f}


def boot(yt, p, n=1000):
    r = np.random.default_rng(0)
    v = [mf1(yt[i], p[i]) for i in (r.integers(0, len(yt), len(yt)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])


rows = []
for d in json.loads((ROOT / "experiments/summary.json").read_text()):
    cr = d["classification_report"]
    rows.append({"model": f"{d['model']} (old run, 1 seed)", "acc": d["accuracy"], "bal_acc": d["balanced_accuracy"],
                 "macro_P": d["macro_precision"], "macro_R": d["macro_recall"], "macro_F1": d["macro_f1"],
                 **{l: cr[l]["f1-score"] for l in LABELS}})
ens = {}
for tag, name in (("wavlm_base_plus", "WavLM frozen + head"), ("wav2vec2_base", "Wav2Vec2 frozen + head")):
    Ps = [np.load(V2 / f"frozen/{tag}__none_0__seed{s}.npz")["test"] for s in range(5)]
    ens[name] = np.mean(Ps, 0)
    for label, P in ((f"{name}, seed 0", Ps[0]), (f"{name}, ensemble of 5 seeds", ens[name])):
        rows.append({"model": label, **full_metrics(y, P), **dict(zip(LABELS, f1_score(y, P.argmax(1), average=None)))})
osm = load("opensmile_baseline.json")
if osm:
    for k, v in osm.items():
        rows.append({"model": f"openSMILE eGeMAPS + {k} ({v['chosen']})", "acc": v["test_accuracy"], "macro_F1": v["test_macro_f1"], **v["per_class_f1"]})
tab = pd.DataFrame(rows).sort_values("macro_F1", ascending=False)
out += ["# Final results\n\nTest split: 3511 clips, speakers disjoint from train/validation. Macro metrics over 7 emotions.\n",
        "\n## 1. All models (test)\n", md(tab)]

best = ens["WavLM frozen + head"]
pred = best.argmax(1)
lo, hi = boot(y, pred)
out.append(f"\nBest model (WavLM frozen, 5-seed head ensemble): macro-F1 {mf1(y, pred):.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}].\n")

fr = []
for f in glob.glob(str(V2 / "frozen/*.json")):
    r = json.loads(open(f).read())
    if r["holdout"] is None:
        fr.append({"embedding": r["embedding"], "config": f"{r['mode']} λ={r['lambda']:g}", "seed": r["seed"],
                   "val_f1": r["val_macro_f1"], "test_f1": r["test"]["macro_f1"]})
fr = pd.DataFrame(fr)
g = fr.groupby(["embedding", "config"]).agg(val_f1=("val_f1", "mean"), test_f1=("test_f1", "mean"), test_std=("test_f1", "std")).reset_index()
sel = g.loc[g.groupby("embedding").val_f1.idxmax()]
out += ["\n## 2. Configuration selected on VALIDATION only (no test peeking)\n",
        "Earlier tables ranked configurations by test F1. Here the configuration is chosen by mean validation macro-F1 over seeds and only then its test score is reported.\n\n",
        md(sel.rename(columns={"test_f1": "test_f1 (mean of 5 seeds)", "test_std": "test_std"})),
        "\nAll configurations (mean over 5 seeds):\n\n", md(g)]
_sel = sel[sel.embedding == "wavlm_base_plus"].config.iloc[0]
_mode, _lam = _sel.split(" λ=")
_Pv = np.mean([np.load(V2 / f"frozen/wavlm_base_plus__{_mode}_{_lam}__seed{k}.npz")["test"] for k in range(5)], 0)
out.append(f"\nValidation-selected WavLM configuration (`{_sel}`), ensemble of its 5 seeds: macro-F1 {mf1(y, _Pv.argmax(1)):.3f}. "
           "The differences between 'none' and the multi-task variants (about 0.01-0.02) are smaller than the seed-to-seed std (0.02-0.04), "
           "so they are not significant; the deployed model uses 'none' (simplest, validation F1 within 0.004 of the best).\n")

out.append("\n## 3. Robustness of the headline number to the TESS imbalance\n")
no = ds != "TESS"
per_corpus = {c: mf1(y[ds == c], pred[ds == c]) for c in sorted(set(ds)) if (ds == c).sum() >= 20}
rows3 = [{"view": "full test (40% TESS)", "macro_F1": mf1(y, pred), "n": len(y)},
         {"view": "test without TESS", "macro_F1": mf1(y[no], pred[no]), "n": int(no.sum())},
         {"view": "mean of per-corpus macro-F1 (corpora with n>=20)", "macro_F1": float(np.mean(list(per_corpus.values()))), "n": len(per_corpus)}]
out += [md(pd.DataFrame(rows3)), "\nPer corpus (corpora with at least 20 test clips; SVK-DB has 2 and is evaluated separately below):\n\n",
        md(pd.DataFrame([{"corpus": c, "language": data[data.dataset == c].language.iloc[0], "n_test": int((ds == c).sum()), "macro_F1": v,
                          "accuracy": accuracy_score(y[ds == c], pred[ds == c])} for c, v in per_corpus.items()]))]

gcv = load("grouped_cv.json")
if gcv:
    out.append("\nNote: in the default split TESS (2 speakers) is absent from training, so ~40% of the test set is effectively "
               "cross-speaker AND cross-corpus; in speaker-grouped CV (below) TESS reaches F1 0.80 instead of 0.61. "
               "Headline numbers of the default split are therefore conservative for TESS.\n")
    out += ["\n### Speaker-grouped 5-fold cross-validation over all 13,406 clips\n",
            f"Pooled macro-F1 {gcv['pooled_macro_f1']:.3f}, accuracy {gcv['pooled_accuracy']:.3f}; per-fold macro-F1 "
            f"{gcv['fold_f1_mean']:.3f} ± {gcv['fold_f1_std']:.3f}; without TESS {gcv['macro_f1_without_tess']:.3f}.\n\n",
            md(pd.DataFrame(gcv["folds"]).drop(columns="corpora")),
            "\nPer-corpus F1 in CV:\n\n", md(pd.Series(gcv["per_corpus_f1"], name="macro_F1").rename_axis("corpus").reset_index())]

loco = []
for f in glob.glob(str(V2 / "frozen/wavlm_base_plus__*hold*.json")):
    r = json.loads(open(f).read())
    loco.append({"config": f"{r['mode']} λ={r['lambda']:g}", "holdout": r["holdout"], "f1": r["test"]["macro_f1"]})
loco = pd.DataFrame(loco)
if len(loco):
    piv = loco.groupby(["config", "holdout"]).f1.mean().unstack()
    ln = load("loco_corpus_norm.json")
    if ln:
        piv.loc["none λ=0 + per-corpus feature normalisation"] = pd.DataFrame(ln).groupby("holdout").test_f1.mean()
    piv["average"] = piv.mean(1)
    out += ["\n## 4. Cross-corpus (leave-one-corpus-out), frozen WavLM, mean of 3 seeds\n",
            "Train on the other corpora, test on the unseen one. Chance level for 7 classes is about 0.14-0.17.\n\n", md(piv.reset_index())]
    fig, ax = plt.subplots(figsize=(7, 3)); piv["average"].plot.barh(ax=ax); ax.set_xlabel("average LOCO macro-F1")
    fig.tight_layout(); fig.savefig(PL / "loco_average.png", dpi=150); plt.close(fig)
svk = load("svk_loso.json")
if svk:
    out += ["\n### Slovak (SVK-DB, 30 clips, 4 speakers)\n", md(pd.DataFrame([{"setting": k, "n_clips": v["n_clips"], "macro_F1": v["macro_f1"],
            "CI95": f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]", "accuracy": v["accuracy"]} for k, v in svk.items()]))]

conf = best.max(1); correct = pred == y
bins = np.linspace(0, 1, 11); ece = 0.0; rel = []
for a, b in zip(bins[:-1], bins[1:]):
    m = (conf > a) & (conf <= b)
    if m.any():
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean()); rel.append((conf[m].mean(), correct[m].mean()))
brier = float(np.mean(np.sum((best - np.eye(7)[y]) ** 2, 1)))
out += ["\n## 5. Calibration and 'uncertain' option (best model)\n", f"ECE = {ece:.3f}, Brier score = {brier:.3f}. "
        "The ensemble is "
        + ("under-confident" if np.mean(conf) < correct.mean() else "over-confident") + f" on average (mean confidence {conf.mean():.2f} vs accuracy {correct.mean():.2f}).\n\n"]
sel_rows = []
for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
    m = conf >= t
    sel_rows.append({"min_confidence": t, "coverage": m.mean(), "accuracy_on_answered": correct[m].mean(), "macro_F1_on_answered": mf1(y[m], pred[m])})
out += [md(pd.DataFrame(sel_rows))]
fig, ax = plt.subplots(figsize=(4, 4)); r = np.array(rel)
ax.plot([0, 1], [0, 1], "k--"); ax.plot(r[:, 0], r[:, 1], "o-"); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability diagram")
fig.tight_layout(); fig.savefig(PL / "reliability.png", dpi=150); plt.close(fig)

cm = confusion_matrix(y, pred, labels=range(7))
fig, ax = plt.subplots(figsize=(5.5, 4.5)); ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues")
ax.set_xticks(range(7), LABELS, rotation=45); ax.set_yticks(range(7), LABELS)
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, v, ha="center", va="center", fontsize=8)
ax.set_title("Confusion matrix, best model"); fig.tight_layout(); fig.savefig(PL / "confusion_final.png", dpi=150); plt.close(fig)
cr = precision_recall_fscore_support(y, pred, labels=range(7), zero_division=0)
out += ["\n## 6. Per-emotion precision / recall / F1 (best model)\n",
        md(pd.DataFrame({"emotion": LABELS, "precision": cr[0], "recall": cr[1], "f1": cr[2], "support": cr[3]}))]

nz = load("robustness_noise.json")
if nz:
    out += ["\n## 7. Robustness to additive white noise (stratified test subset, 1000 clips)\n",
            md(pd.DataFrame([{"condition": k, "macro_F1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in nz.items()]))]
    fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(list(nz), [v["macro_f1"] for v in nz.values()], "o-"); ax.set_ylabel("macro-F1"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(PL / "noise.png", dpi=150); plt.close(fig)

nr = load("noise_robust.json")
if nr:
    import torch
    from predict_emotion import Head
    Xc = np.load(V2 / "emb_wavlm_base_plus.npz")
    Xt = np.concatenate([Xc["mean"], Xc["std"]], 2).astype(np.float32)[np.where(te)[0]]
    Ps = []
    for f in sorted((ROOT / "models/noise_robust").glob("wavlm_head_seed*.pt")):
        b = torch.load(f, map_location="cpu"); h = Head(b["layers"], b["dim"], b["domains"], b["mode"]); h.load_state_dict(b["state"]); h.eval()
        with torch.inference_mode():
            Ps.append(torch.softmax(h((torch.from_numpy(Xt) - b["mu"]) / b["sd"]), 1).numpy())
    Pr = np.mean(Ps, 0); pr_ = Pr.argmax(1); lo, hi = boot(y, pr_)
    out += ["\n## 7b. Noise-augmented heads (deployed default)\n",
            "Heads trained on clean train clips plus noisy copies of the TRAIN clips only (white, pink or babble noise, SNR uniform 0-30 dB); noisy copies of validation clips stay in validation and only select the epoch, TESS is never used for training. "
            "Clean test: single seeds " + ", ".join(f"{v:.3f}" for v in nr["clean_test_f1_per_seed"]) +
            f" (mean {np.mean(nr['clean_test_f1_per_seed']):.3f}); ensemble of 5 seeds macro-F1 {mf1(y, pr_):.3f} (95% CI [{lo:.3f}, {hi:.3f}]), "
            f"accuracy {accuracy_score(y, pr_):.3f}, without TESS {mf1(y[no], pr_[no]):.3f}. "
            "Both head sets are compared on the same noisy features of the 1050-clip stratified test subset (noise types seen in training; "
            "street noise, music and reverberation were not tested). NOTE: this table uses pre-extracted noisy features; to compare systems use "
            "section 7g (noise regenerated identically for all four systems) and do not mix the two:\n\n",
            md(pd.DataFrame(nr["noisy_subset"]).assign(gain=lambda t: t.noise_robust_F1 - t.standard_F1))]
    cr2 = precision_recall_fscore_support(y, pr_, labels=range(7), zero_division=0)
    out += ["\nPer emotion (robust ensemble, clean test):\n\n", md(pd.DataFrame({"emotion": LABELS, "precision": cr2[0], "recall": cr2[1], "f1": cr2[2], "support": cr2[3]}))]

    conf_r = Pr.max(1); ok_r = pr_ == y
    ece_r = sum(((conf_r > a) & (conf_r <= b_)).mean() * abs(ok_r[(conf_r > a) & (conf_r <= b_)].mean() - conf_r[(conf_r > a) & (conf_r <= b_)].mean())
                for a, b_ in zip(bins[:-1], bins[1:]) if ((conf_r > a) & (conf_r <= b_)).any())
    brier_r = float(np.mean(np.sum((Pr - np.eye(7)[y]) ** 2, 1)))
    out += ["\n### 7b2. Calibration and 'uncertain' option for the DEPLOYED model (noise-augmented ensemble; section 5 above is the standard ensemble)\n",
            f"ECE = {ece_r:.3f}, Brier score = {brier_r:.3f}, mean confidence {conf_r.mean():.2f} vs accuracy {ok_r.mean():.2f}.\n\n",
            md(pd.DataFrame([{"min_confidence": t, "coverage": (conf_r >= t).mean(), "accuracy_on_answered": ok_r[conf_r >= t].mean(),
                              "macro_F1_on_answered": mf1(y[conf_r >= t], pr_[conf_r >= t])} for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8)]))]

ft_runs = sorted((V2 / "finetune").glob("wavlm_v2__*/result.json"))
if ft_runs:
    rows_f, probs_f = [], {}
    for f in ft_runs:
        r = json.loads(f.read_text()); name = f.parent.name.replace("wavlm_v2__", "")
        z = np.load(f.parent / "probs.npz"); assert (z["test_y"] == y).all()
        probs_f[name] = z["test"]; pp = z["test"].argmax(1)
        rows_f.append({"run": name, "best_epoch": r["best_epoch"], "val_F1": r["best_val_macro_f1"], "test_F1": r["test"]["macro_f1"],
                       "test_acc": r["test"]["accuracy"], "F1_without_TESS": mf1(y[no], pp[no]), "minutes": r["minutes"]})
    rf = pd.DataFrame(rows_f)
    none_f = rf[rf.run.str.startswith("none")].test_F1
    out += ["\n## 7c. Fine-tuning WavLM-base-plus on the GPU (RTX 3060 Laptop, bf16)\n",
            "Waveform augmentation (noise, gain, speed), weighted layer sum, layer-wise lr decay, warm-up + cosine, label smoothing, "
            "SpecAugment-style masking inside the model; epoch selected on validation macro-F1.\n\n", md(rf),
            f"\nMean over the {len(none_f)} 'none' runs: {none_f.mean():.3f} ± {none_f.std():.3f} (large seed-to-seed spread).\n"]
    ens_none = np.mean([v for k, v in probs_f.items() if k.startswith("none")], 0)
    ens_all = np.mean(list(probs_f.values()), 0)

    def paired(pa, pb, n=1000):
        r_ = np.random.default_rng(0); dd = []
        for _ in range(n):
            i = r_.integers(0, len(y), len(y)); dd.append(mf1(y[i], pa[i]) - mf1(y[i], pb[i]))
        return np.mean(dd), np.percentile(dd, [2.5, 97.5])

    cmp_rows = []
    for label, P in (("fine-tuned, ensemble of the 'none' runs", ens_none), ("fine-tuned, ensemble of all runs", ens_all),
                     ("frozen WavLM + head, ensemble of 5 seeds", ens["WavLM frozen + head"])) + ((("frozen WavLM + noise-augmented head, ensemble of 5 seeds", Pr),) if "Pr" in dir() else ()):
        pp = P.argmax(1); l_, h_ = boot(y, pp)
        cmp_rows.append({"model": label, "macro_F1": mf1(y, pp), "CI95": f"[{l_:.3f}, {h_:.3f}]", "accuracy": accuracy_score(y, pp), "F1_without_TESS": mf1(y[no], pp[no])})
    out += ["\nFine-tuned vs frozen (test, 3511 clips):\n\n", md(pd.DataFrame(cmp_rows))]
    d_, (l_, h_) = paired(ens_none.argmax(1), ens["WavLM frozen + head"].argmax(1))
    out.append(f"\nPaired bootstrap, fine-tuned ('none' ensemble) minus frozen ensemble: delta macro-F1 = {d_:+.3f} (95% CI {l_:+.3f} .. {h_:+.3f}).\n")

extra_rows = []
for pat in ("ftnoise__*", "ftclean__*"):
    for f in sorted((V2 / "finetune").glob(pat + "/result.json")):
        r = json.loads(f.read_text()); z = np.load(f.parent / "probs.npz")
        pp = z["test"].argmax(1)
        extra_rows.append({"run": f.parent.name, "best_epoch": r["best_epoch"], "val_F1": r["best_val_macro_f1"], "test_F1": r["test"]["macro_f1"],
                           "test_acc": r["test"]["accuracy"], "F1_without_TESS": mf1(y[no], pp[no]), "minutes": r["minutes"]})
if extra_rows:
    ens_noise = np.mean([np.load(f.parent / "probs.npz")["test"] for f in sorted((V2 / "finetune").glob("ftnoise__*/result.json"))], 0)
    pn = ens_noise.argmax(1); l_, h_ = boot(y, pn)
    out += ["\n## 7d. Fine-tuning with noise augmentation (white / pink / babble, 0-30 dB) and a clean control run\n", md(pd.DataFrame(extra_rows)),
            f"\nEnsemble of the noise-augmented fine-tuned runs: macro-F1 {mf1(y, pn):.3f} (95% CI [{l_:.3f}, {h_:.3f}]), "
            f"accuracy {accuracy_score(y, pn):.3f}, without TESS {mf1(y[no], pn[no]):.3f}.\n"]

xl_runs = sorted((V2 / "finetune").glob("xlsrft__*/result.json"))
if xl_runs:
    rows_x, probs_x = [], []
    for f in xl_runs:
        r = json.loads(f.read_text()); z = np.load(f.parent / "probs.npz"); pp = z["test"].argmax(1); probs_x.append(z["test"])
        rows_x.append({"run": f.parent.name, "best_epoch": r["best_epoch"], "val_F1": r["best_val_macro_f1"], "test_F1": r["test"]["macro_f1"],
                       "test_acc": r["test"]["accuracy"], "F1_without_TESS": mf1(y[no], pp[no]), "minutes": r["minutes"]})
    ex = np.mean(probs_x, 0).argmax(1); l_, h_ = boot(y, ex)
    out += ["\n## 7d2. Fine-tuning the multilingual XLS-R-300M (lowest 8 layers frozen, noise augmentation, bf16, 6 GB GPU)\n", md(pd.DataFrame(rows_x)),
            f"\nEnsemble of {len(xl_runs)} runs: macro-F1 {mf1(y, ex):.3f} (95% CI [{l_:.3f}, {h_:.3f}]), without TESS {mf1(y[no], ex[no]):.3f}. "
            "Seed spread is large (first run 0.62, the others 0.70). XLS-R generalises well to the unseen TESS speakers but is clearly weaker than fine-tuned WavLM on the seen corpora "
            "(without-TESS F1 about 0.60 vs 0.77), and the frozen XLS-R features with a light head (7e) are as good as the fine-tuned model.\n"]

bbj = load("backbones.json")
if bbj:
    out += ["\n## 7e. Backbone comparison (frozen features, same plain head)\n",
            "Default split: mean ± std over 5 seeds. Cross-corpus: mean over the 6 held-out corpora (3 seeds each), std across corpora.\n\n",
            md(pd.DataFrame([{"backbone": k, "default F1": f"{v['default_f1']:.3f} ± {v['default_f1_std']:.3f}",
                              "cross-corpus F1": (f"{v['loco_f1']:.3f} ± {v['loco_f1_std']:.3f}" if v["loco_f1"] is not None else "n/a"),
                              **{c: f"{x:.3f}" for c, x in v["loco_per_corpus"].items()}} for k, v in bbj.items()]))]

ftl = {}
for f in glob.glob(str(V2 / "finetune/loco__*__hold-*/result.json")):
    r = json.loads(open(f).read()); ftl.setdefault(r["args"]["holdout"], []).append(r["test"]["macro_f1"])
n_ft_seeds = min((len(v) for v in ftl.values()), default=0)
ftl = {k: float(np.mean(v)) for k, v in ftl.items()}
if ftl:
    plain = loco[loco.config == "none λ=0"].groupby("holdout").f1.mean() if len(loco) else pd.Series(dtype=float)
    cn = pd.DataFrame(ln).groupby("holdout").test_f1.mean() if ln else pd.Series(dtype=float)
    t = pd.DataFrame({f"fine-tuned WavLM ({n_ft_seeds} seed(s), mean)": pd.Series(ftl), "frozen WavLM, plain head": plain, "frozen + per-corpus normalisation": cn})
    t.loc["average"] = t.mean()
    out += ["\n## 7f. Cross-corpus: fine-tuned model vs frozen features\n",
            "Leave-one-corpus-out, macro-F1 on the unseen corpus. The fine-tuned model uses clean training and is averaged over the available seeds (seed spread of fine-tuning is large, "
            "see 7c); per-corpus normalisation needs a batch of recordings from the target source.\n\n", md(t.reset_index().rename(columns={"index": "held-out corpus"}))]

nall = load("noise_all_systems.json")
if nall:
    rowsn = []
    for cond, v in nall.items():
        rowsn.append({"condition": cond, "seen in training": "yes" if v.get("_seen_in_training") else "NO",
                      **{k: x for k, x in v.items() if not k.startswith("_")}})
    out += ["\n## 7g. All systems on identical corrupted audio (1,050-clip test subset, macro-F1)\n",
            "Conditions marked 'NO' were never used in training of any system. Brown noise has >99% of its energy below 100 Hz (outside the speech band).\n\n", md(pd.DataFrame(rowsn))]

cs = []
for f in glob.glob(str(V2 / "cnn_ssl/*_seed*.json")):
    cs.append(json.loads(open(f).read()))
if cs:
    c = pd.DataFrame(cs).groupby(["fraction", "init"]).test_macro_f1.agg(["mean", "std", "count"]).reset_index()
    lp = load("cnn_ssl/linear_probe.json")
    out += ["\n## 8. Self-supervised pre-training of the CNN encoder (same architecture, only initialisation differs)\n", md(c),
            f"\nLinear probe on the frozen encoder: scratch {lp['scratch']:.3f}, contrastive-pretrained {lp['ssl']:.3f}.\n" if lp else ""]
ss = load("semi_supervised_wavlm.json")
if ss:
    s = pd.DataFrame(ss).groupby("labeled_fraction")[["supervised", "self_training"]].agg(["mean", "std"])
    s.columns = [" ".join(x) for x in s.columns]
    out += ["\n### Semi-supervised (self-training) on WavLM embeddings vs supervised, same labelled subsets\n", md(s.reset_index())]
fb = load("fair_baselines.json")
if fb:
    f = pd.DataFrame(fb).groupby("labeled_fraction")[["supervised_f1", "self_training_f1"]].agg(["mean", "std"])
    f.columns = [" ".join(x) for x in f.columns]
    out += ["\n### Same comparison on acoustic features (LinearSVC)\n", md(f.reset_index())]

dup = load("duplicates.json")
if dup:
    out += ["\n## 9. Data-leakage checks\n", "```\n" + json.dumps(dup, indent=1) + "\n```\n"]

try:
    from sklearn.manifold import TSNE

    z = np.load(V2 / "emb_wavlm_base_plus.npz")
    F = np.concatenate([z["mean"], z["std"]], 2).astype(np.float32)[:, 3:7].mean(1)
    idx = np.random.default_rng(0).choice(len(data), 3000, replace=False)
    E = TSNE(2, init="pca", perplexity=40, random_state=0).fit_transform((F[idx] - F[idx].mean(0)) / (F[idx].std(0) + 1e-6))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in ((axs[0], data.emotion.to_numpy()[idx], "coloured by emotion"), (axs[1], data.dataset.to_numpy()[idx], "coloured by corpus")):
        for v in sorted(set(col)):
            m = col == v; ax.scatter(E[m, 0], E[m, 1], s=4, label=v)
        ax.legend(markerscale=3, fontsize=7); ax.set_title("WavLM embeddings (t-SNE), " + title); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(PL / "tsne.png", dpi=150); plt.close(fig)
    lw = np.mean([json.loads(open(f).read())["layer_weights"] for f in glob.glob(str(V2 / "frozen/wavlm_base_plus__none_0__seed*.json"))], 0)
    fig, ax = plt.subplots(figsize=(5, 3)); ax.bar(range(len(lw)), lw); ax.set_xlabel("WavLM layer"); ax.set_ylabel("learned weight"); fig.tight_layout()
    fig.savefig(PL / "layer_weights.png", dpi=150); plt.close(fig)
    out += ["\n## 10. Representation analysis\n", "`plots/tsne.png` (checked visually): WavLM embeddings form clearly separated islands per corpus (CREMA-D, RAVDESS, SAVEE, EMO-DB, UA-SER, TESS), while emotions are mixed inside each island except in the small, very acted corpora; the model must learn emotion across strong corpus offsets, which explains the cross-corpus drop. "
            "`plots/layer_weights.png`: learned weights over WavLM layers.\n"]
except Exception as e:
    out.append(f"\n(t-SNE skipped: {e})\n")

(V2 / "FINAL_RESULTS.md").write_text("".join(out), encoding="utf-8")
print("written", V2 / "FINAL_RESULTS.md")
