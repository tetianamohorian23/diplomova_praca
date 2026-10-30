"""All result figures -> experiments/v2/figures/*.png (thesis-ready, one consistent style).

Style follows the dataviz guidance: validated categorical palette in fixed order, one-hue sequential ramp for magnitude,
single y-axis, thin marks, quiet grid, text in ink colours, direct labels where they fit. Missing inputs are skipped.
"""
from __future__ import annotations

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

from common_splits import CACHE, LABELS, ROOT, load_splits

V2 = CACHE.parent
FIG = V2 / "figures"
FIG.mkdir(exist_ok=True)

# ---- palette (reference categorical order; slots are bound to entities, never cycled)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GRAY = "#8d8c86"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
EMO_COLOR = dict(zip(LABELS, [RED, GREEN, VIOLET, YELLOW, GRAY, BLUE, MAGENTA]))  # fixed per emotion
FAMILY = {"classical": GRAY, "cnn": ORANGE, "frozen": BLUE, "finetuned": VIOLET, "ensemble": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "center", "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 6,
})


def load(path):
    p = V2 / path
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def mf1(y, p):
    return float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0))


def boot(y, p, n=500):
    r = np.random.default_rng(0)
    v = [mf1(y[i], p[i]) for i in (r.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])


data = load_splits()
te = (data.experiment_split == "test").to_numpy()
y = data.label_id.to_numpy()[te]
ds = data.dataset.to_numpy()[te]


def frozen_probs(tag, mode="none", lam="0", seeds=5):
    ps = [V2 / f"frozen/{tag}__{mode}_{lam}__seed{s}.npz" for s in range(seeds)]
    return [np.load(p)["test"] for p in ps if p.exists()]


def robust_probs():
    import torch
    from predict_emotion import Head
    z = np.load(V2 / "emb_wavlm_base_plus.npz")
    X = torch.from_numpy(np.concatenate([z["mean"], z["std"]], 2).astype(np.float32)[np.where(te)[0]])
    out = []
    for f in sorted((ROOT / "models/noise_robust").glob("wavlm_head_seed*.pt")):
        b = torch.load(f, map_location="cpu"); h = Head(b["layers"], b["dim"], b["domains"], b["mode"]); h.load_state_dict(b["state"]); h.eval()
        with torch.inference_mode():
            out.append(torch.softmax(h((X - b["mu"]) / b["sd"]), 1).numpy())
    return out


def finetune_probs(prefix="wavlm_v2__none"):
    ps = []
    for f in sorted((V2 / "finetune").glob(f"{prefix}*/probs.npz")):
        if "hold-" in f.parent.name:
            continue
        z = np.load(f)
        if len(z["test"]) == len(y):
            ps.append(z["test"])
    return ps


# ------------------------------------------------------------------ models used in several figures
old = {d["model"]: d for d in json.loads((ROOT / "experiments/summary.json").read_text())}
W = frozen_probs("wavlm_base_plus"); W2 = frozen_probs("wav2vec2_base"); R = robust_probs(); FT = finetune_probs()
ens = {}
if W: ens["WavLM frozen (5-seed ensemble)"] = np.mean(W, 0)
if R: ens["WavLM frozen + noise aug. (ensemble)"] = np.mean(R, 0)
if FT: ens["WavLM fine-tuned (ensemble)"] = np.mean(FT, 0)
FTN = finetune_probs("ftnoise__none")
if FTN: ens["WavLM fine-tuned + noise aug. (ensemble)"] = np.mean(FTN, 0)
XL = finetune_probs("xlsrft__none")
if XL: ens["XLS-R fine-tuned + noise aug. (ensemble)"] = np.mean(XL, 0)

# ================================================================== 01 model overview
rows = []  # (label, family, value, lo, hi)


def add_row(label, fam, vals=None, value=None, ci=None):
    if vals is not None:
        value = float(np.mean(vals)); ci = (value - float(np.std(vals)), value + float(np.std(vals)))
    rows.append((label, fam, value, ci[0] if ci is not None else value, ci[1] if ci is not None else value))


osm = load("opensmile_baseline.json")
if osm: add_row("openSMILE eGeMAPS + logreg", "classical", value=osm["logreg"]["test_macro_f1"])
add_row("Logistic regression (acoustic features)", "classical", value=old["logistic_regression"]["macro_f1"])
add_row("Semi-supervised SVM", "classical", value=old["semi_supervised_svm"]["macro_f1"])
add_row("CNN-BiLSTM (multi-task)", "cnn", value=old["cnn_bilstm"]["macro_f1"])
cs = [json.loads(open(f).read()) for f in glob.glob(str(V2 / "cnn_ssl/*_seed*.json"))]
cs = pd.DataFrame(cs)
if len(cs):
    add_row("CNN, from scratch", "cnn", vals=cs[(cs.init == "scratch") & (cs.fraction == 1.0)].test_macro_f1.to_numpy())
    add_row("CNN, contrastive pre-training", "cnn", vals=cs[(cs.init == "ssl") & (cs.fraction == 1.0)].test_macro_f1.to_numpy())
if W2: add_row("Wav2Vec2 frozen + head (mean of 5 seeds)", "frozen", vals=[f1_score(y, p.argmax(1), average="macro") for p in W2])
add_row("Wav2Vec2 fine-tuned (CPU, 1 run)", "finetuned", value=old["wav2vec2"]["macro_f1"])
add_row("WavLM fine-tuned, first run (CPU, 1 run)", "finetuned", value=old["wavlm_multitask"]["macro_f1"])
if W: add_row("WavLM frozen + head (mean of 5 seeds)", "frozen", vals=[f1_score(y, p.argmax(1), average="macro") for p in W])
if FT: add_row("WavLM fine-tuned, GPU (mean of runs)", "finetuned", vals=[f1_score(y, p.argmax(1), average="macro") for p in FT])
for name, P in ens.items():
    add_row(name, "ensemble", value=mf1(y, P.argmax(1)), ci=boot(y, P.argmax(1)))
rows.sort(key=lambda r: r[2])
fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.4))
for i, (lab, fam, v, lo, hi) in enumerate(rows):
    ax.barh(i, v, color=FAMILY[fam], height=0.62)
    ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
    ax.text(max(hi, v) + 0.008, i, f"{v:.3f}", va="center", color=INK2, fontsize=9)
ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_xlim(0, 0.88); ax.set_xlabel("macro-F1 on the test split (3,511 clips)")
ax.grid(axis="y", visible=False)
ax.set_title("Test macro-F1 of all models")
ax.text(0, -0.09, "whiskers: ± std over seeds (single models) or 95% bootstrap CI (ensembles)", transform=ax.transAxes, fontsize=8, color=INK2)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()]
ax.legend(handles, ["classical", "CNN", "frozen SSL + head", "fine-tuned SSL", "ensembles (deployed candidates)"], loc="lower right")
save(fig, "01_model_comparison.png")

# ================================================================== 02 per-emotion F1 heatmap
cand = {"openSMILE + logreg": None, "CNN-BiLSTM": "cnn_bilstm", "Wav2Vec2 fine-tuned": "wav2vec2", "WavLM fine-tuned (CPU run)": "wavlm_multitask"}
M, names = [], []
if osm:
    M.append([osm["logreg"]["per_class_f1"][l] for l in LABELS]); names.append("openSMILE + logreg")
for lab, key in list(cand.items())[1:]:
    M.append([old[key]["classification_report"][l]["f1-score"] for l in LABELS]); names.append(lab)
for name, P in ens.items():
    M.append(list(f1_score(y, P.argmax(1), average=None))); names.append(name)
M = np.array(M)
fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 1.6))
im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(7), LABELS); ax.set_yticks(range(len(names)), names); ax.grid(False)
for (i, j), v in np.ndenumerate(M):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else INK, fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("F1", color=INK2); cb.outline.set_visible(False)
ax.set_title("F1 per emotion")
save(fig, "02_per_emotion_f1.png")

# ================================================================== 03 confusion matrices of the strongest models
show = [(n, P) for n, P in ens.items() if "noise aug" in n or "fine-tuned (ens" in n][:3]
if show:
    fig, axs = plt.subplots(1, len(show), figsize=(5.4 * len(show), 4.8), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, (n, P) in zip(axs, show):
        cm = confusion_matrix(y, P.argmax(1), labels=range(7)); nrm = cm / cm.sum(1, keepdims=True)
        ax.imshow(nrm, cmap=SEQ, vmin=0, vmax=1); ax.grid(False)
        ax.set_xticks(range(7), LABELS, rotation=45, ha="right"); ax.set_yticks(range(7), LABELS)
        for (i, j), v in np.ndenumerate(nrm):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > 0.5 else INK, fontsize=8)
        ax.set_title(n, fontsize=9); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle("Confusion matrices (row-normalised)", x=0.5, ha="center", fontweight="bold")
    save(fig, "03_confusion_matrices.png")

# ================================================================== 04 corpus-task ablation
fr = []
for f in glob.glob(str(V2 / "frozen/*.json")):
    r = json.loads(open(f).read())
    if r["holdout"] is None:
        fr.append({"emb": r["embedding"], "mode": r["mode"], "lam": r["lambda"], "f1": r["test"]["macro_f1"]})
fr = pd.DataFrame(fr)
if len(fr):
    embs = [e for e in ("wavlm_base_plus", "wav2vec2_base") if e in set(fr.emb)]
    fig, axs = plt.subplots(1, len(embs), figsize=(5.6 * len(embs), 3.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, e in zip(axs, embs):
        g = fr[fr.emb == e]; base = g[g["mode"] == "none"].f1
        ax.axhspan(base.mean() - base.std(), base.mean() + base.std(), color=GRAY, alpha=0.18, lw=0)
        ax.axhline(base.mean(), color=GRAY, lw=1.5)
        ax.text(0.98, 0.97, "grey band: no corpus task (mean ± std)", color=INK2, fontsize=8, ha="right", va="top", transform=ax.transAxes)
        for mode, col, lab in (("aux", BLUE, "multi-task: corpus classification"), ("grl", ORANGE, "adversarial (gradient reversal)")):
            h = g[g["mode"] == mode].groupby("lam").f1.agg(["mean", "std"])
            ax.errorbar(range(len(h)), h["mean"], yerr=h["std"], color=col, marker="o", capsize=3, label=lab)
            ax.set_xticks(range(len(h)), [f"{v:g}" for v in h.index])
        ax.set_xlabel("corpus-loss weight λ"); ax.set_title({"wavlm_base_plus": "WavLM-base-plus", "wav2vec2_base": "Wav2Vec2-base"}[e])
    axs[0].set_ylabel("test macro-F1 (5 seeds)"); axs[0].legend(loc="lower left")
    fig.suptitle("Does a corpus-related auxiliary task help? (frozen features + head)", x=0.5, ha="center", fontweight="bold", y=1.02)
    save(fig, "04_corpus_task_ablation.png")

# ================================================================== 05 leave-one-corpus-out
lo = []
for f in glob.glob(str(V2 / "frozen/wavlm_base_plus*hold*.json")):
    r = json.loads(open(f).read()); lo.append({"cfg": f"{r['mode']} {r['lambda']:g}", "hold": r["holdout"], "f1": r["test"]["macro_f1"]})
lo = pd.DataFrame(lo)
ln = load("loco_corpus_norm.json")
if len(lo):
    P = lo.groupby(["cfg", "hold"]).f1.mean().unstack()
    series = [("none 0", "WavLM frozen, plain", BLUE), ("aux 0.1", "+ multi-task corpus loss", ORANGE), ("grl 0.05", "+ adversarial corpus loss", AQUA)]
    data_series = [(lab, col, P.loc[k]) for k, lab, col in series if k in P.index]
    if ln:
        data_series.append(("+ per-corpus feature normalisation", VIOLET, pd.DataFrame(ln).groupby("holdout").test_f1.mean()))
    ftl = {}
    for f in glob.glob(str(V2 / "finetune/loco__*__hold-*/result.json")):
        r = json.loads(open(f).read()); ftl.setdefault(r["args"]["holdout"], []).append(r["test"]["macro_f1"])
    if ftl:
        n_seeds = min(len(v) for v in ftl.values())
        data_series.append((f"fine-tuned WavLM ({n_seeds} seed{'s' if n_seeds > 1 else ''})", RED, pd.Series({k: np.mean(v) for k, v in ftl.items()})))
    corp = list(P.columns)
    fig, ax = plt.subplots(figsize=(10, 4.2)); w = 0.8 / len(data_series)
    for k, (lab, col, s) in enumerate(data_series):
        vals = [s.get(c, np.nan) for c in corp]
        ax.bar(np.arange(len(corp)) + k * w - 0.4 + w / 2, vals, w * 0.92, color=col, label=lab)
    ax.axhline(1 / 7, color=INK2, lw=1, ls=(0, (4, 3)), label="chance level ≈ 0.14")
    ax.set_xticks(range(len(corp)), corp); ax.set_ylabel("macro-F1 on the unseen corpus"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper right"); ax.set_ylim(0, 0.98)
    ax.set_title("Cross-corpus (leave-one-corpus-out): training excludes the tested corpus")
    save(fig, "05_cross_corpus.png")

# ================================================================== 06 noise robustness
# All four systems come from ONE evaluation (evaluate_noise_all.py -> noise_all_systems.json): identical corrupted audio for every system.
nw = load("noise_all_systems.json")
if nw:
    cond_groups = [("white noise", "white", [20, 10, 5, 0]), ("babble (other speech)", "babble", [20, 10, 5]), ("pink noise", "pink", [10])]
    systems = [("frozen, clean training", GRAY, "o"), ("frozen, noise-augmented", BLUE, "o"),
               ("fine-tuned, clean training", ORANGE, "s"), ("fine-tuned, noise-augmented", VIOLET, "s")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True, gridspec_kw={"width_ratios": [4, 3, 1.4]})
    for ax, (title, key, snr) in zip(axs, cond_groups):
        for lab, col, mk in systems:
            ax.plot(snr, [nw[f"{key}_{s}"][lab] for s in snr], marker=mk, color=col, label=lab)
        ax.set_title(title, fontsize=10); ax.set_xlabel("SNR, dB"); ax.set_xticks(snr); ax.invert_xaxis()
        ax.axhline(1 / 7, color=INK2, lw=0.8, ls=(0, (4, 3)))
    axs[0].set_ylabel("macro-F1 (1,050-clip test subset)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle("Robustness to additive noise, all systems on identical audio (dashed: chance level)", x=0.5, ha="center", fontweight="bold", y=1.03)
    fig.subplots_adjust(bottom=0.25)
    save(fig, "06_noise_robustness.png")

# ================================================================== 07 calibration and abstention
if R:
    P = np.mean(R, 0); conf = P.max(1); ok = P.argmax(1) == y
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    bins = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.sum() > 5: xs.append(conf[m].mean()); ys.append(ok[m].mean()); ns.append(m.sum())
    axs[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3))); axs[0].plot(xs, ys, marker="o", color=BLUE)
    axs[0].set_xlabel("confidence"); axs[0].set_ylabel("accuracy"); ece = sum(((conf > a) & (conf <= b2)).mean() * abs(ok[(conf > a) & (conf <= b2)].mean() - conf[(conf > a) & (conf <= b2)].mean()) for a, b2 in zip(bins[:-1], bins[1:]) if ((conf > a) & (conf <= b2)).any())
    axs[0].set_title(f"Reliability (ECE = {ece:.3f})", fontsize=10)
    ths = np.linspace(0, 0.9, 40); cov = [(conf >= t).mean() for t in ths]; acc = [ok[conf >= t].mean() if (conf >= t).sum() else np.nan for t in ths]
    axs[1].plot(cov, acc, color=BLUE); axs[1].set_xlabel("share of clips answered (coverage)"); axs[1].set_ylabel("accuracy on answered clips")
    axs[1].set_title("Answering only when confident", fontsize=10); fig.suptitle("Deployed model: noise-augmented WavLM head ensemble", x=0.5, ha="center", fontweight="bold", fontsize=10, y=1.03); axs[1].invert_xaxis()
    for t in (0.4, 0.5, 0.6):
        m = conf >= t; axs[1].plot(m.mean(), ok[m].mean(), "o", color=ORANGE); axs[1].annotate(f"conf ≥ {t:g}", (m.mean(), ok[m].mean()), textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK2)
    save(fig, "07_calibration.png")

# ================================================================== 08 SSL CNN and semi-supervised
ss = load("semi_supervised_wavlm.json")
if len(cs) or ss:
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    if len(cs):
        for k, fr_ in enumerate((0.1, 1.0)):
            for j, (init, col, lab) in enumerate((("scratch", GRAY, "from scratch"), ("ssl", ORANGE, "contrastive pre-training"))):
                v = cs[(cs.init == init) & (cs.fraction == fr_)].test_macro_f1.to_numpy()
                x = k + (j - 0.5) * 0.36
                axs[0].bar(x, v.mean(), 0.32, color=col, label=lab if k == 0 else None); axs[0].scatter([x] * len(v), v, color=INK, s=14, zorder=3)
                axs[0].text(x, max(v.max(), v.mean()) + 0.014, f"{v.mean():.2f} (n={len(v)})", ha="center", fontsize=8, color=INK2)
        axs[0].set_xticks([0, 1], ["10% of the labels", "100% of the labels"]); axs[0].set_ylabel("test macro-F1"); axs[0].grid(axis="x", visible=False)
        axs[0].legend(loc="upper left"); axs[0].set_title("CNN: effect of self-supervised pre-training")
    if ss:
        s = pd.DataFrame(ss).groupby("labeled_fraction")[["supervised", "self_training"]].agg(["mean", "std"])
        for col, lab, k in ((BLUE, "supervised only", "supervised"), (ORANGE, "self-training (pseudo-labels)", "self_training")):
            axs[1].errorbar(s.index * 100, s[(k, "mean")], yerr=s[(k, "std")], color=col, marker="o", capsize=3, label=lab)
        axs[1].set_xlabel("share of labelled training clips, %"); axs[1].set_ylabel("test macro-F1 (5 subsets)"); axs[1].legend(loc="lower right")
        axs[1].set_title("Semi-supervised learning on WavLM embeddings")
    save(fig, "08_selfsup_and_semisup.png")

# ================================================================== 09 fine-tuning curves
runs = [(f.parent.name, json.loads(f.read_text())) for f in sorted((V2 / "finetune").glob("*/result.json")) if "hold-" not in f.parent.name]
if runs:
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    palette = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA, YELLOW, RED, GREEN]
    for k, (n, r) in enumerate(runs[:8]):
        ep = [h["epoch"] for h in r["history"]]
        axs[0].plot(ep, [h["val"]["macro_f1"] for h in r["history"]], marker="o", color=palette[k % 8], label=n.replace("wavlm_v2__", "").replace("__", " "))
        axs[1].plot(ep, [h["train_loss"] for h in r["history"]], marker="o", color=palette[k % 8])
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("validation macro-F1"); axs[0].set_title("Fine-tuning: validation F1 per epoch")
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("training loss"); axs[1].set_title("Training loss")
    axs[0].legend(fontsize=7, ncol=1, loc="lower right")
    save(fig, "09_finetuning_curves.png")

# ================================================================== 10 per-corpus F1
best = ens.get("WavLM fine-tuned + noise aug. (ensemble)") if "WavLM fine-tuned + noise aug. (ensemble)" in ens else None
sel = [(n, ens[n], c) for n, c in (("WavLM frozen (5-seed ensemble)", BLUE), ("WavLM frozen + noise aug. (ensemble)", AQUA), ("WavLM fine-tuned (ensemble)", VIOLET),
                                    ("WavLM fine-tuned + noise aug. (ensemble)", ORANGE), ("XLS-R fine-tuned + noise aug. (ensemble)", RED)) if n in ens]
corp = [c for c in sorted(set(ds)) if (ds == c).sum() >= 20]
if sel:
    fig, ax = plt.subplots(figsize=(10, 4)); w = 0.8 / len(sel)
    for k, (n, P, col) in enumerate(sel):
        pred = P.argmax(1)
        ax.bar(np.arange(len(corp)) + k * w - 0.4 + w / 2, [mf1(y[ds == c], pred[ds == c]) for c in corp], w * 0.92, color=col, label=n)
    ax.set_xticks(range(len(corp)), [f"{c}\n(n={int((ds == c).sum())})" for c in corp]); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    ax.set_ylabel("macro-F1"); ax.legend(ncol=2, loc="lower left", fontsize=8); ax.set_title("Test macro-F1 per corpus (seen corpora; TESS has no training speakers)")
    save(fig, "10_per_corpus.png")

# ================================================================== 11 dataset overview
fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
sp = data.pivot_table(index="dataset", columns="experiment_split", values="audio_path", aggfunc="count", fill_value=0)[["train", "validation", "test"]]
left = np.zeros(len(sp))
for col, c in zip(sp.columns, (BLUE, AQUA, ORANGE)):
    axs[0].barh(range(len(sp)), sp[col], left=left, color=c, label=col, height=0.62, edgecolor=SURFACE, linewidth=1.5); left += sp[col].to_numpy()
axs[0].set_yticks(range(len(sp)), [f"{d} ({data[data.dataset == d].language.iloc[0]})" for d in sp.index]); axs[0].invert_yaxis(); axs[0].grid(axis="y", visible=False)
axs[0].set_xlabel("clips"); axs[0].legend(loc="lower right"); axs[0].set_title("Corpora and speaker-disjoint splits")
em = data.pivot_table(index="dataset", columns="emotion", values="audio_path", aggfunc="count", fill_value=0)[LABELS]
axs[1].imshow(np.log1p(em.values), cmap=SEQ, aspect="auto"); axs[1].grid(False)
axs[1].set_xticks(range(7), LABELS, rotation=45, ha="right"); axs[1].set_yticks(range(len(em)), em.index)
for (i, j), v in np.ndenumerate(em.values):
    axs[1].text(j, i, int(v), ha="center", va="center", fontsize=8, color="white" if np.log1p(v) > 4.5 else INK)
axs[1].set_title("Clips per emotion and corpus (colour: log scale)")
axs[0].annotate("30 clips", (60, list(sp.index).index("SVK-DB")), va="center", fontsize=8, color=INK2)
save(fig, "11_dataset_overview.png")

# ================================================================== 12 layer analysis
lp = V2 / "layer_probe.csv"
if lp.exists():
    pr = pd.read_csv(lp)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for e, col, lab in (("wavlm_base_plus", BLUE, "WavLM-base-plus"), ("wav2vec2_base", ORANGE, "Wav2Vec2-base")):
        g = pr[pr.embedding == e]; axs[0].plot(g.layer, g.test_f1, marker="o", color=col, label=lab)
    axs[0].set_xlabel("transformer layer (0 = CNN output)"); axs[0].set_ylabel("macro-F1 of a linear probe (test, exploratory)"); axs[0].legend(); axs[0].set_title("Emotion information by layer (exploratory)", fontsize=10)
    lw = np.mean([json.loads(open(f).read())["layer_weights"] for f in glob.glob(str(V2 / "frozen/wavlm_base_plus__none_0__seed*.json"))], 0)
    axs[1].bar(range(len(lw)), lw, color=BLUE, width=0.65); axs[1].axhline(1 / len(lw), color=INK2, lw=0.8, ls=(0, (4, 3)))
    axs[1].set_xlabel("WavLM layer"); axs[1].set_ylabel("learned weight (mean of 5 seeds)"); axs[1].set_title("Weights the head puts on each layer"); axs[1].grid(axis="x", visible=False)
    save(fig, "12_layers.png")

# ================================================================== 13 speaker-grouped CV
g = load("grouped_cv.json")
if g:
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
    f1s = [f["macro_f1"] for f in g["folds"]]
    axs[0].bar(range(1, len(f1s) + 1), f1s, color=BLUE, width=0.6); axs[0].axhline(np.mean(f1s), color=INK2, lw=1, ls=(0, (4, 3)))
    for i, v in enumerate(f1s): axs[0].text(i + 1, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
    axs[0].set_ylim(0, 1); axs[0].set_xlabel("fold (whole speakers held out)"); axs[0].set_ylabel("macro-F1"); axs[0].set_title(f"5-fold speaker-grouped CV: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    pc = pd.Series(g["per_corpus_f1"]).drop("SVK-DB", errors="ignore").sort_values()
    axs[1].barh(range(len(pc)), pc.values, color=BLUE, height=0.6); axs[1].set_yticks(range(len(pc)), pc.index); axs[1].set_xlim(0, 1)
    for i, v in enumerate(pc.values): axs[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    axs[1].grid(axis="y", visible=False); axs[1].set_title("CV macro-F1 per corpus (SVK-DB omitted: 30 clips)"); axs[1].set_xlabel("macro-F1")
    save(fig, "13_grouped_cv.png")

# ================================================================== 14 t-SNE
try:
    from sklearn.manifold import TSNE
    z = np.load(V2 / "emb_wavlm_base_plus.npz")
    F = np.concatenate([z["mean"], z["std"]], 2).astype(np.float32)[:, 3:7].mean(1)
    idx = np.random.default_rng(0).choice(len(data), 3000, replace=False)
    E = TSNE(2, init="pca", perplexity=40, random_state=0).fit_transform((F[idx] - F[idx].mean(0)) / (F[idx].std(0) + 1e-6))
    corp_col = dict(zip(sorted(data.dataset.unique()), [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED]))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"wspace": 0.12})
    for ax, col, cmap_, title in ((axs[0], data.emotion.to_numpy()[idx], EMO_COLOR, "emotion"), (axs[1], data.dataset.to_numpy()[idx], corp_col, "corpus")):
        for v in sorted(set(col)):
            m = col == v; ax.scatter(E[m, 0], E[m, 1], s=5, color=cmap_[v], label=v, alpha=0.85, linewidths=0)
        ax.legend(markerscale=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, columnspacing=1.0, handletextpad=0.3); ax.set_title(f"WavLM embeddings (t-SNE), coloured by {title}"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, "14_tsne.png")
except Exception as e:
    print("t-SNE skipped:", e)

# ================================================================== 15 backbones (filled when the extra embeddings exist)
bb = load("backbones.json")
if bb:
    names = list(bb)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, key, title in ((axs[0], "default_f1", "Default split (3,511 test clips)"), (axs[1], "loco_f1", "Cross-corpus (mean of 6 held-out corpora)")):
        v = [bb[n][key] if bb[n][key] is not None else 0.0 for n in names]; e = [bb[n].get(key + "_std", 0) for n in names]
        ax.bar(range(len(names)), v, yerr=e, color=[BLUE, ORANGE, AQUA, VIOLET][: len(names)], width=0.6, capsize=3)
        for i, x in enumerate(v): ax.text(i, x + e[i] + 0.012, f"{x:.2f}" if x else "n/a", ha="center", fontsize=9, color=INK2)
        ax.set_xticks(range(len(names)), [n.replace(" (", chr(10) + "(") for n in names], fontsize=8); ax.set_ylim(0, 0.9); ax.set_title(title); ax.set_ylabel("macro-F1"); ax.grid(axis="x", visible=False)
    save(fig, "15_backbones.png")

# ================================================================== 16 corruptions not used in training
if nw:
    systems = [("frozen, clean training", GRAY), ("frozen, noise-augmented", BLUE), ("fine-tuned, clean training", ORANGE), ("fine-tuned, noise-augmented", VIOLET)]
    systems = [(n, c) for n, c in systems if any(n in v for v in nw.values())]
    groups = [("clean", "clean"), ("reverb_0.3", "reverb\nRT60 0.3 s"), ("reverb_0.8", "reverb\nRT60 0.8 s"), ("telephone", "telephone\n300-3400 Hz"),
              ("lowpass", "low-pass\n4 kHz"), ("brown_10", "brown noise\n10 dB*"), ("hum_10", "mains hum\n10 dB"), ("clip", "clipping")]
    fig, ax = plt.subplots(figsize=(11.5, 4.4)); w = 0.8 / len(systems)
    for k, (n, col) in enumerate(systems):
        ax.bar(np.arange(len(groups)) + k * w - 0.4 + w / 2, [nw[g].get(n, np.nan) for g, _ in groups], w * 0.92, color=col, label=n)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups], fontsize=8); ax.set_ylabel("macro-F1 (1,050-clip subset)"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", fontsize=8, columnspacing=1.2); ax.set_ylim(0, 1.0)
    ax.set_title("Corruptions never used in training")
    ax.text(0, -0.24, "* brown noise carries >99% of its energy below 100 Hz, outside the speech band, so it barely affects any system", transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, "16_unseen_corruptions.png")

    seen = [("white_10", "white 10 dB"), ("white_0", "white 0 dB"), ("pink_10", "pink 10 dB"), ("babble_10", "babble 10 dB"), ("babble_5", "babble 5 dB")]
    fig, ax = plt.subplots(figsize=(10, 3.9)); w = 0.8 / len(systems)
    for k, (n, col) in enumerate(systems):
        ax.bar(np.arange(len(seen)) + k * w - 0.4 + w / 2, [nw[g].get(n, np.nan) for g, _ in seen], w * 0.92, color=col, label=n)
    ax.set_xticks(range(len(seen)), [g[1] for g in seen]); ax.set_ylabel("macro-F1"); ax.grid(axis="x", visible=False); ax.legend(ncol=4, fontsize=8, loc="upper center", columnspacing=1.2); ax.set_ylim(0, 1.0)
    ax.set_title("Noise types seen during noise-augmented training, all four systems")
    save(fig, "17_seen_noise_all_systems.png")

print("done; figures in", FIG)
