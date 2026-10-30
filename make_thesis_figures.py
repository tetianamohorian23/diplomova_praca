"""Four data-preparation figures for the thesis (English labels) -> diplomka/obrazky/*.png.
Counts come from experiments/v2/splits.csv = the 13,406 clips actually used (combined_dataset_normalized.csv has 18,399 rows with
duplicated directories and the excluded classes calm / boredom, so it must not be used for counts)."""
from __future__ import annotations

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common_splits import LABELS, ROOT, load_splits


class mf:  # palette and style identical to make_figures.py (not imported: that module regenerates every figure on import)
    BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
    GRAY, SURFACE, INK, INK2, GRID = "#8d8c86", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    EMO_COLOR = dict(zip(LABELS, [RED, GREEN, VIOLET, YELLOW, GRAY, BLUE, MAGENTA]))


plt.rcParams.update({
    "figure.facecolor": mf.SURFACE, "axes.facecolor": mf.SURFACE, "savefig.facecolor": mf.SURFACE, "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": mf.INK, "axes.labelcolor": mf.INK2, "xtick.color": mf.INK2, "ytick.color": mf.INK2, "axes.edgecolor": mf.GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": mf.GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "center", "legend.frameon": False,
})

OUT = ROOT / "diplomka" / "obrazky"
OUT.mkdir(parents=True, exist_ok=True)
data = load_splits()
SK = {l: l for l in LABELS}  # English emotion names


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


# ---------------------------------------------------------------- Obr. 1: signal + spectrogram (CREMA-D, angry, test split)
row = data[(data.dataset == "CREMA-D") & (data.emotion == "angry") & (data.experiment_split == "test")].iloc[0]
w, sr = sf.read(ROOT / row.audio_path)
assert sr == 16000 and w.ndim == 1
t = np.arange(len(w)) / sr
M = librosa.power_to_db(librosa.feature.melspectrogram(y=w.astype(np.float32), sr=sr, n_fft=400, hop_length=160, n_mels=64), ref=np.max)
fig, axs = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
axs[0].plot(t, w, color=mf.BLUE, lw=0.6); axs[0].set_ylabel("amplitude"); axs[0].set_title("Audio signal (16 kHz, mono)")
im = axs[1].imshow(M, origin="lower", aspect="auto", cmap="magma", extent=[0, len(w) / sr, 0, 8000], vmin=-80, vmax=0)
axs[1].grid(False); axs[1].set_ylabel("frequency [Hz]"); axs[1].set_xlabel("time [s]"); axs[1].set_title("Log-mel spectrogram (64 bands)")
fig.colorbar(im, ax=axs[1], label="level [dB]", pad=0.02)
fig.tight_layout()
save(fig, "obr_signal_spektrogram.png")
print("example clip:", row.audio_path, f"{len(w) / sr:.2f} s")

# ---------------------------------------------------------------- Obr. 2: clips per emotion
cnt = data.emotion.value_counts().reindex(LABELS)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar([SK[l] for l in LABELS], cnt.values, color=[mf.EMO_COLOR[l] for l in LABELS], width=0.62, label=None)
for b, v in zip(bars, cnt.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 25, f"{v:,}".replace(",", " "), ha="center", fontsize=9, color=mf.INK2)
ax.set_ylabel("number of recordings"); ax.set_ylim(0, cnt.max() * 1.12); ax.grid(axis="x", visible=False)
ax.set_title("Distribution of recordings by emotion class (13 406 recordings)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=mf.EMO_COLOR[l], label=l) for l in LABELS], title="Emotion class", ncol=4, loc="upper right", fontsize=9, title_fontsize=9)
ax.set_ylim(0, cnt.max() * 1.45)
fig.tight_layout()
save(fig, "obr_emocie.png")

# ---------------------------------------------------------------- Obr. 3: clips per corpus
order = data.dataset.value_counts()
names = {"EMODB": "EMO-DB"}
lang = {l: g.language.iloc[0] for l, g in data.groupby("dataset")}
fig, ax = plt.subplots(figsize=(8, 3.8))
y = np.arange(len(order))[::-1]
ax.barh(y, order.values, color=[mf.ORANGE if d == "SVK-DB" else mf.BLUE for d in order.index], height=0.62)
for yy, (d, v) in zip(y, order.items()):
    ax.text(v + 60, yy, f"{v:,}".replace(",", " "), va="center", fontsize=9, color=mf.INK2)
ax.set_yticks(y, [f"{names.get(d, d)} ({lang[d]})" for d in order.index]); ax.set_xlabel("number of recordings")
ax.set_xlim(0, order.max() * 1.12); ax.grid(axis="y", visible=False)
ax.set_title("Number of recordings per corpus (13 406 recordings)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=mf.BLUE, label="used corpora"), plt.Rectangle((0, 0), 1, 1, color=mf.ORANGE, label="Slovak sample (30 clips)")], loc="center right", fontsize=9)
fig.tight_layout()
save(fig, "obr_korpusy.png")

# ---------------------------------------------------------------- helpers for diagrams
def canvas(w_, h_, xmax=10, ymax=10):
    fig, ax = plt.subplots(figsize=(w_, h_)); ax.axis("off"); ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    return fig, ax


def box(ax, x, y, w_, h, text, fc="#eaf2fd", ec=mf.BLUE, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w_, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w_ / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=mf.INK, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=mf.INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.3, color=color))


# ---------------------------------------------------------------- Obr. 4: dataset preparation pipeline (section 3.5)
sp = data.experiment_split.value_counts()
n = f"{len(data):,}".replace(",", " ")
fig, ax = canvas(8.6, 7.2)
box(ax, 0.6, 8.9, 8.8, 0.95, "7 original corpora\nCREMA-D · RAVDESS · TESS · SAVEE · EMO-DB · UA-SER · SVK-DB", bold=False)
box(ax, 0.6, 7.2, 8.8, 1.2, "Selection and unification of data\nselection of recordings (duplicate copies removed)\nmapping of emotion categories (calm and boredom excluded)")
box(ax, 0.6, 5.7, 8.8, 0.95, f"Uniform audio format\nmono · 16 kHz · {n} recordings")
box(ax, 0.6, 4.2, 8.8, 0.95, "Common experimental set\n7 target emotion classes")
box(ax, 0.6, 2.7, 8.8, 0.95, "Speaker-level split\nno speaker appears in more than one set")
for y1, y2 in ((8.9, 8.4), (7.2, 6.65), (5.7, 5.15), (4.2, 3.65)):
    arrow(ax, 5, y1, 5, y2)
for x, lab, key, col in ((0.2, "Training set", "train", "#eaf2fd"), (3.6, "Validation set", "validation", "#e6f7ef"), (7.0, "Test set", "test", "#fdeee8")):
    box(ax, x, 0.5, 2.8, 1.4, f"{lab}\n{sp[key]:,}".replace(",", " "), fc=col, ec={"train": mf.BLUE, "validation": mf.AQUA, "test": mf.ORANGE}[key], bold=False)
    arrow(ax, 5, 2.7, x + 1.4, 1.9)
save(fig, "obr_priprava_dat.png")

# ---------------------------------------------------------------- Obr. 5: three approaches to SER (section 3.2)
fig, ax = canvas(10.8, 5.6, 11, 8)
rows = [
    (6.2, "#eaf2fd", mf.BLUE, ["Audio", "MFCC / eGeMAPS\nhandcrafted features", "SVM · Logistic Regression\nRandom Forest"], "Classical methods"),
    (3.5, "#fdeee8", mf.ORANGE, ["Audio", "Mel spectrogram", "CNN or CNN-BiLSTM\n(trained from scratch)"], "Networks from scratch"),
    (0.8, "#efecfb", mf.VIOLET, ["Audio", "Pre-trained model\nWavLM · Wav2Vec 2.0 · XLS-R", "Classifier head\nor fine-tuning"], "Pre-trained speech models"),
]
for y0, fc, ec, steps, title in rows:
    ax.text(0.1, y0 + 1.75, title, fontsize=10.5, fontweight="bold", color=mf.INK)
    xs = (0.1, 2.5, 6.7)
    ws = (1.6, 3.4, 2.8)
    for x0, w0, s_ in zip(xs, ws, steps):
        box(ax, x0, y0, w0, 1.5, s_, fc=fc if x0 > 0.5 else "#f4f4f1", ec=ec, fs=9.5)
    arrow(ax, 1.7, y0 + 0.75, 2.5, y0 + 0.75); arrow(ax, 5.9, y0 + 0.75, 6.7, y0 + 0.75); arrow(ax, 9.5, y0 + 0.75, 10.1, y0 + 0.75)
    ax.text(10.15, y0 + 0.75, "emotion", ha="left", va="center", fontsize=9.5, fontweight="bold", color=mf.INK2)
save(fig, "obr_pristupy.png")

# ---------------------------------------------------------------- Obr. 6: domain shift (section 3.4)
fig, ax = canvas(10, 4.8, 10, 6)
box(ax, 0.2, 3.7, 3.4, 1.7, "Training data\nEnglish · several corpora\nknown speakers", fc="#eaf2fd", ec=mf.BLUE)
box(ax, 3.3, 0.3, 3.4, 1.5, "Trained SER model\nlearned acoustic\nrepresentations", fc="#f4f4f1", ec=mf.GRAY)
box(ax, 6.4, 3.7, 3.4, 1.7, "New data\nother speaker · other language\nother recording conditions", fc="#fdeee8", ec=mf.ORANGE)
arrow(ax, 1.9, 3.7, 3.9, 1.8); arrow(ax, 8.1, 3.7, 6.1, 1.8, color=mf.ORANGE)
ax.text(5, 5.0, "domain shift", ha="center", va="center", fontsize=10.5, fontweight="bold", color=mf.ORANGE)
ax.annotate("", xy=(6.3, 4.55), xytext=(3.7, 4.55), arrowprops=dict(arrowstyle="<|-|>", color=mf.ORANGE, lw=1.5))
for x0, lab in ((4.1, "speaker"), (5.0, "language"), (5.9, "corpus")):
    ax.text(x0, 4.1, lab, ha="center", fontsize=8, color=mf.INK2)
ax.text(5, 0.0, "A change in the data distribution can reduce classification accuracy.", ha="center", va="top", fontsize=9, color=mf.INK2)
save(fig, "obr_domain_shift.png")

# ---------------------------------------------------------------- Obr. 7: ADDoG vs CNN, UAR (values from Gideon et al., IEEE TAC 2019, Table 2)
addog = {"MSP-Improv → IEMOCAP": ((0.439, 0.022), (0.474, 0.009)), "IEMOCAP → MSP-Improv": ((0.432, 0.012), (0.444, 0.007))}
fig, ax = plt.subplots(figsize=(7, 4)); xs = np.arange(2); w2 = 0.34
for k, (lab, col) in enumerate((("CNN", mf.GRAY), ("ADDoG", mf.BLUE))):
    m = [v[k][0] for v in addog.values()]; e = [v[k][1] for v in addog.values()]
    b = ax.bar(xs + (k - 0.5) * w2, m, w2 * 0.92, yerr=e, capsize=3, color=col, label=lab, error_kw={"lw": 1, "ecolor": mf.INK2})
    for xx, mm in zip(xs + (k - 0.5) * w2, m):
        ax.text(xx, mm + 0.035, f"{mm:.3f}", ha="center", fontsize=9, color=mf.INK2)
ax.axhline(1 / 3, color=mf.INK2, lw=1, ls=(0, (4, 3)), label="chance level (0.33)")
ax.set_xticks(xs, list(addog)); ax.set_ylabel("UAR (3-class valence)"); ax.set_ylim(0, 0.68); ax.grid(axis="x", visible=False)
ax.set_title("Cross-corpus transfer without target labels: CNN vs. ADDoG"); ax.legend(loc="upper center", ncol=3)
fig.tight_layout(); save(fig, "obr_addog_uar.png")

# ---------------------------------------------------------------- Obr. 8: class presence matrix (real labels from splits.csv)
em = data.pivot_table(index="dataset", columns="emotion", values="audio_path", aggfunc="count", fill_value=0)[LABELS]
order_c = ["CREMA-D", "RAVDESS", "TESS", "SAVEE", "EMODB", "UA-SER", "SVK-DB"]
em = em.loc[order_c]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
pres = (em.values > 0).astype(float)
ax.imshow(pres, cmap=matplotlib.colors.ListedColormap(["#efefec", "#9cc3f2"]), aspect="auto", vmin=0, vmax=1); ax.grid(False)
for (i, j), v in np.ndenumerate(em.values):
    ax.text(j, i, "✓" if v else "–", ha="center", va="center", fontsize=13 if v else 11, color=mf.INK if v else mf.GRAY)
ax.set_xticks(range(7), LABELS); ax.set_yticks(range(7), [("EMO-DB" if d == "EMODB" else d) + f" ({int((em.loc[d] > 0).sum())})" for d in em.index])
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_xticks(np.arange(-.5, 7, 1), minor=True); ax.set_yticks(np.arange(-.5, 7, 1), minor=True); ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
ax.set_title("Presence of the 7 target emotion classes in each corpus (number of classes in brackets)", pad=28)
fig.tight_layout(); save(fig, "obr_triedy_korpusy.png")

# ---------------------------------------------------------------- Obr. 4.1: architecture of the solution (chapter 4.1)
fig, ax = canvas(10.4, 7.6, 10, 10)
box(ax, 0.9, 8.8, 8.2, 1.0, "Data: 7 emotional speech corpora, 13 406 recordings, 7 emotion classes\nSpeaker-disjoint split: training 6 705 / validation 3 190 / test 3 511", fs=9.2)
cols = [
    (0.1, "#eaf2fd", mf.BLUE, "Classical methods", "Handcrafted features (289)\nand eGeMAPS (88)", "Logistic regression\nSVM · Random Forest"),
    (3.45, "#fdeee8", mf.ORANGE, "Networks from scratch", "Log-mel spectrogram\n(64 bands, 4 s)", "CNN · CNN-BiLSTM\n(+ contrastive pre-training)"),
    (6.8, "#efecfb", mf.VIOLET, "Pre-trained speech models", "wav2vec 2.0 · WavLM\nXLS-R (waveform)", "Frozen + light head\nor fine-tuning"),
]
for x0, fc, ec, title, in_, mod in cols:
    ax.text(x0 + 1.55, 8.15, title, ha="center", fontsize=10, fontweight="bold", color=mf.INK)
    box(ax, x0, 5.9, 3.1, 1.6, in_, fc=fc, ec=ec, fs=9.2)
    box(ax, x0, 3.7, 3.1, 1.6, mod, fc=fc, ec=ec, fs=9.2)
    arrow(ax, 5, 8.8, x0 + 1.55, 7.5)
    arrow(ax, x0 + 1.55, 5.9, x0 + 1.55, 5.3)
    arrow(ax, x0 + 1.55, 3.7, 5, 2.95)
box(ax, 0.8, 1.7, 8.4, 1.25, "Common evaluation on identical data\nmacro-F1 · bootstrap CI · without TESS · cross-corpus (LOCO)\nnoise and distortions · calibration", fs=9.2, fc="#e6f7ef", ec=mf.AQUA)
arrow(ax, 5, 1.7, 5, 1.1)
box(ax, 1.4, 0.15, 7.2, 0.95, "Application: web app EmoVoice · Telegram bot\n(command-line tool for checks)", fs=9.2, fc="#f4f4f1", ec=mf.GRAY)
save(fig, "obr_architektura.png")

# ---------------------------------------------------------------- application architecture (chapter 4.9.1)
fig, ax = canvas(10.4, 6.6, 10, 8)
for x0, text, fc, ec in [
    (0.2, "Command-line tool\npredict_emotion.py", "#e6f7ef", mf.AQUA),
    (3.6, "Web app EmoVoice\nFastAPI + HTML/JS\nmood diary", "#eaf2fd", mf.BLUE),
    (7.0, "Telegram bot\naiogram (bot.py)\nprivate and group chats", "#eaf2fd", mf.BLUE),
]:
    box(ax, x0, 6.2, 2.8, 1.5, text, fs=9.0, fc=fc, ec=ec)
    arrow(ax, x0 + 1.4, 6.2, x0 + 1.4, 5.5)
box(ax, 0.2, 3.5, 9.6, 2.0, "Shared recognition core (Predictor, webapp/service.py)\nconversion to 16 kHz mono · refusal of silence and of clips shorter than 0.25 s · speech-share estimate\n"
    "windows (3 s in the web app and the bot, 4 s in the command-line tool) · frozen WavLM-base-plus\nfive noise-augmented heads · average of probabilities · confidence threshold 0.5 (\"uncertain\")",
    fs=9.0, fc="#efecfb", ec=mf.VIOLET)
box(ax, 0.2, 0.3, 4.6, 1.6, "Result\nemotion · probabilities of 7 classes\nconfidence · \"uncertain\" flag · warning", fs=9.0, fc="#e6f7ef", ec=mf.AQUA)
box(ax, 5.2, 0.3, 4.6, 1.6, "SQLite database (data/emotions.db)\nentries · links · google_links · tg_codes\nmood diary shared by the web app and the bot", fs=9.0, fc="#fdeee8", ec=mf.ORANGE)
arrow(ax, 2.5, 3.5, 2.5, 1.9)
arrow(ax, 7.5, 3.5, 7.5, 1.9)
ax.text(7.6, 2.7, "saving (web app: on request,\nprivate Telegram chat: always)", fontsize=8.2, color=mf.INK2, va="center")
save(fig, "obr_aplikacia.png")
