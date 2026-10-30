"""Evaluate every deployed/candidate system on IDENTICAL corrupted audio (1,050-clip stratified test subset).

Systems: frozen WavLM + heads trained on clean audio, frozen WavLM + noise-augmented heads, fine-tuned (clean),
fine-tuned with noise augmentation (ensemble of the saved runs). Conditions: clean, white / pink / babble noise
(seen in training) and corruptions NOT used in training: reverberation, telephone band, brown noise, mains hum, clipping.
Run with the CUDA environment: .venv-gpu\\Scripts\\python.exe evaluate_noise_all.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModel

from common_splits import CACHE, LABELS, ROOT, load_splits
from noise_utils import babble, mix, pink, white
from predict_emotion import Head
from train_wavlm_v2 import SSLClassifier

V2 = CACHE.parent
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16_000
d = load_splits()
waves = np.load(V2 / "cnn_ssl/waves_int16.npy", mmap_mode="r")
lens = np.load(V2 / "cnn_ssl/lengths.npy")
te = d[d.experiment_split == "test"]
sub = np.concatenate([te[te.label_id == c].sample(min(int((te.label_id == c).sum()), 150), random_state=0).index.to_numpy() for c in range(len(LABELS))])
if os.environ.get("SMOKE"):
    sub = sub[:: 25]  # quick plumbing test
y = d.label_id.to_numpy()[sub]
pool_rows = np.where(d.experiment_split.isin(["train", "validation"]))[0]


def clip(i):
    return waves[i, : lens[i]].astype(np.float32) / 32767


def brown(n, rng):
    x = np.cumsum(rng.standard_normal(n)); x -= np.linspace(x[0], x[-1], n)
    return (x / (x.std() + 1e-9)).astype(np.float32)


def hum(n, rng):
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * 50 * k * t + rng.uniform(0, 6.28)) / k for k in (1, 2, 3, 4, 5))
    return (x / x.std()).astype(np.float32)


def reverb(w, rt60, rng):
    n = int(rt60 * SR); t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60); rir[0] = 1.0
    from scipy.signal import fftconvolve  # FFT convolution: np.convolve with a 0.8 s RIR took minutes per condition
    out = fftconvolve(w, rir / np.linalg.norm(rir), mode="full")[: len(w)]
    return (out * (np.sqrt(np.mean(w ** 2)) / (np.sqrt(np.mean(out ** 2)) + 1e-9))).astype(np.float32)


def band(w, lo, hi):
    spec = np.fft.rfft(w); f = np.fft.rfftfreq(len(w), 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(w)).astype(np.float32)


def corrupt(i, cond, rng):
    w = clip(i)
    if cond == "clean": return w
    kind, _, arg = cond.partition("_")
    if kind in ("white", "pink", "brown", "hum", "babble"):
        n = {"white": lambda: white(len(w), rng), "pink": lambda: pink(len(w), rng), "brown": lambda: brown(len(w), rng),
             "hum": lambda: hum(len(w), rng), "babble": lambda: babble(len(w), rng, lambda: clip(rng.choice(pool_rows)))}[kind]()
        return mix(w, n, float(arg))
    if kind == "reverb": return reverb(w, float(arg), rng)
    if kind == "telephone": return band(w, 300, 3400)
    if kind == "lowpass": return band(w, 0, 4000)
    if kind == "clip": return np.clip(w * 6, -0.5 * np.abs(w * 6).max(), 0.5 * np.abs(w * 6).max()).astype(np.float32)
    raise ValueError(cond)


CONDS = (["clean"] + [f"white_{s}" for s in (20, 10, 5, 0)] + ["pink_10"] + [f"babble_{s}" for s in (20, 10, 5)] +
         ["reverb_0.3", "reverb_0.8", "telephone", "lowpass", "brown_10", "hum_10", "clip"])
SEEN = {c for c in CONDS if c.split("_")[0] in ("clean", "white", "pink", "babble")}

# ---- frozen systems
enc = AutoModel.from_pretrained("microsoft/wavlm-base-plus").eval().to(dev)


def load_heads(folder):
    hs = []
    for f in sorted(Path(folder).glob("wavlm_head_seed*.pt")):
        b = torch.load(f, map_location="cpu"); h = Head(b["layers"], b["dim"], b["domains"], b["mode"]); h.load_state_dict(b["state"]); h.eval()
        hs.append((h, b["mu"], b["sd"]))
    return hs


heads = {"frozen, clean training": load_heads(ROOT / "models"), "frozen, noise-augmented": load_heads(ROOT / "models/noise_robust")}


@torch.inference_mode()
def frozen_feats(w):
    w = (w - w.mean()) / (w.std() + 1e-7)
    hs = torch.stack(enc(torch.from_numpy(w)[None].to(dev), output_hidden_states=True).hidden_states, 0)[:, 0]
    return torch.cat([hs.mean(1), hs.std(1)], 1).float().cpu()[None]


# ---- fine-tuned systems
def load_ft(path):
    ck = torch.load(path, map_location="cpu"); a = ck["args"]
    m = SSLClassifier(a["model"], int(d["dataset_id"].max()) + 1, a["mode"], a["mask_prob"]); m.load_state_dict(ck["state"]); return m.eval().to(dev)


ft_sets = {"fine-tuned, clean training": sorted((V2 / "finetune").glob("ftclean__*/best_model.pt")),
           "fine-tuned, noise-augmented": sorted((V2 / "finetune").glob("ftnoise__*/best_model.pt"))}
ft_models = {k: [load_ft(p) for p in v] for k, v in ft_sets.items() if v}


@torch.inference_mode()
def ft_probs(models, w):
    w = (w - w.mean()) / (w.std() + 1e-7); x = torch.from_numpy(w)[None].to(dev); ln = torch.tensor([len(w)], device=dev)
    out = []
    for m in models:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(torch.softmax(m(x, ln)[0].float(), 1).cpu().numpy()[0])
    return np.mean(out, 0)


PARTIAL = V2 / "noise_all_partial.json"  # resume: conditions finished in an earlier (interrupted) run
results = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
for ci, cond in enumerate(CONDS):
    if cond in results:
        continue
    rng = np.random.default_rng(1000 + ci)
    preds = {k: [] for k in list(heads) + list(ft_models)}
    for i in sub:
        w = corrupt(i, cond, rng)
        x = frozen_feats(w)
        for k, hs in heads.items():
            with torch.inference_mode():
                p = torch.stack([torch.softmax(h((x - mu) / sd), 1)[0] for h, mu, sd in hs]).mean(0).numpy()
            preds[k].append(int(p.argmax()))
        for k, ms in ft_models.items():
            preds[k].append(int(ft_probs(ms, w).argmax()))
    results[cond] = {k: float(f1_score(y, p, average="macro")) for k, p in preds.items()}
    results[cond]["_seen_in_training"] = cond in SEEN
    print(cond, {k: round(v, 3) for k, v in results[cond].items() if k[0] != "_"}, flush=True)
    PARTIAL.write_text(json.dumps(results, indent=1))
(V2 / ("noise_all_systems_smoke.json" if os.environ.get("SMOKE") else "noise_all_systems.json")).write_text(json.dumps(results, indent=1))
print("saved results for", len(results), "conditions")
