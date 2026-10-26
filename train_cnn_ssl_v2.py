"""Controlled self-supervised study with one CNN encoder.

Conditions (same architecture, same fine-tuning recipe, only the encoder init differs):
  scratch : random init
  ssl     : contrastive (NT-Xent) pretraining on TRAIN audio only (labels never used)
each with 100% and 10% of the train labels, several seeds. Also reports a linear probe on the
frozen pretrained encoder. Model selection on validation macro-F1, test scored once.

Usage:
  python train_cnn_ssl_v2.py cache
  python train_cnn_ssl_v2.py pretrain --epochs 40
  python train_cnn_ssl_v2.py finetune --seeds 3
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from common_splits import CACHE, LABELS, ROOT, load_splits
from train_deep_models import MelCNN

SR = 16_000
LEN = SR * 4
OUT = CACHE.parent / "cnn_ssl"
WAVES = OUT / "waves_int16.npy"
LENS = OUT / "lengths.npy"
torch.set_num_threads(16)


def build_cache() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_splits()
    waves = np.zeros((len(data), LEN), dtype=np.int16)
    lens = np.zeros(len(data), dtype=np.int64)
    for i, path in enumerate(data["audio_path"]):
        w, _ = sf.read(ROOT / path, dtype="float32")
        if w.ndim == 2:
            w = w.mean(1)
        w = w[:LEN]
        waves[i, : len(w)] = (np.clip(w, -1, 1) * 32767).astype(np.int16)
        lens[i] = len(w)
    np.save(WAVES, waves); np.save(LENS, lens)
    print("cached", waves.shape, "mean length s:", lens.mean() / SR)


class Backbone(nn.Module):
    """log-mel + deltas -> CNN encoder (identical for scratch and pretrained)."""

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, win_length=400, hop_length=160, n_mels=64)
        self.time_mask = torchaudio.transforms.TimeMasking(18)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(8)
        self.encoder = MelCNN(len(LABELS)).encoder

    def forward(self, wave: torch.Tensor, augment: bool = False) -> torch.Tensor:
        s = torch.log(self.mel(wave).clamp_min(1e-5))
        s = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-5)
        if augment:
            s = self.freq_mask(self.time_mask(s))
        d = torchaudio.functional.compute_deltas(s)
        dd = torchaudio.functional.compute_deltas(d)
        return self.encoder(torch.stack((s, d, dd), 1)).flatten(1)


def batch_waves(waves, lens, idx, crop=None, augment=False):
    """Return float batch; crop = random window (samples) inside the valid part of each clip."""
    out = []
    for i in idx:
        n = int(lens[i]); w = torch.from_numpy(waves[i, :n].astype(np.float32) / 32767)
        if crop:
            if n >= crop:
                s = np.random.randint(0, n - crop + 1); w = w[s : s + crop]
            else:
                w = nn.functional.pad(w, (0, crop - n))
        else:
            w = nn.functional.pad(w, (0, LEN - n))
        if augment:
            w = w * 10 ** (np.random.uniform(-6, 6) / 20)
            if np.random.rand() < 0.6:
                snr = np.random.uniform(10, 35)
                w = w + torch.randn_like(w) * (w.pow(2).mean().clamp_min(1e-8) / 10 ** (snr / 10)).sqrt()
            if np.random.rand() < 0.3:
                f = np.random.uniform(0.9, 1.1)
                w = nn.functional.interpolate(w[None, None], scale_factor=1 / f, mode="linear")[0, 0]
                w = w[: (crop or LEN)]
                w = nn.functional.pad(w, (0, (crop or LEN) - len(w)))
        out.append(w)
    return torch.stack(out)


def nt_xent(a, b, temperature=0.2):
    z = nn.functional.normalize(torch.cat([a, b]), dim=1)
    sim = z @ z.T / temperature
    n = len(a)
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return nn.functional.cross_entropy(sim, targets)


def pretrain(epochs: int, batch_size: int) -> None:
    data = load_splits()
    waves, lens = np.load(WAVES, mmap_mode="r"), np.load(LENS)
    train_idx = np.where(data.experiment_split == "train")[0]
    torch.manual_seed(0); np.random.seed(0)
    backbone = Backbone()
    proj = nn.Sequential(nn.Linear(192, 256), nn.ReLU(), nn.Linear(256, 128))
    params = list(backbone.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    steps = epochs * (len(train_idx) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    history, started = [], time.perf_counter()
    crop = SR * 2
    for epoch in range(1, epochs + 1):
        backbone.train()
        perm = np.random.permutation(train_idx)
        total, nb = 0.0, 0
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            v1 = batch_waves(waves, lens, idx, crop, True)
            v2 = batch_waves(waves, lens, idx, crop, True)
            loss = nt_xent(proj(backbone(v1, True)), proj(backbone(v2, True)))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            total += loss.item(); nb += 1
        history.append({"epoch": epoch, "loss": total / nb})
        print(f"pretrain epoch {epoch}/{epochs} loss={total / nb:.4f} elapsed={(time.perf_counter() - started) / 60:.1f}m", flush=True)
        torch.save(backbone.state_dict(), OUT / "backbone_ssl.pt")
        (OUT / "pretrain_history.json").write_text(json.dumps(history))


def evaluate(model, waves, lens, idx, bs=64):
    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            probs.append(torch.softmax(model(batch_waves(waves, lens, idx[i : i + bs])), 1))
    return torch.cat(probs).numpy()


class Classifier(nn.Module):
    def __init__(self, backbone: Backbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(192, len(LABELS)))

    def forward(self, x, augment=False):
        return self.head(self.backbone(x, augment))


def finetune(init: str, fraction: float, seed: int, epochs=20, patience=5, bs=32) -> dict:
    data = load_splits()
    waves, lens = np.load(WAVES, mmap_mode="r"), np.load(LENS)
    y = data["label_id"].to_numpy()
    split = data.experiment_split.to_numpy()
    rng = np.random.default_rng(seed)
    train_idx = np.where(split == "train")[0]
    if fraction < 1:
        keep = []
        for c in range(len(LABELS)):
            ids = train_idx[y[train_idx] == c]
            keep += rng.choice(ids, max(1, int(round(len(ids) * fraction))), replace=False).tolist()
        train_idx = np.array(keep)
    val_idx, test_idx = np.where(split == "validation")[0], np.where(split == "test")[0]
    torch.manual_seed(seed); np.random.seed(seed)
    backbone = Backbone()
    if init == "ssl":
        backbone.load_state_dict(torch.load(OUT / "backbone_ssl.pt"))
    model = Classifier(backbone)
    counts = np.bincount(y[train_idx], minlength=len(LABELS)).astype(np.float32)
    w = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = torch.optim.AdamW([{"params": backbone.parameters(), "lr": 5e-4},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=1e-2)
    steps = epochs * math.ceil(len(train_idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-3, 2e-3], total_steps=steps, pct_start=0.1)
    best, best_state, best_epoch, stale, hist = -1.0, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            loss = crit(model(batch_waves(waves, lens, idx, augment=True), True), torch.from_numpy(y[idx]))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        vp = evaluate(model, waves, lens, val_idx)
        f1 = f1_score(y[val_idx], vp.argmax(1), average="macro")
        hist.append(f1)
        if f1 > best:
            best, best_epoch, stale = f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    tp = evaluate(model, waves, lens, test_idx)
    res = {"init": init, "fraction": fraction, "seed": seed, "n_labeled": int(len(train_idx)), "best_epoch": best_epoch,
           "val_macro_f1": best, "test_macro_f1": float(f1_score(y[test_idx], tp.argmax(1), average="macro")),
           "test_acc": float(accuracy_score(y[test_idx], tp.argmax(1)))}
    name = f"{init}_f{fraction:g}_seed{seed}"
    (OUT / f"{name}.json").write_text(json.dumps(res)); np.save(OUT / f"{name}_test.npy", tp)
    print(name, json.dumps(res), flush=True)
    return res


def linear_probe() -> dict:
    data = load_splits()
    waves, lens = np.load(WAVES, mmap_mode="r"), np.load(LENS)
    y, split = data["label_id"].to_numpy(), data.experiment_split.to_numpy()
    res = {}
    for init in ("scratch", "ssl"):
        torch.manual_seed(0)
        bb = Backbone()
        if init == "ssl":
            bb.load_state_dict(torch.load(OUT / "backbone_ssl.pt"))
        bb.eval(); feats = []
        with torch.no_grad():
            for i in range(0, len(data), 64):
                feats.append(bb(batch_waves(waves, lens, np.arange(i, min(i + 64, len(data))))).numpy())
        F = np.concatenate(feats); mu, sd = F[split == "train"].mean(0), F[split == "train"].std(0) + 1e-6
        F = (F - mu) / sd
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(F[split == "train"], y[split == "train"])
        res[init] = float(f1_score(y[split == "test"], clf.predict(F[split == "test"]), average="macro"))
        print("linear probe", init, res[init], flush=True)
    (OUT / "linear_probe.json").write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("cache", "pretrain", "finetune", "probe"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "cache":
        build_cache()
    elif a.stage == "pretrain":
        pretrain(a.epochs, a.batch_size)
    elif a.stage == "probe":
        linear_probe()
    else:
        for frac in (1.0, 0.1):
            for init in ("scratch", "ssl"):
                for seed in range(a.seeds):
                    if not (OUT / f"{init}_f{frac:g}_seed{seed}.json").exists():
                        finetune(init, frac, seed)
