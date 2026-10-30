"""Improved WavLM / wav2vec2 fine-tuning for speech emotion recognition.

Changes over train_deep_models.train_wavlm_multitask:
  * waveform augmentation (random crop/shift, gain, noise, speed) on the train split only;
  * learnable weighted sum over all hidden layers + masked mean pooling;
  * in-model SpecAugment (mask_time_prob), frozen CNN feature encoder;
  * layer-wise lr decay, linear warmup + cosine schedule, label smoothing, class weights;
  * corpus head as "aux" (original idea) or "grl" (gradient reversal -> corpus-invariant features);
  * every run writes to its own directory (the old ablation overwrote wavlm_multitask.json);
  * saves validation/test probabilities so runs can be ensembled in final_evaluation.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel

from common_splits import CACHE, LABELS, ROOT, load_splits
from train_frozen_heads import GradReverse

SR = 16_000
MAX_SAMPLES = SR * 4
OUT = CACHE.parent / "finetune"
AMP = True  # bfloat16 autocast on CUDA (set from --no-amp)


class WaveDataset(Dataset):
    def __init__(self, data, train: bool, noise_aug: bool = False) -> None:
        self.data = data.reset_index(drop=True)
        self.train = train
        self.noise_aug = noise_aug  # white / pink / babble noise at 0-30 dB SNR (as in train_noise_robust.py)

    def __len__(self) -> int:
        return len(self.data)

    def _pool(self) -> np.ndarray:
        w, _ = sf.read(ROOT / self.data.iloc[random.randrange(len(self.data))]["audio_path"], dtype="float32")
        return w.mean(1) if w.ndim == 2 else w

    def __getitem__(self, i):
        row = self.data.iloc[i]
        wave, _ = sf.read(ROOT / row["audio_path"], dtype="float32")
        wave = torch.from_numpy(np.ascontiguousarray(wave))
        if wave.ndim == 2:
            wave = wave.mean(1)
        if self.train:
            if random.random() < 0.5:  # speed perturbation
                f = random.uniform(0.9, 1.1)
                wave = torch.nn.functional.interpolate(wave[None, None], scale_factor=1 / f, mode="linear")[0, 0]  # fast speed change
            wave = wave * 10 ** (random.uniform(-6, 6) / 20)
            if self.noise_aug and random.random() < 0.7:
                from noise_utils import babble, mix, pink, white

                kind, snr = random.choice(("white", "pink", "babble")), random.uniform(0, 30)
                rng, w_np = np.random.default_rng(random.getrandbits(32)), wave.numpy()
                noise = {"white": lambda: white(len(w_np), rng), "pink": lambda: pink(len(w_np), rng),
                         "babble": lambda: babble(len(w_np), rng, self._pool)}[kind]()
                wave = torch.from_numpy(mix(w_np, noise, snr).astype(np.float32))
            elif random.random() < 0.5:  # mild additive noise at 15-35 dB SNR
                snr = random.uniform(15, 35)
                power = wave.pow(2).mean().clamp_min(1e-8)
                wave = wave + torch.randn_like(wave) * (power / 10 ** (snr / 10)).sqrt()
            if wave.numel() > MAX_SAMPLES:
                s = random.randint(0, wave.numel() - MAX_SAMPLES)
                wave = wave[s : s + MAX_SAMPLES]
        wave = wave[:MAX_SAMPLES]
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        return wave, int(row["label_id"]), int(row["dataset_id"])


def collate(batch):
    lengths = torch.tensor([len(b[0]) for b in batch])
    waves = torch.zeros(len(batch), int(lengths.max()))
    for i, (w, _, _) in enumerate(batch):
        waves[i, : len(w)] = w
    return waves, lengths, torch.tensor([b[1] for b in batch]), torch.tensor([b[2] for b in batch])


class SSLClassifier(nn.Module):
    def __init__(self, name: str, domains: int, mode: str, mask_prob: float, freeze_layers: int = 0, grad_ckpt: bool = False) -> None:
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        cfg.mask_time_prob = mask_prob
        cfg.mask_time_length = 10
        cfg.layerdrop = 0.0
        cfg.apply_spec_augment = mask_prob > 0
        self.encoder = AutoModel.from_pretrained(name, config=cfg)
        self.encoder.freeze_feature_encoder() if hasattr(self.encoder, "freeze_feature_encoder") else None
        if freeze_layers:  # keep the lowest transformer layers fixed (saves VRAM for the 300M-parameter XLS-R)
            for n_, p_ in self.encoder.named_parameters():
                low = n_.startswith("encoder.layers.") and int(n_.split(".")[2]) < freeze_layers
                if low or n_.startswith(("feature_projection", "encoder.pos_conv_embed", "masked_spec_embed")):
                    p_.requires_grad = False
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
        layers, hidden = cfg.num_hidden_layers + 1, cfg.hidden_size
        self.mode = mode
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.drop = nn.Dropout(0.2)
        self.emotion = nn.Linear(hidden, len(LABELS))
        self.domain = nn.Linear(hidden, domains) if mode != "none" else None

    def forward(self, waves, lengths, grl_scale: float = 1.0):
        frame_lengths = self.encoder._get_feat_extract_output_lengths(lengths)
        out = self.encoder(waves, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)  # (L, B, T, H)
        mix = (hs * torch.softmax(self.layer_logits, 0)[:, None, None, None]).sum(0)
        mask = (torch.arange(mix.shape[1], device=mix.device)[None] < frame_lengths[:, None]).unsqueeze(-1).to(mix.dtype)
        pooled = self.drop((mix * mask).sum(1) / mask.sum(1).clamp_min(1))
        dom = None
        if self.domain is not None:
            dom = self.domain(GradReverse.apply(pooled, grl_scale) if self.mode == "grl" else pooled)
        return self.emotion(pooled), dom


def build_optimizer(model: SSLClassifier, lr: float, head_lr: float, decay: float):
    n = len(model.encoder.encoder.layers)
    groups = []
    for i, layer in enumerate(model.encoder.encoder.layers):
        groups.append({"params": layer.parameters(), "lr": lr * decay ** (n - 1 - i)})
    rest = [p for nme, p in model.encoder.named_parameters()
            if not nme.startswith("encoder.layers.") and p.requires_grad]
    groups.append({"params": rest, "lr": lr * decay ** n})
    head = [model.layer_logits] + list(model.emotion.parameters())
    if model.domain is not None:
        head += list(model.domain.parameters())
    groups.append({"params": head, "lr": head_lr})
    for g in groups:
        g["base_lr"] = g["lr"]
    return torch.optim.AdamW(groups, weight_decay=0.01)


def evaluate(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=AMP and device.type == "cuda"):
        for waves, lengths, y, _ in loader:
            logits, _ = model(waves.to(device), lengths.to(device))
            probs.append(torch.softmax(logits.float(), 1).cpu())
            ys.append(y)
    probs, ys = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    pred = probs.argmax(1)
    return probs, ys, {
        "accuracy": accuracy_score(ys, pred),
        "balanced_accuracy": balanced_accuracy_score(ys, pred),
        "macro_f1": f1_score(ys, pred, average="macro", labels=sorted(set(ys)), zero_division=0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/wavlm-base-plus")
    p.add_argument("--tag", default="wavlm_v2")
    p.add_argument("--mode", choices=("none", "aux", "grl"), default="aux")
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--lr-decay", type=float, default=0.85)
    p.add_argument("--mask-prob", type=float, default=0.05)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--workers", type=int, default=4, help="DataLoader processes (waveform augmentation runs on the CPU)")
    p.add_argument("--noise-aug", action="store_true", help="train with white/pink/babble noise (0-30 dB SNR) on 70%% of the clips")
    p.add_argument("--save-model", action="store_true", help="store the best weights as best_model.pt in the run directory")
    p.add_argument("--freeze-layers", type=int, default=0, help="freeze the lowest N transformer layers (use ~8 for XLS-R on a 6 GB GPU)")
    p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less VRAM, slower)")
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 mixed precision on CUDA")
    p.add_argument("--limit", type=int, default=0, help="smoke test: rows per split")
    p.add_argument("--holdout", default=None, help="leave-one-corpus-out: corpus to test on")
    args = p.parse_args()

    global AMP
    AMP = not args.no_amp
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splits()
    split = data["experiment_split"]
    if args.holdout:
        train, val, test = (data[(split == "train") & (data.dataset != args.holdout)],
                            data[(split == "validation") & (data.dataset != args.holdout)],
                            data[data.dataset == args.holdout])
    else:
        train, val, test = (data[split == s] for s in ("train", "validation", "test"))
    if args.limit:
        train, val, test = (x.sample(min(args.limit, len(x)), random_state=0) for x in (train, val, test))
    name = f"{args.tag}__{args.mode}_{args.lam:g}__seed{args.seed}" + ("__noise" if args.noise_aug else "") + (f"__hold-{args.holdout}" if args.holdout else "")
    run_dir = OUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{name}: device={device} train={len(train)} val={len(val)} test={len(test)}", flush=True)

    mk = lambda df, tr, sh: DataLoader(WaveDataset(df, tr, args.noise_aug and tr), batch_size=args.batch_size, shuffle=sh, collate_fn=collate,
                                       num_workers=args.workers, persistent_workers=args.workers > 0,
                                       pin_memory=device.type == "cuda")
    train_loader, val_loader, test_loader = mk(train, True, True), mk(val, False, False), mk(test, False, False)
    model = SSLClassifier(args.model, int(data["dataset_id"].max()) + 1, args.mode, args.mask_prob, args.freeze_layers, args.grad_ckpt).to(device)
    opt = build_optimizer(model, args.lr, args.head_lr, args.lr_decay)
    counts = np.bincount(train["label_id"], minlength=len(LABELS)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1))).to(device)
    emo_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.smoothing)
    dom_loss = nn.CrossEntropyLoss()
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, int(0.1 * total_steps))

    def set_lr(step: int) -> None:
        f = step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f

    history, best, best_state, best_epoch, stale, step = [], -1.0, None, 0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for i, (waves, lengths, y, d) in enumerate(train_loader, 1):
            set_lr(step); step += 1
            scale = 2.0 / (1.0 + math.exp(-10 * step / total_steps)) - 1.0
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=AMP and device.type == "cuda"):
                logits, dom = model(waves.to(device, non_blocking=True), lengths.to(device), scale)
            loss = emo_loss(logits.float(), y.to(device))
            if dom is not None:
                loss = loss + args.lam * dom_loss(dom.float(), d.to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            if i % 50 == 0 or i == len(train_loader):
                el = (time.perf_counter() - started) / 60
                print(f"epoch {epoch} {i}/{len(train_loader)} loss={running / i:.3f} elapsed={el:.0f}m", flush=True)
        _, _, vm = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / len(train_loader), "val": vm})
        print(f"epoch {epoch}: val_macro_f1={vm['macro_f1']:.4f} val_acc={vm['accuracy']:.4f}", flush=True)
        if vm["macro_f1"] > best:
            best, best_epoch, stale = vm["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # keep VRAM free
        else:
            stale += 1
            if stale >= args.patience:
                print("early stopping"); break
    model.load_state_dict(best_state)
    if args.save_model:
        torch.save({"state": best_state, "args": vars(args), "labels": LABELS}, run_dir / "best_model.pt")
    v_probs, v_y, _ = evaluate(model, val_loader, device)
    t_probs, t_y, tm = evaluate(model, test_loader, device)
    tm["report"] = classification_report(t_y, t_probs.argmax(1), labels=list(range(len(LABELS))),
                                         target_names=LABELS, output_dict=True, zero_division=0)
    (run_dir / "result.json").write_text(json.dumps({
        "args": vars(args), "best_epoch": best_epoch, "best_val_macro_f1": best, "history": history,
        "test": tm, "minutes": (time.perf_counter() - started) / 60,
        "layer_weights": torch.softmax(model.layer_logits, 0).tolist()}, indent=1), encoding="utf-8")
    np.savez(run_dir / "probs.npz", val=v_probs, val_y=v_y, test=t_probs, test_y=t_y,
             test_idx=test.index.to_numpy(), val_idx=val.index.to_numpy())
    peak = f", peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else ""
    print(f"DONE {name}: test_macro_f1={tm['macro_f1']:.4f} acc={tm['accuracy']:.4f}{peak}", flush=True)


if __name__ == "__main__":
    main()
