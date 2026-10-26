"""Train light heads on frozen per-layer SSL embeddings (fast: ablations, seeds, cross-corpus).

Model: softmax-weighted sum over layers -> LayerNorm -> MLP -> emotion head,
plus an optional corpus head used as
  * mode "aux": auxiliary multi-task loss (what the original WavLM run did), or
  * mode "grl": adversarial corpus head behind a gradient-reversal layer (domain-invariant).
Selection of the epoch uses validation macro-F1 only; the test split is scored once.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn

from common_splits import CACHE, LABELS, load_splits

OUT = CACHE.parent / "frozen"
torch.set_num_threads(4)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


class Head(nn.Module):
    def __init__(self, layers: int, dim: int, domains: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.norm = nn.LayerNorm(dim)
        self.body = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.3))
        self.emotion = nn.Linear(256, len(LABELS))
        self.domain = nn.Linear(256, domains) if mode != "none" else None

    def forward(self, x, grl_scale: float = 1.0):
        w = torch.softmax(self.layer_logits, 0)
        h = self.body(self.norm((x * w[None, :, None]).sum(1)))
        dom = None
        if self.domain is not None:
            dom = self.domain(GradReverse.apply(h, grl_scale) if self.mode == "grl" else h)
        return self.emotion(h), dom


def load_features(tag: str) -> np.ndarray:
    z = np.load(CACHE.parent / f"emb_{tag}.npz")
    return np.concatenate([z["mean"], z["std"]], axis=2).astype(np.float32)


def metrics(y, p) -> dict:
    labels = sorted(set(y))
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, labels=labels, average="macro", zero_division=0),
        "per_class_f1": dict(
            zip([LABELS[i] for i in labels], f1_score(y, p, labels=labels, average=None, zero_division=0).tolist())
        ),
    }


def run_one(tag, X, data, mode, lam, seed, holdout=None, epochs=60, patience=10, noise=0.1, smoothing=0.1, save_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    split = data["experiment_split"].to_numpy()
    ds = data["dataset"].to_numpy()
    if holdout:
        train_m = (split == "train") & (ds != holdout)
        val_m = (split == "validation") & (ds != holdout)
        test_m = ds == holdout
        if train_m.sum() == 0:
            raise ValueError("empty train")
    else:
        train_m, val_m, test_m = split == "train", split == "validation", split == "test"
    mu = X[train_m].mean(0, keepdims=True)
    sd = X[train_m].std(0, keepdims=True) + 1e-5
    Xn = torch.from_numpy((X - mu) / sd)
    y = torch.from_numpy(data["label_id"].to_numpy())
    d = torch.from_numpy(data["dataset_id"].to_numpy())
    tr = torch.from_numpy(np.where(train_m)[0])
    counts = np.bincount(y[tr].numpy(), minlength=len(LABELS)).astype(np.float32)
    weights = torch.from_numpy(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
    model = Head(X.shape[1], X.shape[2], int(d.max()) + 1, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(tr) + 127) // 128))
    emo_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
    dom_loss = nn.CrossEntropyLoss()

    def predict(mask):
        model.eval()
        idx = torch.from_numpy(np.where(mask)[0])
        with torch.no_grad():
            logits = model(Xn[idx])[0]
        return idx, torch.softmax(logits, 1).numpy()

    best, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 128):
            b = perm[i : i + 128]
            xb = Xn[b] + noise * torch.randn_like(Xn[b])
            p = (epoch - 1) / epochs
            scale = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            logits, dom = model(xb, scale)
            loss = emo_loss(logits, y[b])
            if dom is not None:
                loss = loss + lam * dom_loss(dom, d[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        idx, probs = predict(val_m)
        score = f1_score(y[idx].numpy(), probs.argmax(1), average="macro")
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    if save_path is not None:
        torch.save({"state": best_state, "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd),
                    "layers": X.shape[1], "dim": X.shape[2], "labels": LABELS, "mode": mode,
                    "domains": int(d.max()) + 1}, save_path)
    v_idx, v_probs = predict(val_m)
    t_idx, t_probs = predict(test_m)
    result = {
        "embedding": tag, "mode": mode, "lambda": lam, "seed": seed, "holdout": holdout,
        "best_epoch": best_epoch, "val_macro_f1": best,
        "test": metrics(y[t_idx].numpy(), t_probs.argmax(1)),
        "layer_weights": torch.softmax(model.layer_logits, 0).tolist(),
    }
    return result, v_probs, t_probs, t_idx.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", default=["wavlm_base_plus", "wav2vec2_base"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--loco", action="store_true", help="leave-one-corpus-out instead of the default split")
    parser.add_argument("--loco-config", default="none:0.0", help="comma list mode:lambda for LOCO")
    parser.add_argument("--only-none", action="store_true", help="default split: train only the plain head (no corpus task)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_splits()
    if args.loco:
        configs = [(c.split(":")[0], float(c.split(":")[1])) for c in args.loco_config.split(",")]
        holdouts = [h for h in sorted(data["dataset"].unique()) if h != "SVK-DB"]
    else:
        configs = [("none", 0.0)] + ([] if args.only_none else [(m, l) for m in ("aux", "grl") for l in args.lambdas])
        holdouts = [None]
    for tag in args.embeddings:
        X = load_features(tag)
        for hold in holdouts:
            for mode, lam in configs:
                for seed in range(args.seeds):
                    name = f"{tag}__{mode}_{lam:g}__seed{seed}" + (f"__hold-{hold}" if hold else "")
                    path = OUT / f"{name}.json"
                    if path.exists():
                        continue
                    started = time.perf_counter()
                    res, v_probs, t_probs, t_idx = run_one(tag, X, data, mode, lam, seed, hold)
                    path.write_text(json.dumps(res), encoding="utf-8")
                    np.savez(OUT / f"{name}.npz", val=v_probs, test=t_probs, test_idx=t_idx)
                    print(f"{name}: val={res['val_macro_f1']:.3f} test_f1={res['test']['macro_f1']:.3f} "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
