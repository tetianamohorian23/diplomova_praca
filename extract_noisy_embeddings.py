"""WavLM per-layer embeddings of noisy copies of the clips.
  train : one random-noise copy (type random white/pink/babble, SNR U(0,30) dB) of every train+validation clip
  test  : the stratified 1050-clip test subset of robustness_noise.py under fixed conditions
"""
import sys
import numpy as np
import torch
from transformers import AutoModel

from common_splits import CACHE, LABELS, load_splits
from noise_utils import babble, mix, pink, white

V2 = CACHE.parent
torch.set_num_threads(8)
stage = sys.argv[1]
d = load_splits()
waves = np.load(V2 / "cnn_ssl/waves_int16.npy", mmap_mode="r")
lens = np.load(V2 / "cnn_ssl/lengths.npy")
rng = np.random.default_rng(1 if stage == "train" else 2)
enc = AutoModel.from_pretrained("microsoft/wavlm-base-plus").eval()
train_rows = np.where(d.experiment_split.isin(["train", "validation"]))[0]
pool = lambda: (lambda i: waves[i, : lens[i]].astype(np.float32) / 32767)(rng.choice(train_rows))


def emb(w):
    w = (w - w.mean()) / (w.std() + 1e-7)
    with torch.inference_mode():
        hs = torch.stack(enc(torch.from_numpy(w.astype(np.float32))[None], output_hidden_states=True).hidden_states, 0)[:, 0]
    return np.concatenate([hs.mean(1).numpy(), hs.std(1).numpy()], 1).astype(np.float16)


def noisy(i, kind, snr):
    w = waves[i, : lens[i]].astype(np.float32) / 32767
    n = {"white": lambda: white(len(w), rng), "pink": lambda: pink(len(w), rng), "babble": lambda: babble(len(w), rng, pool)}[kind]()
    return mix(w, n, snr)


if stage == "train":
    out = np.zeros((len(d), 13, 1536), np.float16); meta = []
    for c, i in enumerate(train_rows):
        kind = rng.choice(["white", "pink", "babble"]); snr = float(rng.uniform(0, 30))
        out[i] = emb(noisy(i, kind, snr)); meta.append((int(i), str(kind), snr))
        if c % 500 == 0: print(c, len(train_rows), flush=True)
    np.save(V2 / "emb_noisy_trainval.npy", out); np.save(V2 / "emb_noisy_trainval_rows.npy", train_rows)
else:
    te = d[d.experiment_split == "test"]
    sub = np.concatenate([te[te.label_id == c].sample(min(int((te.label_id == c).sum()), 150), random_state=0).index.to_numpy() for c in range(len(LABELS))])
    conds = [("white", 20), ("white", 10), ("white", 5), ("white", 0), ("pink", 10), ("babble", 20), ("babble", 10), ("babble", 5)]
    res = {"rows": sub, "clean": np.stack([emb(waves[i, : lens[i]].astype(np.float32) / 32767) for i in sub])}
    for kind, snr in conds:
        res[f"{kind}_{snr}"] = np.stack([emb(noisy(i, kind, snr)) for i in sub]); print(kind, snr, flush=True)
    np.savez(V2 / "emb_noisy_test.npz", **res)
