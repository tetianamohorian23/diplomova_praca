"""Heads trained on clean + noisy WavLM embeddings (white/pink/babble noise, SNR 0-30 dB), compared on the same
noisy test subset with the standard heads. Saves models/noise_robust/*.pt and experiments/v2/noise_robust.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from common_splits import CACHE, LABELS, ROOT, load_splits
from predict_emotion import Head
from train_frozen_heads import load_features, run_one

V2 = CACHE.parent
out = ROOT / "models" / "noise_robust"
out.mkdir(parents=True, exist_ok=True)
data = load_splits()
X = load_features("wavlm_base_plus")
rows = np.load(V2 / "emb_noisy_trainval_rows.npy")
Xn = np.load(V2 / "emb_noisy_trainval.npy")[rows].astype(np.float32)
X2 = np.concatenate([X, Xn])
d2 = pd.concat([data, data.iloc[rows]], ignore_index=True)
clean_f1 = []
for seed in range(5):
    res, *_ = run_one("wavlm_base_plus", X2, d2, "none", 0.0, seed, save_path=out / f"wavlm_head_seed{seed}.pt")
    clean_f1.append(res["test"]["macro_f1"])
    print(f"seed {seed}: val_f1(clean+noisy)={res['val_macro_f1']:.3f} clean test_f1={clean_f1[-1]:.3f}", flush=True)


def load_heads(folder):
    hs = []
    for f in sorted(Path(folder).glob("wavlm_head_seed*.pt")):
        b = torch.load(f, map_location="cpu")
        h = Head(b["layers"], b["dim"], b["domains"], b["mode"]); h.load_state_dict(b["state"]); h.eval()
        hs.append((h, b["mu"], b["sd"]))
    return hs


@torch.inference_mode()
def predict(heads, F):
    x = torch.from_numpy(F.astype(np.float32))
    return torch.stack([torch.softmax(h((x - mu) / sd), 1) for h, mu, sd in heads]).mean(0).numpy()


z = np.load(V2 / "emb_noisy_test.npz")
y = data.label_id.to_numpy()[z["rows"]]
std, rob = load_heads(ROOT / "models"), load_heads(out)
table = []
for cond in [k for k in z.files if k != "rows"]:
    F = np.concatenate([z[cond][:, :, :768], z[cond][:, :, 768:]], 2) if z[cond].shape[-1] == 768 else z[cond]
    row = {"condition": cond}
    for name, heads in (("standard_F1", std), ("noise_robust_F1", rob)):
        row[name] = float(f1_score(y, predict(heads, F).argmax(1), average="macro"))
    table.append(row); print(row, flush=True)
(V2 / "noise_robust.json").write_text(json.dumps({"clean_test_f1_per_seed": clean_f1, "noisy_subset": table}, indent=1))
