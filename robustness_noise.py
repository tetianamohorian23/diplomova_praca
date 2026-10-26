"""Noise robustness of the deployed model (frozen WavLM + 5 heads) on a stratified test subset."""
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from common_splits import CACHE, LABELS, load_splits
from predict_emotion import Predictor

torch.set_num_threads(8)
d = load_splits()
te = d[d.experiment_split == "test"]
import pandas as pd
sub = pd.concat([te[te.label_id == c].sample(min(int((te.label_id == c).sum()), 150), random_state=0) for c in range(len(LABELS))])
waves = np.load(CACHE.parent / "cnn_ssl" / "waves_int16.npy", mmap_mode="r")
lens = np.load(CACHE.parent / "cnn_ssl" / "lengths.npy")
pred = Predictor()
y = sub.label_id.to_numpy()
rng = np.random.default_rng(0)
out = {}
for snr in (None, 20, 10, 5, 0):
    preds = []
    for i in sub.index:
        w = waves[i, : lens[i]].astype(np.float32) / 32767
        if snr is not None:
            noise = rng.standard_normal(len(w)).astype(np.float32)
            w = w + noise * np.sqrt(np.mean(w ** 2) / 10 ** (snr / 10) / np.mean(noise ** 2))
        x = pred.features(w)
        p = torch.stack([torch.softmax(h((x - mu) / sd), 1)[0] for h, mu, sd in pred.heads]).mean(0)
        preds.append(int(p.argmax()))
    out["clean" if snr is None else f"snr_{snr}dB"] = {"macro_f1": float(f1_score(y, preds, average="macro")),
                                                       "accuracy": float(accuracy_score(y, preds)), "n": len(y)}
    print(snr, out["clean" if snr is None else f"snr_{snr}dB"], flush=True)
(CACHE.parent / "robustness_noise.json").write_text(json.dumps(out, indent=1))
