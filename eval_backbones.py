"""Backbone comparison with the same plain head: default split (5 seeds) and leave-one-corpus-out (3 seeds).
Reads experiments/v2/frozen/*.json written by train_frozen_heads.py and writes experiments/v2/backbones.json."""
import glob
import json

import numpy as np

from common_splits import CACHE

V2 = CACHE.parent
NAMES = {"wav2vec2_base": "Wav2Vec2-base", "wavlm_base_plus": "WavLM-base-plus", "wavlm_large": "WavLM-large", "xlsr_300m": "XLS-R-300M (multilingual)"}
out = {}
for tag, name in NAMES.items():
    d, lo = [], {}
    for f in glob.glob(str(V2 / f"frozen/{tag}__none_0__seed*.json")):
        r = json.loads(open(f).read())
        if r["holdout"] is None: d.append(r["test"]["macro_f1"])
        else: lo.setdefault(r["holdout"], []).append(r["test"]["macro_f1"])
    if d:
        per = [np.mean(v) for v in lo.values()]
        out[name] = {"default_f1": float(np.mean(d)), "default_f1_std": float(np.std(d, ddof=1)), "n_seeds": len(d),
                     "loco_f1": float(np.mean(per)) if per else None, "loco_f1_std": float(np.std(per, ddof=1)) if per else 0.0,
                     "loco_per_corpus": {k: float(np.mean(v)) for k, v in lo.items()}}
(V2 / "backbones.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: {a: round(b, 3) for a, b in v.items() if isinstance(b, float)} for k, v in out.items()}, indent=1))
