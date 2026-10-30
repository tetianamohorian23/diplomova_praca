"""Extract frozen per-layer mean/std pooled embeddings from a pretrained SSL model.

Output: experiments/v2/emb_<tag>.npz with arrays mean/std of shape (N, layers, hidden)
aligned with the rows of common_splits.load_splits().
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

from common_splits import CACHE, ROOT, load_splits

MAX_SAMPLES = 16_000 * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--tag", default="wavlm_base_plus")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = load_splits()
    model = AutoModel.from_pretrained(args.model).eval().to(args.device)
    means, stds = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for i, path in enumerate(data["audio_path"]):
            wave, _ = sf.read(ROOT / path, dtype="float32")
            if wave.ndim == 2:
                wave = wave.mean(1)
            wave = wave[:MAX_SAMPLES]
            wave = (wave - wave.mean()) / (wave.std() + 1e-7)
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
                out = model(torch.from_numpy(wave)[None].to(args.device), output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 0)[:, 0].float()  # (layers, T, H)
            means.append(hs.mean(1).cpu().numpy().astype(np.float16))
            stds.append(hs.std(1).cpu().numpy().astype(np.float16))
            if i % 200 == 0:
                el = time.perf_counter() - started
                print(f"{i}/{len(data)} elapsed={el/60:.1f}m ETA={el/(i+1)*(len(data)-i)/60:.1f}m", flush=True)
    np.savez(CACHE.parent / f"emb_{args.tag}.npz", mean=np.stack(means), std=np.stack(stds))
    print("done", time.perf_counter() - started)


if __name__ == "__main__":
    main()
