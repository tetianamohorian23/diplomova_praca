from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "features" / "unified_features.csv"


def jitter_shimmer(path: Path) -> tuple[float, float]:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) < 512:
        return 0.0, 0.0
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=1024,
        hop_length=256,
    )
    voiced = np.isfinite(f0) & (f0 > 0)
    periods = 1.0 / np.maximum(f0[voiced], 1e-6)
    jitter = (
        float(np.std(periods) / np.maximum(np.mean(periods), 1e-6))
        if len(periods)
        else 0.0
    )
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    rms = rms[: len(f0)][voiced[: len(rms)]]
    shimmer = float(np.std(rms) / np.maximum(np.mean(rms), 1e-6)) if len(rms) else 0.0
    return jitter, shimmer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add jitter and shimmer acoustic features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input.with_name("unified_features_prosody.csv")
    data = pd.read_csv(args.input)
    values = []
    for index, filename in enumerate(data["filename"], start=1):
        normalized = ROOT / "audio_16k_mono" / Path(filename).relative_to(ROOT)
        jitter, shimmer = jitter_shimmer(normalized)
        values.append((jitter, shimmer))
        if index == 1 or index % 250 == 0 or index == len(data):
            print(f"jitter/shimmer: {index}/{len(data)}")
    data["jitter_mean"] = [value[0] for value in values]
    data["shimmer_mean"] = [value[1] for value in values]
    data.to_csv(output, index=False)
    print(f"Saved {len(data)} rows with jitter/shimmer to {output}")


if __name__ == "__main__":
    main()
