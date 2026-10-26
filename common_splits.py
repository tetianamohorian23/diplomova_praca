"""Shared train/validation/test split used by every experiment (cached to CSV)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "experiments" / "v2" / "splits.csv"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_splits() -> pd.DataFrame:
    """Return one frame with columns audio_path, dataset, language, label_id,
    dataset_id and experiment_split (train/validation/test)."""
    if CACHE.exists():
        return pd.read_csv(CACHE)
    import train_models

    features = pd.read_csv(train_models.FEATURES)
    features = features[features["label"].isin(LABELS)].copy()
    features = train_models.restrict_to_common_audio(features)
    features = train_models.add_validation_split(features)
    audio = pd.read_csv(ROOT / "combined_dataset_normalized.csv")
    audio["key"] = [
        str((ROOT / p).resolve()).lower() for p in audio["audio_path"]
    ]
    features["key"] = [
        str((ROOT / "audio_16k_mono" / Path(f).relative_to(ROOT)).resolve()).lower()
        for f in features["filename"]
    ]
    audio = audio.drop(columns=["split", "filename"]).drop_duplicates("key")
    keep = features[["key", "experiment_split", "speaker_id", "label"]]
    data = keep.merge(audio, on="key", how="left", validate="one_to_one")
    assert data["audio_path"].notna().all()
    data["label_id"] = data["label"].map({l: i for i, l in enumerate(LABELS)})
    names = sorted(data["dataset"].unique())
    data["dataset_id"] = data["dataset"].map({n: i for i, n in enumerate(names)})
    data = data.drop(columns=["key", "emotion", "text"]).reset_index(drop=True)
    data = data.rename(columns={"label": "emotion"})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(CACHE, index=False)
    return data


if __name__ == "__main__":
    d = load_splits()
    print(d.groupby(["experiment_split"]).size())
    print(d.pivot_table(index="dataset", columns="emotion", values="audio_path", aggfunc="count", fill_value=0))
    print(d.pivot_table(index="dataset", columns="experiment_split", values="audio_path", aggfunc="count", fill_value=0))
    print(d.columns.tolist())
