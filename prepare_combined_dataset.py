from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "combined_dataset.csv"
NORMALIZED_OUTPUT = ROOT / "combined_dataset_normalized.csv"
NORMALIZED_ROOT = ROOT / "audio_16k_mono"
TARGET_SAMPLE_RATE = 16_000

EMOTION_MAPS = {
    "CREMA-D": {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    },
    "EMODB": {
        "A": "angry",
        "B": "boredom",
        "E": "disgust",
        "F": "fear",
        "L": "boredom",
        "N": "neutral",
        "T": "sad",
        "W": "happy",
    },
    "RAVDESS": {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    },
    "SAVEE": {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprise",
    },
    "SVK-DB": {
        "A": "angry",
        "E": "disgust",
        "F": "fear",
        "N": "neutral",
        "T": "sad",
        "W": "happy",
    },
    "TESS": {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "ps": "surprise",
        "sad": "sad",
    },
    "UA-SER": {
        "angry": "angry",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
    },
}


def audio_info(path: Path) -> tuple[int, int, float]:
    info = sf.info(path)
    sample_rate = info.samplerate
    channels = info.channels
    duration = info.duration
    return sample_rate, channels, round(duration, 4)


def normalize_audio(source: Path, destination: Path) -> None:
    audio, sample_rate = sf.read(source, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        mono = scipy.signal.resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, mono, TARGET_SAMPLE_RATE, subtype="PCM_16")


def parse_file(dataset: str, path: Path) -> tuple[str, str, str, str]:
    """Return source id, original label, normalized label, and language."""
    name = path.stem
    if dataset == "CREMA-D":
        parts = name.split("_")
        source_id, original = parts[0], parts[2]
        language = "en"
    elif dataset == "EMODB":
        source_id, original = name[:2], name[5]
        language = "de"
    elif dataset == "RAVDESS":
        parts = name.split("-")
        source_id, original = f"actor_{parts[-1]}", parts[2]
        language = "en"
    elif dataset == "SAVEE":
        match = re.match(r"(?P<label>sa|su|[adfhn])\d+", name)
        if match is None:
            raise ValueError(f"Unsupported SAVEE filename: {path.name}")
        source_id, original = path.parent.name, match.group("label")
        language = "en"
    elif dataset == "SVK-DB":
        source_id, original = name[:2], name[5]
        language = "sk"
    elif dataset == "TESS":
        source_id, original = name.split("_")[0], name.split("_")[-1]
        language = "en"
    elif dataset == "UA-SER":
        source_id, original = "", ""
        language = "uk"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return source_id, original, EMOTION_MAPS[dataset][original], language


def collect_non_ua(dataset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / dataset).rglob("*.wav")):
        source_id, original, emotion, language = parse_file(dataset, path)
        sample_rate, channels, duration = audio_info(path)
        rows.append(
            {
                "dataset": dataset,
                "audio_path": path.relative_to(ROOT).as_posix(),
                "filename": path.name,
                "source_id": source_id,
                "emotion_original": original,
                "emotion": emotion,
                "language": language,
                "sample_rate_hz": sample_rate,
                "channels": channels,
                "duration_seconds": duration,
                "split": "",
                "text": "",
            }
        )
    return rows


def collect_ua() -> list[dict[str, object]]:
    metadata = ROOT / "UA-SER" / "dataset.csv"
    rows: list[dict[str, object]] = []
    with metadata.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            path = ROOT / "UA-SER" / "clips" / item["filename"]
            if not path.exists():
                raise FileNotFoundError(path)
            sample_rate, channels, duration = audio_info(path)
            rows.append(
                {
                    "dataset": "UA-SER",
                    "audio_path": path.relative_to(ROOT).as_posix(),
                    "filename": item["filename"],
                    "source_id": item["speaker_id"],
                    "emotion_original": item["emotion"],
                    "emotion": item["emotion"],
                    "language": "uk",
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    "duration_seconds": duration,
                    "split": item["split"],
                    "text": item["text"],
                }
            )
    return rows


def assign_group_splits(rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["dataset"] == "UA-SER":
            continue
        key = f"{row['dataset']}::{row['source_id']}"
        grouped.setdefault(key, []).append(row)

    by_dataset: dict[str, list[str]] = {}
    for key in grouped:
        dataset, _ = key.split("::", 1)
        by_dataset.setdefault(dataset, []).append(key)

    for dataset, keys in by_dataset.items():
        ordered = sorted(
            keys,
            key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
        )
        test_count = max(1, math.ceil(len(ordered) * 0.2))
        test_keys = set(ordered[-test_count:])
        for key in keys:
            split = "test" if key in test_keys else "train"
            for row in grouped[key]:
                row["split"] = split


def write_manifest(rows: list[dict[str, object]], output: Path) -> None:
    fields = [
        "dataset",
        "audio_path",
        "filename",
        "source_id",
        "emotion_original",
        "emotion",
        "language",
        "sample_rate_hz",
        "channels",
        "duration_seconds",
        "split",
        "text",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        source = ROOT / str(row["audio_path"])
        relative_source = Path(str(row["audio_path"]))
        destination = NORMALIZED_ROOT / relative_source.with_suffix(".wav")
        normalize_audio(source, destination)
        normalized = dict(row)
        normalized["audio_path"] = destination.relative_to(ROOT).as_posix()
        normalized["sample_rate_hz"] = TARGET_SAMPLE_RATE
        normalized["channels"] = 1
        normalized["duration_seconds"] = audio_info(destination)[2]
        normalized_rows.append(normalized)
    return normalized_rows


def main() -> None:
    rows = collect_ua()
    for dataset in ("CREMA-D", "EMODB", "RAVDESS", "SAVEE", "SVK-DB", "TESS"):
        rows.extend(collect_non_ua(dataset))
    assign_group_splits(rows)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["audio_path"])))
    write_manifest(rows, OUTPUT)
    normalized_rows = normalize_rows(rows)
    write_manifest(normalized_rows, NORMALIZED_OUTPUT)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(f"Wrote {len(normalized_rows)} normalized files to {NORMALIZED_ROOT}")
    print(f"Wrote normalized manifest to {NORMALIZED_OUTPUT}")


if __name__ == "__main__":
    main()
