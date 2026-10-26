from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_models
from train_deep_models import train_wavlm_multitask


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments" / "wavlm_ablation"
LABELS = train_models.models_labels()


def build_audio_splits(features: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    audio_data = pd.read_csv(ROOT / "combined_dataset_normalized.csv")
    audio_data = audio_data[audio_data["emotion"].isin(LABELS)].copy()
    split_by_key = {
        (row["dataset"], Path(row["filename"]).name): row["experiment_split"]
        for _, row in features.iterrows()
    }
    audio_data["experiment_split"] = [
        split_by_key.get((row["dataset"], row["filename"]))
        for _, row in audio_data.iterrows()
    ]
    audio_data = audio_data.dropna(subset=["experiment_split"])
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    audio_data["label_id"] = audio_data["emotion"].map(label_to_id)
    dataset_to_id = {
        name: index for index, name in enumerate(sorted(audio_data["dataset"].unique()))
    }
    audio_data["dataset_id"] = audio_data["dataset"].map(dataset_to_id)
    splits = tuple(
        audio_data[audio_data["experiment_split"] == split].copy()
        for split in ("train", "validation", "test")
    )
    if any(split.empty for split in splits):
        sizes = [len(split) for split in splits]
        raise RuntimeError(f"Could not build non-empty audio splits: {sizes}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run WavLM multi-task ablation without training other models."
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2],
        help="Dataset-task loss weights to compare.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA-enabled PyTorch device is available")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    from common_splits import load_splits

    data = load_splits()
    train_data, validation_data, test_data = (
        data[data["experiment_split"] == s].copy() for s in ("train", "validation", "test")
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []

    class ModelArgs:
        batch_size = args.batch_size
        wavlm_epochs = args.epochs
        wavlm_learning_rate = args.learning_rate
        wavlm_head_learning_rate = args.head_learning_rate
        wavlm_freeze_epochs = args.freeze_epochs
        wavlm_domain_loss_weight = 0.0
        wavlm_model = "microsoft/wavlm-base-plus"
        early_stopping_patience = args.patience

    for weight in args.weights:
        ModelArgs.wavlm_domain_loss_weight = weight
        ModelArgs.output_tag = f"_ablation_lambda_{weight:.2f}".replace(".", "_")
        print(f"\n=== WavLM ablation: dataset_loss_weight={weight:.2f} ===")
        result = train_wavlm_multitask(
            ModelArgs, train_data, validation_data, test_data, device
        )
        tag = f"lambda_{weight:.2f}".replace(".", "_")
        source_json = ROOT / "experiments" / f"wavlm_multitask{ModelArgs.output_tag}.json"
        target_json = OUTPUT / f"{tag}.json"
        shutil.move(source_json, target_json)
        summary.append(
            {
                "dataset_loss_weight": weight,
                "test_accuracy": result["accuracy"],
                "test_macro_f1": result["macro_f1"],
                "best_validation_macro_f1": result["best_validation_macro_f1"],
                "best_epoch": result["best_epoch"],
                "result_file": str(target_json),
            }
        )
        print(
            f"COMPLETED λ={weight:.2f}: "
            f"validation_macro_f1={result['best_validation_macro_f1']:.4f}, "
            f"test_macro_f1={result['macro_f1']:.4f}"
        )

    summary.sort(key=lambda item: item["best_validation_macro_f1"], reverse=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== BEST WavLM CONFIGURATION BY VALIDATION ===")
    print(json.dumps(summary[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
