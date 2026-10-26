from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
PLOTS = EXPERIMENTS / "plots"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results: list[dict]) -> None:
    frame = pd.DataFrame(
        [
            {
                "model": result["model"],
                "category": result["category"],
                "Accuracy": result["accuracy"],
                "Balanced accuracy": result["balanced_accuracy"],
                "Macro F1": result["macro_f1"],
                "Macro precision": result["macro_precision"],
                "Macro recall": result["macro_recall"],
            }
            for result in results
        ]
    ).set_index("model")
    ax = frame.plot.bar(figsize=(12, 6), ylim=(0, 1), rot=20)
    ax.set_title("Test metrics by model")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    save(ax.get_figure(), "model_comparison.png")


def plot_per_class(results: list[dict]) -> None:
    rows = []
    for result in results:
        for label in LABELS:
            item = result["classification_report"][label]
            rows.append(
                {
                    "model": result["model"],
                    "emotion": label,
                    "Precision": item["precision"],
                    "Recall": item["recall"],
                    "F1": item["f1-score"],
                }
            )
    frame = pd.DataFrame(rows)
    for metric in ("Precision", "Recall", "F1"):
        pivot = frame.pivot(index="emotion", columns="model", values=metric)
        ax = pivot.plot.bar(figsize=(13, 6), ylim=(0, 1), rot=20)
        ax.set_title(f"Test {metric} by emotion")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        save(ax.get_figure(), f"per_class_{metric.lower()}.png")


def plot_confusion_matrices(results: list[dict]) -> None:
    for result in results:
        if "confusion_matrix" not in result:
            continue
        matrix = np.asarray(result["confusion_matrix"], dtype=float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(image, ax=ax, label="Recall proportion")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Predicted emotion")
        ax.set_ylabel("True emotion")
        ax.set_title(f"{result['model']} normalized confusion matrix")
        for row in range(len(LABELS)):
            for column in range(len(LABELS)):
                ax.text(
                    column,
                    row,
                    f"{normalized[row, column]:.2f}\n(n={int(matrix[row, column])})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalized[row, column] > 0.5 else "black",
                )
        save(fig, f"confusion_{result['model']}.png")


def plot_training_curves() -> None:
    histories = []
    for name in ("cnn_mel", "cnn_bilstm", "wav2vec2", "wavlm_multitask"):
        path = EXPERIMENTS / f"{name}.json"
        if not path.exists():
            continue
        data = load_json(path)
        histories.append((name, data))
    for name, data in histories:
        history = data.get("history", [])
        if not history:
            continue
        epochs = [item["epoch"] for item in history]
        train_f1 = [item["train"]["macro_f1"] for item in history]
        validation_f1 = [item["validation"]["macro_f1"] for item in history]
        validation_loss = [item["validation"]["loss"] for item in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, train_f1, marker="o", label="train Macro F1")
        axes[0].plot(epochs, validation_f1, marker="o", label="validation Macro F1")
        axes[0].set_title(f"{name} learning curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Macro F1")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(epochs, validation_loss, marker="o", color="tab:red")
        axes[1].set_title(f"{name} validation loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(alpha=0.3)
        save(fig, f"learning_curves_{name}.png")

    cnn_path = EXPERIMENTS / "cnn_mel.json"
    if cnn_path.exists():
        data = load_json(cnn_path)
        pretraining = data.get("pretraining", [])
        if pretraining and "reconstruction_loss" in pretraining[0]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                [item["epoch"] for item in pretraining],
                [item["reconstruction_loss"] for item in pretraining],
                marker="o",
            )
            ax.set_title("CNN self-supervised reconstruction loss")
            ax.set_xlabel("Pretraining epoch")
            ax.set_ylabel("MSE")
            ax.grid(alpha=0.3)
            save(fig, "cnn_ssl_reconstruction_loss.png")
        elif pretraining and "contrastive_loss" in pretraining[0]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                [item["epoch"] for item in pretraining],
                [item["contrastive_loss"] for item in pretraining],
                marker="o",
            )
            ax.set_title("CNN contrastive self-supervised loss")
            ax.set_xlabel("Pretraining epoch")
            ax.set_ylabel("NT-Xent loss")
            ax.grid(alpha=0.3)
            save(fig, "cnn_contrastive_loss.png")


def plot_class_distribution() -> None:
    manifest = ROOT / "combined_dataset_normalized.csv"
    if not manifest.exists():
        return
    data = pd.read_csv(manifest)
    data = data[data["emotion"].isin(LABELS)]
    counts = data["emotion"].value_counts().reindex(LABELS)
    ax = counts.plot.bar(figsize=(9, 5), color="tab:purple")
    ax.set_title("Seven-class dataset distribution")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Audio files")
    ax.grid(axis="y", alpha=0.3)
    save(ax.get_figure(), "class_distribution.png")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    results = load_json(EXPERIMENTS / "summary.json")
    plot_summary(results)
    plot_per_class(results)
    plot_confusion_matrices(results)
    plot_training_curves()
    plot_class_distribution()
    print(f"Saved experiment plots to {PLOTS}")


if __name__ == "__main__":
    main()
