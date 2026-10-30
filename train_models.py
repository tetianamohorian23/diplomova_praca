from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features" / "unified_features_prosody.csv"
if not FEATURES.exists():
    FEATURES = ROOT / "features" / "unified_features.csv"
OUTPUT = ROOT / "experiments"


def evaluate(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(set(y_train) | set(y_test))
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "weighted_precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": labels,
    }
    total = len(y_test)
    metrics["per_class_accuracy"] = {
        label: float(
            (
                confusion_matrix(y_test, predictions, labels=labels)[index, index]
                + total
                - sum(confusion_matrix(y_test, predictions, labels=labels)[index, :])
                - sum(confusion_matrix(y_test, predictions, labels=labels)[:, index])
                + confusion_matrix(y_test, predictions, labels=labels)[index, index]
            )
            / total
        )
        for index, label in enumerate(labels)
    }
    model_path = OUTPUT / f"{name}.joblib"
    joblib.dump(model, model_path)
    with (OUTPUT / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(
        f"{name}: accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"macro_precision={metrics['macro_precision']:.4f}, "
        f"macro_recall={metrics['macro_recall']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    return metrics


def oversample_training(data: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    counts = data[label_column].value_counts()
    target = int(counts.max())
    parts = []
    for label, group in data.groupby(label_column, sort=False):
        parts.append(group.sample(target, replace=len(group) < target, random_state=42))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42)


def add_validation_split(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["experiment_split"] = data["split"]
    groups = data.loc[data["split"] == "train", ["dataset", "speaker_id"]].drop_duplicates()
    for dataset in groups["dataset"].unique():
        dataset_groups = groups[groups["dataset"] == dataset].copy()
        ordered = sorted(
            dataset_groups.itertuples(index=False),
            key=lambda row: hashlib.sha256(
                f"{row.dataset}::{row.speaker_id}".encode()
            ).hexdigest(),
        )
        validation_count = max(1, round(len(ordered) * 0.2))
        validation_keys = {
            (row.dataset, row.speaker_id) for row in ordered[-validation_count:]
        }
        mask = data.apply(
            lambda row: (row["dataset"], row["speaker_id"]) in validation_keys
            and row["split"] == "train",
            axis=1,
        )
        data.loc[mask, "experiment_split"] = "validation"
    return data


def restrict_to_common_audio(data: pd.DataFrame) -> pd.DataFrame:
    audio = pd.read_csv(ROOT / "combined_dataset_normalized.csv")
    audio = audio[audio["emotion"].isin(models_labels())]
    audio_labels = {
        str((ROOT / row["audio_path"]).resolve()).lower(): row["emotion"]
        for _, row in audio.iterrows()
    }
    keys = []
    canonical_labels = {}
    for _, row in data.iterrows():
        relative_path = Path(row["filename"]).relative_to(ROOT)
        normalized_path = str((ROOT / "audio_16k_mono" / relative_path).resolve()).lower()
        if normalized_path in audio_labels:
            keys.append(row.name)
            canonical_labels[row.name] = audio_labels[normalized_path]
    result = data.loc[keys].copy()
    result["label"] = pd.Series(canonical_labels)
    return result


def print_model_completion(model_name: str, result: dict) -> None:
    print(f"\n--- {model_name} COMPLETED ---")
    if "best_epoch" in result:
        print(
            f"training: completed, best_epoch={result['best_epoch']}, "
            f"best_validation_macro_f1={result['best_validation_macro_f1']:.4f}"
        )
    else:
        print("training: completed")
    print(
        f"testing: completed, accuracy={result['accuracy']:.4f}, "
        f"macro_f1={result['macro_f1']:.4f}"
    )
    print(f"saved: experiments/{model_name}.json")


def tune_classical_model(name, template, train, validation, test, feature_columns) -> dict:
    candidates = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_score = -1.0
    best_c = candidates[0]
    for c_value in candidates:
        candidate = clone(template)
        candidate.set_params(
            **{
                (
                    "logisticregression__C"
                    if name == "logistic_regression"
                    else "linearsvc__C"
                ): c_value
            }
        )
        fit_train = (
            oversample_training(train)
            if name != "logistic_regression"
            else train
        )
        candidate.fit(fit_train[feature_columns], fit_train["label"])
        prediction = candidate.predict(validation[feature_columns])
        score = f1_score(validation["label"], prediction, average="macro")
        print(f"{name}: validation C={c_value} macro_f1={score:.4f}")
        if score > best_score:
            best_score = score
            best_c = c_value
    final_model = clone(template)
    final_model.set_params(
        **{
            (
                "logisticregression__C"
                if name == "logistic_regression"
                else "linearsvc__C"
            ): best_c
        }
    )
    combined = pd.concat([train, validation], ignore_index=True)
    if name != "logistic_regression":
        combined = oversample_training(combined)
    print(f"{name}: selected C={best_c} validation_macro_f1={best_score:.4f}")
    result = evaluate(
        name,
        final_model,
        combined[feature_columns],
        combined["label"],
        test[feature_columns],
        test["label"],
    )
    result["selected_c"] = best_c
    result["best_validation_macro_f1"] = best_score
    return result


def train_semi_supervised_svm(
    train,
    validation,
    test,
    feature_columns,
    labeled_fraction: float = 0.2,
    confidence_threshold: float = 0.8,
    max_iterations: int = 5,
    c_value: float = 0.1,
) -> dict:
    labeled = train.sample(frac=labeled_fraction, random_state=42)
    unlabeled = train.drop(index=labeled.index)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearSVC(
            class_weight="balanced",
            C=c_value,
            dual="auto",
            max_iter=5000,
            random_state=42,
        ),
    )
    pseudo_count = 0
    iterations_completed = 0
    for iteration in range(max_iterations):
        iterations_completed = iteration + 1
        balanced_labeled = oversample_training(labeled)
        model.fit(balanced_labeled[feature_columns], balanced_labeled["label"])
        scores = model.decision_function(unlabeled[feature_columns])
        top_two = np.partition(scores, -2, axis=1)[:, -2:]
        confidence = top_two[:, 1] - top_two[:, 0]
        keep = confidence >= confidence_threshold
        if not keep.any():
            break
        pseudo = unlabeled.loc[keep].copy()
        pseudo["label"] = model.classes_[scores[keep].argmax(axis=1)]
        labeled = pd.concat([labeled, pseudo], ignore_index=True)
        unlabeled = unlabeled.drop(index=pseudo.index)
        pseudo_count += len(pseudo)
        print(
            f"semi_supervised_svm: iteration={iteration + 1} "
            f"pseudo_labels_added={len(pseudo)}"
        )
    balanced_labeled = oversample_training(labeled)
    model.fit(balanced_labeled[feature_columns], balanced_labeled["label"])
    metrics = evaluate(
        "semi_supervised_svm",
        model,
        labeled[feature_columns],
        labeled["label"],
        test[feature_columns],
        test["label"],
    )
    metrics["category"] = "semi-supervised"
    metrics["labeled_fraction"] = labeled_fraction
    metrics["confidence_threshold"] = confidence_threshold
    metrics["max_iterations"] = max_iterations
    metrics["selected_c"] = c_value
    metrics["pseudo_labels_added"] = pseudo_count
    metrics["self_training_iterations"] = iterations_completed
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate classical, CNN, Wav2Vec2, and WavLM models."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Set both CNN and Wav2Vec2 epochs; individual options take precedence.",
    )
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--cnn-ssl-epochs", type=int, default=5)
    parser.add_argument("--lstm-epochs", type=int, default=30)
    parser.add_argument("--lstm-learning-rate", type=float, default=1e-3)
    parser.add_argument("--wav2vec-epochs", type=int, default=5)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--wav2vec-batch-size", type=int, default=4)
    parser.add_argument("--cnn-learning-rate", type=float, default=1e-4)
    parser.add_argument("--wav2vec-learning-rate", type=float, default=1e-5)
    parser.add_argument("--wav2vec-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--wav2vec-freeze-epochs", type=int, default=1)
    parser.add_argument("--wavlm-epochs", type=int, default=5)
    parser.add_argument("--wavlm-learning-rate", type=float, default=1e-5)
    parser.add_argument("--wavlm-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--wavlm-freeze-epochs", type=int, default=1)
    parser.add_argument("--wavlm-domain-loss-weight", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--svm-labeled-fraction", type=float, default=0.5)
    parser.add_argument("--svm-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--svm-iterations", type=int, default=5)
    parser.add_argument("--svm-c", type=float, default=0.1)
    parser.add_argument("--wav2vec-model", default="facebook/wav2vec2-base")
    parser.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs is not None:
        args.cnn_epochs = args.epochs
        args.wav2vec_epochs = args.epochs

    data = pd.read_csv(FEATURES)
    data = data[data["label"].isin(models_labels())].copy()
    data = restrict_to_common_audio(data)
    data = add_validation_split(data)
    feature_columns = [
        column
        for column in data.columns
        if column not in {
            "filename",
            "dataset",
            "speaker_id",
            "label",
            "original_label",
            "split",
            "experiment_split",
        }
    ]
    train = data[data["experiment_split"] == "train"]
    validation = data[data["experiment_split"] == "validation"]
    test = data[data["experiment_split"] == "test"]
    OUTPUT.mkdir(exist_ok=True)

    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        ),
        "linear_svm": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=5000,
                random_state=42,
            ),
        ),
    }

    results = []
    print("\n=== CLASSICAL MODELS: TRAINING AND VALIDATION ===")
    print("\n[1/6] Training Logistic Regression and selecting C on validation...")
    supervised = tune_classical_model(
        "logistic_regression",
        models["logistic_regression"],
        train,
        validation,
        test,
        feature_columns,
    )
    supervised["category"] = "supervised"
    results.append(supervised)
    print_model_completion("logistic_regression", supervised)
    print("\n[2/6] Training semi-supervised SVM and selecting pseudo-labels...")
    results.append(
        train_semi_supervised_svm(
            train,
            validation,
            test,
            feature_columns,
            labeled_fraction=args.svm_labeled_fraction,
            confidence_threshold=args.svm_confidence_threshold,
            max_iterations=args.svm_iterations,
            c_value=args.svm_c,
        )
    )
    print_model_completion("semi_supervised_svm", results[-1])

    from train_deep_models import (
        train_cnn,
        train_cnn_bilstm,
        train_wav2vec,
        train_wavlm_multitask,
    )

    audio_data = pd.read_csv(ROOT / "combined_dataset_normalized.csv")
    audio_data = audio_data[audio_data["emotion"].isin(models_labels())].copy()
    normalized_paths = {
        str((ROOT / "audio_16k_mono" / Path(path).relative_to(ROOT)).resolve()).lower()
        for path in data["filename"]
    }
    audio_data["_absolute_path"] = audio_data["audio_path"].map(
        lambda path: str((ROOT / path).resolve()).lower()
    )
    audio_data = audio_data[audio_data["_absolute_path"].isin(normalized_paths)].copy()
    split_by_path = {
        str(
            (ROOT / "audio_16k_mono" / Path(row["filename"]).relative_to(ROOT)).resolve()
        ).lower(): row["experiment_split"]
        for _, row in data.iterrows()
    }
    audio_data["experiment_split"] = audio_data["_absolute_path"].map(split_by_path)
    audio_data = audio_data.dropna(subset=["experiment_split"]).drop(columns="_absolute_path")
    label_to_id = {label: index for index, label in enumerate(models_labels())}
    audio_data["label_id"] = audio_data["emotion"].map(label_to_id)
    dataset_to_id = {name: index for index, name in enumerate(sorted(audio_data["dataset"].unique()))}
    audio_data["dataset_id"] = audio_data["dataset"].map(dataset_to_id)
    audio_train = audio_data[audio_data["experiment_split"] == "train"]
    audio_validation = audio_data[audio_data["experiment_split"] == "validation"]
    audio_test = audio_data[audio_data["experiment_split"] == "test"]
    if args.limit:
        audio_train = audio_train.groupby("label_id", group_keys=False).head(args.limit)
        audio_validation = audio_validation.groupby("label_id", group_keys=False).head(
            max(1, args.limit // 4)
        )
        audio_test = audio_test.groupby("label_id", group_keys=False).head(max(1, args.limit // 4))

    class DeepArgs:
        batch_size = args.cnn_batch_size
        epochs = args.cnn_epochs
        cnn_ssl_epochs = args.cnn_ssl_epochs
        lstm_epochs = args.lstm_epochs
        lstm_learning_rate = args.lstm_learning_rate
        learning_rate = args.cnn_learning_rate
        wav2vec_learning_rate = args.wav2vec_learning_rate
        wav2vec_head_learning_rate = args.wav2vec_head_learning_rate
        wav2vec_freeze_epochs = args.wav2vec_freeze_epochs
        wavlm_epochs = args.wavlm_epochs
        wavlm_learning_rate = args.wavlm_learning_rate
        wavlm_head_learning_rate = args.wavlm_head_learning_rate
        wavlm_freeze_epochs = args.wavlm_freeze_epochs
        wavlm_domain_loss_weight = args.wavlm_domain_loss_weight
        wav2vec_model = args.wav2vec_model
        wavlm_model = args.wavlm_model
        early_stopping_patience = args.early_stopping_patience

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA-enabled PyTorch device is available")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device in ("auto", "cpu")
        else args.device
    )
    device = torch.device(device_name)
    print(f"\nDEVICE: {device} (CPU training is {'enabled' if device.type == 'cpu' else 'disabled'})")
    print(
        f"\nCOMMON SPLIT: train={len(train)} validation={len(validation)} "
        f"test={len(test)} total={len(data)} (same files for all models)"
    )
    print(
        f"\nCNN: train={len(audio_train)}, validation={len(audio_validation)}, "
        f"test={len(audio_test)}, device={device}"
    )
    print("\n[3/6] CNN contrastive pretraining, fine-tuning, validation, and testing...")
    cnn_metrics = train_cnn(DeepArgs, audio_train, audio_validation, audio_test, device)
    results.append({"model": "cnn_mel", "category": "self-supervised", **cnn_metrics})
    print_model_completion("cnn_mel", results[-1])
    print(
        f"\nCNN-BiLSTM: train={len(audio_train)}, validation={len(audio_validation)}, "
        f"test={len(audio_test)}, device={device}"
    )
    print("\n[4/6] CNN-BiLSTM multi-task training, validation, and testing...")
    lstm_metrics = train_cnn_bilstm(
        DeepArgs, audio_train, audio_validation, audio_test, device
    )
    results.append({"model": "cnn_bilstm", **lstm_metrics})
    print_model_completion("cnn_bilstm", results[-1])
    DeepArgs.batch_size = args.wav2vec_batch_size
    DeepArgs.epochs = args.wav2vec_epochs
    print(
        f"\nWav2Vec2: train={len(audio_train)}, validation={len(audio_validation)}, "
        f"test={len(audio_test)}, device={device}"
    )
    print("\n[5/6] Wav2Vec2 training, validation, checkpoint selection, and testing...")
    wav2vec_metrics = train_wav2vec(
        DeepArgs, audio_train, audio_validation, audio_test, device
    )
    results.append({"model": "wav2vec2", "category": "transfer-learning", **wav2vec_metrics})
    print_model_completion("wav2vec2", results[-1])
    DeepArgs.batch_size = args.wav2vec_batch_size
    print(
        f"\nWavLM multitask: train={len(audio_train)}, validation={len(audio_validation)}, "
        f"test={len(audio_test)}, device={device}"
    )
    print("\n[6/6] WavLM multi-task training, validation, and testing...")
    wavlm_metrics = train_wavlm_multitask(
        DeepArgs, audio_train, audio_validation, audio_test, device
    )
    results.append({"model": "wavlm_multitask", **wavlm_metrics})
    print_model_completion("wavlm_multitask", results[-1])

    with (OUTPUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print("\n=== FINAL RESULTS ===")
    print(
        f"{'Category':<20} {'Model':<22} {'Accuracy':>10} {'Precision':>12} "
        f"{'Recall':>10} {'Macro F1':>10} {'Selection':>28}"
    )
    for result in results:
        if result["model"] == "logistic_regression":
            selection = (
                f"C={result['selected_c']}, val_f1={result['best_validation_macro_f1']:.4f}"
            )
        elif result["model"] == "semi_supervised_svm":
            selection = (
                f"pseudo={result['pseudo_labels_added']}, "
                f"iterations={result['self_training_iterations']}"
            )
        else:
            selection = (
                f"epoch={result['best_epoch']}, "
                f"val_f1={result['best_validation_macro_f1']:.4f}"
            )
        print(
            f"{result['category']:<20} {result['model']:<22} "
            f"{result['accuracy']:>10.4f} "
            f"{result['macro_precision']:>12.4f} "
            f"{result['macro_recall']:>10.4f} "
            f"{result['macro_f1']:>10.4f} "
            f"{selection:>28}"
        )
    print("\n=== WAV2VEC2 TEST F1 BY EMOTION ===")
    print(f"{'Emotion':<12} {'F1-score':>10}")
    for label in models_labels():
        class_metrics = wav2vec_metrics["classification_report"][label]
        print(f"{label:<12} {class_metrics['f1-score']:>10.4f}")


def models_labels() -> list[str]:
    return ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


if __name__ == "__main__":
    main()
