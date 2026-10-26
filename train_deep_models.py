from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForSequenceClassification, WavLMModel


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "combined_dataset_normalized.csv"
OUTPUT = ROOT / "experiments"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SAMPLE_RATE = 16_000
MAX_SECONDS = 4
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest() -> pd.DataFrame:
    data = pd.read_csv(MANIFEST)
    data = data[data["emotion"].isin(LABELS)].copy()
    data["label_id"] = data["emotion"].map(LABEL_TO_ID)
    return data


def load_wave(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate in {path}: {sample_rate}")
    waveform = torch.from_numpy(np.asarray(audio))
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    waveform = waveform[:MAX_SAMPLES]
    if waveform.numel() < MAX_SAMPLES:
        waveform = nn.functional.pad(waveform, (0, MAX_SAMPLES - waveform.numel()))
    return waveform


class AudioDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.data.iloc[index]
        return load_wave(ROOT / row["audio_path"]), int(row["label_id"])


class MultiTaskAudioDataset(AudioDataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        waveform, label = super().__getitem__(index)
        dataset_id = int(self.data.iloc[index]["dataset_id"])
        return waveform, label, dataset_id


class WavLMEmotionDomainModel(nn.Module):
    def __init__(self, model_name: str, emotion_count: int, domain_count: int) -> None:
        super().__init__()
        self.encoder = WavLMModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.attention = nn.Linear(hidden_size, 1)
        self.emotion_head = nn.Linear(hidden_size, emotion_count)
        self.domain_head = nn.Linear(hidden_size, domain_count)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.encoder(input_values=waveform).last_hidden_state
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        hidden = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        hidden = self.dropout(hidden)
        return self.emotion_head(hidden), self.domain_head(hidden)


class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1)


class MelCNN(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64,
            power=2.0,
        )
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=18)
        self.frequency_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.GroupNorm(16, 192),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(192, class_count),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel_spectrogram(waveform)
        spec = torch.log(spec.clamp_min(1e-5))
        spec = (spec - spec.mean(dim=-1, keepdim=True)) / (
            spec.std(dim=-1, keepdim=True) + 1e-5
        )
        if self.training:
            spec = self.time_mask(spec)
            spec = self.frequency_mask(spec)
        delta = torchaudio.functional.compute_deltas(spec)
        delta_delta = torchaudio.functional.compute_deltas(delta)
        spec = torch.stack((spec, delta, delta_delta), dim=1)
        return self.classifier(self.encoder(spec))


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64,
            power=2.0,
        )
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.lstm = nn.LSTM(
            input_size=64 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, class_count),
        )

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel_spectrogram(waveform)
        spec = torch.log(spec.clamp_min(1e-5))
        spec = (spec - spec.mean(dim=-1, keepdim=True)) / (
            spec.std(dim=-1, keepdim=True) + 1e-5
        )
        delta = torchaudio.functional.compute_deltas(spec)
        delta_delta = torchaudio.functional.compute_deltas(delta)
        spec = torch.stack((spec, delta, delta_delta), dim=1)
        features = self.frontend(spec)
        return features.permute(0, 3, 1, 2).flatten(2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(waveform)
        return self.classifier(pooled)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(self._features(waveform))
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1)


class MultiTaskCNNBiLSTMAttention(CNNBiLSTMAttention):
    def __init__(self, emotion_count: int, dataset_count: int) -> None:
        super().__init__(emotion_count)
        self.dataset_classifier = nn.Linear(256, dataset_count)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.encode(waveform)
        return self.classifier(pooled), self.dataset_classifier(pooled)


class ContrastiveEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = MelCNN(7).encoder
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64,
            power=2.0,
        )
        self.projector = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel_spectrogram(waveform)
        spec = torch.log(spec.clamp_min(1e-5))
        spec = (spec - spec.mean(dim=-1, keepdim=True)) / (
            spec.std(dim=-1, keepdim=True) + 1e-5
        )
        delta = torchaudio.functional.compute_deltas(spec)
        delta_delta = torchaudio.functional.compute_deltas(delta)
        spec = torch.stack((spec, delta, delta_delta), dim=1)
        embedding = self.encoder(spec).flatten(1)
        return nn.functional.normalize(self.projector(embedding), dim=1)


def augment_waveforms(waveforms: torch.Tensor) -> torch.Tensor:
    augmented = waveforms.clone()
    noise_scale = torch.rand(len(waveforms), 1, device=waveforms.device) * 0.01
    augmented += noise_scale * torch.randn_like(augmented)
    gain = 0.8 + 0.4 * torch.rand(len(waveforms), 1, device=waveforms.device)
    augmented *= gain
    shifts = torch.randint(-800, 801, (len(waveforms),), device=waveforms.device)
    for index, shift in enumerate(shifts.tolist()):
        augmented[index] = torch.roll(augmented[index], shift)
    return augmented.clamp(-1.0, 1.0)


def contrastive_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    representations = torch.cat((first, second), dim=0)
    similarities = representations @ representations.T / temperature
    batch_size = first.shape[0]
    mask = torch.eye(2 * batch_size, device=first.device, dtype=torch.bool)
    similarities = similarities.masked_fill(mask, float("-inf"))
    targets = torch.arange(batch_size, device=first.device)
    targets = torch.cat((targets + batch_size, targets))
    return nn.functional.cross_entropy(similarities, targets)


def pretrain_cnn_self_supervised(data, device, epochs: int, batch_size: int) -> dict:
    loader = DataLoader(AudioDataset(data), batch_size=batch_size, shuffle=True)
    model = ContrastiveEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for waveforms, _ in loader:
            waveforms = waveforms.to(device)
            first = model(augment_waveforms(waveforms))
            second = model(augment_waveforms(waveforms))
            loss = contrastive_loss(first, second)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(waveforms)
        value = total / len(loader.dataset)
        history.append({"epoch": epoch, "contrastive_loss": value})
        print(f"cnn contrastive epoch={epoch} loss={value:.4f}")
    return {"encoder": model.encoder.state_dict(), "history": history}


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    total = len(y_true)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
        "per_class_accuracy": {
            label: float(
                (matrix[index, index] + total - matrix[index, :].sum()
                 - matrix[:, index].sum() + matrix[index, index]) / total
            )
            for index, label in enumerate(LABELS)
        },
    }


def run_epoch(
    model, loader, criterion, optimizer, device, training: bool, stage: str,
    attention_pool: nn.Module | None = None,
) -> tuple[float, dict]:
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    started = time.perf_counter()
    total_batches = len(loader)
    report_every = max(1, total_batches // 20)
    for batch_index, (waveforms, targets) in enumerate(loader, start=1):
        waveforms, targets = waveforms.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            if attention_pool is None:
                output = model(waveforms)
                logits = output.logits if hasattr(output, "logits") else output
            else:
                sequence = model.wav2vec2(input_values=waveforms).last_hidden_state
                pooled = attention_pool(sequence)
                logits = model.classifier(model.projector(pooled))
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item() * len(targets)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        if batch_index == 1 or batch_index % report_every == 0 or batch_index == total_batches:
            elapsed = time.perf_counter() - started
            eta = (elapsed / batch_index) * (total_batches - batch_index)
            print(
                f"\r{stage}: {batch_index}/{total_batches} "
                f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                end="",
                flush=True,
            )
    print()
    result = metrics(y_true, y_pred)
    result["loss"] = total_loss / len(loader.dataset)
    return result["loss"], result


def train_cnn(args, train_data, validation_data, test_data, device) -> dict:
    train_loader = DataLoader(AudioDataset(train_data), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(AudioDataset(validation_data), batch_size=args.batch_size)
    test_loader = DataLoader(AudioDataset(test_data), batch_size=args.batch_size)
    pretrained = pretrain_cnn_self_supervised(
        train_data, device, args.cnn_ssl_epochs, args.batch_size
    )
    model = MelCNN(len(LABELS)).to(device)
    model.encoder.load_state_dict(pretrained["encoder"])
    counts = np.bincount(train_data["label_id"], minlength=len(LABELS))
    weights = torch.tensor(
        np.sqrt(len(train_data) / (len(LABELS) * counts)), dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )
    history = []
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        _, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, True, f"cnn epoch {epoch} train"
        )
        _, validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer,
            device,
            False,
            f"cnn epoch {epoch} validation",
        )
        score = validation_metrics["macro_f1"]
        scheduler.step(score)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"cnn epoch={epoch}: train_macro_f1={train_metrics['macro_f1']:.4f}, "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}, "
            f"validation_loss={validation_metrics['loss']:.4f}"
            f", lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(f"cnn early stopping at epoch {epoch}")
                break
    if best_state is None:
        raise RuntimeError("CNN did not produce a checkpoint")
    model.load_state_dict(best_state)
    _, test_metrics = run_epoch(
        model, test_loader, criterion, optimizer, device, False, "cnn final test"
    )
    torch.save(model.state_dict(), OUTPUT / "cnn_mel.pt")
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_validation_macro_f1"] = best_score
    test_metrics["stopped_epoch"] = history[-1]["epoch"]
    test_metrics["cnn_ssl_epochs"] = len(pretrained["history"])
    test_metrics["cnn_ssl_method"] = "contrastive_learning"
    test_metrics["cnn_classifier_max_epochs"] = args.epochs
    (OUTPUT / "cnn_mel.json").write_text(
        json.dumps(
            {
                "category": "self-supervised",
                "pretraining": pretrained["history"],
                "history": history,
                "test": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return test_metrics


def run_multitask_epoch(
    model, loader, emotion_criterion, dataset_criterion, optimizer, device,
    training: bool, stage: str, dataset_loss_weight: float,
) -> tuple[float, dict]:
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    started = time.perf_counter()
    total_batches = len(loader)
    report_every = max(1, total_batches // 20)
    for batch_index, (waveforms, emotion_targets, dataset_targets) in enumerate(
        loader, start=1
    ):
        waveforms = waveforms.to(device)
        emotion_targets = emotion_targets.to(device)
        dataset_targets = dataset_targets.to(device)
        with torch.set_grad_enabled(training):
            emotion_logits, dataset_logits = model(waveforms)
            emotion_loss = emotion_criterion(emotion_logits, emotion_targets)
            dataset_loss = dataset_criterion(dataset_logits, dataset_targets)
            loss = emotion_loss + dataset_loss_weight * dataset_loss
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item() * len(emotion_targets)
        y_true.extend(emotion_targets.cpu().tolist())
        y_pred.extend(emotion_logits.argmax(dim=1).cpu().tolist())
        if batch_index == 1 or batch_index % report_every == 0 or batch_index == total_batches:
            elapsed = time.perf_counter() - started
            eta = (elapsed / batch_index) * (total_batches - batch_index)
            print(
                f"\r{stage}: {batch_index}/{total_batches} "
                f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                end="", flush=True,
            )
    print()
    result = metrics(y_true, y_pred)
    result["loss"] = total_loss / len(loader.dataset)
    return result["loss"], result


def train_cnn_bilstm(args, train_data, validation_data, test_data, device) -> dict:
    train_loader = DataLoader(
        MultiTaskAudioDataset(train_data), batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        MultiTaskAudioDataset(validation_data), batch_size=args.batch_size
    )
    test_loader = DataLoader(
        MultiTaskAudioDataset(test_data), batch_size=args.batch_size
    )
    dataset_count = int(
        pd.concat([train_data, validation_data, test_data])["dataset_id"].max()
    ) + 1
    model = MultiTaskCNNBiLSTMAttention(len(LABELS), dataset_count).to(device)
    emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    dataset_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lstm_learning_rate, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )
    history = []
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.lstm_epochs + 1):
        _, train_metrics = run_multitask_epoch(
            model, train_loader, emotion_criterion, dataset_criterion, optimizer,
            device, True, f"cnn-bilstm-multitask epoch {epoch} train", 0.2,
        )
        _, validation_metrics = run_multitask_epoch(
            model, validation_loader, emotion_criterion, dataset_criterion, optimizer,
            device, False, f"cnn-bilstm-multitask epoch {epoch} validation", 0.2,
        )
        score = validation_metrics["macro_f1"]
        scheduler.step(score)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"cnn-bilstm-multitask epoch={epoch}: train_macro_f1={train_metrics['macro_f1']:.4f}, "
            f"validation_macro_f1={score:.4f}, validation_loss={validation_metrics['loss']:.4f}"
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                print(f"cnn-bilstm-multitask early stopping at epoch {epoch}")
                break
    if best_state is None:
        raise RuntimeError("CNN-BiLSTM did not produce a checkpoint")
    model.load_state_dict(best_state)
    _, test_metrics = run_multitask_epoch(
        model, test_loader, emotion_criterion, dataset_criterion, optimizer,
        device, False, "cnn-bilstm-multitask final test", 0.2,
    )
    test_metrics.update(
        {
            "category": "supervised-multi-task",
            "method": "emotion_plus_dataset_classification",
            "dataset_loss_weight": 0.2,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_score,
            "stopped_epoch": history[-1]["epoch"],
            "max_epochs": args.lstm_epochs,
        }
    )
    torch.save(model.state_dict(), OUTPUT / "cnn_bilstm.pt")
    (OUTPUT / "cnn_bilstm.json").write_text(
        json.dumps(
            {
                "category": "supervised-multi-task",
                "method": "emotion_plus_dataset_classification",
                "dataset_loss_weight": 0.2,
                "history": history,
                "test": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return test_metrics


def train_wav2vec(args, train_data, validation_data, test_data, device) -> dict:
    train_loader = DataLoader(AudioDataset(train_data), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(AudioDataset(validation_data), batch_size=args.batch_size)
    test_loader = DataLoader(AudioDataset(test_data), batch_size=args.batch_size)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.wav2vec_model,
        num_labels=len(LABELS),
        ignore_mismatched_sizes=True,
    ).to(device)
    attention_pool = AttentionPool(model.config.hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    encoder_parameters = list(model.wav2vec2.parameters())
    head_parameters = (
        list(model.projector.parameters())
        + list(model.classifier.parameters())
        + list(attention_pool.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.wav2vec_learning_rate},
            {"params": head_parameters, "lr": args.wav2vec_head_learning_rate},
        ],
        weight_decay=0.01,
    )
    if args.wav2vec_freeze_epochs:
        for parameter in encoder_parameters:
            parameter.requires_grad = False
    history = []
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.wav2vec_freeze_epochs + 1:
            for parameter in encoder_parameters:
                parameter.requires_grad = True
            print("wav2vec2: encoder unfrozen")
        _, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
            f"wav2vec2 epoch {epoch} train",
            attention_pool,
        )
        _, validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer,
            device,
            False,
            f"wav2vec2 epoch {epoch} validation",
            attention_pool,
        )
        score = validation_metrics["macro_f1"]
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"wav2vec2 epoch={epoch}: train_macro_f1={train_metrics['macro_f1']:.4f}, "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}, "
            f"validation_loss={validation_metrics['loss']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(attention_pool.state_dict()),
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(f"wav2vec2 early stopping at epoch {epoch}")
                break
    if best_state is None:
        raise RuntimeError("Wav2Vec2 did not produce a checkpoint")
    model.load_state_dict(best_state[0])
    attention_pool.load_state_dict(best_state[1])
    _, test_metrics = run_epoch(
        model, test_loader, criterion, optimizer, device, False, "wav2vec2 final test",
        attention_pool,
    )
    model.save_pretrained(OUTPUT / "wav2vec2")
    torch.save(attention_pool.state_dict(), OUTPUT / "wav2vec2_attention.pt")
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_validation_macro_f1"] = best_score
    test_metrics["stopped_epoch"] = history[-1]["epoch"]
    test_metrics["max_epochs"] = args.epochs
    test_metrics["freeze_epochs"] = args.wav2vec_freeze_epochs
    (OUTPUT / "wav2vec2.json").write_text(
        json.dumps(
            {"category": "transfer-learning", "history": history, "test": test_metrics},
            indent=2,
        ),
        encoding="utf-8",
    )
    return test_metrics


def train_wavlm_multitask(args, train_data, validation_data, test_data, device) -> dict:
    train_loader = DataLoader(
        MultiTaskAudioDataset(train_data), batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        MultiTaskAudioDataset(validation_data), batch_size=args.batch_size
    )
    test_loader = DataLoader(
        MultiTaskAudioDataset(test_data), batch_size=args.batch_size
    )
    model = WavLMEmotionDomainModel(
        args.wavlm_model, len(LABELS), int(train_data["dataset_id"].max()) + 1
    ).to(device)
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.emotion_head.parameters()) + list(model.domain_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.wavlm_learning_rate},
            {"params": head_parameters, "lr": args.wavlm_head_learning_rate},
        ],
        weight_decay=0.01,
    )
    if args.wavlm_freeze_epochs:
        for parameter in encoder_parameters:
            parameter.requires_grad = False
    emotion_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()
    history = []
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale = 0

    def run(loader, training: bool, stage: str) -> dict:
        model.train(training)
        true, pred = [], []
        total = 0.0
        started = time.perf_counter()
        total_batches = len(loader)
        report_every = max(1, total_batches // 20)
        for batch_index, (waves, emotions, domains) in enumerate(loader, start=1):
            waves, emotions, domains = waves.to(device), emotions.to(device), domains.to(device)
            with torch.set_grad_enabled(training):
                emotion_logits, domain_logits = model(waves)
                loss = emotion_loss(emotion_logits, emotions) + args.wavlm_domain_loss_weight * domain_loss(
                    domain_logits, domains
                )
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total += loss.item() * len(emotions)
            true.extend(emotions.cpu().tolist())
            pred.extend(emotion_logits.argmax(1).cpu().tolist())
            if (
                batch_index == 1
                or batch_index % report_every == 0
                or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - started
                eta = (elapsed / batch_index) * (total_batches - batch_index)
                print(
                    f"\r{stage}: {batch_index}/{total_batches} "
                    f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                    end="",
                    flush=True,
                )
        print()
        result = metrics(true, pred)
        result["loss"] = total / len(loader.dataset)
        print(f"{stage}: emotion_macro_f1={result['macro_f1']:.4f}")
        return result

    for epoch in range(1, args.wavlm_epochs + 1):
        if epoch == args.wavlm_freeze_epochs + 1:
            for parameter in encoder_parameters:
                parameter.requires_grad = True
            print("wavlm: encoder unfrozen")
        train_metrics = run(train_loader, True, f"wavlm epoch {epoch} train")
        validation_metrics = run(validation_loader, False, f"wavlm epoch {epoch} validation")
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        score = validation_metrics["macro_f1"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("WavLM did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = run(test_loader, False, "wavlm final test")
    test_metrics.update(
        {
            "category": "transfer-learning-multitask",
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_score,
            "max_epochs": args.wavlm_epochs,
            "auxiliary_task": "dataset_classification",
            "domain_loss_weight": args.wavlm_domain_loss_weight,
        }
    )
    tag = getattr(args, "output_tag", "")
    torch.save(model.state_dict(), OUTPUT / f"wavlm_multitask{tag}.pt")
    (OUTPUT / f"wavlm_multitask{tag}.json").write_text(
        json.dumps({"category": test_metrics["category"], "history": history, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("cnn", "wav2vec2"), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--wav2vec-learning-rate", type=float, default=1e-5)
    parser.add_argument("--cnn-ssl-epochs", type=int, default=5)
    parser.add_argument("--wav2vec-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--wav2vec-freeze-epochs", type=int, default=1)
    parser.add_argument("--wavlm-epochs", type=int, default=5)
    parser.add_argument("--wavlm-learning-rate", type=float, default=1e-5)
    parser.add_argument("--wavlm-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--wavlm-freeze-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--wav2vec-model", default="facebook/wav2vec2-base")
    parser.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    data = load_manifest()
    train_data = data[data["split"] == "train"]
    test_data = data[data["split"] == "test"]
    if args.limit:
        train_data = train_data.groupby("label_id", group_keys=False).head(args.limit)
        test_data = test_data.groupby("label_id", group_keys=False).head(max(1, args.limit // 4))
    OUTPUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model} device={device} train={len(train_data)} test={len(test_data)}")
    if args.model == "cnn":
        train_cnn(args, train_data, test_data, device)
    else:
        train_wav2vec(args, train_data, test_data, device)


if __name__ == "__main__":
    main()
