"""Speech emotion recognition: predict the emotion of audio with the best model
(frozen WavLM-base-plus + ensemble of 5 trained heads, test macro-F1 ~0.7 on the project test split).

  python predict_emotion.py clip.wav                          # one file
  python predict_emotion.py folder --csv result.csv           # every audio file in a folder
  python predict_emotion.py clip.wav --timeline               # emotion over time (3 s windows)
  python predict_emotion.py --record 5                        # record 5 s from the microphone
  python predict_emotion.py folder --batch-normalize          # many clips from ONE source (see below)
  python predict_emotion.py clip.mp3 --json --min-confidence 0.5

Classes: angry, disgust, fear, happy, neutral, sad, surprise.

--batch-normalize: standardises the features with statistics of the whole batch (needs >= 30 windows from the same
recording setup / speaker pool) and uses heads trained that way (models/norm, `python export_model.py --norm`).
In the leave-one-corpus-out experiment this raised macro-F1 on unseen corpora from 0.31 to 0.48. It must NOT be
used for a single clip or for a mix of unrelated sources.

Default heads are trained with noise augmentation (white / pink / babble noise, 0-30 dB SNR): on the noisy test subset macro-F1
is 0.67 at 20 dB white noise and 0.47 at 5 dB (heads trained on clean audio only: 0.51 and 0.10; use --standard for those).
Trained on acted corpora (CREMA-D, RAVDESS, TESS, SAVEE, EMO-DB, UA-SER); accuracy on spontaneous speech, other noise types or
unseen corpora is lower (F1 ~0.3 on unseen corpora without --batch-normalize). See experiments/v2/FINAL_RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ROBUST = MODELS / "noise_robust"
SR = 16_000
WINDOW = SR * 4
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}
MIN_BATCH_WINDOWS = 30


class Head(torch.nn.Module):
    def __init__(self, layers, dim, domains, mode):
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(layers))
        self.norm = torch.nn.LayerNorm(dim)
        self.body = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(dim, 256), torch.nn.GELU(), torch.nn.Dropout(0.3))
        self.emotion = torch.nn.Linear(256, 7)
        self.domain = torch.nn.Linear(256, domains) if mode != "none" else None

    def forward(self, x):
        w = torch.softmax(self.layer_logits, 0)
        return self.emotion(self.body(self.norm((x * w[None, :, None]).sum(1))))


def load_audio(path: Path) -> np.ndarray:
    try:
        wave, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(wave.mean(1))
    except Exception:
        w, sr = torchaudio.load(str(path))
        wave = w.mean(0)
    if sr != SR:
        wave = torchaudio.functional.resample(wave, sr, SR)
    return wave.numpy()


def record(seconds: float, out: Path | None = None) -> Path:
    """Record from the default microphone to a 16 kHz mono wav file."""
    import sounddevice as sd

    print(f"Recording {seconds:g} s ... speak now", file=sys.stderr, flush=True)
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    out = out or Path(tempfile.gettempdir()) / "emotion_recording.wav"
    sf.write(out, audio[:, 0], SR)
    return out


def check_audio(wave: np.ndarray) -> None:
    if len(wave) < SR // 4:
        raise ValueError("audio shorter than 0.25 s")
    if float(np.sqrt(np.mean(wave ** 2))) < 0.0003:
        raise ValueError("audio is almost silent (no speech to analyse)")


def speech_fraction(x: np.ndarray) -> float:
    """Crude energy-based activity estimate: share of 25 ms frames more than 10 dB above the quietest 10% of frames.
    Clean corpus clips: median 0.54, 5th percentile 0.14; pure noise and speech at 0 dB SNR: 0.0."""
    fl, hop = int(0.025 * SR), int(0.010 * SR)
    if len(x) < fl:
        return 1.0
    e = 10 * np.log10(np.array([np.mean(x[i : i + fl] ** 2) for i in range(0, len(x) - fl + 1, hop)]) + 1e-12)
    return float(np.mean(e > np.percentile(e, 10) + 10))


LOW_SPEECH = 0.10


def split_windows(wave: np.ndarray, size: int = WINDOW) -> list[tuple[int, np.ndarray]]:
    starts = list(range(0, max(len(wave), 1), size))
    chunks = [(s, wave[s : s + size]) for s in starts]
    if len(chunks) > 1 and len(chunks[-1][1]) < SR:
        chunks.pop()
    return chunks


class Predictor:
    def __init__(self, device: str = "cpu", heads_dir: Path = MODELS) -> None:
        files = sorted(heads_dir.glob("wavlm_head_seed*.pt"))
        if not files:
            sys.exit(f"No trained heads in {heads_dir}. Run: python export_model.py" + (" --norm" if heads_dir != MODELS else ""))
        self.encoder = AutoModel.from_pretrained("microsoft/wavlm-base-plus").eval().to(device)
        self.device, self.heads = device, []
        for f in files:
            b = torch.load(f, map_location="cpu")
            h = Head(b["layers"], b["dim"], b["domains"], b["mode"]); h.load_state_dict(b["state"]); h.eval()
            self.heads.append((h, b["mu"], b["sd"]))
        self.labels = b["labels"]

    @torch.inference_mode()
    def features(self, wave: np.ndarray) -> torch.Tensor:
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        hs = torch.stack(self.encoder(torch.from_numpy(wave)[None].to(self.device), output_hidden_states=True).hidden_states, 0)[:, 0]
        return torch.cat([hs.mean(1), hs.std(1)], 1).cpu()[None]

    @torch.inference_mode()
    def probs_from_features(self, x: torch.Tensor) -> np.ndarray:
        return torch.stack([torch.softmax(h((x - mu) / sd), 1)[0] for h, mu, sd in self.heads]).mean(0).numpy()

    def chunk_probs(self, wave: np.ndarray) -> np.ndarray:
        """Class probabilities of one <= 4 s window (frozen encoder + heads; overridden by the fine-tuned predictor)."""
        return self.probs_from_features(self.features(wave))

    def _result(self, path, p: np.ndarray, duration: float, windows: int, min_confidence: float, **extra) -> dict:
        order = p.argsort()[::-1]
        if "speech_fraction" in extra and extra["speech_fraction"] < LOW_SPEECH:
            extra["warning"] = "little speech detected or very noisy audio - the prediction is unreliable"
        return {"file": str(path), "emotion": self.labels[order[0]], "confidence": float(p[order[0]]),
                "uncertain": bool(p[order[0]] < min_confidence), "duration_s": round(duration, 2), "windows": windows,
                "probabilities": {self.labels[i]: round(float(p[i]), 4) for i in order}, **extra}

    def predict(self, path: Path, min_confidence: float = 0.0, timeline: bool = False, window_s: float = 3.0) -> dict:
        wave = load_audio(path)
        check_audio(wave)
        sfr = round(speech_fraction(wave), 3)
        if timeline:
            segs = []
            for start, chunk in split_windows(wave, int(window_s * SR)):
                if len(chunk) < SR // 2:
                    continue
                p = self.chunk_probs(chunk)
                segs.append((start / SR, min(len(wave), start + len(chunk)) / SR, p))
            overall = np.mean([s[2] for s in segs], 0)
            r = self._result(path, overall, len(wave) / SR, len(segs), min_confidence, speech_fraction=sfr)
            r["timeline"] = [{"start_s": round(a, 2), "end_s": round(b, 2), "emotion": self.labels[int(p.argmax())],
                              "confidence": round(float(p.max()), 3)} for a, b, p in segs]
            return r
        chunks = split_windows(wave)
        p = np.mean([self.chunk_probs(c) for _, c in chunks], 0)
        return self._result(path, p, len(wave) / SR, len(chunks), min_confidence, speech_fraction=sfr)

    def predict_batch_normalized(self, files: list[Path], min_confidence: float = 0.0) -> list[dict]:
        """Per-source normalisation: statistics of all windows of the batch (use only for one source)."""
        feats, owners, dur, sfrac = [], [], {}, {}
        results = {}
        for f in files:
            try:
                wave = load_audio(f)
                check_audio(wave)
                sfrac[f] = round(speech_fraction(wave), 3)
                dur[f] = len(wave) / SR
                for _, c in split_windows(wave):
                    feats.append(self.features(c)); owners.append(f)
            except Exception as e:
                results[f] = {"file": str(f), "error": str(e)}
        if len(feats) < MIN_BATCH_WINDOWS:
            raise ValueError(f"--batch-normalize needs at least {MIN_BATCH_WINDOWS} windows (4 s each) from one source, got {len(feats)}")
        X = torch.cat(feats)
        X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-5)
        per_file: dict[Path, list[np.ndarray]] = {}
        for x, f in zip(X, owners):
            per_file.setdefault(f, []).append(self.probs_from_features(x[None]))
        for f, ps in per_file.items():
            results[f] = self._result(f, np.mean(ps, 0), dur[f], len(ps), min_confidence, batch_normalized=True, speech_fraction=sfrac[f])
        return [results[f] for f in files]


FINETUNED = MODELS / "finetuned"


class FineTunedPredictor(Predictor):
    """Fully fine-tuned WavLM (ensemble of the checkpoints in models/finetuned/, see export_finetuned.py)."""

    def __init__(self, device: str = "cpu", ckpt_dir: Path = FINETUNED) -> None:
        files = sorted(ckpt_dir.glob("*.pt"))
        if not files:
            sys.exit(f"No fine-tuned checkpoints in {ckpt_dir}. Run: python export_finetuned.py")
        from train_wavlm_v2 import SSLClassifier

        self.device, self.models = device, []
        for f in files:
            ck = torch.load(f, map_location="cpu")
            a, st = ck["args"], ck["state"]
            domains = st["domain.weight"].shape[0] if "domain.weight" in st else 7
            m = SSLClassifier(a["model"], domains, a["mode"], a["mask_prob"]); m.load_state_dict(st)
            self.models.append(m.eval().to(device))
        self.labels = ck["labels"]

    @torch.inference_mode()
    def chunk_probs(self, wave: np.ndarray) -> np.ndarray:
        wave = ((wave - wave.mean()) / (wave.std() + 1e-7)).astype(np.float32)
        x, ln = torch.from_numpy(wave)[None].to(self.device), torch.tensor([len(wave)], device=self.device)
        out = []
        for m in self.models:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                out.append(torch.softmax(m(x, ln)[0].float(), 1).cpu().numpy()[0])
        return np.mean(out, 0)

    def predict_batch_normalized(self, *a, **k):
        raise ValueError("--batch-normalize is only available for the frozen-feature model")


def show(r: dict, name: str, timeline: bool = False) -> None:
    if "error" in r:
        print(f"{name}: ERROR {r['error']}"); return
    top = ", ".join(f"{k} {v:.0%}" for k, v in list(r["probabilities"].items())[:3])
    tag = f"UNCERTAIN, best guess {r['emotion']}" if r["uncertain"] else r["emotion"].upper()
    print(f"{name}: {tag} ({r['confidence']:.0%})   [{top}]")
    if r.get("warning"):
        print(f"    WARNING: {r['warning']} (speech activity {r['speech_fraction']:.0%})")
    if timeline:
        for t in r["timeline"]:
            print(f"    {t['start_s']:6.1f}-{t['end_s']:6.1f} s  {t['emotion']:<9} {t['confidence']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, nargs="?", help="audio file or folder (omit with --record)")
    ap.add_argument("--record", type=float, metavar="SECONDS", help="record from the microphone instead of reading a file")
    ap.add_argument("--save-recording", type=Path, help="where to keep the recording (default: temp file)")
    ap.add_argument("--timeline", action="store_true", help="print the emotion of consecutive 3 s windows")
    ap.add_argument("--batch-normalize", action="store_true", help="per-source feature normalisation for a folder of clips from one source")
    ap.add_argument("--csv", type=Path, help="write all predictions to this CSV")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="mark predictions below this confidence as UNCERTAIN (0.5 answers ~half of the clips at ~87%% accuracy)")
    ap.add_argument("--finetuned", action="store_true",
                    help="use the fully fine-tuned WavLM ensemble (models/finetuned; best on seen corpora and cross-corpus; heavier, GPU recommended)")
    ap.add_argument("--standard", action="store_true", help="use the heads trained on clean audio only (models/) instead of the noise-augmented default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.record:
        a.path = record(a.record, a.save_recording)
    if a.path is None:
        ap.error("give an audio file/folder or use --record SECONDS")
    files = sorted(p for p in a.path.rglob("*") if p.suffix.lower() in AUDIO_EXT) if a.path.is_dir() else [a.path]
    if not files:
        sys.exit(f"No audio files found in {a.path}")
    if a.batch_normalize and a.timeline:
        ap.error("--timeline cannot be combined with --batch-normalize")
    if a.finetuned and a.batch_normalize:
        ap.error("--finetuned cannot be combined with --batch-normalize")
    if a.finetuned:
        pred = FineTunedPredictor(a.device)
    else:
        pred = Predictor(a.device, MODELS / "norm" if a.batch_normalize else (MODELS if a.standard or not ROBUST.exists() else ROBUST))
    if a.batch_normalize:
        try:
            results = pred.predict_batch_normalized(files, a.min_confidence)
        except ValueError as e:
            sys.exit(str(e))
    else:
        results = []
        for f in files:
            try:
                results.append(pred.predict(f, a.min_confidence, a.timeline))
            except Exception as e:
                results.append({"file": str(f), "error": str(e)})
    if a.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    else:
        for f, r in zip(files, results):
            show(r, f.name, a.timeline)
    if a.csv:
        import csv
        ok = [r for r in results if "error" not in r]
        labels = list(ok[0]["probabilities"]) if ok else []
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["file", "emotion", "confidence", "uncertain", *labels])
            for r in ok:
                w.writerow([r["file"], r["emotion"], r["confidence"], r["uncertain"], *[r["probabilities"][k] for k in labels]])
        print("saved", a.csv)


if __name__ == "__main__":
    main()
