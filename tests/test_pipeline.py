"""Sanity tests: python tests/test_pipeline.py   (or: pytest tests)"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_split_is_speaker_disjoint_and_sized():
    from common_splits import load_splits
    d = load_splits()
    assert len(d) == 13406 and (d.experiment_split == "test").sum() == 3511
    for _, g in d.groupby("dataset"):
        sets = [set(g[g.experiment_split == s].speaker_id) for s in ("train", "validation", "test")]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_labels_and_files():
    from common_splits import LABELS, load_splits
    d = load_splits()
    assert set(d.emotion) == set(LABELS)
    assert all((ROOT / p).exists() for p in d.audio_path.sample(50, random_state=0))


def test_predictor_outputs_valid_distribution():
    from predict_emotion import Predictor, SR
    import soundfile as sf, tempfile
    wave = (0.1 * np.random.default_rng(0).standard_normal(SR * 2)).astype("float32")
    with tempfile.TemporaryDirectory() as t:
        sf.write(Path(t) / "noise.wav", wave, SR)
        pred = Predictor()
        r = pred.predict(Path(t) / "noise.wav", 0.5)
    probs = np.array(list(r["probabilities"].values()))
    assert abs(probs.sum() - 1) < 1e-2 and r["emotion"] in pred.labels


def test_known_clip_is_angry():
    from predict_emotion import Predictor
    clip = ROOT / "CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav"
    assert Predictor().predict(clip)["emotion"] == "angry"


def test_silence_is_refused_and_timeline_works():
    from predict_emotion import Predictor, check_audio, SR
    try:
        check_audio(np.zeros(SR, dtype="float32")); raise AssertionError("silence accepted")
    except ValueError:
        pass
    r = Predictor().predict(ROOT / "CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav", 0.5, timeline=True)
    assert r["timeline"] and r["timeline"][0]["start_s"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
