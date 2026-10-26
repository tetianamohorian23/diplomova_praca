"""Noise generators: white, pink (1/f) and babble (sum of other speech clips), mixed at a target SNR."""
import numpy as np


def white(n, rng):
    return rng.standard_normal(n).astype(np.float32)


def pink(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec)); f[0] = 1
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return (x / (x.std() + 1e-9)).astype(np.float32)


def babble(n, rng, pool, k=4):
    """pool: callable returning a random speech waveform (float32)."""
    out = np.zeros(n, np.float32)
    for _ in range(k):
        w = pool()
        w = np.tile(w, int(np.ceil(n / max(len(w), 1)) + 1))
        s = rng.integers(0, max(1, len(w) - n))
        seg = w[s : s + n]
        out += seg / (seg.std() + 1e-9)
    return out / (out.std() + 1e-9)


def mix(wave, noise, snr_db):
    scale = np.sqrt(np.mean(wave ** 2) / 10 ** (snr_db / 10) / (np.mean(noise ** 2) + 1e-12))
    return wave + noise * scale
