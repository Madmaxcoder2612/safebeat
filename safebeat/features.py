"""Turn an ECG window into a few beat-timing numbers.

The detector is a plain "band-pass, differentiate, square, smooth, find peaks"
chain written for this project. It is deliberately simple: on damaged signals it
finds wrong peaks, and that is exactly what the safety layer has to catch.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from .synth import FS

FEATURE_NAMES = ("heart_rate", "rr_cv", "rmssd_norm", "frac_big_jumps", "max_rr_ratio", "n_beats")


def detect_r_peaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Return sample indices of detected R peaks."""
    sos = butter(2, [5, 20], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    d = np.diff(y, prepend=y[0])
    env = np.convolve(d**2, np.ones(max(1, int(0.1 * fs))) / max(1, int(0.1 * fs)), mode="same")
    if env.max() <= 0:
        return np.array([], dtype=int)
    thr = 0.3 * np.percentile(env, 99.5)
    peaks, _ = find_peaks(env, height=thr, distance=int(0.25 * fs))
    return peaks


def extract_features(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Six numbers describing the beat timing of one window."""
    peaks = detect_r_peaks(x, fs)
    n = len(peaks)
    if n < 3:
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0, float(n)])
    rr = np.diff(peaks) / fs
    mean_rr = float(np.mean(rr))
    drr = np.abs(np.diff(rr))
    return np.array(
        [
            60.0 / mean_rr,
            float(np.std(rr) / mean_rr),
            float(np.sqrt(np.mean(np.diff(rr) ** 2)) / mean_rr),
            float(np.mean(drr > 0.1 * mean_rr)),
            float(np.max(rr) / np.median(rr)),
            float(n),
        ]
    )


def extract_features_batch(X: np.ndarray, fs: int = FS) -> np.ndarray:
    return np.array([extract_features(x, fs) for x in X])
