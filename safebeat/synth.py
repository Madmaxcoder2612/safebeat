"""Synthetic single-lead ECG, written from scratch for this project.

Each beat is a sum of five Gaussian bumps (P, Q, R, S, T) placed around the R peak.
Two classes:

* label 0 = regular rhythm (small beat-to-beat variation, sometimes one early beat)
* label 1 = irregular rhythm (large beat-to-beat variation, no P wave, small
  fast ripples instead)

Every number below is an invented teaching value, not a clinical measurement.
Amplitudes are in "normalised units" where a normal R peak is about 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 250  # samples per second
SECONDS = 10  # one window = 10 s

# (amplitude, offset from R peak in s, width in s)
_WAVES = {
    "P": (0.15, -0.20, 0.025),
    "Q": (-0.12, -0.04, 0.010),
    "R": (1.00, 0.00, 0.012),
    "S": (-0.25, 0.04, 0.012),
    "T": (0.30, 0.25, 0.050),
}


@dataclass(frozen=True)
class Subject:
    """One simulated person. Windows from the same subject share these traits."""

    subject_id: int
    hr_bpm: float
    amp: float
    noise_sd: float
    rr_cv_regular: float
    rr_cv_irregular: float


def make_subject(rng: np.random.Generator, subject_id: int) -> Subject:
    return Subject(
        subject_id=subject_id,
        hr_bpm=float(rng.uniform(55, 100)),
        amp=float(rng.uniform(0.7, 1.4)),
        noise_sd=float(rng.uniform(0.01, 0.04)),
        rr_cv_regular=float(rng.uniform(0.02, 0.07)),
        rr_cv_irregular=float(rng.uniform(0.12, 0.30)),
    )


def _beat_times(rng: np.random.Generator, s: Subject, label: int, seconds: float) -> np.ndarray:
    mean_rr = 60.0 / s.hr_bpm
    cv = s.rr_cv_irregular if label == 1 else s.rr_cv_regular
    times = []
    t = float(rng.uniform(0.0, mean_rr))
    while t < seconds + 1.0:
        times.append(t)
        t += max(0.3, mean_rr * (1.0 + cv * float(rng.standard_normal())))
    beats = np.array(times)
    # About a quarter of regular windows get one early beat. This makes the two
    # classes overlap a little, so the classifier cannot be perfect.
    if label == 0 and len(beats) > 4 and rng.random() < 0.25:
        i = int(rng.integers(2, len(beats) - 2))
        beats[i] -= float(rng.uniform(0.2, 0.35)) * mean_rr
    return beats


def render_window(
    rng: np.random.Generator,
    subject: Subject,
    label: int,
    fs: int = FS,
    seconds: float = SECONDS,
) -> np.ndarray:
    """Return one window of ECG as a float array of length fs*seconds."""
    n = int(fs * seconds)
    tt = np.arange(n) / fs
    x = np.zeros(n)
    for tb in _beat_times(rng, subject, label, seconds):
        lo = max(0, int((tb - 0.5) * fs))
        hi = min(n, int((tb + 0.6) * fs))
        if hi <= lo:
            continue
        seg = tt[lo:hi]
        for name, (a, off, w) in _WAVES.items():
            if name == "P" and label == 1:
                continue
            x[lo:hi] += subject.amp * a * np.exp(-0.5 * ((seg - (tb + off)) / w) ** 2)
    if label == 1:
        phase = float(rng.uniform(0, 2 * np.pi))
        x += 0.03 * subject.amp * np.sin(2 * np.pi * 6.0 * tt + phase)
    x += 0.02 * np.sin(2 * np.pi * 0.25 * tt + float(rng.uniform(0, 2 * np.pi)))
    x += subject.noise_sd * rng.standard_normal(n)
    return x


def make_dataset(
    rng: np.random.Generator,
    subject_ids: list[int],
    windows_per_subject: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (signals, labels, subject_ids). Labels alternate so classes are balanced."""
    signals, labels, owners = [], [], []
    for sid in subject_ids:
        subject = make_subject(rng, sid)
        for k in range(windows_per_subject):
            label = k % 2
            signals.append(render_window(rng, subject, label))
            labels.append(label)
            owners.append(sid)
    return np.array(signals), np.array(labels), np.array(owners)
