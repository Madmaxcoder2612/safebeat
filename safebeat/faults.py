"""Fault injection: damage a clean window the way real sensors get damaged.

Every fault takes a severity from 0 (no damage, returns the input unchanged)
to 1 (worst case used in this project). The strengths are chosen for teaching,
not measured on real devices.
"""

from __future__ import annotations

import numpy as np

from .synth import FS

FAULTS = ("noise", "wander", "dropout", "electrode_off", "clipping", "motion")


def _random_segments(rng: np.random.Generator, n: int, fraction: float, lo_s: float, hi_s: float, fs: int):
    """Yield (start, stop) sample ranges that together cover about `fraction` of n."""
    covered = 0
    target = int(fraction * n)
    while covered < target:
        length = int(rng.uniform(lo_s, hi_s) * fs)
        start = int(rng.integers(0, max(1, n - length)))
        yield start, min(n, start + length)
        covered += length


def apply_fault(
    x: np.ndarray, kind: str, severity: float, rng: np.random.Generator, fs: int = FS
) -> np.ndarray:
    """Return a damaged copy of x. Severity 0 returns an unchanged copy."""
    if kind not in FAULTS:
        raise ValueError(f"unknown fault {kind!r}, choose from {FAULTS}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be between 0 and 1")
    y = x.copy()
    if severity == 0.0:
        return y
    n = len(y)
    tt = np.arange(n) / fs

    if kind == "noise":  # broadband interference, e.g. a poor contact or mains pickup
        y += severity * 0.8 * rng.standard_normal(n)
    elif kind == "wander":  # slow baseline drift from breathing or sweat
        y += severity * 2.0 * np.sin(2 * np.pi * 0.3 * tt + float(rng.uniform(0, 6.28)))
    elif kind == "dropout":  # the sensor loses samples and reports zeros
        for a, b in _random_segments(rng, n, severity * 0.5, 0.3, 1.0, fs):
            y[a:b] = 0.0
    elif kind == "electrode_off":  # the electrode lifts: a flat line with tiny noise
        length = int(severity * 0.8 * n)
        start = int(rng.integers(0, n - length + 1))
        y[start : start + length] = y[start] + 0.002 * rng.standard_normal(length)
    elif kind == "clipping":  # the amplifier saturates
        c = (1.0 - 0.85 * severity) * float(np.max(np.abs(y)))
        y = np.clip(y, -c, c)
    elif kind == "motion":  # bursts of large slow movement artefact
        for a, b in _random_segments(rng, n, severity * 0.6, 0.5, 1.5, fs):
            f = float(rng.uniform(2.0, 6.0))
            y[a:b] += severity * 1.5 * np.sin(2 * np.pi * f * tt[a:b])
    return y
