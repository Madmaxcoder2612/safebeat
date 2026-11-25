"""Signal-quality gate: decide whether a window is fit to be classified at all.

The gate learns its limits from CLEAN windows only. It never sees a faulty
signal while it is being set up, so passing the fault tests is not memorising.

Assumption: the device delivers amplitude in calibrated units where a normal
R peak is about 1. A real device would need its own calibrated limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, welch

from .features import detect_r_peaks
from .synth import FS

FLAT_STD = 0.005  # a 0.25 s stretch with less spread than this is "flat"


def _beat_deviation(xh: np.ndarray, peaks: np.ndarray, fs: int) -> float:
    """How little the beats look alike: 0 = identical shapes, 1 = no resemblance.

    Cut a short snippet around every detected peak, build the median snippet, and
    average how well each snippet correlates with it. Movement artefact bends the
    beat shape, so this rises even when the beat detector still finds "peaks".
    """
    half = int(0.2 * fs)
    snippets = []
    for k in peaks:
        lo = max(0, k - int(0.05 * fs))
        hi = min(len(xh), k + int(0.05 * fs) + 1)
        c = lo + int(np.argmax(xh[lo:hi]))  # centre on the R peak itself
        if c - half >= 0 and c + half < len(xh):
            snippets.append(xh[c - half : c + half + 1])
    if len(snippets) < 3:
        return 1.0
    S = np.array(snippets)
    template = np.median(S, axis=0)
    t = template - template.mean()
    corr = []
    for s in S:
        d = s - s.mean()
        denom = np.linalg.norm(t) * np.linalg.norm(d)
        corr.append(float(np.dot(t, d) / denom) if denom > 0 else 0.0)
    return float(1.0 - np.mean(corr))


def quality_metrics(x: np.ndarray, fs: int = FS) -> dict[str, float]:
    f, p = welch(x, fs=fs, nperseg=min(len(x), 1024))

    def band(lo: float, hi: float) -> float:
        m = (f >= lo) & (f < hi)
        return float(np.trapezoid(p[m], f[m]))

    core = band(0.67, 40.0) + 1e-12
    w = max(2, int(0.25 * fs))
    m1 = uniform_filter1d(x, w)
    m2 = uniform_filter1d(x * x, w)
    local_std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    peaks = detect_r_peaks(x, fs)
    half = int(0.05 * fs)
    # Measure peak heights on a copy with the slow drift removed, so baseline
    # wander alone does not look like "uneven beats".
    xh = sosfiltfilt(butter(2, 0.7, btype="high", fs=fs, output="sos"), x)
    amps = np.array([np.max(xh[max(0, k - half) : k + half + 1]) for k in peaks]) if len(peaks) else np.array([])
    amp_cv = float(np.std(amps) / (abs(np.mean(amps)) + 1e-9)) if len(amps) >= 2 else 1.0
    return {
        "beat_dev": _beat_deviation(xh, peaks, fs),
        "hf_noise": band(40.0, fs / 2) / core,
        "wander": band(0.0, 0.67) / core,
        "flat_frac": float(np.mean(local_std < FLAT_STD)),
        "clip_frac": float(np.mean(np.abs(x) >= 0.999 * np.max(np.abs(x)))),
        "peak_amp_cv": amp_cv,
        "n_beats": float(len(peaks)),
    }


@dataclass
class QualityGate:
    """Pass or fail a window, with the reasons for a fail."""

    margin: float = 1.5
    limits: dict[str, float] = field(default_factory=dict)
    warn_limits: dict[str, float] = field(default_factory=dict)
    beats_range: tuple[float, float] = (3.0, 30.0)

    # Checks that make the beat timing untrustworthy: a failure means "abstain".
    REJECT_FLOORS = {"hf_noise": 0.002, "flat_frac": 0.02, "peak_amp_cv": 0.05, "beat_dev": 0.05}
    # Checks that make the signal ugly but do not move the beat timing:
    # a failure only adds a warning. (True for THIS beat detector; see README.)
    WARN_FLOORS = {"wander": 0.05, "clip_frac": 0.005}

    def fit(self, X_clean: np.ndarray, fs: int = FS) -> "QualityGate":
        return self.fit_metrics([quality_metrics(x, fs) for x in X_clean])

    def fit_metrics(self, rows: list[dict[str, float]]) -> "QualityGate":
        """Same as fit(), from quality metrics computed earlier (lets many margins share one pass)."""
        col = {k: np.array([r[k] for r in rows]) for k in rows[0]}

        def limit(k: str, floor: float) -> float:
            return max(self.margin * float(np.quantile(col[k], 0.995)), floor)

        self.limits = {k: limit(k, fl) for k, fl in self.REJECT_FLOORS.items()}
        self.warn_limits = {k: limit(k, fl) for k, fl in self.WARN_FLOORS.items()}
        self.beats_range = (
            max(3.0, float(col["n_beats"].min()) - 2.0),
            float(col["n_beats"].max()) + 2.0,
        )
        return self

    def check(self, metrics: dict[str, float]) -> list[str]:
        """Return the list of failed checks (empty list = window is usable)."""
        failed = [k for k, lim in self.limits.items() if metrics[k] > lim]
        lo, hi = self.beats_range
        if not lo <= metrics["n_beats"] <= hi:
            failed.append("n_beats")
        return failed

    def warnings(self, metrics: dict[str, float]) -> list[str]:
        """Return soft warnings: usable, but the signal looks degraded."""
        return [k for k, lim in self.warn_limits.items() if metrics[k] > lim]
