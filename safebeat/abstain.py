"""SafeBeat: quality gate plus conformal "I am not sure" rule.

Split conformal prediction in plain words: on a calibration set the model has
never trained on, record how wrong it is on the true class (score = 1 - p_true).
Take a high quantile q of those scores. For a new window, keep every class whose
score would be at most q. If exactly one class is left, answer. Otherwise
(both classes left, or none) say "can't tell".

Honest limit: the coverage promise (about 1 - alpha) holds only for windows that
look like the calibration data, meaning CLEAN signals. Damaged signals break that
assumption, which is exactly why the quality gate runs first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import extract_features
from .model import RhythmModel
from .quality import QualityGate, quality_metrics


@dataclass
class Decision:
    label: int | None  # 0 or 1, or None if SafeBeat abstains
    reason: str  # "ok", "uncertain", or "bad_signal:<checks>"
    p_irregular: float
    warnings: list[str] = field(default_factory=list)  # soft quality warnings, e.g. "wander"


class SafeBeat:
    def __init__(self, model: RhythmModel, gate: QualityGate, alpha: float = 0.1) -> None:
        self.model = model
        self.gate = gate
        self.alpha = alpha
        self.q: float | None = None

    def calibrate(self, feats_cal: np.ndarray, y_cal: np.ndarray) -> "SafeBeat":
        p = self.model.predict_proba(feats_cal)
        scores = 1.0 - p[np.arange(len(y_cal)), y_cal]
        n = len(scores)
        k = min(n, int(np.ceil((n + 1) * (1.0 - self.alpha))))
        self.q = float(np.sort(scores)[k - 1])
        return self

    def decide(self, x: np.ndarray) -> Decision:
        return self.decide_from(quality_metrics(x), extract_features(x))

    def decide_from(self, metrics: dict[str, float], feats: np.ndarray) -> Decision:
        """Same as decide(), but with the numbers already computed (used for fast sweeps)."""
        if self.q is None:
            raise RuntimeError("call calibrate() first")
        failed = self.gate.check(metrics)
        warn = self.gate.warnings(metrics)
        probs = self.model.predict_proba(feats[None, :])[0]
        if failed:
            return Decision(None, "bad_signal:" + "+".join(failed), float(probs[1]), warn)
        keep = [k for k in (0, 1) if 1.0 - probs[k] <= self.q]
        if len(keep) == 1:
            return Decision(keep[0], "ok", float(probs[1]), warn)
        return Decision(None, "uncertain", float(probs[1]), warn)
