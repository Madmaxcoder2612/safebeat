"""The rhythm classifier: plain logistic regression on six beat-timing numbers.

Kept simple on purpose. The project is about knowing when NOT to trust a model,
not about squeezing out the last percent of accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ("regular", "irregular")


class RhythmModel:
    def __init__(self, class_weight: str | None = None) -> None:
        self.pipe = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, class_weight=class_weight)
        )

    def fit(self, features: np.ndarray, y: np.ndarray) -> "RhythmModel":
        self.pipe.fit(features, y)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.pipe.predict(features)
