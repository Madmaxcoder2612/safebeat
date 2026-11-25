"""Real-data check on the MIT-BIH Arrhythmia Database (downloaded by the user).

Contains information from the MIT-BIH Arrhythmia Database
(https://physionet.org/content/mitdb/1.0.0/), which is made available under the
ODC Attribution License (https://opendatacommons.org/licenses/by/1-0/).
Full credit and citations: see NOTICE.

What this module does to the data (all changes are listed here on purpose):

1. Keep channel MLII (or the first channel if there is none).
2. Resample from 360 Hz to 250 Hz, the rate the rest of this project uses.
3. Per record, subtract the median and divide by the 99.5th percentile of the
   absolute value, so a normal R peak is about 1. This stands in for a device
   with calibrated gain.
4. Cut into non-overlapping 10 s windows.
5. Label a window from the cardiologists' rhythm annotations, only when ONE
   rhythm covers the whole window:  "(N" (normal sinus rhythm) -> 0 "regular";
   "(AFIB" (atrial fibrillation) -> 1 "irregular". Every other rhythm is skipped.
6. Skip the four records with a pacemaker (102, 104, 107, 217).

Nothing from the database is stored in this repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

from .synth import FS, SECONDS

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mitdb"
PACED = {"102", "104", "107", "217"}
SRC_FS = 360
RHYTHM_LABEL = {"(N": 0, "(AFIB": 1}


def list_records(data_dir: Path = DATA_DIR) -> list[str]:
    recs = sorted(p.stem for p in data_dir.glob("*.hea"))
    return [r for r in recs if r not in PACED]


def load_windows(record: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return (windows, labels) for one record. Windows are 250 Hz, 10 s, label 0 or 1."""
    rec = wfdb.rdrecord(str(data_dir / record))
    ann = wfdb.rdann(str(data_dir / record), "atr")
    ch = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    x = rec.p_signal[:, ch].astype(float)
    # rhythm label per source sample, from the rhythm-change annotations ('+')
    rhythm = np.array([""] * len(x), dtype=object)
    changes = [(s, a) for s, sym, a in zip(ann.sample, ann.symbol, ann.aux_note) if sym == "+"]
    for k, (s, a) in enumerate(changes):
        end = changes[k + 1][0] if k + 1 < len(changes) else len(x)
        rhythm[s:end] = a.strip("\x00")
    x = resample_poly(x - np.median(x), 25, 36)
    scale = np.percentile(np.abs(x), 99.5)
    x = x / scale if scale > 0 else x
    n = FS * SECONDS
    src_n = SRC_FS * SECONDS
    windows, labels = [], []
    for k in range(len(x) // n):
        seg = rhythm[k * src_n : (k + 1) * src_n]
        if len(seg) == 0 or not (seg == seg[0]).all() or seg[0] not in RHYTHM_LABEL:
            continue
        windows.append(x[k * n : (k + 1) * n])
        labels.append(RHYTHM_LABEL[seg[0]])
    if not windows:
        return np.empty((0, n)), np.empty(0, dtype=int)
    return np.array(windows), np.array(labels)
