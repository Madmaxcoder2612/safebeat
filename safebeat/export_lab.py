"""Precompute everything the browser lab shows and write it to lab/data.js.

Run from the repository root:  python -m safebeat.export_lab

The lab page does no machine learning of its own. This script runs the real
Python code on a few synthetic example windows, damages each one with every fault
at 11 severities, records the verdicts of the naive model and of SafeBeat (at 5
gate strictness levels), and stores the signals (halved to 125 Hz for drawing).
The page just replays those results. Nothing is sent anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .abstain import SafeBeat
from .evaluate import MARGINS, build, sweep
from .faults import FAULTS, apply_fault
from .features import extract_features
from .model import CLASSES
from .quality import QualityGate, quality_metrics
from .synth import FS

LAB_SEVERITIES = tuple(round(0.1 * i, 1) for i in range(11))
OUT = Path(__file__).resolve().parent.parent / "lab" / "data.js"


def _pick_windows(model, safe, Xte, yte):
    """Two regular and two irregular windows from different people that start out correct."""
    picks = {0: [], 1: []}
    seen_people = set()
    for i, (x, y) in enumerate(zip(Xte, yte)):
        person = i // 40  # build() uses 40 windows per subject
        if person in seen_people or len(picks[int(y)]) >= 2:
            continue
        d = safe.decide(x)
        naive = int(model.predict(extract_features(x)[None, :])[0])
        if d.label == int(y) and naive == int(y):
            picks[int(y)].append(i)
            seen_people.add(person)
    return picks[0] + picks[1]


def main(seed: int = 7) -> None:
    model, safe, Xtr, Xte, yte = build(seed)
    variants = {m: SafeBeat(model, QualityGate(margin=m).fit(Xtr), alpha=0.1) for m in MARGINS}
    for v in variants.values():
        v.q = safe.q

    windows, cases = [], {}
    for n, idx in enumerate(_pick_windows(model, safe, Xte, yte)):
        wid = f"w{n}"
        windows.append({"id": wid, "truth": CLASSES[int(yte[idx])], "name": f"Person {chr(65 + n)}"})
        cases[wid] = {}
        for fault in FAULTS:
            cases[wid][fault] = []
            for sev in LAB_SEVERITIES:
                rng = np.random.default_rng([seed, n, FAULTS.index(fault), int(sev * 10)])
                x = apply_fault(Xte[idx], fault, sev, rng)
                metrics, feats = quality_metrics(x), extract_features(x)
                p = model.predict_proba(feats[None, :])[0]
                safe_out = {}
                for m, v in variants.items():
                    d = v.decide_from(metrics, feats)
                    safe_out[str(m)] = {"label": d.label, "reason": d.reason, "warn": d.warnings}
                cases[wid][fault].append(
                    {
                        "sig": [int(round(1000 * s)) for s in x[::2]],
                        "naive": {"label": int(p[1] > 0.5), "p_irregular": round(float(p[1]), 3)},
                        "safe": safe_out,
                    }
                )

    faults_tab, margin_tab = sweep(seed, severities=LAB_SEVERITIES, margin_severities=(0.0, 0.3, 0.5))
    data = {
        "meta": {
            "fs": FS // 2,
            "seconds": 10,
            "severities": list(LAB_SEVERITIES),
            "margins": list(MARGINS),
            "faults": list(FAULTS),
            "classes": list(CLASSES),
            "note": "All signals are synthetic. Results precomputed by the Python code in this repository.",
        },
        "windows": windows,
        "cases": cases,
        "sweep": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in faults_tab],
        "margin": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in margin_tab],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.SAFEBEAT_LAB = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), windows: {[w['name'] + '=' + w['truth'] for w in windows]}")


if __name__ == "__main__":
    main()
