"""Train, calibrate and test on synthetic data, then sweep the faults.

Run from the repository root:  python -m safebeat.evaluate

Subjects are split BEFORE windows are made: train, calibration and test subjects
never overlap, so test results are on people the model has never seen.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .abstain import SafeBeat
from .faults import FAULTS, apply_fault
from .features import extract_features_batch
from .model import RhythmModel
from .quality import QualityGate, quality_metrics
from .synth import make_dataset

SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)
MARGINS = (1.5, 3.0, 6.0, 12.0, 24.0)
RESULTS = Path(__file__).resolve().parent.parent / "results"


def build(seed: int = 7, n_train: int = 40, n_cal: int = 20, n_test: int = 30, wps: int = 40):
    """Return (model, safebeat, train signals, test signals, test labels)."""
    rng = np.random.default_rng(seed)
    ids = list(range(n_train + n_cal + n_test))
    tr, ca, te = ids[:n_train], ids[n_train : n_train + n_cal], ids[n_train + n_cal :]
    assert not (set(tr) & set(ca)) and not (set(tr) & set(te)) and not (set(ca) & set(te))
    Xtr, ytr, _ = make_dataset(rng, tr, wps)
    Xca, yca, _ = make_dataset(rng, ca, wps)
    Xte, yte, _ = make_dataset(rng, te, wps)
    model = RhythmModel().fit(extract_features_batch(Xtr), ytr)
    safe = SafeBeat(model, QualityGate().fit(Xtr), alpha=0.1).calibrate(extract_features_batch(Xca), yca)
    return model, safe, Xtr, Xte, yte


def precompute(X: np.ndarray):
    """Quality metrics and beat features for every window (the slow part, done once)."""
    return [quality_metrics(x) for x in X], extract_features_batch(X)


def score(model, safe: SafeBeat, y: np.ndarray, pre) -> dict[str, float]:
    metrics, feats = pre
    naive_pred = model.predict(feats)
    decisions = [safe.decide_from(m, f) for m, f in zip(metrics, feats)]
    answered = np.array([d.label is not None for d in decisions])
    labels = np.array([d.label if d.label is not None else -1 for d in decisions])
    return {
        "naive_wrong_rate": float(np.mean(naive_pred != y)),
        "safebeat_answer_rate": float(np.mean(answered)),
        "safebeat_wrong_rate": float(np.mean(answered & (labels != y))),
        "safebeat_acc_on_answered": float(np.mean(labels[answered] == y[answered])) if answered.any() else float("nan"),
    }


def sweep(seed: int = 7, severities=SEVERITIES, margin_severities=(0.0, 0.25, 0.5)):
    """Fault sweep (main table) and gate-margin sweep (safety vs. availability)."""
    model, safe, Xtr, Xte, yte = build(seed)
    rng = np.random.default_rng(seed + 1000)
    fault_rows, margin_rows = [], []
    variants = {m: SafeBeat(model, QualityGate(margin=m).fit(Xtr), alpha=0.1) for m in MARGINS}
    for v in variants.values():
        v.q = safe.q
    for fault in FAULTS:
        for sev in severities:
            Xf = np.array([apply_fault(x, fault, sev, rng) for x in Xte])
            pre = precompute(Xf)
            fault_rows.append({"fault": fault, "severity": sev, **score(model, safe, yte, pre)})
            if fault in ("noise", "motion") and sev in margin_severities:
                for m, v in variants.items():
                    margin_rows.append({"fault": fault, "severity": sev, "margin": m, **score(model, v, yte, pre)})
    return fault_rows, margin_rows


def _write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def save(fault_rows, margin_rows, out_dir: Path = RESULTS) -> None:
    out_dir.mkdir(exist_ok=True)
    _write_csv(out_dir / "fault_sweep.csv", fault_rows)
    _write_csv(out_dir / "margin_sweep.csv", margin_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, fault in zip(axes.ravel(), FAULTS):
        sub = [r for r in fault_rows if r["fault"] == fault]
        s = [r["severity"] for r in sub]
        ax.plot(s, [r["naive_wrong_rate"] for r in sub], "o-", color="#c0392b", label="naive: wrong answers")
        ax.plot(s, [r["safebeat_wrong_rate"] for r in sub], "o-", color="#2c7a4b", label="SafeBeat: wrong answers")
        ax.plot(s, [r["safebeat_answer_rate"] for r in sub], "--", color="#555555", label="SafeBeat: answered")
        ax.set_title(fault)
        ax.set_xlabel("fault severity")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("share of all windows")
    axes[1, 0].set_ylabel("share of all windows")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Synthetic ECG: wrong answers vs. fault severity (test subjects never seen in training)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "fault_sweep.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for fault, sev, color in (("noise", 0.25, "#2c7a4b"), ("noise", 0.5, "#1f5f8b"), ("motion", 0.25, "#b9770e")):
        sub = [r for r in margin_rows if r["fault"] == fault and r["severity"] == sev]
        ax.plot(
            [r["safebeat_answer_rate"] for r in sub],
            [r["safebeat_wrong_rate"] for r in sub],
            "o-",
            color=color,
            label=f"{fault}, severity {sev}",
        )
    ax.set_xlabel("share of windows answered (availability)")
    ax.set_ylabel("share of windows answered WRONG (unsafe)")
    ax.set_title("Looser gate = more answers, more wrong answers")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "safety_vs_availability.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    faults, margins = sweep()
    save(faults, margins)
    print(f"{'fault':14s} {'sev':>5s} {'naive_wrong':>12s} {'sb_answered':>12s} {'sb_wrong':>9s}")
    for r in faults:
        print(
            f"{r['fault']:14s} {r['severity']:5.2f} {r['naive_wrong_rate']:12.3f} "
            f"{r['safebeat_answer_rate']:12.3f} {r['safebeat_wrong_rate']:9.3f}"
        )
    print()
    print(f"{'fault':8s} {'sev':>5s} {'margin':>7s} {'answered':>9s} {'wrong':>7s}")
    for r in margins:
        print(f"{r['fault']:8s} {r['severity']:5.2f} {r['margin']:7.1f} {r['safebeat_answer_rate']:9.3f} {r['safebeat_wrong_rate']:7.3f}")
