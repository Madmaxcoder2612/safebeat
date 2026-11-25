"""Real-data check: does SafeBeat's gate and "can't tell" rule hold up on MIT-BIH?

Run from the repository root (after python tools/download_mitdb.py):
    python -m safebeat.evaluate_real

Contains information from the MIT-BIH Arrhythmia Database
(https://physionet.org/content/mitdb/1.0.0/), which is made available under the
ODC Attribution License (https://opendatacommons.org/licenses/by/1-0/).
Full credit and citations: see NOTICE.

Design, in plain words:
* Records are split, never windows: a record (person) is entirely in train,
  calibration or test. Three rotations, so every record is tested once.
* Atrial fibrillation (AF) appears in only seven records, so AF numbers rest on
  few people. Read them with that in mind.
* Because AF is rare, results are reported per class and as a class-balanced
  average, so "always say regular" cannot look good.
* Model, gate and "can't tell" thresholds are trained or fitted on the training
  and calibration records only. The same six faults as in the synthetic study are
  then applied to the unseen test records.
* Two SENSITIVITY checks are reported separately and are not tuned results:
  a sweep of the "can't tell" level alpha on clean data, and a sweep of gate
  strictness on movement and noise. They show what a knob does; they are not
  used to pick a setting that is then reported as the main result.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .abstain import SafeBeat
from .evaluate import precompute
from .faults import FAULTS, apply_fault
from .features import extract_features_batch
from .model import RhythmModel
from .quality import QualityGate
from .realdata import list_records, load_windows

RESULTS = Path(__file__).resolve().parent.parent / "results"
AF_FOLDS = [["201", "203"], ["202", "219"], ["210", "221", "222"]]
AF_RECORDS = {r for f in AF_FOLDS for r in f}
CONDITIONS = [("none", 0.0)] + [(f, s) for f in FAULTS for s in (0.5, 1.0)]
ALPHAS = (0.02, 0.05, 0.1, 0.2)
MARGINS = (0.5, 1.0, 1.5, 3.0)
MARGIN_CONDITIONS = {("none", 0.0), ("motion", 0.5), ("motion", 1.0), ("noise", 0.5)}
NOTICE = (
    "Contains information from the MIT-BIH Arrhythmia Database "
    "(physionet.org/content/mitdb/1.0.0/), which is made available under the "
    "ODC Attribution License (opendatacommons.org/licenses/by/1-0/)."
)


def make_folds(records: list[str]) -> list[list[str]]:
    folds = [[r for r in f if r in records] for f in AF_FOLDS]
    others = [r for r in records if r not in AF_RECORDS]
    for i, r in enumerate(others):
        folds[i % len(folds)].append(r)
    return folds


def _take(data, recs, rng, reg_cap):
    """Stack windows of the given records; cap regular windows per record, keep all AF."""
    X, y = [], []
    for r in recs:
        Xr, yr = data[r]
        if len(yr) == 0:
            continue
        reg = np.flatnonzero(yr == 0)
        if len(reg) > reg_cap:
            reg = rng.choice(reg, reg_cap, replace=False)
        idx = np.concatenate([reg, np.flatnonzero(yr == 1)]).astype(int)
        X.append(Xr[idx])
        y.append(yr[idx])
    return np.concatenate(X), np.concatenate(y)


def _summarise(p) -> dict[str, float]:
    y = np.concatenate(p["y"])
    naive = np.concatenate(p["naive"])
    ans = np.concatenate(p["answered"])
    lab = np.concatenate(p["label"])
    row: dict[str, float] = {"n_regular": int((y == 0).sum()), "n_af": int((y == 1).sum())}
    for c, name in ((0, "regular"), (1, "af")):
        m = y == c
        wrong = ans[m] & (lab[m] != y[m])
        row[f"naive_wrong_{name}"] = float(np.mean(naive[m] != y[m]))
        row[f"safebeat_answered_{name}"] = float(np.mean(ans[m]))
        row[f"safebeat_wrong_{name}"] = float(np.mean(wrong))
        row[f"safebeat_error_among_answered_{name}"] = float(np.sum(wrong) / np.sum(ans[m])) if ans[m].any() else float("nan")
    for k in ("naive_wrong", "safebeat_wrong", "safebeat_answered"):
        row[f"{k}_balanced"] = (row[f"{k}_regular"] + row[f"{k}_af"]) / 2
    row["safebeat_error_among_answered_balanced"] = float(
        np.nanmean([row["safebeat_error_among_answered_regular"], row["safebeat_error_among_answered_af"]])
    )
    return row


def run(seed: int = 5):
    rng = np.random.default_rng(seed)
    records = list_records()
    data = {r: load_windows(r) for r in records}
    folds = make_folds(records)
    keys = set()
    pooled: dict[tuple, dict] = {}

    def add(key, y, naive, decisions):
        p = pooled.setdefault(key, {"y": [], "naive": [], "answered": [], "label": []})
        p["y"].append(y)
        p["naive"].append(naive)
        p["answered"].append(np.array([d.label is not None for d in decisions]))
        p["label"].append(np.array([d.label if d.label is not None else -1 for d in decisions]))
        keys.add(key)

    for i, test_recs in enumerate(folds):
        rest = [r for j, f in enumerate(folds) if j != i for r in f]
        af_rest = sorted(r for r in rest if r in AF_RECORDS)
        reg_rest = sorted(r for r in rest if r not in AF_RECORDS)
        cal_recs = af_rest[::3] + reg_rest[::4]
        train_recs = [r for r in rest if r not in cal_recs]
        Xtr, ytr = _take(data, train_recs, rng, 120)
        Xca, yca = _take(data, cal_recs, rng, 60)
        Xte, yte = _take(data, test_recs, rng, 60)
        assert not set(train_recs) & set(test_recs) and not set(cal_recs) & set(test_recs)
        model = RhythmModel(class_weight="balanced").fit(extract_features_batch(Xtr), ytr)
        feats_ca = extract_features_batch(Xca)
        gates = {m: QualityGate(margin=m).fit(Xtr) for m in MARGINS}
        main = SafeBeat(model, gates[1.5], alpha=0.1, class_conditional=True).calibrate(feats_ca, yca)
        alpha_v = {a: SafeBeat(model, gates[1.5], alpha=a, class_conditional=True).calibrate(feats_ca, yca) for a in ALPHAS}
        margin_v = {m: SafeBeat(model, gates[m], alpha=0.1, class_conditional=True).calibrate(feats_ca, yca) for m in MARGINS}
        print(
            f"fold {i}: train {len(ytr)} (AF {int(ytr.sum())}), cal {len(yca)} (AF {int(yca.sum())}), "
            f"test {len(yte)} (AF {int(yte.sum())}), AF test records {[r for r in test_recs if r in AF_RECORDS]}",
            flush=True,
        )
        for cond in CONDITIONS:
            fault, sev = cond
            Xf = Xte if fault == "none" else np.array([apply_fault(x, fault, sev, rng) for x in Xte])
            metrics, feats = precompute(Xf)
            naive = model.predict(feats)

            def decide(s):
                return [s.decide_from(m, f) for m, f in zip(metrics, feats)]

            add(("main", None, fault, sev), yte, naive, decide(main))
            if cond == ("none", 0.0):
                for a, s in alpha_v.items():
                    add(("alpha", a, fault, sev), yte, naive, decide(s))
            if cond in MARGIN_CONDITIONS:
                for m, s in margin_v.items():
                    add(("margin", m, fault, sev), yte, naive, decide(s))
    rows = []
    for (kind, param, fault, sev), p in sorted(pooled.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2], kv[0][3])):
        rows.append({"kind": kind, "param": param if param is not None else "", "fault": fault, "severity": sev, **_summarise(p)})
    return rows


def _write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as f:
        f.write(f"# {NOTICE}\n# Parse with pandas.read_csv(path, comment='#').\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def save(rows, out_dir: Path = RESULTS) -> None:
    out_dir.mkdir(exist_ok=True)
    main = [r for r in rows if r["kind"] == "main"]
    sens = [r for r in rows if r["kind"] != "main"]
    _write_csv(out_dir / "real_mitdb_results.csv", main)
    _write_csv(out_dir / "real_mitdb_sensitivity.csv", sens)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = next(r for r in main if r["fault"] == "none")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharey=True)
    for ax, fault in zip(axes.ravel(), FAULTS):
        sub = [clean] + [r for r in main if r["fault"] == fault]
        s = [0.0] + [r["severity"] for r in sub[1:]]
        ax.plot(s, [r["naive_wrong_balanced"] for r in sub], "o-", color="#c0392b", label="naive: wrong answers")
        ax.plot(s, [r["safebeat_wrong_balanced"] for r in sub], "o-", color="#2c7a4b", label="SafeBeat: wrong answers")
        ax.plot(s, [r["safebeat_answered_balanced"] for r in sub], "--", color="#555555", label="SafeBeat: answered")
        ax.set_title(fault)
        ax.set_xlabel("fault severity")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("class-balanced share of windows")
    axes[1, 0].set_ylabel("class-balanced share of windows")
    axes[0, 0].legend(fontsize=8, loc="center left")
    fig.suptitle("Real ECG (MIT-BIH): regular rhythm vs atrial fibrillation, records never seen in training")
    fig.text(0.5, 0.012, NOTICE, ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "real_mitdb_fault_sweep.png", dpi=130)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    al = sorted((r for r in sens if r["kind"] == "alpha"), key=lambda r: r["param"])
    a1.plot([r["safebeat_answered_balanced"] for r in al], [r["safebeat_error_among_answered_balanced"] for r in al], "o-", color="#2c7a4b", label="SafeBeat, alpha 0.02 to 0.2")
    for r in al:
        a1.annotate(f"alpha {r['param']}", (r["safebeat_answered_balanced"], r["safebeat_error_among_answered_balanced"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    a1.plot([1.0], [clean["naive_wrong_balanced"]], "s", color="#c0392b", label="naive (always answers)")
    a1.set_xlabel("share of windows answered (class-balanced)")
    a1.set_ylabel("error among answered (class-balanced)")
    a1.set_title("Clean real data: asking SafeBeat to be pickier")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    for fault, sev, color in (("motion", 0.5, "#a23b8c"), ("motion", 1.0, "#b9770e"), ("noise", 0.5, "#1f5f8b")):
        mg = sorted((r for r in sens if r["kind"] == "margin" and r["fault"] == fault and r["severity"] == sev), key=lambda r: r["param"])
        a2.plot([r["param"] for r in mg], [r["safebeat_answered_balanced"] for r in mg], "--", color=color)
        a2.plot([r["param"] for r in mg], [r["safebeat_wrong_balanced"] for r in mg], "o-", color=color, label=f"{fault} {sev}: wrong (solid), answered (dashed)")
    a2.set_xlabel("gate margin (smaller = stricter)")
    a2.set_ylabel("class-balanced share of windows")
    a2.set_title("Sensitivity: gate strictness on movement and noise")
    a2.set_ylim(0, 1.02)
    a2.grid(alpha=0.3)
    a2.legend(fontsize=7)
    fig.text(0.5, 0.01, NOTICE, ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / "real_mitdb_sensitivity.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    save(out)
    cols = ["kind", "param", "fault", "severity", "naive_wrong_balanced", "safebeat_answered_balanced", "safebeat_wrong_balanced", "safebeat_error_among_answered_balanced"]
    print(" ".join(f"{c[:15]:>15s}" for c in cols))
    for r in out:
        print(" ".join(f"{r[c]:15.3f}" if isinstance(r[c], float) else f"{str(r[c]):>15s}" for c in cols))
    print(f"windows: regular {out[0]['n_regular']}, AF {out[0]['n_af']}")
