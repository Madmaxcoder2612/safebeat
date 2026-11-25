"""Tests for SafeBeat. Run from the repository root:  python -m pytest -q"""

from __future__ import annotations

import numpy as np
import pytest

from safebeat.abstain import SafeBeat
from safebeat.evaluate import build, precompute, score
from safebeat.faults import FAULTS, apply_fault
from safebeat.features import detect_r_peaks, extract_features
from safebeat.quality import quality_metrics
from safebeat.synth import FS, SECONDS, make_dataset, make_subject, render_window


@pytest.fixture(scope="module")
def trained():
    # Small and fast on purpose; the real run uses more subjects.
    return build(seed=3, n_train=20, n_cal=15, n_test=12, wps=20)


# ---------- synthetic data ----------


def test_window_shape_and_determinism():
    a = render_window(np.random.default_rng(1), make_subject(np.random.default_rng(1), 0), 0)
    b = render_window(np.random.default_rng(1), make_subject(np.random.default_rng(1), 0), 0)
    assert a.shape == (FS * SECONDS,)
    assert np.array_equal(a, b)


def test_dataset_balanced_and_subject_labels():
    X, y, owners = make_dataset(np.random.default_rng(0), [5, 6], 10)
    assert X.shape == (20, FS * SECONDS)
    assert y.sum() == 10
    assert set(owners) == {5, 6}


def test_peak_detector_counts_beats_on_clean_window():
    rng = np.random.default_rng(2)
    s = make_subject(rng, 0)
    x = render_window(rng, s, 0)
    expected = SECONDS * s.hr_bpm / 60.0
    assert abs(len(detect_r_peaks(x)) - expected) <= 2


def test_features_have_fixed_length_even_for_flat_signal():
    assert extract_features(np.zeros(FS * SECONDS)).shape == (6,)


# ---------- faults ----------


@pytest.mark.parametrize("kind", FAULTS)
def test_fault_severity_zero_is_identity_and_one_changes_signal(kind):
    rng = np.random.default_rng(4)
    x = render_window(rng, make_subject(rng, 0), 0)
    assert np.array_equal(apply_fault(x, kind, 0.0, rng), x)
    y = apply_fault(x, kind, 1.0, rng)
    assert y.shape == x.shape
    assert not np.array_equal(x, y)


def test_fault_does_not_modify_input_and_rejects_bad_arguments():
    rng = np.random.default_rng(5)
    x = render_window(rng, make_subject(rng, 0), 0)
    before = x.copy()
    apply_fault(x, "noise", 1.0, rng)
    assert np.array_equal(x, before)
    with pytest.raises(ValueError):
        apply_fault(x, "nonsense", 0.5, rng)
    with pytest.raises(ValueError):
        apply_fault(x, "noise", 1.5, rng)


def test_electrode_off_creates_flat_stretch():
    rng = np.random.default_rng(6)
    x = render_window(rng, make_subject(rng, 0), 0)
    y = apply_fault(x, "electrode_off", 1.0, rng)
    assert quality_metrics(y)["flat_frac"] > 0.5
    assert quality_metrics(x)["flat_frac"] < 0.05


# ---------- quality gate ----------


def _fraction(flags):
    return float(np.mean(flags))


def test_gate_passes_most_clean_heldout_windows(trained):
    _, safe, _, Xte, _ = trained
    ok = [not safe.gate.check(quality_metrics(x)) for x in Xte]
    assert _fraction(ok) >= 0.95


@pytest.mark.parametrize("kind", ["electrode_off", "dropout"])
def test_gate_rejects_flatline_and_dropout(trained, kind):
    _, safe, _, Xte, _ = trained
    rng = np.random.default_rng(8)
    bad = [safe.gate.check(quality_metrics(apply_fault(x, kind, 0.75, rng))) for x in Xte]
    assert _fraction([len(b) > 0 for b in bad]) >= 0.9


def test_baseline_wander_warns_but_does_not_reject(trained):
    _, safe, _, Xte, _ = trained
    rng = np.random.default_rng(9)
    m = [quality_metrics(apply_fault(x, "wander", 1.0, rng)) for x in Xte]
    assert _fraction([not safe.gate.check(k) for k in m]) >= 0.95
    assert _fraction(["wander" in safe.gate.warnings(k) for k in m]) >= 0.9


# ---------- conformal abstain ----------


def test_decide_before_calibrate_raises(trained):
    model, safe, _, Xte, _ = trained
    fresh = SafeBeat(model, safe.gate)
    with pytest.raises(RuntimeError):
        fresh.decide(Xte[0])


def test_conformal_coverage_on_clean_unseen_subjects(trained):
    """The label SafeBeat keeps in its set should contain the truth about 90 % of the time."""
    model, safe, _, Xte, yte = trained
    feats = np.array([extract_features(x) for x in Xte])
    p = model.predict_proba(feats)
    covered = 1.0 - p[np.arange(len(yte)), yte] <= safe.q
    assert covered.mean() >= 0.85  # target 0.90, small test set, allow slack


def test_safebeat_abstains_on_flatline_and_says_why(trained):
    _, safe, _, _, _ = trained
    d = safe.decide(np.zeros(FS * SECONDS))
    assert d.label is None
    assert d.reason.startswith("bad_signal")


# ---------- the headline claims ----------


def test_safebeat_beats_naive_on_electrode_off(trained):
    model, safe, _, Xte, yte = trained
    rng = np.random.default_rng(10)
    Xf = np.array([apply_fault(x, "electrode_off", 0.75, rng) for x in Xte])
    r = score(model, safe, yte, precompute(Xf))
    assert r["naive_wrong_rate"] > 0.2
    assert r["safebeat_wrong_rate"] < 0.05


def test_safebeat_is_not_trivially_abstaining_on_clean_data(trained):
    model, safe, _, Xte, yte = trained
    r = score(model, safe, yte, precompute(Xte))
    assert r["safebeat_answer_rate"] > 0.8
    assert r["safebeat_wrong_rate"] <= r["naive_wrong_rate"] + 1e-9


def test_train_cal_test_subjects_do_not_overlap():
    # build() asserts disjoint subject ids internally; a tiny build proves it runs.
    build(seed=1, n_train=3, n_cal=3, n_test=3, wps=4)
