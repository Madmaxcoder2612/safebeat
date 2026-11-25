"""Tests for the real-data part. Tests that need the downloaded recordings are skipped
when data/mitdb/ is missing, so CI and fresh clones still pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from safebeat.evaluate_real import AF_FOLDS, AF_RECORDS, NOTICE, make_folds
from safebeat.realdata import DATA_DIR, PACED, list_records, load_windows
from safebeat.synth import FS, SECONDS

ROOT = Path(__file__).resolve().parent.parent
HAVE_DATA = (DATA_DIR / "100.hea").exists()


def test_folds_are_disjoint_and_cover_every_record():
    records = [f"{n}" for n in range(100, 135)] + sorted(AF_RECORDS)
    records = sorted(set(records))
    folds = make_folds(records)
    flat = [r for f in folds for r in f]
    assert sorted(flat) == records  # nothing lost
    assert len(flat) == len(set(flat))  # nothing twice
    # AF records are spread over different folds, so each fold tests some AF
    assert all(any(r in AF_RECORDS for r in f) for f in folds)


def test_af_folds_cover_exactly_the_af_records():
    assert {r for f in AF_FOLDS for r in f} == AF_RECORDS


def test_paced_records_are_never_listed():
    if not HAVE_DATA:
        pytest.skip("MIT-BIH data not downloaded")
    assert not set(list_records()) & PACED


def test_loaded_windows_have_expected_shape_and_labels():
    if not HAVE_DATA:
        pytest.skip("MIT-BIH data not downloaded")
    X, y = load_windows("100")
    assert X.ndim == 2 and X.shape[1] == FS * SECONDS
    assert len(X) == len(y) > 0
    assert set(y.tolist()) <= {0, 1}
    assert y.max() == 0  # record 100 is normal sinus rhythm
    Xa, ya = load_windows("203")
    assert ya.sum() > 50  # record 203 has long stretches of atrial fibrillation


# ---------- credit and citation must not get lost ----------


def test_notice_text_names_database_and_licence_links():
    assert "MIT-BIH Arrhythmia Database" in NOTICE
    assert "physionet.org/content/mitdb/1.0.0/" in NOTICE
    assert "ODC Attribution License" in NOTICE
    assert "opendatacommons.org/licenses/by/1-0/" in NOTICE


def test_notice_file_has_all_three_citations_and_the_licence():
    text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for needle in (
        "https://doi.org/10.13026/C2F305",  # database citation
        "PMID: 11446209",  # original publication
        "https://doi.org/10.1038/s44360-026-00096-z",  # PhysioNet citation
        "Open Data Commons Attribution License v1.0",
        "https://opendatacommons.org/licenses/by/1-0/",
        "NOT included in this repository",
    ):
        assert needle in text, needle


def test_readme_carries_the_notice_next_to_real_data_results():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Contains information from the MIT-BIH Arrhythmia Database" in text
    assert "ODC Attribution License" in text


def test_no_recordings_are_tracked_in_the_repository():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in gi
    assert not list(ROOT.glob("*.dat")) and not list(ROOT.glob("*.hea")) and not list(ROOT.glob("*.atr"))
