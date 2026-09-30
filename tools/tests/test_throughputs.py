"""Tests for the Notebook B throughput extract (tools/make_throughputs_extract.py)
and the regenerated syseng reference m5 table (tools/make_m5_reference.py).

The reference system curves in tools/tests/data/syseng_system_1nm.csv were written
by syseng_throughputs' own ``buildHardwareAndSystem`` (via rubin_sim) at the pinned
commit, sampled at 1 nm; see tools/make_m5_reference.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.make_throughputs_extract import (
    BANDS, SYSENG_SHA, read_throughput_text, resample, savitzky_golay,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "week2" / "data"
THRU = DATA / "throughputs.csv"
SKY = DATA / "darksky.csv"
M5REF = DATA / "m5_reference.csv"
FIXTURE = Path(__file__).resolve().parent / "data" / "syseng_system_1nm.csv"


def _read(path):
    return pd.read_csv(path, comment="#")


# ---- pure-function unit tests --------------------------------------------

def test_reader_skips_comment_lines_even_with_data_on_them():
    # syseng filter/glass files put the first data point on the header line;
    # rubin_sim's reader skips that whole line, and so must we.
    text = "#Wavelength(nm)   Throughput(0-1)3.0000e+02\t9.8760e-05\n301 0.5\n300.5 0.25\n\n"
    w, sb = read_throughput_text(text)
    assert np.allclose(w, [300.5, 301.0]) and np.allclose(sb, [0.25, 0.5])


def test_resample_is_zero_outside_the_data():
    w, sb = np.array([400.0, 500.0]), np.array([1.0, 1.0])
    out = resample(w, sb, np.array([300.0, 450.0, 600.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0])


def test_savitzky_golay_preserves_a_cubic():
    x = np.linspace(-1, 1, 201)
    y = 1 + x - 2 * x**2 + 0.5 * x**3
    assert np.allclose(savitzky_golay(y, 31, 3)[20:-20], y[20:-20], atol=1e-10)


# ---- throughputs.csv -------------------------------------------------------

COLS = (["wavelength_nm", "mirrors", "lenses", "detector"]
        + [f"filter_{b}" for b in BANDS] + ["atmos_X1.0", "atmos_X1.2"])


def test_throughput_grid_complete():
    t = _read(THRU)
    assert list(t.columns) == COLS
    assert np.array_equal(t.wavelength_nm.to_numpy(), np.arange(300, 1101))
    assert not t.isna().any().any()


def test_throughput_values_in_unit_interval():
    t = _read(THRU).drop(columns="wavelength_nm")
    assert (t.to_numpy() >= 0).all() and (t.to_numpy() <= 1).all()


def test_header_records_pinned_sha():
    head = THRU.read_text().splitlines()[0:20]
    assert any(SYSENG_SHA in line for line in head)
    assert len(SYSENG_SHA) == 40 and SYSENG_SHA.startswith("00570b3d39")


@pytest.mark.parametrize("atm", ["X1.0", "X1.2"])
@pytest.mark.parametrize("band", BANDS)
def test_system_matches_syseng_buildHardwareAndSystem(band, atm):
    t = _read(THRU)
    ref = _read(FIXTURE)
    assert np.array_equal(ref.wavelength_nm.to_numpy(), t.wavelength_nm.to_numpy())
    ours = (t.mirrors * t.lenses * t.detector * t[f"filter_{band}"] * t[f"atmos_{atm}"]).to_numpy()
    theirs = ref[f"system_{atm}_{band}"].to_numpy()
    inband = theirs > 0.01 * theirs.max()
    rms = np.sqrt(np.mean((ours[inband] - theirs[inband]) ** 2)) / theirs.max()
    assert rms < 0.01


@pytest.mark.parametrize("band", BANDS)
def test_hardware_matches_syseng(band):
    t = _read(THRU)
    ref = _read(FIXTURE)
    ours = (t.mirrors * t.lenses * t.detector * t[f"filter_{band}"]).to_numpy()
    theirs = ref[f"hardware_{band}"].to_numpy()
    inband = theirs > 0.01 * theirs.max()
    # exact replication: agrees to the 6-significant-figure rounding of the CSVs
    assert np.sqrt(np.mean((ours[inband] - theirs[inband]) ** 2)) / theirs.max() < 1e-4


# ---- darksky.csv -----------------------------------------------------------

def test_darksky_grid_and_units():
    s = _read(SKY)
    assert list(s.columns) == ["wavelength_nm", "flambda_erg_s_cm2_nm"]
    assert np.array_equal(s.wavelength_nm.to_numpy(), np.arange(300, 1101))
    assert (s.flambda_erg_s_cm2_nm > 0).all() and (s.flambda_erg_s_cm2_nm < 1e-14).all()
    assert "erg" in SKY.read_text().splitlines()[0] or any(
        "erg" in l for l in SKY.read_text().splitlines()[:15] if l.startswith("#"))


# ---- m5_reference.csv ------------------------------------------------------

def test_m5_reference_table():
    m = _read(M5REF).set_index("band")
    assert list(m.index) == list(BANDS)
    for col in ["m5", "FWHMeff", "skyMag", "Zp_t", "Cm", "dCm_double", "m5_X1.2"]:
        assert col in m.columns and m[col].notna().all()
    assert (m["m5_X1.2"] < m["m5"]).all()           # higher airmass is shallower
    assert 24.0 < m.loc["r", "m5"] < 25.0
    header = [l for l in M5REF.read_text().splitlines() if l.startswith("#")]
    text = "\n".join(header)
    assert SYSENG_SHA in text and "rubin-sim==" in text and "readnoise=8.8" in text


def test_files_total_under_2MB():
    total = sum(p.stat().st_size for p in (THRU, SKY, M5REF))
    assert total < 2 * 1024**2
