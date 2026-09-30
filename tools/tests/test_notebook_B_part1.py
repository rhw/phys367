"""Notebook B, Part 1 ("Depth from first principles").

The notebook's first-principles m5() must reproduce syseng_throughputs' own makeM5
(week2/data/m5_reference.csv, regenerated at the pinned commit) to within 0.05 mag in
every band. Do not loosen the tolerance: if it misses, find out why.
"""
import json

import pytest

from tools import build_notebooks

CHECK = r'''
import json
ref = load("m5_reference").set_index("band")
bands = list("ugrizy")
out = {}
out["m5"] = {b: float(m5(b)) for b in bands}
out["m5_x12"] = {b: float(m5(b, airmass=1.2)) for b in bands}
out["ref_m5"] = {b: float(ref.loc[b, "m5"]) for b in bands}
out["ref_m5_x12"] = {b: float(ref.loc[b, "m5_X1.2"]) for b in bands}
out["zp_mag"] = {b: float(2.5 * np.log10(zeropoint(b))) for b in bands}
out["ref_zp"] = {b: float(ref.loc[b, "Zp_t"]) for b in bands}
out["skymag"] = {b: float(sky_mag_dark(b)) for b in bands}
out["ref_skymag"] = {b: float(ref.loc[b, "skyMag"]) for b in bands}
out["d30"] = {b: float(m5(b, t_exp=30, n_exp=1) - m5(b)) for b in bands}
# sky_mag equal to the dark-sky value must give the same m5 as the default
out["sky_consistent"] = {b: float(m5(b, sky_mag=sky_mag_dark(b)) - m5(b)) for b in bands}
out["deeper_bigger"] = bool(m5("r", area_m2=100.0) > m5("r") > m5("r", area_m2=10.0))
out["brighter_sky_shallower"] = bool(m5("r", sky_mag=19.0) < m5("r"))
out["worse_seeing_shallower"] = bool(m5("r", fwhm_eff=1.5) < m5("r"))

# extreme knob settings: nothing crashes, everything finite
extremes = []
for kw in [dict(read_noise=0.0), dict(read_noise=50.0), dict(fwhm_eff=0.3), dict(fwhm_eff=3.0),
           dict(sky_mag=15.0), dict(sky_mag=25.0), dict(t_exp=1, n_exp=1), dict(t_exp=300, n_exp=10),
           dict(area_m2=0.5), dict(area_m2=200.0), dict(airmass=2.5), dict(dark=0.0),
           dict(airmass=1.0, fwhm_eff=0.3, read_noise=0.0, sky_mag=25.0, t_exp=300, n_exp=10)]:
    for b in bands:
        extremes.append(float(m5(b, **kw)))
out["extremes_finite"] = bool(np.all(np.isfinite(extremes)))
show_m5("u", aperture_m=1.0, read_noise=50.0, fwhm_eff=3.0, sky_mag=16.0, t_exp=1, n_exp=1)
show_m5("y", aperture_m=12.0, read_noise=0.0, fwhm_eff=0.3, sky_mag=24.0, t_exp=300, n_exp=10)
show_components()
show_components(airmass=1.2)
plt.close("all")
import inspect
out["widget_params"] = list(inspect.signature(_m5_widget).parameters)
out["offset0"] = {b: float(m5(b, fwhm_eff=FWHM_EFF_ZENITH[b] + 0.0) - m5(b)) for b in bands}
_m5_widget("u", 0.5, 30.0, 1.6, 5.0, 1, 10); _m5_widget("y", 12.0, 0.0, -0.4, 0.0, 300, 1)
plt.close("all")
out["sys_peak_r"] = float(np.max(system("r")))
out["sys_x12_lower"] = bool(np.all(system("u", 1.2) <= system("u", 1.0) + 1e-12))
out["budget_rubin"] = float(survey_budget(RUBIN_DIAM_EFF_M, RUBIN_FOV_DEG2))
out["budget_4m"] = float(survey_budget(4.0, RUBIN_FOV_DEG2))
out["budget_1deg"] = float(survey_budget(RUBIN_DIAM_EFF_M, 1.0))
out["chain"] = [M5_DESIGN_R, float(m5("r")), M5_SIM_MEDIAN_R]
out["sha"] = SYSENG_SHA
print("CHECKS" + json.dumps(out))
'''

BANDS = "ugrizy"


@pytest.fixture(scope="module")
def nb_checks(run_notebook_checks):
    return run_notebook_checks(CHECK, notebook="B")


def test_m5_matches_syseng_reference(nb_checks):
    c = nb_checks
    for b in BANDS:
        assert abs(c["m5"][b] - c["ref_m5"][b]) < 0.05, (b, c["m5"][b], c["ref_m5"][b])


def test_m5_airmass_12_matches_reference(nb_checks):
    c = nb_checks
    for b in BANDS:
        assert abs(c["m5_x12"][b] - c["ref_m5_x12"][b]) < 0.05, b


def test_zeropoint_and_sky_match_reference(nb_checks):
    c = nb_checks
    for b in BANDS:
        assert abs(c["zp_mag"][b] - c["ref_zp"][b]) < 0.02, b
        assert abs(c["skymag"][b] - c["ref_skymag"][b]) < 0.02, b
        assert abs(c["sky_consistent"][b]) < 1e-6, b


def test_one_30s_vs_two_15s(nb_checks):
    d = nb_checks["d30"]
    # syseng makeM5: u 1x30 s 23.697 vs 2x15 s 23.449 -> +0.248; read noise matters most in u
    assert abs(d["u"] - 0.248) < 0.03
    assert all(d["u"] > d[b] > 0 for b in "grizy")


def test_knobs_and_extremes(nb_checks):
    c = nb_checks
    assert c["deeper_bigger"] and c["brighter_sky_shallower"] and c["worse_seeing_shallower"]
    assert c["extremes_finite"]
    assert 0.3 < c["sys_peak_r"] < 0.8
    assert c["sys_x12_lower"]


def test_seeing_slider_is_offset_from_band_fiducial(nb_checks):
    c = nb_checks
    assert "seeing_offset" in c["widget_params"]
    assert all(abs(c["offset0"][b]) < 1e-12 for b in BANDS)


def test_etendue_budget_and_chain(nb_checks):
    c = nb_checks
    assert abs(c["budget_rubin"] - 1.5e7) < 1
    assert abs(c["budget_4m"] / c["budget_rubin"] - (4.0 / 6.423) ** 2) < 1e-6
    assert abs(c["budget_1deg"] / c["budget_rubin"] - 1 / 9.6) < 1e-6
    design, built, sim = c["chain"]
    assert design == 24.7 and sim == 24.06
    assert 24.43 < built < 24.53
    assert c["sha"].startswith("00570b3d39")


def test_notebook_b_source_rules():
    nb = build_notebooks.build_notebook_b()
    src = "\n".join(cell.source for cell in nb.cells)
    assert nb.cells[0].source.startswith("# Week 2B · Camera and telescope → science")
    assert "import tools" not in src and "from tools" not in src
    for banned in ("healpy", "scipy", "rubin_sim import", "import rubin_sim"):
        assert banned not in src
    assert src.count("Where does this break?") >= 4
    assert "00570b3d39" in src
    for word in ("colour", "centre", "modelling", "optimise", "normalis"):
        assert word not in src.lower()
    assert all(not getattr(cell, "outputs", []) for cell in nb.cells)
    ids = [cell.id for cell in nb.cells]
    assert len(set(ids)) == len(ids)
    assert build_notebooks.build_notebook_b().cells[1].id == nb.cells[1].id
