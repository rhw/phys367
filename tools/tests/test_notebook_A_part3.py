"""Task 5: Notebook A, Part 3 ("one point on the sky"), Part 4 ("go deeper") and hand-in."""
import numpy as np
import pandas as pd
import pytest

from tools import build_notebooks

CHECK = r'''
import json, io, contextlib
_cos = visits_at('COSMOS')
_s0, _e0 = seasons(_cos['mjd'].values)[0]
_hits = [int(observe('COSMOS', kilonova, t0)['detected'].sum())
         for t0 in np.arange(_s0 - 1.0, _e0, 2.0)]

# noise-free, daily-sampled, gap-free synthetic
_syn = lensed_quasar(mjd=61208.0 + np.arange(3 * 365), noise=False)
_d_syn = estimate_delay(_syn)
_rx = lensed_quasar('RXJ1131-1231')
_d_rx = estimate_delay(_rx)
_rx_clean = lensed_quasar('RXJ1131-1231', noise=False)
_d_rx_clean = estimate_delay(_rx_clean)

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _ec = cadence_plot('north-edge')
    _eo = observe('north-edge', kilonova, 61300.0)
    _eq = lensed_quasar('north-edge')
    _ed = estimate_delay(_eq)
    _ef = fraction_caught('north-edge', kilonova)
_msg = _buf.getvalue()

_cc = cadence_plot('COSMOS')
_lc = observe('main-1', sn_ia, 61500.0)
_dm200 = 5 * np.log10(200e6 / 10)
_fr = {p: fraction_caught(p, kilonova, within_days=2) for p in ['COSMOS', 'main-1']}
_fr_repeat = fraction_caught('COSMOS', kilonova, within_days=2)
print("CHECKS" + json.dumps(dict(
    maxhits=max(_hits), merr=float(mag_err(24.0, 24.0)),
    merr_bright=float(mag_err(21.5, 24.0)),
    d_syn=float(_d_syn), d_rx=float(_d_rx), d_rx_clean=float(_d_rx_clean),
    n_rx=len(_rx),
    empty_lens=[len(_ec), len(_eo), len(_eq)], empty_delay_nan=bool(np.isnan(_ed)),
    empty_frac=float(_ef), msg=_msg,
    n_cos=len(_cc),
    kn_r0=float(kilonova(np.array([0.0]), 'r')[0]),
    kn_g2=float(kilonova(np.array([2.0]), 'g')[0]),
    kn_neg_nan=bool(np.isnan(kilonova(np.array([-1.0]), 'r')[0])),
    sn_r0=float(sn_ia(np.array([0.0]), 'r')[0]),
    sn_r_later=float(sn_ia(np.array([30.0]), 'r')[0]),
    sn_early_finite=bool(np.all(np.isfinite(sn_ia(np.array([-500.0]), 'r')))),
    dm03=float(np.interp(0.3, _Z, _DM)), dm200=float(_dm200),
    lc_cols=sorted(_lc.columns), fr=_fr, fr_repeat=float(_fr_repeat),
)))
'''


@pytest.fixture(scope="module")
def nb_checks(run_notebook_checks):
    return run_notebook_checks(CHECK)


def test_part3_brief_checks(nb_checks):
    c = nb_checks
    assert c['maxhits'] >= 1                         # kilonova seen at COSMOS in season 1
    assert abs(c['d_syn'] - 30) <= 2                 # noise-free daily synthetic
    assert np.isfinite(c['d_rx'])                    # RXJ1131 returns a number
    assert abs(c['merr'] - 0.217) < 1e-3


def test_part3_empty_preset(nb_checks):
    c = nb_checks
    assert c['empty_lens'] == [0, 0, 0]
    assert c['empty_delay_nan'] and c['empty_frac'] == 0.0
    assert ("No visits within 1.75° of north-edge in this simulation — why might that be?"
            in c['msg'])


def test_part3_models(nb_checks):
    c = nb_checks
    assert c['n_cos'] > 1000
    assert abs(c['kn_r0'] - (-16 + c['dm200'])) < 1e-9
    assert abs(c['kn_g2'] - (-16 + c['dm200'] + 2.0)) < 1e-9     # g fades 1.0 mag/day
    assert c['kn_neg_nan']
    assert abs(c['sn_r0'] - (-19.3 - 0.1 + c['dm03'])) < 1e-6    # t = 0 is peak
    assert c['sn_r_later'] > c['sn_r0']
    assert c['sn_early_finite']
    assert {'mjd', 't', 'band', 'm5', 'm_true', 'm_obs', 'm_err', 'detected'} <= set(c['lc_cols'])
    assert c['merr_bright'] < 0.03
    assert np.isfinite(c['d_rx_clean'])
    assert 0.0 <= c['fr']['COSMOS'] <= 1.0 and 0.0 <= c['fr']['main-1'] <= 1.0
    assert c['fr']['COSMOS'] == c['fr_repeat']                    # seeded


def _all_text():
    text = "\n".join(t for _, t in build_notebooks.PART3_CELLS + build_notebooks.PART4_CELLS)
    return " ".join(text.split())              # ignore line wrapping


def test_part4_links_and_handin():
    t = _all_text()
    for url in [
        "https://rubin-sim.lsst.io",
        "https://rubin-sim.lsst.io/data-download.html",
        "https://github.com/lsst/rubin_sim_notebooks/tree/main/maf/tutorial",
        "https://github.com/lsst/rubin_sim_notebooks/tree/main/scheduler",
        "https://github.com/lsst/rubin_sim_notebooks/tree/main/maf/science",
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/baseline_v5.3.3_10yrs.db",
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/",
        "https://usdf-maf.slac.stanford.edu/",
        "https://survey-strategy.lsst.io",
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/schedview/reports/",
    ]:
        assert url in t, url
    for s in ["7,267", "17,273", "750 MB", "02_Writing_Metrics", "201_10", "Be the SCOC",
              "Due Monday Oct 5 on Canvas", "Graded complete/incomplete",
              "Do the deep drilling fields win? Why or why not?",
              "MacLeod et al. 2010", "AT2017gfo"]:
        assert s in t, s
    assert "Why do the DDFs win" not in t
    last_kind, last_text = build_notebooks.PART4_CELLS[-1]
    assert last_kind == "md" and "Due Monday" not in last_text   # empty paragraph cell last


def test_presets_blurbs_consistent():
    from tools.make_baseline_extract import PRESETS
    csv = pd.read_csv(build_notebooks.REPO / "week2" / "data" / "presets.csv")
    code = pd.DataFrame(PRESETS, columns=["name", "ra", "dec", "blurb"])
    pd.testing.assert_frame_equal(csv.reset_index(drop=True), code)
    assert "equator" not in " ".join(csv["blurb"])
    for n in ["main-1", "main-2", "main-3"]:
        assert csv.set_index("name").loc[n, "blurb"] == "Main survey (WFD) field"
