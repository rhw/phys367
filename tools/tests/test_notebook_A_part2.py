"""Task 4: Notebook A, Part 2 ("the real plan: baseline v5.3.3")."""
import pytest

CHECK = r'''
import json
import matplotlib
_m = load('maps')
_b = budget_table()
_ax = sky_map('nvis', 'u', which='y1')
_ax2 = sky_map('m5_coadd', 'r')
_main = main_survey_medians()
_cmp = compare_to_mine(plan_survey(18000, 30, DEFAULT_SPLIT))
_pts = real_vs_worksheet()
print("CHECKS" + json.dumps(dict(
    nmaps=len(_m),
    pct_vis=float(_b['% of visits'].sum()), pct_hrs=float(_b['% of hours'].sum()),
    sorted_desc=bool(_b['% of visits'].is_monotonic_decreasing),
    top=str(_b.index[0]),
    main_r=float(_main.loc['r', 'nvis']), main_tot=float(_main['nvis'].sum()),
    main_m5r=float(_main.loc['r', 'm5_coadd']),
    ax_is_axes=isinstance(_ax, matplotlib.axes.Axes),
    ax2_is_axes=isinstance(_ax2, matplotlib.axes.Axes),
    cmp_cols=sorted(_cmp.columns), cmp_rows=list(_cmp.index),
    m5r_med=float(_pts.loc['m5', 'median (simulation)']),
    ddf=sorted(DDF_FIELDS),
)))
'''


@pytest.fixture(scope="module")
def nb_checks(run_notebook_checks):
    return run_notebook_checks(CHECK)


def test_part2_brief_checks(nb_checks):
    c = nb_checks
    assert c['nmaps'] > 0
    assert abs(c['pct_vis'] - 100) < 0.01
    assert abs(c['pct_hrs'] - 100) < 0.01
    assert 150 <= c['main_r'] <= 250
    assert c['ax_is_axes']                      # sparse u-band Year-1 map runs


def test_part2_extras(nb_checks):
    c = nb_checks
    assert c['sorted_desc'] and c['top'] == 'main'
    assert 650 <= c['main_tot'] <= 850          # simulation: ~720-750 total per main field
    assert 26.0 < c['main_m5r'] < 28.0
    assert c['ax2_is_axes']
    assert c['cmp_rows'] == list('ugrizy')
    assert 'yours: visits' in c['cmp_cols'] and 'real: visits' in c['cmp_cols']
    assert 23.8 < c['m5r_med'] < 24.3           # r main visits at preset points
    assert c['ddf'] == sorted(['COSMOS', 'ECDFS', 'EDFS', 'ELAIS-S1', 'XMM-LSS'])
