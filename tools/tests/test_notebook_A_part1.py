"""Task 3: Notebook A, Part 1 ("your survey with sliders")."""
import pytest

CHECK = r'''
import json
p = plan_survey(18000, 30, DEFAULT_SPLIT)
x = plan_survey(30000, 5, {'u':0,'g':.2,'r':.2,'i':.2,'z':.2,'y':.2})
t = plan_survey(35000, 120, DEFAULT_SPLIT)
print("CHECKS" + json.dumps(dict(m5r=m5_single('r',30), n=p['n_visits'], ngal=p['n_gal'],
      iwarn=len(x['warnings']), m5u_nan=bool(x['m5_coadd']['u'] != x['m5_coadd']['u']),
      rev=p['revisit_days'],
      m5r60=m5_single('r', 60), nu_zero=x['n_by_band']['u'],
      nsn=p['n_sn_detected'], pairs=p['pairs_per_15d'],
      twarn=len(t['warnings']), pwarn=len(p['warnings']),
      warn_types=[type(w).__name__ for w in x['warnings']],
      keys=sorted(p.keys()))))
'''


@pytest.fixture(scope="module")
def nb_checks(run_notebook_checks):
    return run_notebook_checks(CHECK)


def test_part1_checks(nb_checks):
    c = nb_checks
    assert abs(c['m5r'] - 24.7) < 1e-9
    assert 800 < c['n'] < 850                    # worksheet: ~825
    assert 3.0e9 < c['ngal'] < 4.5e9             # Science Book: ~4 billion (gold sample)
    assert c['iwarn'] >= 2 and c['m5u_nan']
    assert abs(c['rev'] - 3.0) < 1e-9


def test_part1_extras(nb_checks):
    c = nb_checks
    assert abs(c['m5r60'] - (24.7 + 1.25 * 0.30103)) < 1e-4
    assert c['nu_zero'] == 0
    assert c['nsn'] > 1e5                        # upper bound; bigger than Marshall's ~1e5
    assert c['pairs'] > 0
    assert c['twarn'] >= 2                       # trailing + area beyond Chile
    assert c['pwarn'] == 0                       # LSST defaults raise no warnings
    assert all(t == 'str' for t in c['warn_types'])
    assert c['keys'] == sorted(['n_visits', 'n_by_band', 'm5_single', 'm5_coadd',
                                'revisit_days', 'n_gal', 'n_sn_detected',
                                'pairs_per_15d', 'warnings'])
