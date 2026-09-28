import numpy as np, pandas as pd
from tools.make_baseline_extract import coadd_m5, category, visits_near, pixel_maps

def test_coadd_two_equal_visits_gain_0p376():
    assert abs(coadd_m5(np.array([24.0, 24.0])) - (24.0 + 1.25*np.log10(2))) < 1e-9

def test_coadd_empty_is_nan():
    assert np.isnan(coadd_m5(np.array([])))

def test_category():
    assert category('pairs_iz_33.0_40.0') == 'main'
    assert category('ddf_cosmos') == 'DDF'
    assert category('too_GW_case_B_C_496_i4') == 'ToO'
    assert category('twilight_near_sun') == 'twilight NEO'
    assert category('rges_onseason') == 'Roman bulge'
    assert category('something_new') == 'other'

def _obs(ra, dec, band='r', m5=24.0, mjd=61210.0, night=1):
    return pd.DataFrame(dict(fieldRA=ra, fieldDec=dec, band=band, fiveSigmaDepth=m5,
                             observationStartMJD=mjd, night=night))

def test_visits_near_wraps_ra():
    obs = _obs([359.5, 0.5, 180.0], [0.0, 0.0, 0.0])
    assert len(visits_near(obs, 0.0, 0.0)) == 2

def test_visits_near_pole():
    obs = _obs([0.0, 180.0], [-89.0, -89.0])      # 2 deg apart across the pole
    assert len(visits_near(obs, 90.0, -90.0)) == 2

def test_pixel_maps_empty_band_nan_and_counts():
    obs = _obs([10.0]*3, [-30.0]*3, band=['r','r','g'], m5=[24.0,24.0,25.0],
               mjd=[61210.,61211.,61212.], night=[1,2,3])
    m = pixel_maps(obs, nside=16, radius=1.75, year1_mjd_end=61211.5)
    r = m[(m.band=='r')]
    assert (r.nvis == 2).all() and (r.nvis_y1 == 2).all()
    g = m[(m.band=='g')]
    assert (g.nvis_y1 == 0).all() and g.m5_coadd_y1.isna().all()
