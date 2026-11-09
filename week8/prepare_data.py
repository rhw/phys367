#!/usr/bin/env python
"""Pre-fetch the Week 8 clustering sample: query the DP2 Object catalog, join
photo-z, and apply the MagLim-style cut, exactly as Sections 2-4 of
week8_galaxy_clustering_wtheta.ipynb do.

This is the slow, failure-prone part of the notebook at full scale (the full
~113 deg^2 region pulls millions of objects, which pushes past the ~1 million
row guideline for an in-memory LSDB join). Running it once here and handing
students the resulting parquet files means they skip straight to the mask,
randoms, and w(theta) sections.

Produces, under --output-dir (default: data/ next to this script):
  objects.parquet   extended, clean-flag objects in the region, pre-MagLim-cut
                     (used later to build the E(B-V) systematics map)
  galaxies.parquet  the final MagLim-style tomographic galaxy sample
  meta.json         the parameters used, so the notebook can check a cached
                     copy still matches its own parameters before loading it

Run with the same LSST Science Pipelines environment as the notebook kernel:
  source /opt/lsst/software/stack/loadLSST.bash && setup lsst_distrib
  python prepare_data.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import lsdb
from upath import UPath
from lsst.rsp import RSPDiscovery


def run_query(service, query):
    job = service.submit_job(query)
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR'])
    print('Job phase is', job.phase)
    if job.phase == 'ERROR':
        job.raise_if_error()
    assert job.phase == 'COMPLETED'
    table = job.fetch_result().to_table()
    job.delete()
    return table


def fetch_objects(service, params):
    band = params['band']
    columns = [
        'objectId', 'coord_ra', 'coord_dec', 'refExtendedness', 'ebv',
        '{0}_cModelMag'.format(band), '{0}_cModelMagErr'.format(band),
        '{0}_inputCount'.format(band), '{0}_cModel_flag'.format(band),
        '{0}_invalidPsfFlag'.format(band),
        '{0}_pixelFlags_saturatedCenter'.format(band),
        '{0}_pixelFlags_interpolatedCenter'.format(band),
    ]
    query = (
        "SELECT " + ", ".join(columns) + " "
        "FROM dp2.Object "
        "WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
        "CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1 "
        "AND refExtendedness = 1 "
        "AND {b}_inputCount > 0 "
        "AND {b}_cModel_flag = 0 "
        "AND {b}_invalidPsfFlag = 0 "
        "AND {b}_pixelFlags_saturatedCenter = 0 "
        "AND {b}_pixelFlags_interpolatedCenter = 0 "
        "AND {b}_cModelMag > {bright} "
        "AND {b}_cModelMag < {faint}"
    ).format(ra=params['ra_center'], dec=params['dec_center'], radius=params['radius_deg'],
             b=band, bright=params['mag_bright'], faint=params['faint_precut'])
    print(query)
    obj_df = run_query(service, query).to_pandas()
    print(len(obj_df), 'extended, clean-flag objects in the region')
    return obj_df


def join_photoz(obj_df, params):
    base_path = UPath("/rubin/lsdb_data/dp2")
    obj_df_renamed = obj_df.rename(columns={'coord_ra': 'ra', 'coord_dec': 'dec'})
    obj_cat = lsdb.from_dataframe(obj_df_renamed, catalog_name="object")

    pz_cat_full = lsdb.open_catalog(base_path / "object_photoz")
    pz_codes = sorted({c.split('_')[0] for c in pz_cat_full.columns if c.endswith('_z_best')})
    pz_code = pz_codes[0]
    pz_col = '{}_z_median'.format(pz_code)
    print('Available photo-z estimators:', pz_codes)
    print('Using', pz_col, 'as z_phot')

    pz_cat = lsdb.open_catalog(base_path / "object_photoz",
                               columns=['objectId', 'ra', 'dec', pz_col])
    pz_cat_region = pz_cat.cone_search(ra=params['ra_center'], dec=params['dec_center'],
                                       radius_arcsec=params['radius_deg'] * 3600.0)
    joined_cat = pz_cat_region.join(obj_cat, left_on='objectId', right_on='objectId',
                                    suffix_method='all_columns')
    clustering_df = joined_cat.compute()
    print(len(clustering_df), 'objects with both Object and photo-z measurements')
    return obj_df_renamed, clustering_df, pz_code, pz_col


def apply_maglim_cut(clustering_df, pz_col, params):
    band = params['band']
    i_col = '{0}_cModelMag_object'.format(band)
    z_col = '{0}_object_photoz'.format(pz_col)

    clustering_df = clustering_df.rename(columns={
        i_col: 'i_mag',
        z_col: 'z_phot',
        'ebv_object': 'ebv',
        'objectId_object': 'objectId',
        'ra_object': 'ra',
        'dec_object': 'dec',
    })

    z_edges = np.linspace(0.0, params['z_max'], params['n_zbins'] + 1)
    maglim_cut = (
        (clustering_df['i_mag'] > params['mag_bright'])
        & (clustering_df['i_mag'] < params['maglim_slope'] * clustering_df['z_phot']
           + params['maglim_intercept'])
        & (clustering_df['z_phot'] >= z_edges[0])
        & (clustering_df['z_phot'] < z_edges[-1])
    )
    galaxies = clustering_df.loc[maglim_cut,
                                 ['objectId', 'ra', 'dec', 'i_mag', 'z_phot', 'ebv']].reset_index(drop=True)
    # clustering_df comes from an LSDB .compute(), as a nested_pandas.NestedFrame; cast back
    # to a plain DataFrame so standard pandas/pyarrow round-tripping (e.g. to_parquet) works.
    galaxies = pd.DataFrame(galaxies)
    galaxies['zbin'] = np.digitize(galaxies['z_phot'], z_edges[1:-1])
    print(len(galaxies), 'galaxies pass the MagLim-style cut and lie in [{:.2f}, {:.2f})'.format(
        z_edges[0], z_edges[-1]))
    for b in range(params['n_zbins']):
        print('  bin {}: {:.2f} <= z < {:.2f}, {} galaxies'.format(
            b, z_edges[b], z_edges[b + 1], np.sum(galaxies['zbin'] == b)))
    return galaxies


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ra-center', type=float, default=345.0)
    parser.add_argument('--dec-center', type=float, default=-8.2)
    parser.add_argument('--radius-deg', type=float, default=6.0)
    parser.add_argument('--band', default='i')
    parser.add_argument('--mag-bright', type=float, default=17.5)
    parser.add_argument('--faint-precut', type=float, default=23.0)
    parser.add_argument('--maglim-slope', type=float, default=4.0)
    parser.add_argument('--maglim-intercept', type=float, default=18.0)
    parser.add_argument('--n-zbins', type=int, default=4)
    parser.add_argument('--z-max', type=float, default=1.0)
    parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    args = parser.parse_args()

    params = vars(args).copy()
    output_dir = params.pop('output_dir')
    os.makedirs(output_dir, exist_ok=True)

    discovery = RSPDiscovery("dp2")
    service = discovery.get_tap_client()

    obj_df = fetch_objects(service, params)
    obj_df_renamed, clustering_df, pz_code, pz_col = join_photoz(obj_df, params)
    galaxies = apply_maglim_cut(clustering_df, pz_col, params)

    objects_path = os.path.join(output_dir, 'objects.parquet')
    galaxies_path = os.path.join(output_dir, 'galaxies.parquet')
    meta_path = os.path.join(output_dir, 'meta.json')

    obj_df_renamed.to_parquet(objects_path, index=False)
    galaxies.to_parquet(galaxies_path, index=False)

    meta = dict(params)
    meta['pz_code'] = pz_code
    meta['pz_col'] = pz_col
    meta['n_objects'] = len(obj_df_renamed)
    meta['n_galaxies'] = len(galaxies)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print('Wrote', objects_path)
    print('Wrote', galaxies_path)
    print('Wrote', meta_path)


if __name__ == '__main__':
    main()
