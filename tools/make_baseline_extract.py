"""Baseline v5.3.3 extract for Phys 367 Week 2 Notebook A.

Reads the Rubin baseline v5.3.3 10-year survey-simulation SQLite database
(``observations`` table) and writes three small data files under
``week2/data/`` for the Colab notebook to load:

- ``baseline_v5.3.3_maps.parquet``: healpix pixel maps (nside=64), one row
  per (hpix, band). Columns: hpix, ra, dec, band, nvis, m5_coadd, nvis_y1,
  m5_coadd_y1, median_night_gap. Pixels a band never visits are simply
  absent (no rows) -- the notebook reindexes and fills NaN before plotting.
  Stored as float32 for the float columns to keep file size small.
- ``baseline_v5.3.3_points.parquet``: per-visit rows for a curated list of
  sky-position presets (``PRESETS``), for time-series / cadence plots.
  Columns: preset (short PRESETS name), mjd, night, band, m5 (fiveSigmaDepth),
  seeing (seeingFwhmEff), airmass, sky (skyBrightness), category.
- ``baseline_v5.3.3_budget.csv``: rows = observation category, columns =
  per-band visit counts, total visits, and open-shutter hours (sum of
  visitExposureTime / 3600).

A companion file, ``week2/data/presets.csv``, holds the PRESETS metadata
(name, ra, dec, blurb) since a small presets table inside the points
parquet would be awkward to join against in the notebook.

Column short-name glossary (points parquet):
    preset   -- PRESETS short name (e.g. "COSMOS", "main-1")
    mjd      -- observationStartMJD
    night    -- integer night number
    band     -- observing band (ugrizy)
    m5       -- fiveSigmaDepth (5-sigma point-source depth, mag)
    seeing   -- seeingFwhmEff (effective seeing FWHM, arcsec)
    airmass  -- airmass at observation
    sky      -- skyBrightness (mag/arcsec^2)
    category -- category(observation_reason), see `category()` below

Visit footprint: a circle of radius 1.75 deg around (fieldRA, fieldDec);
chip gaps are ignored (see globals.md).

Coadded depth: m5_coadd = 1.25 * log10( sum_i 10**(0.8 * m5_i) ).
"""

import argparse
import pathlib
import sqlite3

import healpy as hp
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Presets: (name, ra, dec, blurb)
# ---------------------------------------------------------------------------
PRESETS = [
    # main survey (WFD) representative pointings
    ("main-1", 150.0, -30.0, "Main survey (WFD) field near the equator"),
    ("main-2", 30.0, -45.0, "Main survey (WFD) field in the southern sky"),
    ("main-3", 330.0, -20.0, "Main survey (WFD) field in the southern sky"),
    # Deep Drilling Fields
    ("COSMOS", 150.10, 2.18, "COSMOS Deep Drilling Field"),
    ("ECDFS", 53.13, -28.10, "Extended Chandra Deep Field South DDF"),
    ("EDFS", 58.90, -49.32, "Euclid Deep Field South DDF"),
    ("ELAIS-S1", 9.45, -44.00, "ELAIS-S1 Deep Drilling Field"),
    ("XMM-LSS", 35.71, -4.75, "XMM-LSS Deep Drilling Field"),
    # Galactic
    ("Gal-center", 266.42, -29.01, "Galactic centre"),
    ("Gal-plane-l300", 187.4, -62.8, "Galactic plane at l=300 deg"),
    # Local Group / nearby
    ("LMC", 80.89, -69.76, "Large Magellanic Cloud"),
    ("47-Tuc", 6.02, -72.08, "47 Tucanae globular cluster"),
    ("Fornax-dSph", 39.99, -34.45, "Fornax dwarf spheroidal"),
    # survey edges
    ("ecliptic", 0.0, 0.0, "Point on the ecliptic (RA=0, Dec=0)"),
    ("north-edge", 180.0, 25.0, "Northern edge of the main survey footprint"),
    ("south-polar-cap", 0.0, -85.0, "South polar cap"),
    # lensed quasars
    ("RXJ1131-1231", 172.96, -12.53, "Lensed quasar RXJ1131-1231"),
    ("HE0435-1223", 69.56, -12.29, "Lensed quasar HE0435-1223"),
]

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def coadd_m5(m5):
    """Coadded 5-sigma depth from an array of single-visit m5 values.

    m5_coadd = 1.25 * log10( sum_i 10**(0.8 * m5_i) ). NaN for empty input.
    """
    m5 = np.asarray(m5, float)
    return np.nan if m5.size == 0 else 1.25 * np.log10(np.sum(10 ** (0.8 * m5)))


_CATS = [
    ("pairs", "main"),
    ("singles", "main"),
    ("triplet", "main"),
    ("template", "templates"),
    ("ddf", "DDF"),
    ("too", "ToO"),
    ("twilight", "twilight NEO"),
    ("rges", "Roman bulge"),
]


def category(reason):
    """Map an observation_reason string to a short category, by prefix."""
    for p, c in _CATS:
        if reason.startswith(p):
            return c
    return "other"


def _unitvec(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def visits_near(obs, ra, dec, radius=1.75):
    """Rows of `obs` within `radius` degrees of (ra, dec), by angular separation."""
    v = _unitvec(obs.fieldRA.to_numpy(), obs.fieldDec.to_numpy())
    return obs[v @ _unitvec(ra, dec) >= np.cos(np.radians(radius))]


def pixel_maps(obs, nside=64, radius=1.75, year1_mjd_end=None):
    """Per-(healpix pixel, band) visit-count and coadded-depth maps.

    Columns: hpix, ra, dec, band, nvis, m5_coadd, nvis_y1, m5_coadd_y1,
    median_night_gap. Pixels a band never visits are absent (no rows).
    """
    if year1_mjd_end is None:
        year1_mjd_end = obs.observationStartMJD.min() + 365.25
    vecs = _unitvec(obs.fieldRA.to_numpy(), obs.fieldDec.to_numpy())
    pix, idx = [], []
    for i, v in enumerate(vecs):  # ~1.9M visits; about a minute
        p = hp.query_disc(nside, v, np.radians(radius))
        pix.append(p)
        idx.append(np.full(p.size, i))
    pix, idx = np.concatenate(pix), np.concatenate(idx)
    e = pd.DataFrame(
        dict(
            hpix=pix,
            band=obs.band.to_numpy()[idx],
            m5=obs.fiveSigmaDepth.to_numpy()[idx],
            y1=obs.observationStartMJD.to_numpy()[idx] < year1_mjd_end,
            night=obs.night.to_numpy()[idx],
        )
    )
    e["f"] = 10 ** (0.8 * e.m5)
    e["fy1"] = np.where(e.y1, e.f, 0.0)
    g = (
        e.groupby(["hpix", "band"])
        .agg(nvis=("f", "size"), fsum=("f", "sum"), nvis_y1=("y1", "sum"), fsum_y1=("fy1", "sum"))
        .reset_index()
    )
    with np.errstate(divide="ignore"):
        g["m5_coadd"] = np.where(g.fsum > 0, 1.25 * np.log10(g.fsum), np.nan)
        g["m5_coadd_y1"] = np.where(g.fsum_y1 > 0, 1.25 * np.log10(g.fsum_y1), np.nan)
    gaps = (
        e[["hpix", "night"]]
        .drop_duplicates()
        .sort_values(["hpix", "night"])
        .assign(gap=lambda d: d.groupby("hpix").night.diff())
        .groupby("hpix")
        .gap.median()
        .rename("median_night_gap")
    )
    g = g.merge(gaps, on="hpix", how="left")
    theta, phi = hp.pix2ang(nside, g.hpix.to_numpy())
    g["ra"], g["dec"] = np.degrees(phi), 90 - np.degrees(theta)
    g["nvis_y1"] = g.nvis_y1.astype(int)
    return g[
        ["hpix", "ra", "dec", "band", "nvis", "m5_coadd", "nvis_y1", "m5_coadd_y1", "median_night_gap"]
    ]


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

_OBS_COLUMNS = [
    "fieldRA",
    "fieldDec",
    "observationStartMJD",
    "night",
    "band",
    "fiveSigmaDepth",
    "seeingFwhmEff",
    "airmass",
    "skyBrightness",
    "visitExposureTime",
    "observation_reason",
]


def _load_observations(db_path):
    cols = ", ".join(_OBS_COLUMNS)
    with sqlite3.connect(db_path) as con:
        obs = pd.read_sql_query(f"SELECT {cols} FROM observations", con)
    obs["category"] = obs.observation_reason.map(category)
    return obs


def _build_budget(obs):
    """category x band visit counts, total visits, open-shutter hours."""
    counts = obs.pivot_table(index="category", columns="band", values="fieldRA", aggfunc="size", fill_value=0)
    counts = counts.astype(int)
    counts["total_visits"] = counts.sum(axis=1)
    hours = obs.groupby("category").visitExposureTime.sum() / 3600.0
    counts["open_shutter_hours"] = hours
    return counts.reset_index()


def _build_points(obs):
    frames = []
    for name, ra, dec, _blurb in PRESETS:
        near = visits_near(obs, ra, dec)
        if len(near) == 0:
            continue
        frames.append(
            pd.DataFrame(
                dict(
                    preset=name,
                    mjd=near.observationStartMJD.to_numpy(),
                    night=near.night.to_numpy(),
                    band=near.band.to_numpy(),
                    m5=near.fiveSigmaDepth.to_numpy(),
                    seeing=near.seeingFwhmEff.to_numpy(),
                    airmass=near.airmass.to_numpy(),
                    sky=near.skyBrightness.to_numpy(),
                    category=near.category.to_numpy(),
                )
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["preset", "mjd", "night", "band", "m5", "seeing", "airmass", "sky", "category"]
    )


def _write_presets_csv(out_dir):
    df = pd.DataFrame(PRESETS, columns=["name", "ra", "dec", "blurb"])
    df.to_csv(out_dir / "presets.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=str, help="Path to baseline_v5.3.3_10yrs.db")
    parser.add_argument("out_dir", type=str, help="Output directory (e.g. week2/data)")
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = _load_observations(args.db_path)

    maps = pixel_maps(obs, nside=64, radius=1.75)
    float_cols = ["ra", "dec", "m5_coadd", "m5_coadd_y1", "median_night_gap"]
    for c in float_cols:
        maps[c] = maps[c].astype(np.float32)
    maps.to_parquet(out_dir / "baseline_v5.3.3_maps.parquet", index=False)

    points = _build_points(obs)
    points.to_parquet(out_dir / "baseline_v5.3.3_points.parquet", index=False)

    budget = _build_budget(obs)
    budget.to_csv(out_dir / "baseline_v5.3.3_budget.csv", index=False)

    _write_presets_csv(out_dir)

    print(f"Wrote maps ({len(maps)} rows), points ({len(points)} rows), "
          f"budget ({len(budget)} rows), presets ({len(PRESETS)} rows) to {out_dir}")


if __name__ == "__main__":
    main()
