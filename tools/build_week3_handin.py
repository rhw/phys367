"""Build the Week 3 hand-in notebook from a readable cell list.

Run ``python tools/build_week3_handin.py`` to (re)write ``week3/handin.ipynb`` with outputs
cleared. The notebook runs on the Rubin Science Platform (LSST kernel) next to the three
Week 3 tutorial notebooks by Eli Rykoff and Alex Broughton; it must not import anything from
tools/.
"""
from pathlib import Path
import textwrap

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "week3" / "handin.ipynb"


def _c(kind, text):
    return (kind, textwrap.dedent(text).strip("\n"))


INTRO_CELLS = [
    _c("md", """
    # Week 3 · Images → catalogs: hand-in

    On Monday (Oct 5) Alex Broughton and Eli Rykoff showed how Rubin turns images into a
    catalog: instrument signature removal, photometric and astrometric calibration, detection,
    deblending, measurement, and the selection function all of that leaves behind. Wednesday's
    (Oct 7) hands-on used three notebooks, which live in this folder:

    - **`dp2_full_sky_stars_phys367.ipynb`** (Eli): bright stars over the whole DP2 footprint,
      two star/galaxy separators, maps of star density, color, and dust.
    - **`dp2_cosmos_phys367.ipynb`** (Eli): the COSMOS deep field down to i ≈ 26: the stellar
      locus, galaxy colors, and how well the PSF model fits the stars.
    - **`qualify_a_dataset.ipynb`** (Alex): survey property maps, a coadd with its catalog and
      mask planes, blends, galaxy counts, and single-visit vs coadd depth.

    This notebook is the hand-in. **Pick one of the four exercises below**, each one a step
    from Monday's lecture, make its figure, and write the paragraph at the end. The warm-up
    is optional. Each exercise comes with a starter cell that loads what it needs and a few
    questions to steer the paragraph; the measurement and the plot are yours.

    **How to use it.** Run on the Rubin Science Platform (<https://data.lsst.cloud>) with the
    **LSST** kernel, from the same folder as the three notebooks above. Run the setup cell
    first, then go straight to your exercise. Exercises 1–3 use the COSMOS catalog Eli
    extracted for the tutorial and run in seconds. Exercise 4 uses the Butler and takes a
    few minutes per sky position. Budget one to two hours.
    """),
    _c("md", """
    ## Setup

    The COSMOS extract is the `dp2.Object` table within 1° of the COSMOS field center
    (RA 150.12°, Dec +2.21°), about 1.2 million objects with PSF, Sérsic and cModel fluxes,
    flags, moments, and both extendedness columns. It is the file Eli's COSMOS notebook loads.
    If you run somewhere else, point `DATA_DIR` at your copy.

    Magnitudes below are AB magnitudes from fluxes in nJy, $m = -2.5\\log_{10} f + 31.4$,
    the same conversion Alex's notebook uses (and the same thing Eli's `.to_value(units.ABmag)`
    does). The magnitude error is $\\sigma_m = 1.086\\,\\sigma_f / f$.
    """),
    _c("code", """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.table import Table

    DATA_DIR = os.environ.get("PHYS367_DATA", "/home/erykoff/data")

    cosmos = Table.read(os.path.join(DATA_DIR, "cosmos_object_selection.fits"))
    print(f"COSMOS extract: {len(cosmos)} objects, {len(cosmos.colnames)} columns")


    def psf_mag(band, table=cosmos):
        \"\"\"PSF AB magnitude in `band`; NaN where the flux is not positive.\"\"\"
        flux = np.asarray(table[f"{band}_psfFlux"], dtype=float)
        mag = np.full(flux.shape, np.nan)
        positive = flux > 0
        mag[positive] = -2.5 * np.log10(flux[positive]) + 31.4
        return mag


    def psf_mag_err(band, table=cosmos):
        \"\"\"Reported PSF magnitude error in `band`, 1.086 sigma_f / f.\"\"\"
        flux = np.asarray(table[f"{band}_psfFlux"], dtype=float)
        err = np.asarray(table[f"{band}_psfFluxErr"], dtype=float)
        return 1.086 * err / flux


    g, r, i = psf_mag("g"), psf_mag("r"), psf_mag("i")
    g_err, r_err, i_err = psf_mag_err("g"), psf_mag_err("r"), psf_mag_err("i")

    # Objects with a valid PSF measurement in g, r and i (as in Eli's `use` selection)
    ok = np.isfinite(g) & np.isfinite(r) & np.isfinite(i)
    for band in "gri":
        ok &= ~np.asarray(cosmos[f"{band}_invalidPsfFlag"], dtype=bool)

    ref_ext = np.asarray(cosmos["refExtendedness"], dtype=float)      # 0 = point source, 1 = extended
    model_ext = np.asarray(cosmos["griz_model_extendedness"], dtype=float)  # 0 (compact) to 1 (extended)
    star_ref = ok & (ref_ext < 0.5)

    print(f"{ok.sum()} objects with valid g, r, i PSF photometry; {star_ref.sum()} called stars by refExtendedness")
    """),
]

WARMUP_CELLS = [
    _c("md", """
    ## Warm-up (optional) · The stellar locus, colored by magnitude

    This is the exercise at the end of Eli's COSMOS notebook. His color–color plots color each
    hexagon by the *number* of stars in it. Make the same plot (g−r against r−i for the
    `star_ref` sample with `i_err < 0.05`) but color each hexagon by the *median i magnitude*
    of the stars in it. `plt.hexbin` takes a `C=` array and a `reduce_C_function`.

    What does the plot tell you about where the red and blue stars in this field are? If
    bluer stars are systematically brighter in your sample, what does that do to any test that
    bins by magnitude *or* by color, like the PSF size-bias plots in Eli's notebook?
    """),
    _c("code", """
    bright_stars = star_ref & (i_err < 0.05)

    # Your plot here: g - r against r - i, hexagons colored by the median i magnitude
    """),
]

EX1_CELLS = [
    _c("md", """
    ## Exercise 1 · Calibration: are the errors honest?

    Every flux in the catalog comes with an error. That error is *statistical*: photon noise and
    read noise, propagated through the PSF fit. Calibration residuals, PSF model errors, and
    light from neighbors are not in it. A check that needs no outside truth is the **stellar
    locus**: stars of a given type have nearly the same colors, so the width of the locus in
    g−r at fixed r−i is a ceiling on the real color error. If the catalog's errors were the
    whole story, the locus could not be narrower than they predict, and at the faint end it
    should be about as wide.

    *From the tutorial:* Eli's COSMOS notebook, "Compute colors from the PSF fluxes" and the
    first color–color plots.

    **Steps.**

    1. Take the `star_ref` sample on the blue, nearly straight part of the locus
       (0.2 < r−i < 0.9, 0.2 < g−r < 1.6). The starter cell fits a cubic g−r = f(r−i) to the
       bright stars (i < 20) and computes the residual g−r − f(r−i) for every star.
    2. In bins of i (0.5 mag wide from 16 to 25), measure the **width** of the residuals. Use
       `robust_width` (half the 16–84 percentile range), because a few outliers would dominate
       a standard deviation.
    3. In the same bins, take the median **reported** error on g−r,
       $\\sqrt{\\sigma_g^2 + \\sigma_r^2}$ (`reported_color_err`).
    4. Plot both against i on a log axis.

    **For the paragraph.**

    - At the bright end the locus is far wider than the reported error. What sets that floor?
      (Think about what varies from star to star at fixed r−i, and what the pipeline cannot
      know from one object.) Is that floor a failure of the error bars?
    - Treat the bright-end width as an intrinsic term and add it in quadrature to the reported
      error. Where does the measured width leave that curve, and by how much?
    - Before blaming the errors, look at Exercise 2: what happens to the `star_ref` sample at
      the same magnitudes? What else broadens a "stellar" locus there?
    - Which of the systematics from Monday's lecture does a catalog error, by construction,
      know nothing about?
    """),
    _c("code", """
    gmr, rmi = g - r, r - i
    on_locus = star_ref & (rmi > 0.2) & (rmi < 0.9) & (gmr > 0.2) & (gmr < 1.6)

    # Cubic fit to the bright part of the locus, then the residual for every star on it
    bright = on_locus & (i < 20)
    locus_fit = np.polyfit(rmi[bright], gmr[bright], 3)
    resid = gmr - np.polyval(locus_fit, rmi)

    reported_color_err = np.sqrt(g_err**2 + r_err**2)

    mag_bins = np.arange(16.0, 25.01, 0.5)
    mag_mids = 0.5 * (mag_bins[1:] + mag_bins[:-1])


    def robust_width(x):
        \"\"\"Half the 16-84 percentile range: equals sigma for a Gaussian, ignores outliers.\"\"\"
        lo, hi = np.nanpercentile(x, [16, 84])
        return 0.5 * (hi - lo)


    print(f"{on_locus.sum()} stars on the locus, {bright.sum()} of them brighter than i = 20")
    print("fit g-r = f(r-i), coefficients:", np.round(locus_fit, 3))

    # Your measurement here: loop over the bins of i, and for the stars in each bin record
    #   the robust width of `resid` and the median of `reported_color_err`.
    # Then plot both against `mag_mids` (log y axis).
    """),
]

EX2_CELLS = [
    _c("md", """
    ## Exercise 2 · Selection: who gets called a star?

    Star/galaxy separation is a selection function, and the catalog offers two. Eli's full-sky
    notebook compared them for bright objects (i < 18):

    - `refExtendedness` compares the PSF flux with a model flux in the reference band and
      returns 0 (point source) or 1 (extended). The notebooks use `< 0.5` for stars.
    - `griz_model_extendedness` comes from the Sérsic fit to g, r, i, z together and runs
      continuously from 0 to 1. The recommended star/galaxy threshold is 0.2.

    COSMOS lets you push the comparison to i ≈ 26, three orders of magnitude fainter.

    *From the tutorial:* Eli's full-sky notebook, "Perils of star/galaxy separation"; Alex's
    notebook, the galaxy-counts section and its note that the extendedness columns have not
    been characterized for purity or completeness.

    **Steps.**

    1. In bins of i (0.5 mag from 16 to 26), compute the fraction of objects each classifier
       calls a star: `ref_ext < 0.5` and `model_ext < 0.2`.
    2. In the same bins, compute the fraction of objects on which the two **disagree**.
    3. Plot the three curves against i.

    **For the paragraph.**

    - Over what magnitude range do the two agree? Where does each one fail, and on which side:
      calling stars galaxies, or galaxies stars?
    - At the bright end one classifier collapses. What happens to bright stars on the detector,
      and to a Sérsic fit of them?
    - At the faint end `refExtendedness` turns back up. Real star counts at high Galactic
      latitude fall smoothly with magnitude, so is that a real population? If you select faint
      "stars" with `ref_ext < 0.5`, what have you actually selected? And a faint galaxy sample
      with the complement?
    - The DP2 flag guidance says the extendedness columns have not been characterized for
      purity or completeness. What data would you need to measure those here? (COSMOS has
      Hubble imaging that resolves far more galaxies than Rubin can.)
    """),
    _c("code", """
    classified = ok & np.isfinite(model_ext)

    is_star_ref = ref_ext < 0.5
    is_star_model = model_ext < 0.2

    mag_bins = np.arange(16.0, 26.01, 0.5)
    mag_mids = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    print(f"{classified.sum()} objects with both classifiers")
    print(f"called stars: {is_star_ref[classified].sum()} by refExtendedness, "
          f"{is_star_model[classified].sum()} by griz_model_extendedness")

    # Your measurement here: for each bin of i, the fraction of `classified` objects with
    #   is_star_ref, with is_star_model, and with is_star_ref != is_star_model.
    # Then plot the three fractions against `mag_mids`.
    """),
]

EX3_CELLS = [
    _c("md", """
    ## Exercise 3 · Deblending: how much of the catalog is blended?

    Alex's notebook counted the blended objects in one patch with `detect_fromBlend` and set
    the number next to HSC's 58% in its Wide layer (Bosch et al. 2018). That fraction depends
    on depth, on seeing, and on how crowded the sky is, so one number for one patch is not the
    answer. COSMOS is a deep field: expect more.

    Two flags describe each object's history in the deblender. `detect_isIsolated` means the
    object was alone in its detection footprint. `detect_fromBlend` means its footprint held
    more than one object and the deblender split it.

    *From the tutorial:* Alex's notebook, "Add Detection Footprints" and the blend count that
    follows it.

    **Steps.**

    1. In bins of i (0.5 mag from 17 to 27), compute the fraction of objects with
       `detect_fromBlend` set.
    2. The starter cell counts each object's neighbors within 10″ (`n_neighbors`). Split the
       sample into its densest and sparsest quartiles in `n_neighbors` and repeat step 1 for
       each.
    3. Plot the three curves against i, with the HSC 58% as a reference line.

    **For the paragraph.**

    - The blend fraction *rises* toward the bright end. Why? What does a bright star's
      detection footprint look like, and who ends up inside it?
    - Is `detect_fromBlend` a statement about the object or about its footprint? When a faint
      galaxy sits in a bright star's footprint, which of the two is "blended"?
    - Which differences between HSC Wide and a DP2 deep field explain the gap from 58%?
    - About 15% of objects here are isolated. If your science needs isolated galaxies, what
      have you selected on, and where on the sky will your sample be missing?
    - *On the RSP, optional:* in Alex's notebook, move `ra, dec` to a patch at low Galactic
      latitude and rerun the blend count. How does the fraction compare with COSMOS?
    """),
    _c("code", """
    from scipy.spatial import cKDTree

    from_blend = np.asarray(cosmos["detect_fromBlend"], dtype=bool)
    isolated = np.asarray(cosmos["detect_isIsolated"], dtype=bool)
    print(f"{from_blend[ok].mean():.1%} of objects are from a blend; {isolated[ok].mean():.1%} are isolated")

    # Neighbors within 10 arcsec of each object, counted on the unit sphere
    ra_rad = np.radians(np.asarray(cosmos["coord_ra"], dtype=float))
    dec_rad = np.radians(np.asarray(cosmos["coord_dec"], dtype=float))
    xyz = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)])
    chord = 2 * np.sin(np.radians(10 / 3600) / 2)
    n_neighbors = np.array([len(hits) - 1 for hits in cKDTree(xyz).query_ball_point(xyz, chord)])

    q25, q75 = np.percentile(n_neighbors[ok], [25, 75])
    sparse = ok & (n_neighbors <= q25)
    dense = ok & (n_neighbors > q75)
    print(f"neighbors within 10 arcsec: median {np.median(n_neighbors[ok]):.0f}; "
          f"sparsest quartile <= {q25:.0f}, densest quartile > {q75:.0f}")

    mag_bins = np.arange(17.0, 27.01, 0.5)
    mag_mids = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    # Your measurement here: for each bin of i, the fraction of objects with `from_blend`,
    #   for all of `ok`, for `dense`, and for `sparse`. Skip bins with fewer than ~50 objects.
    # Then plot the three fractions against `mag_mids`, with a line at 0.58 for HSC Wide.
    """),
]

EX4_CELLS = [
    _c("md", """
    ## Exercise 4 · Depth: does the map predict the catalog?

    Section 10 of Alex's notebook compares, at one sky position, the single-visit depth
    (`magLim` from the VisitDetector table) with the coadd depth from the `psf_maglim` survey
    property map, against two Week 2 predictions: stacking N visits gains
    $1.25\\log_{10} N$, and a PSF that is wider by a factor $x$ costs $2.5\\log_{10} x$. At one
    position the leftover could be anything. At three it starts to be a measurement: is the
    mismatch random, or does it have a sign?

    The depth map is a prediction about point sources from the PSF size and the noise. The
    catalog itself tells you where it runs out: the magnitude where the histogram of PSF
    magnitudes turns over. The starter cell returns both.

    *From the tutorial:* Alex's notebook, sections 2 (maps), 8 (the S/N slider) and 10.

    **Steps.**

    1. Choose **three** sky positions in the DP2 coadd footprint: Alex's default
       (RA 187.26°, Dec +8.66°), the COSMOS center (RA 150.12°, Dec +2.21°), and one of your
       own from the footprint map in Alex's section 2 or Eli's full-sky maps. A position at low
       Galactic latitude or with very different exposure time is the most informative third
       choice.
    2. Call `depth_at(ra, dec)` for each. It returns the median single-visit depth and PSF
       FWHM, the coadd depth and PSF FWHM from the maps, the number of visits implied by the
       exposure-time map, and the turnover magnitude of the catalog in that patch.
    3. Tabulate measured gain (coadd − single visit) against predicted gain (stacking + PSF
       term), and plot one against the other, or the leftover against the number of visits.
       Add the turnover magnitude to the table.

    **For the paragraph.**

    - Does the unexplained term have the same sign at all three positions? If so, which
      assumption is systematically off: equal visits in the stack, the PSF term, or the
      exposure time divided by 30 s as a visit count?
    - `magLim` and `psf_maglim` are both 5σ point-source depths, but measured differently. Does
      that matter at the level of the leftover?
    - How far below the coadd depth does the catalog turn over, and is that offset the same at
      the three positions? What would you need before quoting the map's depth as your
      sample's depth?
    """),
    _c("code", """
    # Butler access: this cell runs on the RSP LSST kernel.
    try:
        from lsst.daf.butler import Butler
        import hpgeom as hpg
        import pandas as pd
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        butler = Butler("dp2", collections="dp2")
    except ImportError as err:
        butler = None
        print("Exercise 4 needs the Rubin Science Platform (LSST kernel):", err)

    SKYMAP = "lsst_cells_v2"
    NSIDE_COV = 32          # resolution of the coverage map
    NSIDE_NATIVE = 32768    # native resolution of the maps
    PIXEL_SCALE = 0.2       # arcsec per coadd pixel
    SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
    VISIT_EXPTIME = 30.0    # s, the nominal visit

    MAPS = {   # as in Alex's notebook; "scale" converts to the display unit
        "psf_maglim": dict(dtype="deepCoadd_psf_maglim_consolidated_map_weighted_mean", scale=1.0),
        "psf_fwhm": dict(dtype="deepCoadd_psf_size_consolidated_map_weighted_mean", scale=SIGMA_TO_FWHM * PIXEL_SCALE),
        "exposure_time": dict(dtype="deepCoadd_exposure_time_consolidated_map_sum", scale=1.0),
    }

    _visit_table = {}


    def visit_table(band):
        \"\"\"The VisitDetector table for one band, read once (Alex's section 10).\"\"\"
        if band not in _visit_table:
            ref = butler.query_datasets("visit_detector_table", instrument="LSSTCam")[0]
            columns = ["visitId", "detector", "band", "ra", "dec", "magLim", "psfSigma"]
            vd = butler.get(ref, parameters={"columns": columns})
            vd = vd.to_pandas() if hasattr(vd, "to_pandas") else vd
            _visit_table[band] = vd[vd["band"] == band]
        return _visit_table[band]


    def map_median_near(key, ra, dec, band, radius_deg=0.1, nside=4096):
        \"\"\"Median of a survey property map within `radius_deg` of (ra, dec), in display units.\"\"\"
        dtype = MAPS[key]["dtype"]
        coverage = butler.get(f"{dtype}.coverage", band=band, skymap=SKYMAP)
        near = [int(p) for p in hpg.query_circle(NSIDE_COV, ra, dec, 0.5, inclusive=True) if coverage.coverage_mask[p]]
        hsp_map = butler.get(dtype, band=band, skymap=SKYMAP,
                             parameters={"pixels": near, "degrade_nside": nside})
        pixels = hpg.query_circle(nside, ra, dec, radius_deg, nest=True)
        values = np.asarray(hsp_map.get_values_pix(pixels), dtype=float)
        values = values[values > -1e29]
        return float(np.nanmedian(values)) * MAPS[key]["scale"] if values.size else np.nan


    def turnover_mag(ra, dec, band):
        \"\"\"Magnitude where the histogram of PSF magnitudes in the patch at (ra, dec) peaks (Alex's section 8).\"\"\"
        refs = butler.query_datasets("deep_coadd", where="band.name = :band AND patch.region OVERLAPS POINT(:ra, :dec)",
                                     bind={"band": band, "ra": ra, "dec": dec}, order_by="patch")
        data_id = refs[0].dataId
        cat = butler.get("object", tract=data_id["tract"],
                         parameters={"columns": ["patch", f"{band}_psfFlux"]})
        flux = np.asarray(cat[f"{band}_psfFlux"][cat["patch"] == data_id["patch"]], dtype=float)
        mags = -2.5 * np.log10(flux[flux > 0]) + 31.4
        counts, edges = np.histogram(mags, bins=np.arange(16, 30.01, 0.25))
        return 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])


    def depth_at(ra, dec, band="i"):
        \"\"\"Single-visit and coadd depth at one position, with the Week 2 predictions.\"\"\"
        vd = visit_table(band)
        sep = SkyCoord(vd["ra"].values * u.deg, vd["dec"].values * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg))
        near = vd[sep < 0.1 * u.deg]
        out = {
            "ra": ra, "dec": dec, "band": band,
            "n_detector_images": len(near),
            "visit_depth": near["magLim"].median(),
            "visit_fwhm": near["psfSigma"].median() * SIGMA_TO_FWHM * PIXEL_SCALE,
            "coadd_depth": map_median_near("psf_maglim", ra, dec, band),
            "coadd_fwhm": map_median_near("psf_fwhm", ra, dec, band),
            "n_visits": map_median_near("exposure_time", ra, dec, band) / VISIT_EXPTIME,
            "turnover_mag": turnover_mag(ra, dec, band),
        }
        out["gain_measured"] = out["coadd_depth"] - out["visit_depth"]
        out["gain_stacking"] = 1.25 * np.log10(out["n_visits"])
        out["gain_psf"] = -2.5 * np.log10(out["coadd_fwhm"] / out["visit_fwhm"])
        out["unexplained"] = out["gain_measured"] - out["gain_stacking"] - out["gain_psf"]
        return out


    positions = [(187.2645519, 8.6612496), (150.11916667, 2.20583333)]   # add a third of your own

    # Your measurement here, e.g.
    #   rows = [depth_at(ra, dec) for ra, dec in positions]
    #   table = pd.DataFrame(rows).round(2)
    # Then a plot: measured gain against predicted gain, or `unexplained` against `n_visits`.
    """),
]

HANDIN_CELLS = [
    _c("md", """
    ## Hand-in

    Due Monday Oct 12 on Canvas. Turn in: (1) one figure from the exercise you chose;
    (2) one paragraph: what did you measure, what does it say about the catalog, and what
    would it mean for your team's science case?; (3) a sentence on any AI tools you used.
    Upload the .ipynb or a PDF. Graded complete/incomplete.

    Make your figure in the code cell below. Write your paragraph and the AI-tools sentence in
    the last cell.
    """),
    _c("code", """
    # Your hand-in figure
    """),
    _c("md", """
    *Your paragraph here.*
    """),
]


def _make_notebook(cells_src):
    cells = []
    for kind, text in cells_src:
        if kind == "md":
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "lsst", "display_name": "LSST", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def build_notebook():
    return _make_notebook(INTRO_CELLS + WARMUP_CELLS + EX1_CELLS + EX2_CELLS + EX3_CELLS
                          + EX4_CELLS + HANDIN_CELLS)


def build(path=NB):
    """Write the hand-in notebook (outputs cleared) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), path)
    return path


if __name__ == "__main__":
    print(build())
