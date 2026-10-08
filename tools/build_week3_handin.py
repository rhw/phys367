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
    from Monday's lecture, make its figure, and write the paragraph at the end. One exercise is
    the whole assignment; the warm-up and the *Go further* boxes are extra. Each exercise comes
    with a starter cell that loads what it needs and a few questions to steer the paragraph;
    the measurement and the plot are yours.

    Exercises 1–3 run from a catalog file and take a minute or two. Exercise 4 is the
    adventurous choice: it uses the Butler, takes a few minutes per sky position, and is the
    one most exposed to a busy Science Platform.

    **How to use it.** Run on the Rubin Science Platform (<https://data.lsst.cloud>) with the
    **LSST** kernel, from the same folder as the three notebooks above. Run the setup cell,
    then the starter cell of your exercise, then write your code right below it. Each starter
    cell defines its own `mag_bins`, so run it immediately before your own code. Budget one to
    two hours.
    """),
    _c("md", """
    **Words you will meet.** *PSF flux*: the flux from fitting the point-spread function at the
    object's position; the right measure for stars. *Sérsic* and *cModel* flux: fluxes from
    fitting galaxy profiles; better for galaxies. *Extendedness*: a 0-to-1 score of how much
    bigger than the PSF an object is; the catalog has two versions (Exercise 2). *Reference
    band*: the one band whose detection sets each object's position and shape. *Second
    moments* ($I_{xx}, I_{yy}$): the size of an object's light distribution, with
    $T = I_{xx} + I_{yy}$ its area-like size. *Detection footprint*: the connected set of pixels
    above threshold that an object was found in. *Deblender*: the code that splits one footprint
    with several peaks into several objects. *Coadd*: the stacked image of all visits; *tract*
    and *patch* are the sky tiles it is cut into. *Butler*: the Rubin data access layer.
    *Survey property map*: a sky map of something like depth or PSF size, in HEALPix pixels of
    resolution `nside`. *5σ point-source depth*: the magnitude at which a star is detected at
    signal-to-noise 5. Fluxes are in nanojansky (nJy).
    """),
    _c("md", """
    ## Setup

    The COSMOS extract is the `dp2.Object` table within 1° of the COSMOS field center
    (RA 150.12°, Dec +2.21°): about 1.2 million objects with PSF, Sérsic and cModel fluxes,
    flags, moments, and both extendedness columns. It is the 530 MB file Eli's COSMOS notebook
    loads, in a directory every RSP user can read. If you run somewhere else, point `DATA_DIR`
    at your copy (or set the `PHYS367_DATA` environment variable).

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

    # Objects with a valid PSF measurement in g, r and i (like Eli's `use` selection, but for
    # these three bands only, so the sample is a little larger than his)
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
    hexagon by the *number* of stars in it. Plot g−r against r−i for the `star_ref` sample
    with `i_err < 0.05`, but color each hexagon by the *median i magnitude* of the stars in it.
    `plt.hexbin` takes a `C=` array and a `reduce_C_function`. (Eli's bright cut used an i
    error that was accidentally divided by the u-band flux, so his sample differs slightly from
    this one.)

    What does the plot tell you about where the red and blue stars in this field are? If
    bluer stars are systematically brighter in your sample, what does that do to any test that
    bins by magnitude *or* by color, like the PSF size-bias plots in Eli's notebook?

    The warm-up alone is not a complete hand-in.
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
    locus**: stars of a given type have nearly the same colors, so the scatter of g−r about
    the locus at fixed r−i is an upper bound on the real color error. If the catalog's errors
    were the whole story, the scatter could not be smaller than they predict, and at the faint
    end it should be about as large.

    *From the tutorial:* Eli's COSMOS notebook, the code cell commented "Compute colors from
    the PSF fluxes" and the color–color plots that follow it.

    **Steps.**

    1. Take the `star_ref` sample on the blue, nearly straight part of the locus
       (0.2 < r−i < 0.9, 0.2 < g−r < 1.6). The starter cell fits a cubic g−r = f(r−i) to the
       bright stars (i < 20) and computes the residual g−r − f(r−i) for every star.
    2. In bins of i (0.5 mag wide from 16 to 25), measure the **width** of the residuals for
       the stars in `on_locus`. Use `robust_width` (half the 16–84 percentile range), because
       a few outliers would dominate a standard deviation. The starter cell shows the one-line
       pattern; the result is one number per entry of `mag_mids`.
    3. In the same bins, take the median **reported** error on the residual,
       `reported_resid_err`. This is not just $\\sqrt{\\sigma_g^2 + \\sigma_r^2}$: the residual
       is taken about a sloped curve, so the r−i error enters through the local slope $s$ of
       the fit and $\\sigma_r$ appears in both colors. Propagated,
       $\\sigma^2 = \\sigma_g^2 + (1+s)^2\\sigma_r^2 + s^2\\sigma_i^2$. The starter cell computes it.
    4. Plot both against i on a log axis.

    **For the paragraph.**

    - At the bright end the measured width is far above the reported error. What sets that
      bright-end width? (Think about what varies from star to star at fixed r−i: metallicity,
      surface gravity, unresolved binaries; and what the pipeline cannot know from one object:
      calibration residuals across the field.) Is that width a failure of the error bars?
    - Take the median width in the i = 17–19 bins as an intrinsic term $\\sigma_0$ and add a
      third curve to your plot, $\\sqrt{\\sigma_0^2 + \\sigma_\\mathrm{rep}^2}$. At what
      magnitude does the measured width rise above it, and by what factor at i = 23?
    - You do not need to run Exercise 2 for this, but reason about it: past i ≈ 23 a growing
      share of the `star_ref` objects are not stars. What does that do to a "stellar" locus?
      The color box in step 1 also clips the residuals once the scatter exceeds about 0.3 mag,
      so the faintest bin or two are biased narrow.
    - Which of these does a catalog error, by construction, know nothing about: photon noise,
      zero-point calibration errors, PSF model errors, blending, background subtraction?

    > **Go further.** Split the locus stars with `detect_isIsolated` into isolated and blended,
    > and measure the width separately for each. If a neighbor's light is in the PSF flux, which
    > sample should be wider, and by how much? Or fit $\\sigma^2 = \\sigma_0^2 + (a\\,\\sigma_\\mathrm{rep})^2$
    > and report $a$: the single number by which the catalog errors would have to be scaled.
    """),
    _c("code", """
    gmr, rmi = g - r, r - i
    on_locus = star_ref & (rmi > 0.2) & (rmi < 0.9) & (gmr > 0.2) & (gmr < 1.6)

    # Cubic fit to the bright part of the locus, then the residual for every star on it
    bright = on_locus & (i < 20)
    locus_fit = np.polyfit(rmi[bright], gmr[bright], 3)
    resid = gmr - np.polyval(locus_fit, rmi)

    # Reported error on the residual, propagated through the local slope of the fit
    slope = np.polyval(np.polyder(locus_fit), rmi)
    reported_resid_err = np.sqrt(g_err**2 + (1 + slope)**2 * r_err**2 + slope**2 * i_err**2)

    mag_bins = np.arange(16.0, 25.01, 0.5)
    mag_mids = 0.5 * (mag_bins[1:] + mag_bins[:-1])


    def robust_width(x):
        \"\"\"Half the 16-84 percentile range: equals sigma for a Gaussian, ignores outliers.\"\"\"
        lo, hi = np.nanpercentile(x, [16, 84])
        return 0.5 * (hi - lo)


    print(f"{on_locus.sum()} stars on the locus, {bright.sum()} of them brighter than i = 20")
    print("fit g-r = f(r-i), coefficients:", np.round(locus_fit, 3))

    # The pattern for one binned quantity (one value per entry of mag_mids):
    #   width = [robust_width(resid[on_locus & (i >= lo) & (i < hi)])
    #            for lo, hi in zip(mag_bins[:-1], mag_bins[1:])]
    # Your measurement here: the width of `resid` and the median of `reported_resid_err`
    #   in each bin, then both against `mag_mids` on a log y axis.
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
      continuously from 0 to 1. The recommended star/galaxy threshold is 0.2, with Eli's
      caveat: it is calibrated in well-measured fields and only exists for objects detected
      in all four of g, r, i, z.

    COSMOS lets you push the comparison to i ≈ 26, three orders of magnitude fainter.

    *From the tutorial:* Eli's full-sky notebook, "Perils of star/galaxy separation"; Alex's
    notebook, the galaxy-counts section and its note that the extendedness columns have not
    been characterized for purity or completeness.

    **Steps.**

    1. Among the objects in `classified` (both classifiers defined), in bins of i (0.5 mag
       from 16 to 26), compute the fraction each classifier calls a star: `is_star_ref` and
       `is_star_model`. Same one-line pattern as Exercise 1.
    2. In the same bins, compute the fraction on which the two **disagree**.
    3. Plot the three curves against i.

    **For the paragraph.**

    - Over what magnitude range do the two agree? Where does each one fail, and on which side:
      calling stars galaxies, or galaxies stars?
    - What happens at the bright end, and to which classifier? What happens to a bright star
      on the detector, and to a Sérsic fit of it?
    - What happens to the `refExtendedness` star fraction at the faint end? Real star counts
      at high Galactic latitude fall smoothly with magnitude. Is what you see a real
      population? If you select faint "stars" with `ref_ext < 0.5`, what have you actually
      selected? And a faint galaxy sample with the complement?
    - The starter cell prints how many objects lack one of the two classifications. Does
      dropping them bias any of your bins?
    - The DP2 flag guidance says the extendedness columns have not been characterized for
      purity or completeness. What data would you need to measure those here? (COSMOS has
      Hubble imaging that resolves far more galaxies than Rubin can.)

    > **Go further.** Test the faint "stars" directly. The catalog has second moments for each
    > object and for the PSF at its position: `i_ixx + i_iyy` is the object's size $T$ and
    > `i_ixxPSF + i_iyyPSF` the PSF's. For objects with `ref_ext < 0.5` in bins of i, plot the
    > distribution of $T_\\mathrm{obj}/T_\\mathrm{PSF}$. A true star sits at 1. What fraction of
    > the i > 24 "stars" do?
    """),
    _c("code", """
    classified = ok & np.isfinite(ref_ext) & np.isfinite(model_ext)

    is_star_ref = ref_ext < 0.5
    is_star_model = model_ext < 0.2

    mag_bins = np.arange(16.0, 26.01, 0.5)
    mag_mids = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    print(f"{classified.sum()} objects with both classifiers; {(ok & ~classified).sum()} valid objects lack one of them")
    print(f"called stars: {is_star_ref[classified].sum()} by refExtendedness, "
          f"{is_star_model[classified].sum()} by griz_model_extendedness")

    # Your measurement here: for each bin of i, the fraction of `classified` objects with
    #   is_star_ref, with is_star_model, and with is_star_ref != is_star_model
    #   (one value per entry of mag_mids), then the three against `mag_mids`.
    """),
]

EX3_CELLS = [
    _c("md", """
    ## Exercise 3 · Deblending: how much of the catalog is blended?

    Alex's notebook counted the blended objects in one patch with `detect_fromBlend` and set
    the number next to HSC's: 58% of all detected objects in the HSC-SSP Wide layer (depth
    i ≈ 26) came from blended footprints (Bosch et al. 2018). That fraction depends on depth,
    on seeing, and on how crowded the sky is, so one number for one patch is not the answer.
    COSMOS is a deep field: expect more.

    Two flags describe each object's history in the deblender. `detect_isIsolated` means the
    object was alone in its detection footprint. `detect_fromBlend` means its footprint held
    more than one object and the deblender split it.

    *From the tutorial:* Alex's notebook, "Add Detection Footprints" and the blend count that
    follows it.

    **Steps.**

    1. In bins of i (0.5 mag from 17 to 27), compute the fraction of `ok` objects with
       `from_blend` set. Same one-line pattern as Exercise 1; set bins with fewer than about
       50 objects to `np.nan` so they drop out of the plot.
    2. The starter cell counts each object's neighbors within 1′ (`n_neighbors`), a crowding
       measure on a scale much larger than a footprint. Split the sample into its densest and
       sparsest quartiles in `n_neighbors` and repeat step 1 for each. (Integer counts have
       ties, so the two subsets will not be exactly 25% each; that is fine.)
    3. Plot the three curves against i, with the HSC 58% as a reference line.

    **For the paragraph.**

    - The starter cell prints the overall blended and isolated fractions. Are they what you
      expected after Alex's patch and the HSC number? Which differences between HSC Wide and
      a DP2 deep field would move it?
    - Which way does the blend fraction run with magnitude? What does a bright star's detection
      footprint look like, and who ends up inside it?
    - Is `detect_fromBlend` a statement about the object or about its footprint? When a faint
      galaxy sits in a bright star's footprint, which of the two is "blended"?
    - How much does arcminute-scale crowding move the blend fraction, compared with magnitude?
      What would happen if you counted neighbors within 10″ instead, and why would that be
      circular?
    - If your science needs isolated galaxies, what have you selected on, and where on the sky
      will your sample be missing?

    > **Go further.** In Alex's notebook, move `ra, dec` to a patch at low Galactic latitude
    > (or any crowded spot from Eli's star-density map) and rerun the blend count of section 5.
    > How does the fraction compare with COSMOS at the same magnitude, and which of depth,
    > seeing, and crowding moved it?
    """),
    _c("code", """
    from scipy.spatial import cKDTree

    from_blend = np.asarray(cosmos["detect_fromBlend"], dtype=bool)
    isolated = np.asarray(cosmos["detect_isIsolated"], dtype=bool)
    print(f"{from_blend[ok].mean():.1%} of objects are from a blend; {isolated[ok].mean():.1%} are isolated")

    # Neighbors within 1 arcmin of each object, counted on the unit sphere
    ra_rad = np.radians(np.asarray(cosmos["coord_ra"], dtype=float))
    dec_rad = np.radians(np.asarray(cosmos["coord_dec"], dtype=float))
    xyz = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)])
    chord = 2 * np.sin(np.radians(60 / 3600) / 2)
    n_neighbors = cKDTree(xyz).query_ball_point(xyz, chord, return_length=True) - 1

    q25, q75 = np.percentile(n_neighbors[ok], [25, 75])
    sparse = ok & (n_neighbors <= q25)
    dense = ok & (n_neighbors > q75)
    print(f"neighbors within 1 arcmin: median {np.median(n_neighbors[ok]):.0f}; "
          f"sparsest quartile <= {q25:.0f} ({sparse.sum()} objects), densest quartile > {q75:.0f} ({dense.sum()} objects)")

    mag_bins = np.arange(17.0, 27.01, 0.5)
    mag_mids = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    # Your measurement here: for each bin of i, the fraction of objects with `from_blend`,
    #   for all of `ok`, for `dense`, and for `sparse` (np.nan where a bin has < ~50 objects),
    #   then the three against `mag_mids`, with a line at 0.58 for HSC Wide.
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
    catalog itself tells you where it runs out: the magnitude where the histogram of i-band
    PSF magnitudes in the patch peaks. That turnover is a completeness proxy for whatever is
    in the catalog, mostly galaxies, not a 5σ point-source depth, so it need not sit on either
    side of the map value. The starter cell returns both.

    *From the tutorial:* Alex's notebook, sections 2 (maps), 8 (the S/N slider) and 10.

    **If the starter cell prints an ImportError**, you are not on the LSST kernel: switch
    kernels (Kernel → Change Kernel) and rerun from the setup cell. Each `depth_at` call takes
    a few minutes and the results are cached in `depth_cache.json` next to this notebook, so a
    restarted kernel does not cost you the positions you already have.

    **Steps.**

    1. Choose **three** sky positions in the DP2 coadd footprint. Two are given: Alex's default
       (RA 187.26°, Dec +8.66°) and the COSMOS center (RA 150.12°, Dec +2.21°). For the third,
       run the footprint map cell in section 2 of Alex's notebook (it takes about half a
       minute) and pick any point where the map has data; a spot at low Galactic latitude, for
       instance around RA 240°, Dec −25° in the dense region of Eli's star map, gives the most
       contrast. Check it on the footprint map before you commit minutes to it.
    2. Call `depth_at(ra, dec)` for each. It returns the median single-visit depth and PSF
       FWHM, the coadd depth and PSF FWHM from the maps, the number of visits implied by the
       exposure-time map, and the turnover magnitude of the catalog in that patch.
    3. **Your figure:** measured gain (coadd − single visit) against predicted gain (stacking
       + PSF term), one point per position, with a 1:1 line. Print the table of all returned
       numbers above it.

    **For the paragraph.**

    - Does the unexplained term have the same sign at all three positions? If so, which
      assumption is systematically off: equal visits in the stack, the PSF term, or the
      exposure time divided by 30 s as a visit count?
    - `magLim` and `psf_maglim` are both 5σ point-source depths, but measured differently. Does
      that matter at the level of the leftover?
    - Where does the catalog turnover sit relative to the map depth, and is the offset the
      same at the three positions? Why might it be fainter than a point-source depth, and why
      brighter? What would you need before quoting the map's depth as your sample's depth?

    > **Go further.** Repeat in a second band (`depth_at(ra, dec, band="g")`). Does the
    > leftover change sign or size with band? Stacking is achromatic; the PSF term and the sky
    > are not. Or add two more positions and plot the leftover against the number of visits.
    """),
    _c("code", """
    # Butler access: this cell runs on the RSP LSST kernel.
    import json

    try:
        from lsst.daf.butler import Butler
        import hpgeom as hpg
        import pandas as pd
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        butler = Butler("dp2", collections="dp2")
    except ImportError as err:
        butler = None
        print("ImportError:", err)
        print("Exercise 4 needs the LSST kernel on the Rubin Science Platform. Kernel -> Change Kernel -> LSST, then rerun.")

    SKYMAP = "lsst_cells_v2"
    NSIDE_COV = 32          # resolution of the coverage map
    NSIDE_NATIVE = 32768    # native resolution of the maps
    PIXEL_SCALE = 0.2       # arcsec per coadd pixel
    SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
    VISIT_EXPTIME = 30.0    # s, the nominal visit
    CACHE_FILE = "depth_cache.json"

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
        \"\"\"Single-visit and coadd depth at one position, with the Week 2 predictions. Cached on disk.\"\"\"
        cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}
        key = f"{ra:.5f},{dec:.5f},{band}"
        if key in cache:
            return cache[key]
        vd = visit_table(band)
        sep = SkyCoord(vd["ra"].values * u.deg, vd["dec"].values * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg))
        near = vd[sep < 0.1 * u.deg]
        out = {
            "ra": ra, "dec": dec, "band": band,
            "n_detector_images": int(len(near)),
            "visit_depth": float(near["magLim"].median()),
            "visit_fwhm": float(near["psfSigma"].median() * SIGMA_TO_FWHM * PIXEL_SCALE),
            "coadd_depth": map_median_near("psf_maglim", ra, dec, band),
            "coadd_fwhm": map_median_near("psf_fwhm", ra, dec, band),
            "n_visits": map_median_near("exposure_time", ra, dec, band) / VISIT_EXPTIME,
            "turnover_mag": float(turnover_mag(ra, dec, band)),
        }
        out["gain_measured"] = out["coadd_depth"] - out["visit_depth"]
        out["gain_stacking"] = 1.25 * np.log10(out["n_visits"])
        out["gain_psf"] = -2.5 * np.log10(out["coadd_fwhm"] / out["visit_fwhm"])
        out["unexplained"] = out["gain_measured"] - out["gain_stacking"] - out["gain_psf"]
        cache[key] = out
        json.dump(cache, open(CACHE_FILE, "w"), indent=1)
        return out


    positions = [(187.2645519, 8.6612496), (150.11916667, 2.20583333)]   # add your third position here

    # Your measurement here, e.g.
    #   rows = [depth_at(ra, dec) for ra, dec in positions]
    #   table = pd.DataFrame(rows).round(2); print(table)
    # Then the figure: gain_measured against gain_stacking + gain_psf, one point per position,
    #   with a 1:1 line.
    """),
]

HANDIN_CELLS = [
    _c("md", """
    ## Hand-in

    Due Monday Oct 12 on Canvas. Turn in **this notebook** with: (1) one figure from the
    exercise you chose; (2) one paragraph, starting with "Exercise N": what did you measure,
    what does it say about the catalog, and what would it mean for your team's science case
    (the one from the Sep 28 worksheet)? **Quote at least one number read off your figure**;
    (3) the AI-tools line in the last cell filled in. Graded complete/incomplete.

    Do the measurement and the plot in your exercise's cell. Then paste the lines that draw
    the figure into the code cell below, so the figure sits next to the paragraph. Before you
    upload, choose *Run All*, save, and download the .ipynb (File → Download). Submit the
    notebook itself, not a PDF.

    **One exercise is the assignment.** The *Go further* box in each exercise, or a second
    exercise, is optional and does not change the grade. If you do one, add it in new cells
    after the AI-tools line.
    """),
    _c("code", """
    # Your hand-in figure: paste the lines that draw it here
    """),
    _c("md", """
    **Exercise N.** *Your paragraph here.*

    **AI tools:** *none, or which ones and what for.*
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
