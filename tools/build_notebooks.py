"""Build the Week 2 notebooks from readable cell lists.

The notebook source lives here as ordered lists of ("md" | "code", text) cells.
Run ``python tools/build_notebooks.py`` to (re)write week2/A_survey_strategy.ipynb
with outputs cleared. The notebook itself must not import anything from tools/.
"""
from pathlib import Path
import textwrap

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB_A = REPO / "week2" / "A_survey_strategy.ipynb"


def _c(kind, text):
    return (kind, textwrap.dedent(text).strip("\n"))


# ---------------------------------------------------------------------------
# Part 1: your survey with sliders
# ---------------------------------------------------------------------------
PART1_CELLS = [
    _c("md", """
    # Week 2A · Survey strategy → science

    On Monday (Sep 28) you designed your own survey on the "Design your own survey" worksheet:
    one telescope, ten years, one budget, one science goal. Last Wednesday (Sep 23) Phil Marshall
    showed how Rubin's actual survey, the LSST, splits its time. This notebook turns the
    worksheet's rules of thumb into code you can play with, then compares your choices with a full
    simulation of the real ten-year survey.

    **How to use it.** In Colab, choose *Runtime → Run all*, then move the sliders. Every slider
    wraps a plain function you can call directly. Use the direct call for your hand-in figure,
    because sliders don't show up in PDFs.

    **Core path vs. go deeper.** The core path takes about 30–45 minutes. The *Go deeper* boxes are
    optional.

    *Data note: these are public survey simulations, not Rubin data products.*
    """),
    _c("code", """
    # Setup. Everything here ships with Colab.
    from pathlib import Path
    import urllib.request

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from astropy.cosmology import Planck18

    DATA_URL = "https://raw.githubusercontent.com/rhw/phys367/main/week2/data/"
    DATA_DIR = Path("data")
    DATA_FILES = {
        "maps": "baseline_v5.3.3_maps.parquet",
        "points": "baseline_v5.3.3_points.parquet",
        "budget": "baseline_v5.3.3_budget.csv",
        "presets": "presets.csv",
    }

    def load(name):
        \"\"\"Load a data file by short name ('maps', 'points', 'budget', 'presets') or file name.

        Uses the local data/ directory if the file is there; otherwise downloads it from
        GitHub once and caches it in data/.
        \"\"\"
        fname = DATA_FILES.get(name, name)
        path = DATA_DIR / fname
        if not path.exists():
            DATA_DIR.mkdir(exist_ok=True)
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(DATA_URL + fname, path)
        if fname.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    # numpy renamed trapz -> trapezoid in 2.0; use whichever exists.
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    """),
    _c("md", """
    ## Part 1 · Your survey, with sliders

    The worksheet gave you four rules of thumb. Here they are as code, each with where it comes from
    and a question about where it stops being true.

    ### 1. The budget

    $$\\text{Area (deg}^2) \\times N_\\text{visits per field} \\times \\frac{t_\\text{visit}}{30\\,\\text{s}}
    \\approx 1.5\\times10^{7}$$

    *Source: Monday's worksheet* (18,000 deg² × ~825 visits of 30 s; Ivezić et al. 2019 quote ~825
    visits per field for an earlier design of the survey).

    **Where does this break?** The budget treats a 5 s visit as one sixth of a 30 s visit, but each
    visit also costs ~9 s of overhead (readout, shutter, slew). What happens to the real number of
    visits you can afford at very short exposures?

    ### 2. Single-visit depth

    $$m_5^\\text{single} = m_5(30\\,\\text{s}) + 1.25\\log_{10}(t_\\text{visit}/30\\,\\text{s})$$

    with $m_5(30\\,\\text{s})$ = u 23.9, g 25.0, r 24.7, i 24.0, z 23.3, y 22.1
    (*Ivezić et al. 2019, Table 2: dark sky, zenith, 5σ point source*).

    **Where does this break?** These depths assume a dark sky at zenith with good seeing. The
    simulated survey's median r-band visit reaches only 24.06 (median seeing 1.03″, airmass 1.18,
    from the simulation). Which of your science numbers below would shift if every visit were
    0.6 mag shallower?

    ### 3. Coadded depth

    $$m_5^\\text{coadd} = 1.25\\log_{10}\\sum_i 10^{0.8\\,m_{5,i}}
    \\;\\;\\xrightarrow{\\text{identical visits}}\\;\\; m_5^\\text{single} + 1.25\\log_{10} N_\\text{band}$$

    *Source: noise adds in quadrature; the worksheet uses the identical-visit form.* A band with zero
    visits has no coadd depth at all, so it is reported as NaN (not −∞).

    **Where does this break?** Coadding assumes the noise is independent between visits and the
    source doesn't change. Which science cases violate the second assumption?

    ### 4. Revisit time

    $$n \\approx 3\\,\\text{days} \\times \\frac{t_\\text{visit}}{30\\,\\text{s}} \\times
    \\frac{\\text{area}}{18{,}000\\,\\text{deg}^2}$$

    *Source: Monday's worksheet* (average days between visits to a field, observing in pairs).

    **Where does this break?** This is an *average* over ten years. The real survey has seasons,
    weather, and a Moon. Is the average gap the number your science actually cares about?
    """),
    _c("code", """
    # Single-visit 5-sigma depth at 30 s (Ivezic et al. 2019, Table 2; dark sky, zenith).
    M5_30S = {"u": 23.9, "g": 25.0, "r": 24.7, "i": 24.0, "z": 23.3, "y": 22.1}
    BANDS = list(M5_30S)

    # LSST filter split (Ivezic et al. 2019): fraction of visits in each band.
    DEFAULT_SPLIT = {"u": 0.068, "g": 0.097, "r": 0.223, "i": 0.223, "z": 0.194, "y": 0.194}

    BUDGET = 1.5e7          # area x visits x (t/30 s), from the worksheet
    SKY_FROM_CHILE = 30000  # deg^2 reachable from Chile (worksheet)
    FULL_SKY = 41253        # deg^2

    def m5_single(band, t_visit=30.0):
        \"\"\"5-sigma point-source depth of one visit of t_visit seconds.\"\"\"
        return M5_30S[band] + 1.25 * np.log10(t_visit / 30.0)

    def m5_coadd_from_n(m5_one, n):
        \"\"\"Coadd depth of n identical visits; NaN (not -inf) when n == 0.\"\"\"
        return m5_one + 1.25 * np.log10(n) if n > 0 else np.nan
    """),
    _c("md", """
    ### 5. From depth to science

    Three yardsticks, one per kind of science. Each is deliberately crude.

    **Galaxies for weak lensing.** Shapes need S/N ≈ 20, about 1.5 mag brighter than the 5σ limit,
    so $i_\\text{lim} = m_5^\\text{coadd}(i) - 1.5$. The number of galaxies brighter than $i$ is

    $$N(<i) = 46 \\times 10^{0.31\\,(i-25)}\\ \\text{arcmin}^{-2}$$

    *Source: LSST Science Book (2009), eq. 3.7 — a fit to CFHTLS Deep counts over
    20.5 < i < 25.5. Deeper than i ≈ 25.5 it is an extrapolation.* Multiply by the area
    (1 deg² = 3600 arcmin²).

    **Where does this break?** At the LSST defaults your $i_\\text{lim}$ is beyond 25.5, where the
    fit is an extrapolation. It also ignores blending: at LSST depth, many galaxies overlap a
    neighbour. How much would you trust the galaxy count, and does a deeper survey always give you
    more *usable* shapes?

    **Type Ia supernovae.** Peak absolute magnitude $M = -19.3$. A supernova
    counts as "detected near peak" if it is 1 mag brighter than a single i-band visit's limit:
    solve $M + \\mathrm{DM}(z_\\text{max}) = m_5^\\text{single}(i) - 1$ with the Planck 2018
    cosmology. Then

    $$N_\\text{SN} = \\int_0^{z_\\text{max}} \\frac{R\\,(1+z)^{1.5}}{1+z}\\,\\frac{dV}{dz}\\,dz
    \\times \\frac{\\text{area}}{41{,}253} \\times 10\\,\\text{yr} \\times 0.5$$

    with $R = 2.6\\times10^{-5}\\ \\text{Mpc}^{-3}\\,\\text{yr}^{-1}$ (local SN Ia rate), the
    $(1+z)^{1.5}$ rate evolution, $1/(1+z)$ for time dilation, and 0.5 because a field is only
    observable about half the year. The estimate uses the i-band single-visit depth, so it is 0 if
    you give i no visits.

    *Source: volumetric rate $r_V \\propto (1+z)^{1.5}$ from Dilday et al. 2008 (ApJ 682, 262),
    normalized to ≈2.6×10⁻⁵ Mpc⁻³ yr⁻¹ ($h_{70}^3$) — consistent with their measured 2.9×10⁻⁵ at
    z ≈ 0.09. Peak M ≈ −19.3 is the standard SN Ia peak absolute magnitude.*

    **Where does this break?** This is an upper bound. What would you need to add to count
    supernovae that are useful for cosmology?

    **Moving objects.** Linking an asteroid orbit needs roughly 3 same-night pairs within ~15 days
    (Ivezić et al. 2019; Jones et al. 2018: detections on ~3 nights within ~15 nights, 2 visits per
    night).
    Pairs per 15-day window per field:

    $$\\frac{N_\\text{visits}/2}{10 \\times 365.25 \\times 0.5 / 15}$$

    *Source: a counting proxy; assumes every visit is half of a same-night pair and a 6-month
    observing season.*

    **Where does this break?** Visits are not spread evenly: weather, the Moon, and the season
    clump them. Is the average the right statistic for "did we link this asteroid"?
    """),
    _c("code", """
    SN_M_PEAK = -19.3        # SN Ia peak absolute mag
    SN_RATE = 2.6e-5         # Mpc^-3 yr^-1, local SN Ia rate
    YEARS, SEASON = 10.0, 0.5

    # Planck 2018 distances on a grid, integrated with numpy (flat universe, so
    # D_L = (1+z) D_C and dV/dz = 4 pi D_C^2 c / H(z)). Only H(z) comes from astropy.
    C_KMS = 299792.458
    _Z = np.linspace(0.0, 3.0, 3001)
    _INV_E = 1.0 / Planck18.efunc(_Z)
    _DH = C_KMS / Planck18.H0.value                                    # Hubble distance, Mpc
    _DC = _DH * np.concatenate([[0.0], np.cumsum(0.5 * (_INV_E[1:] + _INV_E[:-1]) * np.diff(_Z))])
    _DVDZ = 4 * np.pi * _DC**2 * _DH * _INV_E                           # Mpc^3 per unit z, full sky
    _DM = 5 * np.log10(np.maximum((1 + _Z) * _DC, 1e-10) * 1e6 / 10.0)  # distance modulus

    def z_max_sn(m_lim):
        \"\"\"Redshift at which a M = -19.3 SN Ia has apparent magnitude m_lim (Planck18).\"\"\"
        m_app = SN_M_PEAK + _DM[1:]           # monotonic in z
        if m_lim <= m_app[0]:
            return 0.0
        if m_lim >= m_app[-1]:
            return float(_Z[-1])
        return float(np.interp(m_lim, m_app, _Z[1:]))

    def n_sn_detected(m_lim, area):
        \"\"\"Upper bound on SNe Ia seen near peak.\"\"\"
        zmax = z_max_sn(m_lim)
        if zmax <= 0:
            return 0.0
        z = np.linspace(0.0, zmax, 400)
        dvdz = np.interp(z, _Z, _DVDZ)
        per_year = _trapz(SN_RATE * (1 + z) ** 1.5 / (1 + z) * dvdz, z)
        return float(per_year * (area / FULL_SKY) * YEARS * SEASON)

    def plan_survey(area, t_visit, split, budget=BUDGET):
        \"\"\"Turn (area, exposure time, filter split) into depths and science yields.

        The split is normalized to sum to 1, so you can give any non-negative weights.
        \"\"\"
        warnings = []
        weights = {b: max(float(split.get(b, 0.0)), 0.0) for b in BANDS}
        total = sum(weights.values())
        if total == 0:
            warnings.append("No visits in any band: the filter split is all zeros.")
            frac = {b: 0.0 for b in BANDS}
        else:
            frac = {b: w / total for b, w in weights.items()}

        n_visits = budget / (area * (t_visit / 30.0))
        n_by_band = {b: n_visits * frac[b] for b in BANDS}
        m5s = {b: m5_single(b, t_visit) for b in BANDS}
        m5c = {b: m5_coadd_from_n(m5s[b], n_by_band[b]) for b in BANDS}

        revisit_days = 3.0 * (t_visit / 30.0) * (area / 18000.0)

        i_lim = m5c["i"] - 1.5                                   # S/N ~ 20 for shapes
        n_gal = 46 * 10 ** (0.31 * (i_lim - 25)) * area * 3600   # NaN if no i visits

        n_sn = n_sn_detected(m5s["i"] - 1, area) if n_by_band["i"] > 0 else 0.0

        pairs_per_15d = (n_visits / 2) / (YEARS * 365.25 * SEASON / 15)

        if t_visit < 15:
            warnings.append(f"t_visit = {t_visit:g} s: the ~9 s overhead is more than 1/3 "
                            "of each visit, so the budget overstates your visits.")
        if t_visit > 60:
            warnings.append(f"t_visit = {t_visit:g} s: exposures longer than ~1 min trail "
                            "fast-moving objects.")
        for b in BANDS:
            if total > 0 and n_by_band[b] == 0:
                warnings.append(f"No {b}-band visits: no {b} coadd depth, and no {b} colours "
                                "for photometric redshifts.")
        if area > SKY_FROM_CHILE:
            warnings.append(f"Area = {area:,.0f} deg^2 is more than the ~30,000 deg^2 "
                            "reachable from Chile.")

        return dict(n_visits=n_visits, n_by_band=n_by_band, m5_single=m5s, m5_coadd=m5c,
                    revisit_days=revisit_days, n_gal=n_gal, n_sn_detected=n_sn,
                    pairs_per_15d=pairs_per_15d, warnings=warnings)
    """),
    _c("code", """
    def show_survey(area=18000, t_visit=30, u=DEFAULT_SPLIT["u"], g=DEFAULT_SPLIT["g"],
                    r=DEFAULT_SPLIT["r"], i=DEFAULT_SPLIT["i"], z=DEFAULT_SPLIT["z"],
                    y=DEFAULT_SPLIT["y"]):
        \"\"\"Print the plan for one survey and plot its depth by band. Returns the plan dict.\"\"\"
        split = dict(u=u, g=g, r=r, i=i, z=z, y=y)
        p = plan_survey(area, t_visit, split)
        print(f"Area {area:,.0f} deg^2, t_visit {t_visit:g} s -> "
              f"{p['n_visits']:.0f} visits per field over 10 yr (from the worksheet budget)")
        print(f"{'band':>4} {'visits':>7} {'m5 single':>10} {'m5 coadd':>9}")
        for b in BANDS:
            print(f"{b:>4} {p['n_by_band'][b]:7.0f} {p['m5_single'][b]:10.2f} "
                  f"{p['m5_coadd'][b]:9.2f}")
        print()
        print(f"Revisit time (worksheet rule):        {p['revisit_days']:.1f} days")
        print(f"Weak-lensing galaxies (i < m5 - 1.5): {p['n_gal']:.2e}")
        print("SNe Ia detected near peak (an upper bound — see the question below): "
              f"{p['n_sn_detected']:.2e}")
        print(f"Same-night pairs per field per 15 d:  {p['pairs_per_15d']:.1f}  "
              "(rule of thumb: linking an orbit needs ~3 pairs within ~15 days; "
              "Ivezic et al. 2019, Jones et al. 2018)")
        if p["warnings"]:
            print("\\nWarnings:")
            for w in p["warnings"]:
                print("  -", w)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(BANDS))
        coadd = [p["m5_coadd"][b] for b in BANDS]
        ax.bar(x, coadd, color="0.75", label="10-yr coadd $m_5$")
        ax.scatter(x, [p["m5_single"][b] for b in BANDS], marker="_", s=600, color="k",
                   zorder=3, label="single-visit $m_5$")
        ax.set_xticks(x, BANDS)
        finite = [c for c in coadd if np.isfinite(c)]
        ax.set_ylim(21, max(finite + [25.0]) + 1.0)
        ax.set_ylabel("5σ point-source depth (mag)")
        ax.set_title(f"{area:,.0f} deg², {t_visit:g} s visits")
        ax.legend(loc="upper right", fontsize=8)
        plt.show()
        return p
    """),
    _c("md", """
    ### What the real LSST does

    For reference: 18,000 deg², 30 s visits, and the Ivezić et al. (2019) filter split
    (u 6.8%, g 9.7%, r 22.3%, i 22.3%, z 19.4%, y 19.4%).
    """),
    _c("code", """
    lsst = show_survey(18000, 30, **DEFAULT_SPLIT)
    """),
    _c("md", """
    **Check your number.** Phil Marshall quoted ~100,000 LSST supernovae. Why is your number bigger?
    What does this estimate leave out?

    **Asteroids.** Does the LSST default meet the ~3 pairs in ~15 days rule? What about your survey?

    **Two cadence numbers.** The revisit time and the pairs-per-15-days number don't quite agree.
    Why not? Which assumptions differ?
    """),
    _c("md", """
    ### Now it's yours

    Move the sliders. Area runs from 1,000 to 30,000 deg² and t_visit from 5 to 120 s. The six
    filter sliders are relative weights: `plan_survey` normalizes them to sum to 1, so only their
    ratios matter. Setting a band to 0 removes it.

    If the sliders don't appear (e.g. in a PDF), call `show_survey(area, t_visit, u=..., g=..., ...)`
    directly; that's also what to use for your hand-in figure.
    """),
    _c("code", """
    from ipywidgets import interact, FloatSlider, IntSlider

    _band_sliders = {b: FloatSlider(value=DEFAULT_SPLIT[b], min=0.0, max=0.5, step=0.01,
                                    description=b, continuous_update=False) for b in BANDS}
    interact(show_survey,
             area=IntSlider(value=18000, min=1000, max=30000, step=500,
                            description="area (deg²)", continuous_update=False),
             t_visit=IntSlider(value=30, min=5, max=120, step=5,
                               description="t_visit (s)", continuous_update=False),
             **_band_sliders);
    """),
    _c("md", """
    ### Your team's card from Monday

    Set the knobs for the science goal on your team's card from Monday (weak lensing, solar system,
    fast transients, Milky Way, or time-domain cosmology). Then write down, in the cell below:

    - **What changed?** Compared with the LSST defaults, which numbers went up and which went down?
    - **Can't live without.** Which single number does your science depend on most, and what value
      did you get?
    """),
    _c("md", """
    *Your notes here.*
    """),
    _c("md", """
    > **Go deeper.** Replace the 9 s overhead warning with a real overhead: make the budget count
    > $(t_\\text{visit} + 9\\,\\text{s})$ per visit instead of $t_\\text{visit}$. How many visits per
    > field does a 5 s survey really get?
    """),
]

# ---------------------------------------------------------------------------
# Part 2: the real plan, baseline v5.3.3
# ---------------------------------------------------------------------------
PART2_CELLS = [
    _c("md", """
    ## Part 2 · The real plan: baseline v5.3.3

    Rubin's scheduler team simulates the full ten-year survey visit by visit. We use their
    baseline v5.3.3 simulation: 1,852,100 visits starting in mid-2026 (from the simulation). The
    files in `data/` are small extracts of it:

    - **maps**: for each sky pixel (about 0.84 deg²) and band, the number of visits and the coadded
      depth after 10 years and after Year 1, plus the median gap between nights with a visit.
    - **points**: every visit that covers one of a few named sky positions.
    - **budget**: how the visits and open-shutter hours split among the survey's programs.

    To build these we treated each visit as a circle of radius 1.75° around the pointing centre;
    chip gaps are ignored.

    The coadded depth in each pixel uses the same formula as Part 1, but with each visit's actual
    $m_5$ instead of a single number:
    $m_5^\\text{coadd} = 1.25\\log_{10}\\sum_i 10^{0.8\\,m_{5,i}}$.
    """),
    _c("code", """
    maps = load("maps")
    points = load("points")
    budget = load("budget")
    print(f"maps: {len(maps):,} (pixel, band) rows; points: {len(points):,} visits; "
          f"budget: {budget['total_visits'].sum():,} visits in total")
    maps.head()
    """),
    _c("md", """
    ### Where did ten years go?

    The simulation labels every visit with the program that asked for it: the **main** wide survey,
    the five **DDF** deep drilling fields, **templates** (early visits to build reference images),
    **ToO** (targets of opportunity, such as follow-up of gravitational-wave events), **twilight NEO**
    (short twilight visits hunting near-Earth objects), and the **Roman bulge** field.
    """),
    _c("code", """
    def budget_table():
        \"\"\"Share of visits and of open-shutter hours per program, largest first (from the simulation).\"\"\"
        b = load("budget").set_index("category")
        t = pd.DataFrame({
            "visits": b["total_visits"],
            "% of visits": 100 * b["total_visits"] / b["total_visits"].sum(),
            "open-shutter hours": b["open_shutter_hours"],
            "% of hours": 100 * b["open_shutter_hours"] / b["open_shutter_hours"].sum(),
        })
        return t.sort_values("% of visits", ascending=False)

    budget_table().round(1)
    """),
    _c("md", """
    **Where did ten years go?** Your worksheet budget spent everything on one survey. What fraction
    does the real plan spend on the main survey? Twilight NEO visits are a larger share of visits
    than of hours. Why?

    ### What does a main-survey field get?

    Main-survey fields get roughly 150–170 r-band visits and ~720–750 visits in all bands over
    10 years in this simulation (Ivezić et al. 2019 quoted ~184 r and ~825 total for an earlier
    design).

    `compare_to_mine(plan)` puts your plan from Part 1 next to the medians over main-survey pixels.
    We call a pixel "main survey" if it has at least 500 visits in all bands and lies more than 2°
    from any deep drilling field. **This is a heuristic**, not the simulation's own label: the
    500-visit cut and the 2° radius are our choices, and the medians shift a little if you change
    them (try `main_survey_medians(min_visits=400, ddf_radius=3)`).
    """),
    _c("code", """
    # Deep drilling field centres (RA, Dec in degrees), copied from data/presets.csv.
    DDF_FIELDS = {"COSMOS": (150.10, 2.18), "ECDFS": (53.13, -28.10), "EDFS": (58.90, -49.32),
                  "ELAIS-S1": (9.45, -44.00), "XMM-LSS": (35.71, -4.75)}

    def ang_sep_deg(ra1, dec1, ra2, dec2):
        \"\"\"Angular separation in degrees (haversine; safe at RA = 0/360 and near the poles).\"\"\"
        ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
        h = (np.sin((dec2 - dec1) / 2) ** 2
             + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2)
        return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))

    def main_survey_pixels(maps=None, min_visits=500, ddf_radius=2.0):
        \"\"\"Pixel ids we treat as main survey (heuristic: >= min_visits total, > ddf_radius from a DDF).\"\"\"
        m = load("maps") if maps is None else maps
        pix = m.groupby("hpix").agg(ra=("ra", "first"), dec=("dec", "first"), total=("nvis", "sum"))
        keep = pix["total"] >= min_visits
        for ra0, dec0 in DDF_FIELDS.values():
            keep &= ang_sep_deg(pix["ra"].values, pix["dec"].values, ra0, dec0) > ddf_radius
        return pix.index[keep]

    def main_survey_medians(maps=None, **cuts):
        \"\"\"Median visits and coadd depth per band over main-survey pixels (from the simulation).\"\"\"
        m = load("maps") if maps is None else maps
        pix = main_survey_pixels(m, **cuts)
        sub = m[m["hpix"].isin(pix)]
        nvis = sub.pivot(index="hpix", columns="band", values="nvis").reindex(
            index=pix, columns=BANDS).fillna(0)
        m5c = sub.pivot(index="hpix", columns="band", values="m5_coadd").reindex(
            index=pix, columns=BANDS)
        return pd.DataFrame({"nvis": nvis.median(), "m5_coadd": m5c.median().astype(float)})

    def compare_to_mine(plan):
        \"\"\"Your plan (from plan_survey) next to the real main-survey medians, per band.\"\"\"
        real = main_survey_medians()
        t = pd.DataFrame({
            "yours: visits": [plan["n_by_band"][b] for b in BANDS],
            "real: visits": real["nvis"].values,
            "yours: m5 coadd": [plan["m5_coadd"][b] for b in BANDS],
            "real: m5 coadd": real["m5_coadd"].values,
        }, index=BANDS)
        print(f"{len(main_survey_pixels()):,} main-survey pixels (heuristic); "
              f"real total visits per pixel: {real['nvis'].sum():.0f} (median per band, summed)")

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        x = np.arange(len(BANDS))
        for ax, col, label in [(axes[0], "visits", "visits in 10 yr"),
                               (axes[1], "m5 coadd", "10-yr coadd $m_5$ (mag)")]:
            ax.bar(x - 0.2, t[f"yours: {col}"], 0.4, label="your plan (worksheet rules)")
            ax.bar(x + 0.2, t[f"real: {col}"], 0.4, label="baseline v5.3.3 main survey")
            ax.set_xticks(x, BANDS)
            ax.set_ylabel(label)
        vals = t[["yours: m5 coadd", "real: m5 coadd"]].values
        vals = vals[np.isfinite(vals)]
        axes[1].set_ylim(vals.min() - 1, vals.max() + 0.5)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        plt.show()
        return t.astype(float).round(2)

    compare_to_mine(lsst)
    """),
    _c("md", """
    Try it with your own plan: `compare_to_mine(plan_survey(area, t_visit, split))`.

    ### Why is the real survey shallower per visit?

    Part 1 assumed every r-band visit reaches 24.7 (Ivezić et al. 2019: dark sky, zenith). Across the
    full simulation, the median r-band visit reaches 24.06 (from the full simulation). The cell below
    uses the points file: every r-band main-survey visit that covers one of the named positions.
    The points file covers only the named positions (including star fields like 47 Tuc and the LMC),
    so these medians are a sample of the sky, not the whole survey.
    """),
    _c("code", """
    def real_vs_worksheet(band="r", category="main"):
        \"\"\"Median m5, seeing, airmass and sky of real visits vs the worksheet's assumptions.\"\"\"
        p = load("points")
        sel = p[(p["band"] == band) & (p["category"] == category)]
        assumed = {"m5": M5_30S[band], "seeing": 0.7, "airmass": 1.0,
                   "sky": 21.2 if band == "r" else np.nan}
        t = pd.DataFrame({
            "median (simulation)": sel[["m5", "seeing", "airmass", "sky"]].median(),
            "worksheet assumption": pd.Series(assumed),
        })
        t["units"] = ["mag", "arcsec (FWHM)", "", "mag/arcsec²"]
        print(f"{len(sel):,} {band}-band '{category}' visits in the points file")
        return t.round(2)

    real_vs_worksheet()
    """),
    _c("md", """
    **Why is the median real visit ~0.6 mag shallower than your 24.7?** Which of the three
    (seeing, airmass, sky) matters most? Try estimating each effect on its own before you look
    anything up. Plot `m5` against each column of `points` for r-band visits to check.

    ### Maps

    `sky_map(col, band, which)` draws one column of the maps file on the sky. `col` is `"nvis"`,
    `"m5_coadd"` or `"median_night_gap"`; `which` is `"10yr"` or `"y1"` (Year 1). East is to the
    left, as on the sky. Each dot is a pixel centre. Pixels with no visits in that band (and period)
    are simply absent, so blank sky means "never observed", not zero. The colour scale runs from
    the 1st to the 99th percentile, so the deep drilling fields saturate; pass `vmin=`, `vmax=` to
    change it.
    """),
    _c("code", """
    COL_LABELS = {"nvis": "number of visits", "m5_coadd": "coadd 5σ depth (mag)",
                  "median_night_gap": "median gap between nights with a visit (days)"}

    def sky_map(col="nvis", band="r", which="10yr", ax=None, **scatter_kw):
        \"\"\"Mollweide map of one maps column for one band. Returns the axes.\"\"\"
        m = load("maps")
        m = m[m["band"] == band]
        if which == "y1":
            if col not in ("nvis", "m5_coadd"):
                raise ValueError("Year-1 maps exist only for 'nvis' and 'm5_coadd'")
            m = m[m["nvis_y1"] > 0]
            col_used = col + "_y1"
        elif which == "10yr":
            col_used = col
        else:
            raise ValueError("which must be '10yr' or 'y1'")
        vals = m[col_used].astype(float)
        ok = np.isfinite(vals.values)
        m, vals = m[ok], vals[ok]

        if ax is None:
            fig = plt.figure(figsize=(8, 4.5))
            ax = fig.add_subplot(projection="mollweide")
        # RA in radians in [-pi, pi], flipped so east (increasing RA) is to the left.
        ra = np.radians(((m["ra"].values + 180.0) % 360.0) - 180.0)
        dec = np.radians(m["dec"].values)
        # Colour limits default to the 1st-99th percentiles so a few deep fields don't wash
        # out the rest of the sky; pass vmin=/vmax= to override.
        # median_night_gap counts nights with a visit in any band, so it has no band.
        band_txt = "all bands" if col == "median_night_gap" else f"{band} band"
        kw = dict(s=2.5, cmap="viridis", linewidths=0,
                  vmin=np.percentile(vals, 1) if len(vals) else None,
                  vmax=np.percentile(vals, 99) if len(vals) else None)
        kw.update(scatter_kw)
        sc = ax.scatter(-ra, dec, c=vals.values, **kw)
        ax.set_xticks(np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]))
        ax.set_xticklabels(["150°", "120°", "90°", "60°", "30°", "0°", "330°", "300°",
                            "270°", "240°", "210°"], fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.06, shrink=0.7, extend="both",
                     label=f"{band_txt}: {COL_LABELS.get(col, col)}")
        ax.set_title(f"{band_txt}, {'10 years' if which == '10yr' else 'Year 1'}: "
                     f"{COL_LABELS.get(col, col)}", fontsize=10, pad=14)
        return ax

    sky_map("m5_coadd", "r")
    plt.show()
    """),
    _c("md", """
    **What do you see?** Find the main survey, the deep drilling fields, and the Galactic plane.
    Which parts of the sky get more visits than the main survey, and which get fewer? Try `"nvis"`
    and other bands.

    ### Year 1 vs 10 years

    Compare the r-band visits after Year 1 and after 10 years. Note that the two colour scales are
    different.
    """),
    _c("code", """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), subplot_kw={"projection": "mollweide"})
    sky_map("nvis", "r", which="y1", ax=axes[0])
    sky_map("nvis", "r", which="10yr", ax=axes[1])
    plt.show()
    """),
    _c("code", """
    PIXEL_AREA = 41253 / 49152   # deg^2 per pixel (HEALPix nside = 64: 12 * 64^2 pixels)

    def year1_coadd_area(min_visits=6):
        \"\"\"Area (deg^2) with at least min_visits Year-1 visits in every one of the six bands.\"\"\"
        m = load("maps")
        y1 = m.pivot(index="hpix", columns="band", values="nvis_y1").reindex(columns=BANDS).fillna(0)
        return float((y1.min(axis=1) >= min_visits).sum() * PIXEL_AREA)

    print(f"Area with >= 6 Year-1 visits in every band: {year1_coadd_area(6):,.0f} deg^2 "
          "(from the maps extract of the baseline v5.3.3 simulation; 1.75° circular footprint, "
          "chip gaps ignored)")
    """),
    _c("md", """
    - In the baseline v5.3.3 simulation, about 19,000 deg² get at least 6 visits in every band
      during Year 1 (computed above from the maps extract).
    - Phil Marshall (Sep 23) showed that updated simulations give an "expected number of visits
      between June '26–'27 not sufficient to coadd area more than DDFs", using 6+ visits as the
      "minimum raw visits for coaddition after quality cuts".

    **Two answers to one question.** Both come from simulations of the same survey. List the
    differences in assumptions that could turn ~19,000 deg² into "mostly just the deep fields".
    Which would you check first?

    ### How often does the survey come back?

    The worksheet's revisit rule gave one average number. The map below shows, for each pixel, the
    median gap between nights with a visit in any band, over 10 years (from the simulation).
    """),
    _c("code", """
    # The gap is the same on every band row of a pixel; "r" just picks the r-band rows (pixels r visited).
    sky_map("median_night_gap", "r", vmin=0, vmax=15)
    plt.show()
    """),
    _c("md", """
    **Median gap vs. your revisit time.** Read the typical value off the map. How does it compare
    with the ~3 days from the worksheet rule? Why might a median of a
    few nights still leave long gaps that matter for your science?

    > **Go deeper.** Use `maps` to compute how much sky reaches a 10-yr r-band coadd depth of at
    > least 27.0, and how that area grows from Year 1 to Year 10.
    """),
]
PART3_CELLS = []
PART4_CELLS = []


def build_notebook_a():
    cells = []
    for kind, text in PART1_CELLS + PART2_CELLS + PART3_CELLS + PART4_CELLS:
        if kind == "md":
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def build(path=NB_A):
    """Write the notebook (outputs cleared) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook_a(), path)
    return path


if __name__ == "__main__":
    print(build())
