"""Build the Week 2 notebooks from readable cell lists.

The notebook source lives here as ordered lists of ("md" | "code", text) cells.
Run ``python tools/build_notebooks.py`` to (re)write week2/A_survey_strategy.ipynb and
week2/B_camera_to_science.ipynb with outputs cleared. The notebook itself must not import anything from tools/.
"""
from pathlib import Path
import textwrap

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB_A = REPO / "week2" / "A_survey_strategy.ipynb"
NB_B = REPO / "week2" / "B_camera_to_science.ipynb"


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
    import os
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
            part = path.with_suffix(path.suffix + ".part")   # don't trust a half-finished download
            urllib.request.urlretrieve(DATA_URL + fname, part)
            part.rename(path)
        if fname.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    # numpy renamed trapz -> trapezoid in 2.0; use whichever exists.
    _trapz = getattr(np, "trapezoid", None) or np.trapz

    from ipywidgets import interact, interact_manual

    def sliders(func, **controls):
        \"\"\"Attach sliders and menus to func, like ipywidgets.interact.

        Automated runs with no screen (our tests set PHYS367_HEADLESS=1) build the same widgets
        but don't run func, because live widget output can stall a headless run.
        \"\"\"
        if os.environ.get("PHYS367_HEADLESS") == "1":
            return interact_manual(func, **controls)
        return interact(func, **controls)
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
    (*Ivezić et al. 2019, Table 1: SRD design specification, fiducial zenith, 5σ point source*).

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
    # Single-visit 5-sigma depth at 30 s (Ivezic et al. 2019, Table 1: SRD design specification, fiducial zenith).
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

    **Where does this break?** At the LSST defaults $i_\\text{lim} \\approx 25.3$, just inside the
    fit's range (20.5 < i < 25.5); a deeper survey pushes it into extrapolation. It also ignores
    blending: at LSST depth, many galaxies overlap a
    neighbor. How much would you trust the galaxy count, and does a deeper survey always give you
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
                warnings.append(f"No {b}-band visits: no {b} coadd depth, and no {b} colors "
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
    from ipywidgets import FloatSlider, IntSlider

    _band_sliders = {b: FloatSlider(value=DEFAULT_SPLIT[b], min=0.0, max=0.5, step=0.01,
                                    description=b, continuous_update=False) for b in BANDS}
    def _survey_widget(area, t_visit, u, g, r, i, z, y):
        show_survey(area, t_visit, u, g, r, i, z, y)     # don't echo the returned plan dict

    sliders(_survey_widget,
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

    To build these we treated each visit as a circle of radius 1.75° around the pointing center;
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
    # Deep drilling field centers (RA, Dec in degrees), copied from data/presets.csv.
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
    left, as on the sky. Each dot is a pixel center. Pixels with no visits in that band (and period)
    are simply absent, so blank sky means "never observed", not zero. The color scale runs from
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
        # Color limits default to the 1st-99th percentiles so a few deep fields don't wash
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

    Compare the r-band visits after Year 1 and after 10 years. Note that the two color scales are
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
# ---------------------------------------------------------------------------
# Part 3: one point on the sky, and light curves
# ---------------------------------------------------------------------------
PART3_CELLS = [
    _c("md", """
    ## Part 3 · One point on the sky

    The maps average over ten years. A transient doesn't: what matters is *when* the survey looks.
    The points file lists every visit whose 1.75° circle covers one of a few named positions
    (chip gaps ignored), with that visit's time, band, and depth (from the simulation). Here are
    the positions.
    """),
    _c("code", """
    presets = load("presets").set_index("name")
    presets
    """),
    _c("code", """
    FOOTPRINT_RADIUS = 1.75   # deg; each visit is a circle this size around its pointing (chip gaps ignored)
    SURVEY_START = 61208.0    # MJD of the first night of the simulation (~2026-06-29)
    BAND_COLORS = {"u": "tab:purple", "g": "tab:blue", "r": "tab:green",
                   "i": "tab:orange", "z": "tab:red", "y": "tab:brown"}

    def _check_preset(preset):
        if preset not in presets.index:
            raise ValueError(f"unknown preset {preset!r}; choose one of: {', '.join(presets.index)}")

    def _no_visits(preset):
        print(f"No visits within {FOOTPRINT_RADIUS}° of {preset} in this simulation "
              "— why might that be?")

    def visits_at(preset):
        \"\"\"All simulated visits covering one preset position, in time order (from the points file).\"\"\"
        _check_preset(preset)
        return points[points["preset"] == preset].sort_values("mjd").reset_index(drop=True)

    def seasons(mjd, gap=60.0):
        \"\"\"(start, end) MJD of each observing season: runs of visits with no gap longer than `gap` days.\"\"\"
        mjd = np.sort(np.asarray(mjd, float))
        if mjd.size == 0:
            return []
        breaks = np.where(np.diff(mjd) > gap)[0]
        return list(zip(np.r_[mjd[0], mjd[breaks + 1]], np.r_[mjd[breaks], mjd[-1]]))

    def night_gaps(mjd, gap=60.0):
        \"\"\"Gaps (days) between successive nights with a visit, leaving out the gaps between seasons.\"\"\"
        nights = np.unique(np.floor(np.asarray(mjd, float)))
        d = np.diff(nights)
        return d[d <= gap]

    def cadence_plot(preset, start=None, stop=None, ax=None):
        \"\"\"Raster of visits (time x band) at one preset, seasons shaded. Returns the visits.\"\"\"
        v = visits_at(preset)
        if len(v) == 0:
            _no_visits(preset)
            return v
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
        for s, e in seasons(v["mjd"]):
            ax.axvspan(s - 0.5, e + 0.5, color="0.9", zorder=0)
        for k, b in enumerate(BANDS):
            vb = v[v["band"] == b]
            ax.scatter(vb["mjd"], np.full(len(vb), k), marker="|", s=120, linewidths=0.8,
                       color=BAND_COLORS[b])
        ax.set_yticks(range(len(BANDS)), BANDS)
        ax.set_ylim(len(BANDS) - 0.5, -0.5)
        ax.set_xlim(start if start is not None else v["mjd"].min() - 20,
                    stop if stop is not None else v["mjd"].max() + 20)
        ax.set_xlabel("MJD (shaded: observing seasons)")
        top = ax.secondary_xaxis("top", functions=(lambda m: (m - SURVEY_START) / 365.25,
                                                  lambda y: y * 365.25 + SURVEY_START))
        top.set_xlabel("years since the survey starts")
        n_nights = len(np.unique(np.floor(v["mjd"])))
        ra, dec = presets.loc[preset, ["ra", "dec"]]
        ax.set_title(f"{preset} (RA {ra:g}, Dec {dec:g}): {len(v):,} visits on {n_nights:,} nights "
                     f"in 10 yr; median gap between nights in a season "
                     f"{np.median(night_gaps(v['mjd'])):.1f} d", fontsize=9)
        if show:
            plt.tight_layout()
            plt.show()
        return v

    cadence_plot("main-1");
    """),
    _c("code", """
    def cadence_table():
        \"\"\"One row per preset: visits, nights, seasons, and gaps between nights (from the points file).\"\"\"
        rows = {}
        for name in presets.index:
            v = visits_at(name)
            g = night_gaps(v["mjd"]) if len(v) else np.array([])
            rows[name] = {"visits": len(v),
                          "nights": len(np.unique(np.floor(v["mjd"]))),
                          "seasons": len(seasons(v["mjd"])),
                          "median gap (d)": np.median(g) if len(g) else np.nan,
                          "90th pct gap (d)": np.percentile(g, 90) if len(g) else np.nan}
        return pd.DataFrame(rows).T.round(1)

    cadence_table()
    """),
    _c("md", """
    In rolling-cadence years a field can go more than 60 days between visits in the middle of its
    season, so the "seasons" column counts gaps longer than 60 days, not calendar seasons.

    Pick a position from the menu. If the menu doesn't appear, call `cadence_plot("COSMOS")`
    directly; zoom in on one season with `cadence_plot("COSMOS", start=61400, stop=61550)`.
    """),
    _c("code", """
    from ipywidgets import Dropdown

    def _cadence_widget(preset):
        cadence_plot(preset)

    sliders(_cadence_widget, preset=Dropdown(options=list(presets.index), value="COSMOS",
                                              description="position"));
    """),
    _c("md", """
    **Read the raster.** Compare a deep drilling field with a main-survey field. How long is a
    season? How many nights per season get a visit, and in how many bands? What does the
    `north-edge` position tell you?

    ### Photometric errors

    A visit's $m_5$ is the magnitude at which a point source has S/N = 5. If the noise is dominated
    by the sky background, S/N scales with the source flux, so

    $$\\mathrm{S/N} = 5\\times10^{-0.4\\,(m - m_5)}, \\qquad
    \\sigma_m = \\frac{2.5}{\\ln 10}\\,\\frac{1}{\\mathrm{S/N}} = \\frac{1.0857}{5\\times10^{-0.4\\,(m-m_5)}}$$

    *Source: the definition of $m_5$ plus background-limited noise; a simplified form of the
    error model in Ivezić et al. 2019.* A source at exactly $m_5$ has $\\sigma_m \\approx 0.22$ mag.
    We call a visit a **detection** if the observed magnitude is brighter than that visit's
    $m_5$ (S/N > 5). To simulate a measurement we add Gaussian noise to the *flux*, not the
    magnitude, so that faint sources scatter the right way.

    **Where does this break?** For bright sources the source's own photon noise dominates, and real
    photometry has a systematic floor of a few millimag. Which of the models below are bright
    enough for that to matter?
    """),
    _c("code", """
    def mag_err(m, m5):
        \"\"\"1-sigma magnitude error of a source of magnitude m in a visit of depth m5.\"\"\"
        return 1.0857 / (5 * 10 ** (-0.4 * (np.asarray(m, float) - np.asarray(m5, float))))

    print(f"mag_err(24.0, 24.0) = {mag_err(24.0, 24.0):.3f} mag;  "
          f"mag_err(21.5, 24.0) = {mag_err(21.5, 24.0):.3f} mag")
    """),
    _c("md", """
    ### Three toy transients

    Each model is a function `model(t, band)` that returns the true apparent magnitude at time
    `t` (days after `t0`) in one band, or NaN when the source isn't there.

    **Kilonova** (a neutron-star merger). Peak $M_r \\approx -16$, fading by roughly 0.5–1 mag per
    day, roughly like AT2017gfo, the GW170817 kilonova. We put it at $d = 200$ Mpc by default
    (GW170817 was at about 40 Mpc), start it at peak at $t = 0$, and let it fade linearly in
    magnitude at u 1.2, g 1.0, r 0.7, i 0.5, z 0.4, y 0.35 mag/day. The same peak in every band and
    these per-band rates are toy values chosen to make it redden as it fades.

    $$m(t) = -16 + 5\\log_{10}\\frac{d}{10\\,\\text{pc}} + \\dot m_\\text{band}\\,t, \\qquad t \\ge 0$$

    **Where does this break?** A real kilonova rises in under a day and is blue early, then red.
    It also sits in a host galaxy, behind dust. Which of these matters most for *catching* it?

    **Type Ia supernova** at redshift $z$ (default 0.3). A Bazin et al. (2009) shape in flux,
    $f(t) \\propto e^{-t/\\tau_\\text{fall}} / (1 + e^{-t/\\tau_\\text{rise}})$, with
    $\\tau_\\text{rise} = 5$ d and $\\tau_\\text{fall} = 20$ d in the rest frame, stretched by
    $(1+z)$, shifted so $t = 0$ is peak. Peak $M = -19.3$ (the standard SN Ia peak, as in Part 1)
    plus toy color offsets u +0.5, g 0, r −0.1, i +0.2, z +0.4, y +0.5, at the Planck 2018
    distance modulus.

    **Where does this break?** At $z = 0.3$ each LSST band sees a bluer part of the rest-frame
    spectrum (no K-correction here). Real SNe Ia also differ in stretch and color. What would
    you need to measure to use one for cosmology?
    """),
    _c("code", """
    KN_M_PEAK = -16.0                 # peak absolute mag, roughly AT2017gfo (GW170817)
    KN_FADE = {"u": 1.2, "g": 1.0, "r": 0.7, "i": 0.5, "z": 0.4, "y": 0.35}   # mag/day (toy)
    SN_TAU_RISE, SN_TAU_FALL = 5.0, 20.0                                        # rest-frame days
    SN_COLOR = {"u": 0.5, "g": 0.0, "r": -0.1, "i": 0.2, "z": 0.4, "y": 0.5}   # mag (toy)

    def dist_mod_mpc(d_mpc):
        \"\"\"Distance modulus for a distance in Mpc.\"\"\"
        return 5 * np.log10(d_mpc * 1e6 / 10.0)

    def kilonova(t, band, d_mpc=200):
        \"\"\"Toy kilonova: peak M = -16 at t = 0, then a linear fade per band. NaN for t < 0.\"\"\"
        t = np.asarray(t, float)
        m = KN_M_PEAK + dist_mod_mpc(d_mpc) + KN_FADE[band] * t
        return np.where(t >= 0, m, np.nan)

    def sn_ia(t, band, z=0.3):
        \"\"\"Toy SN Ia: Bazin shape (tau_rise 5 d, tau_fall 20 d, x (1+z)); t = 0 is peak.\"\"\"
        t = np.asarray(t, float)
        tr, tf = SN_TAU_RISE * (1 + z), SN_TAU_FALL * (1 + z)
        t_peak = tr * np.log(tf / tr - 1)          # where the Bazin curve peaks
        def log_f(x):                              # log flux; logaddexp avoids overflow
            return -x / tf - np.logaddexp(0.0, -x / tr)
        m_peak = SN_M_PEAK + SN_COLOR[band] + np.interp(z, _Z, _DM)
        return m_peak - 2.5 / np.log(10) * (log_f(t + t_peak) - log_f(t_peak))
    """),
    _c("code", """
    LC_COLUMNS = ["mjd", "night", "band", "m5", "t", "m_true", "m_obs", "m_err", "detected"]

    def _measure(m_true, m5, rng):
        \"\"\"Add flux noise (sigma = flux at m5 / 5). Returns observed mag (NaN if flux <= 0), error.\"\"\"
        f = 10 ** (-0.4 * (m_true - m5))           # flux in units of the flux at m5
        f_obs = f + rng.normal(0.0, 0.2, len(f))
        with np.errstate(divide="ignore", invalid="ignore"):
            m_obs = np.where(f_obs > 0, m5 - 2.5 * np.log10(f_obs), np.nan)
        return m_obs, mag_err(m_obs, m5)

    def _observe_visits(v, model, t0, rng, **model_kw):
        t = v["mjd"].values - t0
        m_true = np.full(len(v), np.nan)
        for b in BANDS:
            sel = v["band"].values == b
            if sel.any():
                m_true[sel] = model(t[sel], b, **model_kw)
        keep = np.isfinite(m_true)
        lc = v.loc[keep, ["mjd", "night", "band", "m5"]].copy()
        lc["t"], lc["m_true"] = t[keep], m_true[keep]
        lc["m_obs"], lc["m_err"] = _measure(lc["m_true"].values, lc["m5"].values, rng)
        lc["detected"] = lc["m_obs"] < lc["m5"]      # S/N > 5; NaN compares False
        return lc[LC_COLUMNS].reset_index(drop=True)

    def observe(preset, model, t0, seed=367, **model_kw):
        \"\"\"Sample model(t, band) at every simulated visit to `preset`, with noise.

        t0 is the MJD of the event (t = mjd - t0). Returns one row per visit where the model is
        defined: true and observed magnitude, error, and whether it was detected (S/N > 5).
        \"\"\"
        v = visits_at(preset)
        if len(v) == 0:
            _no_visits(preset)
            return pd.DataFrame(columns=LC_COLUMNS)
        return _observe_visits(v, model, t0, np.random.default_rng(seed), **model_kw)

    def show_lightcurve(preset="COSMOS", model=kilonova, t0=61500.0, window=(-5, 20), ax=None,
                        seed=367, **model_kw):
        \"\"\"Plot observe(...) : detections with error bars, non-detections as m5 upper limits.\"\"\"
        lc = observe(preset, model, t0, seed=seed, **model_kw)
        if len(lc) == 0:
            return lc
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        tt = np.linspace(window[0], window[1], 400)
        for b in BANDS:
            ax.plot(tt, model(tt, b, **model_kw), color=BAND_COLORS[b], lw=0.8, alpha=0.5)
            s = lc[(lc["band"] == b) & lc["t"].between(*window)]
            det, lim = s[s["detected"]], s[~s["detected"]]
            ax.errorbar(det["t"], det["m_obs"], det["m_err"], fmt="o", ms=4,
                        color=BAND_COLORS[b], label=b)
            ax.scatter(lim["t"], lim["m5"], marker="v", s=18, color=BAND_COLORS[b], alpha=0.4)
        win = lc[lc["t"].between(*window)]
        faint = np.nanmax(np.r_[win["m5"].values, win["m_obs"].values, 24.0])
        bright = np.nanmin(np.r_[win["m_true"].values, faint - 3])
        ax.set_ylim(faint + 0.5, bright - 0.5)       # faint at the bottom
        ax.set_xlim(*window)
        ax.set_xlabel(f"days since t0 = MJD {t0:.1f}")
        ax.set_ylabel("magnitude")
        ax.set_title(f"{getattr(model, '__name__', 'model')} at {preset}: {int(win['detected'].sum())} "
                     f"detections in {len(win)} visits (lines: model; triangles: m5 of non-detections)",
                     fontsize=9)
        ax.legend(fontsize=7, ncol=6, loc="lower left")
        if show:
            plt.tight_layout()
            plt.show()
        return lc

    show_lightcurve("COSMOS", kilonova, t0=61500.0);
    """),
    _c("md", """
    Pick a position, a model and a start time. The direct call is
    `show_lightcurve("main-1", sn_ia, t0=61500, window=(-30, 80))`; that's the one to use for a
    hand-in figure.
    """),
    _c("code", """
    from ipywidgets import FloatSlider

    def _lightcurve_widget(preset, model, t0):
        show_lightcurve(preset, model, t0, window=(-5, 20) if model is kilonova else (-30, 80))

    sliders(_lightcurve_widget,
             preset=Dropdown(options=list(presets.index), value="COSMOS", description="position"),
             model=Dropdown(options={"kilonova": kilonova, "SN Ia": sn_ia}, description="model"),
             t0=FloatSlider(value=61500, min=SURVEY_START, max=SURVEY_START + 3652, step=1,
                            description="t0 (MJD)", continuous_update=False,
                            layout={"width": "600px"}));
    """),
    _c("md", """
    ### How often do you catch it?

    One `t0` is one lucky or unlucky draw. `fraction_caught` drops the event at many random times
    across the ten years (so events during the off-season count as misses) and returns the
    fraction with at least one detection within `within_days` of `t0`.
    """),
    _c("code", """
    def fraction_caught(preset, model=kilonova, within_days=2.0, n_trials=1000, seed=367,
                        **model_kw):
        \"\"\"Fraction of random event times (uniform over 10 yr) detected within `within_days`.\"\"\"
        v = visits_at(preset)
        if len(v) == 0:
            _no_visits(preset)
            return 0.0
        rng = np.random.default_rng(seed)
        mjd = v["mjd"].values
        t0s = rng.uniform(SURVEY_START, SURVEY_START + 3652.5, n_trials)
        lo = np.searchsorted(mjd, t0s, side="left")
        hi = np.searchsorted(mjd, t0s + within_days, side="right")
        caught = 0
        for t0, a, b in zip(t0s, lo, hi):
            if b > a:
                lc = _observe_visits(v.iloc[a:b], model, t0, rng, **model_kw)
                caught += bool(lc["detected"].any())
        return caught / n_trials

    def catch_table(model=kilonova, within_days=2.0, **kw):
        \"\"\"fraction_caught for every preset that has visits.\"\"\"
        names = [n for n in presets.index if (points["preset"] == n).any()]
        return pd.Series({n: fraction_caught(n, model, within_days, **kw) for n in names},
                         name=f"fraction caught within {within_days:g} d").sort_values(ascending=False)

    catch_table(kilonova, within_days=2)
    """),
    _c("md", """
    **Catching a kilonova.** Which preset catches the kilonova within 2 days? Do the deep drilling
    fields win? Why or why not? What changes at `d_mpc=400` (try
    `catch_table(kilonova, 2, d_mpc=400)`) or with a longer window?

    ### Optional · A lensed quasar: measuring a time delay

    Skip this if you're short on time; it previews Phil Marshall's Oct 26 session.

    A strongly lensed quasar shows two (or more) images of the same quasar. The light paths differ
    in length, so image B repeats image A's flickering after a delay $\\Delta t$ (and is fainter by
    $\\Delta m$). Measure $\\Delta t$ and, with a lens model, you measure $H_0$.

    **Quasar variability** as a damped random walk: on a 1-day grid,

    $$x_{i+1} = x_i\\,e^{-\\Delta/\\tau} + \\sigma\\sqrt{1-e^{-2\\Delta/\\tau}}\\;\\mathcal{N}(0,1),
    \\qquad \\sigma = \\mathrm{SF}_\\infty/\\sqrt{2}$$

    with $\\tau = 200$ d and $\\mathrm{SF}_\\infty = 0.2$ mag, typical quasar values (e.g. MacLeod et
    al. 2010), around a mean of 21 mag. Image A is $21 + x(t)$; image B is
    $21 + x(t - \\Delta t) + \\Delta m$, with $\\Delta t = 30$ d and $\\Delta m = 0.5$ mag (toy
    values). Both images are sampled at the same visits; the variability is the same in every band
    (also a toy).

    **Where does this break?** Real lensed-quasar images are also microlensed by stars in the lens
    galaxy, which adds variability to one image but not the other. And the images are
    arcseconds apart, so they must be deblended. How would each of these fool the estimator below?

    **The estimator** (a toy). For each trial shift $s$ from −100 to +100 days, shift A by $s$,
    interpolate it at B's times (only *within* a season, never across a gap), fit a constant
    offset, and compute the mean squared difference. The best $s$ is the estimate. The real problem,
    with microlensing and honest error bars, is the subject of Phil Marshall's Oct 26 session.
    """),
    _c("code", """
    QSO_COLUMNS = ["mjd", "night", "band", "m5", "mA_true", "mB_true", "mA", "mB", "errA", "errB"]

    def drw(t_grid, tau=200.0, sf=0.2, rng=None):
        \"\"\"Damped random walk (zero mean) on an evenly spaced grid; SF_inf = sqrt(2) * sigma.\"\"\"
        rng = np.random.default_rng(rng)
        sigma = sf / np.sqrt(2)
        a = np.exp(-np.diff(t_grid) / tau)
        kicks = rng.normal(0.0, 1.0, len(t_grid))
        x = np.empty(len(t_grid))
        x[0] = sigma * kicks[0]                   # start in the stationary distribution
        for i in range(1, len(t_grid)):
            x[i] = a[i - 1] * x[i - 1] + sigma * np.sqrt(1 - a[i - 1] ** 2) * kicks[i]
        return x

    def lensed_quasar(preset=None, dt=30.0, dmag=0.5, tau=200.0, sf=0.2, mean=21.0,
                      noise=True, mjd=None, seed=367):
        \"\"\"Two images of a DRW quasar, B = A delayed by dt and fainter by dmag, at the same visits.

        Give a preset name to use that position's simulated visits, or mjd=array for a cadence
        of your own (then every visit is r band with m5 = 24.0). noise=False gives exact magnitudes.
        \"\"\"
        if mjd is None:
            v = visits_at(preset)
            if len(v) == 0:
                _no_visits(preset)
                return pd.DataFrame(columns=QSO_COLUMNS)
        else:
            mjd = np.sort(np.asarray(mjd, float))
            v = pd.DataFrame({"mjd": mjd, "night": np.floor(mjd - SURVEY_START).astype(int),
                              "band": "r", "m5": 24.0})
        rng = np.random.default_rng(seed)
        t = v["mjd"].values
        grid = np.arange(np.floor(t.min()) - abs(dt) - 2, np.ceil(t.max()) + abs(dt) + 2, 1.0)
        x = drw(grid, tau, sf, rng)
        lc = v[["mjd", "night", "band", "m5"]].copy().reset_index(drop=True)
        lc["mA_true"] = mean + np.interp(t, grid, x)
        lc["mB_true"] = mean + np.interp(t - dt, grid, x) + dmag
        for img in "AB":
            if noise:
                m_obs, err = _measure(lc[f"m{img}_true"].values, lc["m5"].values, rng)
                m_obs[~(m_obs < lc["m5"].values)] = np.nan   # keep detections only
                lc[f"m{img}"], lc[f"err{img}"] = m_obs, err
            else:
                lc[f"m{img}"], lc[f"err{img}"] = lc[f"m{img}_true"], 0.0
        return lc[QSO_COLUMNS]

    def estimate_delay(lc, shifts=np.arange(-100, 101), min_overlap=10, return_curve=False):
        \"\"\"Toy delay estimator: the shift s minimizing the mean squared B - (A(t - s) + offset).

        Nights are averaged over bands; A is interpolated only within a season. Returns the best
        shift in days (NaN if no shift has min_overlap overlapping nights), or
        (best, shifts, mse) if return_curve=True.
        \"\"\"
        shifts = np.asarray(shifts, float)
        mse = np.full(len(shifts), np.nan)
        d = lc.dropna(subset=["mA", "mB"]) if len(lc) else lc
        if len(d) >= min_overlap:
            g = d.groupby("night").agg(mjd=("mjd", "mean"), A=("mA", "mean"), B=("mB", "mean"))
            tn, A, B = g["mjd"].values, g["A"].values, g["B"].values
            seas = seasons(tn)
            for k, s in enumerate(shifts):
                q = tn - s                          # B at time t looks like A at t - s
                ok = np.zeros(len(q), bool)
                for a, b in seas:
                    ok |= (q >= a) & (q <= b)
                if ok.sum() >= min_overlap:
                    r = B[ok] - np.interp(q[ok], tn, A)
                    mse[k] = np.mean((r - r.mean()) ** 2)
        best = float(shifts[np.nanargmin(mse)]) if np.isfinite(mse).any() else np.nan
        return (best, shifts, mse) if return_curve else best

    def show_delay(preset="RXJ1131-1231", dt=30.0, noise=True, seed=367, **qso_kw):
        \"\"\"Light curves of both images at a preset, and the estimator's curve. Returns the estimate.\"\"\"
        lc = lensed_quasar(preset, dt=dt, noise=noise, seed=seed, **qso_kw)
        if len(lc) == 0:
            return np.nan
        best, shifts, mse = estimate_delay(lc, return_curve=True)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1]})
        for img, c in [("A", "tab:blue"), ("B", "tab:red")]:
            a1.errorbar(lc["mjd"], lc[f"m{img}"], lc[f"err{img}"], fmt=".", ms=3, color=c,
                        label=f"image {img}")
        a1.invert_yaxis()
        a1.set_xlabel("MJD")
        a1.set_ylabel("magnitude (all bands)")
        a1.legend(fontsize=8)
        a1.set_title(f"{preset}: {lc['mA'].notna().sum()} visits with image A detected", fontsize=9)
        a2.plot(shifts, mse, color="k")
        a2.axvline(dt, color="tab:green", ls="--", label=f"true delay {dt:g} d")
        a2.axvline(best, color="tab:orange", label=f"estimate {best:g} d")
        a2.set_xlabel("trial shift s (days)")
        a2.set_ylabel("mean squared difference (mag²)")
        a2.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        print(f"True delay {dt:g} d; estimated {best:g} d ({'with' if noise else 'without'} noise)")
        return best

    show_delay("RXJ1131-1231");
    """),
    _c("code", """
    from ipywidgets import Checkbox, fixed

    _lens_names = [n for n in presets.index if n.startswith(("RXJ", "HE"))] + \\
                  [n for n in presets.index if not n.startswith(("RXJ", "HE"))]
    def _delay_widget(preset, dt, noise, seed):
        show_delay(preset, dt=dt, noise=noise, seed=seed)   # don't echo the returned estimate

    sliders(_delay_widget,
             preset=Dropdown(options=_lens_names, value="RXJ1131-1231", description="position"),
             dt=FloatSlider(value=30, min=-90, max=90, step=5, description="true Δt (d)",
                            continuous_update=False),
             noise=Checkbox(value=True, description="noise"),
             seed=fixed(367));
    """),
    _c("md", """
    **Is the delay recovered at RXJ1131?** What limits it: season gaps or noise? Try
    `show_delay("RXJ1131-1231", noise=False)`, other seeds (`seed=1`, `seed=2`, ...), and a
    cadence of your own, e.g.
    `estimate_delay(lensed_quasar(mjd=SURVEY_START + np.arange(0, 3650, 3)))`.
    """),
]

# ---------------------------------------------------------------------------
# Part 4: go deeper, and the hand-in
# ---------------------------------------------------------------------------
PART4_CELLS = [
    _c("md", """
    ## Part 4 · Go deeper (optional)

    Everything in this notebook came from small extracts of one simulation. Here is where the real
    tools and data live.

    - **rubin_sim**, the survey-simulation and metrics package: docs at
      <https://rubin-sim.lsst.io>, with its data download at
      <https://rubin-sim.lsst.io/data-download.html>.
    - **The full simulation used here** (baseline v5.3.3, 10 years, an SQLite database of about
      750 MB):
      <https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/baseline_v5.3.3_10yrs.db>.
      Alternative v5.3 survey strategies are in the same directory tree:
      <https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/>.
    - **MAF tutorials** (the Metrics Analysis Framework; start with 01–03):
      <https://github.com/lsst/rubin_sim_notebooks/tree/main/maf/tutorial>.
      **Writing your own metric** is tutorial `02_Writing_Metrics`. Scheduler tutorials:
      <https://github.com/lsst/rubin_sim_notebooks/tree/main/scheduler>. Science-metric notebooks
      (e.g. KNeMetric for kilonovae, TDC_TimeDelayAccuracy for lensed quasars, SNIa):
      <https://github.com/lsst/rubin_sim_notebooks/tree/main/maf/science>.
    - **Be the SCOC.** The Survey Cadence Optimization Committee chooses among strategies by
      comparing metrics like these across many simulations. Precomputed metric results for the
      simulations: <https://usdf-maf.slac.stanford.edu/>. The survey-strategy hub:
      <https://survey-strategy.lsst.io>. Pick your science case's metric and find the strategy
      that does best for it. What does that strategy cost someone else?
    - **Plan vs. reality.** Phil Marshall (Sep 23): in the first ~15 days of the LSST, the survey
      took 7,267 science visits out of 17,273 possible (bad weather on 5 nights). On July 14 the
      summit was evacuated for the worst storm in 50+ years; the target is to be back on sky in
      mid-October. **What does your science lose?** Rerun your plan with fewer visits, e.g.
      `plan_survey(area, t_visit, split, budget=0.6 * BUDGET)`, and compare it with your
      full-budget plan.
    - **What was actually observed.** Nightly scheduler reports compare each night's plan with what
      happened: <https://s3df.slac.stanford.edu/data/rubin/sim-data/schedview/reports/>.
    - **On the Rubin Science Platform** (once your accounts are live): the DP1 Visit table
      tutorial, 201_10, gives the measured seeing and depth of the LSSTComCam commissioning visits
      in DP1. Compare them with the simulated `points` file (different camera, and pre-survey).
    """),
    _c("md", """
    ## Hand-in

    Due Monday Oct 5 on Canvas. Turn in: (1) one figure you made (use a direct function call, not
    a slider); (2) one paragraph: what did your team's science case need, and does the baseline
    survey give it?; (3) a sentence on any AI tools you used. Upload the .ipynb or a PDF. Graded
    complete/incomplete.

    Make your figure in the code cell below. Write your paragraph and the AI-tools sentence in the
    last cell.
    """),
    _c("code", """
    # Your hand-in figure (use a direct function call, not a slider), e.g.
    # show_lightcurve("main-1", sn_ia, t0=61500, window=(-30, 80))
    """),
    _c("md", """
    *Your paragraph here.*
    """),
]


# ===========================================================================
# Notebook B: Week 2B · Camera and telescope → science
# ===========================================================================
# Cell text is written as raw strings (r"""...""" for markdown, r'''...''' for code, so code
# cells can hold """docstrings"""), so LaTeX backslashes need no doubling here.

# ---------------------------------------------------------------------------
# Notebook B intro and setup
# ---------------------------------------------------------------------------
B_INTRO_CELLS = [
    _c("md", r"""
    # Week 2B · Camera and telescope → science

    Monday's (Sep 28) "Design your own survey" worksheet handed you a single-visit depth,
    r ≈ 24.7, as a given. In Notebook A the simulated survey's median r-band visit reached only
    24.06. This notebook builds that depth from the hardware up: mirrors, lenses, filters, detector,
    atmosphere, and the night sky. It then follows the camera through to science: seeing and weak
    lensing, and the layout of the focal plane. It goes with Aaron Roodman's lecture on Wednesday
    (Sep 30), and it compares with the first on-sky numbers from Phil Marshall's Sep 23 overview.

    **How to use it.** In Colab, choose *Runtime → Run all*, then move the sliders. Every slider
    wraps a plain function you can call directly. Use the direct call for your hand-in figure,
    because sliders don't show up in PDFs.

    **Core path vs. go deeper.** The core path takes about 30–45 minutes. The *Go deeper* boxes are
    optional.

    *Data note: the throughput curves are the Rubin project's engineering model of the as-built
    hardware (public, from GitHub), and the survey numbers are from public simulations. None of this
    is measured Rubin data.*
    """),
    _c("code", r'''
    # Setup. Everything here ships with Colab.
    import os
    from pathlib import Path
    import urllib.request

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    DATA_URL = "https://raw.githubusercontent.com/rhw/phys367/main/week2/data/"
    DATA_DIR = Path("data")
    DATA_FILES = {
        "throughputs": "throughputs.csv",
        "darksky": "darksky.csv",
        "m5_reference": "m5_reference.csv",
    }

    def _fetch(name):
        """Local path of a data file; downloads it from GitHub once if it isn't in data/."""
        fname = DATA_FILES.get(name, name)
        path = DATA_DIR / fname
        if not path.exists():
            DATA_DIR.mkdir(exist_ok=True)
            print(f"downloading {fname} ...")
            part = path.with_suffix(path.suffix + ".part")   # don't trust a half-finished download
            urllib.request.urlretrieve(DATA_URL + fname, part)
            part.rename(path)
        return path

    def load(name):
        """Load a data file by short name ('throughputs', 'darksky', 'm5_reference') or file name.

        Uses the local data/ directory if the file is there; otherwise downloads it from
        GitHub once and caches it in data/. Lines starting with '#' are the file's provenance
        header; read them with header(name).
        """
        path = _fetch(name)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, comment="#")

    def header(name):
        """The provenance header (the '#' lines) of a data file, as one string."""
        with open(_fetch(name)) as f:
            return "".join(line for line in f if line.startswith("#"))

    from ipywidgets import interact, interact_manual

    def sliders(func, **controls):
        """Attach sliders and menus to func, like ipywidgets.interact.

        Automated runs with no screen (our tests set PHYS367_HEADLESS=1) build the same widgets
        but don't run func, because live widget output can stall a headless run.
        """
        if os.environ.get("PHYS367_HEADLESS") == "1":
            return interact_manual(func, **controls)
        return interact(func, **controls)

    BANDS = list("ugrizy")
    BAND_COLORS = {"u": "tab:purple", "g": "tab:blue", "r": "tab:green",
                   "i": "tab:orange", "z": "tab:red", "y": "tab:brown"}
    '''),
]

# ---------------------------------------------------------------------------
# Notebook B, Part 1: depth from first principles
# ---------------------------------------------------------------------------
B_PART1_CELLS = [
    _c("md", r"""
    ## Part 1 · Depth from first principles

    The 5σ limiting magnitude $m_5$ is the brightness of a point source that a single visit detects
    at signal-to-noise 5. It depends on how many photons from the source reach the detector, how
    many sky photons land in the same patch of pixels, and how much noise the camera adds. You will
    build it in four steps: throughput, source photons, sky photons, and signal-to-noise.

    ### 1. The throughput chain

    The fraction of photons at wavelength $\lambda$ that survive to become photoelectrons is a
    product of components:

    $$S_b(\lambda) = \underbrace{M_1 M_2 M_3}_{\text{mirrors}}\;\underbrace{L_1 L_2 L_3}_{\text{lenses}}
    \;\underbrace{\text{QE}}_{\text{detector}}\;\underbrace{F_b}_{\text{filter}},
    \qquad T_b(\lambda, X) = S_b(\lambda)\,A(\lambda, X)$$

    where $S_b$ is the *hardware* throughput in band $b$ and $A(\lambda, X)$ is the atmosphere's
    transmission at airmass $X$.

    *Source: the Rubin systems-engineering throughput model, `lsst-pst/syseng_throughputs` at commit
    `00570b3d39` (release 1.9, with silver coatings on all three mirrors). We rebuilt its
    component products on a 1 nm grid (`tools/make_throughputs_extract.py` in the course repo);
    they match syseng's own `buildHardwareAndSystem` output to about 1 part in 10⁶. Each component
    includes syseng's loss terms (contamination, condensation), and the detector is the minimum of
    the two CCD vendors' QE curves.* The atmosphere is tabulated at X = 1.0
    (`atmos_10_aerosol.dat`, syseng's standard X = 1.0 atmosphere with an aerosol component) and
    X = 1.2 (`pachonModtranAtm_12_aerosol.dat`), both from syseng. For other airmasses we
    interpolate (and beyond X = 1.2, extrapolate) $\ln A$ linearly in airmass between the two curves. That interpolation is our choice, not syseng's.

    **Where does this break?** Every factor here is a single curve for the whole focal plane and
    the whole survey. Which of them would you expect to change across the field of view, from night
    to night, or over ten years?
    """),
    _c("code", r'''
    SYSENG_SHA = "00570b3d391b5a8671d55341ed51b5e534dab6b4"   # syseng_throughputs, release 1.9

    thr = load("throughputs")
    dark_sed = load("darksky")
    m5_ref = load("m5_reference").set_index("band")
    print(header("throughputs"))

    WAVE = thr["wavelength_nm"].to_numpy(dtype=float)   # nm, 1 nm grid, 300-1100 nm
    DLAM = 1.0                                           # nm

    def hardware(band):
        """Hardware throughput S_b (mirrors x lenses x detector x filter), no atmosphere."""
        return (thr["mirrors"] * thr["lenses"] * thr["detector"] * thr[f"filter_{band}"]).to_numpy()

    def atmosphere(airmass=1.0):
        """Atmospheric transmission at this airmass.

        Exact at X = 1.0 and 1.2 (the two tabulated curves); elsewhere ln(A) is interpolated
        (or extrapolated) linearly in X between them.
        """
        if airmass < 1.0:
            raise ValueError("airmass must be >= 1 (1 = zenith)")
        a10 = thr["atmos_X1.0"].to_numpy()
        a12 = thr["atmos_X1.2"].to_numpy()
        ratio = np.divide(a12, a10, out=np.zeros_like(a10), where=a10 > 0)
        return np.clip(a10 * ratio ** ((airmass - 1.0) / 0.2), 0.0, 1.0)

    def system(band, airmass=1.0):
        """Total throughput T_b = hardware x atmosphere, on the WAVE grid."""
        return hardware(band) * atmosphere(airmass)
    '''),
    _c("code", r'''
    def show_components(airmass=1.0):
        """Plot every component of the throughput chain, then the total system curve per band."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        keep = WAVE >= 320        # below ~320 nm nothing gets through (and the lens file has an edge artifact)
        w = WAVE[keep]
        for col, label, style in [("mirrors", "mirrors M1·M2·M3", "-"),
                                  ("lenses", "lenses L1·L2·L3", "--"),
                                  ("detector", "detector QE", "-.")]:
            ax1.plot(w, thr[col].to_numpy()[keep], "k", ls=style, lw=1.2, label=label)
        ax1.plot(w, atmosphere(airmass)[keep], color="0.6", lw=1.5, label=f"atmosphere, X = {airmass:g}")
        for b in BANDS:
            ax1.plot(w, thr[f"filter_{b}"].to_numpy()[keep], color=BAND_COLORS[b], lw=1, alpha=0.7)
        ax1.set_ylabel("throughput")
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="lower right", fontsize=8)
        ax1.set_title("Components (colored: the six filters)")
        for b in BANDS:
            ax2.plot(w, system(b, airmass)[keep], color=BAND_COLORS[b], label=b)
            ax2.plot(w, hardware(b)[keep], color=BAND_COLORS[b], lw=0.7, ls=":")
        ax2.set_ylabel("total throughput")
        ax2.set_xlabel("wavelength (nm)")
        ax2.set_title(f"System = product of all of the above (dotted: hardware only), X = {airmass:g}")
        ax2.legend(ncol=6, fontsize=8, loc="upper right")
        ax2.set_ylim(0, 0.7)
        plt.tight_layout()
        plt.show()

    show_components()
    '''),
    _c("md", r"""
    **Read the plot.** Which component limits the u band? Which one sets the red edge of y? Where
    does the atmosphere matter most?

    ### 2. Photons from a source: the zeropoint

    An AB magnitude is defined against a flat spectrum, $f_\nu = 3631$ Jy at $m = 0$. The rate of
    detected photons (photoelectrons) from such a source is

    $$N_0 = A_\text{eff}\int \frac{f_\nu\,T_b(\lambda)}{h\lambda}\,d\lambda ,
    \qquad N(m) = N_0\,10^{-0.4\,m}\ \ \text{per second}$$

    where $A_\text{eff}$ is the effective collecting area. Divide $f_\nu$ by the photon energy
    $h\nu$ to count photons, and use $d\nu/\nu = d\lambda/\lambda$ to integrate in wavelength.
    $2.5\log_{10}N_0$ is the magnitude that gives one photoelectron per second.

    *Source: AB system (Oke & Gunn 1983); the same sum as rubin_sim's `Sed.calc_adu`, which
    syseng uses. The collecting area is rubin_sim's default, $A_\text{eff} = \pi(6.423\ \text{m}/2)^2$
    = 32.4 m². Rubin's primary mirror is 8.4 m across; the central obscuration makes the effective aperture
    smaller (Ivezić et al. 2019 quote 6.5 m effective; rubin_sim's slightly smaller value is used
    throughout).*

    **Where does this break?** Real stars and galaxies don't have flat $f_\nu$. What happens to
    the photon count for a very red or very blue source with the same AB magnitude?
    """),
    _c("code", r'''
    H_ERG_S = 6.626068e-27           # Planck constant, erg s (rubin_sim's value)
    C_NM_S = 2.99792458e17           # speed of light, nm/s
    AB_ZERO_FNU = 3631e-23           # erg s^-1 cm^-2 Hz^-1 (3631 Jy)
    RUBIN_AREA_M2 = np.pi * (6.423 / 2) ** 2   # rubin_sim PhotometricParameters default

    def zeropoint(band, airmass=1.0, area_m2=None, throughput=None):
        """Photoelectrons per second from an m = 0 AB source (flat f_nu = 3631 Jy).

        Uses the full system throughput (hardware x atmosphere) unless you pass `throughput`.
        """
        area_cm2 = (RUBIN_AREA_M2 if area_m2 is None else area_m2) * 1e4
        T = system(band, airmass) if throughput is None else throughput
        return area_cm2 * np.sum(AB_ZERO_FNU * T / (H_ERG_S * WAVE)) * DLAM

    print("Zeropoint: magnitude giving 1 photoelectron/s (X = 1.0)")
    print(f"{'band':>4} {'this notebook':>14} {'syseng makeM5':>14}")
    for b in BANDS:
        print(f"{b:>4} {2.5 * np.log10(zeropoint(b)):14.3f} {m5_ref.loc[b, 'Zp_t']:14.3f}")
    '''),
    _c("md", r"""
    ### 3. Photons from the sky

    The night sky is not dark. Airglow, zodiacal light and scattered light give a surface
    brightness $F_\lambda^\text{sky}$ per arcsec². The rate of sky photoelectrons per arcsec² is

    $$B = A_\text{eff}\int F_\lambda^\text{sky}\,\frac{\lambda}{hc}\,S_b(\lambda)\,d\lambda$$

    through the **hardware only**: the sky glow is made in the atmosphere, and syseng's dark-sky
    spectrum already describes the light arriving at the telescope.

    *Source: `siteProperties/darksky.dat` in syseng (same commit), a dark, moonless sky at zenith.
    The file's header gives units of erg s⁻¹ cm⁻² nm⁻¹ only. We treat it as per arcsec² because
    that is how syseng's `makeM5` uses it (it multiplies by the pixel area in arcsec²).* To try a
    brighter sky, `sky_counts(band, sky_mag=...)` rescales the same spectrum to the surface brightness
    you give it, in mag/arcsec².

    **Where does this break?** This is one dark sky at zenith. What does the Moon do, and would it
    matter equally in every band? What about twilight, or looking toward the Galactic plane or the
    ecliptic?
    """),
    _c("code", r'''
    def sky_counts(band, sky_mag=None, area_m2=None):
        """Sky photoelectrons per second per arcsec^2, through the hardware (no atmosphere).

        sky_mag=None uses the dark-sky spectrum as is; otherwise the same spectrum is rescaled
        to that surface brightness (AB mag/arcsec^2 through the hardware).
        """
        area_cm2 = (RUBIN_AREA_M2 if area_m2 is None else area_m2) * 1e4
        flam = dark_sed["flambda_erg_s_cm2_nm"].to_numpy()
        dark = area_cm2 * np.sum(flam * WAVE / (H_ERG_S * C_NM_S) * hardware(band)) * DLAM
        if sky_mag is None:
            return dark
        return dark * 10 ** (-0.4 * (sky_mag - sky_mag_dark(band)))

    def sky_mag_dark(band):
        """Surface brightness of the dark-sky spectrum in this band (AB mag/arcsec^2)."""
        return -2.5 * np.log10(sky_counts(band) / zeropoint(band, throughput=hardware(band)))

    print("Dark-sky surface brightness at zenith (mag/arcsec^2)")
    print(f"{'band':>4} {'this notebook':>14} {'syseng makeM5':>14} {'sky e-/s/pixel':>15}")
    for b in BANDS:
        print(f"{b:>4} {sky_mag_dark(b):14.3f} {m5_ref.loc[b, 'skyMag']:14.3f} "
              f"{sky_counts(b) * 0.2**2:15.1f}")
    '''),
    _c("md", r"""
    ### 4. Signal-to-noise and $m_5$

    A visit is $n_\text{exp}$ exposures of $t_\text{exp}$ seconds each, so the total open-shutter time is
    $t = n_\text{exp}t_\text{exp}$. For a point source giving $C$ photoelectrons, spread over a PSF
    that covers $n_\text{eff}$ pixels,

    $$\frac{S}{N} = \frac{C}{\sqrt{C + n_\text{eff}\,\big(B\,p^2\,t + n_\text{exp}\,\sigma_\text{read}^2 + D\,t\big)}},
    \qquad n_\text{eff} = 2.266\left(\frac{\text{FWHM}_\text{eff}}{p}\right)^2$$

    Here $p$ is the pixel scale, $\sigma_\text{read}$ the read noise per pixel per exposure, and
    $D$ the dark current. Set $S/N = 5$ and solve the quadratic for $C$. Then
    $m_5 = -2.5\log_{10}\!\big[C/(N_0\,t)\big]$.

    For a Gaussian PSF of width $\sigma$, $n_\text{eff} = 4\pi\sigma^2/p^2 = 2.266\,(\text{FWHM}/p)^2$:
    the number of pixels whose noise an optimal (PSF-weighted) measurement effectively averages over.

    *Source: the LSST signal-to-noise document LSE-40 (eq. 27 for $n_\text{eff}$, eq. 45 for
    $m_5$), as coded in rubin_sim's `signaltonoise.calc_m5` and used by syseng's `makeM5`.*

    | parameter | value | source |
    |---|---|---|
    | effective area | π(6.423 m / 2)² = 32.4 m² | rubin_sim 2.6.2 `phot_utils/photometric_parameters.py`, `DefaultPhotometricParameters` |
    | pixel scale $p$ | 0.2″ | same file |
    | read noise | 8.8 e⁻ per pixel per exposure | same file (from the camera spec, LSE-30) |
    | dark current | 0.2 e⁻/s per pixel | same file |
    | gain | 1 (we count electrons) | syseng `makeM5` sets gain = 1; rubin_sim's default is 2.3 e⁻/ADU, which cancels when you work in electrons |
    | FWHM$_\text{eff}$ | u 0.92, g 0.87, r 0.83, i 0.80, z 0.78, y 0.76″ at zenith, × $X^{0.6}$ | syseng `m5Utils.fwhm_eff_zenith` (fiducial seeing from the LSST overview paper, Ivezić et al. 2019) |
    | visit | 2 × 15 s in every band | the reference convention used here; syseng's `makeM5` default is 1 × 30 s in u |

    The reference table `m5_reference.csv` was made by running syseng's own `makeM5` at the pinned
    commit with exactly these settings at X = 1.0 (see its header). Your function should agree with it.

    **Where does this break?** The formula assumes a faint point source on a smooth, perfectly
    subtracted sky, with a Gaussian-like PSF. Name a kind of source, or a part of the sky, where
    each of those assumptions fails. What would happen to $m_5$?
    """),
    _c("code", r'''
    PIXEL_SCALE = 0.2                       # arcsec per pixel (rubin_sim default)
    FWHM_EFF_ZENITH = {"u": 0.92, "g": 0.87, "r": 0.83, "i": 0.80, "z": 0.78, "y": 0.76}  # arcsec, syseng

    def m5(band, t_exp=15, n_exp=2, fwhm_eff=None, sky_mag=None, read_noise=8.8, dark=0.2,
           area_m2=None, airmass=1.0):
        """5-sigma point-source depth (AB mag) of one visit of n_exp exposures of t_exp seconds.

        fwhm_eff=None uses syseng's fiducial zenith FWHM_eff x airmass^0.6 (arcsec).
        sky_mag=None uses the dark-sky spectrum; otherwise the sky in mag/arcsec^2.
        read_noise in e-/pixel/exposure; dark in e-/pixel/s; area_m2=None means Rubin's 32.4 m^2.
        """
        if fwhm_eff is None:
            fwhm_eff = FWHM_EFF_ZENITH[band] * airmass ** 0.6
        t = t_exp * n_exp                                        # open-shutter seconds
        n_eff = 2.266 * (fwhm_eff / PIXEL_SCALE) ** 2            # pixels in the PSF (LSE-40 eq. 27)
        sky_per_pixel = sky_counts(band, sky_mag, area_m2) * PIXEL_SCALE**2 * t
        var_bkg = n_eff * (sky_per_pixel + n_exp * read_noise**2 + dark * t)
        snr = 5.0
        counts_5sig = snr**2 / 2 + np.sqrt(snr**4 / 4 + snr**2 * var_bkg)   # solves S/N = 5 for C
        return -2.5 * np.log10(counts_5sig / (zeropoint(band, airmass, area_m2) * t))

    print("Single-visit m5, 2 x 15 s, X = 1.0, dark sky, fiducial seeing")
    print(f"{'band':>4} {'this notebook':>14} {'syseng makeM5':>14} {'difference':>11}")
    for b in BANDS:
        mine, ref = m5(b), m5_ref.loc[b, "m5"]
        print(f"{b:>4} {mine:14.3f} {ref:14.3f} {mine - ref:+11.4f}")
    '''),
    _c("md", r"""
    ### Design → as-built → as-scheduled

    Three numbers for the same thing, the depth of one r-band visit:

    | r-band $m_5$ | what it is | source |
    |---|---|---|
    | **24.7** | the design ("fiducial") depth | Ivezić et al. 2019, Table 1 (SRD design specification, fiducial zenith); Monday's worksheet |
    | **≈ 24.48** | the as-built hardware (release 1.9) at the reference settings: 2 × 15 s, zenith, dark sky, FWHM$_\text{eff}$ 0.83″ | this notebook's `m5("r")`; syseng `makeM5` gives 24.479 |
    | **24.06** | the median r-band visit in the simulated ten-year survey | baseline v5.3.3 simulation (Notebook A); median seeing 1.03″, airmass 1.18, sky 21.0 mag/arcsec² |
    """),
    _c("code", r'''
    M5_DESIGN_R = 24.7        # Ivezic et al. 2019, Table 1 (SRD design specification, fiducial zenith; also the worksheet)
    M5_SIM_MEDIAN_R = 24.06   # median r-band visit, baseline v5.3.3 simulation (Notebook A)

    print(f"design (Ivezic+2019 Table 1, SRD spec):    {M5_DESIGN_R:.2f}")
    print(f"as-built hardware (this notebook, m5('r')): {m5('r'):.2f}")
    print(f"median simulated visit (baseline v5.3.3): {M5_SIM_MEDIAN_R:.2f}")
    '''),
    _c("md", r"""
    **Account for each step.** From 24.7 to ≈ 24.48: what went into 24.7, and which of those inputs
    differ from the as-built calculation? From ≈ 24.48 to 24.06: call `m5("r", ...)`
    with the simulation's median seeing, airmass and sky. How much of the drop does each one explain
    on its own? Is the depth at the median conditions the same thing as the median depth?

    **Your science.** Your plan in Notebook A used 24.7. Which of your numbers change if you use
    24.06 instead, and by how much?
    """),
    _c("md", r"""
    ### Read noise, and why the u band is different

    The reference visit is two 15 s exposures. syseng's own `makeM5` defaults to a single 30 s
    exposure in u, and two 15 s exposures in the other bands. The cell below computes, with your
    `m5()`, how much deeper one 30 s exposure is than two 15 s exposures in each band. Next to it:
    the sky electrons per pixel in one 15 s exposure, and syseng's `dCm_double` from the reference table: the depth that a visit of twice the
    exposure time still loses to camera noise, compared with a noiseless camera.
    """),
    _c("code", r'''
    RN = 8.8   # e- per pixel per exposure
    print(f"{'band':>4} {'sky e-/pix/15s':>15} {'read noise^2':>13} "
          f"{'m5(1x30s) - m5(2x15s)':>22} {'dCm_double (syseng)':>20}")
    for b in BANDS:
        sky_pix = sky_counts(b) * PIXEL_SCALE**2 * 15
        gain_30 = m5(b, t_exp=30, n_exp=1) - m5(b, t_exp=15, n_exp=2)
        print(f"{b:>4} {sky_pix:15.1f} {RN**2:13.1f} {gain_30:22.3f} "
              f"{m5_ref.loc[b, 'dCm_double']:20.3f}")
    '''),
    _c("md", r"""
    **Why u?** Why does the u band gain the most from one 30 s exposure, and lose the most to read
    noise? Compare the first two columns. What would you change about the camera, or the visit, to
    fix it, and what would each fix cost?

    **Where does this break?** Two exposures per visit let you reject cosmic rays and catch
    fast-moving objects. What do you give up with one 30 s exposure?
    """),
    _c("md", r"""
    ### Your camera, with sliders

    Change the telescope and camera. The aperture is the *effective* diameter (Rubin: 6.423 m).
    "Sky brighter by" makes the sky brighter than the dark sky by that many mag/arcsec². The seeing
    slider is an *offset* from the band's fiducial FWHM$_\text{eff}$ (0.92″ in u down to 0.76″ in
    y), so 0 reproduces the reference; +0.2 means 0.2″ worse than fiducial. The plot
    shows $m_5$ against open-shutter time for your camera and for the Rubin reference.

    If the sliders don't appear (e.g. in a PDF), call
    `show_m5(band, aperture_m=..., read_noise=..., fwhm_eff=..., sky_mag=..., t_exp=..., n_exp=...)`
    directly; that's also what to use for your hand-in figure.
    """),
    _c("code", r'''
    def show_m5(band="r", aperture_m=6.423, read_noise=8.8, fwhm_eff=None, sky_mag=None,
                t_exp=15, n_exp=2, airmass=1.0):
        """Print m5 for one visit with these settings and plot m5 vs exposure time. Returns m5."""
        area = np.pi * (aperture_m / 2) ** 2
        mine = m5(band, t_exp, n_exp, fwhm_eff, sky_mag, read_noise, 0.2, area, airmass)
        ref = m5(band)
        fwhm_used = FWHM_EFF_ZENITH[band] * airmass ** 0.6 if fwhm_eff is None else fwhm_eff
        sky_used = sky_mag_dark(band) if sky_mag is None else sky_mag
        print(f"{band} band: {n_exp} x {t_exp:g} s, aperture {aperture_m:.2f} m, read noise "
              f"{read_noise:.1f} e-, FWHM_eff {fwhm_used:.2f}\", sky {sky_used:.2f} mag/arcsec^2, "
              f"X = {airmass:.2f}")
        print(f"  m5 = {mine:.2f}   (Rubin reference, 2 x 15 s: {ref:.2f}; "
              f"difference {mine - ref:+.2f} mag)")

        t_grid = np.geomspace(1, 300, 60)                   # exposure time per exposure, s
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(t_grid * n_exp, [m5(band, t, n_exp, fwhm_eff, sky_mag, read_noise, 0.2, area, airmass)
                                 for t in t_grid], color=BAND_COLORS[band], label="your camera")
        ax.plot(t_grid * 2, [m5(band, t, 2) for t in t_grid], "k--", lw=1,
                label="Rubin reference (2 exposures)")
        ax.scatter([t_exp * n_exp], [mine], color=BAND_COLORS[band], zorder=3)
        ax.scatter([30], [ref], color="k", zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("open-shutter time per visit (s)")
        ax.set_ylabel(f"{band}-band $m_5$ (AB mag)")
        ax.legend(fontsize=8)
        plt.show()
        return mine
    '''),
    _c("code", r'''
    from ipywidgets import Dropdown, FloatSlider, IntSlider

    _W = dict(continuous_update=False, style={"description_width": "initial"})

    def _m5_widget(band, aperture_m, read_noise, seeing_offset, sky_brighter, t_exp, n_exp):
        fwhm_eff = FWHM_EFF_ZENITH[band] + seeing_offset     # offset from the band's fiducial
        show_m5(band, aperture_m, read_noise, fwhm_eff, sky_mag_dark(band) - sky_brighter,
                t_exp, n_exp)                               # don't echo the returned m5

    sliders(_m5_widget,
            band=Dropdown(options=BANDS, value="r", description="band"),
            aperture_m=FloatSlider(value=6.423, min=0.5, max=12.0, step=0.1,
                                   description="aperture (m)", **_W),
            read_noise=FloatSlider(value=8.8, min=0.0, max=30.0, step=0.2,
                                   description="read noise (e⁻)", **_W),
            seeing_offset=FloatSlider(value=0.0, min=-0.4, max=1.6, step=0.01,
                                      description="seeing relative to fiducial (″)", **_W),
            sky_brighter=FloatSlider(value=0.0, min=0.0, max=5.0, step=0.1,
                                     description="sky brighter by (mag)", **_W),
            t_exp=IntSlider(value=15, min=1, max=300, step=1,
                            description="t_exp (s)", **_W),
            n_exp=IntSlider(value=2, min=1, max=10, step=1,
                            description="n_exp", **_W));
    '''),
    _c("md", r"""
    **Where are you limited?** For each band, find the exposure time at which read noise stops
    mattering. Does it depend on the sky brightness? On the seeing?

    **Buy one thing.** You can afford one upgrade: 10% more aperture, half the read noise, or
    0.1″ better seeing. Which buys the most depth in r? In u? Is it the same answer?
    """),
    _c("md", r"""
    ### Étendue: aperture × field of view

    Depth per visit depends on the aperture. How much sky you can cover depends on the field of view.
    A survey telescope is rated by their product, the **étendue** $A_\text{eff}\,\Omega$.
    Here is a toy argument for why.

    1. If every visit is limited by sky noise, $m_5$ depends on $A_\text{eff}\,t$. So reaching the
       same depth takes $t \propto 1/A_\text{eff}$.
    2. Covering the same area takes $\propto 1/\Omega$ pointings.
    3. So the worksheet's ten-year budget, Area × visits × $(t/30\,\text{s}) \approx 1.5\times10^7$
       (Monday's worksheet, Notebook A), scales as $A_\text{eff}\,\Omega$ when every visit has
       Rubin's 30 s depth.

    *Sources: Rubin's field of view is 9.6 deg² (Ivezić et al. 2019); the effective area is rubin_sim's
    default, as above. This is a toy model.*

    **Where does this break?** The argument assumes every visit is sky-noise limited, the overhead
    per visit is negligible, and the telescope has the same number of good nights. Which of these
    fails first for a much smaller telescope? For a much smaller field?
    """),
    _c("code", r'''
    BUDGET = 1.5e7               # area x visits x (t/30 s) over 10 yr, Monday's worksheet
    RUBIN_DIAM_EFF_M = 6.423     # effective aperture, m (rubin_sim default area)
    RUBIN_FOV_DEG2 = 9.6         # field of view, deg^2 (Ivezic et al. 2019)

    def survey_budget(diam_eff_m, fov_deg2):
        """Toy: the worksheet's 10-yr budget, scaled by etendue relative to Rubin."""
        etendue = np.pi * (diam_eff_m / 2) ** 2 * fov_deg2
        etendue_rubin = np.pi * (RUBIN_DIAM_EFF_M / 2) ** 2 * RUBIN_FOV_DEG2
        return BUDGET * etendue / etendue_rubin

    print("Toy: ten-year budget at Rubin's 30 s single-visit depth, scaled by etendue")
    print(f"{'telescope':<28} {'etendue (m^2 deg^2)':>20} {'budget':>9} "
          f"{'visits/field @18k deg^2':>24} {'r m5, 2x15 s':>13}")
    for label, d, fov in [("Rubin (6.423 m, 9.6 deg^2)", RUBIN_DIAM_EFF_M, RUBIN_FOV_DEG2),
                          ("4 m aperture, 9.6 deg^2", 4.0, RUBIN_FOV_DEG2),
                          ("6.423 m, 1 deg^2 field", RUBIN_DIAM_EFF_M, 1.0)]:
        et = np.pi * (d / 2) ** 2 * fov
        b = survey_budget(d, fov)
        print(f"{label:<28} {et:20.0f} {b:9.2e} {b / 18000:24.0f} "
              f"{m5('r', area_m2=np.pi * (d / 2) ** 2):13.2f}")
    '''),
    _c("md", r"""
    **What would it do to the survey?** Take your team's plan from Notebook A. With a 4 m telescope,
    or with a 1 deg² field, would you rather keep the area and lose visits, or keep the visits and
    lose area? Which science case suffers most from each?

    > **Go deeper.** The atmosphere here interpolates between two tabulated curves. The textbook
    > alternative is Beer–Lambert: $A(\lambda, X) = A(\lambda, 1)^X$. Compute
    > `thr["atmos_X1.0"]**1.2` and compare it with `thr["atmos_X1.2"]`. Where do they differ, and why
    > might a single exponent not describe every part of the atmosphere?
    """),
]

# ---------------------------------------------------------------------------
# Notebook B, Part 2: seeing -> weak lensing
# ---------------------------------------------------------------------------
B_PART2_CELLS = [
    _c("md", r"""
    ## Part 2 · Seeing → weak lensing

    Weak lensing measures a tiny, coherent stretch in the shapes of distant galaxies. The telescope
    and the atmosphere blur every galaxy with the point-spread function (PSF). A galaxy much smaller
    than the PSF comes out looking like the PSF, and its shape tells you almost nothing. So the
    seeing decides how many galaxies you can use.

    Here are the image-quality numbers you'll compare. FWHM is the full width at half maximum of the PSF.

    | PSF FWHM | what it is | source |
    |---|---|---|
    | **0.7″** | the design seeing: the reference value in the depth formula, and on Monday's worksheet | Ivezić et al. 2019, eq. 6 |
    | **0.73″** | median seeing over the same first ~15 days | Phil Marshall, Sep 23 overview |
    | **0.91″** | median delivered image quality, first night | Phil Marshall, Sep 23 overview |
    | **1.1″** | median delivered image quality, first ~15 days | Phil Marshall, Sep 23 overview |

    Marshall also reported a median PSF ellipticity of 0.078 on the first night, against a
    requirement of 0.04. He put the telescope's and camera's own contribution at 0.45″ in
    total: dome seeing 0.3–0.4″ and tracking ~0.1″. These are early commissioning numbers.
    """),
    _c("md", r"""
    ### 1. A toy model for usable galaxies

    You need three ingredients.

    **How many galaxies.** The counts from Notebook A, $N(<i) = 46\times10^{0.31(i-25)}$ arcmin⁻²
    (*LSST Science Book 2009, eq. 3.7, a fit to CFHTLS Deep counts over 20.5 < i < 25.5*). Down to
    the Science Book's "gold sample" limit, $i < 25.3$ (S/N > 20 for point sources in median
    conditions), that is ≈ 55 arcmin⁻² (*Science Book §3.7.2*).

    **How big they are.** Miller et al. 2013 (the CFHTLenS shape paper, Appendix B1, eq. B1)
    fit the median disk scale length $r_d$ of galaxies measured with Hubble (Simard et al. 2002)
    against magnitude, over $18.5 < i_{814} < 25.5$:

    $$\ln\!\left(\frac{r_d}{\text{arcsec}}\right) = -1.145 - 0.269\,(i - 23)$$

    Around that median, the sizes scatter as $p(r) \propto r\,\exp[-(r/a)^{4/3}]$ (same appendix).
    Fainter galaxies are smaller: the median $r_d$ is ≈ 0.32″ at $i = 23$ and ≈ 0.17″ at $i = 25.3$.

    **Which ones are resolved.** Treat the galaxy and the PSF as Gaussians of width
    $\sigma_\text{gal}$ and $\sigma_\text{PSF}$. The *resolution factor* is

    $$R_2 = \frac{\sigma_\text{gal}^2}{\sigma_\text{gal}^2 + \sigma_\text{PSF}^2}$$

    It is 1 for a galaxy much larger than the PSF and 0 for one much smaller. A galaxy is
    *usable* if $R_2 > 1/3$. *Source: the cut used for the SDSS lensing shape catalog,
    Mandelbaum et al. 2005, §2, eq. 10; the Gaussian form of $R_2$ is from the caption of their
    Fig. 5.*

    **Our choices (not from any paper).** An exponential disk's half-light radius is
    $1.678\,r_d$. We replace each galaxy by a Gaussian with the same half-light radius
    ($\sigma_\text{gal} = 1.678\,r_d/1.1774$). We use the PSF FWHM in the table above as a Gaussian
    FWHM ($\sigma_\text{PSF} = \text{FWHM}/2.355$). We ignore bulges, ellipticity, blending,
    and the S/N of the shape measurement.

    **Where does this break?** The size relation comes from disk-dominated galaxies in one small
    Hubble field, measured along the major axis. Mandelbaum et al. warn that the Gaussian form of
    $R_2$ fails for bulge-dominated (de Vaucouleurs) profiles. Which way would each of these push
    the number of usable galaxies?
    """),
    _c("code", r'''
    FWHM_DESIGN = 0.7        # arcsec: design seeing (Ivezic et al. 2019, eq. 6; worksheet)
    SEEING_MARKERS = {       # PSF FWHM in arcsec
        "design": 0.7,               # Ivezic et al. 2019, eq. 6
        "median seeing": 0.73,       # Marshall, Sep 23: median atmospheric seeing, first ~15 days
        "first night": 0.91,         # Marshall, Sep 23: median delivered image quality, first night
        "first ~15 days": 1.1,       # Marshall, Sep 23: median delivered image quality, first ~15 days
    }
    I_GOLD = 25.3            # Science Book 3.7.2 "gold sample" limit, i < 25.3
    I_BRIGHT = 18.5          # bright end of the Miller et al. size fit; brighter galaxies are few (< 0.5 arcmin^-2)
    N_EFF_CHANG = 37.0       # Chang et al. 2013: LSST n_eff (r+i, 10 yr), before blending and masking

    from math import erfc
    _erfc = np.vectorize(erfc)

    def counts_per_mag(i):
        """dN/di in galaxies per arcmin^2 per mag: derivative of Science Book eq. 3.7."""
        return np.log(10) * 0.31 * 46 * 10 ** (0.31 * (i - 25))

    def median_scalelength(i):
        """Median disk scale length r_d (arcsec) at i-band magnitude i: Miller et al. 2013, eq. B1."""
        return np.exp(-1.145 - 0.269 * (i - 23))

    def frac_resolved(i, fwhm, r_min=1/3):
        """Fraction of galaxies at magnitude i with resolution factor R2 > r_min, for a PSF of this FWHM.

        Galaxies: Miller et al. 2013 sizes, each replaced by a Gaussian with the same half-light
        radius. PSF: a Gaussian with this FWHM (arcsec).
        """
        sigma_psf = fwhm / 2.3548
        sigma_gal_min = sigma_psf * np.sqrt(r_min / (1 - r_min))   # R2 > r_min  <=>  sigma_gal > this
        rd_min = sigma_gal_min * 1.1774 / 1.678                   # the scale length with that sigma
        # Miller et al.: p(r) ~ r exp[-(r/a)^(4/3)], with a set so the median is eq. B1.
        # For this distribution the median is 1.134 a.
        a = median_scalelength(i) / 1.134
        x = (rd_min / a) ** (4 / 3)
        # P(r > rd_min): with x = (r/a)^(4/3) this is a gamma distribution of shape 3/2,
        # whose upper tail has the closed form below.
        return _erfc(np.sqrt(x)) + 2 * np.sqrt(x / np.pi) * np.exp(-x)

    for i in [21, 23, 24, 25, 25.3]:
        print(f"i = {i:4}: median r_d = {median_scalelength(i):.2f}\"   "
              f"resolved at 0.7\": {frac_resolved(i, 0.7):.2f}   at 1.1\": {frac_resolved(i, 1.1):.2f}")
    '''),
    _c("md", r"""
    ### 2. Worse seeing also costs depth

    From Part 1: a wider PSF spreads a point source over more sky pixels, so $m_5$ gets brighter.
    The next cell uses your `m5()` to shift the magnitude limit. At the design seeing it is
    $i < 25.3$, and it moves by exactly as much as the single-visit $m_5$ in i moves. That
    pretends *every* visit had this seeing.

    One conversion is needed. `m5()` takes FWHM$_\text{eff}$ (the width of the equivalent
    single Gaussian, Part 1), while the numbers in the table are measured FWHMs. rubin_sim
    converts them with FWHM = 0.822 FWHM$_\text{eff}$ + 0.052″ (*rubin_sim 2.6.2,
    `phot_utils/signaltonoise.py`, `fwhm_geom2_fwhm_eff`*).

    **Where does this break?** The $m_5$ loss is for point sources. A resolved galaxy is already
    spread over its own area. Does seeing cost you more or less depth for a large galaxy than for a
    star? And a real survey mixes good and bad nights in one coadd.
    """),
    _c("code", r'''
    def fwhm_to_eff(fwhm):
        """Measured PSF FWHM -> FWHM_eff for m5() (rubin_sim fwhm_geom2_fwhm_eff), arcsec."""
        return (fwhm - 0.052) / 0.822

    def i_limit(fwhm):
        """Toy magnitude limit: i < 25.3 at the design seeing, shifted by the change in i-band m5."""
        return I_GOLD + m5("i", fwhm_eff=fwhm_to_eff(fwhm)) - m5("i", fwhm_eff=fwhm_to_eff(FWHM_DESIGN))

    def usable_density(fwhm, r_min=1/3, depth=True):
        """Toy: galaxies per arcmin^2 with R2 > r_min, down to the magnitude limit.

        depth=True moves the limit with the seeing (i_limit); depth=False keeps i < 25.3.
        This is a count of resolved galaxies, not Chang et al.'s weighted n_eff.
        """
        i_max = i_limit(fwhm) if depth else I_GOLD
        edges = np.linspace(I_BRIGHT, i_max, 401)
        mid = 0.5 * (edges[1:] + edges[:-1])
        return float(np.sum(counts_per_mag(mid) * frac_resolved(mid, fwhm, r_min) * np.diff(edges)))

    n_design = usable_density(FWHM_DESIGN)
    print(f"{'':16} {'FWHM':>6} {'i limit':>8} {'resolved only':>14} {'+ depth loss':>13} {'vs design':>10}")
    for name, f in SEEING_MARKERS.items():
        n_d = usable_density(f)
        print(f"{name:16} {f:5.2f}\" {i_limit(f):8.2f} {usable_density(f, depth=False):14.1f} "
              f"{n_d:13.1f} {n_d / n_design:10.2f}")
    print("(galaxies per arcmin^2; toy model)")
    print(f"Chang et al. 2013 n_eff: {N_EFF_CHANG:.0f} arcmin^-2. Science Book 3.7.2: ~40 arcmin^-2 (+/- 20%).")
    '''),
    _c("md", r"""
    ### 3. Usable galaxies vs. seeing

    The left panel shows the usable density against PSF FWHM, with and without the depth loss.
    The markers are the four seeing values from the table. The gray line is the published LSST
    estimate: Chang et al. 2013 (MNRAS 434, 2121) found $n_\text{eff} \approx 37$ arcmin⁻² for
    r+i over ten years before blending and masking, 31 after rejecting serious blends, and 26
    after a further 15% loss to masking. Theirs is a full simulation with a weighted $n_\text{eff}$;
    ours is a toy count, so compare shapes and ratios, not the last digit. The right panel shows
    which magnitudes you lose. Left of the design seeing, the depth-shifted limit passes
    $i = 25.5$, the faint end of the eq. 3.7 fit, so that part of the solid curve is an
    extrapolation.

    If the slider doesn't appear, call `show_lensing(fwhm, r_min=1/3, depth=True)` directly.
    """),
    _c("code", r'''
    def show_lensing(fwhm=0.73, r_min=1/3, depth=True):
        """Plot usable density vs PSF FWHM, and what is lost by magnitude at this FWHM. Returns the density."""
        grid = np.linspace(0.4, 1.6, 61)
        n_here = usable_density(fwhm, r_min, depth)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        ax1.plot(grid, [usable_density(f, r_min, True) for f in grid], "k", label="resolved + depth loss")
        ax1.plot(grid, [usable_density(f, r_min, False) for f in grid], "k--", lw=1,
                 label="resolved only (i < 25.3)")
        ax1.axhline(N_EFF_CHANG, color="0.6", lw=1, label="Chang et al. 2013 $n_{eff}$ = 37")
        for (name, f), color in zip(SEEING_MARKERS.items(), ["tab:blue", "tab:green", "tab:orange", "tab:red"]):
            ax1.axvline(f, color=color, ls=":", lw=1)
            ax1.plot(f, usable_density(f, r_min, depth), "o", color=color, label=f"{name} ({f}\")")
        ax1.plot(fwhm, n_here, "k*", ms=12, label=f"you: {fwhm:.2f}\" -> {n_here:.1f}")
        ax1.set_xlabel("PSF FWHM (arcsec)")
        ax1.set_ylabel("usable galaxies per arcmin$^2$ (toy)")
        ax1.set_ylim(0, None)
        ax1.legend(fontsize=7)

        i_max = i_limit(fwhm) if depth else I_GOLD
        mags = np.linspace(I_BRIGHT, 26.0, 200)
        ax2.plot(mags, counts_per_mag(mags), "0.5", label="all galaxies (eq. 3.7)")
        ok = mags <= i_max
        ax2.fill_between(mags[ok], 0, (counts_per_mag(mags) * frac_resolved(mags, fwhm, r_min))[ok],
                         color="tab:blue", alpha=0.5, label=f"usable at {fwhm:.2f}\"")
        ax2.axvline(i_max, color="k", ls="--", lw=1, label=f"i limit {i_max:.2f}")
        ax2.set_xlabel("i magnitude")
        ax2.set_ylabel("galaxies per arcmin$^2$ per mag")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        return n_here

    show_lensing(1.1)
    '''),
    _c("code", r'''
    from ipywidgets import Checkbox

    def _lensing_widget(fwhm, r_min, depth):
        show_lensing(fwhm, r_min, depth)          # don't echo the returned density

    sliders(_lensing_widget,
            fwhm=FloatSlider(value=0.73, min=0.4, max=1.6, step=0.01,
                             description="PSF FWHM (″)", **_W),
            r_min=FloatSlider(value=1/3, min=0.1, max=0.6, step=0.01,
                              description="resolution cut R₂ >", **_W),
            depth=Checkbox(value=True, description="include depth loss"));
    '''),
    _c("md", r"""
    **What does 1.1″ cost?** Compare the usable density at 1.1″ with 0.73″. How much of the loss
    comes from resolution and how much from depth? How does the answer change if you move the
    resolution cut?

    **How rough is "figure of merit ∝ $n_\text{eff}$"?** Very rough. The error on the lensing power
    spectrum (Chang et al. 2013, eq. 11) is proportional to $P(\ell) + \sigma_\text{SN}^2/n_\text{eff}$ (plus
    systematics), where $P$ is the signal and $\sigma_\text{SN} \approx 0.26$ is the scatter in
    intrinsic galaxy shapes. $n_\text{eff}$ enters only through the second term. So treat
    "FoM ∝ $n_\text{eff}$" as a way to rank options, not a number to quote. On which scales does
    losing galaxies cost you almost nothing, and where does it cost more than proportionally?

    **Where does the extra come from?** Marshall's breakdown: 0.45″ in total from the telescope
    and camera (dome seeing 0.3–0.4″, tracking ~0.1″), against a median seeing of 0.73″.
    Do blurs from independent sources add linearly or in quadrature? Does 0.45″ account for the
    gap between 0.73″ and 1.1″? Which of those contributions can be fixed, and which can't?

    **Why does PSF ellipticity matter?** The first-night median PSF ellipticity was 0.078, against
    a requirement of 0.04. The lensing shear you're after is a few percent at most (Chang et al.
    2013 note it is over an order of magnitude smaller than the 0.26 shape scatter). A galaxy at the
    cut, $R_2 = 1/3$, is two-thirds PSF by second moment. If your model of the PSF's ellipticity is
    off by a small fraction, what happens to the shear you infer? Why would a small PSF ellipticity
    be easier to correct than a large one?

    > **Go deeper.** Miller et al. write the size distribution's scale as $a = r_d/0.833$. For
    > $p(r) \propto r\,e^{-(r/a)^{4/3}}$, the median is $1.134\,a$, so $a = r_d/0.833$ would put the
    > median at $1.36\,r_d$, not at $r_d$. The paper also says it chose $a$ to match the median of
    > eq. B1, and we follow that. Check the median numerically. Then redo the design value with
    > their $a$ (larger galaxies). How much does the answer move, and how does that compare with
    > the gap between our toy and Chang et al.?
    """),
]

B_PART3_CELLS = []   # Part 3: the focal plane (Task 4)
B_PART4_CELLS = []   # Part 4: go deeper and the hand-in (Task 4)


def _make_notebook(cells_src):
    cells = []
    for index, (kind, text) in enumerate(cells_src):
        if kind == "md":
            cell = nbformat.v4.new_markdown_cell(text)
        else:
            cell = nbformat.v4.new_code_cell(text)
        cell.id = f"cell-{index:03d}"            # deterministic ids: rebuilds don't churn
        cells.append(cell)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def build_notebook_a():
    return _make_notebook(PART1_CELLS + PART2_CELLS + PART3_CELLS + PART4_CELLS)


def build_notebook_b():
    return _make_notebook(B_INTRO_CELLS + B_PART1_CELLS + B_PART2_CELLS + B_PART3_CELLS
                          + B_PART4_CELLS)


def build(path=NB_A):
    """Write Notebook A (outputs cleared) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook_a(), path)
    return path


def build_b(path=NB_B):
    """Write Notebook B (outputs cleared) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook_b(), path)
    return path


if __name__ == "__main__":
    print(build())
    print(build_b())
