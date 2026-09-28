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

    *Source: LSST Science Book, eq. 3.7.* Multiply by the area (1 deg² = 3600 arcmin²).

    **Where does this break?** The fit was calibrated for $i \\lesssim 25$–$27$. It also ignores
    blending: at LSST depth, a large fraction of galaxies overlap a neighbour. Does a deeper survey
    always give you more *usable* shapes?

    **Type Ia supernovae.** Peak absolute magnitude $M = -19.3$ with no K-correction. A supernova
    counts as "detected near peak" if it is 1 mag brighter than a single i-band visit's limit:
    solve $M + \\mathrm{DM}(z_\\text{max}) = m_5^\\text{single}(i) - 1$ with the Planck 2018
    cosmology. Then

    $$N_\\text{SN} = \\int_0^{z_\\text{max}} \\frac{R\\,(1+z)^{1.5}}{1+z}\\,\\frac{dV}{dz}\\,dz
    \\times \\frac{\\text{area}}{41{,}253} \\times 10\\,\\text{yr} \\times 0.5$$

    with $R = 2.6\\times10^{-5}\\ \\text{Mpc}^{-3}\\,\\text{yr}^{-1}$ (local SN Ia rate), the
    $(1+z)^{1.5}$ rate evolution, $1/(1+z)$ for time dilation, and 0.5 because a field is only
    observable about half the year.

    **Where does this break?** This is an upper bound: no K-corrections, and no requirement that you
    actually get a well-sampled light curve. What would you need to add to count supernovae that are
    useful for cosmology?

    **Moving objects.** Linking an asteroid orbit needs roughly 3 same-night pairs within ~15 days.
    Pairs per 15-day window per field:

    $$\\frac{N_\\text{visits}/2}{10 \\times 365.25 \\times 0.5 / 15}$$

    *Source: a counting proxy; assumes every visit is half of a same-night pair and a 6-month
    observing season.*

    **Where does this break?** Visits are not spread evenly: weather, the Moon, and the season
    clump them. Is the average the right statistic for "did we link this asteroid"?
    """),
    _c("code", """
    SN_M_PEAK = -19.3        # SN Ia peak absolute mag (no K-correction)
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
        \"\"\"Upper bound on SNe Ia seen near peak (no K-corrections, no light-curve cuts).\"\"\"
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
        print("SNe Ia detected near peak (upper bound; no K-corrections, no light-curve quality "
              f"cuts): {p['n_sn_detected']:.2e}")
        print(f"Same-night pairs per field per 15 d:  {p['pairs_per_15d']:.1f}  "
              "(rule of thumb: linking an orbit needs ~3 pairs within ~15 days)")
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

    **Asteroids.** Does the LSST default meet the ~3 pairs in ~15 days rule? What about your survey?
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

PART2_CELLS = []
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
