"""Build the Week 2 debrief notebook from a readable cell list.

Run ``python tools/build_week2_debrief.py`` to (re)write ``week2/debrief.ipynb`` with outputs
cleared. The debrief reruns the most instructive analyses students added to Notebooks A and B
and walks through the places where the reasoning slipped, anonymously. It reuses the functions
and data of the two notebooks by executing their non-widget code cells (fetched from GitHub,
or from the local week2/ folder when present). The notebook must not import from tools/.
"""
from pathlib import Path
import textwrap

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "week2" / "debrief.ipynb"


def _c(kind, text):
    return (kind, textwrap.dedent(text).strip("\n"))


INTRO_CELLS = [
    _c("md", """
    # Week 2 · Debrief: what the class found

    Every notebook students turned in ran through to the hand-in with no errors, and about
    half of you went past the template and built something of your own.
    This notebook reruns the most instructive of those additions, with the code, so everyone
    can see them, and then goes through the places where the reasoning slipped. Nothing here is
    attributed; where it says "one of you", that is all it says.

    **How to use it.** *Runtime → Run all*. The first cell loads the functions and data of
    Notebooks A and B (about 20 s). After that every section stands alone.

    1. **What you found**: eight analyses, regenerated.
    2. **Where the reasoning slipped**: the misconceptions, each with the two-line computation that settles it.
    3. **The numbers** the class converged on.
    4. **What the class got right.**
    """),
    _c("code", """
    # Load the functions and data of Notebooks A and B without their sliders.
    import io, json, os, re, contextlib, urllib.request
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    RAW = "https://raw.githubusercontent.com/rhw/phys367/main/week2/"


    def load_notebook_functions(name):
        \"\"\"Run the code cells of a Week 2 notebook here, skipping the slider cells.\"\"\"
        local = os.path.join(os.path.dirname(os.path.abspath("__file__")), name)
        src = open(local).read() if os.path.exists(local) else urllib.request.urlopen(RAW + name).read().decode()
        for cell in json.loads(src)["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            if re.search(r"^\\s*sliders\\(|^def _\\w+_widget\\(", code, re.M) or code.strip().startswith("# Your hand-in"):
                continue
            code = "\\n".join(line for line in code.splitlines() if not re.match(r"\\s*[%!]", line))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, name, "exec"), globals())
            plt.close("all")


    _show = plt.show
    plt.show = lambda *a, **k: plt.close("all")        # the templates' demo plots stay hidden
    load_notebook_functions("A_survey_strategy.ipynb")
    load_a = load                                       # both notebooks define load(); keep A's for maps/points
    load_notebook_functions("B_camera_to_science.ipynb")
    load = load_a
    plt.show = _show
    print("Notebook A:", len(points), "visits in `points`,", len(maps), "map rows;  Notebook B: m5('r') =", round(m5("r"), 3))
    """),
]

PART1_CELLS = [
    _c("md", """
    ## Part 1 · What you found

    ### 1. Where did the 0.6 mag go?

    The design single-visit r-band depth is 24.7; the simulated median is 24.06. Several of
    you asked which of seeing, airmass and sky explains the drop. Two approaches appeared: a
    scatter of every main-survey visit's $m_5$ against each quantity, and a stepwise call of
    `m5()` changing one input at a time. Here are both.

    **Why it matters.** Every requirement in Week 1 was set against the design depth. If the
    real survey is 0.6 mag shallower, knowing *which* input causes it tells you whether the
    loss is permanent (the site's seeing), seasonal (sky brightness), or something the
    scheduler can trade against (airmass).
    """),
    _c("code", """
    r_visits = points[(points["band"] == "r") & (points["category"] == "main")]
    labels = {"seeing": "seeing FWHM (arcsec)", "airmass": "airmass", "sky": "sky brightness (mag/arcsec²)"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col in zip(axes, labels):
        ax.scatter(r_visits[col], r_visits["m5"], s=4, alpha=0.15)
        slope, intercept = np.polyfit(r_visits[col], r_visits["m5"], 1)
        x = np.linspace(r_visits[col].min(), r_visits[col].max(), 50)
        ax.plot(x, slope * x + intercept, color="red")
        ax.set_xlabel(labels[col]); ax.set_ylabel("$m_5$ (r)")
        ax.set_title(f"correlation {r_visits[col].corr(r_visits['m5']):+.2f}")
    plt.tight_layout(); plt.show()

    med = r_visits[["m5", "seeing", "airmass", "sky"]].median()
    print("main-survey r medians:", med.round(2).to_dict())

    steps = {
        "as built, fiducial":          m5("r"),
        "+ seeing 1.03\\"":               m5("r", fwhm_eff=1.03),
        "+ airmass 1.18":               m5("r", fwhm_eff=1.03, airmass=1.18),
        "+ sky 21.0":                   m5("r", fwhm_eff=1.03, airmass=1.18, sky_mag=21.0),
    }
    table = pd.Series(steps).to_frame("m5")
    table["step"] = table["m5"].diff().fillna(0.0)
    print(table.round(3))
    """),
    _c("md", """
    Seeing takes 0.23 mag, the sky 0.09, airmass 0.02; the stepwise total, 24.14, lands
    within 0.08 of the simulation's 24.06. Four of you concluded "seeing dominates", three
    with evidence like this.

    **A cautionary tale from the same question.** One notebook reported that airmass alone
    explained 0.48 of the 0.6 mag. The code had `airmass=1.8` where the simulation median is
    1.18. The three terms then summed to 24.06 almost exactly, which made the typo look
    right. When one term comes out much larger than you expected, check the inputs before
    building an explanation on it; the coincidence of the total was the misleading part.
    (A subtlety: with `fwhm_eff=None`, `m5()` also widens the PSF by airmass$^{0.6}$, so to
    separate the terms hold the delivered seeing fixed, as the stepwise table does.)
    """),
    _c("code", """
    for x in (1.18, 1.8):
        print(f"airmass {x} at fixed 1.03 arcsec seeing: m5 changes by {m5('r', fwhm_eff=1.03, airmass=x) - m5('r', fwhm_eff=1.03):+.3f} mag")
    print(f"airmass 1.18 letting the model also widen the PSF (X^0.6): {m5('r', airmass=1.18) - m5('r'):+.3f} mag")
    print(f"airmass 1.8 with the PSF widened too (the hand-in's call):  {m5('r', airmass=1.8) - m5('r'):+.3f} mag")
    """),
    _c("md", """
    ### 2. Seeing, priced in time

    Instead of asking how much depth a worse PSF costs at fixed exposure, one of you asked
    how much *exposure* it costs at fixed depth. The answer is the more useful one for a survey
    with a fixed number of hours.
    """),
    _c("code", """
    times = np.geomspace(5, 300, 100)
    target = 24.5
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for seeing in (0.7, 1.0, 1.3):
        depths = np.array([m5("r", t_exp=t / 2, n_exp=2, fwhm_eff=seeing) for t in times])
        t_needed = np.interp(target, depths, times)
        line, = ax.plot(times, depths, label=f"FWHM_eff {seeing:.1f}\\": {t_needed:.0f} s to reach {target}")
        ax.plot([t_needed], [target], "o", color=line.get_color())
    ax.axhline(target, color="gray", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("open-shutter time per visit (s)"); ax.set_ylabel("r-band $m_5$")
    ax.legend(); plt.show()
    """),
    _c("md", """
    From 0.7″ to 1.3″ the time to reach the same depth triples.

    **Takeaway.** Seeing is usually discussed as an image-quality number, but for a survey
    with a fixed number of hours it is a budget number: every 0.1″ of extra PSF width is
    time you do not get back. That is why the 0.9″ to 1.1″ of the first commissioning weeks
    matters beyond the pictures.

    ### 3. The 9 s of overhead per visit

    The worksheet budget counts a 5 s visit as one sixth of a 30 s visit. Notebook A flagged
    the readout, shutter and slew that every visit also costs; two of you put the 9 s into
    the budget, one in code and one by hand.
    """),
    _c("code", """
    area, t_visit, overhead = 18_000, 5, 9
    print(f"5 s visits, no overhead:  {BUDGET / (area * t_visit / 30):.0f} visits per field")
    print(f"5 s visits, 9 s overhead: {BUDGET / (area * (t_visit + overhead) / 30):.0f} visits per field")
    print(f"30 s visits, 9 s overhead:{BUDGET / (area * (30 + overhead) / 30):.0f} visits per field (vs 833 without)")
    """),
    _c("md", """
    **Takeaway.** Short visits look free in a budget that only counts open-shutter time, and
    they are not: at 5 s the overhead cuts the visit count by a factor of 2.8. This is the
    reason Rubin's visits are 30 s and not 5, and why the fast-cadence science cases had to
    argue for time rather than assume it.

    ### 4. What the storm costs

    Phil Marshall quoted 7,267 visits taken out of 17,273 possible in the first two weeks.
    Two of you reran the plan at 60% of the budget. One read the bar chart and wrote that u
    and y were hit hardest. The other printed the table:
    """),
    _c("code", """
    full = plan_survey(18_000, 30, DEFAULT_SPLIT)
    reduced = plan_survey(18_000, 30, DEFAULT_SPLIT, budget=0.6 * BUDGET)
    print(pd.DataFrame({
        "full visits": full["n_by_band"], "60% visits": reduced["n_by_band"],
        "full coadd": full["m5_coadd"], "60% coadd": reduced["m5_coadd"],
    }).round(2))
    print(f"\\ncoadd loss in every band: {1.25 * np.log10(0.6):+.2f} mag")
    print(f"lensing galaxies: {full['n_gal']:.2e} -> {reduced['n_gal']:.2e};  pairs per 15 d: {full['pairs_per_15d']:.2f} -> {reduced['pairs_per_15d']:.2f}")
    print(f"SNe Ia detected: {full['n_sn_detected']:.2e} -> {reduced['n_sn_detected']:.2e}  (unchanged: the toy depends on single-visit depth and area only)")
    """),
    _c("md", """
    With identical visits the loss is the same 0.28 mag in every band; the chart's y-range
    made it look band-dependent. The second notebook also noted that a contiguous gap is
    worse than a uniform 40% cut, because it removes whole seasons from some fields.

    **Takeaway.** Lost time costs every band the same depth, so "which band suffers" is the
    wrong question; "which science needs the visits that were lost" is the right one. The
    lensing count drops 18% and the pair rate for asteroids 40%, while the supernova count
    in this toy does not move, because it depends on single-visit depth, not on how many
    visits there are.

    ### 5. How much sky actually reaches r = 27?

    The median ten-year r-band coadd is 26.94, so "r ≥ 27 over 18,000 deg²" is not what the
    simulation delivers. One of you measured how much sky does reach it.
    """),
    _c("code", """
    r_maps = maps[maps["band"] == "r"]
    for label, col in (("Year 1", "m5_coadd_y1"), ("10 years", "m5_coadd")):
        print(f"{label:9s} area with r coadd >= 27.0: {(r_maps[col] >= 27.0).sum() * PIXEL_AREA:8,.0f} deg²")
    print(f"{'':9s} main-survey median 10-yr r coadd: {main_survey_medians(maps).loc['r', 'm5_coadd']:.2f}")
    """),
    _c("md", """
    **Takeaway.** A survey's depth is a distribution, not a number, and a requirement written
    as a threshold ("r ≥ 27") is met over a very different area than the median suggests.
    When your science case needs a depth, ask how many square degrees reach it, and when.

    ### 6. Kilonovae: one table for three questions

    "Which preset catches the kilonova within 2 days? At 400 Mpc? Within 5 days?" One
    notebook answered all three at once.
    """),
    _c("code", """
    catch = pd.concat({
        "200 Mpc, 2 d": catch_table(kilonova, 2),
        "400 Mpc, 2 d": catch_table(kilonova, 2, d_mpc=400),
        "200 Mpc, 5 d": catch_table(kilonova, 5),
    }, axis=1)
    catch.round(3)
    """),
    _c("md", """
    The deep drilling fields catch about half, the main survey about a sixth, the south polar
    cap almost none. Four of you drew the right conclusion: the deep fields win on *cadence*
    (a median night gap of 1 day against 3), not depth, and since they cover a tiny fraction
    of the sky, gravitational-wave alerts and target-of-opportunity time are what makes
    kilonova science work. One notebook wrote that the deep fields rank highest "because the
    observations are extremely deep"; see Part 2.

    **Takeaway.** For anything that fades in days, the question is not "how deep" but "how
    often", and the answer is set by where the scheduler points, not by the telescope. That
    is why the fast-transient science cases depend on alerts and target-of-opportunity time
    as much as on Rubin itself.

    ### 7. Time delays: season gaps or noise?

    One of you ran the lensed-quasar estimator once with and once without noise, got 28 and
    30 days, and concluded the season gaps were the limit. Another ran six noise seeds and a
    gap-free regular cadence:
    """),
    _c("code", """
    print("RXJ1131, no noise:     ", estimate_delay(lensed_quasar("RXJ1131-1231", noise=False)))
    print("RXJ1131, noise, seeds 1-6:", [round(estimate_delay(lensed_quasar("RXJ1131-1231", noise=True, seed=s))) for s in range(1, 7)])
    regular = SURVEY_START + np.arange(0, 3650, 3)
    print("every 3 d, no gaps, no noise:", estimate_delay(lensed_quasar(mjd=regular, noise=False)))
    print("every 3 d, no gaps, noise:   ", estimate_delay(lensed_quasar(mjd=regular, noise=True)))
    """),
    _c("md", """
    The gapped, noise-free case recovers the true 30 days exactly; it is the noise that moves
    the estimate. One run with each setting cannot tell you that.

    **Takeaway.** Whenever a result depends on a random realization, one run is an anecdote.
    Changing the seed a few times costs seconds and is the difference between naming the
    right limiting factor and the wrong one. The same habit applies to real data: a
    conclusion from one patch of sky needs a second patch.

    ### 8. When does the camera saturate?

    Everything in Notebook B was about the faint end. One of you turned it around and asked
    at what magnitude the central pixel of a star fills the well, using the notebook's
    zeropoints, dark-sky brightness and PSF widths. The code below is theirs, lightly tidied.
    """),
    _c("code", """
    band = "r"
    full_well = 1e5                                   # electrons, approximate
    sigma = FWHM_EFF_ZENITH[band] / 2.355             # Gaussian PSF at the fiducial zenith FWHM_eff
    xx, yy = np.meshgrid(np.linspace(-2, 2, 401), np.linspace(-2, 2, 401), indexing="ij")
    psf = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    frac_in_center_pixel = psf[(np.abs(xx) < 0.1) & (np.abs(yy) < 0.1)].sum() / psf.sum()

    t = np.logspace(1, 4, 100)
    sky_per_pixel_per_s = sky_counts(band) * PIXEL_SCALE**2      # sky_counts is e-/s/arcsec^2
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cmap, norm = plt.get_cmap("viridis"), plt.Normalize(16, 26)
    for mag in np.arange(16, 26.5, 0.5):
        star_rate = frac_in_center_pixel * zeropoint(band) * 10 ** (-0.4 * mag)   # zeropoint = e-/s for m = 0
        total = np.minimum((star_rate + sky_per_pixel_per_s) * t, full_well)
        ax.loglog(t, total, color=cmap(norm(mag)))
    ax.axhline(full_well, color="gray", ls="--", label="full well (approx.)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=f"{band}-band magnitude")
    ax.set_xlabel("exposure time (s)"); ax.set_ylabel("electrons in the central pixel"); ax.legend()
    ax.set_title("When does LSSTCam saturate?"); plt.show()
    print(f"fraction of a star's light in the central 0.2\\" pixel at this PSF: {frac_in_center_pixel:.3f}")
    """),
    _c("md", """
    Stars brighter than about r = 16 saturate in a 30 s visit, close to what Rubin quotes.

    **Takeaway.** A survey has a bright limit as well as a faint one. Saturated stars are not
    lost, but their fluxes come from a different measurement path with its own systematics,
    and they are exactly the objects Week 3's star/galaxy classifier fails on. Calibration
    against bright reference stars has to live inside this limit.

    **Also seen in the hand-ins:** a six-panel map of visits per band showing
    the Galactic plane covered in griz but barely in u and y; a table of the photometric error
    each toy transient would have at peak, with the result that the 5 mmag calibration floor
    matters only for sources brighter than $m_5 - 4.1$; and a three-way sensitivity study
    (18,000 vs 10,000 deg², 30 vs 60 s) showing that doubling the visit length at fixed area
    leaves the coadd depth and lensing count unchanged while the supernova count rises.
    """),
]

PART2_CELLS = [
    _c("md", """
    ## Part 2 · Where the reasoning slipped

    Each item below appeared in at least one hand-in. The computation under it is the one to
    remember.

    ### Aperture "doesn't help much"

    One notebook moved the aperture slider, saw a modest change, and concluded that exposure
    time is the better lever. In the sky-limited regime $m_5$ gains $2.5\\log_{10}$ of the
    aperture ratio, and aperture buys depth without spending visits:
    """),
    _c("code", """
    base = m5("r")
    for D in (4.0, 6.423, 12.85):
        print(f"aperture {D:5.2f} m: m5 = {m5('r', area_m2=np.pi * (D / 2)**2):.2f}  ({m5('r', area_m2=np.pi * (D / 2)**2) - base:+.2f})")
    for t in (15, 30, 60):
        print(f"2 x {t:2d} s:          m5 = {m5('r', t_exp=t):.2f}  ({m5('r', t_exp=t) - base:+.2f})")
    """),
    _c("md", """
    Doubling the aperture gains 0.82 mag; doubling the exposure gains 0.42. Doubling the
    aperture is worth quadrupling the exposure, and it costs no visits.

    ### Which depth is this?

    Three numbers for the single-visit r-band depth went around: 24.7 (design, from the
    worksheet), 24.48 (the as-built camera at fiducial conditions, Notebook B), 24.06 (the
    simulated ten-year median visit). All three are *single visits*. Several hand-ins mixed
    them with coadds: one compared a 1000 × 30 s `m5()` call against the 2 × 15 s reference
    and concluded the system falls short of its Milky Way requirements, when those numbers
    exceed the requirements by 3 mag; another labeled a 20 × 30 s stack "as-built single-visit
    depth". Keep the two columns apart:
    """),
    _c("code", """
    med = main_survey_medians(maps)
    rows = {}
    for b in BANDS:
        rows[b] = {
            "design visit (worksheet)": M5_30S[b],
            "as-built visit, 2x15 s": m5(b),
            "simulated median visit": points[(points["band"] == b) & (points["category"] == "main")]["m5"].median(),
            "simulated 10-yr coadd": med.loc[b, "m5_coadd"],
            "1000 x 30 s in m5()": m5(b, t_exp=30, n_exp=1000),
        }
    pd.DataFrame(rows).T.round(2)
    """),
    _c("md", """
    The last column is not a visit, and not the survey either: `m5()` stacks identical
    exposures with no change of seeing or sky, while the real coadd is 700-odd different
    visits. Say which depth you mean, every time.

    ### The budget is fixed

    "We changed to longer exposures per visit, which slightly increases the total exposure
    time." It cannot: the ten years are the budget. Longer visits mean fewer of them.
    Likewise "point-source depth increased when I raised the area to 30,000 deg²": the
    single-visit depth does not know the area, and the coadd gets *shallower* because each
    field gets fewer visits. What rises with area is the total galaxy count.
    """),
    _c("code", """
    plans = {
        "18,000 deg², 30 s": plan_survey(18_000, 30, DEFAULT_SPLIT),
        "18,000 deg², 120 s": plan_survey(18_000, 120, DEFAULT_SPLIT),
        "30,000 deg², 30 s": plan_survey(30_000, 30, DEFAULT_SPLIT),
        "10,000 deg², 30 s": plan_survey(10_000, 30, DEFAULT_SPLIT),
    }
    pd.DataFrame({k: {"visits per field": p["n_visits"], "r single visit": p["m5_single"]["r"], "r coadd": p["m5_coadd"]["r"],
                      "lensing galaxies": p["n_gal"], "revisit (d)": p["revisit_days"], "pairs / 15 d": p["pairs_per_15d"]}
                  for k, p in plans.items()}).T.round(2)
    """),
    _c("md", """
    ### Density is not the figure of merit

    One notebook plotted lensing galaxies per deg² against survey area, saw it fall, called it
    "weak-lensing resolution", and argued for a smaller survey. The mechanism in the notebook
    was right: less area means more visits per field, a deeper coadd, and more usable galaxies
    per square arcminute. But shear statistics scale with the *total* number of galaxies (and
    with area, against cosmic variance). Plot both:
    """),
    _c("code", """
    areas = np.linspace(1_000, 30_000, 60)
    n_gal = np.array([plan_survey(a, 30, DEFAULT_SPLIT)["n_gal"] for a in areas])
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(areas, n_gal / areas, color="tab:red"); ax1.set_ylabel("galaxies per deg²", color="tab:red")
    ax2 = ax1.twinx(); ax2.plot(areas, n_gal, color="tab:blue"); ax2.set_ylabel("total lensing galaxies", color="tab:blue")
    ax1.set_xlabel("survey area (deg²)"); plt.show()
    """),
    _c("md", """
    ### 1.1″ is not "lower" than 0.7″, and it is not what we expect

    Two slips about seeing. "The first 15 days PSF is 1.1″, lower than the designed 0.7″":
    a larger FWHM is worse image quality. And "the effective seeing we expect is 1.1″, which
    cuts usable galaxies from 40 to 30, acceptable": 1.1″ is the first two weeks of
    commissioning, not the ten-year expectation, and 40 → 30 is the resolution-only loss.
    With the depth loss included the toy model gives:
    """),
    _c("code", """
    for label, fwhm in (("design", 0.7), ("first night delivered", 0.91), ("first ~15 days delivered", 1.1)):
        print(f"{label:26s} {fwhm:.2f}\\"  usable galaxies: {usable_density(fwhm, depth=False):5.1f} (resolution only)  {usable_density(fwhm):5.1f} per arcmin² (with depth loss)")
    """),
    _c("md", """
    A 45% cut in usable galaxies is not "acceptable" for a survey whose lensing figure of
    merit scales with that number. One more from the same paragraph: "observing sharper
    images through coadd". Coadding does not sharpen; it averages the PSFs of the input
    visits, which is a Week 3 topic.

    ### The deep fields catch kilonovae because they look every night

    Not because they are deep. The toy kilonova at 200 Mpc peaks at 20.5 in every band,
    3 to 4 mag brighter than any single visit's limit, so depth is not the constraint:
    """),
    _c("code", """
    print("kilonova peak magnitude at 200 Mpc:", {b: round(float(kilonova(np.array([0.0]), b)[0]), 2) for b in BANDS})
    ct = cadence_table()
    cols = [c for c in ct.columns if "gap" in c.lower()][:2]
    print(ct.loc[["main-1", "COSMOS"], cols] if cols else ct.loc[["main-1", "COSMOS"]])
    """),
    _c("md", """
    ### The u band and read noise

    Three hand-ins touched the question of why one 30 s exposure beats two 15 s exposures in
    u. The right reason, given by one of you: the u sky is dark, so for a 15 s exposure the
    read noise squared is comparable to the sky counts, and splitting the visit pays the read
    noise twice. Not because u sources are faint, and not because the throughput is low as
    such. The exposure time at which sky noise overtakes read noise differs strongly by band:
    """),
    _c("code", """
    read_noise = 8.8
    for b in BANDS:
        rate_per_pixel = sky_counts(b) * PIXEL_SCALE**2       # e-/pixel/s (sky_counts is per arcsec^2)
        print(f"{b}: dark sky {rate_per_pixel:5.1f} e-/pix/s -> sky noise equals read noise after {read_noise**2 / rate_per_pixel:5.1f} s;"
              f"  1x30 s minus 2x15 s = {m5(b, t_exp=30, n_exp=1) - m5(b):+.2f} mag")
    """),
    _c("md", """
    ### A visit is not an exposure

    Three hand-ins called the 30 s visit "an exposure" when working out whether a moving
    object trails. Notebook B's reference visit was 2 × 15 s, so an asteroid at 0.04″/s
    moves 0.6″ per exposure and 1.2″ per visit. The project has since moved to single 30 s
    exposures, so the trailing question is now the per-visit one. Either way, state which you
    mean.

    ### 833 is the worksheet, 725 is the survey

    833 visits per field is the worksheet budget at 18,000 deg². The simulation gives a
    main-survey median of 725 and coadds about 0.1 mag shallower than the worksheet's. Two
    hand-ins contrasted them correctly; one quoted 833 as what fields get.

    ### Smaller slips, each from one notebook

    - **A "season" is the ~200-day stretch each year when a field is observable**, not a
      calendar quarter. A main-survey field gets about 35 nights per season, COSMOS about 130.
    - **A shallower visit is not a shorter one.** Reading 24.06 as "a 9 s exposure, so 620
      visits instead of 186" inverts the question: the survey loses 0.6 mag *at* 30 s and gets
      no visits back.
    - **Visit-rich pixels shorten the median gap, not lengthen it.**
    - **A light curve at MJD 50000** is 30 years before the survey starts (MJD 61208); no
      detections can appear. Check the time axis before interpreting an empty plot.
    - **A smaller sky magnitude is a brighter sky.** r and i have the *brightest* sky of the
      optical bands, not the darkest; they are preferred for shapes because of seeing and
      galaxy colors.
    - **Fill factor (91%) and the never-seen fraction (8.3%) are different quantities** that
      happen to be numerically close.
    - **Fast transients need deep single visits and short gaps**, not "short single-visit
      depth".
    - **Noise does not scale with the telescope diameter**, and field of view and plate scale
      are independent.
    - **One verification paragraph was copied from the other notebook.** Check that the numbers
      you quote come from the figure above them.
    """),
]

PART3_CELLS = [
    _c("md", """
    ## Part 3 · The numbers

    The values the class quoted most, with where they come from. "Design" is the worksheet
    and Ivezić et al. (2019); "simulated" is baseline v5.3.3; "as built" is Notebook B's
    camera model.

    | quantity | value | source |
    |---|---|---|
    | single-visit r depth | 24.7 design · 24.48 as built · 24.06 simulated median | A, B |
    | where the 0.6 mag goes | seeing −0.23 · sky −0.09 · airmass −0.02 | B `m5()` |
    | visits per field, 18,000 deg² | 833 worksheet · 725 simulated median | A |
    | 10-yr coadd r depth | 26.94 simulated median | A `compare_to_mine` |
    | revisit | 3 d median gap · 11 d at the 90th percentile · 1 d in deep fields | A cadence table |
    | kilonova caught within 2 d | 16% main survey · 42–52% deep fields | A `catch_table` |
    | delivered seeing | 0.7″ design · 0.91″ first night · 1.1″ first 15 days | B |
    | usable lensing galaxies | 41 at 0.7″ · 23 at 1.1″ per arcmin² (toy, with depth loss) | B `usable_density` |
    | time to reach r = 24.5 | 23 s at 0.7″ · 43 s at 1.0″ · 70 s at 1.3″ | B `m5()` |
    | 5 s visits per field | 5000 without overhead · 1786 with 9 s | A |
    | 60% budget | −0.28 mag in every band · 833 → 500 visits | A `plan_survey` |
    | main survey share | 80% of visits and of hours | A budget table |
    | chip gaps | 8.3% of a field never seen without dithering · 0% with | B `show_dithering` |
    | u band | 1 × 30 s is 0.25 mag deeper than 2 × 15 s | B `m5()` |
    """),
]

PART4_CELLS = [
    _c("md", """
    ## Part 4 · What the class got right

    Reached independently by several of you, and worth keeping:

    - **Seeing dominates the depth loss** (four hand-ins, three with evidence).
    - **The longest exposure before an asteroid trails is about the seeing divided by its
      angular speed**, 16 to 20 s at 0.04″/s (three hand-ins).
    - **Deep fields win on cadence, cover too little sky, so kilonova science needs
      gravitational-wave alerts and target-of-opportunity time** (four).
    - **Twilight NEO sweeps take more visits than hours** because the exposures are short at
      the same overhead (three).
    - **The main survey is 80% of visits and of hours** (three).
    - **Mirrors limit u; detector QE sets the red edge of y** (five).
    - **More area means fewer visits per field and a shallower coadd; less area means a
      shorter revisit** (five).
    - **You cannot stack a kilonova**: transients and moving objects break the coadd's
      constant-source assumption (three).
    - **The median gap hides a long tail**: 11 days at the 90th percentile in a main-survey
      field, a month at the south polar cap (four).
    - **The toy supernova count is an upper bound**: no cadence, no light-curve sampling, no
      host redshift (two).
    - **Dithering removes the never-seen fraction at the cost of a spread in visit counts**
      (three).

    And on disclosure: every hand-in that used an AI tool said what it was used for, and the
    best ones said how the result was checked. Keep doing that.
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
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def build_notebook():
    return _make_notebook(INTRO_CELLS + PART1_CELLS + PART2_CELLS + PART3_CELLS + PART4_CELLS)


def build(path=NB):
    """Write the debrief notebook (outputs cleared) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), path)
    return path


if __name__ == "__main__":
    print(build())
