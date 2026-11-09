# Physics 367: Astrophysical Data Analysis with the Rubin Observatory (Autumn 2026, Stanford)

Welcome to Physics 367! In this course, you will work hands-on with real Rubin Observatory data (Data Preview 1 and early Data Preview 2 on the Rubin Science Platform) as well as public Legacy Survey of Space and Time (LSST) survey simulations to explore the fundamentals of astrophysical data analysis.

For course details and lecture materials, visit: [Physics 367 Course Page](https://web.stanford.edu/group/gfc/Fall26_367/)

---

## Week 2: First Steps with Survey Data

**2A · Survey strategy → science** (due Mon Oct 5)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhw/phys367/blob/main/week2/A_survey_strategy.ipynb)

**2B · Camera → science** (due Mon Oct 5)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhw/phys367/blob/main/week2/B_camera_to_science.ipynb)

**Week 2 debrief** · what the class found, and where the reasoning slipped (posted after the hand-in)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhw/phys367/blob/main/week2/debrief.ipynb)

---

## Week 3: From Images to Catalogs

Week 3 runs on the **Rubin Science Platform** (<https://data.lsst.cloud>, LSST kernel), not Colab: the notebooks use the Butler, the TAP service, and catalog extracts stored on the RSP. In an RSP terminal, `git clone https://github.com/rhw/phys367.git` (or `git pull` if you already have it) and open the notebooks under `week3/`.

Tutorials from Wednesday's hands-on (Alex Broughton and Eli Rykoff):

- [`qualify_a_dataset.ipynb`](week3/qualify_a_dataset.ipynb) · Is this catalog any good? Survey property maps, a coadd with its catalog and masks, counts and depth.
- [`dp2_full_sky_stars_phys367.ipynb`](week3/dp2_full_sky_stars_phys367.ipynb) · Bright stars over the DP2 footprint, star/galaxy separation, sky maps.
- [`dp2_cosmos_phys367.ipynb`](week3/dp2_cosmos_phys367.ipynb) · The COSMOS deep field: stellar locus, galaxy colors, PSF size residuals.

**3 · Hand-in** (due Mon Oct 12): [`handin.ipynb`](week3/handin.ipynb). Pick one of four exercises (calibration, selection, deblending, depth), make its figure, write the paragraph.


## Week 4: Photometric redshifts

Week 4 will need to download some data.

Tutorials from Wednesday's hands-on (Eric Charles and Tianqing (TQ) Zhang):

- [`week4_A_ExploreFluxes.ipynb`](week4/week4_A_ExploreFluxes.ipynb) · Looking at different types of flux measurements that are present in the Rubin catalogs.
- [`week4_B_ExploreInputs.ipynb`](week4/week4_B_ExploreInputs.ipynb) · Looking at the inputs for photometric redshift estimation algorithms.
- [`week4_C_SklearnRegression.ipynb`](week4/week4_C_SklearnRegression.ipynb) · A very simple example using scikit-learn to build a photometric redshift estimator.

- [`week4_Extra_Morphology.ipynb`](week4/week4_Extra_Morphology.ipynb) · Explores some of the quantities in the catalog related to object morphology.
- [`week4_Extra_SklearnMorphology.ipynb`](week4/week4_Extra_SklearnMorphology.ipynb) · Uses scikit-learn to build a photometric redshift estimator that includes object morphology. 
- [`week4_Extra_GalaxySize.ipynb`](week4/week4_Extra_GalaxySize.ipynb) · Explores the relationship between redshift and galaxy size.
- [`week4_Extra_AdmixStudy.ipynb`](week4/week4_Extra_AdmixStudy.ipynb) · Explores the effect of flux contamination (i.e., blending) on redshift estimation.
- [`week4_Extra_NoiseStudy.ipynb`](week4/week4_Extra_NoiseStudy.ipynb) · Explores the effect of photometric noise on redshift estimation.

**4 · Hand-in** (due Mon Oct 19): [`week4_handin.ipynb`](week4/week4_handin.ipynb). Pick one of six exercises, make its figure, write the paragraph.

Note: you may need to add the packages in `week4/requirements.txt` (e.g., with `pip install -r week4/requirements.txt`)  to your conda environment to run these notebooks. 

## Week 8: Galaxy clustering with Rubin DP2

Week 8 runs on the **Rubin Science Platform** (<https://data.lsst.cloud>, LSST kernel), like Week 3.

[`week8_galaxy_clustering.ipynb`](week8/week8_galaxy_clustering.ipynb) · Build a DES-style MagLim lens sample from real DP2 imaging, mask it and correct it for observational systematics, and measure its angular clustering two ways: the angular correlation function w(θ) and the angular power spectrum C(ℓ).

Before opening the notebook, in an RSP terminal: `git clone https://github.com/rhw/phys367.git` (or `git pull`), then `pip install -r week8/requirements.txt`. This covers everything except `pymaster` (NaMaster), which builds from source and isn't needed until Section 5 — the notebook tells you exactly when and how to install it there.

Section 2 of the notebook queries ~3.4 million objects over the full ~113 deg² region and joins photo-z — about **6 minutes**, mostly spent waiting on the TAP query rather than on CPU. It only has to happen once: the notebook caches its own result to `week8/data/` (not checked into the repo — it's over GitHub's 100 MB file-size limit) and reuses that cache on every later run. If you'd rather not eat those 6 minutes inside your first read-through, run `python week8/prepare_data.py` once beforehand instead — it does exactly that same step, standalone, and leaves the same cache behind.

## Run Locally

To run the notebooks on your own computer:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rhw/phys367.git
   cd phys367
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas pyarrow matplotlib astropy ipywidgets jupyter
   
   # for week4 you might need to also do:
   pip install -r week4/requirements.txt
   ```

3. **Open a notebook** (Week 2 Notebook A or B; the Week 3 and Week 8 notebooks need the RSP instead):
   ```bash
   jupyter notebook week2/A_survey_strategy.ipynb
   jupyter notebook week2/B_camera_to_science.ipynb
   ```

---

## Hand-in Policy

Each notebook assignment is graded **complete/incomplete** based on the hand-in cell.

- **Total notebooks:** ~9 throughout the course
- **Required to pass:** Turn in at least 7
- **Due date:** Each notebook is due on **Canvas the Monday after it is released**
- **Hand-in format:** Your final submission must include:
  - **One figure** (can be generated by code or a plot)
  - **One paragraph** (your interpretation or summary)
  - **An AI-use note** (if you used any generative AI tools, list them; if not, you can write "none")

Upload the .ipynb or a PDF to Canvas.

---

## About the Data

The survey simulation data in this repository are public; they are not Rubin data products, so the course's rules about not uploading Rubin data to external AI services do not apply to them. (Please still note any AI tools you use in your hand-in.)

---

## Support

Questions: ask in the course Slack channel #fall2026-367-rubindata.
