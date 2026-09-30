"""Notebook B, Part 3 ("The focal plane"), Part 4 (go deeper) and the hand-in.

Focal-plane geometry is the nominal LSSTCam layout in lsst/obs_lsst (policy/rafts.yaml,
policy/cameraHeader.yaml, policy/lsstCam/R*.yaml). The fill factor computed from it must land
within 0.01 of Veres & Chesley 2017's 90.8% and inside the published range (Science Book 93%,
O'Connor et al. 2019 ">90%"). Dithering must reduce the fraction of never-covered pixels.
"""
import re

from tools import build_notebooks


def _handin_example():
    """The example call written (commented out) in the hand-in code cell."""
    kind, text = build_notebooks.B_PART4_CELLS[-2]
    assert kind == "code"
    calls = [line.lstrip("# ").strip() for line in text.splitlines()
             if re.match(r"#\s*_?\s*=?\s*\w+\(", line)]
    assert calls, text
    return calls[0]


CHECK = r'''
import json, time
out = {}
ccds = science_ccds()
out["n_ccds"] = int(len(ccds))
out["n_rafts"] = int(ccds["raft"].nunique())
out["vendors"] = {k: int(v) for k, v in ccds["vendor"].value_counts().items()}
out["n_corner"] = int(len(corner_sensors()))
out["fill"] = float(fill_factor())
out["area_deg2"] = float(active_area_deg2())
out["pix_arcsec"] = float(PIXEL_MM * PLATE_SCALE)
out["center_on"] = bool(on_chip(np.array([0.0]), np.array([0.0]))[0])
# halfway between R22 S11 and S12 (x = 21.125 mm): a gap
out["gap_off"] = bool(on_chip(np.array([21.125]), np.array([0.0]))[0])
# center of a corner raft: not a science CCD
out["corner_off"] = bool(on_chip(np.array([254.0]), np.array([254.0]))[0])
out["circle"] = {k: float(v) for k, v in circle_vs_silicon().items()}

t0 = time.time()
s1 = show_dithering(n_visits=20, dither_deg=0.7, rotate=True, seed=367)
out["t_show"] = time.time() - t0
s2 = show_dithering(n_visits=20, dither_deg=0.7, rotate=True, seed=367)
out["stats"] = s1
out["repeat_same"] = bool(s1 == s2)
s3 = show_dithering(n_visits=20, dither_deg=0.7, rotate=True, seed=1)
out["other_seed_differs"] = bool(s3 != s1)
out["widget_returns"] = repr(_dither_widget(10, 0.5, True))
draw_focal_plane()
EXAMPLE
plt.close("all")
print("CHECKS" + json.dumps(out))
'''


def test_part3(run_notebook_checks):
    c = run_notebook_checks(CHECK.replace("EXAMPLE", _handin_example()), notebook="B")
    assert c["n_ccds"] == 189 and c["n_rafts"] == 21
    assert c["vendors"] == {"E2V": 117, "ITL": 72}
    assert c["n_corner"] == 16             # 4 corner rafts x (2 guiders + 2 half wavefront sensors)
    # fill factor: computed from the drawn geometry vs the published values
    assert abs(c["fill"] - 0.908) < 0.01, c["fill"]
    assert 0.90 < c["fill"] < 0.93
    assert abs(c["area_deg2"] - 9.6) < 0.2, c["area_deg2"]
    assert abs(c["pix_arcsec"] - 0.2) < 0.001
    assert c["center_on"] and not c["gap_off"] and not c["corner_off"]
    circ = c["circle"]
    assert 0.8 < circ["circle_on_silicon"] < 0.95
    assert 0.0 < circ["silicon_outside_circle"] < 0.2
    und, dit = c["stats"]["undithered"], c["stats"]["dithered"]
    # without dithering, the zero-coverage pixels are the gaps: roughly 1 - fill factor
    assert 0.04 < und["frac_zero"] < 0.15, und
    assert dit["frac_zero"] < 0.5 * und["frac_zero"], (und, dit)
    assert dit["frac_zero"] < 0.01
    assert und["max"] == 20 and dit["max"] <= 20
    assert dit["std"] < und["std"]
    assert c["repeat_same"] and c["other_seed_differs"]
    assert c["t_show"] < 5.0, c["t_show"]
    assert c["widget_returns"] == "None"


def test_part3_part4_source_rules():
    part3 = "\n".join(text for _, text in build_notebooks.B_PART3_CELLS)
    part4 = "\n".join(text for _, text in build_notebooks.B_PART4_CELLS)
    assert "## Part 3" in part3
    for cite in ("Ivezić et al. 2019", "obs_lsst", "Vereš & Chesley 2017", "Science Book",
                 "O'Connor et al. 2019", "90.8%", "93%", "1.75°", "Notebook A",
                 "camera_footprint_demo"):
        assert cite in part3, cite
    assert part3.count("Where does this break?") >= 1
    assert "selection function" in part3
    for link in ("SilverVsAluminum", "201_10_Visit_table",
                 "https://lsstdesc.org/pages/DESchool.html",
                 "https://github.com/GalSim-developers/GalSim",
                 "camera_footprint_demo.ipynb", "Roodman"):
        assert link in part4, link
    nb = build_notebooks.build_notebook_b()
    src = "\n".join(cell.source for cell in nb.cells)
    for word in ("colour", "centre", "modelling", "optimise", "normalis", "analyse"):
        assert word not in src.lower(), word
    for pkg in ("scipy", "healpy", "rubin_sim", "galsim"):
        assert f"import {pkg}" not in src and f"from {pkg}" not in src


def test_handin_last():
    nb = build_notebooks.build_notebook_b()
    last, code, handin = nb.cells[-1], nb.cells[-2], nb.cells[-3]
    assert last.cell_type == "markdown" and "Your paragraph" in last.source
    assert code.cell_type == "code" and "direct function call" in code.source
    assert handin.cell_type == "markdown" and handin.source.startswith("## Hand-in")
    for s in ("Monday Oct 5", "Canvas", "one figure", "one paragraph", "AI tools",
              "complete/incomplete", ".ipynb or a PDF"):
        assert s in handin.source, s
    assert sum("## Hand-in" in cell.source for cell in nb.cells) == 1


def test_readme_badge():
    readme = (build_notebooks.REPO / "README.md").read_text()
    assert "coming Wednesday" not in readme
    assert ("https://colab.research.google.com/github/rhw/phys367/blob/main/week2/"
            "B_camera_to_science.ipynb") in readme
    assert "2B · Camera" in readme
