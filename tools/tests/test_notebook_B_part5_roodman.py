"""Notebook B: additions aligned with Aaron Roodman's Sep 30 lecture ("Rubin Observatory Design
Rationale").

The camera calculator in Part 3 must reproduce his slide numbers from first principles:
0.2" per 10 um pixel at f = 10.3 m (slide 23), f/1.23 for an 8.4 m primary (slides 10, 23),
~3.1e9 pixels for 9.6 deg^2 at 0.2" (he quotes 3.2 billion for "10 square degrees"), and a 2 s
readout for 16 Mpix / 16 amplifiers at 500 kHz (slides 25-26). Part 2's new camera_fwhm term
must leave every existing density unchanged at its default (0) and lower the density at 0.3".
"""
from tools import build_notebooks

CHECK = r'''
import json, inspect
out = {}
cam = camera_numbers(verbose=False)
out["cam"] = {k: float(v) for k, v in cam.items()}
out["cam_15um"] = float(camera_numbers(pixel_um=15, verbose=False)["pixel_scale"])
out["cam_amps"] = float(camera_numbers(amps_per_ccd=32, verbose=False)["t_read"])
camera_numbers()                                   # the printed version runs
out["widget_returns"] = repr(_camera_widget(10.3, 8.4, 10.0, 16, 500))
out["dens0"] = {k: float(usable_density(v, camera_fwhm=0.0)) for k, v in SEEING_MARKERS.items()}
out["dens"] = {k: float(usable_density(v)) for k, v in SEEING_MARKERS.items()}
out["dens_nodepth0"] = float(usable_density(0.91, depth=False, camera_fwhm=0.0))
out["dens_nodepth"] = float(usable_density(0.91, depth=False))
out["dens_cam03"] = float(usable_density(0.7, camera_fwhm=0.3))
out["dens_cam03_equiv"] = float(usable_density(float(np.hypot(0.7, 0.3))))
out["diffusion_fwhm"] = float(DIFFUSION_FWHM)
out["qe_1000"] = float(QE_1000)
out["camera_fwhm_default"] = inspect.signature(usable_density).parameters["camera_fwhm"].default
plt.close("all")
print("CHECKS" + json.dumps(out))
'''


def test_roodman_numbers(run_notebook_checks):
    c = run_notebook_checks(CHECK, notebook="B")
    cam = c["cam"]
    assert abs(cam["plate_scale"] - 206265 / 10300) < 1e-6        # arcsec per mm
    assert abs(cam["pixel_scale"] - 0.2) < 0.001                  # 0.2" per 10 um pixel
    assert abs(cam["f_ratio"] - 1.23) < 0.005
    assert 3.05e9 < cam["n_pixels"] < 3.15e9                      # 9.6 deg^2 at 0.2"
    assert abs(cam["n_pixels_10deg2"] - 3.24e9) < 0.02e9          # Roodman's 10 deg^2
    assert abs(cam["t_read"] - 2.0) < 1e-9
    assert abs(cam["pixels_per_fwhm"] - 2.0) < 0.01               # 0.4" best image blur
    assert abs(cam["shutter_open_no_slew"] - 30 / 34) < 1e-9
    assert abs(cam["shutter_open"] - 30 / 39) < 1e-9             # plus a 5 s slew and settle
    assert abs(c["cam_15um"] - 0.3) < 0.002
    assert abs(c["cam_amps"] - 1.0) < 1e-9
    assert c["widget_returns"] == "None"
    # camera_fwhm defaults to 0 and changes nothing
    assert c["camera_fwhm_default"] == 0
    for k, v in c["dens"].items():
        assert c["dens0"][k] == v, k
    assert c["dens_nodepth0"] == c["dens_nodepth"]
    # a 0.3" camera term, added in quadrature, lowers the density
    assert c["dens_cam03"] < c["dens"]["design"]
    assert abs(c["dens_cam03"] - c["dens_cam03_equiv"]) < 1e-9
    # sigma = 4 um of diffusion at 0.2"/10 um: ~0.08" sigma, ~0.19" FWHM
    assert abs(c["diffusion_fwhm"] - 0.19) < 0.005
    assert 0.0 < c["qe_1000"] < 1.0


def test_roodman_source_rules():
    part1 = "\n".join(t for _, t in build_notebooks.B_PART1_CELLS)
    part2 = "\n".join(t for _, t in build_notebooks.B_PART2_CELLS)
    part3 = "\n".join(t for _, t in build_notebooks.B_PART3_CELLS)
    intro = build_notebooks.B_INTRO_CELLS[0][1]
    assert "Sep 30" in intro and "Roodman" in intro
    for part in ("Part 1", "Part 2", "Part 3"):
        assert part in intro, part
    for cite in ("Roodman, Sep 30 lecture, slide 18", "Roodman, Sep 30 lecture, slide 23",
                 "Roodman, Sep 30 lecture, slides 25–26", "Roodman, Sep 30 lecture, slide 27"):
        assert cite in part3, cite
    assert "The numbers that set the camera" in part3
    assert part3.index("The numbers that set the camera") < part3.index("def draw_focal_plane")
    assert "0.5mm gaps" in part3 and "What else sits between two CCDs' imaging pixels?" in part3
    assert "3.09 billion" in part3 and "3.2 billion" in part3
    assert "5 secs" in part3 and "~5 second slews" in part3
    assert "15 µm" in part3 and "300 s" in part3
    assert "Point Spread Function from Camera < 0.3”" in part2
    assert "slide 35" in part2 and "slide 20" in part2
    assert "double count" in part2
    assert "30% QE at λ=1000nm" in part1 and "slide 22" in part1
    assert "319 m² deg²" in part1 and "slide 3" in part1
    # his slides are not public: cite by name and date, never link a file
    src = "\n".join(cell.source for cell in build_notebooks.build_notebook_b().cells)
    assert ".key" not in src and ".pdf" not in src.lower()
