"""Notebook B, Part 2 ("Seeing -> weak lensing").

Toy model: Science Book eq. 3.7 counts x Miller et al. 2013 (eq. B1) size distribution x a
resolution cut R2 > 1/3 (Mandelbaum et al. 2005), with the magnitude limit shifted by the
notebook's own m5() as the seeing changes. The usable density must fall monotonically with
seeing, and at the design seeing (0.7") it must land within 25% of Chang et al. 2013's
n_eff ~ 37 arcmin^-2.
"""
from tools import build_notebooks

CHECK = r'''
import json
out = {}
grid = np.linspace(0.4, 1.8, 57)
out["dens"] = [float(usable_density(f)) for f in grid]
out["dens_nodepth"] = [float(usable_density(f, depth=False)) for f in grid]
out["markers"] = {k: float(usable_density(v)) for k, v in SEEING_MARKERS.items()}
out["fid"] = float(usable_density(FWHM_DESIGN))
out["ilim_fid"] = float(i_limit(FWHM_DESIGN))
out["ilim_worse"] = float(i_limit(1.1))
out["frac_bright_vs_faint"] = [float(frac_resolved(20.0, 0.7)), float(frac_resolved(25.0, 0.7))]
out["frac_limits"] = [float(frac_resolved(25.0, 0.01)), float(frac_resolved(25.0, 20.0))]
out["stricter_cut_fewer"] = bool(usable_density(0.7, r_min=0.5) < usable_density(0.7))
out["widget_returns"] = repr(_lensing_widget(0.9, 1/3, True))
show_lensing(1.1, r_min=0.25, depth=False)
plt.close("all")
print("CHECKS" + json.dumps(out))
'''


def test_part2(run_notebook_checks):
    c = run_notebook_checks(CHECK, notebook="B")
    d = c["dens"]
    assert all(b < a for a, b in zip(d, d[1:])), d
    dn = c["dens_nodepth"]
    assert all(b < a for a, b in zip(dn, dn[1:])), dn
    # depth loss only ever removes galaxies beyond the design seeing
    assert all(x <= y + 1e-9 for x, y in zip(d[12:], dn[12:]))
    assert abs(c["fid"] / 37.0 - 1) < 0.25, c["fid"]
    m = c["markers"]
    assert set(m) == {"design", "median seeing", "first night", "first ~15 days"}
    assert m["design"] > m["median seeing"] > m["first night"] > m["first ~15 days"]
    assert abs(c["ilim_fid"] - 25.3) < 1e-9
    assert c["ilim_worse"] < 25.3
    bright, faint = c["frac_bright_vs_faint"]
    assert 0 < faint < bright <= 1
    tiny_psf, huge_psf = c["frac_limits"]
    assert tiny_psf > 0.99 and huge_psf < 1e-3
    assert c["stricter_cut_fewer"]
    assert c["widget_returns"] == "None"


def test_part2_source_rules():
    nb = build_notebooks.build_notebook_b()
    src = "\n".join(cell.source for cell in nb.cells)
    part2 = "\n".join(text for _, text in build_notebooks.B_PART2_CELLS)
    assert "## Part 2" in part2
    for cite in ("Miller et al. 2013", "Mandelbaum et al. 2005", "Chang et al. 2013",
                 "eq. 3.7", "Marshall"):
        assert cite in part2, cite
    for val in ("0.73", "0.91", "1.1", "0.078", "0.04", "0.45"):
        assert val in part2, val
    assert part2.count("Where does this break?") >= 2
    for word in ("colour", "centre", "modelling", "optimise", "normalis"):
        assert word not in src.lower()
    assert "scipy" not in src
