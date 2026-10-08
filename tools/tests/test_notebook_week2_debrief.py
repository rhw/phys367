"""Structure checks for the Week 2 debrief notebook (no network needed)."""
import nbformat

from tools.build_week2_debrief import build_notebook


def _sources(nb, kind):
    return ["".join(c.source) for c in nb.cells if c.cell_type == kind]


def test_notebook_is_valid_with_outputs_cleared():
    nb = build_notebook()
    nbformat.validate(nb)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def test_four_parts_in_order_and_anonymous():
    md = "\n".join(_sources(build_notebook(), "markdown"))
    parts = ["## Part 1", "## Part 2", "## Part 3", "## Part 4"]
    assert [md.index(p) for p in parts] == sorted(md.index(p) for p in parts)
    # no student surnames from the Canvas export may appear
    for name in ("Zhang", "Wang", "Xia", "Baptista", "Hartley", "Murgia", "Sherwin", "Cheng", "Choi",
                 "Gendreau", "Yang", "Barac", "Caudillo", "Sajkov"):
        assert name not in md, name


def test_code_cells_compile_and_bootstrap_loads_both_notebooks():
    code = _sources(build_notebook(), "code")
    for src in code:
        compile(src, "<cell>", "exec")
        assert "from tools" not in src and "import tools" not in src
    assert 'load_notebook_functions("A_survey_strategy.ipynb")' in code[0]
    assert 'load_notebook_functions("B_camera_to_science.ipynb")' in code[0]
    assert "raw.githubusercontent.com/rhw/phys367/main/week2/" in code[0]
