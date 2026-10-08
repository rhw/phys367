"""Structure checks for the Week 3 hand-in notebook (no RSP needed)."""
import nbformat

from tools.build_week3_handin import build_notebook


def _sources(nb, kind):
    return ["".join(c.source) for c in nb.cells if c.cell_type == kind]


def test_notebook_is_valid_and_outputs_cleared():
    nb = build_notebook()
    nbformat.validate(nb)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    assert nb.metadata["kernelspec"]["name"] == "lsst"


def test_four_exercises_and_handin_in_order():
    md = "\n".join(_sources(build_notebook(), "markdown"))
    headings = [h for h in ("## Exercise 1", "## Exercise 2", "## Exercise 3", "## Exercise 4", "## Hand-in")
                if h in md]
    assert headings == ["## Exercise 1", "## Exercise 2", "## Exercise 3", "## Exercise 4", "## Hand-in"]
    assert [md.index(h) for h in headings] == sorted(md.index(h) for h in headings)
    assert "Oct 12" in md
    assert md.count("> **Go further.**") == 4 and "**AI tools:**" in md and "at least one number" in md


def test_starter_cells_compile_and_stay_off_tools():
    for src in _sources(build_notebook(), "code"):
        compile(src, "<cell>", "exec")
        assert "from tools" not in src and "import tools" not in src


def test_data_path_is_overridable():
    code = "\n".join(_sources(build_notebook(), "code"))
    assert 'os.environ.get("PHYS367_DATA", "/home/erykoff/data")' in code
    assert "cosmos_object_selection.fits" in code
