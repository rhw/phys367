"""Shared pytest fixtures for the notebook tests (controller ruling R2).

``run_notebook_checks(check_code)`` builds week2/A_survey_strategy.ipynb from
tools/build_notebooks.py, appends ``check_code`` as a final cell, executes the
notebook headless with nbclient (cwd week2/, kernel "phys367", 600 s per cell),
and returns the JSON dict printed on the line that starts with ``CHECKS``.
"""
import json
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
WEEK2 = REPO / "week2"
KERNEL = "phys367"


def _execute_with_check(check_code: str) -> dict:
    from tools import build_notebooks

    nb_path = build_notebooks.build()
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(check_code))
    client = NotebookClient(nb, timeout=600, kernel_name=KERNEL,
                            resources={"metadata": {"path": str(WEEK2)}})
    client.execute()
    for out in nb.cells[-1].get("outputs", []):
        if out.get("output_type") == "stream":
            for line in out.get("text", "").splitlines():
                if line.startswith("CHECKS"):
                    return json.loads(line[len("CHECKS"):])
    raise AssertionError("no CHECKS line in final cell output: "
                         f"{nb.cells[-1].get('outputs')}")


@pytest.fixture(scope="session")
def run_notebook_checks():
    """Factory: run_notebook_checks(check_code) -> dict."""
    return _execute_with_check
