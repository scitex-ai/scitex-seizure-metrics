"""Sphinx configuration for scitex-seizure-metrics."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "scitex-seizure-metrics"
author = "Yusuke Watanabe"
copyright = "2026, Yusuke Watanabe"

try:
    from importlib.metadata import version as _v

    release = _v("scitex-seizure-metrics")
except Exception:
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# Enable myst markdown features needed by docs/math/sample_to_alarm.md:
# - dollarmath: $...$ and $$...$$ render as inline / display math
# - amsmath: \begin{align*}...\end{align*} blocks
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Robust docs-CI defense (ecosystem template): suppress only the
# docutils reST-parser rendering noise that free-form module/function
# docstrings routinely produce under autodoc — e.g. a wrapped bullet
# whose continuation line docutils reads as a nested block quote
# ("Unexpected indentation" / "Block quote ends without a blank line").
# These are benign formatting artifacts, NOT broken docs.
#
# Scoped to the "docutils" category ONLY — real structural breakage
# (missing toctree pages [toc.not_readable], broken cross-references
# [ref.*], missing math renderer, undefined directives) is emitted under
# OTHER categories and still trips `sphinx-build -W`. This keeps -W
# strict where it matters while not whack-a-moling every docstring.
suppress_warnings = ["docutils"]

html_theme = "furo"
html_title = f"scitex-seizure-metrics {release}"
html_static_path = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
