---
description: |
  [TOPIC] epileval Installation
  [DETAILS] How to install epileval and its optional extras — `[plots]` for matplotlib, `[test]` / `[dev]` for the test/lint stack, `[docs]` for Sphinx, `[all]` for everything; through the SciTeX umbrella as `pip install scitex[epileval]`; editable contributor install with `pip install -e ".[dev]"` and `make test` / `make docs`. Python 3.10+ supported.
tags: [epileval-installation]
---

# Installation

## Core install

```bash
pip install epileval
```

Pulls only the runtime deps: `numpy >= 1.26`, `pandas >= 2.0`,
`scikit-learn >= 1.3`, `timescoring >= 0.0.5`.

## Optional extras

| Extra        | What it adds                                                            |
| ------------ | ----------------------------------------------------------------------- |
| `[plots]`    | `matplotlib` — required for the `epileval.plots` submodule.             |
| `[test]`     | `pytest`, `pytest-cov`, `pytest-timeout`, `matplotlib`.                 |
| `[dev]`      | Everything in `[test]` + `ruff` + `scitex-dev` for the full toolchain.  |
| `[docs]`     | `sphinx`, `furo`, `myst-parser`, `sphinx-copybutton`, type-hints.       |
| `[all]`      | `[plots]` + `[docs]` + `[dev]`.                                         |

```bash
pip install "epileval[plots]"
pip install "epileval[dev]"
pip install "epileval[all]"
```

## Through the SciTeX umbrella

```bash
pip install "scitex[epileval]"
```

Imports become `scitex.epileval.detection`, `scitex.epileval.forecasting`,
… via the umbrella's thin shim. They resolve to the same objects as
`epileval.detection` / `epileval.forecasting`.

## Editable install (contributors)

```bash
git clone https://github.com/ywatanabe1989/epileval
cd epileval
pip install -e ".[dev]"
make test     # runs pytest with coverage
make docs     # builds Sphinx HTML in docs/sphinx/_build/html
```

## Python compatibility

Python ≥ 3.10. CI tests 3.10 / 3.11 / 3.12.
