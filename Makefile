# epileval — thin Makefile dispatcher.
# All real logic lives in pyproject.toml + scripts/. Keep this short.

.PHONY: help install install-dev test test-fast lint format clean build docs

help:
	@echo "make install       Editable install (no extras)"
	@echo "make install-dev   Editable install with [dev] extras"
	@echo "make test          Run pytest with coverage"
	@echo "make test-fast     Run pytest -x -q (stop on first failure)"
	@echo "make lint          Run ruff check"
	@echo "make format        Run ruff format"
	@echo "make docs          Build Sphinx HTML in docs/sphinx/_build/html"
	@echo "make clean         Remove build artefacts"
	@echo "make build         Build sdist + wheel"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=src/epileval --cov-report=term-missing

test-fast:
	pytest tests/ -x -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

docs:
	sphinx-build -b html docs/sphinx docs/sphinx/_build/html

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build:
	python -m build
