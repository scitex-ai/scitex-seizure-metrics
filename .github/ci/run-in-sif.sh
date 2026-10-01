#!/usr/bin/env bash
# Runs inside the digest-verified CI SIF. Install this checkout's complete
# declared test environment into job-owned writable scratch; never retry
# with fewer extras. Tests use only package-owned temporary state.
set -euo pipefail
V="${1:?python version required}"
VENV="/opt/venv-$V"
PY="$VENV/bin/python"
test -x "$PY" || { echo "::error::baked python missing in $VENV"; exit 1; }
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export TMPDIR="/tmp/ci-scitex_seizure_metrics-${GITHUB_RUN_ID:-0}-${GITHUB_RUN_ATTEMPT:-0}-$V"
rm -rf "${TMPDIR:?}"
mkdir -p "$TMPDIR/site" "$TMPDIR/uv-cache" "$TMPDIR/scitex-state"
export UV_CACHE_DIR="$TMPDIR/uv-cache" XDG_CACHE_HOME="$TMPDIR/cache"
export PIP_CACHE_DIR="$TMPDIR/pip-cache" MPLCONFIGDIR="$TMPDIR/mpl"
export SCITEX_DIR="$TMPDIR/scitex-state" MPLBACKEND=Agg RUN_E2E=1
mkdir -p "$MPLCONFIGDIR"
unset VIRTUAL_ENV || true
export PATH="$VENV/bin:$PATH"
uv pip install --python "$PY" --target="$TMPDIR/site" -e ".[all,dev]"
export PYTHONPATH="$TMPDIR/site:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
NPROC="$(nproc 2>/dev/null || echo 4)"
WORKERS=$NPROC
[ "$WORKERS" -lt 4 ] && WORKERS=4
echo "py=$("$PY" -V) xdist workers=$WORKERS (nproc=$NPROC) RUN_E2E=$RUN_E2E"
if "$PY" -c "import matplotlib" 2>/dev/null; then
    "$PY" -c "from matplotlib import font_manager; font_manager.fontManager"
fi
exec nice -n 19 ionice -c 3 \
    "$PY" -m pytest tests/ -n "$WORKERS" --dist load -q \
    --cov=src/scitex_seizure_metrics --cov-report=xml --cov-report=term \
    -p no:cacheprovider
