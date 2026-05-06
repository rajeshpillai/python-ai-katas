#!/usr/bin/env bash
set -euo pipefail

#
# build-static.sh — Build a static, self-contained version of the app for
# GitHub Pages (or any static host).
#
# What it does:
#   1. Exports the kata index from backend/python/app/routes/katas.py to
#      frontend/public/katas-index.json
#   2. Copies backend/python/content/ to frontend/public/content/
#   3. Runs the Vite build with VITE_STATIC_BUILD=true and the configured
#      base path, producing frontend/dist/
#
# Usage:
#   ./scripts/build-static.sh
#
# Env:
#   GHPAGES_BASE_PATH   Base path for assets (default: derived from origin
#                       remote, e.g. /python-ai-katas/). Set to "/" for
#                       custom-domain or root deploys.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Helpers ----------------------------------------------------------------

info()  { printf "\033[1;34m=> %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m=> %s\033[0m\n" "$*"; }
error() { printf "\033[1;31m=> %s\033[0m\n" "$*" >&2; exit 1; }

# --- Pre-flight -------------------------------------------------------------

command -v npm     >/dev/null || error "npm not found in PATH"
command -v python3 >/dev/null || error "python3 not found in PATH"

PYTHON_BACKEND="$REPO_ROOT/backend/python"
[[ -d "$PYTHON_BACKEND" ]] || error "Python backend not found at $PYTHON_BACKEND"

# Pick a Python interpreter — prefer the backend venv, fall back to system.
if [[ -x "$PYTHON_BACKEND/.venv/bin/python" ]]; then
    PY="$PYTHON_BACKEND/.venv/bin/python"
elif [[ -x "$PYTHON_BACKEND/venv/bin/python" ]]; then
    PY="$PYTHON_BACKEND/venv/bin/python"
else
    PY="python3"
    warn "No backend venv found. Falling back to system python3."
    warn "If the export step fails (missing fastapi), run:"
    warn "    cd backend/python && uv venv && uv pip install -r requirements.txt"
fi

# --- Resolve base path ------------------------------------------------------

if [[ -z "${GHPAGES_BASE_PATH:-}" ]]; then
    if origin_url="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null)"; then
        # Strip .git, take repo name from URL — works for both ssh and https
        repo_name="${origin_url%.git}"
        repo_name="${repo_name##*/}"
        if [[ -n "$repo_name" ]]; then
            GHPAGES_BASE_PATH="/$repo_name/"
        fi
    fi
fi
GHPAGES_BASE_PATH="${GHPAGES_BASE_PATH:-/}"

info "Base path: $GHPAGES_BASE_PATH"

# --- 1. Export kata index ---------------------------------------------------

INDEX_OUT="$REPO_ROOT/frontend/public/katas-index.json"
mkdir -p "$(dirname "$INDEX_OUT")"

info "Exporting kata index → $INDEX_OUT"
cd "$PYTHON_BACKEND"
"$PY" -c "
import json, sys
sys.path.insert(0, '.')
from app.routes.katas import TRACK_REGISTRY
out = {
    track_id: {
        'katas': data['katas'],
        'phases': {str(k): v for k, v in data['phases'].items()},
    }
    for track_id, data in TRACK_REGISTRY.items()
}
print(json.dumps(out, indent=2))
" > "$INDEX_OUT"
cd "$REPO_ROOT"

# --- 2. Copy kata markdown content (per-language) --------------------------

CONTENT_DST="$REPO_ROOT/frontend/public/content"
info "Copying kata content → $CONTENT_DST/{python,rust}"
rm -rf "$CONTENT_DST"
mkdir -p "$CONTENT_DST"

# Python content is required.
PY_CONTENT="$REPO_ROOT/backend/python/content"
[[ -d "$PY_CONTENT" ]] || error "Python content not found at $PY_CONTENT"
mkdir -p "$CONTENT_DST/python"
cp -a "$PY_CONTENT/." "$CONTENT_DST/python/"

# Rust content is optional (deploys are still useful with Python alone).
RUST_CONTENT="$REPO_ROOT/backend/rust/content"
if [[ -d "$RUST_CONTENT" ]]; then
    mkdir -p "$CONTENT_DST/rust"
    cp -a "$RUST_CONTENT/." "$CONTENT_DST/rust/"
else
    warn "Rust content not found at $RUST_CONTENT — skipping (Rust katas will 404 on the static deploy)."
fi

# --- 3. Vite build ----------------------------------------------------------

cd "$REPO_ROOT/frontend"

if [[ ! -d node_modules ]]; then
    info "Installing frontend deps (npm install)"
    npm install
fi

info "Running Vite build (VITE_STATIC_BUILD=true, base=$GHPAGES_BASE_PATH)"
VITE_STATIC_BUILD=true VITE_BASE_PATH="$GHPAGES_BASE_PATH" npm run build

# --- Done -------------------------------------------------------------------

DIST="$REPO_ROOT/frontend/dist"
[[ -d "$DIST" ]] || error "Build produced no dist/ directory"

info "Static build ready in $DIST"
info "Try it locally:"
info "    npx serve $DIST -l 4173"
info "    # or: python3 -m http.server -d $DIST 4173"
