#!/usr/bin/env bash
set -euo pipefail

#
# publish-gh-pages.sh — Publish the static build to a GitHub Pages branch.
#
# Usage:
#   ./scripts/publish-gh-pages.sh                 # dry-run (default)
#   ./scripts/publish-gh-pages.sh --push          # actually push
#
# Env:
#   GHPAGES_REMOTE      Git remote name or URL (default: origin)
#   GHPAGES_BRANCH      Branch to publish to (default: gh-pages)
#   GHPAGES_BASE_PATH   Asset base path (default: derived from remote URL)
#   GHPAGES_MESSAGE     Custom commit message (optional)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Helpers ----------------------------------------------------------------

info()  { printf "\033[1;34m=> %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m=> %s\033[0m\n" "$*"; }
error() { printf "\033[1;31m=> %s\033[0m\n" "$*" >&2; exit 1; }

# --- Args & config ----------------------------------------------------------

DRY_RUN=true
if [[ "${1:-}" == "--push" ]]; then
    DRY_RUN=false
fi

GHPAGES_REMOTE="${GHPAGES_REMOTE:-origin}"
GHPAGES_BRANCH="${GHPAGES_BRANCH:-gh-pages}"
GHPAGES_MESSAGE="${GHPAGES_MESSAGE:-}"

# --- Pre-flight -------------------------------------------------------------

command -v git  >/dev/null || error "git not found in PATH"
command -v npm  >/dev/null || error "npm not found in PATH"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]]; then
    warn "Working tree has uncommitted changes — proceeding anyway."
fi

# --- 1. Run the static build ------------------------------------------------

info "Running static build"
export GHPAGES_BASE_PATH="${GHPAGES_BASE_PATH:-}"
bash "$SCRIPT_DIR/build-static.sh"

DIST="$REPO_ROOT/frontend/dist"
[[ -d "$DIST" ]] || error "build-static.sh did not produce $DIST"

# GitHub Pages bypasses Jekyll only with this marker file
touch "$DIST/.nojekyll"

# --- 2. Summary -------------------------------------------------------------

info "Static dist tree (top 3 levels):"
(cd "$DIST" && find . -maxdepth 3 -not -path '*/.*' | sort | head -50)

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    warn "DRY RUN — nothing pushed."
    warn "Review the tree above. To publish:"
    warn "    $0 --push"
    exit 0
fi

# --- 3. Resolve remote URL --------------------------------------------------

# If GHPAGES_REMOTE is already a URL, use it; otherwise resolve via git
if [[ "$GHPAGES_REMOTE" =~ ^(https?|git|ssh):// || "$GHPAGES_REMOTE" =~ ^git@ ]]; then
    REMOTE_URL="$GHPAGES_REMOTE"
else
    REMOTE_URL="$(git -C "$REPO_ROOT" remote get-url "$GHPAGES_REMOTE" 2>/dev/null)" \
        || error "Remote '$GHPAGES_REMOTE' not found. Set GHPAGES_REMOTE or add the remote."
fi

info "Publishing to $REMOTE_URL ($GHPAGES_BRANCH)"

# --- 4. Stage on a fresh worktree of the gh-pages branch --------------------

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

if git clone --depth=1 --branch "$GHPAGES_BRANCH" "$REMOTE_URL" "$WORK_DIR" 2>/dev/null; then
    info "Cloned existing $GHPAGES_BRANCH branch"
    # Wipe everything except .git so the new build is the full state
    find "$WORK_DIR" -mindepth 1 -maxdepth 1 -not -name '.git' -exec rm -rf {} +
else
    warn "Branch $GHPAGES_BRANCH not found on remote — initializing fresh"
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"
    git -C "$WORK_DIR" init -b "$GHPAGES_BRANCH" -q
    git -C "$WORK_DIR" remote add origin "$REMOTE_URL"
fi

cp -a "$DIST"/. "$WORK_DIR"/

cd "$WORK_DIR"
git add -A

if git diff --cached --quiet 2>/dev/null; then
    warn "No changes — nothing to push."
    exit 0
fi

SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
if [[ -n "$GHPAGES_MESSAGE" ]]; then
    COMMIT_MSG="$GHPAGES_MESSAGE

Source: $SOURCE_SHA"
else
    COMMIT_MSG="Deploy static build $(date +%Y-%m-%d)

Source: $SOURCE_SHA"
fi

git commit -q -m "$COMMIT_MSG"
git push origin "$GHPAGES_BRANCH"

info "Done. Pushed to $REMOTE_URL ($GHPAGES_BRANCH)"
info "Enable Pages at: Settings → Pages → Source: $GHPAGES_BRANCH branch / root"
