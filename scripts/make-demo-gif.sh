#!/usr/bin/env bash
set -euo pipefail

#
# make-demo-gif.sh — convert a screen-recording MP4 into a Reddit-ready GIF.
#
# Reddit auto-converts GIFs > ~20MB into looping video ("RedGifs"), which
# strips the autoplay-everywhere appeal. This script targets ~10-15MB by:
#   - capping width (default 800px — readable, not bloated)
#   - capping frame rate (default 15fps — smooth, half the bytes of 30fps)
#   - two-pass encoding with a generated palette (clean colors, no banding)
#
# Usage:
#   ./scripts/make-demo-gif.sh INPUT.mp4 [OUTPUT.gif]
#
# Examples:
#   ./scripts/make-demo-gif.sh demo.mp4
#   ./scripts/make-demo-gif.sh demo.mp4 out/ai-katas.gif
#   WIDTH=1080 FPS=20 ./scripts/make-demo-gif.sh demo.mp4
#   START=00:00:03 DURATION=00:00:30 ./scripts/make-demo-gif.sh demo.mp4
#
# Env:
#   WIDTH       Output width in px            (default: 800)
#   FPS         Output frame rate             (default: 15)
#   START       Trim start (HH:MM:SS or sec)  (optional)
#   DURATION    Length to keep                (optional, e.g. 00:00:45)
#   DITHER      none | bayer | sierra2_4a     (default: sierra2_4a — best quality)
#

# --- Args -------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 INPUT.mp4 [OUTPUT.gif]" >&2
    echo "Run with --help for env-var options." >&2
    exit 1
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '3,30p' "$0"
    exit 0
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}.gif}"

[[ -f "$INPUT" ]] || { echo "Input not found: $INPUT" >&2; exit 1; }

# --- Config -----------------------------------------------------------------

WIDTH="${WIDTH:-800}"
FPS="${FPS:-15}"
DITHER="${DITHER:-sierra2_4a}"

# --- Dependencies -----------------------------------------------------------

command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg not found. Install it first:" >&2
    echo "  macOS:  brew install ffmpeg" >&2
    echo "  Ubuntu: sudo apt install ffmpeg" >&2
    exit 1
}

# --- Helpers ----------------------------------------------------------------

info()  { printf "\033[1;34m=> %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m=> %s\033[0m\n" "$*"; }

# --- Trim args (optional) ---------------------------------------------------

TRIM_ARGS=()
[[ -n "${START:-}" ]]    && TRIM_ARGS+=(-ss "$START")
[[ -n "${DURATION:-}" ]] && TRIM_ARGS+=(-t "$DURATION")

# --- Filter graph -----------------------------------------------------------

# fps + scale (lanczos = sharp); split palettegen and paletteuse so
# the final GIF uses an optimal 256-color palette derived from the clip.
FILTERS="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"

PALETTE="$(mktemp -t demo-palette.XXXXXX.png)"
trap 'rm -f "$PALETTE"' EXIT

# --- Pass 1: generate palette ----------------------------------------------

info "Pass 1/2 — generating palette (width=${WIDTH}, fps=${FPS})"
ffmpeg -loglevel error -y \
    "${TRIM_ARGS[@]}" \
    -i "$INPUT" \
    -vf "${FILTERS},palettegen=stats_mode=diff" \
    "$PALETTE"

# --- Pass 2: encode GIF ----------------------------------------------------

info "Pass 2/2 — encoding GIF → $OUTPUT"
mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -loglevel error -y \
    "${TRIM_ARGS[@]}" \
    -i "$INPUT" \
    -i "$PALETTE" \
    -lavfi "${FILTERS}[v];[v][1:v]paletteuse=dither=${DITHER}" \
    "$OUTPUT"

# --- Report -----------------------------------------------------------------

SIZE_BYTES=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT")
SIZE_MB=$(awk "BEGIN { printf \"%.1f\", ${SIZE_BYTES} / 1048576 }")

info "Done. ${OUTPUT} — ${SIZE_MB} MB"

# --- Soft warnings ----------------------------------------------------------

if (( $(awk "BEGIN { print (${SIZE_MB} > 20) }") )); then
    warn "Over 20MB — Reddit will probably convert this to MP4 ('RedGif')."
    warn "To shrink: lower WIDTH (e.g. 640), lower FPS (e.g. 12), or trim with START/DURATION."
elif (( $(awk "BEGIN { print (${SIZE_MB} > 8) }") )); then
    info "Looks good for a Reddit upload (under 20MB threshold)."
else
    info "Comfortably under the Reddit GIF size limit. Embeds will autoplay everywhere."
fi
