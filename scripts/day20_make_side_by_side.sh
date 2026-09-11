#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 REFERENCE_MP4 RESIDUAL_MP4 OUTPUT_MP4"
  exit 2
fi

REF="$1"
RES="$2"
OUT="$3"

ffmpeg -y \
  -i "$REF" \
  -i "$RES" \
  -filter_complex \
  "[0:v]setpts=PTS-STARTPTS[left];[1:v]setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2[v]" \
  -map "[v]" \
  -shortest \
  -c:v libx264 \
  -crf 18 \
  -pix_fmt yuv420p \
  "$OUT"

echo "Created: $OUT"
