#!/usr/bin/env bash
# Gerrard Hall and Person Hall (COLMAP example datasets): MV matching vs COLMAP
# SIFT, pairwise metrics as in the paper's Tables II-IV. The paper disables the
# cosine filter on these image collections (eps = 1). Every command behind the
# results table is in this file.
#
#   bash eval/run_colmap_datasets.sh              # both datasets, pairwise metrics
#   SFM=1 bash eval/run_colmap_datasets.sh        # also incremental mapping
set -euo pipefail
cd "$(dirname "$0")/.."

NAMES="${NAMES:-gerrard-hall person-hall}"
SRC="${SRC:-data/colmap}"
BASE="https://github.com/colmap/colmap/releases/download/3.11.1"
step() { local out="$1"; shift; [ -e "$out" ] && { echo "skip: $out exists"; return; }; "$@"; }

fetch() {  # fetch <name> <files...>: download and unpack a dataset release asset
  local name="$1"; shift
  [ -d "$SRC/$name/images" ] && return
  mkdir -p "$SRC"
  for f in "$@"; do curl -sSL --retry 3 -o "$SRC/$f" "$BASE/$f"; done
  if [ "$#" -gt 1 ]; then  # split archive: join, then unpack
    zip -q -s 0 "$SRC/$name.zip" --out "$SRC/$name-joined.zip"
    unzip -q "$SRC/$name-joined.zip" -d "$SRC"
    rm -f "$SRC/$name-joined.zip"
  else
    unzip -q "$SRC/$name.zip" -d "$SRC"
  fi
  for f in "$@"; do rm -f "$SRC/$f"; done
}

for NAME in $NAMES; do
  case "$NAME" in
    gerrard-hall) fetch gerrard-hall gerrard-hall.zip ;;
    person-hall) fetch person-hall person-hall.z01 person-hall.zip ;;
  esac
  R="runs/$NAME"
  # Longest run of same-size consecutive images, long side 1920 px, shared camera.
  step "$R/prepare.json" uv run python eval/prepare_colmap_dataset.py "$SRC/$NAME" "$R"
  K="$(cat "$R/camera.txt")"

  step "$R/mv.json" uv run av1sfm match "$R/img" "$R/mv.db" --ivf "$R/clip.ivf" --encode \
    --eps 1 --camera-params "$K" --stats "$R/mv.json"
  step "$R/mv_trust.json" uv run av1sfm match "$R/img" "$R/mv_trust.db" --ivf "$R/clip.ivf" \
    --eps 1 --camera-params "$K" --two-view trust --stats "$R/mv_trust.json"
  step "$R/mv_svt.json" uv run av1sfm match "$R/img" "$R/mv_svt.db" --ivf "$R/clip_svt.ivf" \
    --encode --encoder svtav1 --eps 1 --camera-params "$K" --stats "$R/mv_svt.json"

  step "$R/sift_seq.json" uv run python eval/run_sift.py "$R/img" "$R/sift_seq.db" --matching sequential \
    --overlap 10 --camera-params "$K" --stats "$R/sift_seq.json"
  step "$R/sift_exh.json" uv run python eval/run_sift.py "$R/img" "$R/sift_exh.db" --matching exhaustive \
    --camera-params "$K" --stats "$R/sift_exh.json"

  for M in mv mv_svt sift_seq sift_exh; do
    step "$R/score_$M.json" uv run av1sfm score "$R/$M.db" --repeats 3 --out "$R/score_$M.json"
  done

  if [ "${SFM:-0}" = 1 ]; then
    for M in mv mv_trust mv_svt sift_seq sift_exh; do
      step "$R/map_$M.json" uv run python eval/run_mapper.py "$R/$M.db" "$R/img" "$R/rec_$M" \
        --fix-intrinsics --stats "$R/map_$M.json"
    done
  fi
done
uv run python eval/make_table.py runs > runs/results.md
cat runs/results.md
