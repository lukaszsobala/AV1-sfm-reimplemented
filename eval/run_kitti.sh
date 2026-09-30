#!/usr/bin/env bash
# Full KITTI odometry 00 evaluation: MV matching vs COLMAP SIFT (exhaustive and
# sequential). Every command used for the results table is in this file.
#
#   bash eval/run_kitti.sh            # 117-frame SfM demo + 230-frame pairwise evaluation
#   SETS="117" bash eval/run_kitti.sh # only the 117-frame set
set -euo pipefail
cd "$(dirname "$0")/.."

SETS="${SETS:-117 230}"
# Steps whose output already exists are skipped; delete runs/kitti<N> to redo.
step() { local out="$1"; shift; [ -e "$out" ] && { echo "skip: $out exists"; return; }; "$@"; }
# Rectified left colour camera (P2): f, cx, cy; k = 0 for SIMPLE_RADIAL.
K="718.856,607.1928,185.2157,0"
# KITTI drives forward: the default initial-pair constraints never accept a pair.
MAPPER_ARGS="--init-max-forward-motion 1.0 --init-min-tri-angle 4"

[ -d data/kitti/00/image_2 ] && [ "$(ls data/kitti/00/image_2 | wc -l)" -ge 230 ] \
  || uv run python eval/fetch_kitti.py --sequence 00 --frames 230 --camera image_2 --out data/kitti

for N in $SETS; do
  R="runs/kitti${N}"
  mkdir -p "$R/img"
  for i in $(seq 0 $((N - 1))); do
    ln -sf "$PWD/data/kitti/00/image_2/$(printf %06d "$i").png" "$R/img/"
  done

  # --- AV1 motion vectors (eps = 0.1, tau = 1 px, min track length 3) ---
  # Encoder: libaom, -usage realtime -cpu-used 6 -lag-in-frames 0, CRF 32 (defaults),
  # through the ffmpeg built by scripts/build_ffmpeg.sh when present.
  step "$R/mv.json" uv run av1sfm match "$R/img" "$R/mv.db" --ivf "$R/clip.ivf" --encode \
    --camera-params "$K" --stats "$R/mv.json"
  step "$R/mv_trust.json" uv run av1sfm match "$R/img" "$R/mv_trust.db" --ivf "$R/clip.ivf" \
    --camera-params "$K" --two-view trust --stats "$R/mv_trust.json"
  # Same pipeline on an SVT-AV1 stream (low-delay RTC, preset 10, CRF 32).
  step "$R/mv_svt.json" uv run av1sfm match "$R/img" "$R/mv_svt.db" --ivf "$R/clip_svt.ivf" \
    --encode --encoder svtav1 --camera-params "$K" --stats "$R/mv_svt.json"

  # --- COLMAP SIFT baselines (CPU) ---
  step "$R/sift_seq.json" uv run python eval/run_sift.py "$R/img" "$R/sift_seq.db" --matching sequential \
    --overlap 10 --camera-params "$K" --stats "$R/sift_seq.json"
  step "$R/sift_exh.json" uv run python eval/run_sift.py "$R/img" "$R/sift_exh.db" --matching exhaustive \
    --camera-params "$K" --stats "$R/sift_exh.json"

  # --- identical pairwise geometric scoring of raw matches ---
  for M in mv mv_svt sift_seq sift_exh; do
    step "$R/score_$M.json" uv run av1sfm score "$R/$M.db" --repeats 3 --out "$R/score_$M.json"
  done

  # --- SfM demo (117 frames only) ---
  # Primary: intrinsics fixed at the KITTI calibration (identical for all methods).
  # Secondary: COLMAP's default intrinsics refinement (map_refine_*), reported
  # because it breaks MV reconstructions (see ASSUMPTIONS.md R5).
  if [ "$N" = 117 ]; then
    for M in mv mv_trust mv_svt sift_seq sift_exh; do
      step "$R/map_$M.json" uv run python eval/run_mapper.py "$R/$M.db" "$R/img" "$R/rec_$M" \
        $MAPPER_ARGS --fix-intrinsics --stats "$R/map_$M.json"
      step "$R/map_refine_$M.json" uv run python eval/run_mapper.py "$R/$M.db" "$R/img" \
        "$R/rec_refine_$M" $MAPPER_ARGS --stats "$R/map_refine_$M.json"
    done
  fi
done
uv run python eval/make_table.py runs > runs/results.md
cat runs/results.md
