#!/usr/bin/env bash
# Full KITTI odometry 00 evaluation: MV matching vs COLMAP SIFT (exhaustive and
# sequential). Every command used for the results table is in this file.
#
#   bash eval/run_kitti.sh            # 117-frame SfM demo + 230-frame pairwise evaluation
#   SETS="117" bash eval/run_kitti.sh # only the 117-frame set
#
# Options (environment):
#   SETS     frame counts to run (default "117 230")
#   HW       hardware encoders to add as methods mv_<enc>, e.g. HW="qsv vaapi"
#   TORCH    1: add the PyTorch methods sift_seq_exact / sift_exh_exact (exact SIFT
#            matching) and disk_seq / disk_exh (DISK + LightGlue); docs/USAGE.md "GPU matchers"
#   DEVICE   PyTorch device for those: auto (default; Intel GPU, CUDA, else CPU), xpu, cpu
#   ONLY     run only these methods, e.g. ONLY="mv mv_qsv mv_vaapi" (default: all)
#   RUN      command prefix for Python tools (default "uv run"; RUN= for an active venv)
#
#   RUN= HW="qsv vaapi" ONLY="mv mv_qsv mv_vaapi" SETS=117 bash eval/run_kitti.sh
#   RUN= TORCH=1 ONLY="sift_seq_exact disk_seq" SETS=117 bash eval/run_kitti.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SETS="${SETS:-117 230}"
HW="${HW:-}"
TORCH="${TORCH:-}"
DEVICE="${DEVICE:-auto}"
ONLY="${ONLY:-}"
RUN="${RUN-uv run}"
METHODS="mv mv_trust mv_svt sift_seq sift_exh"
for e in $HW; do METHODS="$METHODS mv_$e"; done
[ -n "$TORCH" ] && METHODS="$METHODS sift_seq_exact sift_exh_exact disk_seq disk_exh"
want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }
# Steps whose output already exists are skipped; delete runs/kitti<N> to redo.
step() { local out="$1"; shift; [ -e "$out" ] && { echo "skip: $out exists"; return; }; "$@"; }
# Rectified left colour camera (P2): f, cx, cy; k = 0 for SIMPLE_RADIAL.
K="718.856,607.1928,185.2157,0"
# KITTI drives forward: the default initial-pair constraints never accept a pair.
MAPPER_ARGS="--init-max-forward-motion 1.0 --init-min-tri-angle 4"

[ -d data/kitti/00/image_2 ] && [ "$(ls data/kitti/00/image_2 | wc -l)" -ge 230 ] \
  || $RUN python eval/fetch_kitti.py --sequence 00 --frames 230 --camera image_2 --out data/kitti

for N in $SETS; do
  R="runs/kitti${N}"
  mkdir -p "$R/img"
  for i in $(seq 0 $((N - 1))); do
    ln -sf "$PWD/data/kitti/00/image_2/$(printf %06d "$i").png" "$R/img/"
  done

  # --- AV1 motion vectors (eps = 0.1, tau = 1 px, min track length 3) ---
  # Encoder: libaom, -usage realtime -cpu-used 6 -lag-in-frames 0, CRF 32 (defaults),
  # through the ffmpeg built by scripts/build_ffmpeg.sh when present.
  want mv && step "$R/mv.json" $RUN av1sfm match "$R/img" "$R/mv.db" --ivf "$R/clip.ivf" --encode \
    --camera-params "$K" --stats "$R/mv.json"
  want mv_trust && step "$R/mv_trust.json" $RUN av1sfm match "$R/img" "$R/mv_trust.db" \
    --ivf "$R/clip.ivf" --camera-params "$K" --two-view trust --stats "$R/mv_trust.json"
  # Same pipeline on an SVT-AV1 stream (low-delay RTC, preset 10, CRF 32).
  want mv_svt && step "$R/mv_svt.json" $RUN av1sfm match "$R/img" "$R/mv_svt.db" \
    --ivf "$R/clip_svt.ivf" --encode --encoder svtav1 --camera-params "$K" --stats "$R/mv_svt.json"
  # Hardware encoders (constant qindex 128, see ASSUMPTIONS.md E5b-E5f).
  for e in $HW; do
    want "mv_$e" && step "$R/mv_$e.json" $RUN av1sfm match "$R/img" "$R/mv_$e.db" \
      --ivf "$R/clip_$e.ivf" --encode --encoder "$e" --camera-params "$K" --stats "$R/mv_$e.json"
  done

  # --- COLMAP SIFT baselines (CPU) ---
  want sift_seq && step "$R/sift_seq.json" $RUN python eval/run_sift.py "$R/img" "$R/sift_seq.db" --matching sequential \
    --overlap 10 --camera-params "$K" --stats "$R/sift_seq.json"
  want sift_exh && step "$R/sift_exh.json" $RUN python eval/run_sift.py "$R/img" "$R/sift_exh.db" --matching exhaustive \
    --camera-params "$K" --stats "$R/sift_exh.json"

  # --- PyTorch baselines (TORCH=1): same pairs and verification as above ---
  # Exact SIFT matching (= COLMAP's --brute-force result, ASSUMPTIONS.md R4).
  for P in seq:sequential exh:exhaustive; do
    M="sift_${P%%:*}_exact"
    [[ " $METHODS " == *" $M "* ]] && want "$M" || continue
    step "$R/$M.json" $RUN python eval/run_sift.py "$R/img" "$R/$M.db" --matching "${P#*:}" \
      --overlap 10 --matcher exact --device "$DEVICE" --camera-params "$K" --stats "$R/$M.json"
  done
  # DISK + LightGlue (the paper's Table I learned baseline).
  for P in seq:sequential exh:exhaustive; do
    M="disk_${P%%:*}"
    [[ " $METHODS " == *" $M "* ]] && want "$M" || continue
    step "$R/$M.json" $RUN python eval/run_lightglue.py "$R/img" "$R/$M.db" --matching "${P#*:}" \
      --overlap 10 --device "$DEVICE" --camera-params "$K" --stats "$R/$M.json"
  done

  # --- identical pairwise geometric scoring of raw matches ---
  for M in $METHODS; do
    [ "$M" = mv_trust ] && continue  # same raw matches as mv
    want "$M" || continue
    step "$R/score_$M.json" $RUN av1sfm score "$R/$M.db" --repeats 3 --out "$R/score_$M.json"
  done

  # --- SfM demo (117 frames only) ---
  # Primary: intrinsics fixed at the KITTI calibration (identical for all methods).
  # Secondary: COLMAP's default intrinsics refinement (map_refine_*), reported
  # because it breaks MV reconstructions (see ASSUMPTIONS.md R5).
  if [ "$N" = 117 ]; then
    for M in $METHODS; do
      want "$M" || continue
      step "$R/map_$M.json" $RUN python eval/run_mapper.py "$R/$M.db" "$R/img" "$R/rec_$M" \
        $MAPPER_ARGS --fix-intrinsics --stats "$R/map_$M.json"
      step "$R/map_refine_$M.json" $RUN python eval/run_mapper.py "$R/$M.db" "$R/img" \
        "$R/rec_refine_$M" $MAPPER_ARGS --stats "$R/map_refine_$M.json"
    done
  fi
done
$RUN python eval/make_table.py runs > runs/results.md
cat runs/results.md
