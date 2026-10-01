#!/usr/bin/env bash
# Quality sweep for a constant-QP (hardware) encoder on KITTI 00, frames 0..N-1:
# MV matching and pairwise scoring at each AV1 qindex, then a summary table.
#
#   RUN= ENCODER=vaapi QPS="64 96 128 160" bash eval/sweep_qp.sh
#
# Options (environment): ENCODER (vaapi), QPS ("64 96 128 160"), N (117),
# RUN (command prefix, default "uv run"; RUN= for an active virtualenv).
# Outputs go to runs/sweep_<encoder>/; finished settings are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN-uv run}"
ENCODER="${ENCODER:-vaapi}"
QPS="${QPS:-64 96 128 160}"
N="${N:-117}"
K="718.856,607.1928,185.2157,0"  # KITTI 00 left colour camera (P2), k = 0

[ "$(ls data/kitti/00/image_2 2>/dev/null | wc -l)" -ge "$N" ] \
  || $RUN python eval/fetch_kitti.py --sequence 00 --frames "$N" --camera image_2 --out data/kitti
IMG="runs/sweep_${ENCODER}/img"
mkdir -p "$IMG"
for i in $(seq 0 $((N - 1))); do
  ln -sf "$PWD/data/kitti/00/image_2/$(printf %06d "$i").png" "$IMG/"
done

OUT="runs/sweep_${ENCODER}"
for q in $QPS; do
  if [ ! -e "$OUT/q$q.json" ]; then
    echo "== $ENCODER qp $q: matching"
    rm -f "$OUT/q$q.db"  # left over from an interrupted run
    $RUN av1sfm match "$IMG" "$OUT/q$q.db" --ivf "$OUT/q$q.ivf" --encode --encoder "$ENCODER" \
      --qp "$q" --camera-params "$K" --stats "$OUT/q$q.json" > /dev/null
  fi
  if [ ! -e "$OUT/score_q$q.json" ]; then
    echo "== $ENCODER qp $q: scoring"
    $RUN av1sfm score "$OUT/q$q.db" --out "$OUT/score_q$q.json" > /dev/null
  fi
done

$RUN python - "$OUT" $QPS <<'EOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
print("| qindex | Encode (s) | Matching (s) | Keypoints / img | Raw matches / img "
      "| Inlier ratio (pooled) | Inlier ratio (median of pairs) | Median Sampson (px) |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
for q in sys.argv[2:]:
    st = json.loads((out / f"q{q}.json").read_text())
    s = json.loads((out / f"score_q{q}.json").read_text())["summary"]
    t, m = st["timings"], st["matches"]
    match = sum(t[k]["wall_s"] for k in t if k != "encode")
    print(f"| {q} | {t['encode']['wall_s']:.2f} | {match:.1f} | {m['keypoints_per_image']:,.0f} "
          f"| {m['raw_matches_per_image']:,.0f} | {s['inlier_ratio_pooled']:.3f} "
          f"| {s.get('inlier_ratio_median_of_pairs', float('nan')):.3f} "
          f"| {s['sampson_px_median_of_pairs']:.3f} |")
EOF
