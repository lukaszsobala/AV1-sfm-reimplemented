"""Descriptive statistics of the block MVs in an IVF (for ASSUMPTIONS.md).

uv run python eval/mv_stats.py runs/kitti117/clip.ivf
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from av1sfm.blocks import AOM_BLOCK_SIZES, block_origins
from av1sfm.extract import load_frame_motion


def main(ivf: str) -> None:
    frames = load_frame_motion(ivf)
    blocks = inter = zero = compound = odd = comps = 0
    gaps: Counter[int] = Counter()
    sizes: Counter[str] = Counter()
    for f in frames:
        if f.is_intra:
            continue
        gy, gx = block_origins(f.block_map)
        ref = f.ref[gy, gx]
        mv = f.mv[gy, gx]
        blocks += len(gy)
        is_inter = ref[:, 0] >= 1
        inter += int(is_inter.sum())
        compound += int((is_inter & (ref[:, 1] >= 1)).sum())
        m0 = mv[is_inter, :2]
        zero += int(np.all(m0 == 0, axis=1).sum())
        odd += int(np.sum(m0 % 2 != 0))
        comps += m0.size
        slot_frame = np.array((-1,) + f.ref_frame_index)
        for lst in (0, 1):
            ok = ref[:, lst] >= 1
            gaps.update((f.index - slot_frame[ref[ok, lst]]).tolist())
        sizes.update(
            f"{AOM_BLOCK_SIZES[b][0]}x{AOM_BLOCK_SIZES[b][1]}" for b in f.block_map[gy, gx]
        )
    tot = sum(gaps.values())
    out = {
        "frames": len(frames),
        "coded_blocks": blocks,
        "inter_block_fraction": inter / blocks,
        "compound_fraction_of_inter": compound / inter,
        "zero_mv_fraction_of_inter": zero / inter,
        "odd_eighth_pel_component_fraction": odd / comps,
        "reference_gap_fraction": {int(k): v / tot for k, v in sorted(gaps.items())},
        "top_block_sizes": dict(sizes.most_common(8)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
