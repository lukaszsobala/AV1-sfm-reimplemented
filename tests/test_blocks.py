import numpy as np
import pytest
from synth import bsize, make_frame, set_block

from av1sfm.blocks import MotionLookup, block_centers, block_origins, frame_block_motion


def test_block_origins_collapse_mixed_partition():
    bm = np.full((8, 8), bsize(8, 8), np.uint8)  # 32x32 px of 8x8 blocks = 16 blocks
    bm[0:4, 0:4] = bsize(16, 16)  # top-left 16x16 replaces 4 of them
    bm[4:5, 4:8] = bsize(16, 4)  # a 16x4 strip
    bm[5:6, 4:8] = bsize(16, 4)
    gy, gx = block_origins(bm)
    origins = set(zip(gy.tolist(), gx.tolist()))
    assert (0, 0) in origins and (0, 2) not in origins and (2, 0) not in origins
    assert (4, 4) in origins and (5, 4) in origins and (4, 6) not in origins
    # 1 (16x16) + 12 (8x8) - 2 (8x8 replaced by the two strips) + 2 strips
    assert len(origins) == 1 + 12 - 2 + 2 + 0


def test_block_centers_colmap_convention_and_clipping():
    c = block_centers(np.array([0, 56]), np.array([0, 32]), np.array([16, 16]),
                      np.array([16, 16]), width=60, height=40)  # fmt: skip
    # first block covers pixels 0..15 -> centre 8.0; second overhangs to x=60, y=40
    np.testing.assert_allclose(c, [[8.0, 8.0], [58.0, 36.0]])


def test_frame_block_motion_one_keypoint_per_block_and_mv_geometry():
    fm = make_frame(3, 64, 64, block=(32, 32), mv_px=(-2.5, 1.25))
    bm = frame_block_motion(fm)
    assert len(bm) == 4  # four 32x32 blocks, not 64 grid cells
    np.testing.assert_allclose(
        sorted(map(tuple, bm.center)), [(16, 16), (16, 48), (48, 16), (48, 48)]
    )
    np.testing.assert_allclose(bm.mv, np.tile([-2.5, 1.25], (4, 1)))
    np.testing.assert_allclose(bm.target, bm.center + [-2.5, 1.25])
    assert (bm.ref_frame == 2).all() and (bm.ref_list == 0).all()


def test_frame_block_motion_filters_zero_intra_and_out_of_frame():
    fm = make_frame(5, 64, 64, block=(16, 16), mv_px=(3.0, 0.0))
    set_block(fm, 0, 0, 16, 16, mv_px=(0.0, 0.0))  # ambiguous zero MV
    set_block(fm, 16, 0, 16, 16, ref0=0)  # intra block inside an inter frame
    set_block(fm, 48, 16, 16, 16, mv_px=(20.0, 0.0))  # target leaves the frame
    bm = frame_block_motion(fm)
    assert len(bm) == 16 - 3
    assert len(frame_block_motion(fm, skip_zero=False)) == 16 - 2
    assert len(frame_block_motion(fm, keep_out_of_frame=True)) == 16 - 2
    assert len(frame_block_motion(make_frame(0, intra=True))) == 0


def test_reference_slot_resolution_compound_and_prev_only():
    fm = make_frame(10, 32, 32, block=(16, 16), mv_px=(1.0, 1.0),
                    ref_frames=(9, 8, 7, 6, 5, 4, 3))  # fmt: skip
    set_block(fm, 0, 0, 16, 16, ref0=4)  # GOLDEN slot -> frame 6
    set_block(fm, 16, 0, 16, 16, ref1=2, mv1_px=(2.0, 2.0))  # compound LAST + LAST2
    bm = frame_block_motion(fm)
    assert len(bm) == 5
    assert sorted(bm.ref_frame.tolist()) == [6, 8, 9, 9, 9]
    two = bm.ref_list == 1
    np.testing.assert_allclose(bm.mv[two], [[2.0, 2.0]])
    assert len(frame_block_motion(fm, use_compound=False)) == 4
    assert sorted(frame_block_motion(fm, prev_only=True).ref_frame.tolist()) == [9, 9, 9]


def test_motion_lookup_prefers_nearest_reference():
    fm = make_frame(10, 32, 32, block=(16, 16), mv_px=(1.0, 0.0),
                    ref_frames=(9, 8, 7, 6, 5, 4, 3))  # fmt: skip
    set_block(fm, 0, 0, 16, 16, ref0=4, ref1=1, mv1_px=(-1.0, 0.0))  # frame 6 vs frame 9
    lk = MotionLookup.from_frame(fm)
    mv, ref = lk.query(np.array([[3.0, 3.0], [20.0, 3.0], [-1.0, 3.0], [31.9, 31.9]]))
    assert ref.tolist() == [9, 9, -1, 9]
    np.testing.assert_allclose(mv[0], [-1.0, 0.0])
    np.testing.assert_allclose(mv[1], [1.0, 0.0])


@pytest.mark.parametrize("prev_only", [False, True])
def test_motion_lookup_zero_and_prev_only(prev_only):
    fm = make_frame(4, 32, 32, block=(16, 16), mv_px=(1.0, 0.0), ref_frames=(3, 2, 1, 0, 0, 0, 0))
    set_block(fm, 0, 0, 16, 16, mv_px=(0.0, 0.0))
    set_block(fm, 16, 0, 16, 16, ref0=2)
    lk = MotionLookup.from_frame(fm, prev_only=prev_only)
    _, ref = lk.query(np.array([[1.0, 1.0], [17.0, 1.0], [1.0, 17.0]]))
    assert ref.tolist() == [-1, -1 if prev_only else 2, 3]
