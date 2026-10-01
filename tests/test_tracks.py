import numpy as np
import pytest
from synth import make_frame, set_block

from av1sfm.blocks import MotionLookup
from av1sfm.tracks import TrackParams, build_tracks, tracks_to_matches


def pan_sequence(n=5, w=64, h=64, mv=(-2.0, -1.0), block=(16, 16)):
    """Frame 0 intra, frames 1..n-1 each predicted from the previous frame."""
    return [make_frame(0, w, h, intra=True)] + [
        make_frame(i, w, h, block=block, mv_px=mv) for i in range(1, n)
    ]


def test_pan_tracks_follow_the_motion_chain():
    tr = build_tracks(pan_sequence(), TrackParams())
    L = tr.lengths()
    assert L.max() == 5  # frames 4 -> 3 -> 2 -> 1 -> 0
    t = int(np.argmax(L))
    sel = tr.track == t
    frames, xy = tr.frame[sel], tr.xy[sel]
    assert frames.tolist() == [4, 3, 2, 1, 0]
    np.testing.assert_allclose(np.diff(xy, axis=0), np.tile([-2.0, -1.0], (4, 1)))


def test_uncovered_seeding_only_where_no_track_arrives():
    tr = build_tracks(
        pan_sequence(n=3, mv=(0.25, 0.25)), TrackParams(min_length=1, seed="uncovered")
    )
    # frame 2 seeds 16 blocks; they land in the same 16 blocks of frame 1 (shift < 4 px),
    # so frame 1 seeds nothing new; frame 0 is intra and seeds nothing either.
    assert tr.stats["seeds"] == 16
    assert (tr.lengths() == 3).all()


def test_seed_all_emits_every_block_of_every_frame():
    # Paper: every block of every inter frame is a source keypoint (16 + 16), and
    # tracks arriving from frame 2 add their target points to frame 1 as well.
    tr = build_tracks(pan_sequence(n=3, mv=(-4.0, 0.0)), TrackParams(min_length=1))
    assert tr.stats["seeds"] == 32
    kp_frame1 = int(np.sum(tr.frame == 1))
    assert kp_frame1 == 16 + 16  # own centres + all arrivals (a 4 px shift stays inside)


def test_cell_grid_uses_every_4x4_unit():
    frames = pan_sequence(n=3, mv=(-4.0, 0.0), block=(32, 32))
    blk = build_tracks(frames, TrackParams(min_length=1, grid="block"))
    cell = build_tracks(frames, TrackParams(min_length=1, grid="cell"))
    assert blk.stats["seeds"] == 2 * 4  # four 32x32 blocks per frame
    assert cell.stats["seeds"] == 2 * 16 * 16  # every 4x4 unit of a 64x64 frame
    np.testing.assert_allclose(sorted(set(cell.xy[:, 0] % 4)), [2.0])  # cell centres


def test_min_length_drops_short_tracks():
    frames = pan_sequence(n=2)  # a single inter frame: every track has length 2
    assert build_tracks(frames, TrackParams(min_length=3)).num_tracks == 0
    assert build_tracks(frames, TrackParams(min_length=2)).num_tracks == 16


def turning_sequence():
    """Frames 3,2 move left, frame 1 moves up: a 90-degree turn at frame 2 -> 1."""
    f = [make_frame(0, 64, 64, intra=True), make_frame(1, 64, 64, mv_px=(0.0, -4.0))]
    f += [make_frame(i, 64, 64, mv_px=(-4.0, 0.0)) for i in (2, 3)]
    return f


def test_cosine_violation_splits_track():
    tr = build_tracks(turning_sequence(), TrackParams(eps=0.1, min_length=2, on_violation="split"))
    assert tr.stats["cosine_violations"] > 0
    # The turn is in triple (2, 1, 0): tracks are split at frame 1, so (3, 2, 1)
    # survives but no track spans 2 -> 1 -> 0.
    spans = [set(tr.frame[tr.track == t].tolist()) for t in range(tr.num_tracks)]
    assert any({3, 2, 1} <= fr for fr in spans)
    assert not any({2, 1, 0} <= fr for fr in spans)
    assert any(fr == {1, 0} for fr in spans)  # the continuation after the split


def test_cosine_violation_cut_terminates_track():
    tr = build_tracks(turning_sequence(), TrackParams(eps=0.1, min_length=2, on_violation="cut"))
    spans = [tr.frame[tr.track == t].tolist() for t in range(tr.num_tracks)]
    assert [3, 2, 1] in spans  # terminated at frame 1, not continued to 0
    assert not any({2, 1, 0} <= set(fr) for fr in spans)
    assert [1, 0] in spans  # frame 1's own block centres still start tracks


def test_cosine_violation_drop_mode_and_eps_one():
    drop = build_tracks(turning_sequence(), TrackParams(eps=0.1, on_violation="drop"))
    assert all(3 not in set(drop.frame[drop.track == t].tolist()) for t in range(drop.num_tracks))
    off = build_tracks(turning_sequence(), TrackParams(eps=1.0))
    assert off.lengths().max() == 4 and off.stats["cosine_violations"] == 0


def test_tau_skips_test_for_small_motion():
    f = [make_frame(0, 64, 64, intra=True), make_frame(1, 64, 64, mv_px=(0.0, -0.5))]
    f += [make_frame(i, 64, 64, mv_px=(-4.0, 0.0)) for i in (2, 3)]
    tr = build_tracks(f, TrackParams(eps=0.1, tau=1.0))
    assert tr.stats["cosine_violations"] == 0 and tr.lengths().max() == 4


def test_track_stops_at_zero_or_intra_block():
    frames = pan_sequence(n=4, mv=(-4.0, 0.0))
    set_block(frames[2], 0, 0, 64, 64, mv_px=(0.0, 0.0))  # whole frame 2 static
    tr = build_tracks(frames, TrackParams(min_length=1))
    spans = [set(tr.frame[tr.track == t].tolist()) for t in range(tr.num_tracks)]
    assert {3, 2} in spans  # tracks from frame 3 end in static frame 2
    assert not any(3 in fr and 1 in fr for fr in spans)


def test_non_previous_reference_links_frames_with_gap():
    frames = pan_sequence(n=4, mv=(-2.0, 0.0))
    frames[3] = make_frame(
        3, 64, 64, mv_px=(-4.0, 0.0), ref_slot=4, ref_frames=(2, 1, 0, 1, 0, 0, 0)
    )
    tr = build_tracks(frames, TrackParams())
    spans = [tr.frame[tr.track == t].tolist() for t in range(tr.num_tracks)]
    assert [3, 1, 0] in spans  # 3 -> 1 directly (GOLDEN slot 4 = frame 1), then 1 -> 0
    t = spans.index([3, 1, 0])
    np.testing.assert_allclose(np.diff(tr.xy[tr.track == t], axis=0), [[-4.0, 0.0], [-2.0, 0.0]])
    tr2 = build_tracks(frames, TrackParams(prev_only=True))
    assert 3 not in tr2.frame.tolist()


def test_matches_are_triangular_and_consistent():
    tr = build_tracks(pan_sequence(n=5), TrackParams())
    mg = tracks_to_matches(tr)
    pairs = set(mg.matches)
    assert (0, 4) in pairs and (1, 3) in pairs and (3, 4) in pairs
    assert all(a < b for a, b in pairs)
    for (a, b), m in mg.matches.items():
        d = mg.keypoints[b][m[:, 1]] - mg.keypoints[a][m[:, 0]]
        np.testing.assert_allclose(
            d, np.tile([2.0 * (b - a), 1.0 * (b - a)], (len(m), 1)), atol=1e-4
        )
    gap = tracks_to_matches(tr, max_pair_gap=1)
    assert set(gap.matches) == {(0, 1), (1, 2), (2, 3), (3, 4)}


def test_keypoints_unique_per_frame_track():
    tr = build_tracks(pan_sequence(n=6, mv=(-3.0, 2.0)), TrackParams())
    for f in np.unique(tr.frame):
        ids = tr.track[tr.frame == f]
        assert len(ids) == len(np.unique(ids))


def test_invalid_violation_mode():
    with pytest.raises(ValueError):
        TrackParams(on_violation="bogus")
    with pytest.raises(ValueError):
        TrackParams(grid="pixel")


def test_match_count_guard():
    from av1sfm.tracks import TooManyMatches

    tr = build_tracks(pan_sequence(n=5), TrackParams())
    with pytest.raises(TooManyMatches):
        tracks_to_matches(tr, max_matches=10)
    assert tracks_to_matches(tr, max_pair_gap=1, max_matches=10).matches  # gap bypasses guard


def future_reference_sequence():
    """Content pans +4 px per frame. Decode order: 0 (key), 2, 1; frame 1 uses frame 2."""
    key = make_frame(0, intra=True)
    f2 = make_frame(2, mv_px=(-8.0, 0.0), ref_frames=(0,) * 7)
    f1 = make_frame(1, mv_px=(4.0, 0.0), ref_frames=(2,) * 7)
    f2.decode_index, f2.ref_decode_index = 1, (0,) * 7
    f1.decode_index, f1.ref_decode_index = 2, (1,) * 7
    return [key, f2, f1]


def test_tracks_follow_future_references_with_consistent_direction():
    tr = build_tracks(future_reference_sequence(), TrackParams())
    assert tr.stats["cosine_violations"] == 0  # +4 px forward and -8 px back: same motion
    spans = [tr.frame[tr.track == t].tolist() for t in range(tr.num_tracks)]
    assert [2, 1, 0] in spans
    xy = tr.xy[tr.track == spans.index([2, 1, 0])]
    np.testing.assert_allclose(xy[:, 0] - xy[1, 0], [4.0, 0.0, -4.0])  # frames 2, 1, 0
    mg = tracks_to_matches(tr)
    assert {(0, 1), (0, 2), (1, 2)} <= set(mg.matches)


def test_hidden_frame_and_overlay_never_put_a_track_twice_in_one_image():
    # Decode order 0 (key), 1 = hidden alt-ref shown at 2, 2 = frame 1, 3 = overlay at 2.
    key = make_frame(0, intra=True)
    arf = make_frame(2, mv_px=(-8.0, 0.0), ref_frames=(0,) * 7)
    f1 = make_frame(1, mv_px=(4.0, 0.0), ref_frames=(2,) * 7)
    ovl = make_frame(2, mv_px=(-4.0, 0.0), ref_frames=(1, 2, 0, 0, 0, 0, 0))
    arf.decode_index, arf.ref_decode_index = 1, (0,) * 7
    f1.decode_index, f1.ref_decode_index = 2, (1,) * 7
    ovl.decode_index, ovl.ref_decode_index = 3, (2, 1, 0, 0, 0, 0, 0)
    set_block(ovl, 0, 0, 16, 16, ref0=2, mv_px=(0.0, 0.5))  # overlay -> its own hidden frame
    frames = [key, arf, f1, ovl]
    lk = MotionLookup.from_frame(ovl)
    assert lk.query(np.array([[8.0, 8.0], [40.0, 8.0]]))[1].tolist() == [-1, 1]
    tr = build_tracks(frames, TrackParams(min_length=1))
    assert tr.stats["revisits_stopped"] > 0  # overlay -> frame 1 -> alt-ref stops at frame 1
    for t in range(tr.num_tracks):
        f = tr.frame[tr.track == t]
        assert len(f) == len(np.unique(f))
