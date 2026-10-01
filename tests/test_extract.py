"""Display order and reference resolution from AV1 order hints (no decoder needed)."""

from av1sfm.extract import KEY_FRAME, resolve_order

INTER = 1


def test_low_delay_stream_is_numbered_in_decode_order():
    ohs = [0, 1, 2, 3]
    refs = [[0] * 7, [0] * 7, [1, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0]]
    display, nodes = resolve_order(ohs, [KEY_FRAME, INTER, INTER, INTER], refs)
    assert display == [0, 1, 2, 3]
    assert nodes[3] == (2, 1, 0, 0, 0, 0, 0)
    assert nodes[0] == (-1,) * 7  # a key frame references nothing


def test_random_access_alt_ref_is_decoded_first_and_referenced_ahead():
    # libaom's default good mode (eval: 40 KITTI frames): the alt-ref with order
    # hint 6 is decoded right after the key frame, frames 1..5 then use it.
    ohs = [0, 6, 3, 1, 2]
    refs = [[0] * 7, [0] * 7, [0, 0, 0, 0, 0, 0, 6], [0, 0, 0, 0, 3, 0, 6], [1, 0, 0, 0, 3, 0, 6]]
    display, nodes = resolve_order(ohs, [KEY_FRAME] + [INTER] * 4, refs)
    assert display == [0, 4, 3, 1, 2]  # ranks of the display times 0, 1, 2, 3, 6
    assert nodes[4] == (3, 0, 0, 0, 2, 0, 1)  # order hints 1, 0, 3, 6 -> decoded frames


def test_overlay_shares_the_display_index_of_its_hidden_frame():
    # Hidden alt-ref (order hint 4) decoded second, shown later by an overlay frame.
    ohs = [0, 4, 1, 2, 3, 4, 5]
    refs = [[0] * 7] + [[o - 1 if o else 0, 0, 0, 0, 0, 0, 4] for o in ohs[1:]]
    display, nodes = resolve_order(ohs, [KEY_FRAME] + [INTER] * 6, refs)
    assert display == [0, 4, 1, 2, 3, 4, 5]
    assert nodes[5][6] == 1  # the overlay references the hidden frame itself
    assert nodes[6][0] == 5  # later frames get the most recent frame with that order hint


def test_key_frame_restarting_order_hints_continues_the_numbering():
    ohs = [0, 1, 2, 0, 1]
    refs = [[0] * 7, [0] * 7, [1] * 7, [0] * 7, [0] * 7]
    display, nodes = resolve_order(ohs, [KEY_FRAME, INTER, INTER, KEY_FRAME, INTER], refs)
    assert display == [0, 1, 2, 3, 4]
    assert nodes[4] == (3,) * 7  # the new key frame, not the first one


def test_order_hints_wrap_around():
    ohs = [(120 + k) % 128 for k in range(14)]
    refs = [[0] * 7] + [[(o - 1) % 128] * 7 for o in ohs[1:]]
    display, nodes = resolve_order(ohs, [KEY_FRAME] + [INTER] * 13, refs)
    assert display == list(range(14))
    assert all(n == (k - 1,) * 7 for k, n in enumerate(nodes) if k)
