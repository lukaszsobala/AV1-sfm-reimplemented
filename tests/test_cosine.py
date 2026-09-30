import numpy as np
import pytest

from av1sfm.tracks import cosine_ok


@pytest.mark.parametrize(
    ("v1", "v2", "expected"),
    [
        ((4, 0), (8, 0), True),  # parallel, different magnitude
        ((4, 0), (4, 1), True),  # 14 deg: cos 0.970 >= 0.9
        ((4, 0), (4, 2.5), False),  # 32 deg: cos 0.848 < 0.9
        ((4, 0), (0, 4), False),  # orthogonal
        ((4, 0), (-4, 0), False),  # opposite
        ((0.5, 0), (-4, 0), True),  # |v1| < tau: not testable
        ((4, 0), (0, -0.9), True),  # |v2| < tau
        ((np.nan, np.nan), (-4, 0), True),  # no previous step
    ],
)
def test_cosine_cases(v1, v2, expected):
    assert cosine_ok(np.array([v1]), np.array([v2]), eps=0.1, tau=1.0)[0] == expected


def test_threshold_is_inclusive_at_one_minus_eps():
    ang = np.arccos(0.9)
    v2 = np.array([[np.cos(ang) * 10, np.sin(ang) * 10]])
    assert cosine_ok(np.array([[10.0, 0.0]]), v2 * (1 - 1e-9) + [[1e-7, 0]], 0.1, 1.0)[0]
    assert not cosine_ok(
        np.array([[10.0, 0.0]]), np.array([[np.cos(ang + 1e-3), np.sin(ang + 1e-3)]]) * 10, 0.1, 1.0
    )[0]


def test_eps_one_disables_filter():
    v1 = np.array([[4.0, 0.0], [4.0, 0.0]])
    v2 = np.array([[-4.0, 0.0], [0.0, 4.0]])
    assert cosine_ok(v1, v2, eps=1.0, tau=1.0).all()
    assert not cosine_ok(v1, v2, eps=0.99, tau=1.0).any()


def test_vectorised_shapes():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(100, 2)) * 5
    out = cosine_ok(v, v * 2.0, eps=0.1, tau=1.0)
    assert out.shape == (100,) and out.all()
