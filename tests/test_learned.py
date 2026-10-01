"""LightGlue's dual softmax written with logsumexp (av1sfm.learned) against kornia's."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lightglue = pytest.importorskip("kornia.feature.lightglue")

from av1sfm.learned import double_softmax_maxima, mutual_matches


def test_double_softmax_maxima_matches_kornia():
    g = torch.Generator().manual_seed(0)
    sim = 8 * torch.randn(1, 300, 200, generator=g)
    z0, z1 = torch.randn(1, 300, 1, generator=g), torch.randn(1, 200, 1, generator=g)
    full = lightglue.sigmoid_log_double_softmax(sim, z0, z1)
    scores = full[0, :-1, :-1]
    v0, i0, i1 = double_softmax_maxima(sim, z0, z1)
    np.testing.assert_allclose(v0, scores.max(1).values.numpy(), rtol=0, atol=1e-4)
    np.testing.assert_array_equal(i0, scores.max(1).indices.numpy())
    np.testing.assert_array_equal(i1, scores.max(0).indices.numpy())
    # Same mutual matches as kornia's filter_matches.
    m0 = lightglue.filter_matches(full, 0.1)[0][0].numpy()
    want = np.stack([np.flatnonzero(m0 > -1), m0[m0 > -1]], axis=1)
    assert len(want) > 20
    np.testing.assert_array_equal(mutual_matches(v0, i0, i1, 0.1), want)
