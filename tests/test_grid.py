"""Plan-free grid: michell_grid unchanged by the refactor, offsets consistent."""

from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from shipd_jax import SAMPLE_HULLS, bits_from_vector, michell_grid, offsets  # noqa: E402

DATA = Path(__file__).parent / "data" / "michell_grid_before.npz"


def _baseline():
    x = np.loadtxt(SAMPLE_HULLS, delimiter=",")[11]
    return x, bits_from_vector(x), 6.15 / (x[4] * 142.0)


def test_michell_grid_unchanged():
    ref = np.load(DATA)
    x, bits, df = _baseline()
    X, Z, Y, WL = michell_grid(x, df, bits, 21, 101)
    np.testing.assert_allclose(np.asarray(X), ref["X"], atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(Z), ref["Z"], atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(Y), ref["Y"], atol=0, rtol=0)
    assert float(WL) == float(ref["WL"])


def test_offsets_match_michell_grid():
    x, bits, df = _baseline()
    X, Z, Y, _ = michell_grid(x, df, bits, 21, 101)
    # recover the absolute ShipD-frame stations and heights
    from shipd_jax import build_hull

    h = build_hull(x, bits)
    Z_abs = np.asarray(Z) + float(h.Dd) * df
    from shipd_jax.grid import waterline_extent

    xb, _ = waterline_extent(x, bits, Z_abs)
    X_abs = np.asarray(X) + float(np.min(xb))
    np.testing.assert_allclose(
        np.asarray(offsets(x, bits, X_abs, Z_abs)), np.asarray(Y), atol=1e-12
    )
