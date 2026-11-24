"""mod_michell gradients stay finite where exp(Kz) nears the float64 overflow limit."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from shipd_jax import SAMPLE_HULLS, bits_from_vector, michell_grid, mod_michell  # noqa: E402


def _scaled(x, LOA, B, T):
    """Sample row ``x`` at LOA, B, T, chine and bilge kept in proportion to B."""
    x = jnp.asarray(x)
    Bd = B / LOA
    x = x.at[0].set(LOA).at[3].set(Bd).at[6].set(T / (x[4] * LOA))
    for i in (7, 10, 28, 30):
        x = x.at[i].set(Bd * x[i] / x[3])
    return x


def test_gradient_finite_near_exp_overflow():
    x0 = np.loadtxt(SAMPLE_HULLS, delimiter=",")[0]
    bits = bits_from_vector(x0)

    def wave_resistance(p):
        x = _scaled(x0, *p)
        X, Z, Y = michell_grid(x, x[6], bits, 21, 101)[:3]
        return mod_michell(Y, 12.346656, X, Z, 1025.0, N=61)

    # at this design an angle's Filon z-weight has exp(Kz) close to 1e308
    g = jax.jacfwd(wave_resistance)(jnp.array([247.95, 32.44, 10.19]))
    assert np.isfinite(np.asarray(g)).all(), g
