"""Differentiable (JAX) implementation of the Ship-D hull pipeline.

Typical use::

    import numpy as np, jax
    from shipd_jax import (SAMPLE_HULLS, bits_from_vector, build_volume_plan,
                           build_cw_plan, gen_pointcloud, volume_properties,
                           gen_pc_for_cw, mod_michell, constraints)

    x_np = np.loadtxt(SAMPLE_HULLS, delimiter=',')[0]
    bits = bits_from_vector(x_np)

    # constraints and their jacobian
    c = constraints(x_np, bits)
    Jc = jax.jacfwd(lambda x: constraints(x, bits))(x_np)

    # surface points and their sensitivities (topology fixed by the plan)
    plan = build_volume_plan(x_np, num_wl=101, ppw=1000)
    pc = gen_pointcloud(x_np, plan)
    dpc = jax.jacfwd(lambda x: gen_pointcloud(x, plan))(x_np)

    # wave resistance
    cw_plan = build_cw_plan(x_np, draft_frac=0.33, num_wl=51, ppw=301)
    def cw(x, Fn=0.3):
        X, Z, Y, WL = gen_pc_for_cw(x, cw_plan)
        U = Fn * jnp.sqrt(9.81 * WL)
        return mod_michell(Y, U, X, Z, rho=1000.0, N=21)
    g = jax.grad(cw)(x_np)

For an optimization loop, where the design vector moves and the frozen plan
above goes stale, use the plan-free ``michell_grid`` /
``grid_volume_properties`` instead (see ``grid.py`` and the
"Optimizer readiness" section of docs/shipd.md) -- one compiled function covers
every design (and, for ``michell_grid``, every draft) with no
recompilation, which is what the ship-mdo CSDL resistance model
(``ship_mdo.models.shipd_resistance``) uses.
"""

from importlib.resources import files as _files

from . import viz
from .constraints import constraints
from .hull import (
    Bits,
    Hull,
    bits_from_vector,
    bow_profile,
    build_cw_plan,
    build_hull,
    build_volume_plan,
    delta_bow,
    delta_stern,
    gen_pc_for_cw,
    gen_pointcloud,
    half_beam_midbody,
    stern_profile,
    volume_properties,
)
from .michell import calc_cf, calc_drag, interp_cw, michspace, mod_michell
from .stl_export import gen_stl, gen_stl_trivec
from .grid import (
    grid_volume_properties,
    michell_grid,
    offsets,
    pointcloud,
    section_counts,
    waterline_extent,
)

# The 12 sample hulls of the Ship-D paper (Table 1), one 45-entry design
# vector per row; shipped with the package so a non-editable install has it.
SAMPLE_HULLS = _files(__name__).joinpath("data", "Input_Vectors_SampleHulls.csv")

__all__ = [
    "SAMPLE_HULLS",
    "Bits",
    "Hull",
    "bits_from_vector",
    "bow_profile",
    "build_cw_plan",
    "build_hull",
    "build_volume_plan",
    "calc_cf",
    "calc_drag",
    "constraints",
    "delta_bow",
    "delta_stern",
    "gen_pc_for_cw",
    "gen_pointcloud",
    "gen_stl",
    "gen_stl_trivec",
    "half_beam_midbody",
    "interp_cw",
    "michspace",
    "mod_michell",
    "section_counts",
    "stern_profile",
    "grid_volume_properties",
    "michell_grid",
    "offsets",
    "pointcloud",
    "waterline_extent",
    "viz",
    "volume_properties",
]
