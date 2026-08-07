"""shipd_jax: differentiable (JAX) implementation of the Ship-D hull pipeline.

Typical use::

    import numpy as np, jax
    from shipd_jax import (bits_from_vector, build_volume_plan, build_cw_plan,
                           gen_pointcloud, volume_properties, gen_pc_for_cw,
                           mod_michell, constraints)

    x_np = np.loadtxt('Input_Vectors_SampleHulls.csv', delimiter=',')[0]
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
"""
from .hull import (Bits, Hull, bits_from_vector, build_hull,
                   build_volume_plan, build_cw_plan,
                   gen_pointcloud, volume_properties, gen_pc_for_cw,
                   half_beam_midbody, bow_profile, stern_profile,
                   delta_bow, delta_stern)
from .constraints import constraints
from .michell import mod_michell, michspace, calc_cf, calc_drag, interp_cw
from .stl_export import gen_stl, gen_stl_trivec
from .universal import (uniform_volume_plan, gen_pc_for_cw_universal,
                        plan_signature, topology_safe_bounds,
                        check_box_topology)
from . import viz

__all__ = [
    "Bits", "Hull", "bits_from_vector", "build_hull",
    "build_volume_plan", "build_cw_plan",
    "gen_pointcloud", "volume_properties", "gen_pc_for_cw",
    "half_beam_midbody", "bow_profile", "stern_profile",
    "delta_bow", "delta_stern",
    "constraints", "mod_michell", "michspace", "calc_cf", "calc_drag",
    "interp_cw", "gen_stl", "gen_stl_trivec", "viz",
    "uniform_volume_plan", "gen_pc_for_cw_universal", "plan_signature",
    "topology_safe_bounds", "check_box_topology",
]
