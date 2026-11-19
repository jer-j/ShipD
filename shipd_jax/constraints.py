#!/usr/bin/env python3
"""
Differentiable port of the 49 Ship-D feasibility constraints.

Order and semantics match ``Hull_Parameterization.input_Constraints()``:

* general hull form (2)
* cross section (7)
* bow form (15)
* stern form (12)
* bulb forms (13)

A hull is feasible when every entry is <= 0.
"""

from __future__ import annotations

import jax.numpy as jnp

from .hull import (
    Bits,
    Hull,
    _quad_vertex,
    bow_profile,
    build_hull,
    delta_bow,
    delta_stern,
    drift_angle,
    half_beam_midbody,
    stern_profile,
)


def _vertex_guarded(a, b, lo, hi, fallback):
    """Vertex of ``a z^2 + b z`` plus an in-range flag, with a safe stand-in
    value used to evaluate downstream expressions when out of range."""
    zv = _quad_vertex(a, b)
    cond = (zv >= lo) & (zv <= hi)
    zv_safe = jnp.where(cond, zv, fallback)
    return cond, zv_safe


def constraints_general(h: Hull):
    return [-h.LOA + h.Ls + h.Lb, h.WL - h.Dd]


def constraints_cross_section(h: Hull):
    return [
        -h.Rc_UG_int[1] + h.Dc,
        -h.Rc,
        -h.Bc,
        -h.Dc,
        h.Rc_LG_int[0] - h.Bc,
        h.Rk_LG_int[0] - h.Rc_LG_int[0],
        0.00000001 - jnp.abs(h.Rk),
    ]


def constraints_bow(h: Hull):
    half = h.Dd * 0.5  # safe stand-in z, strictly inside (0, Dd)

    cond, zv = _vertex_guarded(h.DRIFT[0], h.DRIFT[1], 0.0, h.Dd, half)
    vert_drift_0 = jnp.where(cond, drift_angle(h, zv) - jnp.pi / 2.0, -1.0)
    vert_drift_1 = jnp.where(cond, -drift_angle(h, zv), -1.0)

    cond, zv = _vertex_guarded(h.DELTA_BOW[0], h.DELTA_BOW[1], 0.0, h.Dd, half)
    vert_delta_bow = jnp.where(cond, -delta_bow(h, zv) + bow_profile(h, zv), -1.0)

    cond, zv = _vertex_guarded(h.BOW[0], h.BOW[1], 0.0, h.Dd, half)
    vert_bow = jnp.where(cond, -delta_bow(h, zv) + bow_profile(h, zv), -1.0)

    return [
        h.Kappa_BOW * h.Lb - delta_bow(h, 0.0),
        drift_angle(h, 0.0) - jnp.pi / 2.0,
        -drift_angle(h, 0.0),
        drift_angle(h, h.Dd) - jnp.pi / 2.0,
        -drift_angle(h, h.Dd),
        vert_drift_0,
        vert_drift_1,
        -h.BK[0],
        h.BK[0] - h.Kappa_BOW * h.Lb,
        -h.BK[1],
        h.BK[1] - h.Dd,
        -delta_bow(h, h.Dd) + bow_profile(h, h.Dd),
        -delta_bow(h, h.BK[1]) + h.BK[0],
        vert_delta_bow,
        vert_bow,
    ]


def constraints_stern(h: Hull, bits: Bits):
    half = h.Dd * 0.5

    cond, zv = _vertex_guarded(h.DELTA_STERN[0], h.DELTA_STERN[1], 0.0, h.Dd, half)
    vert_delta_stern = jnp.where(
        cond, delta_stern(h, zv) - stern_profile(h, bits, zv), -1.0
    )

    return [
        delta_stern(h, 0.0) - (h.Lb + h.Lm + h.Ls * h.Kappa_STERN),
        delta_stern(h, h.SK[1]) - h.SK[0],
        vert_delta_stern,
        delta_stern(h, h.Dd) - stern_profile(h, bits, h.Dd),
        (h.Lb + h.Lm + h.Ls * h.Kappa_STERN) - h.SK[0],
        h.Bc_trans - half_beam_midbody(h, h.Dc_trans),
        -h.Rc_UG_int_trans[1] + h.Dc_trans,
        -h.Rc_trans,
        -h.Bc_trans,
        -h.Dc_trans,
        h.Rc_LG_int_trans[0] - h.Bc_trans,
        h.Rk_LG_int_trans[0] - h.Rc_LG_int_trans[0],
    ]


def constraints_bulb(h: Hull, bits: Bits):
    minus1 = -jnp.ones(())
    plus1 = jnp.ones(())
    half = h.Dd * 0.5

    if bits.BB:
        c_beta0 = [minus1, minus1, h.BB_Prof[3] - half_beam_midbody(h, h.BB_Prof[2])]
        c_rk = [
            h.BB_Prof[2] - h.Rk,
            h.BB_Prof[3] - h.Rk,
            h.BB_Prof[3] - half_beam_midbody(h, h.BB_Prof[2]),
        ]
        c_else = [plus1, plus1, plus1]
        beta0 = h.Beta == 0.0
        rkpos = h.Rk > 0.0
        c03 = [
            jnp.where(beta0, c_beta0[i], jnp.where(rkpos, c_rk[i], c_else[i]))
            for i in range(3)
        ]

        cond, zv = _vertex_guarded(h.DELTA_BOW[0], h.DELTA_BOW[1], 0.0, h.WL, half)
        vert_delta_bow = jnp.where(cond, delta_bow(h, zv) - h.BB_Prof[5], -1.0)

        c36 = [
            h.BB_Prof[5] - delta_bow(h, 0.0),
            h.BB_Prof[5] - delta_bow(h, h.WL),
            vert_delta_bow,
        ]
    else:
        c03 = [minus1, minus1, minus1]
        c36 = [minus1, minus1, minus1]

    if bits.SB:
        c_beta0 = [
            minus1,
            minus1,
            h.SB_Prof[3] - half_beam_midbody(h, h.SB_Prof[2]),
            h.WL * h.HSBOA - h.SK[1],
        ]
        c_rk = [
            h.SB_Prof[2] - h.Rk,
            h.SB_Prof[3] - h.Rk,
            h.SB_Prof[3] - half_beam_midbody(h, h.SB_Prof[2]),
            h.WL * h.HSBOA - h.SK[1],
        ]
        c_else = [plus1, plus1, plus1, plus1]
        beta0 = h.Beta == 0.0
        rkpos = h.Rk > 0.0
        c610 = [
            jnp.where(beta0, c_beta0[i], jnp.where(rkpos, c_rk[i], c_else[i]))
            for i in range(4)
        ]

        cond, zv = _vertex_guarded(
            h.DELTA_STERN[0], h.DELTA_STERN[1], 0.0, h.WL * h.HSBOA, half
        )
        vert_delta_stern = jnp.where(cond, delta_stern(h, zv) - h.SB_Prof[5], -1.0)

        c1013 = [
            delta_stern(h, 0.0) - h.SB_Prof[5],
            delta_stern(h, h.WL * h.HSBOA) - h.SB_Prof[5],
            vert_delta_stern,
        ]
    else:
        c610 = [minus1, minus1, minus1, minus1]
        c1013 = [minus1, minus1, minus1]

    return c03 + c36 + c610 + c1013


def constraints(x, bits: Bits) -> jnp.ndarray:
    """All 49 feasibility constraints (feasible when every entry <= 0)."""
    h = build_hull(x, bits)
    cs = (
        constraints_general(h)
        + constraints_cross_section(h)
        + constraints_bow(h)
        + constraints_stern(h, bits)
        + constraints_bulb(h, bits)
    )
    return jnp.stack([jnp.asarray(c, dtype=jnp.float64) for c in cs])
