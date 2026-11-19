#!/usr/bin/env python3
"""
Differentiable (JAX) re-implementation of the Ship-D hull parameterization.

This module reproduces, function by function, the geometry defined in the
original ``HullParameterization.py`` by Noah Bagazinski, restructured as pure
functions on JAX arrays so that every output (surface points, constraints,
volumetric properties, wave resistance) is differentiable with respect to the
45-entry design vector.

Design
------
The original code makes a number of *structural* decisions from the concrete
values of the design vector: how many sample points each waterline section
receives, which root of a cubic is the bulb/hull intersection, whether the
stern waterline is an ellipse or a parabola, etc.  Those decisions are
piecewise-constant in the design variables, so they are computed once with
concrete numpy values in a "plan" (see :func:`build_volume_plan` /
:func:`build_cw_plan`) and treated as static.  All *coordinates* are then
recomputed inside JAX from the traced design vector, so:

* with the plan built from the same vector, forward evaluation matches the
  original implementation to floating-point accuracy, and
* gradients are the derivatives of the geometry holding the (discrete) mesh
  topology fixed, which is the standard interpretation for shape sensitivities.

Known quirks of the reference implementation are reproduced intentionally:

* ``GenBowForm`` checks the bow-rise vertex against ``BK[1]*Dd`` even though
  ``BK[1]`` is already dimensional (double scaling),
* the fillet quartic systems use ``6*x**2`` in the second-derivative row where
  ``6*x`` was presumably intended,
* ``Calc_WettedSurface`` contains a no-op statement instead of adding the
  transom width to the arc length (so it is *not* added here either).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

Array = jax.Array

_NEWTON_ITERS = 8


# Static bits (structural switches carried in the design vector)


class Bits(NamedTuple):
    """The four 0/1 entries of the design vector that select structure."""

    EP_S: bool  # x[20]: stern shape below SK intersect is ellipse (1) / parabola (0)
    EP_T: bool  # x[21]: stern shape above SK intersect is ellipse (1) / parabola (0)
    BB: bool  # x[31]: bulbous bow present
    SB: bool  # x[32]: bulbous stern present


def bits_from_vector(x) -> Bits:
    """Extract the structural bits from a concrete design vector.

    Matches the truthiness semantics of the reference code (``if self.bit_BB:``).
    """
    x = np.asarray(x, dtype=np.float64)
    return Bits(
        bool(x[20] != 0.0), bool(x[21] != 0.0), bool(x[31] != 0.0), bool(x[32] != 0.0)
    )


# Derived hull state (everything the reference computes in __init__/Gen*)


class Hull(NamedTuple):
    LOA: Array
    Lb: Array
    Ls: Array
    Bd: Array  # half breadth at deck
    Dd: Array
    Bs: Array  # half breadth at stern deck
    WL: Array
    Bc: Array  # half breadth at chine
    Beta: Array
    Rc: Array
    Rk: Array
    BOW: Array  # (3,)
    BK: Array  # (2,) [x, z]
    Kappa_BOW: Array
    DELTA_BOW: Array  # (3,)
    DRIFT: Array  # (3,)
    TRANS: Array  # (2,)
    SK: Array  # (2,) [x, z]
    Kappa_STERN: Array
    DELTA_STERN: Array  # (3,)
    Beta_trans: Array
    Bc_trans: Array
    Rc_trans: Array
    Rk_trans: Array
    Lbb: Array
    Hbb: Array
    Bbb: Array
    Lbbm: Array
    Rbb: Array
    Kappa_SB: Array
    Lsb: Array
    HSBOA: Array
    Hsb: Array
    Bsb: Array
    Lsbm: Array
    Rsb: Array
    Lm: Array
    # midbody cross section
    Rk_Center: Array  # (2,)
    Rk_LG_int: Array  # (2,)
    LG: Array  # (3,)
    Dc: Array
    UG: Array  # (3,)
    Rc_UG_int: Array  # (2,)
    Rc_LG_int: Array  # (2,)
    Rc_Center: Array  # (2,)
    # bow
    KEEL_BOW: Array
    # stern
    STERNRISE: Array
    Rk_Center_trans: Array
    Rk_LG_int_trans: Array
    LG_trans: Array
    Dc_trans: Array
    UG_trans: Array
    Rc_UG_int_trans: Array
    Rc_LG_int_trans: Array
    Rc_Center_trans: Array
    # bulbs
    BB_Prof: Array  # (7,)
    SB_Prof: Array  # (7,)


def _safe(den: Array) -> Array:
    """Replace exact zeros in a denominator so the division never NaNs.

    Only used where the reference either cannot reach the zero (feasible
    hulls) or explicitly branches away from it, so values are unchanged.
    """
    return jnp.where(den == 0.0, 1.0, den)


def _quad_vertex(a: Array, b: Array) -> Array:
    """z-position of the vertex of ``a z^2 + b z``; -1.0 when ``a == 0``."""
    return jnp.where(a == 0.0, -1.0, -b / (2.0 * _safe(a)))


def _sqrt_abs(w: Array) -> Array:
    """``sqrt(abs(w))`` with a finite (zero) gradient at ``w == 0``.

    The bulb/stern curves evaluate ``sqrt(abs(1-u^2))`` at section endpoints
    where the argument is exactly zero (the value there is either masked out
    or overwritten).  The primal matches ``jnp.sqrt(jnp.abs(w))`` everywhere;
    the double-where keeps the infinite square-root slope out of reverse-mode
    accumulation (0 * inf = NaN otherwise).
    """
    a = jnp.abs(w)
    pos = a > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, a, 1.0)), 0.0)


def _linspace(a: Array, b: Array, n: int) -> Array:
    """np.linspace clone (endpoint set exactly) that is traceable in JAX."""
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    if n <= 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    if n == 1:
        return a[None]
    t = jnp.arange(n, dtype=jnp.float64) * ((b - a) / (n - 1)) + a
    return t.at[-1].set(b)


def _cross_section(Rk, Beta_deg, z_keel, D_top, B_top, Bc, Rc):
    """Shared solver for the midbody / transom cross sections.

    Returns (Rk_Center, Rk_LG_int, LG, Dc, UG, Rc_UG_int, Rc_LG_int, Rc_Center)
    exactly as ``GenCrossSection`` / ``GenSternForm`` compute them, with the
    keel-radius center offset ``z_keel`` (0 for midbody, SK_z for the transom).
    """
    Rk_Center = jnp.stack(
        [-Rk * (0.5 - 0.5 * jnp.sign(Rk)), z_keel + Rk * (0.5 + 0.5 * jnp.sign(Rk))]
    )
    Rk_LG_int = jnp.stack(
        [
            Rk_Center[0] + Rk * jnp.sin(jnp.pi * Beta_deg / 180.0),
            Rk_Center[1] - Rk * jnp.cos(jnp.pi * Beta_deg / 180.0),
        ]
    )

    A = jnp.stack(
        [
            jnp.stack([jnp.ones(()), jnp.ones(()), jnp.ones(())]),
            jnp.stack([Rk_LG_int[1], Rk_LG_int[0], jnp.ones(())]),
            jnp.stack(
                [
                    -(Rk_LG_int[0] - Rk_Center[0]),
                    (Rk_LG_int[1] - Rk_Center[1]),
                    jnp.zeros(()),
                ]
            ),
        ]
    )
    b = jnp.array([1.0, 0.0, 0.0])
    LG = jnp.linalg.solve(A, b)

    Dc = -(LG[1] * Bc + LG[2]) / LG[0]

    A = jnp.stack(
        [
            jnp.stack([Dc, Bc, jnp.ones(())]),
            jnp.stack([D_top, B_top, jnp.ones(())]),
            jnp.stack([jnp.ones(()), jnp.ones(()), jnp.ones(())]),
        ]
    )
    b = jnp.array([0.0, 0.0, 1.0])
    UG = jnp.linalg.solve(A, b)

    A1 = UG[0]
    B1 = UG[1]
    theta = jnp.arctan2(-B1, A1)
    theta = jnp.where(theta < 0.0, theta + jnp.pi, theta)
    beta = Beta_deg * jnp.pi / 180.0
    A2 = LG[0]
    B2 = LG[1]

    zero = jnp.zeros(())
    one = jnp.ones(())
    A6 = jnp.stack(
        [
            jnp.stack([B1, A1, zero, zero, zero, zero]),
            jnp.stack([zero, zero, B2, A2, zero, zero]),
            jnp.stack([one, zero, zero, zero, -one, zero]),
            jnp.stack([zero, -one, zero, zero, zero, one]),
            jnp.stack([zero, zero, one, zero, -one, zero]),
            jnp.stack([zero, zero, zero, -one, zero, one]),
        ]
    )
    b6 = jnp.stack(
        [
            -UG[2],
            -LG[2],
            Rc * jnp.sin(theta),
            Rc * jnp.cos(theta),
            Rc * jnp.sin(beta),
            Rc * jnp.cos(beta),
        ]
    )
    C = jnp.linalg.solve(A6, b6)

    return Rk_Center, Rk_LG_int, LG, Dc, UG, C[0:2], C[2:4], C[4:6]


def build_hull(x, bits: Bits) -> Hull:
    """Compute every derived quantity of the parameterization (traced)."""
    x = jnp.asarray(x, dtype=jnp.float64)

    LOA = x[0]
    Lb = x[1] * LOA
    Ls = x[2] * LOA
    Bd = x[3] / 2.0 * LOA
    Dd = x[4] * LOA
    Bs = x[5] * Bd
    WL = x[6] * Dd
    Bc = x[7] / 2.0 * LOA
    Beta = x[8]
    Rc = x[9] * Bc
    Rk = x[10] * Dd

    BOW_a = x[11] * 0.5 * Lb / Dd**2.0
    BOW_b = x[12] * 0.5 * Lb / Dd
    BK_z = x[13] * Dd
    Kappa_BOW = x[14]
    DELTA_BOW_a = x[15] * 0.5 * Lb / Dd**2.0
    DELTA_BOW_b = x[16] * 0.5 * Lb / Dd
    DRIFT = jnp.stack([x[17] * 60.0 / Dd**2.0, x[18] * 60.0 / Dd, x[19]])

    TRANS_a = x[22]
    SK_frac = x[23]
    Kappa_STERN = x[24]
    DELTA_STERN_a = x[25] * 0.5 * Ls / Dd**2.0
    DELTA_STERN_b = x[26] * 0.5 * Ls / Dd
    Beta_trans = x[27]
    Bc_trans = x[28] / 2.0 * LOA
    Rc_trans = x[29] * Bc_trans
    Rk_trans = x[30] * Dd * (1.0 - SK_frac)

    Lbb = x[33]
    Hbb = x[34]
    Bbb = x[35]
    Lbbm = x[36]
    Rbb = x[37]
    Kappa_SB = x[38]
    Lsb = x[39]
    HSBOA = x[40]
    Hsb = x[41]
    Bsb = x[42]
    Lsbm = x[43]
    Rsb = x[44]

    Lm = LOA - Ls - Lb

    # midbody cross section
    (Rk_Center, Rk_LG_int, LG, Dc, UG, Rc_UG_int, Rc_LG_int, Rc_Center) = (
        _cross_section(Rk, Beta, jnp.zeros(()), Dd, Bd, Bc, Rc)
    )

    # bow form
    Zv = _quad_vertex(BOW_a, BOW_b)
    C0 = BOW_a * Dd**2.0 + BOW_b * Dd
    C1 = BOW_a * BK_z**2.0 + BOW_b * BK_z
    C2 = BOW_a * Zv**2.0 + BOW_b * Zv
    # NOTE: reference compares against BK[1]*Dd with BK[1] already dimensional.
    in_range = (Zv >= BK_z * Dd) & (Zv <= Dd)
    BOW_c = -jnp.where(
        in_range, jnp.minimum(jnp.minimum(C0, C1), C2), jnp.minimum(C0, C1)
    )
    BOW = jnp.stack([BOW_a, BOW_b, BOW_c])

    BK_x = BOW_a * BK_z**2.0 + BOW_b * BK_z + BOW_c
    BK = jnp.stack([BK_x, BK_z])

    KEEL_BOW = BK_z / ((BK_x - Kappa_BOW * Lb) ** 2.0)

    Zv = _quad_vertex(DELTA_BOW_a, DELTA_BOW_b)
    C0 = DELTA_BOW_a * Dd**2.0 + DELTA_BOW_b * Dd
    C2 = DELTA_BOW_a * Zv**2.0 + DELTA_BOW_b * Zv
    in_range = (Zv >= 0.0) & (Zv <= Dd)
    DELTA_BOW_c = -jnp.where(
        in_range, jnp.maximum(jnp.maximum(C0, 0.0), C2), jnp.maximum(C0, 0.0)
    )
    DELTA_BOW = jnp.stack([DELTA_BOW_a, DELTA_BOW_b, DELTA_BOW_c])

    # stern form
    SK_z = SK_frac * Dd
    TRANS_b = jnp.where(TRANS_a >= 0.0, LOA - TRANS_a * Dd, LOA - TRANS_a * SK_z)
    TRANS = jnp.stack([TRANS_a, TRANS_b])
    SK_x = TRANS_a * SK_z + TRANS_b
    SK = jnp.stack([SK_x, SK_z])

    STERNRISE = SK_z / (SK_x - (Lb + Lm + Ls * Kappa_STERN)) ** 2.0

    Zv = _quad_vertex(DELTA_STERN_a, DELTA_STERN_b)
    C0 = DELTA_STERN_a * Dd**2.0 + DELTA_STERN_b * Dd
    C2 = DELTA_STERN_a * Zv**2.0 + DELTA_STERN_b * Zv
    in_range = (Zv >= 0.0) & (Zv <= Dd)
    DELTA_STERN_c = -jnp.where(
        in_range, jnp.minimum(jnp.minimum(C0, 0.0), C2), jnp.minimum(C0, 0.0)
    )
    DELTA_STERN = jnp.stack([DELTA_STERN_a, DELTA_STERN_b, DELTA_STERN_c])

    (
        Rk_Center_trans,
        Rk_LG_int_trans,
        LG_trans,
        Dc_trans,
        UG_trans,
        Rc_UG_int_trans,
        Rc_LG_int_trans,
        Rc_Center_trans,
    ) = _cross_section(Rk_trans, Beta_trans, SK_z, Dd, Bs, Bc_trans, Rc_trans)

    hull = Hull(
        LOA=LOA,
        Lb=Lb,
        Ls=Ls,
        Bd=Bd,
        Dd=Dd,
        Bs=Bs,
        WL=WL,
        Bc=Bc,
        Beta=Beta,
        Rc=Rc,
        Rk=Rk,
        BOW=BOW,
        BK=BK,
        Kappa_BOW=Kappa_BOW,
        DELTA_BOW=DELTA_BOW,
        DRIFT=DRIFT,
        TRANS=TRANS,
        SK=SK,
        Kappa_STERN=Kappa_STERN,
        DELTA_STERN=DELTA_STERN,
        Beta_trans=Beta_trans,
        Bc_trans=Bc_trans,
        Rc_trans=Rc_trans,
        Rk_trans=Rk_trans,
        Lbb=Lbb,
        Hbb=Hbb,
        Bbb=Bbb,
        Lbbm=Lbbm,
        Rbb=Rbb,
        Kappa_SB=Kappa_SB,
        Lsb=Lsb,
        HSBOA=HSBOA,
        Hsb=Hsb,
        Bsb=Bsb,
        Lsbm=Lsbm,
        Rsb=Rsb,
        Lm=Lm,
        Rk_Center=Rk_Center,
        Rk_LG_int=Rk_LG_int,
        LG=LG,
        Dc=Dc,
        UG=UG,
        Rc_UG_int=Rc_UG_int,
        Rc_LG_int=Rc_LG_int,
        Rc_Center=Rc_Center,
        KEEL_BOW=KEEL_BOW,
        STERNRISE=STERNRISE,
        Rk_Center_trans=Rk_Center_trans,
        Rk_LG_int_trans=Rk_LG_int_trans,
        LG_trans=LG_trans,
        Dc_trans=Dc_trans,
        UG_trans=UG_trans,
        Rc_UG_int_trans=Rc_UG_int_trans,
        Rc_LG_int_trans=Rc_LG_int_trans,
        Rc_Center_trans=Rc_Center_trans,
        BB_Prof=jnp.zeros((7,)),
        SB_Prof=jnp.zeros((7,)),
    )

    # bulb forms (need bow/midbody functions)
    if bits.BB:
        FP = bow_profile(hull, WL)
        BB_Prof = jnp.stack(
            [
                FP,
                (1.0 - Hbb) * WL,
                Hbb * WL,
                half_beam_midbody(hull, Hbb * WL) * Bbb,
                Lbb * LOA * (1.0 - Lbbm),
                FP - LOA * Lbb * Lbbm,
                jnp.zeros(()),
            ]
        )
        hull = hull._replace(BB_Prof=BB_Prof)

    if bits.SB:
        SBs = Kappa_SB * Ls + Lm + Lb
        SB_Prof = jnp.stack(
            [
                SBs,
                (1.0 - Hsb) * WL * HSBOA,
                Hsb * WL * HSBOA,
                half_beam_midbody(hull, Hsb * WL * HSBOA) * Bsb,
                Lsb * LOA * (1.0 - Lsbm),
                SBs + LOA * Lsb * Lsbm,
                jnp.zeros(()),
            ]
        )
        hull = hull._replace(SB_Prof=SB_Prof)

    return hull


# Sectional geometry (broadcastable over z / x)


def half_beam_midbody(h: Hull, z) -> Array:
    """Half beam of the parallel midbody cross section at height ``z``."""
    z = jnp.asarray(z, dtype=jnp.float64)
    c_keel = z < h.Rk_LG_int[1]
    c_lg = z < h.Rc_LG_int[1]
    c_chine = z < h.Rc_UG_int[1]

    eff_keel = c_keel
    eff_chine = (~c_keel) & (~c_lg) & c_chine

    keel_arg = (h.Rk**2) - (z - h.Rk_Center[1]) ** 2
    y_keel = (
        jnp.sign(h.Rk) * jnp.sqrt(jnp.where(eff_keel, keel_arg, 1.0)) + h.Rk_Center[0]
    )
    y_lg = -(h.LG[0] * z + h.LG[2]) / h.LG[1]
    chine_arg = (h.Rc**2) - (z - h.Rc_Center[1]) ** 2
    y_chine = jnp.sqrt(jnp.where(eff_chine, chine_arg, 1.0)) + h.Rc_Center[0]
    y_ug = -(h.UG[0] * z + h.UG[2]) / h.UG[1]

    y = jnp.where(
        c_keel, y_keel, jnp.where(c_lg, y_lg, jnp.where(c_chine, y_chine, y_ug))
    )
    return jnp.where((z < 0.0) | (z > h.Dd), -1.0, y)


def bowrise(h: Hull, z) -> Array:
    return h.BOW[0] * z**2.0 + h.BOW[1] * z + h.BOW[2]


def keelrise_bow(h: Hull, z) -> Array:
    # callers guarantee z >= 0
    return (
        -jnp.sqrt(jnp.asarray(z, dtype=jnp.float64) / h.KEEL_BOW) + h.Kappa_BOW * h.Lb
    )


def delta_bow(h: Hull, z) -> Array:
    return h.Lb + h.DELTA_BOW[0] * z**2.0 + h.DELTA_BOW[1] * z + h.DELTA_BOW[2]


def drift_angle(h: Hull, z) -> Array:
    return jnp.pi * (h.DRIFT[0] * z**2.0 + h.DRIFT[1] * z + h.DRIFT[2]) / 180.0


def bow_profile(h: Hull, z) -> Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    return jnp.where(z <= h.BK[1], keelrise_bow(h, z), bowrise(h, z))


def solve_waterline_bow(h: Hull, z) -> Array:
    """Cubic ``y(x)`` between the bow profile and delta_bow at height ``z``."""
    X1 = bow_profile(h, z)
    X2 = delta_bow(h, z)
    Y2 = half_beam_midbody(h, z)
    zero = jnp.zeros(())
    one = jnp.ones(())
    A = jnp.stack(
        [
            jnp.stack([X1**3.0, X1**2.0, X1, one]),
            jnp.stack([3.0 * X1**2.0, 2.0 * X1, one, zero]),
            jnp.stack([X2**3.0, X2**2.0, X2, one]),
            jnp.stack([3.0 * X2**2.0, 2.0 * X2, one, zero]),
        ]
    )
    b = jnp.stack([zero, jnp.tan(drift_angle(h, z)), Y2, zero])
    return jnp.linalg.solve(A, b)


def _cubic(prof: Array, x) -> Array:
    return prof[0] * x**3.0 + prof[1] * x**2.0 + prof[2] * x + prof[3]


def _cubic_dydx(prof: Array, x) -> Array:
    return 3.0 * prof[0] * x**2.0 + 2.0 * prof[1] * x + prof[2]


def _quartic(pr: Array, x) -> Array:
    return pr[0] * x**4.0 + pr[1] * x**3.0 + pr[2] * x**2.0 + pr[3] * x + pr[4]


def transom_x(h: Hull, z) -> Array:
    return h.TRANS[0] * z + h.TRANS[1]


def sternrise(h: Hull, z) -> Array:
    return (
        jnp.sqrt(jnp.asarray(z, dtype=jnp.float64) / h.STERNRISE)
        + h.Lb
        + h.Lm
        + h.Ls * h.Kappa_STERN
    )


def stern_profile(h: Hull, bits: Bits, z) -> Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    base = jnp.where(z <= h.SK[1], sternrise(h, z), transom_x(h, z))
    if bits.SB:
        return jnp.where(z <= h.WL * h.HSBOA, h.SB_Prof[0], base)
    return base


def delta_stern(h: Hull, z) -> Array:
    return (
        h.Lb
        + h.Lm
        + h.DELTA_STERN[0] * z**2.0
        + h.DELTA_STERN[1] * z
        + h.DELTA_STERN[2]
    )


def half_beam_transom(h: Hull, bits: Bits, z) -> tuple[Array, Array]:
    """(x, y) of the transom edge at height ``z`` (priority order as reference)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    x = stern_profile(h, bits, z)

    c_zero = (z <= h.SK[1]) & (z > 0.0)
    c_err = (z < 0.0) | (z > h.Dd)
    c_keel = (z > h.SK[1]) & (z < h.Rk_LG_int_trans[1])
    c_lg = (z >= h.Rk_LG_int_trans[1]) & (z < h.Rc_LG_int_trans[1])
    c_chine = (z >= h.Rc_LG_int_trans[1]) & (z < h.Rc_UG_int_trans[1])

    eff_keel = (~c_zero) & (~c_err) & c_keel
    eff_chine = (~c_zero) & (~c_err) & (~c_keel) & (~c_lg) & c_chine

    keel_arg = (h.Rk_trans**2) - (z - h.Rk_Center_trans[1]) ** 2
    y_keel = (
        jnp.sign(h.Rk_trans) * jnp.sqrt(jnp.where(eff_keel, keel_arg, 1.0))
        + h.Rk_Center_trans[0]
    )
    y_lg = -(h.LG_trans[0] * z + h.LG_trans[2]) / h.LG_trans[1]
    chine_arg = (h.Rc_trans**2) - (z - h.Rc_Center_trans[1]) ** 2
    y_chine = jnp.sqrt(jnp.where(eff_chine, chine_arg, 1.0)) + h.Rc_Center_trans[0]
    y_ug = -(h.UG_trans[0] * z + h.UG_trans[2]) / h.UG_trans[1]

    y = jnp.where(
        c_zero,
        0.0,
        jnp.where(
            c_err,
            -1.0,
            jnp.where(
                c_keel, y_keel, jnp.where(c_lg, y_lg, jnp.where(c_chine, y_chine, y_ug))
            ),
        ),
    )
    return x, y


def solve_waterline_stern(h: Hull, bits: Bits, z) -> tuple[Array, Array]:
    """Return (parabola (3,), ellipse (4,)) stern waterline curves at ``z``.

    The caller selects which family applies (plan-static per waterline);
    the ellipse is ``[Rx, Ry, Cx, Cy]`` as in the reference PROF[4:8].
    """
    x1 = delta_stern(h, z)
    y1 = half_beam_midbody(h, z)
    x2, y2 = half_beam_transom(h, bits, z)

    zero = jnp.zeros(())
    one = jnp.ones(())
    A = jnp.stack(
        [
            jnp.stack([x1**2.0, x1, one]),
            jnp.stack([x2**2.0, x2, one]),
            jnp.stack([2.0 * x1, one, zero]),
        ]
    )
    b = jnp.stack([y1, y2, zero])
    parab = jnp.linalg.solve(A, b)

    ell = jnp.stack([x2 - x1, y1 - y2, x1, y2])
    return parab, ell


def _stern_curve(parab: Array, ell: Array, is_ellipse: bool, x) -> Array:
    if is_ellipse:
        return (
            _sqrt_abs(ell[1] ** 2.0 * (1.0 - ((x - ell[2]) / ell[0]) ** 2.0)) + ell[3]
        )
    return parab[0] * x**2.0 + parab[1] * x + parab[2]


def _stern_curve_dydx(parab: Array, ell: Array, is_ellipse: bool, x) -> Array:
    if is_ellipse:
        return (
            -ell[1]
            * (x - ell[2])
            / (ell[0] ** 2.0)
            * 1.0
            / jnp.sqrt(jnp.abs(1.0 - ((x - ell[2]) / ell[0]) ** 2.0))
        )
    return 2.0 * parab[0] * x + parab[1]


# Bulb geometry


def bb_profile(h: Hull, z) -> Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    Rz = jnp.where(z >= h.BB_Prof[2], h.BB_Prof[1], h.BB_Prof[2])
    return (
        h.BB_Prof[5]
        - _sqrt_abs(1.0 - ((z - h.Hbb * h.WL) / _safe(Rz)) ** 2.0) * h.BB_Prof[4]
    )


def half_beam_bb(h: Hull, z, x) -> Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    Rz = jnp.where(z >= h.BB_Prof[2], h.BB_Prof[1], h.BB_Prof[2])
    Ry = _sqrt_abs(1.0 - ((z - h.BB_Prof[2]) / _safe(Rz)) ** 2.0) * h.BB_Prof[3]
    Rx = h.BB_Prof[5] - bb_profile(h, z)
    return jnp.where(
        x >= h.BB_Prof[5],
        Ry,
        _sqrt_abs(1.0 - ((x - h.BB_Prof[5]) / _safe(Rx)) ** 2.0) * Ry,
    )


def bb_dydx(h: Hull, z, x) -> Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    Rz = jnp.where(z >= h.BB_Prof[2], h.BB_Prof[1], h.BB_Prof[2])
    Ry = _sqrt_abs(1.0 - ((z - h.BB_Prof[2]) / _safe(Rz)) ** 2.0) * h.BB_Prof[3]
    Rx = h.BB_Prof[5] - bb_profile(h, z)
    inner = _sqrt_abs(1.0 - ((x - h.BB_Prof[5]) / _safe(Rx)) ** 2.0)
    return jnp.where(
        x >= h.BB_Prof[5],
        0.0,
        -Ry * (x - h.BB_Prof[5]) / (Rx**2.0) * 1.0 / _safe(inner),
    )


def sb_profile(h: Hull, z) -> Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    zc = h.Hsb * h.WL * h.HSBOA
    Rz = jnp.where(z >= zc, h.SB_Prof[1], h.SB_Prof[2])
    return (
        h.SB_Prof[5]
        + jnp.sqrt(jnp.abs(1.0 - ((z - zc) / _safe(Rz)) ** 2.0)) * h.SB_Prof[4]
    )


def half_beam_sb(h: Hull, z, x) -> Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    zc = h.Hsb * h.WL * h.HSBOA
    Rz = jnp.where(z >= zc, h.SB_Prof[1], h.SB_Prof[2])
    Ry = jnp.sqrt(jnp.abs(1.0 - ((z - zc) / _safe(Rz)) ** 2.0)) * h.SB_Prof[3]
    Rx = sb_profile(h, z) - h.SB_Prof[5]
    return jnp.where(
        x <= h.SB_Prof[5],
        Ry,
        jnp.sqrt(jnp.abs(1.0 - ((x - h.SB_Prof[5]) / _safe(Rx)) ** 2.0)) * Ry,
    )


def sb_dydx(h: Hull, z, x) -> Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    zc = h.Hsb * h.WL * h.HSBOA
    Rz = jnp.where(z >= zc, h.SB_Prof[1], h.SB_Prof[2])
    Ry = jnp.sqrt(jnp.abs(1.0 - ((z - zc) / _safe(Rz)) ** 2.0)) * h.SB_Prof[3]
    Rx = sb_profile(h, z) - h.SB_Prof[5]
    inner = jnp.sqrt(jnp.abs(1.0 - ((x - h.SB_Prof[5]) / _safe(Rx)) ** 2.0))
    return jnp.where(
        x <= h.SB_Prof[5],
        0.0,
        -Ry * (x - h.SB_Prof[5]) / (Rx**2.0) * 1.0 / _safe(inner),
    )


# Fillet solves (bulb <-> hull blending), Newton-refined from static inits


def _newton_cubic(c0, c1, c2, c3, x0: float, iters: int = _NEWTON_ITERS) -> Array:
    """Refine a (statically initialized) root of ``c0 x^3 + c1 x^2 + c2 x + c3``."""
    x = jnp.asarray(x0, dtype=jnp.float64)
    for _ in range(iters):
        f = c0 * x**3.0 + c1 * x**2.0 + c2 * x + c3
        fp = 3.0 * c0 * x**2.0 + 2.0 * c1 * x + c2
        x = x - f / _safe(fp)
    return x


# Modes for the bow-fillet intersection "root" (see _bb_root_init):
_ROOT_REAL = 0  # a true real root of the cubic
_ROOT_CPAIR = 1  # real part of a complex pair: (-c1/c0 - r)/2 via Vieta
_ROOT_CPAIR2 = 2  # degenerate (quadratic) complex pair: -c2/(2*c1)


def _bb_fillet(h: Hull, z, x0):
    """Bow-bulb fillet at height ``z``: returns (prof, Xr0, Xr1, profrad).

    ``x0`` is the plan tuple (mode, init) for the intersection root.
    """
    mode, init = x0
    prof = solve_waterline_bow(h, z)
    x1 = bb_profile(h, z)
    x2 = delta_bow(h, z)

    ybbC = half_beam_bb(h, z, h.BB_Prof[5])
    c0, c1, c2 = prof[0], prof[1], prof[2]
    c3 = prof[3] - ybbC
    if mode == _ROOT_REAL:
        x_int = _newton_cubic(c0, c1, c2, c3, init)
    elif mode == _ROOT_CPAIR:
        # np.roots yields a complex pair; the reference takes np.real of it,
        # which by Vieta equals (-c1/c0 - r)/2 with r the single real root.
        r = _newton_cubic(c0, c1, c2, c3, init)
        x_int = (-c1 / _safe(c0) - r) / 2.0
    else:  # _ROOT_CPAIR2, leading coefficient exactly zero
        x_int = -c2 / (2.0 * _safe(c1))

    dxf = jnp.abs(jnp.minimum(x_int - x1, x2 - x_int))
    Xr0 = x_int - dxf * h.Rbb / 2.0
    Xr1 = x_int + dxf * h.Rbb / 2.0

    Y0 = half_beam_bb(h, z, Xr0)
    Y1 = _cubic(prof, Xr1)
    D0 = bb_dydx(h, z, Xr0)
    D1 = _cubic_dydx(prof, Xr1)

    zero = jnp.zeros(())
    one = jnp.ones(())
    # NOTE: row 5 uses 6*Xr0**2 (not 6*Xr0), replicating the reference.
    Arad = jnp.stack(
        [
            jnp.stack([Xr0**4.0, Xr0**3.0, Xr0**2.0, Xr0, one]),
            jnp.stack([Xr1**4.0, Xr1**3.0, Xr1**2.0, Xr1, one]),
            jnp.stack([4.0 * Xr0**3.0, 3.0 * Xr0**2.0, 2.0 * Xr0, one, zero]),
            jnp.stack([4.0 * Xr1**3.0, 3.0 * Xr1**2.0, 2.0 * Xr1, one, zero]),
            jnp.stack([12.0 * Xr0**2.0, 6.0 * Xr0**2.0, 2.0 * one, zero, zero]),
        ]
    )
    brad = jnp.stack([Y0, Y1, D0, D1, zero])
    profrad = jnp.linalg.solve(Arad, brad)
    return prof, Xr0, Xr1, profrad


def _sb_fillet(h: Hull, bits: Bits, z, is_ellipse: bool, branch_a: bool):
    """Stern-bulb fillet: returns (parab, ell, Xr0, Xr1, profrad)."""
    parab, ell = solve_waterline_stern(h, bits, z)
    x1 = delta_stern(h, z)
    x2 = sb_profile(h, z)

    y_int = half_beam_sb(h, z, h.SB_Prof[5])

    if branch_a:
        x_int = h.SB_Prof[0]
    else:
        if is_ellipse:
            x_int = (
                ell[0] * jnp.sqrt(jnp.abs(1.0 - ((y_int - ell[3]) / ell[1]) ** 2.0))
                + ell[2]
            )
        else:
            aa = parab[0]
            bb = parab[1]
            cc = parab[2] - y_int
            disc = bb**2.0 - 4.0 * aa * cc
            # double-where so the sqrt-at-zero branch cannot poison gradients
            # (disc < 0 is the reference's complex-pair case: both "roots"
            # collapse to the real part -b/2a, matching np.real(np.roots(...)))
            pos = disc > 0.0
            sq = jnp.where(pos, jnp.sqrt(jnp.where(pos, disc, 1.0)), 0.0)
            r1 = (-bb + sq) / (2.0 * _safe(aa))
            r2 = (-bb - sq) / (2.0 * _safe(aa))
            neg = jnp.asarray(-jnp.inf)
            x_int = jnp.maximum(
                jnp.where(r1 >= x1, r1, neg), jnp.where(r2 >= x1, r2, neg)
            )

    dxf = jnp.abs(jnp.minimum(x_int - x1, x2 - x_int))
    Xr0 = x_int - dxf * h.Rsb / 2.0
    Xr1 = x_int + dxf * h.Rsb / 2.0

    Y0 = _stern_curve(parab, ell, is_ellipse, Xr0)
    Y1 = half_beam_sb(h, z, Xr1)
    D0 = _stern_curve_dydx(parab, ell, is_ellipse, Xr0)
    D1 = sb_dydx(h, z, Xr1)

    zero = jnp.zeros(())
    one = jnp.ones(())
    # NOTE: row 5 uses Xr1 (and the 6*x**2 quirk), replicating the reference.
    Arad = jnp.stack(
        [
            jnp.stack([Xr0**4.0, Xr0**3.0, Xr0**2.0, Xr0, one]),
            jnp.stack([Xr1**4.0, Xr1**3.0, Xr1**2.0, Xr1, one]),
            jnp.stack([4.0 * Xr0**3.0, 3.0 * Xr0**2.0, 2.0 * Xr0, one, zero]),
            jnp.stack([4.0 * Xr1**3.0, 3.0 * Xr1**2.0, 2.0 * Xr1, one, zero]),
            jnp.stack([12.0 * Xr1**2.0, 6.0 * Xr1**2.0, 2.0 * one, zero, zero]),
        ]
    )
    brad = jnp.stack([Y0, Y1, D0, D1, zero])
    profrad = jnp.linalg.solve(Arad, brad)
    return parab, ell, Xr0, Xr1, profrad


# Waterline section builders (space mode: fixed number of points per section)


def _wl_bow_plain(h: Hull, z, n: int) -> Array:
    x1 = bow_profile(h, z)
    x2 = delta_bow(h, z)
    x = _linspace(x1, x2, n)
    prof = solve_waterline_bow(h, z)
    y = _cubic(prof, x)
    y = y.at[0].set(0.0)
    return jnp.stack([x, y], axis=1)


def _wl_bow_bb(h: Hull, z, n: int, x0: float) -> Array:
    x1 = bb_profile(h, z)
    x2 = delta_bow(h, z)
    x = _linspace(x1, x2, n)

    prof, Xr0, Xr1, profrad = _bb_fillet(h, z, x0)

    y = jnp.where(
        x <= Xr0,
        half_beam_bb(h, z, x),
        jnp.where(x < Xr1, _quartic(profrad, x), _cubic(prof, x)),
    )
    y = y.at[0].set(0.0)
    return jnp.stack([x, y], axis=1)


def _wl_mid(h: Hull, z, n: int) -> Array:
    Xm = _linspace(delta_bow(h, z), delta_stern(h, z), n + 2)
    x = Xm[1 : n + 1]
    y = jnp.broadcast_to(half_beam_midbody(h, z), (n,))
    return jnp.stack([x, y], axis=1)


def _wl_stern_plain(h: Hull, bits: Bits, z, n: int, is_ellipse: bool) -> Array:
    x1 = delta_stern(h, z)
    x2 = stern_profile(h, bits, z)
    x = _linspace(x1, x2, n)
    parab, ell = solve_waterline_stern(h, bits, z)
    y = _stern_curve(parab, ell, is_ellipse, x)
    tx, ty = half_beam_transom(h, bits, z)
    xs = jnp.concatenate([x[: n - 1], tx[None]])
    ys = jnp.concatenate([y[: n - 1], ty[None]])
    return jnp.stack([xs, ys], axis=1)


def _wl_stern_sb(
    h: Hull, bits: Bits, z, n: int, is_ellipse: bool, branch_a: bool
) -> Array:
    x1 = delta_stern(h, z)
    x2 = sb_profile(h, z)
    x = _linspace(x1, x2, n)

    parab, ell, Xr0, Xr1, profrad = _sb_fillet(h, bits, z, is_ellipse, branch_a)

    y = jnp.where(
        x <= Xr0,
        _stern_curve(parab, ell, is_ellipse, x),
        jnp.where(x < Xr1, _quartic(profrad, x), half_beam_sb(h, z, x)),
    )
    xs = jnp.concatenate([x[: n - 1], x2[None]])
    ys = jnp.concatenate([y[: n - 1], jnp.zeros((1,))])
    return jnp.stack([xs, ys], axis=1)


def _wl_zero(h: Hull, bits: Bits, n: int) -> Array:
    """The keel-line waterline used by the reference when ``z == 0``."""
    if bits.BB:
        bow_start = h.BB_Prof[5]
    else:
        bow_start = h.Kappa_BOW * h.Lb
    if bits.SB:
        stern_end = h.SB_Prof[5]
    else:
        stern_end = h.Lm + h.Lb + h.Kappa_STERN * h.Ls
    x = _linspace(bow_start, stern_end, n)
    return jnp.stack([x, jnp.zeros((n,))], axis=1)


# Plans (static topology computed from a concrete design vector)


@dataclass(frozen=True)
class WLPlan:
    pts_bow: int
    pts_mid: int
    pts_stern: int
    bb: bool  # this waterline uses the bow-bulb fillet
    sb: bool  # this waterline uses the stern-bulb fillet
    is_ellipse: bool  # stern curve family at this height
    sb_branch_a: bool  # stern fillet takes the "y_int too large" branch
    bb_x0: tuple  # (mode, init) for the bow fillet intersection root


@dataclass(frozen=True)
class VolumePlan:
    bits: Bits
    num_wl: int
    ppw: int
    wls: tuple[WLPlan, ...]


def _wl_topology(h: Hull, bits: Bits, z: float, ppw: int) -> WLPlan:
    """Replicates gen_WLPoints' integer point allocation and branch choices."""
    WL_c = float(h.WL)
    HSBOA_c = float(h.HSBOA)
    SK1_c = float(h.SK[1])

    if bits.BB and z <= WL_c:
        bow_start = min(float(bb_profile(h, z)), float(bow_profile(h, z)))
    else:
        bow_start = float(bow_profile(h, z))
    if bits.SB and z <= HSBOA_c * WL_c:
        stern_end = max(float(sb_profile(h, z)), float(stern_profile(h, bits, z)))
    else:
        stern_end = float(stern_profile(h, bits, z))

    wl_loa = stern_end - bow_start
    pts_bow = abs(int((float(delta_bow(h, z)) - bow_start) / wl_loa * ppw)) + 1
    pts_stern = abs(int((stern_end - float(delta_stern(h, z))) / wl_loa * ppw)) + 1
    if (pts_bow + pts_stern) > ppw:
        over = pts_bow + pts_stern - ppw
        pts_mid = 0
        pts_stern = pts_stern - over
    else:
        pts_mid = ppw - pts_bow - pts_stern
    if pts_stern < 1 or pts_bow < 1:
        raise ValueError(
            f"degenerate point allocation at z={z}: "
            f"bow={pts_bow}, mid={pts_mid}, stern={pts_stern}"
        )

    use_bb = bool(bits.BB and z < WL_c)
    use_sb = bool(bits.SB and z < WL_c * HSBOA_c)
    is_ellipse = bool(bits.EP_T if z >= SK1_c else bits.EP_S)

    bb_x0 = (_ROOT_REAL, 0.0)
    if use_bb:
        bb_x0 = _bb_root_init(h, z)

    sb_branch_a = _sb_branch(h, bits, z, is_ellipse) if use_sb else False

    return WLPlan(
        pts_bow=pts_bow,
        pts_mid=pts_mid,
        pts_stern=pts_stern,
        bb=use_bb,
        sb=use_sb,
        is_ellipse=is_ellipse,
        sb_branch_a=sb_branch_a,
        bb_x0=bb_x0,
    )


def _sb_branch(h: Hull, bits: Bits, z: float, is_ellipse: bool) -> bool:
    """Concrete stern-fillet branch decision, mirroring the reference.

    Also raises where the reference's parabola/y_int intersection filter
    would be empty (``np.amax`` of an empty list crashes there).
    """
    parab, ell = solve_waterline_stern(h, bits, z)
    y_int = float(half_beam_sb(h, z, h.SB_Prof[5]))
    y_at_x1 = float(_stern_curve(parab, ell, is_ellipse, delta_stern(h, z)))
    branch_a = bool(y_int >= y_at_x1)
    if not branch_a and not is_ellipse:
        aa, bb_, cc = float(parab[0]), float(parab[1]), float(parab[2]) - y_int
        roots = np.roots(np.array([aa, bb_, cc]))
        ds = np.float64(delta_stern(h, z))
        if not [r for r in np.atleast_1d(roots) if r >= ds]:
            raise ValueError(
                f"no admissible stern-fillet intersection at z={z} "
                "(the reference would raise here too)"
            )
    return branch_a


def _bb_root_init(h: Hull, z: float):
    """np.roots-based (mode, init) plan for the bow-fillet intersection.

    Replicates the reference selection
    ``np.amin(np.real([i for i in ROOTS if i >= bow_profile and i >= BB_profile]))``
    including numpy's lexicographic ordering of complex values: when the
    selected "root" is one of a complex-conjugate pair, the reference silently
    uses its *real part*, which we reproduce via Vieta's formulas.
    """
    prof = np.asarray(solve_waterline_bow(h, z), dtype=np.float64)
    ybbC = float(half_beam_bb(h, z, h.BB_Prof[5]))
    coeffs = np.array([prof[0], prof[1], prof[2], prof[3] - ybbC])
    roots = np.roots(coeffs)
    t1 = np.float64(bow_profile(h, z))
    t2 = np.float64(bb_profile(h, z))
    passing = [r for r in np.atleast_1d(roots) if (r >= t1) and (r >= t2)]
    if not passing:
        raise ValueError(
            f"no admissible bow-fillet intersection root at z={z} "
            "(the reference would raise here too)"
        )
    x_ref = float(np.amin(np.real(passing)))
    winners = [r for r in passing if float(np.real(r)) == x_ref]
    if any(np.imag(r) == 0.0 for r in winners):
        return (_ROOT_REAL, x_ref)
    # complex pair selected
    if len(coeffs) - len(roots) > 1 or coeffs[0] == 0.0:
        return (_ROOT_CPAIR2, 0.0)
    real_roots = [float(np.real(r)) for r in np.atleast_1d(roots) if np.imag(r) == 0.0]
    if not real_roots:
        raise ValueError(f"cubic with no real root at z={z}?")
    return (_ROOT_CPAIR, real_roots[0])


def build_volume_plan(x_np, num_wl: int = 101, ppw: int = 1000) -> VolumePlan:
    """Plan for the volume-measurement point cloud (Z = linspace(1e-5, Dd))."""
    x_np = np.asarray(x_np, dtype=np.float64)
    bits = bits_from_vector(x_np)
    h = build_hull(jnp.asarray(x_np), bits)  # eager (concrete) evaluation
    Dd = float(x_np[4] * x_np[0])
    Z = np.linspace(0.00001, Dd, num=num_wl)
    wls = tuple(_wl_topology(h, bits, float(z), ppw) for z in Z)
    return VolumePlan(bits=bits, num_wl=num_wl, ppw=ppw, wls=wls)


def gen_pointcloud(x, plan: VolumePlan) -> Array:
    """(num_wl, ppw, 3) surface point grid; replicates ``gen_pointCloud``."""
    h = build_hull(x, plan.bits)
    Z = _linspace(jnp.asarray(0.00001), h.Dd, plan.num_wl)
    rows = []
    for i, wl in enumerate(plan.wls):
        z = Z[i]
        parts = []
        if wl.bb:
            parts.append(_wl_bow_bb(h, z, wl.pts_bow, wl.bb_x0))
        else:
            parts.append(_wl_bow_plain(h, z, wl.pts_bow))
        parts.append(_wl_mid(h, z, wl.pts_mid))
        if wl.sb:
            parts.append(
                _wl_stern_sb(
                    h, plan.bits, z, wl.pts_stern, wl.is_ellipse, wl.sb_branch_a
                )
            )
        else:
            parts.append(_wl_stern_plain(h, plan.bits, z, wl.pts_stern, wl.is_ellipse))
        xy = jnp.concatenate(parts, axis=0)
        rows.append(
            jnp.concatenate([xy, jnp.broadcast_to(z, (plan.ppw,))[:, None]], axis=1)
        )
    return jnp.stack(rows, axis=0)


# Volumetric / hydrostatic properties (replicates Calc_VolumeProperties)


def volume_properties(x, plan: VolumePlan) -> dict:
    """All hydrostatic measures of ``Calc_VolumeProperties`` (differentiable).

    Returns a dict with Z, pointcloud, areas_wp, volumes, lcf, lcb, vcb,
    ixx, iyy, area_ws, wl_lengths.
    """
    h = build_hull(x, plan.bits)
    pc = gen_pointcloud(x, plan)
    Z = pc[:, 0, 2]

    X = pc[:, :, 0]
    Y = pc[:, :, 1]

    dx = X[:, 1:] - X[:, :-1]
    a = 2.0 * Y[:, 1:]
    b = 2.0 * Y[:, :-1]

    # waterplane areas: 2 * trapz(y, x)
    areas = 2.0 * jnp.sum(0.5 * (Y[:, 1:] + Y[:, :-1]) * dx, axis=1)

    # volumes: cumulative trapezoid over Z, seeded with the bottom slice
    vol0 = 0.5 * areas[0] * Z[0]
    dvol = 0.5 * (areas[1:] + areas[:-1]) * (Z[1:] - Z[:-1])
    volumes = vol0 + jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(dvol)])

    # longitudinal center of flotation per waterplane
    ab = a + b
    cx = X[:, :-1] + dx / 3.0 * (2.0 * a + b) / _safe(ab)
    moment = jnp.sum(cx * 0.5 * ab * dx, axis=1)
    lcf = moment / areas

    # centers of buoyancy: cumulative trapz of moments over Z
    momx = lcf * areas
    momz = Z * areas

    def _cumtrapz(f):
        df = 0.5 * (f[1:] + f[:-1]) * (Z[1:] - Z[:-1])
        return jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(df)])

    lcb = _cumtrapz(momx) / volumes
    vcb = _cumtrapz(momz) / volumes

    # waterplane second moments
    ixx = jnp.sum(dx / 48.0 * ab * (a**2.0 + b**2.0), axis=1)
    iyy = jnp.sum(
        dx**3.0 * (a**2.0 + 4.0 * a * b + b**2.0) / (36.0 * _safe(ab))
        + 0.5 * ab * dx * (lcf[:, None] - cx) ** 2.0,
        axis=1,
    )

    # wetted surface (transom width intentionally not added; see module docs)
    dy = Y[:, 1:] - Y[:, :-1]
    seg2 = dx**2.0 + dy**2.0
    seg = jnp.where(seg2 > 0.0, jnp.sqrt(jnp.where(seg2 > 0.0, seg2, 1.0)), 0.0)
    arc = jnp.sum(seg, axis=1)
    wsa0 = 2.0 * areas[0]
    dwsa = (arc[1:] + arc[:-1]) * (Z[1:] - Z[:-1])
    area_ws = wsa0 + jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(dwsa)])

    wl_lengths = X[:, -1] - X[:, 0]

    return dict(
        Z=Z,
        pointcloud=pc,
        areas_wp=areas,
        volumes=volumes,
        lcf=lcf,
        lcb=lcb,
        vcb=vcb,
        ixx=ixx,
        iyy=iyy,
        area_ws=area_ws,
        wl_lengths=wl_lengths,
        hull=h,
    )


# Michell point grid (replicates gen_PC_for_Cw through gen_MeshGridWL)


@dataclass(frozen=True)
class CwWLPlan:
    bb: bool
    sb: bool
    is_ellipse: bool
    sb_branch_a: bool
    bb_x0: tuple  # (mode, init) for the bow fillet intersection root
    mask: tuple  # bool per grid station: strictly inside (x1, x2)
    # When a grid station coincides exactly with delta_bow, the reference
    # double-counts it (bow filter is (x1, db], midbody filter is [db, ds))
    # and its Y-grid assembly shifts by one slot from that station on.  These
    # fields replicate that assembly: i0 = first row, cnt = station count in
    # (x1, x2), dup_k = number of bow rows up to and including the duplicate
    # (or -1 when no coincidence occurs and `mask` alone applies).
    i0: int = 0
    cnt: int = 0
    dup_k: int = -1


@dataclass(frozen=True)
class CwPlan:
    bits: Bits
    draft_frac: float  # draft as a fraction of Dd (dataset uses Dd * z_ref)
    num_wl: int
    ppw: int
    wls: tuple[CwWLPlan, ...]


def build_cw_plan(x_np, draft_frac: float, num_wl: int = 51, ppw: int = 301) -> CwPlan:
    x_np = np.asarray(x_np, dtype=np.float64)
    bits = bits_from_vector(x_np)
    h = build_hull(jnp.asarray(x_np), bits)

    Dd = float(h.Dd)
    WL_c = float(h.WL)
    HSBOA_c = float(h.HSBOA)
    SK1_c = float(h.SK[1])
    draft = Dd * draft_frac

    Z = np.asarray(
        _linspace(jnp.asarray(0.00000001) * h.Dd, jnp.asarray(draft), num_wl)
    )
    x_bow = np.zeros(num_wl)
    x_stern = np.zeros(num_wl)
    for i, z in enumerate(Z):
        if bits.BB and z <= WL_c:
            x_bow[i] = float(bb_profile(h, z))
        else:
            x_bow[i] = float(bow_profile(h, z))
        if bits.SB and z <= HSBOA_c * WL_c:
            x_stern[i] = float(sb_profile(h, z))
        else:
            x_stern[i] = float(stern_profile(h, bits, z))

    Xg = np.asarray(
        _linspace(jnp.asarray(x_bow.min()), jnp.asarray(x_stern.max()), ppw)
    )

    wls = []
    for i, z in enumerate(Z):
        z = float(z)
        use_bb = bool(bits.BB and z < WL_c)
        use_sb = bool(bits.SB and z < WL_c * HSBOA_c)
        is_ellipse = bool(bits.EP_T if z >= SK1_c else bits.EP_S)

        x1 = x_bow[i]
        x2 = x_stern[i]
        mask = (Xg > x1) & (Xg < x2)
        inside = np.where(mask)[0]
        i0 = int(inside[0]) if inside.size else 0
        cnt = int(inside.size)

        # A grid station exactly on delta_bow is double-counted by the
        # reference's section filters; replicate its shifted assembly.
        db = float(delta_bow(h, z))
        dup_idx = np.where(mask & (Xg == db))[0]
        dup_k = int(dup_idx[0]) - i0 + 1 if dup_idx.size else -1

        bb_x0 = _bb_root_init(h, z) if use_bb else (_ROOT_REAL, 0.0)

        sb_branch_a = _sb_branch(h, bits, z, is_ellipse) if use_sb else False

        wls.append(
            CwWLPlan(
                bb=use_bb,
                sb=use_sb,
                is_ellipse=is_ellipse,
                sb_branch_a=sb_branch_a,
                bb_x0=bb_x0,
                mask=tuple(bool(m) for m in mask),
                i0=i0,
                cnt=cnt,
                dup_k=dup_k,
            )
        )
    return CwPlan(
        bits=bits, draft_frac=float(draft_frac), num_wl=num_wl, ppw=ppw, wls=tuple(wls)
    )


def _y_at_grid(h: Hull, bits: Bits, z, Xg: Array, wl: CwWLPlan) -> Array:
    """Half beam at every grid station for one waterline (before masking)."""
    if wl.bb:
        prof, Xr0, Xr1, profrad = _bb_fillet(h, z, wl.bb_x0)
        y_bow = jnp.where(
            Xg <= Xr0,
            half_beam_bb(h, z, Xg),
            jnp.where(Xg < Xr1, _quartic(profrad, Xg), _cubic(prof, Xg)),
        )
    else:
        prof = solve_waterline_bow(h, z)
        y_bow = _cubic(prof, Xg)

    y_mid = jnp.broadcast_to(half_beam_midbody(h, z), Xg.shape)

    if wl.sb:
        parab, ell, Xr0s, Xr1s, profrad_s = _sb_fillet(
            h, bits, z, wl.is_ellipse, wl.sb_branch_a
        )
        y_stern = jnp.where(
            Xg <= Xr0s,
            _stern_curve(parab, ell, wl.is_ellipse, Xg),
            jnp.where(Xg < Xr1s, _quartic(profrad_s, Xg), half_beam_sb(h, z, Xg)),
        )
    else:
        parab, ell = solve_waterline_stern(h, bits, z)
        y_stern = _stern_curve(parab, ell, wl.is_ellipse, Xg)

    x2b = delta_bow(h, z)
    x1s = delta_stern(h, z)
    return jnp.where(Xg <= x2b, y_bow, jnp.where(Xg < x1s, y_mid, y_stern))


def gen_pc_for_cw(x, plan: CwPlan):
    """(X, Z, Y, WL) inputs for the Michell integral; matches gen_PC_for_Cw."""
    h = build_hull(x, plan.bits)
    draft = h.Dd * plan.draft_frac
    Z = _linspace(0.00000001 * h.Dd, draft, plan.num_wl)

    if plan.bits.BB:
        x_bow = jnp.where(Z <= h.WL, bb_profile(h, Z), bow_profile(h, Z))
    else:
        x_bow = bow_profile(h, Z)
    if plan.bits.SB:
        x_stern = jnp.where(
            Z <= h.HSBOA * h.WL, sb_profile(h, Z), stern_profile(h, plan.bits, Z)
        )
    else:
        x_stern = stern_profile(h, plan.bits, Z)

    WLlen = x_stern[-1] - x_bow[-1]
    Xg = _linspace(jnp.min(x_bow), jnp.max(x_stern), plan.ppw)

    cols = []
    for i, wl in enumerate(plan.wls):
        y = _y_at_grid(h, plan.bits, Z[i], Xg, wl)
        if wl.dup_k < 0:
            mask = jnp.asarray(np.array(wl.mask, dtype=bool))
            cols.append(jnp.where(mask, y, 0.0))
        else:
            # replicate the reference's one-slot shift after the duplicated
            # delta_bow station (see CwWLPlan)
            i0, cnt, k = wl.i0, wl.cnt, wl.dup_k
            col = jnp.zeros((plan.ppw,))
            col = col.at[i0 : i0 + k].set(y[i0 : i0 + k])
            col = col.at[i0 + k].set(half_beam_midbody(h, Z[i]))
            col = col.at[i0 + k + 1 : i0 + cnt + 1].set(y[i0 + k : i0 + cnt])
            cols.append(col)
    Y = jnp.stack(cols, axis=1)  # (ppw, num_wl)

    return Xg - Xg[0], Z - Z[-1], Y, WLlen
