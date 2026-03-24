"""Functional JAX implementation of the ShipD hull parameterization core steps."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class HullJaxConfig(NamedTuple):
    """Static configuration (int dimensions + feature bits) for JIT-safe execution."""

    NUM_WL: int = 101
    POINTS_PER_WL: int = 100
    bit_EP_S: bool = False
    bit_EP_T: bool = False
    bit_BB: bool = False
    bit_SB: bool = False


DEFAULT_CONFIG = HullJaxConfig()


class HullInputs(NamedTuple):
    LOA: float
    Lb: float
    Ls: float
    Bd: float
    Dd: float
    Bs: float
    WL: float
    Bc: float
    Beta: float
    Rc: float
    Rk: float
    BOW: jax.Array
    BK: jax.Array
    Kappa_BOW: float
    DELTA_BOW: jax.Array
    DRIFT: jax.Array
    TRANS: jax.Array
    SK: jax.Array
    Kappa_STERN: float
    DELTA_STERN: jax.Array
    Beta_trans: float
    Bc_trans: float
    Rc_trans: float
    Rk_trans: float
    Kappa_SB: float
    Lbb: float
    Hbb: float
    Bbb: float
    Lbbm: float
    Rbb: float
    Lsb: float
    HSBOA: float
    Hsb: float
    Bsb: float
    Lsbm: float
    Rsb: float


class HullState(NamedTuple):
    inp: HullInputs
    Lm: jax.Array
    # Cross section
    Rk_Center: jax.Array
    Rk_LG_int: jax.Array
    LG: jax.Array
    Dc: jax.Array
    UG: jax.Array
    Rc_UG_int: jax.Array
    Rc_LG_int: jax.Array
    Rc_Center: jax.Array
    # Bow
    BOW: jax.Array
    BK: jax.Array
    KEEL_BOW: jax.Array
    DELTA_BOW: jax.Array
    # Stern
    TRANS: jax.Array
    SK: jax.Array
    STERNRISE: jax.Array
    DELTA_STERN: jax.Array
    Rk_Center_trans: jax.Array
    Rk_LG_int_trans: jax.Array
    LG_trans: jax.Array
    Dc_trans: jax.Array
    UG_trans: jax.Array
    Rc_UG_int_trans: jax.Array
    Rc_LG_int_trans: jax.Array
    Rc_Center_trans: jax.Array
    # Bulbs
    BB_Prof: jax.Array
    SB_Prof: jax.Array


INPUT_ORDER = [
    "LOA", "Lb", "Ls", "Bd", "Dd", "Bs", "WL", "Bc", "Beta", "Rc", "Rk",
    "BOW_0", "BOW_1", "BK_1", "Kappa_BOW", "DELTA_BOW_0", "DELTA_BOW_1",
    "DRIFT_0", "DRIFT_1", "DRIFT_2", "bit_EP_S", "bit_EP_T", "TRANS_0", "SK_1",
    "Kappa_STERN", "DELTA_STERN_0", "DELTA_STERN_1", "Beta_trans", "Bc_trans",
    "Rc_trans", "Rk_trans", "bit_BB", "bit_SB", "Lbb", "Hbb", "Bbb", "Lbbm",
    "Rbb", "Kappa_SB", "Lsb", "HSBOA", "Hsb", "Bsb", "Lsbm", "Rsb",
]


def parse_design_vector(x: jax.Array, config: HullJaxConfig = DEFAULT_CONFIG) -> HullInputs:
    """Parse the 45 design variables in the exact legacy order from Hull_Parameterization.__init__."""
    x = jnp.asarray(x)
    if x.shape[-1] != 45:
        raise ValueError(f"Expected 45 design variables, got {x.shape[-1]}")

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

    BOW = jnp.array([x[11] * 0.5 * Lb / Dd**2.0, x[12] * 0.5 * Lb / Dd, 0.0])
    BK = jnp.array([0.0, x[13] * Dd])
    Kappa_BOW = x[14]
    DELTA_BOW = jnp.array([x[15] * 0.5 * Lb / Dd**2.0, x[16] * 0.5 * Lb / Dd, 0.0])
    DRIFT = jnp.array([x[17] * 60.0 / Dd**2.0, x[18] * 60.0 / Dd, x[19]])

    TRANS = jnp.array([x[22], 0.0])
    SK = jnp.array([0.0, x[23]])
    Kappa_STERN = x[24]
    DELTA_STERN = jnp.array([x[25] * 0.5 * Ls / Dd**2.0, x[26] * 0.5 * Ls / Dd, 0.0])
    Beta_trans = x[27]
    Bc_trans = x[28] / 2.0 * LOA
    Rc_trans = x[29] * Bc_trans
    Rk_trans = x[30] * Dd * (1.0 - SK[1])

    return HullInputs(
        LOA=LOA, Lb=Lb, Ls=Ls, Bd=Bd, Dd=Dd, Bs=Bs, WL=WL, Bc=Bc, Beta=Beta, Rc=Rc, Rk=Rk,
        BOW=BOW, BK=BK, Kappa_BOW=Kappa_BOW, DELTA_BOW=DELTA_BOW, DRIFT=DRIFT,
        TRANS=TRANS, SK=SK, Kappa_STERN=Kappa_STERN, DELTA_STERN=DELTA_STERN,
        Beta_trans=Beta_trans, Bc_trans=Bc_trans, Rc_trans=Rc_trans, Rk_trans=Rk_trans,
        Kappa_SB=x[38], Lbb=x[33], Hbb=x[34], Bbb=x[35], Lbbm=x[36], Rbb=x[37],
        Lsb=x[39], HSBOA=x[40], Hsb=x[41], Bsb=x[42], Lsbm=x[43], Rsb=x[44],
    )


def _solve_chine(UG, LG, Rc, Beta_deg):
    A1, B1 = UG[0], UG[1]
    theta = jnp.arctan2(-B1, A1)
    theta = jnp.where(theta < 0.0, theta + jnp.pi, theta)
    beta = Beta_deg * jnp.pi / 180.0
    A2, B2 = LG[0], LG[1]
    A = jnp.array([
        [B1, A1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, B2, A2, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
    ])
    b = jnp.array([
        -UG[2], -LG[2], Rc * jnp.sin(theta), Rc * jnp.cos(theta), Rc * jnp.sin(beta), Rc * jnp.cos(beta)
    ])
    C = jnp.linalg.solve(A, b)
    return C[0:2], C[2:4], C[4:6]


def gen_general_hullform(inp: HullInputs):
    return inp.LOA - inp.Ls - inp.Lb


def gen_cross_section(inp: HullInputs):
    Rk_Center = jnp.array([-inp.Rk * (0.5 - 0.5 * jnp.sign(inp.Rk)), inp.Rk * (0.5 + 0.5 * jnp.sign(inp.Rk))])
    Rk_LG_int = jnp.array([
        Rk_Center[0] + inp.Rk * jnp.sin(jnp.pi * inp.Beta / 180.0),
        Rk_Center[1] - inp.Rk * jnp.cos(jnp.pi * inp.Beta / 180.0),
    ])
    A_lg = jnp.array([
        [1.0, 1.0, 1.0],
        [Rk_LG_int[1], Rk_LG_int[0], 1.0],
        [-(Rk_LG_int[0] - Rk_Center[0]), (Rk_LG_int[1] - Rk_Center[1]), 0.0],
    ])
    LG = jnp.linalg.solve(A_lg, jnp.array([1.0, 0.0, 0.0]))
    Dc = -(LG[1] * inp.Bc + LG[2]) / LG[0]
    A_ug = jnp.array([[Dc, inp.Bc, 1.0], [inp.Dd, inp.Bd, 1.0], [1.0, 1.0, 1.0]])
    UG = jnp.linalg.solve(A_ug, jnp.array([0.0, 0.0, 1.0]))
    Rc_UG_int, Rc_LG_int, Rc_Center = _solve_chine(UG, LG, inp.Rc, inp.Beta)
    return Rk_Center, Rk_LG_int, LG, Dc, UG, Rc_UG_int, Rc_LG_int, Rc_Center


def _bowrise(BOW, z):
    return BOW[0] * z**2.0 + BOW[1] * z + BOW[2]


def _bow_profile(BK, KEEL_BOW, Kappa_BOW, Lb, BOW, z):
    return lax.cond(z <= BK[1], lambda _: -jnp.sqrt(z / KEEL_BOW) + Kappa_BOW * Lb, lambda _: _bowrise(BOW, z), operand=None)


def _halfbeam_midbody(z, inp: HullInputs, Rk_Center, Rk_LG_int, LG, Rc_UG_int, Rc_LG_int, Rc_Center, UG):
    y0 = jnp.sign(inp.Rk) * jnp.sqrt(jnp.maximum(0.0, (inp.Rk**2) - (z - Rk_Center[1]) ** 2)) + Rk_Center[0]
    y1 = -(LG[0] * z + LG[2]) / LG[1]
    y2 = jnp.sqrt(jnp.maximum(0.0, (inp.Rc**2) - (z - Rc_Center[1]) ** 2)) + Rc_Center[0]
    y3 = -(UG[0] * z + UG[2]) / UG[1]
    return jnp.where(
        z < Rk_LG_int[1],
        y0,
        jnp.where(z < Rc_LG_int[1], y1, jnp.where(z < Rc_UG_int[1], y2, y3)),
    )


def gen_bow_form(inp: HullInputs):
    z_vertex = lax.cond(inp.BOW[0] == 0.0, lambda _: -1.0, lambda _: -inp.BOW[1] / (2.0 * inp.BOW[0]), operand=None)
    C = jnp.array([
        inp.BOW[0] * inp.Dd**2.0 + inp.BOW[1] * inp.Dd,
        inp.BOW[0] * inp.BK[1] ** 2.0 + inp.BOW[1] * inp.BK[1],
        inp.BOW[0] * z_vertex**2.0 + inp.BOW[1] * z_vertex,
    ])
    bow2 = lax.cond((z_vertex >= inp.BK[1] * inp.Dd) & (z_vertex <= inp.Dd), lambda _: -jnp.min(C), lambda _: -jnp.min(C[:2]), operand=None)
    BOW = inp.BOW.at[2].set(bow2)
    bk0 = _bowrise(BOW, inp.BK[1])
    BK = inp.BK.at[0].set(bk0)
    KEEL_BOW = BK[1] / ((BK[0] - inp.Kappa_BOW * inp.Lb) ** 2.0)

    z_vertex_d = lax.cond(inp.DELTA_BOW[0] == 0.0, lambda _: -1.0, lambda _: -inp.DELTA_BOW[1] / (2.0 * inp.DELTA_BOW[0]), operand=None)
    C2 = jnp.array([
        inp.DELTA_BOW[0] * inp.Dd**2.0 + inp.DELTA_BOW[1] * inp.Dd,
        0.0,
        inp.DELTA_BOW[0] * z_vertex_d**2.0 + inp.DELTA_BOW[1] * z_vertex_d,
    ])
    delta2 = lax.cond((z_vertex_d >= 0.0) & (z_vertex_d <= inp.Dd), lambda _: -jnp.max(C2), lambda _: -jnp.max(C2[:2]), operand=None)
    DELTA_BOW = inp.DELTA_BOW.at[2].set(delta2)
    return BOW, BK, KEEL_BOW, DELTA_BOW


def gen_stern_form(inp: HullInputs, Lm):
    SK = inp.SK.at[1].set(inp.SK[1] * inp.Dd)
    t1 = jnp.where(inp.TRANS[0] >= 0.0, inp.LOA - inp.TRANS[0] * inp.Dd, inp.LOA - inp.TRANS[0] * SK[1])
    TRANS = inp.TRANS.at[1].set(t1)
    SK = SK.at[0].set(TRANS[0] * SK[1] + TRANS[1])
    STERNRISE = SK[1] / (SK[0] - (inp.Lb + Lm + inp.Ls * inp.Kappa_STERN)) ** 2.0

    zv = lax.cond(inp.DELTA_STERN[0] == 0.0, lambda _: -1.0, lambda _: -inp.DELTA_STERN[1] / (2.0 * inp.DELTA_STERN[0]), operand=None)
    C = jnp.array([
        inp.DELTA_STERN[0] * inp.Dd**2.0 + inp.DELTA_STERN[1] * inp.Dd,
        0.0,
        inp.DELTA_STERN[0] * zv**2.0 + inp.DELTA_STERN[1] * zv,
    ])
    delta2 = lax.cond((zv >= 0.0) & (zv <= inp.Dd), lambda _: -jnp.min(C), lambda _: -jnp.min(C[:2]), operand=None)
    DELTA_STERN = inp.DELTA_STERN.at[2].set(delta2)

    Rk_Center_t = jnp.array([-inp.Rk_trans * (0.5 - 0.5 * jnp.sign(inp.Rk_trans)), SK[1] + inp.Rk_trans * (0.5 + 0.5 * jnp.sign(inp.Rk_trans))])
    Rk_LG_int_t = jnp.array([
        Rk_Center_t[0] + inp.Rk_trans * jnp.sin(jnp.pi * inp.Beta_trans / 180.0),
        Rk_Center_t[1] - inp.Rk_trans * jnp.cos(jnp.pi * inp.Beta_trans / 180.0),
    ])
    A_lg = jnp.array([
        [1.0, 1.0, 1.0],
        [Rk_LG_int_t[1], Rk_LG_int_t[0], 1.0],
        [-(Rk_LG_int_t[0] - Rk_Center_t[0]), (Rk_LG_int_t[1] - Rk_Center_t[1]), 0.0],
    ])
    LG_t = jnp.linalg.solve(A_lg, jnp.array([1.0, 0.0, 0.0]))
    Dc_t = -(LG_t[1] * inp.Bc_trans + LG_t[2]) / LG_t[0]
    A_ug = jnp.array([[Dc_t, inp.Bc_trans, 1.0], [inp.Dd, inp.Bs, 1.0], [1.0, 1.0, 1.0]])
    UG_t = jnp.linalg.solve(A_ug, jnp.array([0.0, 0.0, 1.0]))
    Rc_UG_t, Rc_LG_t, Rc_C_t = _solve_chine(UG_t, LG_t, inp.Rc_trans, inp.Beta_trans)

    return TRANS, SK, STERNRISE, DELTA_STERN, Rk_Center_t, Rk_LG_int_t, LG_t, Dc_t, UG_t, Rc_UG_t, Rc_LG_t, Rc_C_t


def gen_bulb_forms(inp: HullInputs, BOW, BK, KEEL_BOW, cross_data, Lm, config: HullJaxConfig):
    Rk_Center, Rk_LG_int, LG, _, UG, Rc_UG_int, Rc_LG_int, Rc_Center = cross_data
    BB_Prof = jnp.zeros((7,))
    SB_Prof = jnp.zeros((7,))

    FP = _bow_profile(BK, KEEL_BOW, inp.Kappa_BOW, inp.Lb, BOW, inp.WL)
    bb = jnp.array([
        FP,
        (1.0 - inp.Hbb) * inp.WL,
        inp.Hbb * inp.WL,
        _halfbeam_midbody(inp.Hbb * inp.WL, inp, Rk_Center, Rk_LG_int, LG, Rc_UG_int, Rc_LG_int, Rc_Center, UG) * inp.Bbb,
        inp.Lbb * inp.LOA * (1.0 - inp.Lbbm),
        FP - inp.LOA * inp.Lbb * inp.Lbbm,
        0.0,
    ])
    SBs = inp.Kappa_SB * inp.Ls + Lm + inp.Lb
    sb = jnp.array([
        SBs,
        (1.0 - inp.Hsb) * inp.WL * inp.HSBOA,
        inp.Hsb * inp.WL * inp.HSBOA,
        _halfbeam_midbody(inp.Hsb * inp.WL * inp.HSBOA, inp, Rk_Center, Rk_LG_int, LG, Rc_UG_int, Rc_LG_int, Rc_Center, UG) * inp.Bsb,
        inp.Lsb * inp.LOA * (1.0 - inp.Lsbm),
        SBs + inp.LOA * inp.Lsb * inp.Lsbm,
        0.0,
    ])
    BB_Prof = lax.cond(config.bit_BB, lambda _: bb, lambda _: BB_Prof, operand=None)
    SB_Prof = lax.cond(config.bit_SB, lambda _: sb, lambda _: SB_Prof, operand=None)
    return BB_Prof, SB_Prof


def build_hull_state(x: jax.Array, config: HullJaxConfig = DEFAULT_CONFIG) -> HullState:
    """Single functional entry point returning a structured state for downstream JAX calls."""
    inp = parse_design_vector(x, config)
    Lm = gen_general_hullform(inp)
    cross_data = gen_cross_section(inp)
    BOW, BK, KEEL_BOW, DELTA_BOW = gen_bow_form(inp)
    stern = gen_stern_form(inp, Lm)
    BB_Prof, SB_Prof = gen_bulb_forms(inp, BOW, BK, KEEL_BOW, cross_data, Lm, config)
    return HullState(
        inp=inp,
        Lm=Lm,
        Rk_Center=cross_data[0], Rk_LG_int=cross_data[1], LG=cross_data[2], Dc=cross_data[3], UG=cross_data[4],
        Rc_UG_int=cross_data[5], Rc_LG_int=cross_data[6], Rc_Center=cross_data[7],
        BOW=BOW, BK=BK, KEEL_BOW=KEEL_BOW, DELTA_BOW=DELTA_BOW,
        TRANS=stern[0], SK=stern[1], STERNRISE=stern[2], DELTA_STERN=stern[3],
        Rk_Center_trans=stern[4], Rk_LG_int_trans=stern[5], LG_trans=stern[6], Dc_trans=stern[7], UG_trans=stern[8],
        Rc_UG_int_trans=stern[9], Rc_LG_int_trans=stern[10], Rc_Center_trans=stern[11],
        BB_Prof=BB_Prof, SB_Prof=SB_Prof,
    )


build_hull_state_jit = jax.jit(build_hull_state, static_argnames=("config",))
