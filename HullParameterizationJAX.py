"""JAX-compatible hull input constraint evaluation.

This module mirrors ``Hull_Parameterization.input_Constraints`` from
``HullParameterization.py`` and returns a fixed-size (49,) constraint vector.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax import lax


PI = jnp.pi


def _cfg_value(cfg: Mapping[str, Any] | None, key: str, default: float) -> float:
    if cfg is None:
        return float(default)
    value = cfg.get(key, default)
    return float(value)


def input_constraints_jax(x: jnp.ndarray, cfg: Mapping[str, Any] | None = None) -> jnp.ndarray:
    """Return the 49 Ship-D input constraints as a JAX array.

    Ordering and semantics match ``Hull_Parameterization.input_Constraints``.
    """

    eps = _cfg_value(cfg, "eps", 1e-12)

    x = jnp.asarray(x, dtype=jnp.float64)

    LOA = x[0]
    Lb = x[1] * LOA
    Ls = x[2] * LOA
    Bd = x[3] * 0.5 * LOA
    Dd = x[4] * LOA
    Bs = x[5] * Bd
    WL = x[6] * Dd
    Bc = x[7] * 0.5 * LOA
    Beta = x[8]
    Rc = x[9] * Bc
    Rk = x[10] * Dd

    BOW0 = x[11] * 0.5 * Lb / (Dd**2)
    BOW1 = x[12] * 0.5 * Lb / Dd

    BK1 = x[13] * Dd
    Kappa_BOW = x[14]

    DELTA_BOW0 = x[15] * 0.5 * Lb / (Dd**2)
    DELTA_BOW1 = x[16] * 0.5 * Lb / Dd

    DRIFT0 = x[17] * 60.0 / (Dd**2)
    DRIFT1 = x[18] * 60.0 / Dd
    DRIFT2 = x[19]

    bit_EP_S = x[20]
    bit_EP_T = x[21]
    del bit_EP_S, bit_EP_T  # kept for complete input mapping

    TRANS0 = x[22]
    SK1 = x[23]
    Kappa_STERN = x[24]

    DELTA_STERN0 = x[25] * 0.5 * Ls / (Dd**2)
    DELTA_STERN1 = x[26] * 0.5 * Ls / Dd

    Beta_trans = x[27]
    Bc_trans = x[28] * 0.5 * LOA
    Rc_trans = x[29] * Bc_trans
    Rk_trans = x[30] * Dd * (1.0 - SK1)

    bit_BB = x[31]
    bit_SB = x[32]

    Lbb = x[33]
    Hbb = x[34]
    Bbb = x[35]
    Lbbm = x[36]
    Rbb = x[37]
    del Rbb

    Kappa_SB = x[38]
    Lsb = x[39]
    HSBOA = x[40]
    Hsb = x[41]
    Bsb = x[42]
    Lsbm = x[43]
    Rsb = x[44]
    del Rsb

    Lm = LOA - Ls - Lb

    # ---------- Cross section geometry ----------
    Rk_center = jnp.array(
        [
            -Rk * (0.5 - 0.5 * jnp.sign(Rk)),
            Rk * (0.5 + 0.5 * jnp.sign(Rk)),
        ]
    )
    Rk_lg_int = jnp.array(
        [
            Rk_center[0] + Rk * jnp.sin(PI * Beta / 180.0),
            Rk_center[1] - Rk * jnp.cos(PI * Beta / 180.0),
        ]
    )

    A_lg = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [Rk_lg_int[1], Rk_lg_int[0], 1.0],
            [-(Rk_lg_int[0] - Rk_center[0]), (Rk_lg_int[1] - Rk_center[1]), 0.0],
        ]
    )
    b_lg = jnp.array([1.0, 0.0, 0.0])
    LG = jnp.linalg.solve(A_lg, b_lg)

    Dc = -(LG[1] * Bc + LG[2]) / LG[0]

    A_ug = jnp.array([[Dc, Bc, 1.0], [Dd, Bd, 1.0], [1.0, 1.0, 1.0]])
    b_ug = jnp.array([0.0, 0.0, 1.0])
    UG = jnp.linalg.solve(A_ug, b_ug)

    theta = jnp.arctan2(-UG[1], UG[0])
    theta = jnp.where(theta < 0.0, theta + PI, theta)
    beta = Beta * PI / 180.0

    A_chine = jnp.array(
        [
            [UG[1], UG[0], 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, LG[1], LG[0], 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
        ]
    )
    b_chine = jnp.array(
        [
            -UG[2],
            -LG[2],
            Rc * jnp.sin(theta),
            Rc * jnp.cos(theta),
            Rc * jnp.sin(beta),
            Rc * jnp.cos(beta),
        ]
    )
    C_chine = jnp.linalg.solve(A_chine, b_chine)
    Rc_UG_int = C_chine[0:2]
    Rc_LG_int = C_chine[2:4]
    Rc_center = C_chine[4:6]

    def halfbeam_midbody(z: jnp.ndarray) -> jnp.ndarray:
        invalid = (z < 0.0) | (z > Dd)
        keel = jnp.sign(Rk) * jnp.sqrt(jnp.abs((Rk**2) - (z - Rk_center[1]) ** 2)) + Rk_center[0]
        lg = -(LG[0] * z + LG[2]) / LG[1]
        chine = jnp.sqrt(jnp.abs((Rc**2) - (z - Rc_center[1]) ** 2)) + Rc_center[0]
        ug = -(UG[0] * z + UG[2]) / UG[1]

        y = jnp.where(z < Rk_lg_int[1], keel, lg)
        y = jnp.where(z < Rc_LG_int[1], y, chine)
        y = jnp.where(z < Rc_UG_int[1], y, ug)
        y = jnp.where(invalid, -1.0, y)
        return y

    # ---------- Bow geometry ----------
    bow_zv = jnp.where(jnp.abs(BOW0) < eps, -1.0, -BOW1 / (2.0 * BOW0))
    bow_candidates = jnp.array(
        [
            BOW0 * Dd**2 + BOW1 * Dd,
            BOW0 * BK1**2 + BOW1 * BK1,
            BOW0 * bow_zv**2 + BOW1 * bow_zv,
        ]
    )
    bow2_in_range = (bow_zv >= BK1 * Dd) & (bow_zv <= Dd)
    BOW2 = jnp.where(bow2_in_range, -jnp.min(bow_candidates), -jnp.min(bow_candidates[:2]))

    def bowrise(z: jnp.ndarray) -> jnp.ndarray:
        return BOW0 * z**2 + BOW1 * z + BOW2

    BK0 = bowrise(BK1)

    KEEL_BOW = BK1 / ((BK0 - Kappa_BOW * Lb) ** 2)

    delta_bow_zv = jnp.where(jnp.abs(DELTA_BOW0) < eps, -1.0, -DELTA_BOW1 / (2.0 * DELTA_BOW0))
    delta_bow_candidates = jnp.array(
        [
            DELTA_BOW0 * Dd**2 + DELTA_BOW1 * Dd,
            0.0,
            DELTA_BOW0 * delta_bow_zv**2 + DELTA_BOW1 * delta_bow_zv,
        ]
    )
    delta_bow2_in_range = (delta_bow_zv >= 0.0) & (delta_bow_zv <= Dd)
    DELTA_BOW2 = jnp.where(
        delta_bow2_in_range,
        -jnp.max(delta_bow_candidates),
        -jnp.max(delta_bow_candidates[:2]),
    )

    def keelrise_bow(z: jnp.ndarray) -> jnp.ndarray:
        return -jnp.sqrt(jnp.abs(z / KEEL_BOW)) + Kappa_BOW * Lb

    def delta_bow(z: jnp.ndarray) -> jnp.ndarray:
        return Lb + DELTA_BOW0 * z**2 + DELTA_BOW1 * z + DELTA_BOW2

    def drift(z: jnp.ndarray) -> jnp.ndarray:
        return PI * (DRIFT0 * z**2 + DRIFT1 * z + DRIFT2) / 180.0

    def bow_profile(z: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(z <= BK1, keelrise_bow(z), bowrise(z))

    # ---------- Stern geometry ----------
    SK1 = SK1 * Dd
    TRANS1 = jnp.where(TRANS0 >= 0.0, LOA - TRANS0 * Dd, LOA - TRANS0 * SK1)

    def transom(z: jnp.ndarray) -> jnp.ndarray:
        return TRANS0 * z + TRANS1

    SK0 = transom(SK1)
    STERNRISE = SK1 / (SK0 - (Lb + Lm + Ls * Kappa_STERN)) ** 2

    delta_stern_zv = jnp.where(jnp.abs(DELTA_STERN0) < eps, -1.0, -DELTA_STERN1 / (2.0 * DELTA_STERN0))
    delta_stern_candidates = jnp.array(
        [
            DELTA_STERN0 * Dd**2 + DELTA_STERN1 * Dd,
            0.0,
            DELTA_STERN0 * delta_stern_zv**2 + DELTA_STERN1 * delta_stern_zv,
        ]
    )
    delta_stern2_in_range = (delta_stern_zv >= 0.0) & (delta_stern_zv <= Dd)
    DELTA_STERN2 = jnp.where(
        delta_stern2_in_range,
        -jnp.min(delta_stern_candidates),
        -jnp.min(delta_stern_candidates[:2]),
    )

    Rk_center_t = jnp.array(
        [
            -Rk_trans * (0.5 - 0.5 * jnp.sign(Rk_trans)),
            SK1 + Rk_trans * (0.5 + 0.5 * jnp.sign(Rk_trans)),
        ]
    )
    Rk_lg_int_t = jnp.array(
        [
            Rk_center_t[0] + Rk_trans * jnp.sin(PI * Beta_trans / 180.0),
            Rk_center_t[1] - Rk_trans * jnp.cos(PI * Beta_trans / 180.0),
        ]
    )

    A_lg_t = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [Rk_lg_int_t[1], Rk_lg_int_t[0], 1.0],
            [-(Rk_lg_int_t[0] - Rk_center_t[0]), (Rk_lg_int_t[1] - Rk_center_t[1]), 0.0],
        ]
    )
    LG_t = jnp.linalg.solve(A_lg_t, b_lg)
    Dc_trans = -(LG_t[1] * Bc_trans + LG_t[2]) / LG_t[0]

    A_ug_t = jnp.array([[Dc_trans, Bc_trans, 1.0], [Dd, Bs, 1.0], [1.0, 1.0, 1.0]])
    UG_t = jnp.linalg.solve(A_ug_t, b_ug)

    theta_t = jnp.arctan2(-UG_t[1], UG_t[0])
    theta_t = jnp.where(theta_t < 0.0, theta_t + PI, theta_t)
    beta_t = Beta_trans * PI / 180.0

    A_chine_t = jnp.array(
        [
            [UG_t[1], UG_t[0], 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, LG_t[1], LG_t[0], 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
        ]
    )
    b_chine_t = jnp.array(
        [
            -UG_t[2],
            -LG_t[2],
            Rc_trans * jnp.sin(theta_t),
            Rc_trans * jnp.cos(theta_t),
            Rc_trans * jnp.sin(beta_t),
            Rc_trans * jnp.cos(beta_t),
        ]
    )
    C_chine_t = jnp.linalg.solve(A_chine_t, b_chine_t)

    Rc_UG_int_t = C_chine_t[0:2]
    Rc_LG_int_t = C_chine_t[2:4]

    def sternrise(z: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.abs(z / STERNRISE)) + Lb + Lm + Ls * Kappa_STERN

    def delta_stern(z: jnp.ndarray) -> jnp.ndarray:
        return Lb + Lm + DELTA_STERN0 * z**2 + DELTA_STERN1 * z + DELTA_STERN2

    # ---------- Bulb geometry (needed by stern constraints and bulb constraints) ----------
    bb_on = bit_BB > 0.5
    sb_on = bit_SB > 0.5

    FP = bow_profile(WL)
    BB_Prof = jnp.array(
        [
            FP,
            (1.0 - Hbb) * WL,
            Hbb * WL,
            halfbeam_midbody(Hbb * WL) * Bbb,
            Lbb * LOA * (1.0 - Lbbm),
            FP - LOA * Lbb * Lbbm,
            0.0,
        ]
    )
    BB_Prof = jnp.where(bb_on, BB_Prof, jnp.zeros_like(BB_Prof))

    SBs = Kappa_SB * Ls + Lm + Lb
    SB_Prof = jnp.array(
        [
            SBs,
            (1.0 - Hsb) * WL * HSBOA,
            Hsb * WL * HSBOA,
            halfbeam_midbody(Hsb * WL * HSBOA) * Bsb,
            Lsb * LOA * (1.0 - Lsbm),
            SBs + LOA * Lsb * Lsbm,
            0.0,
        ]
    )
    SB_Prof = jnp.where(sb_on, SB_Prof, jnp.zeros_like(SB_Prof))

    def stern_profile(z: jnp.ndarray) -> jnp.ndarray:
        with_sb = (bit_SB > 0.5) & (z <= WL * HSBOA)
        return jnp.where(with_sb, SB_Prof[0], jnp.where(z <= SK1, sternrise(z), transom(z)))

    # ---------- Constraints ----------
    C_general = jnp.array([-LOA + Ls + Lb, WL - Dd])

    C_cross = jnp.array(
        [
            -Rc_UG_int[1] + Dc,
            -Rc,
            -Bc,
            -Dc,
            Rc_LG_int[0] - Bc,
            Rk_lg_int[0] - Rc_LG_int[0],
            1e-8 - jnp.abs(Rk),
        ]
    )

    drift_zv = jnp.where(jnp.abs(DRIFT0) < eps, -1.0, -DRIFT1 / (2.0 * DRIFT0))
    drift_in_range = (drift_zv >= 0.0) & (drift_zv <= Dd)
    vert_drift = jnp.where(
        drift_in_range,
        jnp.array([drift(drift_zv) - PI / 2.0, -drift(drift_zv)]),
        jnp.array([-1.0, -1.0]),
    )

    delta_bow_c_zv = jnp.where(jnp.abs(DELTA_BOW0) < eps, -1.0, -DELTA_BOW1 / (2.0 * DELTA_BOW0))
    delta_bow_c_in_range = (delta_bow_c_zv >= 0.0) & (delta_bow_c_zv <= Dd)
    vert_delta_bow = jnp.where(delta_bow_c_in_range, -delta_bow(delta_bow_c_zv) + bow_profile(delta_bow_c_zv), -1.0)

    bow_c_zv = jnp.where(jnp.abs(BOW0) < eps, -1.0, -BOW1 / (2.0 * BOW0))
    bow_c_in_range = (bow_c_zv >= 0.0) & (bow_c_zv <= Dd)
    vert_bow = jnp.where(bow_c_in_range, -delta_bow(bow_c_zv) + bow_profile(bow_c_zv), -1.0)

    C_bow = jnp.array(
        [
            Kappa_BOW * Lb - delta_bow(0.0),
            drift(0.0) - PI / 2.0,
            -drift(0.0),
            drift(Dd) - PI / 2.0,
            -drift(Dd),
            vert_drift[0],
            vert_drift[1],
            -BK0,
            BK0 - Kappa_BOW * Lb,
            -BK1,
            BK1 - Dd,
            -delta_bow(Dd) + bow_profile(Dd),
            -delta_bow(BK1) + BK0,
            vert_delta_bow,
            vert_bow,
        ]
    )

    delta_stern_c_zv = jnp.where(jnp.abs(DELTA_STERN0) < eps, -1.0, -DELTA_STERN1 / (2.0 * DELTA_STERN0))
    delta_stern_c_in_range = (delta_stern_c_zv >= 0.0) & (delta_stern_c_zv <= Dd)
    vert_delta_stern = jnp.where(
        delta_stern_c_in_range,
        delta_stern(delta_stern_c_zv) - stern_profile(delta_stern_c_zv),
        -1.0,
    )

    C_stern = jnp.array(
        [
            delta_stern(0.0) - (Lb + Lm + Ls * Kappa_STERN),
            delta_stern(SK1) - SK0,
            vert_delta_stern,
            delta_stern(Dd) - stern_profile(Dd),
            (Lb + Lm + Ls * Kappa_STERN) - SK0,
            Bc_trans - halfbeam_midbody(Dc_trans),
            -Rc_UG_int_t[1] + Dc_trans,
            -Rc_trans,
            -Bc_trans,
            -Dc_trans,
            Rc_LG_int_t[0] - Bc_trans,
            Rk_lg_int_t[0] - Rc_LG_int_t[0],
        ]
    )

    bb_base = jnp.array([-1.0, -1.0, BB_Prof[3] - halfbeam_midbody(BB_Prof[2])])
    bb_rk = jnp.array([BB_Prof[2] - Rk, BB_Prof[3] - Rk, BB_Prof[3] - halfbeam_midbody(BB_Prof[2])])
    bb_bad = jnp.array([1.0, 1.0, 1.0])
    C_bb_03 = jnp.where(Beta == 0.0, bb_base, jnp.where(Rk > 0.0, bb_rk, bb_bad))

    bb_dzv = jnp.where(jnp.abs(DELTA_BOW0) < eps, -1.0, -DELTA_BOW1 / (2.0 * DELTA_BOW0))
    bb_vert = jnp.where((bb_dzv >= 0.0) & (bb_dzv <= WL), delta_bow(bb_dzv) - BB_Prof[5], -1.0)
    C_bb_36 = jnp.array([BB_Prof[5] - delta_bow(0.0), BB_Prof[5] - delta_bow(WL), bb_vert])
    C_bb = jnp.where(bb_on, jnp.concatenate([C_bb_03, C_bb_36]), -jnp.ones((6,)))

    sb_base = jnp.array(
        [-1.0, -1.0, SB_Prof[3] - halfbeam_midbody(SB_Prof[2]), WL * HSBOA - SK1]
    )
    sb_rk = jnp.array(
        [
            SB_Prof[2] - Rk,
            SB_Prof[3] - Rk,
            SB_Prof[3] - halfbeam_midbody(SB_Prof[2]),
            WL * HSBOA - SK1,
        ]
    )
    sb_bad = jnp.array([1.0, 1.0, 1.0, 1.0])
    C_sb_610 = jnp.where(Beta == 0.0, sb_base, jnp.where(Rk > 0.0, sb_rk, sb_bad))

    sb_dzv = jnp.where(jnp.abs(DELTA_STERN0) < eps, -1.0, -DELTA_STERN1 / (2.0 * DELTA_STERN0))
    sb_vert = jnp.where((sb_dzv >= 0.0) & (sb_dzv <= WL * HSBOA), delta_stern(sb_dzv) - SB_Prof[5], -1.0)
    C_sb_1013 = jnp.array(
        [
            delta_stern(0.0) - SB_Prof[5],
            delta_stern(WL * HSBOA) - SB_Prof[5],
            sb_vert,
        ]
    )
    C_sb = jnp.where(sb_on, jnp.concatenate([C_sb_610, C_sb_1013]), -jnp.ones((7,)))

    C_bulb = jnp.concatenate([C_bb, C_sb])

    C = jnp.concatenate([C_general, C_cross, C_bow, C_stern, C_bulb])
    return C


def _jacobian_checks_once(x: jnp.ndarray, cfg: Mapping[str, Any] | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    f = lambda x_in: input_constraints_jax(x_in, cfg)
    return jax.jacrev(f)(x), jax.jacfwd(f)(x)

