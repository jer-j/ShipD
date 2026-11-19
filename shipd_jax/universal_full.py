#!/usr/bin/env python3
"""
Fully general plan-free pipeline: **all** hull types, no recompilation.

This supersedes the bulb-free-only :mod:`shipd_jax.universal`. The key
realisation is that the reference plan freezes two *different* kinds of thing,
and only one of them has to be static:

======================  ==========================================  ============
plan entry              what it is                                  must be static?
======================  ==========================================  ============
``pts_bow/mid/stern``   array **shapes**                            yes
``bb`` / ``sb``         "is the bulb present at this waterline?"    **no**
``is_ellipse``          "ellipse or parabola at this height?"       **no**
``sb_branch_a``         "which stern-fillet intersection branch?"   **no**
``bb_x0``               "which root of the fillet cubic?"           **no**
======================  ==========================================  ============

Fixing the shapes with a global scalar (``ppw`` split by fixed section
fractions) removes the only genuinely static entry.  Every remaining flag is a
*value* decision, so it can be expressed with ``jax.lax.cond`` / ``jnp.where``
and evaluated from the traced design vector -- including the flags that depend
on *combinations* of variables, which is exactly where per-axis bounds on the
frozen plan break down.

Two consequences:

* one compiled executable serves every design vector (per ``Bits`` setting --
  at most 16 variants, and the bits are structural switches, not design
  variables), and
* the branch decisions move *with* the design instead of being frozen at the
  point where the plan was built, so gradients stay correct arbitrarily far
  from that point.

``lax.cond`` rather than ``jnp.where`` is used for the bulb branches because
the untaken branch would evaluate bulb geometry outside its valid domain
(producing NaN that ``where`` propagates into reverse-mode gradients even
though it selects the right value forward).  ``cond`` executes only the taken
branch, so both the value and its gradient stay clean.

The one thing that *cannot* be made continuous is the discrete root/branch
selection itself: where the selected root changes, the geometry genuinely
jumps.  That is a property of the parameterization, not of this implementation
-- see :func:`branch_switch_report` for locating such points before handing the
model to an optimizer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .hull import (
    Bits,
    Hull,
    _cubic,
    _cubic_dydx,
    _linspace,
    _quartic,
    _safe,
    _sqrt_abs,
    _wl_bow_plain,
    _wl_mid,
    bb_dydx,
    bb_profile,
    bow_profile,
    build_hull,
    delta_bow,
    delta_stern,
    half_beam_bb,
    half_beam_midbody,
    half_beam_sb,
    half_beam_transom,
    sb_dydx,
    sb_profile,
    solve_waterline_bow,
    solve_waterline_stern,
    stern_profile,
)

__all__ = [
    "branch_switch_report",
    "cubic_root_real_parts",
    "section_counts",
    "universal_pc_for_cw",
    "universal_pointcloud",
    "universal_volume_properties",
]

_TINY = 1e-30
_BIG = 1e30


# Traced polynomial roots (replacing the plan's frozen root initializer)


def _cbrt(y):
    """Real cube root, sign preserving, with a finite (zero) gradient at 0.

    ``|y|**(1/3)`` has an infinite derivative at the origin; the double-where
    keeps that out of reverse-mode accumulation without changing any value.
    """
    ay = jnp.abs(y)
    nz = ay > 0.0
    return jnp.sign(y) * jnp.where(nz, jnp.where(nz, ay, 1.0) ** (1.0 / 3.0), 0.0)


def cubic_root_real_parts(c0, c1, c2, c3):
    """Real parts of the three roots of ``c0 x^3 + c1 x^2 + c2 x + c3``.

    Fully traced and differentiable (closed form -- no iteration, no
    ``np.roots``).  For a complex-conjugate pair, *both* returned entries carry
    the pair's shared real part.  That reproduces what the reference
    implementation effectively does: it filters ``np.roots`` output with
    NumPy's lexicographic complex ordering and then takes ``np.real``, so a
    complex root contributes its real part.

    Degenerate leading coefficient (``|c0|`` below tolerance) falls back to the
    quadratic roots, matching ``np.roots``' behaviour of returning fewer roots.
    """
    # quadratic fallback (|c0| ~ 0)
    qden = 2.0 * jnp.where(jnp.abs(c1) < _TINY, 1.0, c1)
    qdisc = c2 * c2 - 4.0 * c1 * c3
    qpos = qdisc > 0.0
    qsq = jnp.where(qpos, jnp.sqrt(jnp.where(qpos, qdisc, 1.0)), 0.0)
    qhalf = -c2 / qden
    qr1 = jnp.where(qpos, (-c2 + qsq) / qden, qhalf)
    qr2 = jnp.where(qpos, (-c2 - qsq) / qden, qhalf)

    # generic cubic
    # Every branch below is wrapped in the "double where" pattern: the argument
    # handed to sqrt / cbrt / arccos is replaced by a benign constant whenever
    # that branch is NOT the selected one. Without this, the *unselected*
    # branch still evaluates at a point where its derivative is infinite
    # (sqrt' and cbrt' at 0, arccos' at +-1), and reverse mode turns the
    # resulting inf * 0 into NaN even though the forward value is correct.
    c0s = jnp.where(jnp.abs(c0) < _TINY, 1.0, c0)
    a = c1 / c0s
    b = c2 / c0s
    c = c3 / c0s
    p = b - a * a / 3.0
    q = 2.0 * a**3 / 27.0 - a * b / 3.0 + c
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    use_trig = disc < 0.0
    use_card = ~use_trig

    # one real root (disc >= 0): real Cardano
    sq = jnp.where(use_card, jnp.sqrt(jnp.where(use_card, disc, 1.0)), 0.0)
    u_arg = jnp.where(use_card, -q / 2.0 + sq, 1.0)
    v_arg = jnp.where(use_card, -q / 2.0 - sq, 1.0)
    t_card = _cbrt(u_arg) + _cbrt(v_arg)

    # three real roots (disc < 0, which forces p < 0): trigonometric form
    p_neg = jnp.where(use_trig, jnp.minimum(p, -_TINY), -1.0)
    m = 2.0 * jnp.sqrt(-p_neg / 3.0)
    arg = jnp.where(
        use_trig, jnp.clip(3.0 * q / (p_neg * m), -1.0 + 1e-12, 1.0 - 1e-12), 0.0
    )
    t_trig = m * jnp.cos(jnp.arccos(arg) / 3.0)

    r1 = jnp.where(use_trig, t_trig, t_card) - a / 3.0

    # deflate:  x^3+ax^2+bx+c = (x - r1)(x^2 + (a+r1)x + (b + a r1 + r1^2))
    B = a + r1
    C = b + a * r1 + r1 * r1
    D2 = B * B - 4.0 * C
    half = -B / 2.0
    dpos = D2 > 0.0
    sq2 = jnp.where(dpos, jnp.sqrt(jnp.where(dpos, D2, 1.0)), 0.0) / 2.0
    r2 = half + sq2  # complex pair -> both collapse to `half`
    r3 = half - sq2

    is_quad = jnp.abs(c0) < _TINY
    return (
        jnp.where(is_quad, qr1, r1),
        jnp.where(is_quad, qr2, r2),
        jnp.where(is_quad, qr2, r3),
    )


def _min_root_at_least(roots, thr, fallback):
    """Smallest root whose real part is >= ``thr`` (reference's selection);
    ``fallback`` when none qualify. Uses a finite sentinel so reverse-mode
    gradients never see ``inf * 0``."""
    vals = [jnp.where(r >= thr, r, _BIG) for r in roots]
    m = jnp.minimum(jnp.minimum(vals[0], vals[1]), vals[2])
    return jnp.where(m >= 0.5 * _BIG, fallback, m)


def _max_root_at_least(r1, r2, thr, fallback):
    """Largest of two roots that are >= ``thr`` (stern-fillet selection)."""
    vals = [jnp.where(r >= thr, r, -_BIG) for r in (r1, r2)]
    m = jnp.maximum(vals[0], vals[1])
    return jnp.where(m <= -0.5 * _BIG, fallback, m)


# Bulb half-breadths evaluated over the FULL station array
#
# The reference evaluates half_beam_bb / half_beam_sb only on the bulb's own
# x-interval. The plan-free code evaluates them at every station and masks
# afterwards, which walks them far outside that interval. Where the bulb has
# (near) zero longitudinal extent -- Rx -> 0 at its vertical extremes -- the
# ratio ((x - Cx)/Rx)**2 then overflows to +inf, and although the caller masks
# the value away, reverse mode still carries a NaN back from it.
#
# Clamping the ratio fixes that. Inside the bulb's own range |u| <= 1, so the
# clamp is inactive and values are identical to the reference; outside, the
# value is discarded by the caller anyway.

_U_CLAMP = 1e6


def _half_beam_bb_safe(h: Hull, z, x):
    Rz = jnp.where(z >= h.BB_Prof[2], h.BB_Prof[1], h.BB_Prof[2])
    Ry = _sqrt_abs(1.0 - ((z - h.BB_Prof[2]) / _safe(Rz)) ** 2.0) * h.BB_Prof[3]
    Rx = h.BB_Prof[5] - bb_profile(h, z)
    u = jnp.clip((x - h.BB_Prof[5]) / _safe(Rx), -_U_CLAMP, _U_CLAMP)
    return jnp.where(x >= h.BB_Prof[5], Ry, _sqrt_abs(1.0 - u * u) * Ry)


def _half_beam_sb_safe(h: Hull, z, x):
    zc = h.Hsb * h.WL * h.HSBOA
    Rz = jnp.where(z >= zc, h.SB_Prof[1], h.SB_Prof[2])
    Ry = _sqrt_abs(1.0 - ((z - zc) / _safe(Rz)) ** 2.0) * h.SB_Prof[3]
    Rx = sb_profile(h, z) - h.SB_Prof[5]
    u = jnp.clip((x - h.SB_Prof[5]) / _safe(Rx), -_U_CLAMP, _U_CLAMP)
    return jnp.where(x <= h.SB_Prof[5], Ry, _sqrt_abs(1.0 - u * u) * Ry)


# Traced stern curve family (ellipse vs parabola chosen from a traced height)


def _stern_curve_t(parab, ell, use_ell, x):
    """Stern waterline half-beam; ``use_ell`` is a *traced* boolean."""
    y_par = parab[0] * x**2.0 + parab[1] * x + parab[2]
    y_ell = (
        _sqrt_abs(ell[1] ** 2.0 * (1.0 - ((x - ell[2]) / _safe(ell[0])) ** 2.0))
        + ell[3]
    )
    return jnp.where(use_ell, y_ell, y_par)


def _stern_curve_dydx_t(parab, ell, use_ell, x):
    d_par = 2.0 * parab[0] * x + parab[1]
    inner = _sqrt_abs(1.0 - ((x - ell[2]) / _safe(ell[0])) ** 2.0)
    d_ell = -ell[1] * (x - ell[2]) / _safe(ell[0] ** 2.0) / _safe(inner)
    return jnp.where(use_ell, d_ell, d_par)


def _use_ellipse(h: Hull, bits: Bits, z):
    """Traced ellipse/parabola flag at height ``z`` (bits stay static)."""
    if bits.EP_S == bits.EP_T:
        return jnp.asarray(bool(bits.EP_S))
    # families differ above/below the stern-keel intersect
    above = jnp.asarray(bool(bits.EP_T))
    below = jnp.asarray(bool(bits.EP_S))
    return jnp.where(z >= h.SK[1], above, below)


# Traced fillets


def _bb_fillet_t(h: Hull, z):
    """Bow-bulb fillet with the intersection root found in-trace."""
    prof = solve_waterline_bow(h, z)
    x1 = bb_profile(h, z)
    x2 = delta_bow(h, z)
    ybbC = half_beam_bb(h, z, h.BB_Prof[5])

    roots = cubic_root_real_parts(prof[0], prof[1], prof[2], prof[3] - ybbC)
    thr = jnp.maximum(bow_profile(h, z), bb_profile(h, z))
    x_int = _min_root_at_least(roots, thr, 0.5 * (x1 + x2))

    dxf = jnp.abs(jnp.minimum(x_int - x1, x2 - x_int))
    Xr0 = x_int - dxf * h.Rbb / 2.0
    Xr1 = x_int + dxf * h.Rbb / 2.0

    Y0 = half_beam_bb(h, z, Xr0)
    Y1 = _cubic(prof, Xr1)
    D0 = bb_dydx(h, z, Xr0)
    D1 = _cubic_dydx(prof, Xr1)

    zero, one = jnp.zeros(()), jnp.ones(())
    Arad = jnp.stack(
        [
            jnp.stack([Xr0**4.0, Xr0**3.0, Xr0**2.0, Xr0, one]),
            jnp.stack([Xr1**4.0, Xr1**3.0, Xr1**2.0, Xr1, one]),
            jnp.stack([4.0 * Xr0**3.0, 3.0 * Xr0**2.0, 2.0 * Xr0, one, zero]),
            jnp.stack([4.0 * Xr1**3.0, 3.0 * Xr1**2.0, 2.0 * Xr1, one, zero]),
            jnp.stack([12.0 * Xr0**2.0, 6.0 * Xr0**2.0, 2.0 * one, zero, zero]),
        ]
    )
    profrad = jnp.linalg.solve(Arad, jnp.stack([Y0, Y1, D0, D1, zero]))
    return prof, Xr0, Xr1, profrad


def _sb_fillet_t(h: Hull, bits: Bits, z):
    """Stern-bulb fillet with branch and curve family resolved in-trace."""
    parab, ell = solve_waterline_stern(h, bits, z)
    use_ell = _use_ellipse(h, bits, z)
    x1 = delta_stern(h, z)
    x2 = sb_profile(h, z)

    y_int = half_beam_sb(h, z, h.SB_Prof[5])
    y_at_x1 = _stern_curve_t(parab, ell, use_ell, x1)
    branch_a = y_int >= y_at_x1  # traced

    # branch B, ellipse form
    x_int_ell = (
        ell[0] * _sqrt_abs(1.0 - ((y_int - ell[3]) / _safe(ell[1])) ** 2.0) + ell[2]
    )
    # branch B, parabola form: largest root >= delta_stern
    aa, bb_, cc = parab[0], parab[1], parab[2] - y_int
    disc = bb_**2.0 - 4.0 * aa * cc
    pos = disc > 0.0
    sq = jnp.where(pos, jnp.sqrt(jnp.where(pos, disc, 1.0)), 0.0)
    den = 2.0 * _safe(aa)
    x_int_par = _max_root_at_least((-bb_ + sq) / den, (-bb_ - sq) / den, x1, x1)

    x_int = jnp.where(branch_a, h.SB_Prof[0], jnp.where(use_ell, x_int_ell, x_int_par))

    dxf = jnp.abs(jnp.minimum(x_int - x1, x2 - x_int))
    Xr0 = x_int - dxf * h.Rsb / 2.0
    Xr1 = x_int + dxf * h.Rsb / 2.0

    Y0 = _stern_curve_t(parab, ell, use_ell, Xr0)
    Y1 = half_beam_sb(h, z, Xr1)
    D0 = _stern_curve_dydx_t(parab, ell, use_ell, Xr0)
    D1 = sb_dydx(h, z, Xr1)

    zero, one = jnp.zeros(()), jnp.ones(())
    Arad = jnp.stack(
        [
            jnp.stack([Xr0**4.0, Xr0**3.0, Xr0**2.0, Xr0, one]),
            jnp.stack([Xr1**4.0, Xr1**3.0, Xr1**2.0, Xr1, one]),
            jnp.stack([4.0 * Xr0**3.0, 3.0 * Xr0**2.0, 2.0 * Xr0, one, zero]),
            jnp.stack([4.0 * Xr1**3.0, 3.0 * Xr1**2.0, 2.0 * Xr1, one, zero]),
            jnp.stack([12.0 * Xr1**2.0, 6.0 * Xr1**2.0, 2.0 * one, zero, zero]),
        ]
    )
    profrad = jnp.linalg.solve(Arad, jnp.stack([Y0, Y1, D0, D1, zero]))
    return parab, ell, use_ell, Xr0, Xr1, profrad


# Waterline sections (fixed shapes, traced branches)


def _wl_bow_bb_t(h: Hull, z, n: int) -> jax.Array:
    x1 = bb_profile(h, z)
    x2 = delta_bow(h, z)
    x = _linspace(x1, x2, n)
    prof, Xr0, Xr1, profrad = _bb_fillet_t(h, z)
    y = jnp.where(
        x <= Xr0,
        _half_beam_bb_safe(h, z, x),
        jnp.where(x < Xr1, _quartic(profrad, x), _cubic(prof, x)),
    )
    return jnp.stack([x, y.at[0].set(0.0)], axis=1)


def _wl_stern_plain_t(h: Hull, bits: Bits, z, n: int) -> jax.Array:
    x1 = delta_stern(h, z)
    x2 = stern_profile(h, bits, z)
    x = _linspace(x1, x2, n)
    parab, ell = solve_waterline_stern(h, bits, z)
    y = _stern_curve_t(parab, ell, _use_ellipse(h, bits, z), x)
    tx, ty = half_beam_transom(h, bits, z)
    return jnp.stack(
        [
            jnp.concatenate([x[: n - 1], tx[None]]),
            jnp.concatenate([y[: n - 1], ty[None]]),
        ],
        axis=1,
    )


def _wl_stern_sb_t(h: Hull, bits: Bits, z, n: int) -> jax.Array:
    x1 = delta_stern(h, z)
    x2 = sb_profile(h, z)
    x = _linspace(x1, x2, n)
    parab, ell, use_ell, Xr0, Xr1, profrad = _sb_fillet_t(h, bits, z)
    y = jnp.where(
        x <= Xr0,
        _stern_curve_t(parab, ell, use_ell, x),
        jnp.where(x < Xr1, _quartic(profrad, x), _half_beam_sb_safe(h, z, x)),
    )
    return jnp.stack(
        [
            jnp.concatenate([x[: n - 1], x2[None]]),
            jnp.concatenate([y[: n - 1], jnp.zeros((1,))]),
        ],
        axis=1,
    )


def _wl_universal(h: Hull, bits: Bits, z, n_bow: int, n_mid: int, n_stern: int):
    """One waterline, all branches traced, shapes fixed.

    ``lax.cond`` (not ``jnp.where``) guards the bulb sections: the untaken
    branch would evaluate bulb geometry outside its domain, and ``where``
    would carry the resulting NaN into reverse-mode gradients.
    """
    if bits.BB:
        bow = lax.cond(
            z < h.WL,
            lambda zz: _wl_bow_bb_t(h, zz, n_bow),
            lambda zz: _wl_bow_plain(h, zz, n_bow),
            z,
        )
    else:
        bow = _wl_bow_plain(h, z, n_bow)

    mid = _wl_mid(h, z, n_mid)

    if bits.SB:
        stern = lax.cond(
            z < h.WL * h.HSBOA,
            lambda zz: _wl_stern_sb_t(h, bits, zz, n_stern),
            lambda zz: _wl_stern_plain_t(h, bits, zz, n_stern),
            z,
        )
    else:
        stern = _wl_stern_plain_t(h, bits, z, n_stern)

    return jnp.concatenate([bow, mid, stern], axis=0)


# Public API


def section_counts(
    ppw: int, bow_frac: float = 0.34, stern_frac: float = 0.33
) -> tuple[int, int, int]:
    """Fixed per-section point counts derived from the global scalar ``ppw``."""
    n_bow = max(2, round(bow_frac * ppw))
    n_stern = max(2, round(stern_frac * ppw))
    n_mid = ppw - n_bow - n_stern
    if n_mid < 0:
        raise ValueError("bow_frac + stern_frac leaves no midbody points")
    return n_bow, n_mid, n_stern


def universal_pointcloud(
    x,
    bits: Bits,
    num_wl: int = 51,
    ppw: int = 300,
    bow_frac: float = 0.34,
    stern_frac: float = 0.33,
):
    """``(num_wl, ppw, 3)`` surface grid for **any** hull, with no plan.

    ``bits`` stays static (they are structural switches, not design variables);
    everything else is traced, so one compiled executable serves every design
    vector with those bits.
    """
    n_bow, n_mid, n_stern = section_counts(ppw, bow_frac, stern_frac)
    h = build_hull(x, bits)
    Z = _linspace(jnp.asarray(0.00001), h.Dd, num_wl)
    rows = []
    for i in range(num_wl):
        z = Z[i]
        xy = _wl_universal(h, bits, z, n_bow, n_mid, n_stern)
        rows.append(jnp.concatenate([xy, jnp.broadcast_to(z, (ppw,))[:, None]], axis=1))
    return jnp.stack(rows, axis=0)


def universal_pc_for_cw(x, draft_frac, bits: Bits, num_wl: int = 51, ppw: int = 301):
    """``(X, Z, Y, WL)`` Michell inputs for **any** hull, plan-free.

    ``draft_frac`` is traced, so the draft is a differentiable input and one
    compile covers every (design, draft) pair.
    """
    h = build_hull(x, bits)
    draft = h.Dd * jnp.asarray(draft_frac, dtype=jnp.float64)
    Z = _linspace(0.00000001 * h.Dd, draft, num_wl)

    x_bow = bow_profile(h, Z)
    if bits.BB:
        x_bow = jnp.where(Z <= h.WL, bb_profile(h, Z), x_bow)
    x_stern = stern_profile(h, bits, Z)
    if bits.SB:
        x_stern = jnp.where(Z <= h.HSBOA * h.WL, sb_profile(h, Z), x_stern)

    WLlen = x_stern[-1] - x_bow[-1]
    Xg = _linspace(jnp.min(x_bow), jnp.max(x_stern), ppw)

    def y_bow_bulb(zz):
        prof, Xr0, Xr1, profrad = _bb_fillet_t(h, zz)
        return jnp.where(
            Xg <= Xr0,
            _half_beam_bb_safe(h, zz, Xg),
            jnp.where(Xg < Xr1, _quartic(profrad, Xg), _cubic(prof, Xg)),
        )

    def y_bow_plain(zz):
        return _cubic(solve_waterline_bow(h, zz), Xg)

    def y_stern_bulb(zz):
        parab, ell, use_ell, Xr0, Xr1, profrad = _sb_fillet_t(h, bits, zz)
        return jnp.where(
            Xg <= Xr0,
            _stern_curve_t(parab, ell, use_ell, Xg),
            jnp.where(Xg < Xr1, _quartic(profrad, Xg), _half_beam_sb_safe(h, zz, Xg)),
        )

    def y_stern_plain(zz):
        parab, ell = solve_waterline_stern(h, bits, zz)
        return _stern_curve_t(parab, ell, _use_ellipse(h, bits, zz), Xg)

    cols = []
    for i in range(num_wl):
        z = Z[i]
        y_bow = (
            lax.cond(z < h.WL, y_bow_bulb, y_bow_plain, z)
            if bits.BB
            else y_bow_plain(z)
        )
        y_mid = jnp.broadcast_to(half_beam_midbody(h, z), Xg.shape)
        y_stern = (
            lax.cond(z < h.WL * h.HSBOA, y_stern_bulb, y_stern_plain, z)
            if bits.SB
            else y_stern_plain(z)
        )

        y = jnp.where(
            Xg <= delta_bow(h, z),
            y_bow,
            jnp.where(Xg < delta_stern(h, z), y_mid, y_stern),
        )
        cols.append(jnp.where((Xg > x_bow[i]) & (Xg < x_stern[i]), y, 0.0))

    return Xg - Xg[0], Z - Z[-1], jnp.stack(cols, axis=1), WLlen


def universal_volume_properties(
    x,
    bits: Bits,
    num_wl: int = 51,
    ppw: int = 300,
    bow_frac: float = 0.34,
    stern_frac: float = 0.33,
):
    """Hydrostatics from the plan-free grid, using the same trapezoidal
    formulas as :func:`shipd_jax.volume_properties` (which requires a plan)."""
    pc = universal_pointcloud(x, bits, num_wl, ppw, bow_frac, stern_frac)
    return _vol_props_from_pc(x, bits, pc)


def _vol_props_from_pc(x, bits: Bits, pc):
    """Hydrostatic integrals computed directly from a surface grid."""
    h = build_hull(x, bits)
    Z = pc[:, 0, 2]
    X, Y = pc[:, :, 0], pc[:, :, 1]
    dx = X[:, 1:] - X[:, :-1]
    a = 2.0 * Y[:, 1:]
    b = 2.0 * Y[:, :-1]
    areas = 2.0 * jnp.sum(0.5 * (Y[:, 1:] + Y[:, :-1]) * dx, axis=1)
    vol0 = 0.5 * areas[0] * Z[0]
    dvol = 0.5 * (areas[1:] + areas[:-1]) * (Z[1:] - Z[:-1])
    volumes = vol0 + jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(dvol)])
    ab = a + b
    cx = X[:, :-1] + dx / 3.0 * (2.0 * a + b) / _safe(ab)
    lcf = jnp.sum(cx * 0.5 * ab * dx, axis=1) / areas
    dy = Y[:, 1:] - Y[:, :-1]
    seg2 = dx**2.0 + dy**2.0
    arc = jnp.sum(
        jnp.where(seg2 > 0, jnp.sqrt(jnp.where(seg2 > 0, seg2, 1.0)), 0.0), axis=1
    )
    wsa0 = 2.0 * areas[0]
    dwsa = (arc[1:] + arc[:-1]) * (Z[1:] - Z[:-1])
    area_ws = wsa0 + jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(dwsa)])
    return dict(
        Z=Z,
        pointcloud=pc,
        areas_wp=areas,
        volumes=volumes,
        lcf=lcf,
        area_ws=area_ws,
        wl_lengths=X[:, -1] - X[:, 0],
        hull=h,
    )


# Locating the genuine (model-level) branch switches


def branch_switch_report(
    x_np, bits: Bits, num_wl: int, ppw: int, direction, t_max: float, n: int = 201
):
    """Scan ``x0 + t * direction`` for ``t`` in ``[0, t_max]`` and report where
    the *discrete* branch decisions flip.

    These are points where the geometry itself is non-smooth -- unlike a stale
    plan (an implementation artefact), they are real and an optimizer should be
    kept away from them.  Returns ``(t_values, n_flips_per_step)``.
    """
    x_np = np.asarray(x_np, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    ts = np.linspace(0.0, t_max, n)
    sigs = []
    for t in ts:
        xv = x_np + t * direction
        h = build_hull(jnp.asarray(xv), bits)
        Dd = float(h.Dd)
        Z = np.linspace(0.00001, Dd, num_wl)
        s = []
        for z in Z:
            zz = float(z)
            s.append(
                (
                    bool(zz < float(h.WL)) if bits.BB else False,
                    bool(zz < float(h.WL * h.HSBOA)) if bits.SB else False,
                    bool(_use_ellipse(h, bits, jnp.asarray(zz))),
                )
            )
        sigs.append(tuple(s))
    flips = [0] + [
        sum(1 for p, q in zip(sigs[i - 1], sigs[i], strict=True) if p != q)
        for i in range(1, len(sigs))
    ]
    return ts, np.array(flips)
