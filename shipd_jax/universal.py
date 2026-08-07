#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan-free ("universal") evaluation for bulb-free Ship-D hulls.

The reference-exact pipeline in :mod:`shipd_jax.hull` freezes discrete
"topology" (per-waterline integer point counts, curve-family flags, bulb
root initializers) into a plan built at one concrete design vector.  That
reproduces the original implementation bit-for-bit, but the plan goes stale
as an optimizer moves the design point, and rebuilding it bakes new Python
constants into the JIT closure -- forcing an XLA recompile.

For hulls **without bulbs** (``bit_BB = bit_SB = 0``) and with equal stern
curve-family bits (``bit_EP_S == bit_EP_T``, so the ellipse/parabola choice
does not depend on a traced quantity), *none* of the remaining topology needs
to depend on the design vector:

* the integer point allocation can simply be fixed (uniform plan), and
* the Cw offset-grid mask can be computed inside the trace
  (``jnp.where``), like every other value-level branch already is.

Everything in this module is a pure function of traced inputs, so a single
``jax.jit`` compilation serves *every* design vector (and, for
:func:`gen_pc_for_cw_universal`, every draft -- the draft fraction is a
traced argument and therefore differentiable).

The cost is that sample locations differ slightly from the reference's
adaptive allocation: integrated quantities (volume, wetted surface, Cw)
agree to discretization error, which shrinks with resolution -- quantified
in the accompanying analysis notebook.

Why not ``jax.lax.cond``?  ``cond``/``where`` handle *value* branches (and
this code, like hull.py, already uses the batched equivalent ``jnp.where``
throughout); neither can make an array's *shape* depend on a traced value,
and shapes are exactly what the reference plan varies.  Fixing the shapes,
as done here, is the rewrite that removes the recompile -- see the notebook
for a demonstration.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .hull import (Bits, VolumePlan, WLPlan, _ROOT_REAL, _linspace,
                   bits_from_vector, build_hull, gen_pointcloud,
                   bow_profile, stern_profile, delta_bow, delta_stern,
                   half_beam_midbody, solve_waterline_bow,
                   solve_waterline_stern, _cubic, _stern_curve, _wl_topology)

__all__ = [
    "uniform_volume_plan", "gen_pc_for_cw_universal",
    "plan_signature", "topology_safe_bounds", "check_box_topology",
]


def _require_universal(bits: Bits) -> None:
    if bits.BB or bits.SB:
        raise ValueError(
            "the universal (plan-free) pipeline covers bulb-free hulls only "
            f"(bit_BB={int(bits.BB)}, bit_SB={int(bits.SB)}); bulb hulls need "
            "the plan's root initializers and fillet branch flags")
    if bits.EP_S != bits.EP_T:
        raise ValueError(
            "bit_EP_S != bit_EP_T makes the stern curve family switch at the "
            "traced height SK_z, which the static per-waterline flag cannot "
            "represent; the universal pipeline requires bit_EP_S == bit_EP_T")


# =============================================================================
# Universal surface grid: a VolumePlan with x-independent topology
# =============================================================================

def uniform_volume_plan(bits: Bits, num_wl: int = 51, ppw: int = 300,
                        bow_frac: float = 0.30, stern_frac: float = 0.30) -> VolumePlan:
    """A :class:`VolumePlan` whose per-waterline point counts are fixed
    fractions of ``ppw`` instead of the reference's design-dependent integer
    allocation.

    Valid for **any** bulb-free design vector (with ``EP_S == EP_T``), so a
    function jitted against it never needs recompilation as the design
    changes.  Sample *locations* still adapt continuously to the geometry
    (each section is a linspace between traced boundaries); only the *counts*
    are frozen.
    """
    _require_universal(bits)
    pts_bow = max(2, int(round(bow_frac * ppw)))
    pts_stern = max(2, int(round(stern_frac * ppw)))
    pts_mid = ppw - pts_bow - pts_stern
    if pts_mid < 0:
        raise ValueError("bow_frac + stern_frac must leave room for the midbody")
    wl = WLPlan(pts_bow=pts_bow, pts_mid=pts_mid, pts_stern=pts_stern,
                bb=False, sb=False, is_ellipse=bool(bits.EP_S),
                sb_branch_a=False, bb_x0=(_ROOT_REAL, 0.0))
    return VolumePlan(bits=bits, num_wl=num_wl, ppw=ppw, wls=(wl,) * num_wl)


# =============================================================================
# Universal Michell offset grid: traced masks, traced draft
# =============================================================================

def gen_pc_for_cw_universal(x, draft_frac, bits: Bits,
                            num_wl: int = 51, ppw: int = 301):
    """Plan-free equivalent of :func:`shipd_jax.gen_pc_for_cw` for bulb-free
    hulls: ``(X, Z, Y, WL)`` for the Michell integral, as a pure traced
    function of ``x`` **and** ``draft_frac``.

    Because the draft is a traced scalar, ``jax.grad`` w.r.t. the draft (and
    any design variable) is available, and one compilation covers every
    (design, draft) pair.  Matches the reference-exact generator up to the
    measure-zero grid/boundary coincidences that the frozen mask replicates.
    """
    _require_universal(bits)
    is_ellipse = bool(bits.EP_S)

    h = build_hull(x, bits)
    draft = h.Dd * jnp.asarray(draft_frac, dtype=jnp.float64)
    Z = _linspace(0.00000001 * h.Dd, draft, num_wl)

    x_bow = bow_profile(h, Z)
    x_stern = stern_profile(h, bits, Z)
    WLlen = x_stern[-1] - x_bow[-1]
    Xg = _linspace(jnp.min(x_bow), jnp.max(x_stern), ppw)

    cols = []
    for i in range(num_wl):
        z = Z[i]
        prof = solve_waterline_bow(h, z)
        y_bow = _cubic(prof, Xg)
        y_mid = jnp.broadcast_to(half_beam_midbody(h, z), Xg.shape)
        parab, ell = solve_waterline_stern(h, bits, z)
        y_stern = _stern_curve(parab, ell, is_ellipse, Xg)
        y = jnp.where(Xg <= delta_bow(h, z), y_bow,
                      jnp.where(Xg < delta_stern(h, z), y_mid, y_stern))
        mask = (Xg > x_bow[i]) & (Xg < x_stern[i])     # traced, not frozen
        cols.append(jnp.where(mask, y, 0.0))
    Y = jnp.stack(cols, axis=1)     # (ppw, num_wl)

    return Xg - Xg[0], Z - Z[-1], Y, WLlen


# =============================================================================
# Topology-safe bounds around a design point (for the reference-exact plan)
# =============================================================================

def plan_signature(x_np, bits: Bits, num_wl: int, ppw: int) -> tuple:
    """The discrete topology the reference-exact plan would freeze at ``x_np``:
    per-waterline (counts, branch flags), hashable for equality tests."""
    h = build_hull(jnp.asarray(np.asarray(x_np, dtype=np.float64)), bits)
    Dd = float(h.Dd)
    Z = np.linspace(0.00001, Dd, num_wl)
    sig = []
    for z in Z:
        try:
            wl = _wl_topology(h, bits, float(z), ppw)
            sig.append((wl.pts_bow, wl.pts_mid, wl.pts_stern,
                        wl.bb, wl.sb, wl.is_ellipse, wl.sb_branch_a))
        except Exception as e:      # degenerate allocation etc.
            sig.append(("FAIL", type(e).__name__))
    return tuple(sig)


def topology_safe_bounds(x0_np, bits: Bits, num_wl: int, ppw: int,
                         search_lo: Sequence[float], search_hi: Sequence[float],
                         skip: Sequence[int] = (20, 21, 31, 32),
                         n_bisect: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Per-variable interval around ``x0`` inside which the frozen plan's
    topology provably matches a freshly built plan **along that axis**.

    For each design variable independently, bisect for the largest one-sided
    perturbation (within ``[search_lo, search_hi]``) that leaves
    :func:`plan_signature` unchanged.  Returns ``(lo, hi)`` arrays (absolute
    variable values).  The true topology-safe region is not a box, so the
    per-axis box is an *outer* estimate along the axes -- use
    :func:`check_box_topology` to measure how much of the box interior is
    actually safe.
    """
    x0_np = np.asarray(x0_np, dtype=np.float64)
    sig0 = plan_signature(x0_np, bits, num_wl, ppw)
    lo = x0_np.copy()
    hi = x0_np.copy()

    def same_sig(xv):
        return plan_signature(xv, bits, num_wl, ppw) == sig0

    for i in range(len(x0_np)):
        if i in skip:
            lo[i], hi[i] = search_lo[i], search_hi[i]   # unused by the graph
            continue
        for sign, limit in ((+1, search_hi[i]), (-1, search_lo[i])):
            span = limit - x0_np[i]
            if sign * span <= 0:
                continue
            xv = x0_np.copy()
            xv[i] = limit
            if same_sig(xv):
                good = span              # the whole range is safe
            else:
                good, bad = 0.0, span
                for _ in range(n_bisect):
                    mid = 0.5 * (good + bad)
                    xv[i] = x0_np[i] + mid
                    if same_sig(xv):
                        good = mid
                    else:
                        bad = mid
            if sign > 0:
                hi[i] = x0_np[i] + good
            else:
                lo[i] = x0_np[i] + good
    return lo, hi


def check_box_topology(x0_np, bits: Bits, num_wl: int, ppw: int,
                       lo: np.ndarray, hi: np.ndarray, n_samples: int = 200,
                       seed: int = 0, shrink: float = 1.0) -> float:
    """Fraction of uniform random samples in ``shrink * [lo, hi]`` (scaled
    about ``x0``) whose plan topology matches ``x0``'s."""
    rng = np.random.default_rng(seed)
    x0_np = np.asarray(x0_np, dtype=np.float64)
    sig0 = plan_signature(x0_np, bits, num_wl, ppw)
    lo_s = x0_np + shrink * (np.asarray(lo) - x0_np)
    hi_s = x0_np + shrink * (np.asarray(hi) - x0_np)
    ok = 0
    for _ in range(n_samples):
        xv = rng.uniform(lo_s, hi_s)
        for k in (20, 21, 31, 32):
            xv[k] = x0_np[k]
        if plan_signature(xv, bits, num_wl, ppw) == sig0:
            ok += 1
    return ok / n_samples
