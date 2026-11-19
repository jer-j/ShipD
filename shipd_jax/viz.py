#!/usr/bin/env python3
"""
Reusable matplotlib visualizations for shipd_jax.

Four families of plots:

1. **Resistance curves** -- Michell-vs-reference overlays, multi-draft Cw(Fn)
   curves with tank data, and dCw/dFn.
2. **Sensitivity diagnostics** -- gradient bar charts, constraint-Jacobian
   heatmaps, and KKT-matrix panels.
3. **3D hull rendering** -- plain surfaces, scalar-colored surfaces, waterline
   plan views, and Chamfer-distance comparison bars.
4. **Traditional lines drawings** -- profile/sheer plan (with buttock lines),
   half-breadth plan (waterlines), body plan (stations), and the composite
   three-view drawing.
5. **Shape sensitivity** -- surface deformation fields for a single design
   variable (`plot_param_sensitivity_3d`) or for the steepest-ascent
   direction of a scalar objective such as resistance or a constraint
   (`plot_objective_sensitivity_3d`).

None of these functions perform their own `jax.jit` compilation of the
expensive pipeline stages (`gen_pointcloud`, `gen_pc_for_cw`, `mod_michell`,
...); callers are expected to compile and reuse those themselves and pass in
plain arrays (or, for the sensitivity functions, a pre-built ``VolumePlan``
and a scalar objective closure) so that repeated calls -- e.g. across Froude
numbers or design variables -- do not pay JIT compile cost more than once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, SymLogNorm, to_rgb

from .constraints import constraints as _constraints_fn
from .hull import (
    Bits,
    Hull,
    VolumePlan,
    bb_profile,
    bits_from_vector,
    bow_profile,
    build_hull,
    gen_pc_for_cw,
    gen_pointcloud,
    sb_profile,
    stern_profile,
)
from .michell import G, mod_michell

__all__ = [
    "BIT_PARAM_INDICES",
    "CONSTRAINT_GROUPS",
    "PARAM_LABELS",
    "hessian_definiteness",
    "kkt_matrix",
    "make_constraint_objective",
    "make_resistance_objective",
    "plot_body_plan",
    "plot_chamfer_comparison",
    "plot_constraint_jacobian",
    "plot_dcw_dfn",
    "plot_halfbreadth_plan",
    "plot_hull_3d",
    "plot_hull_surface_scalar",
    "plot_kkt_matrices",
    "plot_lines_drawing",
    "plot_michell_validation",
    "plot_objective_sensitivity_3d",
    "plot_param_gradient_bars",
    "plot_param_sensitivity_3d",
    "plot_profile_plan",
    "plot_resistance_vs_froude",
    "plot_waterlines_plan",
    "surface_sensitivity_to_objective",
    "surface_sensitivity_to_param",
]

# Shared constants

#: Names of the 45 design variables, in vector order (matches shipd_jax.demo).
PARAM_LABELS = [
    "LOA",
    "Lb",
    "Ls",
    "Bd",
    "Dd",
    "Bs",
    "WL",
    "Bc",
    "Beta",
    "Rc",
    "Rk",
    "Abow",
    "Bbow",
    "BKz",
    "Kappa_bow",
    "Adel_b",
    "Bdel_b",
    "Adrft",
    "Bdrft",
    "Cdrft",
    "bit_EP_S",
    "bit_EP_T",
    "A_trans",
    "SKz",
    "Kappa_stern",
    "Adel_s",
    "Bdel_s",
    "Beta_trans",
    "Bc_trans",
    "Rc_trans",
    "Rk_trans",
    "bit_BB",
    "bit_SB",
    "Lbb",
    "Hbb",
    "Bbb",
    "Lbbm",
    "Rbb",
    "Kappa_SB",
    "Lsb",
    "HSBOA",
    "Hsb",
    "Bsb",
    "Lsbm",
    "Rsb",
]

#: Indices of the four discrete 0/1 structural switches (see hull.Bits) --
#: these have no meaningful derivative and are excluded from sensitivity plots.
BIT_PARAM_INDICES = {20, 21, 31, 32}

#: Row ranges of the 49 feasibility constraints, in the order
#: constraints.constraints() concatenates them.
CONSTRAINT_GROUPS = {
    "general": slice(0, 2),
    "cross_section": slice(2, 9),
    "bow": slice(9, 24),
    "stern": slice(24, 36),
    "bulb": slice(36, 49),
}

#: Density of fresh water [kg/m^3], the convention the Ship-D dataset uses
#: for its Michell wave drag.
RHO_FRESHWATER = 1000.0


# 1. Resistance curves


def plot_michell_validation(
    fn,
    cw_numeric,
    cw_reference=None,
    ref_label: str = "reference",
    numeric_label: str = "shipd_jax (differentiable)",
    ax=None,
    title: str | None = None,
    ylabel: str = r"$C_w = R_w/(\rho U^2 L^2)$",
):
    """Overlay a computed :math:`C_w(F_n)` curve against a reference curve.

    Parameters
    ----------
    fn, cw_numeric : 1D arrays of equal length -- the curve to validate.
    cw_reference : optional ``(fn_ref, cw_ref)`` pair (may sit on a different
        Froude grid; plotted as-is, no interpolation).
    """
    ax = ax or plt.gca()
    if cw_reference is not None:
        fr, cr = cw_reference
        ax.plot(fr, cr, "r--", lw=2, label=ref_label)
    ax.plot(fn, cw_numeric, "b-", lw=1.5, label=numeric_label)
    ax.set_xlabel("Froude number, $F_n$")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_resistance_vs_froude(
    fn,
    curves: Mapping[str, np.ndarray],
    tank_points: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    reference_curve: tuple[np.ndarray, np.ndarray] | None = None,
    reference_label: str = "reference curve",
    ax=None,
    title: str | None = None,
    ylabel: str = r"$C_w$",
    ylim=None,
):
    """Plot several :math:`C_w(F_n)` curves (e.g. different drafts) against
    optional towing-tank scatter points and a reference curve (Figure 7 of
    the Ship-D paper is exactly this plot).

    Parameters
    ----------
    curves : ``{label: cw_array}``, all evaluated on the same ``fn`` grid.
    tank_points : optional ``{label: (fn_array, cw_array)}``, drawn as markers.
    reference_curve : optional ``(fn_array, cw_array)``, drawn as a thick grey line.
    """
    ax = ax or plt.gca()
    if reference_curve is not None:
        ax.plot(
            reference_curve[0],
            reference_curve[1],
            "-",
            color="0.6",
            lw=3,
            alpha=0.8,
            label=reference_label,
        )
    if tank_points:
        markers = ["x", "+", "*", "o", "^", "s"]
        colors = [
            "tab:red",
            "tab:green",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
        ]
        for (label, (fx, fy)), mk, c in zip(
            tank_points.items(), markers, colors, strict=False
        ):
            ax.plot(fx, fy, mk, color=c, ms=5, label=label)
    styles = ["-", "-.", ":", "--"]
    for i, (label, cw) in enumerate(curves.items()):
        ax.plot(fn, cw, styles[i % len(styles)], color="tab:blue", lw=1.6, label=label)
    ax.set_xlabel("Froude number, $F_n$")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_dcw_dfn(
    fn, cw, dcw_dfn, ax=None, title: str | None = None, mark_extrema: bool = True
):
    """Dual-axis :math:`C_w(F_n)` and :math:`dC_w/dF_n` plot.

    With ``mark_extrema=True``, vertical guide lines are drawn at the sign
    changes of ``dcw_dfn`` -- the humps and hollows of the resistance curve.
    Returns ``(ax_left, ax_right)``.
    """
    ax1 = ax or plt.gca()
    ax1.plot(fn, cw, "b-", lw=1.8, label="$C_w$")
    ax1.set_xlabel("Froude number, $F_n$")
    ax1.set_ylabel("$C_w$", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(fn, dcw_dfn, "r-", lw=1.2, label="$dC_w/dF_n$ (autodiff)")
    ax2.axhline(0, color="r", lw=0.6, alpha=0.5)
    ax2.set_ylabel("$dC_w/dF_n$", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    if mark_extrema:
        fn_np = np.asarray(fn)
        dcw_np = np.asarray(dcw_dfn)
        sign_change = np.where(np.diff(np.sign(dcw_np)) != 0)[0]
        for k in sign_change:
            ax1.axvline(float(fn_np[k]), color="0.75", lw=0.7, zorder=0)
    if title:
        ax1.set_title(title)
    return ax1, ax2


# 2. Sensitivity diagnostics: gradients, constraint Jacobian, KKT matrices


def plot_param_gradient_bars(
    grads: Mapping[str, np.ndarray] | np.ndarray,
    param_labels: Sequence[str] = PARAM_LABELS,
    ax=None,
    ylabel: str | None = None,
    title: str | None = None,
    yscale: str = "symlog",
    linthresh: float = 1e-6,
):
    """Grouped bar chart of one or more 45-entry gradient vectors.

    ``grads`` may be a single array (wrapped as ``{'gradient': grads}``) or a
    ``{label: 45-vector}`` mapping, e.g. one entry per Froude number.
    """
    if not isinstance(grads, Mapping):
        grads = {"gradient": np.asarray(grads)}
    ax = ax or plt.gca()
    n = len(param_labels)
    ng = len(grads)
    w = 0.8 / max(ng, 1)
    for i, (label, g) in enumerate(grads.items()):
        g = np.asarray(g)
        ax.bar(np.arange(n) + (i - (ng - 1) / 2) * w, g, width=w, label=label)
    ax.set_xticks(range(n))
    ax.set_xticklabels(param_labels, rotation=90, fontsize=7)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh)
    elif yscale:
        ax.set_yscale(yscale)
    ax.set_ylabel(ylabel or r"$\partial f/\partial x_i$")
    if title:
        ax.set_title(title)
    if ng > 1:
        ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_constraint_jacobian(
    Jc,
    param_labels: Sequence[str] = PARAM_LABELS,
    ax=None,
    title: str | None = None,
    linthresh: float = 1e-6,
    cmap: str = "RdBu_r",
    show_groups: bool = True,
):
    """Heatmap of a ``(n_constraints, 45)`` constraint Jacobian."""
    Jc = np.asarray(Jc)
    ax = ax or plt.gca()
    vmax = np.abs(Jc).max()
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(
        Jc,
        aspect="auto",
        cmap=cmap,
        norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
    )
    ax.set_xticks(range(len(param_labels)))
    ax.set_xticklabels(param_labels, rotation=90, fontsize=7)
    ax.set_ylabel("constraint index")
    if show_groups and Jc.shape[0] == 49:
        for name, sl in CONSTRAINT_GROUPS.items():
            ax.axhline(sl.stop - 0.5, color="k", lw=0.5, alpha=0.4)
            ax.text(
                len(param_labels) + 0.5,
                (sl.start + sl.stop) / 2 - 0.5,
                name,
                fontsize=7,
                va="center",
            )
    ax.set_title(
        title or f"Constraint Jacobian ({Jc.shape[0]} x {Jc.shape[1]}, symlog color)"
    )
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax


def kkt_matrix(H, JA) -> np.ndarray:
    """Assemble ``[[H, JA^T], [JA, 0]]`` from a Hessian and an active-constraint
    Jacobian. Pass an empty ``JA`` (shape ``(0, n)``) for the unconstrained case.
    """
    H = np.asarray(H)
    JA = np.asarray(JA)
    n = H.shape[0]
    m = JA.shape[0]
    K = np.zeros((n + m, n + m))
    K[:n, :n] = H
    K[:n, n:] = JA.T
    K[n:, :n] = JA
    return K


def hessian_definiteness(H) -> tuple[np.ndarray, str]:
    """Eigenvalues of a symmetric Hessian and a one-word definiteness label."""
    ev = np.linalg.eigvalsh(np.asarray(H))
    if ev[0] >= -1e-12:
        kind = "positive semidefinite"
    elif ev[-1] <= 1e-12:
        kind = "negative semidefinite"
    else:
        kind = "indefinite"
    return ev, kind


def plot_kkt_matrices(
    hessians: Mapping[str, np.ndarray],
    JA,
    axes=None,
    linthresh: float = 1e-8,
    cmap: str = "RdBu_r",
    suptitle: str | None = None,
    figsize_per_panel=4.2,
):
    """Grid of KKT matrices ``[[H, JA^T], [JA, 0]]`` for several parameter
    values (e.g. Froude numbers), sharing one active-constraint Jacobian.

    ``hessians``: ``{label: (n, n) Hessian}``. Returns ``(fig, axes)``.
    """
    JA = np.asarray(JA)
    n = JA.shape[1]
    m = JA.shape[0]
    labels = list(hessians.keys())
    if axes is None:
        fig, axes = plt.subplots(
            1,
            len(labels),
            figsize=(figsize_per_panel * len(labels), figsize_per_panel + 0.4),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
    else:
        fig = axes[0].figure
    for ax, label in zip(axes, labels, strict=True):
        K = kkt_matrix(hessians[label], JA)
        vmax = np.abs(K).max()
        vmax = vmax if vmax > 0 else 1.0
        ax.imshow(
            K, cmap=cmap, norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
        )
        ax.axhline(n - 0.5, color="k", lw=0.8)
        ax.axvline(n - 0.5, color="k", lw=0.8)
        ax.set_title(str(label), fontsize=11)
        ax.set_xticks([n / 2, n + m / 2])
        ax.set_xticklabels(["$x$", r"$\lambda$"])
        ax.set_yticks([n / 2, n + m / 2])
        ax.set_yticklabels(["$x$", r"$\lambda$"])
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return fig, axes


# 3. 3D hull rendering


def _box_aspect(X, Y, Z, mirror):
    return (np.ptp(X), (2.0 if mirror else 1.0) * np.ptp(Y), 2.5 * np.ptp(Z))


def plot_hull_3d(
    pc,
    ax=None,
    color: str = "steelblue",
    alpha: float = 0.95,
    title: str | None = None,
    mirror: bool = True,
    view: tuple[float, float] = (22, -55),
    show_axes: bool = False,
):
    """Render a ``(num_wl, ppw, 3)`` surface point grid as a plain 3D hull."""
    pc = np.asarray(pc)
    if ax is None:
        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, antialiased=True)
    if mirror:
        ax.plot_surface(
            X, -Y, Z, color=color, alpha=alpha, linewidth=0, antialiased=True
        )
    ax.set_box_aspect(_box_aspect(X, Y, Z, mirror))
    ax.view_init(*view)
    if not show_axes:
        ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


# CAD-style shaded surface
# `plot_hull_3d` above is the quick look; `plot_hull_3d_cad` below draws the
# same geometry as one watertight, smoothly shaded surface for the manuscript
# figures. It draws *only* the surface -- the axes, its rulers, the station
# lines and the canvas sizing all belong to the caller.


def _closed_hull_grid(pc: np.ndarray, deck_rows: int = 3) -> np.ndarray:
    """``(num_wl, ppw, 3)`` half-hull grid -> closed watertight surface grid.

    The y>=0 half is mirrored into one continuous ring per waterline so that a
    *single* ``Poly3DCollection`` owns the whole skin: matplotlib depth-sorts
    faces within a collection but only whole artists against each other, so
    one ring renders its own self-occlusion correctly where two half-surfaces
    would fight. The deck edge is then swept inboard to the centreline,
    capping the top -- without it a view from above looks into an open shell.
    """
    pc = np.asarray(pc, float)
    stbd = pc[:, ::-1].copy()
    stbd[..., 1] *= -1.0
    ring = np.concatenate([pc, stbd[:, 1:-1], pc[:, :1]], axis=1)
    edge = ring[-1]
    t = np.linspace(1.0, 0.0, deck_rows + 1)[1:]
    deck = np.repeat(edge[None], len(t), axis=0)
    deck[..., 1] *= t[:, None]
    return np.concatenate([ring, deck], axis=0)


def _view_dir(elev: float, azim: float) -> np.ndarray:
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _cad_facecolors(
    P: np.ndarray,
    base,
    elev: float,
    azim: float,
    key: tuple[float, float] = (42.0, -34.0),
    fill: tuple[float, float] = (18.0, 115.0),
    ambient: float = 0.20,
    k_key: float = 0.72,
    k_fill: float = 0.16,
    k_spec: float = 0.16,
    shininess: float = 22.0,
    dark: float = 0.45,
    light: float = 0.58,
) -> np.ndarray:
    """Per-face RGB for a smooth two-light render of the surface grid ``P``.

    Vertex normals are averaged onto each quad, so the flat per-face colour
    that ``plot_surface`` draws still varies smoothly across the hull -- a
    Gouraud look without leaving matplotlib. Both lights are aimed relative to
    the camera (``key``/``fill`` are an elevation and an azimuth *offset*),
    which keeps the face turned toward the reader the lit one at any view.
    The key's elevation is taken *away from the horizon on the camera's own
    side*, so a view from below lights the underbody instead of leaving the
    whole visible face in shadow; the fill answers weakly from the other
    hemisphere.
    """
    tu, tv = np.gradient(P, axis=0), np.gradient(P, axis=1)
    n = np.cross(tu, tv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    outward = np.zeros_like(P)
    outward[..., 1] = P[..., 1]
    if np.einsum("ijk,ijk->", n, outward) < 0:
        n = -n
    nf = 0.25 * (n[:-1, :-1] + n[1:, :-1] + n[:-1, 1:] + n[1:, 1:])
    nf /= np.maximum(np.linalg.norm(nf, axis=-1, keepdims=True), 1e-12)

    def unit(v):
        return np.asarray(v, float) / np.linalg.norm(v)

    side = 1.0 if elev >= 0.0 else -1.0
    V = unit(_view_dir(elev, azim))
    L1 = unit(_view_dir(side * min(abs(elev) + key[0], 82.0), azim + key[1]))
    L2 = unit(_view_dir(-side * abs(fill[0]), azim + fill[1]))
    refl = 2.0 * (nf @ L1)[..., None] * nf - L1
    spec = k_spec * np.clip(refl @ V, 0, None) ** shininess
    inten = np.clip(
        ambient
        + k_key * np.clip(nf @ L1, 0, None)
        + k_fill * np.clip(nf @ L2, 0, None),
        0,
        1,
    )[..., None]

    # Shade between a darkened and a whitened version of the base colour
    # rather than by scaling it: scaling drives a mid grey to near-black in
    # shadow, which is exactly the initial hull's colour.
    rgb = np.asarray(to_rgb(base), float)
    c_dark, c_light = rgb * dark, rgb + (1.0 - rgb) * light
    return np.clip(c_dark + (c_light - c_dark) * inten + spec[..., None], 0, 1)


def plot_hull_3d_cad(
    pc,
    ax=None,
    color: str = "steelblue",
    rasterized: bool = True,
    zorder: int = 1,
    shade_kw: dict | None = None,
):
    """Draw ``pc`` as one smoothly shaded, watertight CAD-style hull surface.

    Only the surface is drawn: the caller owns the axes, its view, limits,
    box aspect, rulers and any station lines, so this composes with whatever
    layout a given figure needs. ``ax`` should already carry its final view --
    the lighting is aimed from ``ax.elev``/``ax.azim`` so the face turned
    toward the reader is the lit one -- and setting the view afterwards only
    changes where the light falls, not whether the render is valid.

    With ``ax=None`` the surface goes on the *current* figure, reusing its 3D
    axes or adding one. It deliberately never opens a figure of its own: a
    caller who has just sized a canvas by hand and then passes ``ax=None``
    would otherwise save that canvas empty while the hull went elsewhere.

    The surface is rasterized by default: it is ~20k quads, which as vector
    art dominates the file, while everything else on the axes stays vector.
    Save at ``dpi>=600`` so the shading does not soften in print.
    """
    pc = np.asarray(pc, float)
    if ax is None:
        fig = plt.gcf()
        ax = next((a for a in fig.axes if hasattr(a, "zaxis")), None)
        if ax is None:
            ax = fig.add_subplot(projection="3d")
            # Station lines and the skin are separate artists; mplot3d's own
            # depth sort compares them as wholes, so honour explicit zorder.
            ax.computed_zorder = False
    P = _closed_hull_grid(pc)
    fc = _cad_facecolors(P, color, float(ax.elev), float(ax.azim), **(shade_kw or {}))
    return ax.plot_surface(
        P[..., 0],
        P[..., 1],
        P[..., 2],
        facecolors=fc,
        shade=False,
        rcount=P.shape[0],
        ccount=P.shape[1],
        linewidth=0,
        antialiased=False,
        rasterized=rasterized,
        zorder=zorder,
    )


def station_x(pc, n: int = 10) -> np.ndarray:
    """The ``n + 1`` classic station abscissae, station 0 forward.

    ShipD point clouds run bow-to-stern in +x (``bow_profile`` bounds the low-x
    end of every waterline), so station 0 lands on the forward perpendicular
    and station ``n`` on the aft perpendicular, as the drawing convention
    requires. Both end stations are degenerate -- the hull has no breadth
    there -- which is why they coincide with the stem and stern profiles.
    """
    pc = np.asarray(pc, float)
    return np.linspace(float(pc[..., 0].min()), float(pc[..., 0].max()), n + 1)


def plot_hull_surface_scalar(
    pc,
    scalar_field,
    ax=None,
    cmap: str = "viridis",
    mirror: bool = True,
    title: str | None = None,
    view: tuple[float, float] = (22, -55),
    show_axes: bool = False,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    symlog: bool = False,
    linthresh: float = 1e-6,
):
    """Render a hull surface colored by a per-point scalar field.

    ``scalar_field`` must have the same leading ``(num_wl, ppw)`` shape as
    ``pc[..., 0]`` (e.g. a sensitivity magnitude or a signed component).
    Used internally by the sensitivity plotting functions below; also handy
    directly for e.g. local curvature or wetted-arc-length fields.
    """
    pc = np.asarray(pc)
    s = np.asarray(scalar_field)
    if ax is None:
        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
    lo = vmin if vmin is not None else float(np.nanmin(s))
    hi = vmax if vmax is not None else float(np.nanmax(s))
    if symlog:
        m = max(abs(lo), abs(hi), linthresh)
        norm = SymLogNorm(linthresh=linthresh, vmin=-m, vmax=m)
    else:
        if lo == hi:
            hi = lo + 1.0
        norm = Normalize(lo, hi)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(s))
    ax.plot_surface(
        X, Y, Z, facecolors=colors, linewidth=0, antialiased=True, shade=False
    )
    if mirror:
        ax.plot_surface(
            X, -Y, Z, facecolors=colors, linewidth=0, antialiased=True, shade=False
        )
    ax.set_box_aspect(_box_aspect(X, Y, Z, mirror))
    ax.view_init(*view)
    if not show_axes:
        ax.set_axis_off()
    if title:
        ax.set_title(title)
    if colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array(s)
        plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, label=colorbar_label)
    return ax


def plot_waterlines_plan(
    pc,
    ax=None,
    stride: int = 6,
    mirror: bool = True,
    title: str | None = None,
    color: str = "b",
    xlabel: str = "x [m]",
    ylabel: str = "y [m]",
):
    """Plan-view overlay of every ``stride``-th waterline in ``pc``."""
    pc = np.asarray(pc)
    ax = ax or plt.gca()
    for i in range(0, pc.shape[0], stride):
        ax.plot(pc[i, :, 0], pc[i, :, 1], "-", color=color, lw=0.6)
        if mirror:
            ax.plot(pc[i, :, 0], -pc[i, :, 1], "-", color=color, lw=0.6)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_chamfer_comparison(
    names: Sequence[str],
    rms_pct: Sequence[float],
    reference_pct: Sequence[float] | None = None,
    ax=None,
    label: str = "shipd_jax vs original",
    reference_label: str = "paper Table 1",
    ylabel: str = "RMS of Chamfer distance (% LOA)",
    title: str | None = None,
    log_scale: bool = True,
):
    """Bar chart of per-hull Chamfer-distance RMS (paper Eq. 1), optionally
    next to a reference set of values (e.g. the paper's own Table 1)."""
    ax = ax or plt.gca()
    x = np.arange(len(names))
    w = 0.35 if reference_pct is not None else 0.6
    ax.bar(
        x - (w / 2 if reference_pct is not None else 0), rms_pct, width=w, label=label
    )
    if reference_pct is not None:
        ax.bar(x + w / 2, reference_pct, width=w, label=reference_label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return ax


# 4. Traditional naval-architecture lines drawings
#
# These functions post-process a `gen_pointcloud` grid with plain numpy (no
# new differentiable geometry): each waterline row is x-monotonic from bow to
# stern by construction (see hull.py's `_wl_*` builders), which is what makes
# per-row interpolation for stations/buttocks well posed.


def _profile_extent(
    h: Hull, bits: Bits, z_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Outer (bow_x, stern_x) hull extent at each z, replicating the bulb-aware
    min/max logic ``gen_WLPoints`` uses to bound each waterline."""
    z_np = np.asarray(z_arr, dtype=np.float64)
    z_j = jnp.asarray(z_np)
    bow_x = np.asarray(bow_profile(h, z_j))
    if bits.BB:
        mask = z_np <= float(h.WL)
        bb_x = np.asarray(bb_profile(h, z_j))
        bow_x = np.where(mask, np.minimum(bb_x, bow_x), bow_x)
    stern_x = np.asarray(stern_profile(h, bits, z_j))
    if bits.SB:
        mask = z_np <= float(h.HSBOA * h.WL)
        sb_x = np.asarray(sb_profile(h, z_j))
        stern_x = np.where(mask, np.maximum(sb_x, stern_x), stern_x)
    return bow_x, stern_x


def _station_curve(pc: np.ndarray, x0: float) -> tuple[np.ndarray, np.ndarray]:
    """(z, y) half-beam values at longitudinal position ``x0``, one per
    waterline row that spans ``x0`` (rows are x-monotonic bow-to-stern)."""
    zs, ys = [], []
    for row in pc:
        xr = row[:, 0]
        if xr[0] <= x0 <= xr[-1]:
            zs.append(float(row[0, 2]))
            ys.append(float(np.interp(x0, xr, row[:, 1])))
    if not zs:
        return np.array([]), np.array([])
    order = np.argsort(zs)
    return np.asarray(zs)[order], np.asarray(ys)[order]


def _buttock_xz(pc: np.ndarray, y0: float):
    """Bow-branch and stern-branch (x, z) points of the y=y0 buttock line,
    found by inverting each waterline's monotonic bow/stern half-beam segment."""
    xs_b, zs_b, xs_s, zs_s = [], [], [], []
    for row in pc:
        x, y, z = row[:, 0], row[:, 1], float(row[0, 2])
        if len(x) < 3:
            continue
        i_first = int(np.argmax(y))
        i_last = len(y) - 1 - int(np.argmax(y[::-1]))
        bx, by = x[: i_first + 1], y[: i_first + 1]
        if len(bx) >= 2 and by[-1] > by[0]:
            order = np.argsort(by, kind="stable")
            byu, idx = np.unique(by[order], return_index=True)
            bxu = bx[order][idx]
            if byu[0] <= y0 <= byu[-1]:
                xs_b.append(float(np.interp(y0, byu, bxu)))
                zs_b.append(z)
        sx, sy = x[i_last:], y[i_last:]
        if len(sx) >= 2 and sy[0] > sy[-1]:
            order = np.argsort(sy, kind="stable")
            syu, idx = np.unique(sy[order], return_index=True)
            sxu = sx[order][idx]
            if syu[0] <= y0 <= syu[-1]:
                xs_s.append(float(np.interp(y0, syu, sxu)))
                zs_s.append(z)
    ob, os_ = np.argsort(zs_b), np.argsort(zs_s)
    return (
        np.asarray(xs_b)[ob],
        np.asarray(zs_b)[ob],
        np.asarray(xs_s)[os_],
        np.asarray(zs_s)[os_],
    )


def plot_profile_plan(
    x,
    vol_plan: VolumePlan,
    buttock_fracs: Sequence[float] = (0.25, 0.5, 0.75),
    pc: np.ndarray | None = None,
    n_profile: int = 400,
    ax=None,
    title: str | None = None,
    xlabel: str = "x [m]",
    ylabel: str = "z [m]",
):
    """Profile / sheer plan: the keel-stem-sternpost-deck outline plus buttock
    lines (constant lateral-offset longitudinal sections) -- the classic side
    view of a lines drawing.

    ``buttock_fracs`` are fractions of the maximum half-beam at which to draw
    buttock lines (pass ``()``/``None`` to draw only the outline).
    """
    x_np = np.asarray(x, dtype=np.float64)
    bits = bits_from_vector(x_np)
    h = build_hull(jnp.asarray(x_np), bits)
    Dd = float(h.Dd)
    z_arr = np.linspace(0.0, Dd, n_profile)
    bow_x, stern_x = _profile_extent(h, bits, z_arr)

    ax = ax or plt.gca()
    ax.plot(bow_x, z_arr, "k-", lw=1.6)
    ax.plot(stern_x, z_arr, "k-", lw=1.6)
    ax.plot([bow_x[-1], stern_x[-1]], [Dd, Dd], "k-", lw=1.6)
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")

    if buttock_fracs:
        if pc is None:
            pc = np.asarray(gen_pointcloud(jnp.asarray(x_np), vol_plan))
        ymax = float(np.max(pc[..., 1]))
        cmap = plt.get_cmap("cool")
        for i, frac in enumerate(buttock_fracs):
            y0 = frac * ymax
            xb, zb, xs, zs = _buttock_xz(pc, y0)
            c = cmap(i / max(len(buttock_fracs) - 1, 1))
            if len(xb):
                ax.plot(
                    xb,
                    zb,
                    "-",
                    color=c,
                    lw=1.0,
                    label=f"buttock {frac:.2f} $B_{{max}}$",
                )
            if len(xs):
                ax.plot(xs, zs, "-", color=c, lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Profile / Sheer Plan")
    if buttock_fracs:
        ax.legend(
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=len(buttock_fracs),
        )
    ax.grid(alpha=0.25)
    return ax


def plot_halfbreadth_plan(
    pc,
    waterline_fracs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    ax=None,
    title: str | None = None,
    xlabel: str = "x [m]",
    ylabel: str = "y [m]",
):
    """Half-breadth plan: waterlines at selected fractions of the draft,
    mirrored port/starboard -- the classic plan view of a lines drawing.

    ``pc``: a ``(num_wl, ppw, 3)`` grid from :func:`gen_pointcloud`; its own
    z-resolution determines how closely the requested fractions are matched
    (the nearest available waterline row is used for each).
    """
    pc = np.asarray(pc)
    z_all = pc[:, 0, 2]
    z_max = float(z_all[-1])
    ax = ax or plt.gca()
    cmap = plt.get_cmap("viridis")
    for i, frac in enumerate(waterline_fracs):
        row = int(np.argmin(np.abs(z_all - frac * z_max)))
        c = cmap(i / max(len(waterline_fracs) - 1, 1))
        ax.plot(
            pc[row, :, 0],
            pc[row, :, 1],
            "-",
            color=c,
            lw=1.1,
            label=f"WL {frac:.2f} (z={z_all[row]:.3f})",
        )
        ax.plot(pc[row, :, 0], -pc[row, :, 1], "-", color=c, lw=1.1)
    ax.axhline(0.0, color="0.6", lw=0.6, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Half-Breadth Plan")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    return ax


def plot_body_plan(
    x,
    vol_plan: VolumePlan,
    pc: np.ndarray | None = None,
    stations: Sequence[float] | None = None,
    n_stations: int = 10,
    ax=None,
    title: str | None = None,
    flip: bool = False,
    annotate: bool = True,
    xlabel: str = "half-breadth, y [m]",
    ylabel: str = "z [m]",
):
    """Body plan: hull cross-sections (stations), forward stations on the
    right of the centerline and aft stations on the left (traditional
    convention -- pass ``flip=True`` to mirror it).

    ``stations``: explicit x-positions; if omitted, ``n_stations`` positions
    evenly spaced across the hull's x-extent are used.
    """
    x_np = np.asarray(x, dtype=np.float64)
    bits = bits_from_vector(x_np)
    h = build_hull(jnp.asarray(x_np), bits)
    if pc is None:
        pc = np.asarray(gen_pointcloud(jnp.asarray(x_np), vol_plan))
    x_mid = float(h.Lb + h.Lm / 2.0)

    if stations is None:
        xmin, xmax = float(np.min(pc[..., 0])), float(np.max(pc[..., 0]))
        stations = np.linspace(xmin, xmax, n_stations + 2)[1:-1]
    stations = np.asarray(stations, dtype=float)

    ax = ax or plt.gca()
    cmap = plt.get_cmap("viridis")
    for i, xk in enumerate(stations):
        zc, yc = _station_curve(pc, float(xk))
        if len(zc) < 2:
            continue
        sign = (1.0 if xk < x_mid else -1.0) * (-1.0 if flip else 1.0)
        c = cmap(i / max(len(stations) - 1, 1))
        order = np.argsort(zc)
        ax.plot(sign * yc[order], zc[order], "-", color=c, lw=1.2)
        if annotate:
            j = int(np.argmax(yc))  # label at the point of maximum breadth
            stagger = 9 * (i % 3)  # vertical stagger so nearby labels don't collide
            ax.annotate(
                f"{xk:.2f}",
                (sign * yc[j], zc[j]),
                fontsize=6.5,
                color=c,
                ha="center",
                va="center",
                xytext=(10 * sign, stagger),
                textcoords="offset points",
            )
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fwd_side, aft_side = ("left", "right") if flip else ("right", "left")
    ax.set_title(
        title or f"Body Plan (forward stations {fwd_side}, aft stations {aft_side})"
    )
    ax.grid(alpha=0.25)
    return ax


def plot_lines_drawing(
    x,
    vol_plan: VolumePlan,
    pc: np.ndarray | None = None,
    n_stations: int = 8,
    waterline_fracs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    buttock_fracs: Sequence[float] = (0.25, 0.5, 0.75),
    figsize: tuple[float, float] = (13, 6),
    title: str | None = None,
):
    """Composite traditional three-view lines drawing: profile/sheer plan
    (with buttocks) top-left, half-breadth plan bottom-left, body plan on the
    right -- built from a single :func:`gen_pointcloud` evaluation shared by
    all three views. Returns ``(fig, (ax_profile, ax_halfbreadth, ax_body))``.
    """
    x_np = np.asarray(x, dtype=np.float64)
    if pc is None:
        pc = np.asarray(gen_pointcloud(jnp.asarray(x_np), vol_plan))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, width_ratios=(2.2, 1), height_ratios=(1, 1), hspace=0.35, wspace=0.25
    )
    ax_profile = fig.add_subplot(gs[0, 0])
    ax_half = fig.add_subplot(gs[1, 0], sharex=ax_profile)
    # Single row, not gs[:, 1]: plot_body_plan uses an equal aspect ratio, so
    # spanning both rows leaves the axes shrunk and centered in an oversized
    # cell -- dead space above/below it that no tight_layout/bbox_inches
    # setting removes, since it's inside the subplot's own reserved area.
    ax_body = fig.add_subplot(gs[0, 1])

    plot_profile_plan(x_np, vol_plan, buttock_fracs=buttock_fracs, pc=pc, ax=ax_profile)
    plot_halfbreadth_plan(pc, waterline_fracs=waterline_fracs, ax=ax_half)
    plot_body_plan(x_np, vol_plan, pc=pc, n_stations=n_stations, ax=ax_body)

    fig.suptitle(title or "Lines Drawing", fontsize=14)
    return fig, (ax_profile, ax_half, ax_body)


# 5. Shape sensitivity


def _resolve_param_index(
    param: str | int, param_labels: Sequence[str] = PARAM_LABELS
) -> int:
    if isinstance(param, str):
        if param not in param_labels:
            raise ValueError(
                f"unknown parameter name {param!r}; must be one of {list(param_labels)}"
            )
        idx = list(param_labels).index(param)
    else:
        idx = int(param)
        if not (0 <= idx < len(param_labels)):
            raise ValueError(
                f"parameter index {idx} out of range [0, {len(param_labels)})"
            )
    if idx in BIT_PARAM_INDICES:
        raise ValueError(
            f"parameter {param_labels[idx]!r} (index {idx}) is a discrete structural "
            "switch (0/1), not a continuous design variable, so it has no meaningful "
            "surface derivative. Choose a different parameter, or compare two plans "
            "built with the bit set to 0 and to 1 to see the effect of that topology "
            "choice directly."
        )
    return idx


def surface_sensitivity_to_param(
    x,
    vol_plan: VolumePlan,
    param: str | int,
    param_labels: Sequence[str] = PARAM_LABELS,
):
    """:math:`\\partial p/\\partial x_i` over the whole surface grid, via a
    single forward-mode ``jax.jvp``.

    Returns ``(pc, jac_field, magnitude, idx)``: ``pc`` and ``jac_field`` have
    shape ``(num_wl, ppw, 3)`` (surface points and their derivative), and
    ``magnitude`` is ``||jac_field||`` per point.
    """
    idx = _resolve_param_index(param, param_labels)
    x_j = jnp.asarray(np.asarray(x, dtype=np.float64))
    tangent = jnp.zeros_like(x_j).at[idx].set(1.0)
    pc, dpc = jax.jvp(lambda v: gen_pointcloud(v, vol_plan), (x_j,), (tangent,))
    dpc = np.asarray(dpc)
    magnitude = np.linalg.norm(dpc, axis=-1)
    return np.asarray(pc), dpc, magnitude, idx


def plot_param_sensitivity_3d(
    x,
    vol_plan: VolumePlan,
    param: str | int,
    param_labels: Sequence[str] = PARAM_LABELS,
    ax=None,
    component: str | None = None,
    cmap: str | None = None,
    title: str | None = None,
    **kwargs,
):
    """3D hull surface colored by its sensitivity to one design variable.

    ``param``: a name from ``PARAM_LABELS`` or an integer index; must not be
    one of the four bit switches (raises ``ValueError`` -- they are discrete).
    ``component``: ``None`` shows :math:`\\Vert\\partial p/\\partial x_i\\Vert`
    (sequential colormap); pass ``'x'``/``'y'``/``'z'`` to show a signed
    component instead (diverging colormap, symmetric about zero).
    Extra ``kwargs`` are forwarded to :func:`plot_hull_surface_scalar`.
    """
    pc, dpc, magnitude, idx = surface_sensitivity_to_param(
        x, vol_plan, param, param_labels
    )
    label = param if isinstance(param, str) else param_labels[idx]
    if component is None:
        field = magnitude
        use_cmap = cmap or "viridis"
        cbar_label = rf"$\Vert \partial p/\partial \mathrm{{{label}}}\Vert$"
    else:
        comp_idx = {"x": 0, "y": 1, "z": 2}[component]
        field = dpc[..., comp_idx]
        use_cmap = cmap or "RdBu_r"
        vmax = float(np.abs(field).max())
        kwargs.setdefault("vmin", -vmax)
        kwargs.setdefault("vmax", vmax)
        cbar_label = rf"$\partial {component}/\partial \mathrm{{{label}}}$"
    return plot_hull_surface_scalar(
        pc,
        field,
        ax=ax,
        cmap=use_cmap,
        title=title or f"Surface sensitivity to {label}",
        colorbar_label=cbar_label,
        **kwargs,
    )


def make_resistance_objective(
    cw_plan,
    fn: float,
    rho: float = RHO_FRESHWATER,
    mode: str = "Rw",
    michell_N: int = 21,
) -> Callable:
    """Build ``objective(x) -> scalar`` for :func:`surface_sensitivity_to_objective`
    / ``jax.grad``, evaluating wave resistance at a fixed Froude number and draft.

    ``mode``: ``'Rw'`` (Newtons) or ``'Cw'`` (:math:`R_w/(\\rho U^2 L_{OA}^2)`,
    the Ship-D paper's normalization).
    """
    if mode not in ("Rw", "Cw"):
        raise ValueError("mode must be 'Rw' or 'Cw'")

    def objective(xv):
        X, Z, Y, WL = gen_pc_for_cw(xv, cw_plan)
        U = fn * jnp.sqrt(G * WL)
        Rw = mod_michell(Y, U, X, Z, rho, michell_N)
        if mode == "Rw":
            return Rw
        return Rw / (rho * U**2 * xv[0] ** 2)

    objective.__name__ = f"{mode}(Fn={fn})"
    return objective


def make_constraint_objective(bits: Bits, k: int) -> Callable:
    """Build ``objective(x) -> constraints(x, bits)[k]``, a single feasibility
    constraint as a scalar callable for the sensitivity functions below."""

    def objective(xv):
        return _constraints_fn(xv, bits)[k]

    objective.__name__ = f"constraint[{k}]"
    return objective


def surface_sensitivity_to_objective(x, vol_plan: VolumePlan, objective_fn: Callable):
    """Steepest-ascent surface deformation for a scalar objective.

    Computes ``g = grad(objective_fn)(x)`` and the surface ``jvp`` along
    ``g/||g||`` -- i.e. how the hull surface would move under a gradient step
    that most increases ``objective_fn`` (built with :func:`make_resistance_objective`
    or :func:`make_constraint_objective`, or any other scalar function of ``x``).

    Returns ``(pc, jac_field, magnitude, grad_vector, objective_value)``.
    """
    x_j = jnp.asarray(np.asarray(x, dtype=np.float64))
    value, grad = jax.value_and_grad(objective_fn)(x_j)
    gnorm = jnp.linalg.norm(grad)
    direction = grad / jnp.where(gnorm > 0, gnorm, 1.0)
    pc, dpc = jax.jvp(lambda v: gen_pointcloud(v, vol_plan), (x_j,), (direction,))
    dpc = np.asarray(dpc)
    magnitude = np.linalg.norm(dpc, axis=-1)
    return np.asarray(pc), dpc, magnitude, np.asarray(grad), float(value)


def plot_objective_sensitivity_3d(
    x,
    vol_plan: VolumePlan,
    objective_fn: Callable,
    objective_name: str | None = None,
    ax=None,
    cmap: str = "magma",
    title: str | None = None,
    **kwargs,
):
    """3D hull surface colored by the steepest-ascent shape-sensitivity field
    of a scalar objective -- wave resistance, :math:`C_w`, or a single
    constraint (build ``objective_fn`` with :func:`make_resistance_objective`
    or :func:`make_constraint_objective`).

    Extra ``kwargs`` are forwarded to :func:`plot_hull_surface_scalar`.
    """
    pc, _dpc, magnitude, _grad, value = surface_sensitivity_to_objective(
        x, vol_plan, objective_fn
    )
    name = objective_name or getattr(objective_fn, "__name__", "objective")
    cbar_label = rf"$\Vert \partial p/\partial \hat g\Vert$  ({name})"
    default_title = f"Surface sensitivity to {name}  (value = {value:.4g})"
    return plot_hull_surface_scalar(
        pc,
        magnitude,
        ax=ax,
        cmap=cmap,
        title=title or default_title,
        colorbar_label=cbar_label,
        **kwargs,
    )
