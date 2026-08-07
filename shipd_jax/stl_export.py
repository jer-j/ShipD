#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL export replicating ``Hull_Parameterization.gen_stl``.

The mesh-grid point lists and the triangulation are assembled concretely
(numpy indices), while every coordinate comes from the same geometry
functions as the differentiable code, so the resulting mesh matches the
reference ``gen_stl`` output.  STL writing itself is inherently a
non-differentiable I/O step; use :func:`gen_stl_trivec` if you want the
triangle vertex array as a numpy array without writing a file.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .hull import (Bits, Hull, bits_from_vector, build_hull, _linspace,
                   _bb_fillet, _sb_fillet, _bb_root_init, _sb_branch,
                   _cubic, _quartic,
                   _stern_curve, solve_waterline_bow, solve_waterline_stern,
                   half_beam_midbody, half_beam_transom, half_beam_bb,
                   half_beam_sb, bow_profile, bb_profile, sb_profile,
                   stern_profile, delta_bow, delta_stern)


def _wl_grid_points(h: Hull, bits: Bits, X: np.ndarray, z: float) -> np.ndarray:
    """Concrete replication of ``gen_MeshGridWL`` for one waterline."""
    WL = []

    if z == 0.0:
        bow_start = float(h.BB_Prof[5]) if bits.BB else float(h.Kappa_BOW * h.Lb)
        if bits.SB:
            stern_end = float(h.SB_Prof[5])
        else:
            stern_end = float(h.Lm + h.Lb + h.Kappa_STERN * h.Ls)
        WL.append([bow_start, 0.0, 0.0])
        for xi in X:
            if bow_start < xi < stern_end:
                WL.append([float(xi), 0.0, 0.0])
        WL.append([stern_end, 0.0, 0.0])
        return np.array(WL)

    # ---------------- bow section ----------------
    if bits.BB and z < float(h.WL):
        x1 = float(bb_profile(h, z))
        x2 = float(delta_bow(h, z))
        xs = [xi for xi in X if x1 < xi <= x2]
        prof, Xr0, Xr1, profrad = _bb_fillet(h, z, _bb_root_init(h, z))
        Xr0 = float(Xr0)
        Xr1 = float(Xr1)
        WL.append([x1, 0.0, z])
        for xi in xs:
            if xi <= Xr0:
                y = float(half_beam_bb(h, z, xi))
            elif xi < Xr1:
                y = float(_quartic(profrad, xi))
            else:
                y = float(_cubic(prof, xi))
            WL.append([float(xi), y, z])
    else:
        x1 = float(bow_profile(h, z))
        x2 = float(delta_bow(h, z))
        xs = [xi for xi in X if x1 < xi <= x2]
        prof = solve_waterline_bow(h, z)
        WL.append([x1, 0.0, z])
        for xi in xs:
            WL.append([float(xi), float(_cubic(prof, xi)), z])

    # ---------------- midbody section ----------------
    db = float(delta_bow(h, z))
    ds = float(delta_stern(h, z))
    y_mid = float(half_beam_midbody(h, z))
    for xi in X:
        if db <= xi < ds:
            WL.append([float(xi), y_mid, z])

    # ---------------- stern section ----------------
    is_ellipse = bool(bits.EP_T if z >= float(h.SK[1]) else bits.EP_S)
    if bits.SB and z < float(h.WL * h.HSBOA):
        x1s = ds
        x2s = float(sb_profile(h, z))
        xs = [xi for xi in X if x1s <= xi < x2s]
        branch_a = _sb_branch(h, bits, z, is_ellipse)
        parab, ell, Xr0, Xr1, profrad = _sb_fillet(h, bits, z, is_ellipse, branch_a)
        Xr0 = float(Xr0)
        Xr1 = float(Xr1)
        for xi in xs:
            if xi <= Xr0:
                y = float(_stern_curve(parab, ell, is_ellipse, xi))
            elif xi < Xr1:
                y = float(_quartic(profrad, xi))
            else:
                y = float(half_beam_sb(h, z, xi))
            WL.append([float(xi), y, z])
        WL.append([x2s, 0.0, z])
    else:
        x1s = ds
        x2s = float(stern_profile(h, bits, z))
        xs = [xi for xi in X if x1s <= xi < x2s]
        parab, ell = solve_waterline_stern(h, bits, z)
        for xi in xs:
            WL.append([float(xi), float(_stern_curve(parab, ell, is_ellipse, xi)), z])
        tx, ty = half_beam_transom(h, bits, z)
        WL.append([float(tx), float(ty), z])

    return np.array(WL)


def gen_meshgrid_pointcloud(x_np, NUM_WL=51, PointsPerLOA=501, Z=None, X=None):
    """Concrete replication of ``gen_MeshGridPointCloud`` (grid mode)."""
    x_np = np.asarray(x_np, dtype=np.float64)
    bits = bits_from_vector(x_np)
    h = build_hull(jnp.asarray(x_np), bits)

    if Z is None:
        Z = np.linspace(0.0001 * float(h.Dd), float(h.Dd), NUM_WL)
    if X is None:
        X = np.asarray(_linspace(-h.LOA * 0.5, 1.5 * h.LOA, 2 * PointsPerLOA - 1))

    return [_wl_grid_points(h, bits, X, float(z)) for z in Z], h, bits


def gen_stl_trivec(x_np, NUM_WL=50, PointsPerWL=300, bit_AddTransom=1,
                   bit_AddDeckLid=0, bit_RefineBowAndStern=0):
    """Triangle vertex array ((n,3,3)) replicating ``gen_stl``."""
    x_np = np.asarray(x_np, dtype=np.float64)
    bits = bits_from_vector(x_np)
    h = build_hull(jnp.asarray(x_np), bits)
    Dd = float(h.Dd)
    LOA = float(h.LOA)

    z = np.zeros((NUM_WL,))
    z[0] = 0.0001 * Dd
    z[1] = 0.001 * Dd
    z[2:] = np.linspace(0.0, Dd, NUM_WL - 2)
    z = np.sort(z)

    x = np.linspace(-LOA * 0.5, 1.5 * LOA, 2 * PointsPerWL - 1)
    if bit_RefineBowAndStern:
        x_sub1 = x[0:int(0.75 * PointsPerWL)] + 0.5 * (x[1] - x[0])
        x_sub2 = x[-int(0.75 * PointsPerWL):] + 0.5 * (x[1] - x[0])
        x = np.concatenate((x_sub1, x_sub2, x))
        x = np.sort(x)

    pts = [_wl_grid_points(h, bits, x, float(zz)) for zz in z]

    # ---- triangulation (verbatim port of the reference index logic) ----
    TriVec = []
    for i in range(0, NUM_WL - 1):
        bow = np.argmax([pts[i][0, 0], pts[i + 1][0, 0]])
        stern = np.argmin([pts[i][-1, 0], pts[i + 1][-1, 0]])

        if bow:
            idx_WLB1 = 1
            idx_WLB0 = np.where(pts[i][:, 0] == pts[i + 1][idx_WLB1, 0])[0][0]
        else:
            idx_WLB0 = 1
            idx_WLB1 = np.where(pts[i + 1][:, 0] == pts[i][idx_WLB0, 0])[0][0]

        if stern:
            idx_WLS1 = len(pts[i + 1]) - 2
            idx_WLS0 = np.where(pts[i][:, 0] == pts[i + 1][idx_WLS1, 0])[0][0]
        else:
            idx_WLS0 = len(pts[i]) - 2
            idx_WLS1 = np.where(pts[i + 1][:, 0] == pts[i][idx_WLS0, 0])[0][0]

        if bow:
            TriVec.append([pts[i + 1][idx_WLB1], pts[i][0], pts[i + 1][0]])
            for j in range(0, idx_WLB0):
                TriVec.append([pts[i + 1][idx_WLB1], pts[i][j + 1], pts[i][j]])
        else:
            for j in range(0, idx_WLB1):
                TriVec.append([pts[i][0], pts[i + 1][j], pts[i + 1][j + 1]])
            TriVec.append([pts[i][0], pts[i + 1][idx_WLB1], pts[i][idx_WLB0]])

        for j in range(0, idx_WLS1 - idx_WLB1):
            TriVec.append([pts[i][idx_WLB0 + j], pts[i + 1][idx_WLB1 + j],
                           pts[i + 1][idx_WLB1 + j + 1]])
            TriVec.append([pts[i][idx_WLB0 + j], pts[i + 1][idx_WLB1 + j + 1],
                           pts[i][idx_WLB0 + j + 1]])

        if stern:
            for j in range(idx_WLS0, len(pts[i]) - 1):
                TriVec.append([pts[i + 1][idx_WLS1], pts[i][j + 1], pts[i][j]])
            TriVec.append([pts[i + 1][idx_WLS1], pts[i + 1][-1], pts[i][-1]])
        else:
            TriVec.append([pts[i][idx_WLS0], pts[i + 1][idx_WLS1], pts[i][-1]])
            for j in range(idx_WLS1, len(pts[i + 1]) - 1):
                TriVec.append([pts[i][-1], pts[i + 1][j], pts[i + 1][j + 1]])

    TriVec = np.array(TriVec)
    hullTriangles = 2 * len(TriVec)

    vectors = [TriVec]
    TriVec_stbd = np.copy(TriVec[:, ::-1])
    TriVec_stbd[:, :, 1] *= -1
    vectors.append(TriVec_stbd)

    if bit_AddTransom:
        SK1 = float(h.SK[1])
        wl_above = len([zz for zz in z if zz > SK1])
        z_idx = NUM_WL - wl_above - 1

        pts_trans = np.zeros((wl_above + 1, 3))
        for i in range(0, len(pts_trans)):
            pts_trans[i] = pts[z_idx + i][-1, :]
        pts_tranp = np.array(pts_trans)
        pts_tranp[:, 1] *= -1.0

        trans = np.zeros((2 * wl_above - 1, 3, 3))
        trans[0] = np.array([pts_trans[0], pts_trans[1], pts_tranp[1]])
        for i in range(1, wl_above):
            trans[2 * i - 1] = np.array([pts_trans[i], pts_trans[i + 1], pts_tranp[i]])
            trans[2 * i] = np.array([pts_tranp[i], pts_trans[i + 1], pts_tranp[i + 1]])
        vectors.append(trans)

    if bit_AddDeckLid:
        pts_Lids = pts[NUM_WL - 1]
        pts_Lidp = np.array(pts_Lids)
        pts_Lidp[:, 1] *= -1.0

        lid = np.zeros((2 * len(pts_Lids) - 3, 3, 3))
        lid[0] = np.array([pts_Lids[0], pts_Lidp[1], pts_Lids[1]])
        for i in range(1, len(pts_Lids) - 1):
            lid[2 * i - 1] = np.array([pts_Lids[i], pts_Lidp[i], pts_Lids[i + 1]])
            lid[2 * i] = np.array([pts_Lids[i + 1], pts_Lidp[i], pts_Lidp[i + 1]])
        vectors.append(lid)

    return np.concatenate(vectors, axis=0)


def write_stl_binary(trivec: np.ndarray, path: str, name: bytes = b"shipd_jax"):
    """Minimal binary STL writer (normals from the right-hand rule)."""
    tri = np.asarray(trivec, dtype=np.float64)
    n = len(tri)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normals = np.cross(e1, e2)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lens > 0, normals / np.where(lens == 0, 1, lens), 0.0)

    rec = np.zeros(n, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                             ("attr", "<u2")])
    rec["n"] = normals.astype(np.float32)
    rec["v"] = tri.astype(np.float32)
    with open(path, "wb") as fh:
        header = name[:80].ljust(80, b"\0")
        fh.write(header)
        fh.write(np.uint32(n).tobytes())
        fh.write(rec.tobytes())


def gen_stl(x_np, NUM_WL=50, PointsPerWL=300, bit_AddTransom=1,
            bit_AddDeckLid=0, bit_RefineBowAndStern=0,
            namepath="Hull_Mesh"):
    """Drop-in analogue of the reference ``gen_stl`` (writes namepath + '.stl')."""
    tv = gen_stl_trivec(x_np, NUM_WL=NUM_WL, PointsPerWL=PointsPerWL,
                        bit_AddTransom=bit_AddTransom,
                        bit_AddDeckLid=bit_AddDeckLid,
                        bit_RefineBowAndStern=bit_RefineBowAndStern)
    write_stl_binary(tv, namepath + ".stl")
    return tv
