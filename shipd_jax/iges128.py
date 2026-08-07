#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal IGES reader for untrimmed rational B-spline surfaces (entity 128).

Just enough IGES to read hull geometries like the SIMMAN 2008 DTMB 5415
distribution (six untrimmed NURBS patches): parses the directory/parameter
sections, extracts each type-128 entity, and evaluates the tensor-product
NURBS with `scipy.interpolate.BSpline.design_matrix`.

This module performs no downloading; callers fetch the .igs file themselves
(e.g. from http://www.simman2008.dk/5415/5415_geometry.htm) so that the
geometry always comes from its official source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class NurbsSurface:
    deg_u: int
    deg_v: int
    knots_u: np.ndarray          # full knot vector, length n_u + deg_u + 1
    knots_v: np.ndarray
    weights: np.ndarray          # (n_u, n_v)
    ctrl: np.ndarray             # (n_u, n_v, 3)
    u_range: tuple
    v_range: tuple

    def evaluate(self, nu: int = 60, nv: int = 60) -> np.ndarray:
        """Sample the surface on an (nu, nv) parameter grid -> (nu, nv, 3)."""
        u = np.linspace(self.u_range[0], self.u_range[1], nu)
        v = np.linspace(self.v_range[0], self.v_range[1], nv)
        # clip to the strictly valid basis support to dodge end-knot roundoff
        eps_u = 1e-12 * (self.u_range[1] - self.u_range[0])
        eps_v = 1e-12 * (self.v_range[1] - self.v_range[0])
        u = np.clip(u, self.u_range[0] + eps_u, self.u_range[1] - eps_u)
        v = np.clip(v, self.v_range[0] + eps_v, self.v_range[1] - eps_v)
        Bu = BSpline.design_matrix(u, self.knots_u, self.deg_u).toarray()  # (nu, n_u)
        Bv = BSpline.design_matrix(v, self.knots_v, self.deg_v).toarray()  # (nv, n_v)
        wp = self.ctrl * self.weights[..., None]        # (n_u, n_v, 3)
        num = np.einsum("ui,ijk,vj->uvk", Bu, wp, Bv)
        den = np.einsum("ui,ij,vj->uv", Bu, self.weights, Bv)
        return num / den[..., None]


def read_iges_128(path: str) -> List[NurbsSurface]:
    """Read all entity-128 (rational B-spline surface) entities of an IGES file."""
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    dir_lines = [ln for ln in lines if len(ln) > 72 and ln[72] == "D"]
    par_lines = [ln for ln in lines if len(ln) > 72 and ln[72] == "P"]

    # parameter section text keyed by the DE pointer in cols 66..72
    par_by_de: dict = {}
    for ln in par_lines:
        de = int(ln[64:72])
        par_by_de.setdefault(de, []).append(ln[:64])

    surfaces = []
    for i in range(0, len(dir_lines), 2):
        d1 = dir_lines[i]
        etype = int(d1[0:8])
        if etype != 128:
            continue
        de_number = int(d1[73:80])       # sequence number of this D entry
        text = "".join(par_by_de[de_number])
        # parameter data: comma separated, record terminated by ';'
        text = text.split(";")[0]
        vals = [float(t) for t in text.replace(",", " ").split()][0:]
        # first value repeats the entity type
        assert int(vals[0]) == 128, vals[0]
        K1, K2, M1, M2 = (int(v) for v in vals[1:5])
        # PROP1..PROP5 at vals[5:10]
        p = 10
        n_u, n_v = K1 + 1, K2 + 1
        nk_u = K1 + M1 + 2
        nk_v = K2 + M2 + 2
        knots_u = np.array(vals[p:p + nk_u]); p += nk_u
        knots_v = np.array(vals[p:p + nk_v]); p += nk_v
        w = np.array(vals[p:p + n_u * n_v]); p += n_u * n_v
        xyz = np.array(vals[p:p + 3 * n_u * n_v]); p += 3 * n_u * n_v
        u0, u1, v0, v1 = vals[p:p + 4]
        # IGES orders control points with the FIRST index (u) varying fastest
        weights = w.reshape(n_v, n_u).T
        ctrl = xyz.reshape(n_v, n_u, 3).transpose(1, 0, 2)
        surfaces.append(NurbsSurface(M1, M2, knots_u, knots_v, weights, ctrl,
                                     (u0, u1), (v0, v1)))
    return surfaces


def sample_iges_pointcloud(path: str, nu: int = 80, nv: int = 80) -> np.ndarray:
    """All 128-entities sampled and stacked into one (N, 3) point cloud."""
    pts = [s.evaluate(nu, nv).reshape(-1, 3) for s in read_iges_128(path)]
    return np.vstack(pts)
