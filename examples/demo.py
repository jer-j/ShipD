#!/usr/bin/env python3
"""
Demo: derivatives of the Ship-D hull surface, constraints, and resistance
with respect to the 45 design variables.

Run:  python examples/demo.py [hull_index]
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from shipd_jax import (
    SAMPLE_HULLS,
    bits_from_vector,
    build_cw_plan,
    build_volume_plan,
    calc_cf,
    constraints,
    gen_pc_for_cw,
    gen_pointcloud,
    mod_michell,
    volume_properties,
)

LABELS = [
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


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    x_np = np.loadtxt(SAMPLE_HULLS, delimiter=",", dtype=np.float64)[idx]
    x = jnp.asarray(x_np)
    bits = bits_from_vector(x_np)
    print(f"hull csv[{idx}]  bits={bits}")

    # ------------------------------------------------------------------
    # 1) Hull surface: point grid + full Jacobian d(surface)/d(params)
    # ------------------------------------------------------------------
    plan = build_volume_plan(x_np, num_wl=51, ppw=300)
    t0 = time.time()
    pc = gen_pointcloud(x, plan)
    J_surf = jax.jacfwd(lambda v: gen_pointcloud(v, plan))(x)
    print(
        f"\nsurface grid {pc.shape}, jacobian {J_surf.shape} "
        f"({time.time() - t0:.1f}s, finite={bool(jnp.all(jnp.isfinite(J_surf)))})"
    )
    # example: sensitivity of the total displaced volume
    props = volume_properties(x, plan)
    dVol = jax.grad(lambda v: volume_properties(v, plan)["volumes"][-1])(x)
    print(f"displaced volume at deck: {float(props['volumes'][-1]):.4f}")
    top = np.argsort(-np.abs(np.asarray(dVol)))[:5]
    print(
        "  largest dVol/dx: "
        + ", ".join(f"{LABELS[i]}={float(dVol[i]):+.3f}" for i in top)
    )

    # ------------------------------------------------------------------
    # 2) Constraints: values + Jacobian (49 x 45)
    # ------------------------------------------------------------------
    c = constraints(x, bits)
    Jc = jax.jacfwd(lambda v: constraints(v, bits))(x)
    print(
        f"\nconstraints: {int(jnp.sum(c > 0))} violated of {c.shape[0]}, "
        f"jacobian {Jc.shape} (finite={bool(jnp.all(jnp.isfinite(Jc)))})"
    )

    # ------------------------------------------------------------------
    # 3) Resistance: Michell Rw + total drag, gradient w.r.t. params
    # ------------------------------------------------------------------
    cw_plan = build_cw_plan(x_np, draft_frac=0.33, num_wl=51, ppw=301)
    rho, g = 1000.0, 9.81

    def total_drag(v, Fn=0.3):
        X, Z, Y, WL = gen_pc_for_cw(v, cw_plan)
        U = Fn * jnp.sqrt(g * WL)
        Rw = mod_michell(Y, U, X, Z, rho, 21)
        pr = volume_properties(v, plan)
        # wetted surface interpolated at the 0.33*Dd draft mark
        h = pr["hull"]
        Zm = pr["Z"]
        SA = jnp.interp(0.33 * h.Dd, Zm, pr["area_ws"])
        Rf = 0.5 * calc_cf(U, WL) * rho * SA * U**2.0
        return Rw + Rf

    t0 = time.time()
    D = float(total_drag(x))
    gD = jax.grad(total_drag)(x)
    print(
        f"\ntotal drag (Fn=0.3, T=0.33Dd): {D:.6g}  "
        f"({time.time() - t0:.1f}s, grad finite={bool(jnp.all(jnp.isfinite(gD)))})"
    )
    top = np.argsort(-np.abs(np.asarray(gD)))[:5]
    print(
        "  largest dDrag/dx: "
        + ", ".join(f"{LABELS[i]}={float(gD[i]):+.4g}" for i in top)
    )


if __name__ == "__main__":
    main()
