#!/usr/bin/env python3
"""
Numerical validation of shipd_jax (JAX) against the original
HullParameterization (shipd_jax.reference, Cython).

Compares, per hull:
  * the 49 feasibility constraints,
  * the (NUM_WL x PointsPerWL x 3) measurement point cloud,
  * every hydrostatic quantity of Calc_VolumeProperties,
  * the Michell wave-resistance inputs (X, Z, Y, WL) and the resistance value
    across drafts and Froude numbers,
and finally checks JAX gradients against central finite differences of the
JAX forward (fixed plan/topology).

Run:
    python tools/validate_vs_reference.py [--n30k 20] [--full-res]

--n30k reads InputVectors_30k.npy (the ShipD dataset's design
vectors, not committed) and is skipped when that file is absent.
"""

from __future__ import annotations

import argparse
import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from shipd_jax import (
    SAMPLE_HULLS,
    bits_from_vector,
    build_cw_plan,
    build_volume_plan,
    constraints,
    gen_pc_for_cw,
    gen_pointcloud,
    mod_michell,
    volume_properties,
)
from shipd_jax.reference import modified_michell_cw as MWCw
from shipd_jax.reference.hull_parameterization import (
    Hull_Parameterization as HP,
)

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent.parent / "data" / "shipd"


def relerr(a, b, floor=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    return np.max(np.abs(a - b) / denom)


def maxabs(a, b):
    return np.max(
        np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    )


def compare_hull(
    vec,
    num_wl=51,
    ppw=300,
    cw_drafts=(0.25, 0.33, 0.5, 0.67),
    cw_fns=(0.15, 0.3, 0.45),
    cw_num_wl=51,
    cw_ppw=301,
    verbose=False,
):
    """Return dict of max abs/rel errors for one hull."""
    out = {}
    bits = bits_from_vector(vec)

    ref = HP(np.array(vec, dtype=np.float64))

    # constraints
    c_ref = np.asarray(ref.input_Constraints(), dtype=np.float64)
    c_jax = np.asarray(constraints(jnp.asarray(vec), bits))
    out["constraints_maxabs"] = maxabs(c_ref, c_jax)

    # point cloud + volume properties
    try:
        with redirect_stdout(io.StringIO()):
            ref.Calc_VolumeProperties(NUM_WL=num_wl, PointsPerWL=ppw)
        pc_ref = np.asarray(ref.PCMeasurement, dtype=np.float64)
        ref_vol_ok = True
    except Exception:  # the reference can crash on some hulls
        ref_vol_ok = False

    if ref_vol_ok:
        plan = build_volume_plan(vec, num_wl=num_wl, ppw=ppw)
        props = volume_properties(jnp.asarray(vec), plan)
        pc_jax = np.asarray(props["pointcloud"])

        out["pointcloud_maxabs"] = maxabs(pc_ref, pc_jax)
        out["areas_wp_rel"] = relerr(ref.Areas_WP, props["areas_wp"])
        out["volumes_rel"] = relerr(ref.Volumes, props["volumes"])
        out["lcf_rel"] = relerr(ref.LCFs, props["lcf"])
        out["lcb_rel"] = relerr(ref.VolumeCentroids[:, 0], props["lcb"])
        out["vcb_rel"] = relerr(ref.VolumeCentroids[:, 1], props["vcb"])
        out["ixx_rel"] = relerr(ref.I_WP[:, 0], props["ixx"])
        out["iyy_rel"] = relerr(ref.I_WP[:, 1], props["iyy"])
        out["wsa_rel"] = relerr(ref.Area_WS, props["area_ws"])
        out["wll_rel"] = relerr(ref.WL_Lengths, props["wl_lengths"])
    else:
        for k in (
            "pointcloud_maxabs",
            "areas_wp_rel",
            "volumes_rel",
            "lcf_rel",
            "lcb_rel",
            "vcb_rel",
            "ixx_rel",
            "iyy_rel",
            "wsa_rel",
            "wll_rel",
        ):
            out[k] = np.nan  # reference itself crashed; nothing to compare

    # Michell wave resistance
    g = 9.81
    rho = 1000.0
    worst_pc = 0.0
    worst_rw = 0.0
    cw_ok = True
    for tfrac in cw_drafts:
        try:
            Xr, Zr, Yr, WLr = ref.gen_PC_for_Cw(
                ref.Dd * tfrac, NUM_WL=cw_num_wl, PointsPerWL=cw_ppw
            )
        except Exception:
            cw_ok = False
            continue
        cw_plan = build_cw_plan(vec, tfrac, num_wl=cw_num_wl, ppw=cw_ppw)
        Xj, Zj, Yj, WLj = gen_pc_for_cw(jnp.asarray(vec), cw_plan)

        worst_pc = max(
            worst_pc, maxabs(Xr, Xj), maxabs(Zr, Zj), maxabs(Yr, Yj), maxabs(WLr, WLj)
        )

        for fn in cw_fns:
            U = fn * np.sqrt(g * WLr)
            rw_ref = MWCw.ModMichell(Yr, U, Xr, Zr, rho, 21)
            Uj = fn * jnp.sqrt(g * WLj)
            rw_jax = float(mod_michell(Yj, Uj, Xj, Zj, rho, 21))
            worst_rw = max(worst_rw, relerr(rw_ref, rw_jax, floor=1e-20))
    out["cw_grid_maxabs"] = worst_pc if cw_ok else np.nan
    out["cw_value_rel"] = worst_rw if cw_ok else np.nan

    if verbose:
        for k, v in out.items():
            print(f"    {k:24s} {v:9.3e}")
    return out


# Gradient checks (JAX AD vs central finite differences on the JAX forward)


def _fd_grad(f, x, eps_rel=1e-6):
    x = np.array(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = eps_rel * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += e
        xm = x.copy()
        xm[i] -= e
        g[i] = (float(f(jnp.asarray(xp))) - float(f(jnp.asarray(xm)))) / (2 * e)
    return g


def _grad_agreement(g_ad, g_fd, tol_scale):
    """Max normalized disagreement, using the gradient's norm as scale."""
    scale = max(np.max(np.abs(g_fd)), np.max(np.abs(g_ad)), tol_scale)
    return np.max(np.abs(g_ad - g_fd)) / scale


def gradient_checks(vec, num_wl=21, ppw=101, verbose=True):
    # modest resolutions keep XLA compile times manageable; the gradient
    # math being verified is resolution-independent
    bits = bits_from_vector(vec)
    plan = build_volume_plan(vec, num_wl=num_wl, ppw=ppw)
    cw_plan = build_cw_plan(vec, 0.33, num_wl=21, ppw=101)
    x0 = jnp.asarray(np.array(vec, dtype=np.float64))
    rho, g0 = 1000.0, 9.81

    results = {}

    def vol_total(x):
        return volume_properties(x, plan)["volumes"][-1]

    def wsa_total(x):
        return volume_properties(x, plan)["area_ws"][-1]

    def cw_val(x):
        X, Z, Y, WL = gen_pc_for_cw(x, cw_plan)
        U = 0.3 * jnp.sqrt(g0 * WL)
        return mod_michell(Y, U, X, Z, rho, 21)

    def cons_sum(x):
        # smooth aggregate touching every constraint
        return jnp.sum(jnp.tanh(constraints(x, bits)))

    def surf_pt(x):
        pc = gen_pointcloud(x, plan)
        return pc[num_wl // 2, ppw // 3, 0] + pc[num_wl // 2, ppw // 3, 1]

    for name, f in [
        ("volume", vol_total),
        ("wetted_surface", wsa_total),
        ("michell_rw", cw_val),
        ("constraints_tanh_sum", cons_sum),
        ("surface_point", surf_pt),
    ]:
        f = jax.jit(f)
        g_ad = np.asarray(jax.jit(jax.grad(f))(x0))
        g_fd = _fd_grad(f, vec)
        err = _grad_agreement(g_ad, g_fd, 1e-12)
        nan_free = bool(np.all(np.isfinite(g_ad)))
        results[name] = (err, nan_free)
        if verbose:
            print(f"    grad {name:22s} rel_err={err:9.3e}  finite={nan_free}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n30k",
        type=int,
        default=20,
        help="number of hulls from InputVectors_30k.npy to test",
    )
    ap.add_argument(
        "--full-res",
        action="store_true",
        help="also run one hull at dataset resolution (101x1000)",
    )
    ap.add_argument("--skip-grad", action="store_true")
    args = ap.parse_args()

    csv = np.loadtxt(SAMPLE_HULLS, delimiter=",", dtype=np.float64)
    vecs = [csv[i] for i in range(csv.shape[0])]
    labels = [f"csv[{i}]" for i in range(csv.shape[0])]

    npy_path = _ROOT / "InputVectors_30k.npy"
    if args.n30k > 0 and npy_path.exists():
        big = np.load(npy_path)
        idx = np.linspace(0, len(big) - 1, args.n30k, dtype=int)
        vecs += [big[i] for i in idx]
        labels += [f"30k[{i}]" for i in idx]

    keys = None
    worst = {}
    t0 = time.time()
    for lab, v in zip(labels, vecs, strict=True):
        try:
            out = compare_hull(v)
        except Exception as e:
            print(f"{lab:10s} FAILED: {type(e).__name__}: {e}")
            continue
        if keys is None:
            keys = list(out.keys())
            print("hull       " + "  ".join(f"{k[:14]:>14s}" for k in keys))
        print(f"{lab:10s} " + "  ".join(f"{out[k]:14.3e}" for k in keys))
        for k, val in out.items():
            if np.isfinite(val):
                worst[k] = max(worst.get(k, 0.0), val)

    print("-" * 100)
    if keys:
        print("WORST      " + "  ".join(f"{worst[k]:14.3e}" for k in keys))
    print(f"[{time.time() - t0:.1f}s]")

    if args.full_res:
        print("\nFull dataset resolution (101x1000), hull csv[0]:")
        out = compare_hull(
            vecs[0],
            num_wl=101,
            ppw=1000,
            verbose=True,
            cw_drafts=(0.33,),
            cw_fns=(0.3,),
        )

    if not args.skip_grad:
        print("\nGradient checks (hull csv[0]):")
        gradient_checks(vecs[0])
        # also on a hull without bulbs for branch coverage
        for lab, v in zip(labels, vecs, strict=True):
            b = bits_from_vector(v)
            if not b.BB and not b.SB:
                print(f"\nGradient checks ({lab}, no bulbs):")
                gradient_checks(v)
                break


if __name__ == "__main__":
    main()
