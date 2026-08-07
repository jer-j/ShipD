#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare shipd_jax.stl_export.gen_stl_trivec against the reference gen_stl."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")

spec = importlib.util.spec_from_file_location(
    "HullParameterization_ref",
    os.path.join(_ROOT, "spring_26_rewrite", "HullParameterization.py"))
_ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_ref)
HP = _ref.Hull_Parameterization

from shipd_jax.stl_export import gen_stl_trivec


def main():
    csv = np.loadtxt(os.path.join(_ROOT, "Input_Vectors_SampleHulls.csv"),
                     delimiter=",", dtype=np.float64)
    tmp = tempfile.mkdtemp()
    for i in range(csv.shape[0]):
        vec = csv[i]
        try:
            ref = HP(vec)
            mesh_ref = ref.gen_stl(NUM_WL=40, PointsPerWL=200, bit_AddTransom=1,
                                   bit_AddDeckLid=1,
                                   namepath=os.path.join(tmp, f"ref_{i}"))
            v_ref = np.asarray(mesh_ref.vectors, dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            print(f"csv[{i}]: reference gen_stl failed ({type(e).__name__}) - skipped")
            continue
        try:
            v_jax = gen_stl_trivec(vec, NUM_WL=40, PointsPerWL=200,
                                   bit_AddTransom=1, bit_AddDeckLid=1)
        except Exception as e:  # noqa: BLE001
            print(f"csv[{i}]: shipd_jax gen_stl_trivec FAILED: {type(e).__name__}: {e}")
            continue
        if v_ref.shape != v_jax.shape:
            print(f"csv[{i}]: SHAPE MISMATCH ref={v_ref.shape} jax={v_jax.shape}")
            continue
        # reference stores float32 vectors; compare at float32 resolution
        err = np.max(np.abs(v_ref.astype(np.float64) - v_jax.astype(np.float32)))
        print(f"csv[{i}]: {len(v_ref)} triangles, max|dv| = {err:.3e}")


if __name__ == "__main__":
    main()
