#!/usr/bin/env python3
"""Validation for JAX Ship-D constraint implementation."""

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from HullParameterization import Hull_Parameterization as HP
from HullParameterizationJAX import input_constraints_jax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Row index from Input_Vectors_SampleHulls.csv")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    vectors = np.loadtxt("Input_Vectors_SampleHulls.csv", delimiter=",", dtype=np.float64)
    x_np = vectors[args.index]

    hull = HP(x_np)
    c_np = hull.input_Constraints()
    c_jax = np.asarray(input_constraints_jax(jnp.asarray(x_np)))

    print(f"constraint vector length: {c_jax.shape[0]}")
    print(f"max |numpy-jax|: {np.max(np.abs(c_np - c_jax)):.6e}")

    if not np.allclose(c_np, c_jax, rtol=args.rtol, atol=args.atol):
        raise AssertionError("JAX constraints do not match HullParameterization.input_Constraints")

    f = lambda x: input_constraints_jax(x)
    jac_rev = jax.jacrev(f)(jnp.asarray(x_np))
    jac_fwd = jax.jacfwd(f)(jnp.asarray(x_np))

    print(f"jacrev shape: {tuple(jac_rev.shape)}")
    print(f"jacfwd shape: {tuple(jac_fwd.shape)}")

    if not jnp.all(jnp.isfinite(jac_rev)):
        raise AssertionError("jacrev contains non-finite values")
    if not jnp.all(jnp.isfinite(jac_fwd)):
        raise AssertionError("jacfwd contains non-finite values")

    diff = np.max(np.abs(np.asarray(jac_rev) - np.asarray(jac_fwd)))
    print(f"max |jacrev-jacfwd|: {diff:.6e}")
    print("Validation passed.")


if __name__ == "__main__":
    main()
