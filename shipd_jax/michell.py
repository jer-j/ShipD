#!/usr/bin/env python3
"""
Differentiable port of ``ModifiedMichellCw.py`` (Michell thin-ship wave drag).

The triple loop of the reference is vectorized, keeping the exact Filon
weights and operation forms.  Propagation angles whose Filon z-weights would
overflow ``exp`` are masked out: in the reference those angles produce NaN
which ``np.nan_to_num`` maps to a zero contribution, so forward values match
while gradients stay finite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

G = 9.81

# largest x with np.exp(x) finite in float64
_EXP_OVERFLOW = 709.782712893384


def michspace(N: int) -> np.ndarray:
    """Log spacing over N propagation angles in [0, pi/2] (dense near pi/2)."""
    xm = (np.logspace(0, 1, N, base=10.0) - 1.0) * np.pi / 18 - np.pi / 2
    return -xm[::-1]


def mod_michell(Y, U, X, Z, rho, N: int = 21):
    """Wave-resistance integral; drop-in for ``ModMichell(Y, U, X, Z, RHO, N)``.

    Y : (Nx, Nz) half-beam offsets (Nx odd), X : (Nx,) with X[0] = 0,
    Z : (Nz,) from -draft to 0, U : speed, rho : density.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    X = jnp.asarray(X, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    Nx, Nz = Y.shape
    if Nx % 2 == 0:
        raise ValueError("Nx must be odd for the Filon x-integration")

    dz = Z[1] - Z[0]
    L = X[-1]
    dx = L / (Nx - 1)

    theta = jnp.asarray(michspace(N))
    k0 = G / U**2.0
    c = (4.0 * rho * U**2) / jnp.pi
    a = 1.0 / jnp.cos(theta)  # (N,)
    k = k0 * a**2.0

    # Z integral (Filon trapezoid)
    Kz = k0 * dz * a**2.0
    ok = Kz <= _EXP_OVERFLOW  # reference: NaN -> 0 beyond this
    Kz_s = jnp.where(ok, Kz, 1.0)
    Kzsq = Kz_s**2.0
    w0 = (jnp.exp(Kz_s) - 1.0 - Kz_s) / Kzsq
    wn = (jnp.exp(Kz_s) - 1.0 + jnp.exp(-Kz_s) - 1.0) / Kzsq
    wN = (jnp.exp(-Kz_s) - 1.0 + Kz_s) / Kzsq

    # weights per (j, n)
    W = jnp.broadcast_to(wn[:, None], (N, Nz))
    W = W.at[:, 0].set(w0)
    W = W.at[:, -1].set(wN)

    # exp(k0 * Z[n] * a[j]^2): Z <= 0 so this never overflows
    EZ = jnp.exp(k0 * Z[:, None] * (a**2.0)[None, :])  # (Nz, N)

    F = dz * (Y @ (W.T * EZ))  # (Nx, N)

    # X integral (Filon)
    Kx = k0 * dx * a
    alp = (Kx**2 + 0.5 * Kx * jnp.sin(2.0 * Kx) + jnp.cos(2 * Kx) - 1) / Kx**3.0
    bet = (3.0 * Kx + Kx * jnp.cos(2.0 * Kx) - 2.0 * jnp.sin(2 * Kx)) / Kx**3.0
    gam = 4.0 * (jnp.sin(Kx) - Kx * jnp.cos(Kx)) / Kx**3.0

    phase = k0 * X[:, None] * a[None, :]  # (Nx, N)
    FC = F * jnp.cos(phase)
    FS = F * jnp.sin(phase)

    Pt = FC[-1, :]
    Qt = FS[-1, :]

    Pev = jnp.sum(FC[0::2, :], axis=0) - 0.5 * Pt
    Pod = jnp.sum(FC[1::2, :], axis=0)
    Qev = jnp.sum(FS[0::2, :], axis=0) - 0.5 * Qt
    Qod = jnp.sum(FS[1::2, :], axis=0)

    P = dx * (alp * Qt + bet * Pev + gam * Pod)
    Q = dx * (-alp * Pt + bet * Qev + gam * Qod)

    R = (
        c
        * k**2.0
        / (a**3.0)
        * (
            k**2.0 * (P**2.0 + Q**2.0)
            + 2.0 * k * a * (Q * Pt - P * Qt)
            + a**2.0 * (Pt**2.0 + Qt**2.0)
        )
    )
    R = jnp.where(ok, R, 0.0)
    R = jnp.nan_to_num(R)

    # theta integral (trapezoid)
    return jnp.sum(0.5 * (R[:-1] + R[1:]) * (theta[1:] - theta[:-1]))


def calc_cf(U, WL, v=1.19e-6):
    """1957 ITTC skin-friction line."""
    Re = U * WL / v
    return 0.075 / (jnp.log10(Re) - 2.0) ** 2.0


# Froude numbers and draft fractions of the dataset's precomputed Cw table
CW_TABLE_FN = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
CW_TABLE_T = np.array([0.25, 0.33, 0.5, 0.67])


def interp_cw(Fn, T, CW, Fn_list=CW_TABLE_FN, T_list=CW_TABLE_T):
    """Bilinear interpolation of a precomputed Cw table (differentiable)."""
    Fn_list = jnp.asarray(Fn_list)
    T_list = jnp.asarray(T_list)
    tab = jnp.reshape(jnp.asarray(CW), (T_list.shape[0], Fn_list.shape[0]))

    def _locate(v, grid):
        idx = jnp.clip(jnp.sum(grid < v) - 1, 0, grid.shape[0] - 2)
        frac = (v - grid[idx]) / (grid[idx + 1] - grid[idx])
        frac = jnp.where(v >= grid[-1], 1.0, jnp.where(v <= grid[0], 0.0, frac))
        idx = jnp.where(
            v >= grid[-1], grid.shape[0] - 2, jnp.where(v <= grid[0], 0, idx)
        )
        return idx, frac

    fi, ff = _locate(Fn, Fn_list)
    ti, tf = _locate(T, T_list)

    cwl = tab[ti, fi] + ff * (tab[ti, fi + 1] - tab[ti, fi])
    cwu = tab[ti + 1, fi] + ff * (tab[ti + 1, fi + 1] - tab[ti + 1, fi])
    return cwl + tf * (cwu - cwl)


def calc_drag(U, LOA, WL, CW, T, SA, rho=1025.0):
    """Total drag (wave via interpolated Cw table + ITTC friction)."""
    Cf = calc_cf(U=U, WL=WL)
    Rf = 0.5 * Cf * rho * SA * U**2.0
    Fn = U / jnp.sqrt(G * WL)
    Cw = interp_cw(Fn=Fn, T=T, CW=CW)
    Rw = 0.5 * Cw * rho * (LOA**2.0) * (U**2.0)
    return (Rw + Rf), Fn
