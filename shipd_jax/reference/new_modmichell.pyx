# ModifiedMichellCw.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport cython
from libc.math cimport cos, sin, exp, sqrt, log10, M_PI, isnan, isinf, pow

cdef double g = 9.81

# Pre-allocate default arrays for interp_CW to avoid re-allocation on every call
cdef double[::1] DEFAULT_FN_ARR = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], dtype=np.float64)
cdef double[::1] DEFAULT_T_ARR  = np.array([0.25, 0.33, 0.5, 0.67], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[::1] michspace(int N):
    """
    De-vectorized to avoid NumPy overhead. Builds the array directly in C.
    """
    cdef double[::1] xm = np.empty(N, dtype=np.float64)
    cdef double val
    cdef int i
    cdef double denom = float(N - 1) if N > 1 else 1.0
    
    for i in range(N):
        # Equivalent to reversed np.logspace(0, 1, N)
        val = pow(10.0, float(N - 1 - i) / denom)
        xm[i] = -((val - 1.0) * M_PI / 18.0 - M_PI / 2.0)
        
    return xm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double ModMichell(
    double[:, ::1] Y,
    double U,
    double[::1] X,
    double[::1] Z,
    double RHO,
    int N
):
    cdef:
        int Nx = Y.shape[0]
        int Nz = Y.shape[1]
        int j, m, n
        
        double dz, L, dx
        double k0, cconst
        double sum_f, term, exp_term, angle_term
        
        # Loop variables flattened from arrays
        double a_j, k_j, Kz_j, Kzsq_j
        double w0_j, wn_j, wN_j
        double Kx_j, alp_j, bet_j, gam_j
        double sum_pev, sum_qev, sum_pod, sum_qod
        double Pt_j, Qt_j, Pev_j, Qev_j, Pod_j, Qod_j
        double P_j, Q_j, Rk, rw_sum
        
        double[::1] theta
        double[::1] R = np.zeros(N, dtype=np.float64)

    if Nx % 2 == 0:
        raise ValueError("Nx must be odd.")

    dz = Z[1] - Z[0]
    L = X[Nx - 1]
    dx = L / (Nx - 1)

    theta = michspace(N)

    k0 = g / (U * U)
    cconst = (4.0 * RHO * U * U) / M_PI

    # Collapse massive array allocations into a single J loop
    for j in range(N):
        a_j = 1.0 / cos(theta[j])
        k_j = k0 * a_j * a_j
        
        Kz_j = k0 * dz * a_j * a_j
        Kzsq_j = Kz_j * Kz_j
        
        w0_j = (exp(Kz_j) - 1.0 - Kz_j) / Kzsq_j
        wn_j = (exp(Kz_j) - 1.0 + exp(-Kz_j) - 1.0) / Kzsq_j
        wN_j = (exp(-Kz_j) - 1.0 + Kz_j) / Kzsq_j

        sum_pev = 0.0
        sum_qev = 0.0
        sum_pod = 0.0
        sum_qod = 0.0
        Pt_j = 0.0
        Qt_j = 0.0

        # Inner matrix processing
        for m in range(Nx):
            sum_f = 0.0
            for n in range(Nz):
                exp_term = exp(k_j * Z[n])
                
                if n == 0:
                    term = w0_j * Y[m, n] * exp_term * dz
                elif n == Nz - 1:
                    term = wN_j * Y[m, n] * exp_term * dz
                else:
                    term = wn_j * Y[m, n] * exp_term * dz
                    
                sum_f += term

            # Immediate consolidation into Ev/Od totals (avoids F matrix, pev, qev arrays)
            angle_term = k0 * X[m] * a_j
            if m % 2 == 0:
                sum_pev += sum_f * cos(angle_term)
                sum_qev += sum_f * sin(angle_term)
            else:
                sum_pod += sum_f * cos(angle_term)
                sum_qod += sum_f * sin(angle_term)
                
            if m == Nx - 1:
                Pt_j = sum_f * cos(angle_term)
                Qt_j = sum_f * sin(angle_term)

        Pev_j = sum_pev - 0.5 * Pt_j
        Qev_j = sum_qev - 0.5 * Qt_j
        Pod_j = sum_pod
        Qod_j = sum_qod

        Kx_j = k0 * dx * a_j
        alp_j = (Kx_j * Kx_j + 0.5 * Kx_j * sin(2.0 * Kx_j) + cos(2.0 * Kx_j) - 1.0) / (Kx_j * Kx_j * Kx_j)
        bet_j = (3.0 * Kx_j + Kx_j * cos(2.0 * Kx_j) - 2.0 * sin(2.0 * Kx_j)) / (Kx_j * Kx_j * Kx_j)
        gam_j = 4.0 * (sin(Kx_j) - Kx_j * cos(Kx_j)) / (Kx_j * Kx_j * Kx_j)

        P_j = dx * (alp_j * Qt_j + bet_j * Pev_j + gam_j * Pod_j)
        Q_j = dx * (-alp_j * Pt_j + bet_j * Qev_j + gam_j * Qod_j)

        Rk = cconst * (k_j * k_j) / (a_j * a_j * a_j) * (
            (k_j * k_j) * (P_j * P_j + Q_j * Q_j)
            + 2.0 * k_j * a_j * (Q_j * Pt_j - P_j * Qt_j)
            + a_j * a_j * (Pt_j * Pt_j + Qt_j * Qt_j)
        )

        if isnan(Rk) or isinf(Rk):
            R[j] = 0.0
        else:
            R[j] = Rk

    # Final Trapz integration
    rw_sum = 0.0
    for j in range(N - 1):
        rw_sum += 0.5 * (R[j] + R[j + 1]) * (theta[j + 1] - theta[j])

    return rw_sum


@cython.cdivision(True)
cpdef double Calc_Cf(double U, double WL, double v=1.19e-6):
    cdef double Re = U * WL / v
    return 0.075 / (log10(Re) - 2.0) ** 2.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double interp_CW(
    double Fn,
    double T,
    double[::1] CW,
    object Fn_list=None,
    object T_list=None
):
    cdef:
        int nFn, nT, i
        int Fn_idx = 0
        int T_idx = 0
        double Fn_frac = 0.0
        double T_frac = 0.0
        double CwL, CwU
        double[::1] Fn_arr
        double[::1] T_arr

    Fn_arr = DEFAULT_FN_ARR if Fn_list is None else np.asarray(Fn_list, dtype=np.float64)
    T_arr = DEFAULT_T_ARR if T_list is None else np.asarray(T_list, dtype=np.float64)

    nFn = Fn_arr.shape[0]
    nT = T_arr.shape[0]

    # Find Fn bounds
    if Fn >= Fn_arr[nFn - 1]:
        Fn_idx = nFn - 2
        Fn_frac = 1.0
    elif Fn <= Fn_arr[0]:
        Fn_idx = 0
        Fn_frac = 0.0
    else:
        for i in range(nFn - 1):
            if Fn_arr[i] <= Fn <= Fn_arr[i + 1]:
                Fn_idx = i
                Fn_frac = (Fn - Fn_arr[i]) / (Fn_arr[i + 1] - Fn_arr[i])
                break

    # Find T bounds
    if T >= T_arr[nT - 1]:
        T_idx = nT - 2
        T_frac = 1.0
    elif T <= T_arr[0]:
        T_idx = 0
        T_frac = 0.0
    else:
        for i in range(nT - 1):
            if T_arr[i] <= T <= T_arr[i + 1]:
                T_idx = i
                T_frac = (T - T_arr[i]) / (T_arr[i + 1] - T_arr[i])
                break

    # Simulated 2D lookup via index math (T_idx * nFn + Fn_idx) avoids array reshaping overhead
    CwL = CW[T_idx * nFn + Fn_idx] + Fn_frac * (CW[T_idx * nFn + Fn_idx + 1] - CW[T_idx * nFn + Fn_idx])
    CwU = CW[(T_idx + 1) * nFn + Fn_idx] + Fn_frac * (CW[(T_idx + 1) * nFn + Fn_idx + 1] - CW[(T_idx + 1) * nFn + Fn_idx])
    
    return CwL + T_frac * (CwU - CwL)


cpdef tuple CalcDrag(
    double U,
    double LOA,
    double WL,
    double[::1] CW,
    double T,
    double SA,
    double rho=1025.0
):
    cdef double Cf, Rf, Fn, Cw, Rw

    Cf = Calc_Cf(U, WL)
    Rf = 0.5 * Cf * rho * SA * U * U

    Fn = U / sqrt(g * WL)
    Cw = interp_CW(Fn, T, CW)
    Rw = 0.5 * Cw * rho * LOA * LOA * U * U

    return (Rw + Rf), Fn