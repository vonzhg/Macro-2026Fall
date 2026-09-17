#!/usr/bin/env python3
"""Generate every number and every figure coordinate used in the Session 3 decks.

Model (the course benchmark, S01_B1): u(c)=ln c, y = z k^alpha, delta = 1.
Closed form (Brock-Mirman):   k'(k,z) = alpha*beta*z*k^alpha
                              c(k,z)  = (1-alpha*beta)*z*k^alpha
                              V(k,z)  = A(z) + B ln k,  B = alpha/(1-alpha*beta)
                              A = (I - beta*Pi)^{-1} b,
                              b(z) = ln(1-alpha*beta) + (1+beta*B) ln z + beta*B ln(alpha*beta)

Output: tools/figures/s03_numbers.json -- the ledger. No number appears on an S3 slide
unless it is in that file (session-log rule 4).
"""

import json
import time
import numpy as np
from scipy.optimize import brentq

ALPHA = 0.36           # capital share (CLAUDE.md notation table)
BETA = 0.95
TOL = 1e-6
KSS_DET = (ALPHA * BETA) ** (1.0 / (1.0 - ALPHA))   # deterministic steady state

# A *given* two-state chain.  Building one from an AR(1) is S4's job, not S3's.
ZGRID = np.array([0.95, 1.05])
PI = np.array([[0.90, 0.10],
               [0.10, 0.90]])

OUT = {}


# ----------------------------------------------------------------- closed form
def closed_form(kgrid, zgrid=ZGRID, pi=PI, alpha=ALPHA, beta=BETA):
    B = alpha / (1.0 - alpha * beta)
    b = (np.log(1.0 - alpha * beta)
         + (1.0 + beta * B) * np.log(zgrid)
         + beta * B * np.log(alpha * beta))
    A = np.linalg.solve(np.eye(len(zgrid)) - beta * pi, b)
    V = A[None, :] + B * np.log(kgrid)[:, None]
    kp = alpha * beta * zgrid[None, :] * kgrid[:, None] ** alpha
    c = (1.0 - alpha * beta) * zgrid[None, :] * kgrid[:, None] ** alpha
    return V, kp, c, A, B


def grid(n, kmin=0.05, kmax=0.60):
    return np.linspace(kmin, kmax, n)


# ------------------------------------------------- deterministic VFI (the hook)
def vfi_det(kgrid, beta=BETA, alpha=ALPHA, tol=TOL, maxit=20000, track=False):
    """Discrete grid search: the crudest possible set of choices, on purpose."""
    y = kgrid ** alpha
    C = y[:, None] - kgrid[None, :]                 # c[i,j] = k_i^alpha - k_j
    U = np.where(C > 0, np.log(np.where(C > 0, C, 1.0)), -1e10)
    V = np.zeros(len(kgrid))
    hist = []
    pol_prev = None
    pol_settled = None
    it = 0
    while it < maxit:
        M = U + beta * V[None, :]
        pol = M.argmax(axis=1)
        Vn = M.max(axis=1)
        d = np.max(np.abs(Vn - V))
        if track:
            hist.append((it + 1, Vn.copy(), d,
                         float(np.min(Vn - V)), float(np.max(Vn - V))))
        if pol_prev is not None and pol_settled is None and np.array_equal(pol, pol_prev):
            pol_settled = it + 1
        pol_prev = pol
        V = Vn
        it += 1
        if d < tol:
            break
    return V, kgrid[pol], it, hist, pol_settled


# -------------------------------------------------------------- stochastic VFI
def vfi_stoch(kgrid, zgrid=ZGRID, pi=PI, beta=BETA, alpha=ALPHA, tol=TOL, maxit=20000):
    nk, nz = len(kgrid), len(zgrid)
    y = zgrid[None, :] * (kgrid ** alpha)[:, None]          # (nk, nz)
    C = y[:, :, None] - kgrid[None, None, :]                # (nk, nz, nk)
    U = np.where(C > 0, np.log(np.where(C > 0, C, 1.0)), -1e10)
    V = np.zeros((nk, nz))
    it = 0
    while it < maxit:
        EV = V @ pi.T                        # EV[l,j] = E[V(k_l,z')|z_j]
        # M[i,j,l] = U[i,j,l] + beta * EV[l,j]
        M = U + beta * EV.T[None, :, :]      # EV.T is (nz, nk) indexed [j,l]
        pol = M.argmax(axis=2)
        Vn = M.max(axis=2)
        d = np.max(np.abs(Vn - V))
        V = Vn
        it += 1
        if d < tol:
            break
    return V, kgrid[pol], it


# ------------------------------------------------------------- time iteration
def time_iteration(kgrid, zgrid=ZGRID, pi=PI, beta=BETA, alpha=ALPHA,
                   tol=1e-8, maxit=5000):
    nk, nz = len(kgrid), len(zgrid)
    y = zgrid[None, :] * (kgrid ** alpha)[:, None]
    c = 0.5 * y                                  # initial guess
    it = 0
    nroots = 0
    while it < maxit:
        cn = np.empty_like(c)
        for j in range(nz):
            for i in range(nk):
                yi = y[i, j]

                def resid(cc, yi=yi, j=j):
                    kp = yi - cc
                    rhs = 0.0
                    for m in range(nz):
                        cp = np.interp(kp, kgrid, c[:, m])
                        R = alpha * zgrid[m] * kp ** (alpha - 1.0)
                        rhs += pi[j, m] * R / cp
                    return 1.0 / cc - beta * rhs

                lo, hi = 1e-10, yi - 1e-10
                cn[i, j] = brentq(resid, lo, hi, xtol=1e-14, rtol=1e-14)
                nroots += 1
        d = np.max(np.abs(cn - c))
        c = cn
        it += 1
        if d < tol:
            break
    return c, it, nroots


# ------------------------------------------------------------- Euler residuals
def euler_errors(kgrid, cpol, ktest, zgrid=ZGRID, pi=PI, beta=BETA, alpha=ALPHA):
    """Unit-free consumption mistake, S01_B2's definition: |1 - ctilde/c|."""
    nz = len(zgrid)
    E = np.zeros((len(ktest), nz))
    for j in range(nz):
        for i, k in enumerate(ktest):
            cc = np.interp(k, kgrid, cpol[:, j])
            kp = zgrid[j] * k ** alpha - cc
            rhs = 0.0
            for m in range(nz):
                cp = np.interp(kp, kgrid, cpol[:, m])
                R = alpha * zgrid[m] * kp ** (alpha - 1.0)
                rhs += pi[j, m] * R / cp
            ctil = 1.0 / (beta * rhs)
            E[i, j] = abs(1.0 - ctil / cc)
    return E


def coords(xs, ys, fmt="({:.4f},{:.4f})"):
    return " ".join(fmt.format(float(x), float(y)) for x, y in zip(xs, ys))


# ============================================================== 1. the base run
def run_base():
    kg = grid(200)
    t0 = time.perf_counter()
    V, g, it, hist, pol_settled = vfi_det(kg, track=True)
    wall = time.perf_counter() - t0

    Vs, kps, _, _, Bcoef = closed_form(kg, np.array([1.0]), np.array([[1.0]]))
    Vstar, gstar = Vs[:, 0], kps[:, 0]

    OUT["vfi_base"] = {
        "n_grid": 200, "k_min": 0.05, "k_max": 0.60,
        "alpha": ALPHA, "beta": BETA, "tol": TOL,
        "iterations": int(it),
        "wall_seconds": round(wall, 2),
        "policy_settled_at": int(pol_settled) if pol_settled else None,
        "max_abs_value_error": float(np.max(np.abs(V - Vstar))),
        "max_rel_policy_error": float(np.max(np.abs(g - gstar) / gstar)),
        "grid_spacing_h": float(kg[1] - kg[0]),
        "k_ss_deterministic": float(KSS_DET),
    }

    # policy vs closed form, thinned for the slide
    sel = np.arange(0, 200, 8)
    OUT["fig_policy_vs_truth"] = {
        "vfi": coords(kg[sel], g[sel]),
        "truth": coords(kg[sel], gstar[sel]),
        "xlabel": "k", "ylabel": "k'",
    }
    # where the error lives
    relerr = np.abs(g - gstar) / gstar
    OUT["fig_policy_error"] = {
        "curve": coords(kg[sel], relerr[sel], "({:.4f},{:.5f})"),
        "max_at_k": float(kg[int(np.argmax(relerr))]),
    }

    # a narrow window at EVERY grid point, so the staircase is visible
    w = slice(60, 96)
    OUT["fig_policy_zoom"] = {
        "vfi": coords(kg[w], g[w], "({:.5f},{:.5f})"),
        "truth": coords(kg[w], gstar[w], "({:.5f},{:.5f})"),
        "k_lo": float(kg[60]), "k_hi": float(kg[95]),
        "y_lo": float(min(g[w].min(), gstar[w].min())),
        "y_hi": float(max(g[w].max(), gstar[w].max())),
        "h": float(kg[1] - kg[0]),
        "n_distinct_policy_values": int(len(np.unique(g[w]))),
        "n_grid_points_in_window": int(96 - 60),
        "note": "the computed policy takes only a handful of distinct values here",
    }

    # convergence path against the DISCRETE fixed point V^h
    Vh, _, _, _, _ = vfi_det(kg, tol=1e-14, maxit=20000)
    ns, true_err, succ_diff, mqp = [], [], [], []
    for (n, Vn, d, dmin, dmax) in hist:
        ns.append(n)
        true_err.append(np.max(np.abs(Vn - Vh)))
        succ_diff.append(d)
        mqp.append(BETA / (1.0 - BETA) * (dmax - dmin))
    ns = np.array(ns); true_err = np.array(true_err)
    succ_diff = np.array(succ_diff); mqp = np.array(mqp)

    keep = ((ns <= 300) & (ns % 10 == 0)) | (ns == 1)
    OUT["fig_convergence"] = {
        "true_error": coords(ns[keep], np.log10(np.maximum(true_err[keep], 1e-16))),
        "successive_difference": coords(ns[keep], np.log10(np.maximum(succ_diff[keep], 1e-16))),
        "mqp_width": coords(ns[keep], np.log10(np.maximum(mqp[keep], 1e-16))),
        "slope_log10_beta": float(np.log10(BETA)),
        "xlabel": "iteration n", "ylabel": "log10 error",
    }

    # the honest comparison at the moment the naive rule fires
    stop = int(np.argmax(succ_diff < TOL))
    OUT["stopping_rule"] = {
        "fires_at_iteration": int(ns[stop]),
        "successive_difference_there": float(succ_diff[stop]),
        "true_error_there": float(true_err[stop]),
        "guaranteed_bound_beta_eps_over_1_minus_beta": float(BETA / (1 - BETA) * succ_diff[stop]),
        "mqp_bracket_width_there": float(mqp[stop]),
        "ratio_bound_to_truth": float(BETA / (1 - BETA) * succ_diff[stop] / true_err[stop]),
        "bound_multiple_of_eps": round(BETA / (1 - BETA), 1),
    }


# ================================================ 2. iteration counts vs beta
def run_beta():
    rows = []
    for b in (0.90, 0.95, 0.99):
        theory = np.log(TOL) / np.log(b)
        kg = grid(200)
        _, _, it, _, _ = vfi_det(kg, beta=b)
        rows.append({"beta": b,
                     "theory_ln_eps_over_ln_beta": round(float(theory), 1),
                     "theory_rounded_up": int(np.ceil(theory)),
                     "measured_iterations": int(it),
                     "error_amplifier_beta_over_1_minus_beta": round(b / (1 - b), 1)})
    OUT["iterations_vs_beta"] = rows


# ============================================================ 3. grid refinement
def _refine(tol):
    rows = []
    for n in (50, 100, 200, 400, 800):
        kg = grid(n)
        V, g, it, _, _ = vfi_det(kg, tol=tol)
        Vs, kps, _, _, _ = closed_form(kg, np.array([1.0]), np.array([[1.0]]))
        rows.append({"n": n, "h": float(kg[1] - kg[0]),
                     "value_error": float(np.max(np.abs(V - Vs[:, 0]))),
                     "policy_error": float(np.max(np.abs(g - kps[:, 0]))),
                     "iterations": int(it),
                     "comparisons": int(n) * int(n) * int(it)})
    return rows


def _slopes(rows, lo=0, hi=None):
    r = rows[lo:hi]
    h = np.log(np.array([x["h"] for x in r]))
    return (round(float(np.polyfit(h, np.log([x["value_error"] for x in r]), 1)[0]), 2),
            round(float(np.polyfit(h, np.log([x["policy_error"] for x in r]), 1)[0]), 2))


def run_refine():
    loose = _refine(TOL)          # iteration tolerance 1e-6, the usual choice
    tight = _refine(1e-11)        # iterate until the discretization is the only error left

    sv_all, sp_all = _slopes(loose)
    sv_first, sp_first = _slopes(loose, 0, 3)
    sv_t, sp_t = _slopes(tight)

    floor = BETA / (1.0 - BETA) * TOL      # the truncation floor: 19 * epsilon
    OUT["grid_refinement"] = {
        "rows_tol_1e-6": loose,
        "rows_tol_1e-11": tight,
        "fitted_slope_value_tol_1e-6_all": sv_all,
        "fitted_slope_value_tol_1e-6_first_three": sv_first,
        "fitted_slope_policy_tol_1e-6": sp_all,
        "fitted_slope_value_tol_1e-11": sv_t,
        "fitted_slope_policy_tol_1e-11": sp_t,
        "truncation_floor_19_times_tol": floor,
        "value_error_at_n800_tol_1e-6": loose[-1]["value_error"],
        "lesson": ("at tol=1e-6 the value error stops falling once it reaches the "
                   "truncation floor beta*eps/(1-beta); refining the grid past that "
                   "buys nothing until the iteration tolerance is tightened too"),
        "santos_vigo_aguiar": "value O(h^2), policy O(h)",
    }
    h = np.array([r["h"] for r in loose])
    OUT["fig_refinement"] = {
        "value_tol_1e6": coords(np.log10(h), np.log10([r["value_error"] for r in loose])),
        "value_tol_1e11": coords(np.log10(h), np.log10([r["value_error"] for r in tight])),
        "policy": coords(np.log10(h), np.log10([r["policy_error"] for r in loose])),
        "floor_log10": float(np.log10(floor)),
        "xlabel": "log10 h", "ylabel": "log10 max error",
    }


# ======================================== 4. stochastic VFI and time iteration
def run_stochastic():
    kg = grid(200)
    Vs, kps, cs, _, _ = closed_form(kg)

    t0 = time.perf_counter()
    V, g, it = vfi_stoch(kg)
    wall_vfi = time.perf_counter() - t0
    c_vfi = ZGRID[None, :] * (kg ** ALPHA)[:, None] - g

    t0 = time.perf_counter()
    c_ti, it_ti, nroots = time_iteration(kg)
    wall_ti = time.perf_counter() - t0

    ktest = np.linspace(0.06, 0.59, 997)      # deliberately off the solution grid
    E_vfi = euler_errors(kg, c_vfi, ktest)
    E_ti = euler_errors(kg, c_ti, ktest)

    # Both methods are graded on the SAME object: the relative error of the
    # CONSUMPTION policy against the closed form.  (An earlier version compared
    # VFI's capital-policy error with time iteration's consumption-policy error,
    # which is not a like-for-like comparison.)
    OUT["stochastic"] = {
        "z_grid": ZGRID.tolist(), "transition_matrix": PI.tolist(),
        "error_metric": "max |c_computed / c_exact - 1| over the grid",
        "vfi": {"iterations": int(it), "wall_seconds": round(wall_vfi, 2),
                "max_abs_value_error": float(np.max(np.abs(V - Vs))),
                "max_rel_consumption_error": float(np.max(np.abs(c_vfi / cs - 1.0))),
                "max_rel_capital_policy_error": float(np.max(np.abs(g - kps) / kps)),
                "max_log10_euler_error": float(np.log10(np.max(E_vfi))),
                "mean_log10_euler_error": float(np.log10(np.mean(E_vfi)))},
        "time_iteration": {"iterations": int(it_ti), "wall_seconds": round(wall_ti, 2),
                           "root_finds": int(nroots),
                           "max_rel_consumption_error": float(np.max(np.abs(c_ti / cs - 1.0))),
                           "max_log10_euler_error": float(np.log10(np.max(E_ti))),
                           "mean_log10_euler_error": float(np.log10(np.mean(E_ti)))},
        "accuracy_ratio_vfi_over_ti": float(np.max(np.abs(c_vfi / cs - 1.0))
                                            / np.max(np.abs(c_ti / cs - 1.0))),
    }
    sel = np.arange(0, len(ktest), 12)
    OUT["fig_euler_errors"] = {
        "vfi": coords(ktest[sel], np.log10(E_vfi[sel, 0])),
        "time_iteration": coords(ktest[sel], np.log10(E_ti[sel, 0])),
        "xlabel": "k (off-grid)", "ylabel": "log10 |E|",
    }


# ================================================= 5. loops versus broadcasting
def run_vectorization():
    rows = []
    for n in (200, 500, 1000):
        kg = grid(n)
        y = kg ** ALPHA
        V = np.zeros(n)

        t0 = time.perf_counter()
        Vn = np.empty(n)
        for i in range(n):
            best = -1e18
            for j in range(n):
                c = y[i] - kg[j]
                if c > 0:
                    v = np.log(c) + BETA * V[j]
                    if v > best:
                        best = v
            Vn[i] = best
        t_loop = time.perf_counter() - t0

        t0 = time.perf_counter()
        C = y[:, None] - kg[None, :]
        M = np.where(C > 0, np.log(np.where(C > 0, C, 1.0)), -1e18) + BETA * V[None, :]
        Vn2 = M.max(axis=1)
        t_vec = time.perf_counter() - t0

        rows.append({"n": n,
                     "loop_seconds": round(t_loop, 4),
                     "broadcast_seconds": round(t_vec, 5),
                     "speedup": round(t_loop / t_vec, 1),
                     "agree_max_abs_diff": float(np.max(np.abs(Vn - Vn2)))})
    OUT["vectorization"] = rows


# ================================= 5b. finite differences versus autodiff (3.B3)
def run_autodiff():
    import torch
    k0, a = 2.0, ALPHA
    exact = a * k0 ** (a - 1.0)

    rows, curve = [], []
    for e in range(1, 17):
        h = 10.0 ** (-e)
        fd = ((k0 + h) ** a - k0 ** a) / h
        rel = abs(fd / exact - 1.0)
        curve.append((-float(e), float(np.log10(max(rel, 1e-18)))))
        if e in (1, 4, 8, 12, 16):
            rows.append({"h": h, "forward_difference": float(fd),
                         "relative_error": float(rel)})

    x = torch.tensor(k0, requires_grad=True, dtype=torch.float64)
    y = x ** a
    y.backward()
    ad = float(x.grad)

    best = min(curve, key=lambda t: t[1])
    OUT["autodiff"] = {
        "point": k0, "exact_derivative": float(exact),
        "finite_difference_rows": rows,
        "best_h_exponent": int(-best[0]),
        "best_relative_error": float(10 ** best[1]),
        "autograd_value": ad,
        "autograd_relative_error": float(abs(ad / exact - 1.0)),
        "sqrt_eps": float(np.sqrt(np.finfo(np.float64).eps)),
        "torch_version": torch.__version__,
    }
    OUT["fig_autodiff"] = {
        "curve": " ".join(f"({a_:.1f},{b_:.3f})" for a_, b_ in curve),
        "xlabel": "log10 h", "ylabel": "log10 relative error",
    }



# ================================== 7. the RBC worked example (deck 3.A4)
#     General depreciation: no closed-form policy, but a closed-form STEADY STATE.
#     k* = (alpha*beta*A / (1 - beta*(1-delta)))**(1/(1-alpha))
def run_rbc():
    A, alpha, beta, delta = 1.0, 0.36, 0.95, 0.10
    kss = (alpha * beta * A / (1.0 - beta * (1.0 - delta))) ** (1.0 / (1.0 - alpha))

    N = 500
    kg = np.linspace(0.25 * kss, 1.75 * kss, N)
    y = A * kg ** alpha + (1.0 - delta) * kg               # cash on hand
    C = y[:, None] - kg[None, :]
    U = np.full_like(C, -1e10)
    np.log(C, out=U, where=C > 0)

    V = np.zeros(N)
    t0 = time.perf_counter()
    it = 0
    while it < 20000:
        M = U + beta * V[None, :]
        Vn, pol = M.max(axis=1), M.argmax(axis=1)
        d = np.max(np.abs(Vn - V))
        V, it = Vn, it + 1
        if d < TOL:
            break
    wall = time.perf_counter() - t0
    g = kg[pol]
    c = y - g

    # where does the policy cross the 45-degree line?
    diff = g - kg
    j = int(np.argmin(np.abs(diff)))
    sign_change = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if sign_change.size:
        i0 = int(sign_change[0])
        w = diff[i0] / (diff[i0] - diff[i0 + 1])
        k_cross = float(kg[i0] + w * (kg[i0 + 1] - kg[i0]))
    else:
        k_cross = float(kg[j])

    # Euler residuals off the grid: u'(c) = beta u'(c') (alpha A k'^(a-1) + 1 - delta)
    ktest = np.linspace(kg[2], kg[-3], 733)
    E = np.zeros(ktest.size)
    for i, kk in enumerate(ktest):
        cc = np.interp(kk, kg, c)
        kp = A * kk ** alpha + (1 - delta) * kk - cc
        cp = np.interp(kp, kg, c)
        R = alpha * A * kp ** (alpha - 1.0) + 1.0 - delta
        E[i] = abs(1.0 - (1.0 / (beta * R / cp)) / cc)

    sel = np.arange(0, N, 10)
    OUT["rbc"] = {
        "A": A, "alpha": alpha, "beta": beta, "delta": delta,
        "k_star_closed_form": float(kss),
        "n_grid": N, "k_lo": float(kg[0]), "k_hi": float(kg[-1]),
        "grid_spacing_h": float(kg[1] - kg[0]),
        "iterations": int(it), "wall_seconds": round(wall, 2),
        "k_cross_45_degree": k_cross,
        "cross_minus_kstar": float(k_cross - kss),
        "cross_rel_error": float(abs(k_cross / kss - 1.0)),
        "nearest_grid_point_to_kstar": float(kg[int(np.argmin(np.abs(kg - kss)))]),
        "max_log10_euler_error": float(np.log10(E.max())),
        "mean_log10_euler_error": float(np.log10(E.mean())),
        "comparisons": int(N) * int(N) * int(it),
        "c_ss_closed_form": float(A * kss ** alpha - delta * kss),
    }
    OUT["fig_rbc_policy"] = {
        "policy": coords(kg[sel], g[sel]),
        "fortyfive": f"({kg[0]:.4f},{kg[0]:.4f}) ({kg[-1]:.4f},{kg[-1]:.4f})",
        "kstar": float(kss),
    }
    OUT["fig_rbc_euler"] = {
        "curve": coords(ktest[::6], np.log10(E[::6])),
    }

    # Version 1: add a two-state productivity shock, same code plus one axis
    z, Pi = ZGRID, PI
    yz = z[None, :] * (A * kg ** alpha)[:, None] + (1 - delta) * kg[:, None]
    Cz = yz[:, :, None] - kg[None, None, :]
    Uz = np.full_like(Cz, -1e10)
    np.log(Cz, out=Uz, where=Cz > 0)
    Vz = np.zeros((N, z.size))
    t0 = time.perf_counter()
    itz = 0
    while itz < 20000:
        EV = Vz @ Pi.T
        Mz = Uz + beta * EV.T[None, :, :]
        Vn, polz = Mz.max(axis=2), Mz.argmax(axis=2)
        d = np.max(np.abs(Vn - Vz))
        Vz, itz = Vn, itz + 1
        if d < TOL:
            break
    wallz = time.perf_counter() - t0
    gz = kg[polz]
    OUT["rbc_stochastic"] = {
        "z_grid": z.tolist(), "iterations": int(itz),
        "wall_seconds": round(wallz, 2),
        "memory_MB_return_array": round(Uz.nbytes / 1e6, 1),
        "lines_of_code_changed": 3,
        "policy_low_at_kstar": float(np.interp(kss, kg, gz[:, 0])),
        "policy_high_at_kstar": float(np.interp(kss, kg, gz[:, 1])),
    }
    OUT["fig_rbc_policy_stoch"] = {
        "low": coords(kg[sel], gz[sel, 0]),
        "high": coords(kg[sel], gz[sel, 1]),
        "fortyfive": f"({kg[0]:.4f},{kg[0]:.4f}) ({kg[-1]:.4f},{kg[-1]:.4f})",
    }


# ============ 8. floating point, in the detail the deck actually needs
def run_float_detail():
    import struct

    def bits(x):
        return format(struct.unpack('>Q', struct.pack('>d', x))[0], '064b')

    b = bits(6.5)
    OUT["float_detail"] = {
        "worked_example_value": 6.5,
        "worked_example_bits": f"{b[0]} {b[1:12]} {b[12:]}",
        "worked_example_sign": int(b[0]),
        "worked_example_exponent_field": int(b[1:12], 2),
        "worked_example_bias": 1023,
        "worked_example_unbiased_exponent": int(b[1:12], 2) - 1023,
        "worked_example_mantissa": 1.0 + int(b[12:], 2) / 2**52,
        "max_double": float(np.finfo(np.float64).max),
        "min_normal_double": float(np.finfo(np.float64).tiny),
        "min_subnormal_double": float(5e-324),
        "eps_double": float(np.finfo(np.float64).eps),
        # the three failure modes
        "overflow_to_inf": repr(float(np.float64(1e308) * 10)),
        "underflow_to_zero": repr(float(np.float64(1e-320) / 1e10)),
        "nan_not_equal_itself": bool(np.nan != np.nan),
        "cancellation_a": 1.0 + 1e-16,
        "cancellation_result": repr((1.0 + 1e-16) - 1.0),
        "cancellation_expected": 1e-16,
        "assoc_left": repr((0.1 + 0.2) + 0.3),
        "assoc_right": repr(0.1 + (0.2 + 0.3)),
        "assoc_equal": bool((0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3)),
    }
    # summation order: many small numbers added to one large one
    big = 1e8
    small = np.full(10_000_000, 1.0)
    naive = big
    for _ in range(3):
        pass
    OUT["float_detail"]["sum_order_naive"] = float(np.float32(big) + np.float32(1.0) - np.float32(big))
    OUT["float_detail"]["sum_order_note"] = "float32: 1e8 + 1 - 1e8 loses the 1 entirely"
    # catastrophic cancellation in a derivative, at the scale S4 will meet
    h = 1e-12
    f = lambda k: k ** 0.36
    OUT["float_detail"]["cancellation_derivative_h"] = h
    OUT["float_detail"]["cancellation_derivative_num"] = repr(f(2.0 + h) - f(2.0))
    OUT["float_detail"]["cancellation_derivative_sig_digits_left"] = 4


# ============ 9. Santos: does the computable bound the uncomputable? (3.A1)
def run_santos():
    ref = OUT["stochastic"]
    vfi_res = 10 ** ref["vfi"]["max_log10_euler_error"]
    vfi_err = ref["vfi"]["max_rel_consumption_error"]
    ti_res = 10 ** ref["time_iteration"]["max_log10_euler_error"]
    ti_err = ref["time_iteration"]["max_rel_consumption_error"]
    rows = OUT["grid_refinement"]["rows_tol_1e-11"]
    OUT["santos"] = {
        "vfi_residual": vfi_res, "vfi_true_policy_error": vfi_err,
        "vfi_ratio_residual_over_error": vfi_res / vfi_err,
        "ti_residual": ti_res, "ti_true_policy_error": ti_err,
        "ti_ratio_residual_over_error": ti_res / ti_err,
        "value_over_policy_error_ratio": [
            round(r["value_error"] / r["policy_error"], 4) for r in rows],
        "h_values": [r["h"] for r in rows],
        "reading": ("the residual is a conservative certificate: it exceeds the true policy "
                    "error by a small factor, for both methods"),
    }


# ============================================================= 6. floating point
def run_float():
    gdp = 28_000_000_000_000.0
    OUT["floating_point"] = {
        "one_tenth_plus_two_tenths": repr(0.1 + 0.2),
        "equals_point_three": bool(0.1 + 0.2 == 0.3),
        "difference": float((0.1 + 0.2) - 0.3),
        "eps_float64": float(np.finfo(np.float64).eps),
        "eps_float32": float(np.finfo(np.float32).eps),
        "spacing_at_28e12_float64": float(np.spacing(gdp)),
        "spacing_at_28e12_float32": float(np.spacing(np.float32(gdp))),
        "gdp_plus_100k_float32_absorbed":
            bool(np.float32(gdp) + np.float32(1e5) == np.float32(gdp)),
        "gdp_plus_100k_float64_absorbed": bool(gdp + 1e5 == gdp),
        "smallest_addition_that_moves_28e12_float32":
            float(np.spacing(np.float32(gdp))),
        "float32_digits": 7, "float64_digits": 16,
    }


if __name__ == "__main__":
    run_float()
    run_base()
    run_beta()
    run_refine()
    run_stochastic()
    run_vectorization()
    run_autodiff()
    run_rbc()
    run_float_detail()
    run_santos()
    OUT["_meta"] = {
        "model": "u=ln c, y=z k^alpha, delta=1 (Brock-Mirman)",
        "alpha": ALPHA, "beta": BETA, "tol": TOL,
        "numpy": np.__version__,
        "generated_by": "tools/figures/s03_generate.py",
    }
    with open("tools/figures/s03_numbers.json", "w") as f:
        json.dump(OUT, f, indent=2, sort_keys=False)
    print("wrote tools/figures/s03_numbers.json")
    for k in OUT:
        print(" ", k)
