#!/usr/bin/env python3
"""Every number and figure coordinate used in the Session 5 decks.

Session 5 stops iterating on a grid.  Block A differentiates the equilibrium
conditions at a point (perturbation); Block B chooses coefficients so that a
residual vanishes (projection).  Both blocks end by solving the SAME quarterly
RBC model that 3.A4 and 4.B5 solved globally, so the three methods can be put
in one table.

Calibration continues Session 4's quarterly numbers (Quant_Macro labs/Lab7
RBCModel), which is also HW1 Part B Q3's:

    alpha = 0.33, beta = 0.99, delta = 0.025, rho = 0.95, sigma_eps = 0.007

Block A develops the theory on the source deck's own model -- log utility with
delta = 1, the Brock-Mirman benchmark of S01_B1:192 -- because that one has a
closed-form policy.

Output: tools/figures/s05_numbers.json -- the ledger.  No number appears on an
S5 slide unless it is a key in that file.  The Smolyak point counts are NOT
recomputed here: 5.B3 reuses key "smolyak" from s04_numbers.json, per
slides/SOURCES.md.

Run:  sbatch tools/run_code.slurm python3 tools/figures/s05_generate.py
"""

import json
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev as npcheb
from scipy.linalg import ordqz
from scipy.optimize import root
from scipy.stats import norm

# ----------------------------------------------------------------- calibration
ALPHA, BETA, DELTA = 0.33, 0.99, 0.025          # quarterly RBC, as in S4
RHO, SIG_EPS = 0.95, 0.007
BM_ALPHA, BM_BETA = 0.36, 0.95                  # Brock-Mirman, delta = 1, as in S3
TOL = 1e-6

OUT = {}


def coords(xs, ys, fmt="({:.4f},{:.4f})"):
    return " ".join(fmt.format(float(x), float(y)) for x, y in zip(xs, ys))


# ===================================================================== helpers
def rouwenhorst(n, rho, sigma):
    """Rouwenhorst (1995).  Matches rho and the unconditional variance exactly.
    Copied from tools/figures/s04_generate.py so the two ledgers use one chain."""
    p = (1.0 + rho) / 2.0
    P = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        Z = np.zeros((k, k))
        Z[:-1, :-1] += p * P
        Z[:-1, 1:] += (1 - p) * P
        Z[1:, :-1] += (1 - p) * P
        Z[1:, 1:] += p * P
        Z[1:-1, :] /= 2.0
        P = Z
    sz = sigma / np.sqrt(1.0 - rho ** 2)
    psi = sz * np.sqrt(n - 1)
    y = np.linspace(-psi, psi, n)
    return y, P


def gauss_hermite(n):
    """Nodes/weights for E[f(eps)], eps ~ N(0,1)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def steady_state(alpha, beta, delta):
    k = ((1.0 / beta - 1.0 + delta) / alpha) ** (1.0 / (alpha - 1.0))
    y = k ** alpha
    c = y - delta * k
    return k, y, c


# ================================================= first order by QZ (Klein 2000)
def klein(A, B, n_x):
    """Solve  A E_t s_{t+1} = B s_t  with s = (x; y), x predetermined.

    Returns P (x' = P x), F (y = F x), the generalized eigenvalue moduli sorted
    ascending, and the number of them outside the unit circle.  This is the
    generalized Schur (QZ) route of Klein (2000); Blanchard-Kahn is the count.
    """
    def stable(a, b):
        return np.abs(b) < np.abs(a)          # |lambda| = |b/a| < 1

    S, T, aa, bb, Q, Z = ordqz(A, B, sort=stable, output="real")
    lam = np.abs(bb) / np.where(np.abs(aa) < 1e-14, 1e-14, np.abs(aa))
    n_unstable = int(np.sum(lam > 1.0 + 1e-9))

    Z11 = Z[:n_x, :n_x]
    Z21 = Z[n_x:, :n_x]
    S11 = S[:n_x, :n_x]
    T11 = T[:n_x, :n_x]
    try:
        # When the count is wrong, Z11 is singular and there is no P, F to form.
        # That is not a numerical accident -- it is what Blanchard-Kahn failing
        # looks like from inside the algorithm.
        Z11i = np.linalg.inv(Z11)
        F = Z21 @ Z11i
        P = Z11 @ np.linalg.solve(S11, T11) @ Z11i
    except np.linalg.LinAlgError:
        P = F = None
    return P, F, np.sort(lam), n_unstable


def growth_matrices(alpha, beta, delta, rho):
    """The stochastic growth model log-linearized, in A E_t s' = B s form.

    s = (khat, z, chat);  x = (khat, z) predetermined, y = chat a jump.
      resource :  k* k'   = (alpha y* + (1-delta)k*) k  + y* z - c* c
      Euler    :  E c'    = c + kappa E z' + kappa(alpha-1) k'
      shock    :  E z'    = rho z
    """
    k, y, c = steady_state(alpha, beta, delta)
    kappa = alpha * k ** (alpha - 1.0) * beta          # share of R* that moves
    A = np.array([[k,                 0.0,    0.0],
                  [0.0,               1.0,    0.0],
                  [-kappa * (alpha - 1.0), -kappa, 1.0]])
    B = np.array([[alpha * y + (1 - delta) * k, y,   -c],
                  [0.0,                        rho,  0.0],
                  [0.0,                        0.0,  1.0]])
    return A, B, (k, y, c, kappa)


# ============================================ perturbation by matching residuals
def make_residuals(alpha, beta, delta, rho, sigma, order, n_gh=7):
    """Return R(theta) -> the stacked conditions the source deck describes:
    F = 0 and every derivative of F up to `order`, at the expansion point.

    The policy functions are written in log deviations,
        chat(k,z) = a0 + a1 k + a2 z + 1/2 (a3 k^2 + 2 a4 k z + a5 z^2)
        khat'(k,z)= b0 + b1 k + b2 z + 1/2 (b3 k^2 + 2 b4 k z + b5 z^2)
    so first order is 6 unknowns and second order is 12 -- which is exactly the
    "12 equations on 12 unknowns" of the source's second-order frame.
    """
    kss, yss, css = steady_state(alpha, beta, delta)
    xe, we = gauss_hermite(n_gh)

    def unpack(th):
        if order == 1:
            a = np.array([th[0], th[1], th[2], 0.0, 0.0, 0.0])
            b = np.array([th[3], th[4], th[5], 0.0, 0.0, 0.0])
        else:
            a, b = np.asarray(th[:6]), np.asarray(th[6:])
        return a, b

    def poly(p, kh, z):
        return (p[0] + p[1] * kh + p[2] * z
                + 0.5 * (p[3] * kh ** 2 + 2.0 * p[4] * kh * z + p[5] * z ** 2))

    def F(th, kh, z):
        a, b = unpack(th)
        c = max(css * np.exp(np.clip(poly(a, kh, z), -50, 50)), 1e-12)
        khp = poly(b, kh, z)
        kp = max(kss * np.exp(np.clip(khp, -50, 50)), 1e-12)
        k = kss * np.exp(kh)
        # resource constraint
        f2 = c + kp - np.exp(z) * k ** alpha - (1.0 - delta) * k
        # Euler, expectation by Gauss-Hermite over next period's innovation
        zp = rho * z + sigma * xe
        cp = np.maximum(css * np.exp(np.clip(poly(a, khp, zp), -50, 50)), 1e-12)
        R = alpha * np.exp(zp) * kp ** (alpha - 1.0) + (1.0 - delta)
        f1 = 1.0 / c - beta * np.sum(we * R / cp)
        return np.array([f1, f2])

    h = 1e-5 if order == 1 else 1e-3

    def conditions(th):
        f00 = F(th, 0.0, 0.0)
        fk = (F(th, h, 0.0) - F(th, -h, 0.0)) / (2 * h)
        fz = (F(th, 0.0, h) - F(th, 0.0, -h)) / (2 * h)
        out = [f00, fk, fz]
        if order >= 2:
            fkk = (F(th, h, 0.0) - 2 * f00 + F(th, -h, 0.0)) / h ** 2
            fzz = (F(th, 0.0, h) - 2 * f00 + F(th, 0.0, -h)) / h ** 2
            fkz = (F(th, h, h) - F(th, h, -h)
                   - F(th, -h, h) + F(th, -h, -h)) / (4 * h ** 2)
            out += [fkk, fkz, fzz]
        return np.concatenate(out)

    return conditions, (kss, yss, css)


def solve_perturbation(alpha, beta, delta, rho, sigma, order, guess=None):
    cond, ss = make_residuals(alpha, beta, delta, rho, sigma, order)
    n = 6 if order == 1 else 12
    if guess is None:
        guess = np.zeros(n)
        guess[1] = guess[4 if order == 1 else 7] = 0.5
    sol = root(cond, guess, method="hybr", tol=1e-13)
    return sol.x, float(np.max(np.abs(cond(sol.x)))), sol.success, ss


# =====================================================================  5.A1-A3
def run_brock_mirman():
    """The source deck's own model: log utility, delta = 1.  Its policy is known
    in closed form, and in LOGS it is exactly linear -- so first-order
    perturbation is not approximately right here, it is exactly right."""
    a, b = BM_ALPHA, BM_BETA
    kss = (a * b) ** (1.0 / (1.0 - a))
    css = kss ** a - kss

    A, B, (k_, y_, c_, kappa) = growth_matrices(a, b, 1.0, RHO)
    P, Fm, lam, n_unst = klein(A, B, 2)

    # closed form in logs: khat' = alpha khat + z, chat = alpha khat + z
    P_exact = np.array([[a, 1.0], [0.0, RHO]])
    F_exact = np.array([[a, 1.0]])
    err_P = float(np.max(np.abs(P - P_exact)))
    err_F = float(np.max(np.abs(Fm - F_exact)))

    # and in LEVELS, over a wide grid, against k' = alpha beta e^z k^alpha
    kg = np.linspace(0.25 * kss, 2.5 * kss, 400)
    zs = np.array([-0.05, 0.0, 0.05])
    worst = 0.0
    for z in zs:
        exact = a * b * np.exp(z) * kg ** a
        approx = kss * np.exp(P[0, 0] * np.log(kg / kss) + P[0, 1] * z)
        worst = max(worst, float(np.max(np.abs(approx / exact - 1.0))))

    # independent check: the residual-matching solver must find the same thing
    th1, res1, ok1, _ = solve_perturbation(a, b, 1.0, RHO, SIG_EPS, 1)
    th2, res2, ok2, _ = solve_perturbation(a, b, 1.0, RHO, SIG_EPS, 2,
                                           guess=np.concatenate([th1[:3], np.zeros(3),
                                                                 th1[3:], np.zeros(3)]))

    OUT["brock_mirman"] = {
        "alpha": a, "beta": b, "delta": 1.0, "rho": RHO, "sigma_eps": SIG_EPS,
        "k_ss": float(kss), "c_ss": float(css),
        "closed_form_policy": "k' = alpha beta e^z k^alpha",
        "log_policy": "khat' = alpha khat + z, exactly linear",
        "P_qz": [[float(v) for v in r] for r in P],
        "F_qz": [float(v) for v in Fm.ravel()],
        "max_abs_dev_P_from_exact": err_P,
        "max_abs_dev_F_from_exact": err_F,
        "max_rel_policy_error_in_levels": worst,
        "grid_span": "0.25 k* to 2.5 k*, z in {-0.05, 0, 0.05}",
        "eigenvalue_moduli": [float(v) for v in lam],
        "n_outside_unit_circle": n_unst,
        "n_jump_variables": 1,
        "residual_match_first_order": {
            "a0_a1_a2": [float(v) for v in th1[:3]],
            "b0_b1_b2": [float(v) for v in th1[3:]],
            "max_condition_residual": res1, "solved": bool(res1 < 1e-8),
            "agrees_with_qz_to": float(max(abs(th1[4] - P[0, 0]), abs(th1[5] - P[0, 1]),
                                           abs(th1[1] - Fm[0, 0]), abs(th1[2] - Fm[0, 1]))),
        },
        "second_order_terms": {
            "c_const_a0": float(th2[0]), "k_const_b0": float(th2[6]),
            "c_kk": float(th2[3]), "c_kz": float(th2[4]), "c_zz": float(th2[5]),
            "k_kk": float(th2[9]), "k_kz": float(th2[10]), "k_zz": float(th2[11]),
            "max_abs_second_order_term": float(np.max(np.abs(
                np.concatenate([th2[3:6], th2[9:12]])))),
            "max_condition_residual": res2, "solved": bool(res2 < 1e-7),
        },
        "note": ("delta=1 with log utility is exactly log-linear, so every "
                 "second-order term and the risk correction are zero.  The "
                 "benchmark cannot show perturbation error -- which is why "
                 "5.A6 moves to the quarterly model."),
    }


def run_first_order():
    """The quarterly RBC, first order, by QZ and independently by matching."""
    A, B, (kss, yss, css, kappa) = growth_matrices(ALPHA, BETA, DELTA, RHO)
    P, Fm, lam, n_unst = klein(A, B, 2)
    th1, res1, ok1, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, SIG_EPS, 1)

    # The QZ solution is the DETERMINISTIC expansion, so the independent check
    # must be run at sigma = 0 too; at sigma > 0 a linear FIT is a different
    # object and does move with the shock size.
    th0, res0, ok0, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, 0.0, 1)

    # impulse response to a one-standard-deviation innovation
    T = 40
    z = np.zeros(T); kh = np.zeros(T); ch = np.zeros(T)
    z[0] = SIG_EPS
    ch[0] = Fm[0, 0] * kh[0] + Fm[0, 1] * z[0]
    for t in range(T - 1):
        kh[t + 1] = P[0, 0] * kh[t] + P[0, 1] * z[t]
        z[t + 1] = RHO * z[t]
        ch[t + 1] = Fm[0, 0] * kh[t + 1] + Fm[0, 1] * z[t + 1]

    OUT["pert_first_order"] = {
        "calibration": {"alpha": ALPHA, "beta": BETA, "delta": DELTA,
                        "rho": RHO, "sigma_eps": SIG_EPS},
        "k_ss": float(kss), "y_ss": float(yss), "c_ss": float(css),
        "kappa": float(kappa),
        "P": [[float(v) for v in r] for r in P],
        "F": [float(v) for v in Fm.ravel()],
        "k_on_k": float(P[0, 0]), "k_on_z": float(P[0, 1]),
        "c_on_k": float(Fm[0, 0]), "c_on_z": float(Fm[0, 1]),
        "eigenvalue_moduli": [float(v) for v in lam],
        "n_outside_unit_circle": n_unst, "n_jump_variables": 1,
        "bk_satisfied": bool(n_unst == 1),
        "certainty_equivalence": {
            "a0_at_sigma0": float(th0[0]), "b0_at_sigma0": float(th0[3]),
            "claim": "c_chi = k_chi = 0: the shock scale enters no first-order "
                     "coefficient.  Measured in key 'higher_order' as "
                     "first_order_coeff_shift -- what adding the whole "
                     "second-order block does to the four linear coefficients.",
        },
        "qz_vs_matching_max_abs_diff": float(max(
            abs(th0[4] - P[0, 0]), abs(th0[5] - P[0, 1]),
            abs(th0[1] - Fm[0, 0]), abs(th0[2] - Fm[0, 1]))),
        "matching_residual": res0, "matching_solved": bool(res0 < 1e-8),
        "matching_note": "two independent routes to the same four numbers: "
                         "generalized Schur on the linearized system, and "
                         "root-finding on F=0, F_k=0, F_z=0",
        "irf_peak_c_pct": float(100 * ch.max()),
        "irf_peak_k_pct": float(100 * kh.max()),
        "irf_k_peak_quarter": int(np.argmax(kh)),
        "irf_halflife_z_quarters": float(np.log(0.5) / np.log(RHO)),
    }
    sel = np.arange(0, T, 1)
    OUT["fig_irf"] = {
        "k": coords(sel, 100 * kh[sel], "({:.0f},{:.4f})"),
        "c": coords(sel, 100 * ch[sel], "({:.0f},{:.4f})"),
        "z": coords(sel, 100 * z[sel], "({:.0f},{:.4f})"),
    }


def run_blanchard_kahn():
    """The count, and the three outcomes it distinguishes.  The growth model can
    only ever show case 1, so determinacy is illustrated on the textbook
    three-equation New Keynesian model, where the Taylor principle decides it."""
    rows = []
    for delta in (0.025, 1.0):
        alpha, beta = (ALPHA, BETA) if delta == 0.025 else (BM_ALPHA, BM_BETA)
        A, B, _ = growth_matrices(alpha, beta, delta, RHO)
        Pg, _, lam, n_unst = klein(A, B, 2)
        rows.append({"model": f"growth, delta={delta}", "alpha": alpha, "beta": beta,
                     "solution_exists": bool(Pg is not None),
                     "eigenvalue_moduli": [float(v) for v in lam],
                     "n_unstable": n_unst, "n_jumps": 1,
                     "verdict": "unique" if n_unst == 1 else
                                ("indeterminate" if n_unst < 1 else "no stable solution")})

    # three-equation NK model: pi = beta E pi' + kappa x ; x = E x' - (1/s)(i - E pi')
    kap, sig_is, rho_u = 0.1275, 1.0, 0.8
    nk = []
    for phi in (0.0, 0.8, 1.0, 1.5, 2.5):
        A = np.array([[1.0, 0.0,           0.0],
                      [0.0, BETA,          0.0],
                      [0.0, 1.0 / sig_is,  1.0]])
        B = np.array([[rho_u, 0.0,              0.0],
                      [0.0,   1.0,             -kap],
                      [0.0,   phi / sig_is,     1.0]])
        Pk, _, lam, n_unst = klein(A, B, 1)
        nk.append({"phi_pi": phi, "eigenvalue_moduli": [float(v) for v in lam],
                   "n_unstable": n_unst, "n_jumps": 2,
                   "solution_exists": bool(Pk is not None),
                   "verdict": "unique" if n_unst == 2 else
                              ("indeterminate" if n_unst < 2 else "no stable solution")})

    OUT["blanchard_kahn"] = {
        "rule": "unique iff (# generalized eigenvalues outside the unit circle) "
                "= (# non-predetermined variables)",
        "growth_models": rows,
        "new_keynesian": {"kappa": kap, "sigma_is": sig_is, "rho_u": rho_u,
                          "rows": nk,
                          "taylor_principle_threshold": 1.0},
        "method": "generalized Schur (QZ) with stable eigenvalues ordered first; "
                  "Klein (2000).  scipy.linalg.ordqz.",
    }


def run_higher_order():
    """Second order on the quarterly model: the risk correction is the constant
    the first-order solution is missing."""
    th1, _, _, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, SIG_EPS, 1)
    g0 = np.concatenate([th1[:3], np.zeros(3), th1[3:], np.zeros(3)])
    th2, res2, ok2, (kss, yss, css) = solve_perturbation(
        ALPHA, BETA, DELTA, RHO, SIG_EPS, 2, guess=g0)
    # baseline: the deterministic first-order expansion, which is what
    # "the first-order solution" means.  Caught by L05, which used sigma = 0.
    th1d, _, _, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, 0.0, 1)

    sweep = []
    for mult in (1, 5, 10, 20):
        s = SIG_EPS * mult
        t1, _, _, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, s, 1)
        gg = np.concatenate([t1[:3], np.zeros(3), t1[3:], np.zeros(3)])
        t2, r2, okk, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, s, 2, guess=gg)
        sweep.append({"sigma_eps": float(s), "sigma_multiple": mult,
                      "c_risk_correction_pct": float(100 * t2[0]),
                      "k_risk_correction_pct": float(100 * t2[6]),
                      "condition_residual": float(r2),
                      "solved": bool(r2 < 1e-8)})

    OUT["higher_order"] = {
        "unknowns_first_order": 6, "unknowns_second_order": 12,
        "conditions": "F=0, F_k=0, F_z=0, F_kk=0, F_kz=0, F_zz=0, each a 2-vector",
        "c_const_a0_pct": float(100 * th2[0]),
        "k_const_b0_pct": float(100 * th2[6]),
        "c_k": float(th2[1]), "c_z": float(th2[2]),
        "c_kk": float(th2[3]), "c_kz": float(th2[4]), "c_zz": float(th2[5]),
        "k_k": float(th2[7]), "k_z": float(th2[8]),
        "k_kk": float(th2[9]), "k_kz": float(th2[10]), "k_zz": float(th2[11]),
        "first_order_coeff_shift": float(max(abs(th2[1] - th1d[1]),
                                             abs(th2[2] - th1d[2]),
                                             abs(th2[7] - th1d[4]),
                                             abs(th2[8] - th1d[5]))),
        "first_order_coeff_shift_baseline": "deterministic (sigma = 0) first-order solve",
        "max_condition_residual": res2, "solved": bool(res2 < 1e-8),
        "risk_sweep": sweep,
        "reading": "the risk correction scales with sigma^2: ten times the shock "
                   "is a hundred times the correction",
    }


def simulate_second_order(b, rho, sigma, T=20000, seed=0, pruned=False):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T)
    z = 0.0
    if not pruned:
        kh = 0.0
        worst = 0.0
        for t in range(T):
            z = rho * z + sigma * eps[t]
            kh = (b[0] + b[1] * kh + b[2] * z
                  + 0.5 * (b[3] * kh ** 2 + 2 * b[4] * kh * z + b[5] * z ** 2))
            if not np.isfinite(kh) or abs(kh) > 1e3:
                return float("inf"), t
            worst = max(worst, abs(kh))
        return worst, T
    k1 = k2 = 0.0
    worst = 0.0
    for t in range(T):
        z = rho * z + sigma * eps[t]
        k1n = b[1] * k1 + b[2] * z
        k2n = (b[0] + b[1] * k2
               + 0.5 * (b[3] * k1 ** 2 + 2 * b[4] * k1 * z + b[5] * z ** 2))
        k1, k2 = k1n, k2n
        if not np.isfinite(k1 + k2) or abs(k1 + k2) > 1e3:
            return float("inf"), t
        worst = max(worst, abs(k1 + k2))
    return worst, T


def run_pruning():
    rows = []
    for mult in (1, 10, 30, 60):
        s = SIG_EPS * mult
        t1, _, _, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, s, 1)
        gg = np.concatenate([t1[:3], np.zeros(3), t1[3:], np.zeros(3)])
        t2, r2, ok, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, s, 2, guess=gg)
        b = t2[6:]
        wu, tu = simulate_second_order(b, RHO, s, pruned=False)
        wp, tp = simulate_second_order(b, RHO, s, pruned=True)
        rows.append({"sigma_multiple": mult, "sigma_eps": float(s),
                     "unpruned_max_abs_khat": (None if not np.isfinite(wu) else float(wu)),
                     "unpruned_exploded_at_t": (int(tu) if not np.isfinite(wu) else None),
                     "pruned_max_abs_khat": (None if not np.isfinite(wp) else float(wp)),
                     "pruned_exploded": bool(not np.isfinite(wp)),
                     "condition_residual": float(r2),
                     "solved": bool(r2 < 1e-8)})
    OUT["pruning"] = {
        "T": 20000, "seed": 0,
        "rows": rows,
        "rule": "Kim, Kim, Schaumburg & Sims (2008): propagate the first-order "
                "state, and feed only first-order terms into the second-order "
                "correction, so no term above the approximation order is ever "
                "squared.",
        "reference": "Andreasen, Fernandez-Villaverde & Rubio-Ramirez (2018) for "
                     "the general pruned state-space and its moments.",
    }


# =====================================================================  5.B1-B2
def run_projection_residual():
    """What a residual function looks like, and why 'solved' means 'R = 0'.

    The candidate is a Chebyshev polynomial in k fitted by collocation, so the
    residual is a genuine function of the state: it is zero AT the collocation
    nodes and non-zero between them, and it shrinks as the basis grows.

    Redrawn from scratch rather than copied: the source's figure is
    Fernandez-Villaverde & Guerron-Quintana's, uncredited.
    """
    a, b = BM_ALPHA, BM_BETA
    kss = (a * b) ** (1.0 / (1.0 - a))
    klo, khi = 0.4 * kss, 2.0 * kss

    def psi(k):
        return 2.0 * (k - klo) / (khi - klo) - 1.0

    def resid(th, k):
        kp = np.clip(npcheb.chebval(psi(k), th), 1e-10, None)
        c = k ** a - kp
        kpp = np.clip(npcheb.chebval(psi(kp), th), 1e-10, None)
        cp = kp ** a - kpp
        ok = (c > 0) & (cp > 0)
        out = np.full_like(k, np.nan)
        out[ok] = 1.0 - b * a * kp[ok] ** (a - 1.0) * c[ok] / cp[ok]
        return out

    kt = np.linspace(klo, khi, 2001)
    exact = a * b * kt ** a
    rows, series, nodes_out = [], {}, {}
    for n in (1, 3, 7):
        nodes = np.cos(np.pi * (2 * np.arange(1, n + 2) - 1) / (2 * (n + 1)))
        kn = klo + 0.5 * (nodes + 1.0) * (khi - klo)
        th0 = npcheb.chebfit(psi(kn), a * b * kn ** a * 1.15, n)   # deliberately off
        sol = root(lambda th: resid(th, kn), th0, method="hybr", tol=1e-13)
        th = sol.x
        R = resid(th, kt)
        g = npcheb.chebval(psi(kt), th)
        rows.append({"degree": n, "coefficients": n + 1,
                     "converged": bool(sol.success),
                     "max_abs_residual_at_nodes": float(np.max(np.abs(resid(th, kn)))),
                     "max_abs_residual_on_domain": float(np.nanmax(np.abs(R))),
                     "max_rel_policy_error": float(np.max(np.abs(g / exact - 1.0)))})
        sel = np.arange(0, kt.size, 20)
        series[f"n{n}"] = coords(kt[sel], R[sel], "({:.5f},{:.5f})")
        nodes_out[f"n{n}"] = coords(kn, np.zeros_like(kn), "({:.5f},{:.5f})")

    OUT["projection_residual"] = {
        "model": "Brock-Mirman, deterministic, delta=1; exact policy "
                 "k' = alpha beta k^alpha",
        "domain": [float(klo), float(khi)],
        "basis": "Chebyshev in k, collocation at the Chebyshev nodes",
        "rows": rows,
        "point": "the residual is a FUNCTION, not a number.  Collocation makes it "
                 "exactly zero at n+1 points and leaves it free in between; more "
                 "coefficients push the whole curve down.",
    }
    OUT["fig_residual"] = dict(series, kss=float(kss),
                               **{f"nodes_{k}": v for k, v in nodes_out.items()})


def _si_pi():
    """Si(pi) = int_0^pi sin(t)/t dt.  The Gibbs overshoot is (2/pi)Si(pi) - 1."""
    t = np.linspace(1e-15, np.pi, 2000001)
    return float(np.trapz(np.sin(t) / t, t))


def run_basis_conditioning():
    """Why not monomials: the measured reason."""
    # cond(H) is an error-amplification factor: storing b at all perturbs it by
    # about one machine epsilon, and the solution can move by cond times that.
    eps = float(np.finfo(float).eps)
    hil = []
    for n in (3, 5, 7, 9, 11, 13):
        H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        c = float(np.linalg.cond(H))
        hil.append({"n": n, "cond": c,
                    "digits_lost": float(np.log10(c)),
                    "rel_error_bound": c * eps})

    # Is the cond*eps bound attained?  Recover a KNOWN coefficient vector and
    # measure what actually comes back -- and, separately, what happens to the
    # fitted FUNCTION, which is a different question.
    rec = []
    rng = np.random.default_rng(0)
    for n in (3, 7, 9, 11, 13):
        H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        th_true = rng.standard_normal(n)
        b = H @ th_true
        th_hat = np.linalg.solve(H, b)
        xg = np.linspace(0.0, 1.0, 2001)
        Vg = np.vander(xg, n, increasing=True)
        rec.append({
            "n": n,
            "cond": float(np.linalg.cond(H)),
            "bound_cond_times_eps": float(np.linalg.cond(H) * eps),
            "actual_rel_error_theta": float(np.linalg.norm(th_hat - th_true)
                                            / np.linalg.norm(th_true)),
            "rel_error_of_fitted_function": float(
                np.max(np.abs(Vg @ (th_hat - th_true))) / np.max(np.abs(Vg @ th_true))),
        })

    # the entries of H, read as cosines of angles between basis functions
    ang = []
    for (a, b_) in ((1, 2), (5, 6), (10, 11)):
        cos = (1.0 / (a + b_ + 1)) / np.sqrt((1.0 / (2 * a + 1)) * (1.0 / (2 * b_ + 1)))
        ang.append({"i": a, "j": b_, "cos": float(cos),
                    "degrees": float(np.degrees(np.arccos(min(cos, 1.0))))})

    vand = []
    for n in (5, 9, 13, 17, 21):
        xs = np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))
        Vm = np.vander(xs, n, increasing=True)
        Vc = np.polynomial.chebyshev.chebvander(xs, n - 1)
        vand.append({"n": n, "cond_monomial": float(np.linalg.cond(Vm)),
                     "cond_chebyshev": float(np.linalg.cond(Vc))})

    # The payoff: same nodes, same solve, change only the basis.  Can the
    # coefficients be recovered at all?
    # Two design choices, not one: WHERE you evaluate, and WHICH basis.  Separate
    # them, because the nodes turn out to do most of the work and the basis does
    # the rest -- and only the basis makes it independent of degree.
    vrec = []
    rng2 = np.random.default_rng(1)
    for n in (5, 9, 13, 21, 41, 61):
        cheb_x = np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))
        even_x = np.linspace(-1.0, 1.0, n)
        row = {"n": n}
        for name, V in (
                ("mono_equispaced", np.vander(even_x, n, increasing=True)),
                ("mono_chebnodes", np.vander(cheb_x, n, increasing=True)),
                ("cheb_chebnodes", np.polynomial.chebyshev.chebvander(cheb_x, n - 1))):
            th_true = rng2.standard_normal(n)
            f = V @ th_true
            try:
                th_hat = np.linalg.solve(V, f)
                err = float(np.linalg.norm(th_hat - th_true) / np.linalg.norm(th_true))
            except np.linalg.LinAlgError:
                err = float("inf")
            row[f"cond_{name}"] = float(np.linalg.cond(V))
            row[f"rel_error_{name}"] = err
        vrec.append(row)

    # Where the Chebyshev sqrt(2) comes from: at the Chebyshev roots the columns of
    # V are orthogonal, so V'V is diagonal -- n for the constant column, n/2 for
    # the rest -- and cond(V) = sqrt(n / (n/2)) = sqrt(2), at every n.
    gram = []
    for n in (5, 9, 21):
        xs = np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))
        Vc = np.polynomial.chebyshev.chebvander(xs, n - 1)
        G = Vc.T @ Vc
        off = G - np.diag(np.diag(G))
        gram.append({"n": n,
                     "diag_first": float(G[0, 0]),
                     "diag_rest_min": float(np.min(np.diag(G)[1:])),
                     "diag_rest_max": float(np.max(np.diag(G)[1:])),
                     "max_abs_offdiagonal": float(np.max(np.abs(off))),
                     "cond_V": float(np.linalg.cond(Vc)),
                     "sqrt_ratio": float(np.sqrt(G[0, 0] / np.diag(G)[1]))})

    # Gibbs: Fourier partial sums of a square wave
    x = np.linspace(-np.pi, np.pi, 20001)
    gibbs = []
    series = {}
    for n in (5, 11, 21, 51, 101):
        s = np.zeros_like(x)
        for m in range(1, n + 1, 2):
            s += (4.0 / (np.pi * m)) * np.sin(m * x)
        gibbs.append({"terms": (n + 1) // 2, "max_partial_sum": float(s.max()),
                      "overshoot_pct_of_half_amplitude": float(100 * (s.max() - 1.0)),
                      "overshoot_pct_of_jump": float(100 * (s.max() - 1.0) / 2.0)})
        if n in (5, 21, 101):
            sel = np.arange(0, x.size, 40)
            series[f"n{n}"] = coords(x[sel], s[sel], "({:.4f},{:.4f})")

    # The source's own suggestion: "compare the graph of x^10 with x^11".
    xm = np.linspace(0.0, 1.0, 201)
    sel = np.arange(0, xm.size, 4)
    a, b = 10, 11
    cos_ab = (1.0 / (a + b + 1)) / np.sqrt((1.0 / (2 * a + 1)) * (1.0 / (2 * b + 1)))
    OUT["fig_monomials"] = {
        "x10": coords(xm[sel], xm[sel] ** a, "({:.3f},{:.4f})"),
        "x11": coords(xm[sel], xm[sel] ** b, "({:.3f},{:.4f})"),
        "cos_angle": float(cos_ab),
        "angle_degrees": float(np.degrees(np.arccos(min(cos_ab, 1.0)))),
        "note": "normalized inner product of x^10 and x^11 under <f,g> = int_0^1 f g dx",
    }

    OUT["basis_conditioning"] = {
        "hilbert": hil,
        "machine_eps": eps,
        "hilbert_angles": ang,
        "hilbert_recovery": rec,
        "vandermonde": vand,
        "chebyshev_gram": gram,
        "vandermonde_recovery": vrec,
        "gibbs": gibbs,
        "gibbs_limit_pct_of_half_amplitude": float(100 * (_si_pi() * 2.0 / np.pi - 1.0)),
        "gibbs_limit_pct_of_jump": float(100 * (_si_pi() * 2.0 / np.pi - 1.0) / 2.0),
        "gibbs_convention": "the textbook 8.95% is measured against the JUMP (here 2); "
                            "against the half-amplitude (here 1) the same number is 17.9%",
        "note": "the Hilbert matrix IS the least-squares normal matrix for "
                "monomials on [0,1]; its condition number is why the basis fails "
                "long before the approximation theory does",
    }
    OUT["fig_gibbs"] = dict(series, square="(-3.1416,-1.0) (0.0,-1.0) (0.0,1.0) (3.1416,1.0)")


def run_chebyshev():
    """Nodes, coefficient decay, and what smoothness buys."""
    def smooth(x):
        return np.exp(x) * np.sin(3.0 * x)          # entire: geometric decay

    def kinked(x):
        return np.abs(x)

    rows = []
    decay = {}
    for name, f in (("analytic", smooth), ("kinked", kinked)):
        errs = []
        for n in (5, 9, 17, 33, 65):
            nodes = np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))
            cf = npcheb.chebfit(nodes, f(nodes), n - 1)
            xt = np.linspace(-1, 1, 20001)
            e = float(np.max(np.abs(npcheb.chebval(xt, cf) - f(xt))))
            errs.append({"n": n, "max_error": e, "log10": float(np.log10(e))})
            if n == 33:
                decay[name] = coords(np.arange(len(cf)),
                                     np.log10(np.maximum(np.abs(cf), 1e-18)),
                                     "({:.0f},{:.3f})")
        rows.append({"function": name, "errors": errs})

    # tensor vs complete polynomial counts
    from math import comb
    counts = []
    for d in (1, 2, 3, 5, 10):
        for kap in (2, 4):
            counts.append({"d": d, "kappa": kap,
                           "tensor": int((kap + 1) ** d),
                           "complete": int(comb(kap + d, d))})

    OUT["chebyshev"] = {
        "nodes": "x_j = cos((2j-1)pi / 2n), the roots of T_n",
        "recursion": "T_{n+1} = 2 x T_n - T_{n-1}",
        "interpolation_theorem": "Chebyshev-node interpolation is within a factor "
                                 "(2/pi)log(n+1)+1 of the best possible polynomial "
                                 "of that degree",
        "lebesgue_factor_n33": float(2.0 / np.pi * np.log(34) + 1.0),
        "rows": rows,
        "tensor_vs_complete": counts,
        "reading": "analytic: error falls geometrically.  kinked: it falls like 1/n. "
                   "The basis is not the problem -- the function is.",
    }
    OUT["fig_cheb_decay"] = decay


# =====================================================================  5.B4
def tent(i, xs, x):
    """Piecewise-linear finite-element basis on a uniform mesh, written correctly."""
    h = xs[1] - xs[0]
    return np.clip(1.0 - np.abs(x - xs[i]) / h, 0.0, None)


def tent_as_printed(i, xs, x):
    """The source deck's formula, transcribed literally.

    Its second branch reads (x_{i-1} - x)/(x_i - x_{i-1}) on [x_i, x_{i+1}];
    at x = x_i that is -1, so the printed basis is not a partition of unity.
    """
    out = np.zeros_like(x)
    h = xs[1] - xs[0]
    if i > 0:
        m = (x >= xs[i - 1]) & (x <= xs[i])
        out[m] = (x[m] - xs[i - 1]) / h
    if i < len(xs) - 1:
        m = (x > xs[i]) & (x <= xs[i + 1])
        out[m] = (xs[i - 1] - x[m]) / h if i > 0 else 0.0
    return out


def run_fem():
    n = 9
    xs = np.linspace(0.0, 1.0, n)
    xt = np.linspace(0.0, 1.0, 5001)
    tot = np.zeros_like(xt)
    tot_bad = np.zeros_like(xt)
    for i in range(n):
        tot += tent(i, xs, xt)
        tot_bad += tent_as_printed(i, xs, xt)

    # assembled Galerkin matrix on the tent basis: tridiagonal by construction
    M = np.zeros((n, n))
    gl_x, gl_w = np.polynomial.legendre.leggauss(8)
    for e in range(n - 1):
        a_, b_ = xs[e], xs[e + 1]
        q = 0.5 * (b_ - a_) * gl_x + 0.5 * (a_ + b_)
        w = 0.5 * (b_ - a_) * gl_w
        for i in (e, e + 1):
            for j in (e, e + 1):
                M[i, j] += float(np.sum(w * tent(i, xs, q) * tent(j, xs, q)))
    nz = int(np.sum(np.abs(M) > 1e-14))

    OUT["fem"] = {
        "elements": n - 1, "nodes": n,
        "partition_of_unity_max_dev": float(np.max(np.abs(tot - 1.0))),
        "as_printed_in_source_max_dev": float(np.max(np.abs(tot_bad - 1.0))),
        "as_printed_value_at_node": float(tent_as_printed(4, xs, np.array([xs[4] + 1e-9]))[0]),
        "mass_matrix_nonzeros": nz,
        "mass_matrix_entries": n * n,
        "sparsity_pct": float(100.0 * nz / (n * n)),
        "bandwidth": 1,
        "refinements": ["h: more elements", "r: move the nodes", "p: raise the "
                        "degree inside each element"],
        "note": "compact support is the whole point: basis i and basis j overlap "
                "only if they share an element, so the assembled matrix is banded",
    }
    sel = np.arange(0, xt.size, 25)
    OUT["fig_tent"] = {
        "b3": coords(xt[sel], tent(3, xs, xt)),
        "b4": coords(xt[sel], tent(4, xs, xt)),
        "b5": coords(xt[sel], tent(5, xs, xt)),
        "sum": coords(xt[sel], tot[sel]),
    }


# =====================================================================  5.B5-B6
class Proj:
    """Chebyshev collocation / Galerkin machinery for the quarterly RBC."""

    def __init__(self, n_coef=7, n_z=7, width=0.5):
        self.alpha, self.beta, self.delta = ALPHA, BETA, DELTA
        self.logz, self.Pi = rouwenhorst(n_z, RHO, SIG_EPS)
        self.z = np.exp(self.logz)
        self.n_z, self.n = n_z, n_coef
        self.kss, self.yss, self.css = steady_state(ALPHA, BETA, DELTA)
        self.klo = (1 - width) * self.kss
        self.khi = (1 + width) * self.kss

    def psi(self, k):
        return 2.0 * (k - self.klo) / (self.khi - self.klo) - 1.0

    def cpol(self, th, k, j):
        return npcheb.chebval(self.psi(np.atleast_1d(k)), th[j])

    def resid(self, th, k, j):
        k = np.atleast_1d(k)
        c = self.cpol(th, k, j)
        c = np.maximum(c, 1e-10)
        y = self.z[j] * k ** self.alpha + (1 - self.delta) * k
        kp = np.clip(y - c, self.klo, self.khi)
        rhs = np.zeros_like(k)
        for l in range(self.n_z):
            cp = np.maximum(self.cpol(th, kp, l), 1e-10)
            R = self.alpha * self.z[l] * kp ** (self.alpha - 1) + (1 - self.delta)
            rhs += self.Pi[j, l] * R / cp
        return 1.0 - c * self.beta * rhs

    def guess(self):
        """Warm start from the first-order perturbation solution -- which is
        exactly how this is done in practice."""
        A, B, _ = growth_matrices(ALPHA, BETA, DELTA, RHO)
        _, Fm, _, _ = klein(A, B, 2)
        nodes = np.cos(np.pi * (2 * np.arange(1, self.n + 2) - 1) / (2 * (self.n + 1)))
        kn = self.klo + 0.5 * (nodes + 1.0) * (self.khi - self.klo)
        th = np.zeros((self.n_z, self.n + 1))
        for j in range(self.n_z):
            ch = Fm[0, 0] * np.log(kn / self.kss) + Fm[0, 1] * self.logz[j]
            th[j] = npcheb.chebfit(self.psi(kn), self.css * np.exp(ch), self.n)
        return th


def _weight_system(P, kind):
    """Return f(theta_flat) -> equations, one family per weight function."""
    n1 = P.n + 1
    gl_x, gl_w = np.polynomial.legendre.leggauss(40)
    kq = P.klo + 0.5 * (gl_x + 1.0) * (P.khi - P.klo)
    wq = 0.5 * (P.khi - P.klo) * gl_w
    cheb_roots = np.cos(np.pi * (2 * np.arange(1, n1 + 1) - 1) / (2 * n1))
    k_cheb = P.klo + 0.5 * (cheb_roots + 1.0) * (P.khi - P.klo)
    k_even = np.linspace(P.klo, P.khi, n1)
    edges = np.linspace(P.klo, P.khi, n1 + 1)

    def eqs(flat):
        th = flat.reshape(P.n_z, n1)
        out = []
        for j in range(P.n_z):
            if kind == "collocation_equispaced":
                out.append(P.resid(th, k_even, j))
            elif kind == "orthogonal_collocation":
                out.append(P.resid(th, k_cheb, j))
            else:
                Rq = P.resid(th, kq, j)
                if kind == "galerkin":
                    out.append(np.array([float(np.sum(wq * npcheb.chebval(
                        P.psi(kq), np.eye(n1)[i]) * Rq)) for i in range(n1)]))
                elif kind == "moments":
                    out.append(np.array([float(np.sum(wq * P.psi(kq) ** i * Rq))
                                         for i in range(n1)]))
                elif kind == "subdomain":
                    row = []
                    for i in range(n1):
                        m = (kq >= edges[i]) & (kq <= edges[i + 1])
                        row.append(float(np.sum(wq[m] * Rq[m])))
                    out.append(np.array(row))
        return np.concatenate(out)

    return eqs


def euler_err_from_cpol(cfun, P, n_test=2000, seed=0, band=None):
    """Unit-free Euler errors for ANY consumption policy c(k, j)."""
    rng = np.random.default_rng(seed)
    lo, hi = (P.klo, P.khi) if band is None else band
    kt = rng.uniform(lo, hi, n_test)
    jt = rng.integers(0, P.n_z, n_test)
    E = np.empty(n_test)
    for i in range(n_test):
        j = int(jt[i])
        c = float(cfun(np.array([kt[i]]), j)[0])
        if c <= 0:
            E[i] = np.nan
            continue
        y = P.z[j] * kt[i] ** P.alpha + (1 - P.delta) * kt[i]
        kp = min(max(y - c, P.klo), P.khi)
        rhs = 0.0
        for l in range(P.n_z):
            cp = float(cfun(np.array([kp]), l)[0])
            R = P.alpha * P.z[l] * kp ** (P.alpha - 1) + (1 - P.delta)
            rhs += P.Pi[j, l] * R / max(cp, 1e-12)
        E[i] = abs(1.0 - c * P.beta * rhs)
    E = E[np.isfinite(E)]
    return E


def run_collocation():
    """Six weight functions, one problem, one basis size."""
    P = Proj(n_coef=6)
    th0 = P.guess().ravel()
    rows = []
    for kind in ("orthogonal_collocation", "collocation_equispaced",
                 "galerkin", "subdomain", "moments"):
        eqs = _weight_system(P, kind)
        t0 = time.perf_counter()
        sol = root(eqs, th0, method="hybr", tol=1e-12)
        dt = time.perf_counter() - t0
        th = sol.x.reshape(P.n_z, P.n + 1)
        E = euler_err_from_cpol(lambda k, j, th=th: P.cpol(th, k, j), P)
        rows.append({"weight_function": kind,
                     "converged": bool(sol.success),
                     "seconds": round(dt, 3),
                     "residual_norm": float(np.max(np.abs(eqs(sol.x)))),
                     "max_log10_euler": float(np.log10(E.max())),
                     "mean_log10_euler": float(np.log10(E.mean()))})

    # least squares: minimise the integral of R^2 rather than zero it
    from scipy.optimize import least_squares
    gl_x, gl_w = np.polynomial.legendre.leggauss(40)
    kq = P.klo + 0.5 * (gl_x + 1.0) * (P.khi - P.klo)
    wq = 0.5 * (P.khi - P.klo) * gl_w

    def ls(flat):
        th = flat.reshape(P.n_z, P.n + 1)
        return np.concatenate([np.sqrt(wq) * P.resid(th, kq, j)
                               for j in range(P.n_z)])

    t0 = time.perf_counter()
    sol = least_squares(ls, th0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    dt = time.perf_counter() - t0
    th = sol.x.reshape(P.n_z, P.n + 1)
    E = euler_err_from_cpol(lambda k, j, th=th: P.cpol(th, k, j), P)
    rows.append({"weight_function": "least_squares", "converged": bool(sol.success),
                 "seconds": round(dt, 3),
                 "residual_norm": float(np.max(np.abs(ls(sol.x)))),
                 "max_log10_euler": float(np.log10(E.max())),
                 "mean_log10_euler": float(np.log10(E.mean()))})

    OUT["collocation"] = {
        "n_coefficients_per_z": P.n + 1, "n_z": P.n_z,
        "unknowns": (P.n + 1) * P.n_z,
        "quadrature": "Gauss-Legendre, 40 nodes, for every integral condition",
        "warm_start": "first-order perturbation",
        "rows": rows,
        "reading": "collocation zeroes the residual at points; Galerkin zeroes its "
                   "projection on the basis; least squares minimises it in L2.  "
                   "Same unknowns, same basis, different senses of 'small'.",
    }


def run_worked_proj():
    """The B6 walkthrough: the same quarterly RBC, solved by collocation."""
    rows = []
    best = None
    for n in (3, 5, 7, 9, 11):
        P = Proj(n_coef=n)
        eqs = _weight_system(P, "orthogonal_collocation")
        th0 = P.guess().ravel()
        t0 = time.perf_counter()
        sol = root(eqs, th0, method="hybr", tol=1e-12)
        dt = time.perf_counter() - t0
        th = sol.x.reshape(P.n_z, P.n + 1)
        E = euler_err_from_cpol(lambda k, j, th=th: P.cpol(th, k, j), P)
        rows.append({"degree": n, "unknowns": (n + 1) * P.n_z,
                     "seconds": round(dt, 3), "converged": bool(sol.success),
                     "max_log10_euler": float(np.log10(E.max())),
                     "mean_log10_euler": float(np.log10(E.mean())),
                     "coef_decay_log10": [float(np.log10(max(abs(v), 1e-18)))
                                          for v in th[P.n_z // 2]]})
        if n == 7:
            best = (P, th, dt, sol)

    P, th, dt, sol = best
    kg = np.linspace(P.klo, P.khi, 300)
    jm = P.n_z // 2
    c_mid = P.cpol(th, kg, jm)
    y_mid = P.z[jm] * kg ** P.alpha + (1 - P.delta) * kg
    g_mid = y_mid - c_mid
    sel = np.arange(0, 300, 6)
    OUT["worked_proj"] = {
        "degree": 7, "unknowns": (7 + 1) * P.n_z,
        "seconds": round(dt, 3),
        "k_ss": float(P.kss), "k_lo": float(P.klo), "k_hi": float(P.khi),
        "rows": rows,
        "coefficients_middle_z": [float(v) for v in th[jm]],
        "warm_start": "first-order perturbation coefficients, fitted at the "
                      "Chebyshev nodes",
        "saving_at_kss": float(np.interp(P.kss, kg, g_mid) - P.kss),
    }
    OUT["fig_proj_policy"] = {
        "saving_mid": coords(kg[sel], g_mid[sel] - kg[sel], "({:.4f},{:.5f})"),
        "kss": float(P.kss),
    }


def global_solution(P, n_k=300, n_h=20, tol=TOL, maxit=3000):
    """Session 4's continuous-choice solver, on Session 4's grid, so the
    three-method table is like for like.  Copied from s04_generate.py."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    kgr = np.linspace(P.klo, P.khi, n_k)
    y = P.z[None, :] * kgr[:, None] ** P.alpha + (1 - P.delta) * kgr[:, None]
    V = np.zeros((n_k, P.n_z))
    gk = np.tile(kgr[:, None] * 0.98, (1, P.n_z))
    t0 = time.perf_counter()
    outer = 0
    while outer < maxit:
        EV = V @ P.Pi.T
        Vn = np.empty_like(V)
        for j in range(P.n_z):
            a = np.full(n_k, kgr[0])
            b = np.minimum(y[:, j] - 1e-8, kgr[-1])
            c_ = b - phi * (b - a)
            d_ = a + phi * (b - a)

            def obj(kp, j=j, EV=EV):
                cc = y[:, j] - kp
                out = np.full_like(kp, -1e10)
                ok = cc > 1e-12
                out[ok] = np.log(cc[ok]) + P.beta * np.interp(kp[ok], kgr, EV[:, j])
                return out

            fc, fd = obj(c_), obj(d_)
            for _ in range(40):
                left = fc > fd
                b = np.where(left, d_, b)
                a = np.where(left, a, c_)
                c_ = b - phi * (b - a)
                d_ = a + phi * (b - a)
                fc, fd = obj(c_), obj(d_)
            gk[:, j] = 0.5 * (a + b)
            Vn[:, j] = obj(gk[:, j])
        d = np.max(np.abs(Vn - V))
        V = Vn
        outer += 1
        for _ in range(n_h):
            EV = V @ P.Pi.T
            cc = np.maximum(y - gk, 1e-12)
            EVi = np.empty_like(V)
            for j in range(P.n_z):
                EVi[:, j] = np.interp(gk[:, j], kgr, EV[:, j])
            V = np.log(cc) + P.beta * EVi
        if d < tol:
            break
    secs = time.perf_counter() - t0
    cg = y - gk

    def cfun(k, j):
        return np.interp(k, kgr, cg[:, j])

    return cfun, outer, secs, n_k * P.n_z


def run_three_methods():
    """The payoff frame of the whole session: one model, three methods."""
    P = Proj(n_coef=7)
    band_wide = (P.klo, P.khi)
    band_near = (0.95 * P.kss, 1.05 * P.kss)

    cfun_g, it_g, sec_g, store_g = global_solution(P)

    A, B, _ = growth_matrices(ALPHA, BETA, DELTA, RHO)
    t0 = time.perf_counter()
    Pm, Fm, _, _ = klein(A, B, 2)
    sec_p1 = time.perf_counter() - t0

    def c1(k, j):
        kh = np.log(np.atleast_1d(k) / P.kss)
        return P.css * np.exp(Fm[0, 0] * kh + Fm[0, 1] * P.logz[j])

    th1, _, _, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, SIG_EPS, 1)
    g0 = np.concatenate([th1[:3], np.zeros(3), th1[3:], np.zeros(3)])
    t0 = time.perf_counter()
    th2, _, ok2, _ = solve_perturbation(ALPHA, BETA, DELTA, RHO, SIG_EPS, 2, guess=g0)
    sec_p2 = time.perf_counter() - t0
    a = th2[:6]

    def c2(k, j):
        kh = np.log(np.atleast_1d(k) / P.kss)
        z = P.logz[j]
        ch = (a[0] + a[1] * kh + a[2] * z
              + 0.5 * (a[3] * kh ** 2 + 2 * a[4] * kh * z + a[5] * z ** 2))
        return P.css * np.exp(ch)

    eqs = _weight_system(P, "orthogonal_collocation")
    th0 = P.guess().ravel()
    t0 = time.perf_counter()
    sol = root(eqs, th0, method="hybr", tol=1e-12)
    sec_pr = time.perf_counter() - t0
    thp = sol.x.reshape(P.n_z, P.n + 1)

    def cpr(k, j):
        return P.cpol(thp, k, j)

    rows = []
    for name, f, store, secs in (
            ("global VFI (4.B5)", cfun_g, store_g, sec_g),
            ("perturbation, 1st order", c1, 4, sec_p1),
            ("perturbation, 2nd order", c2, 12, sec_p2),
            ("Chebyshev collocation", cpr, (P.n + 1) * P.n_z, sec_pr)):
        Ew = euler_err_from_cpol(f, P, band=band_wide)
        En = euler_err_from_cpol(f, P, band=band_near)
        rows.append({"method": name, "numbers_stored": int(store),
                     "solve_seconds": round(float(secs), 3),
                     "max_log10_euler_wide": float(np.log10(Ew.max())),
                     "mean_log10_euler_wide": float(np.log10(Ew.mean())),
                     "max_log10_euler_near": float(np.log10(En.max())),
                     "mean_log10_euler_near": float(np.log10(En.mean()))})

    # where perturbation stops being good enough, measured
    ks = np.linspace(P.klo, P.khi, 60)
    jm = P.n_z // 2
    e1 = []
    for kk in ks:
        Ei = euler_err_from_cpol(c1, P, n_test=40, band=(kk * 0.999, kk * 1.001))
        e1.append(float(np.log10(Ei.max())))
    s4 = None
    try:
        with open("tools/figures/s04_numbers.json") as fh:
            s4 = json.load(fh)["worked_stochastic"]["continuous"]
    except Exception:
        pass
    check = None
    if s4 is not None:
        check = {"s04_continuous_max_log10_euler": s4["max_log10_euler"],
                 "s05_rerun_max_log10_euler": rows[0]["max_log10_euler_wide"],
                 "s04_policy_updates": s4["policy_updates"],
                 "s05_rerun_policy_updates": it_g,
                 "agrees": bool(abs(s4["max_log10_euler"]
                                    - rows[0]["max_log10_euler_wide"]) < 0.25
                                and s4["policy_updates"] == it_g)}

    OUT["three_methods"] = {
        "model": "quarterly RBC, Rouwenhorst 7 states, k in [0.5 k*, 1.5 k*]",
        "s04_cross_check": check,
        "k_ss": float(P.kss),
        "global_sweeps": it_g,
        "rows": rows,
        "bands": {"wide": [float(v) for v in band_wide],
                  "near": [float(v) for v in band_near]},
        "second_order_solved": bool(ok2 is not None),
    }
    OUT["fig_pert_error"] = {
        "first_order": coords(ks, e1, "({:.3f},{:.3f})"),
        "kss": float(P.kss),
    }


ALL_RUNS = (run_brock_mirman, run_first_order, run_blanchard_kahn,
            run_higher_order, run_pruning, run_projection_residual,
            run_basis_conditioning, run_chebyshev, run_fem,
            run_collocation, run_worked_proj, run_three_methods)


if __name__ == "__main__":
    # With no arguments, regenerate the whole ledger.  With arguments, run only
    # the named sections and MERGE them into the existing JSON -- so that adding
    # one figure does not silently move every measured timing already on a slide.
    wanted = sys.argv[1:]
    merge = bool(wanted)
    if merge:
        by_name = {fn.__name__.replace("run_", ""): fn for fn in ALL_RUNS}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise SystemExit(f"unknown section(s) {missing}; "
                             f"choose from {sorted(by_name)}")
        runs = [by_name[w] for w in wanted]
    else:
        runs = list(ALL_RUNS)

    for fn in runs:
        t = time.perf_counter()
        fn()
        print(f"  {fn.__name__:28s} {time.perf_counter() - t:7.1f}s", flush=True)

    path = "tools/figures/s05_numbers.json"
    if merge:
        base = json.load(open(path))
        base.update(OUT)
        base.setdefault("_meta", {})["last_partial_update"] = " ".join(wanted)
        OUT.clear()
        OUT.update(base)
    else:
        OUT["_meta"] = {
            "calibration": "quarterly RBC, Quant_Macro Lab7 (Block A theory and "
                           "Block B use the Brock-Mirman delta=1 benchmark where "
                           "a closed form is needed)",
            "alpha": ALPHA, "beta": BETA, "delta": DELTA,
            "rho": RHO, "sigma_eps": SIG_EPS,
            "bm_alpha": BM_ALPHA, "bm_beta": BM_BETA,
            "tol": TOL,
            "numpy": np.__version__,
            "smolyak": "NOT recomputed here -- 5.B3 reuses key 'smolyak' from "
                       "tools/figures/s04_numbers.json, per slides/SOURCES.md",
            "generated_by": "tools/figures/s05_generate.py",
        }
    with open(path, "w") as f:
        json.dump(OUT, f, indent=2)
    print(f"wrote {path}  ({len(OUT)} keys)")
