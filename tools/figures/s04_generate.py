#!/usr/bin/env python3
"""Every number and figure coordinate used in the Session 4 decks.

Session 4 opens the boxes Session 3 sealed, so each block ends in a measured
pay-off.  Calibration moves to the instructor's own QUARTERLY numbers
(Quant_Macro labs/Lab7 RBCModel), which is also HW1 Part B Q3's calibration:

    alpha = 0.33, beta = 0.99, delta = 0.025, rho = 0.95, sigma_eps = 0.007

Output: tools/figures/s04_numbers.json -- the ledger.  No number appears on an
S4 slide unless it is a key in that file.
"""

import json
import time

import numpy as np
from numpy.polynomial import chebyshev as npcheb
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import norm

# ----------------------------------------------------------------- calibration
ALPHA, BETA, DELTA = 0.33, 0.99, 0.025
RHO, SIG_EPS = 0.95, 0.007
TOL = 1e-6

OUT = {}


def coords(xs, ys, fmt="({:.4f},{:.4f})"):
    return " ".join(fmt.format(float(x), float(y)) for x, y in zip(xs, ys))


# =====================================================================  A1
# Root-finding.  Market clearing, the example that works (Quant_Macro Lab4).
def demand(p):
    return 100.0 * p ** -0.5


def supply(p):
    return np.exp(0.5 * p) - 1.0


def excess(p):
    return demand(p) - supply(p)


def d_excess(p):
    return -50.0 * p ** -1.5 - 0.5 * np.exp(0.5 * p)


def run_rootfinding():
    pstar = brentq(excess, 0.1, 20.0, xtol=1e-15, rtol=8.9e-16)

    def track(step, x0, x1=None, maxit=200):
        """Run an iteration, return the error history |x_n - pstar|."""
        hist = []
        a, b = 0.1, 20.0
        x, xm1 = x0, x1
        for _ in range(maxit):
            if step == "bisect":
                m = 0.5 * (a + b)
                if excess(a) * excess(m) <= 0:
                    b = m
                else:
                    a = m
                x = 0.5 * (a + b)
            elif step == "newton":
                x = x - excess(x) / d_excess(x)
            elif step == "secant":
                f0, f1 = excess(xm1), excess(x)
                if f1 == f0:
                    break
                x, xm1 = x - f1 * (x - xm1) / (f1 - f0), x
            hist.append(abs(x - pstar))
            if hist[-1] < 1e-15:
                break
        return np.array(hist)

    h_bis = track("bisect", 10.0)
    h_new = track("newton", 10.0)
    h_sec = track("secant", 6.0, 9.0)

    def order(h):
        """Observed convergence order from the last clean triple."""
        h = h[h > 1e-14]
        if h.size < 4:
            return None
        e2, e1, e0 = h[-1], h[-2], h[-3]
        return float(np.log(e2 / e1) / np.log(e1 / e0))

    def to(h, tgt):
        idx = np.where(h < tgt)[0]
        return int(idx[0] + 1) if idx.size else None

    OUT["rootfinding"] = {
        "problem": "100 p^-0.5 = exp(0.5 p) - 1",
        "p_star": float(pstar),
        "q_star": float(demand(pstar)),
        "methods": {
            "bisection": {"iters_to_1e-6": to(h_bis, 1e-6), "iters_to_1e-12": to(h_bis, 1e-12),
                          "observed_order": order(h_bis), "theoretical_order": 1.0,
                          "needs": "a bracket with a sign change"},
            "newton": {"iters_to_1e-6": to(h_new, 1e-6), "iters_to_1e-12": to(h_new, 1e-12),
                       "observed_order": order(h_new), "theoretical_order": 2.0,
                       "needs": "the derivative"},
            "secant": {"iters_to_1e-6": to(h_sec, 1e-6), "iters_to_1e-12": to(h_sec, 1e-12),
                       "observed_order": order(h_sec), "theoretical_order": 1.618,
                       "needs": "two starting points"},
        },
    }
    keep = lambda h: [(i + 1, float(np.log10(max(v, 1e-16)))) for i, v in enumerate(h)]
    OUT["fig_rootfinding"] = {
        "bisection": " ".join(f"({i},{v:.3f})" for i, v in keep(h_bis)),
        "newton": " ".join(f"({i},{v:.3f})" for i, v in keep(h_new)),
        "secant": " ".join(f"({i},{v:.3f})" for i, v in keep(h_sec)),
    }

    # ---- the cautionary example: an equation with no root -------------------
    def bad_excess(p):
        p = np.maximum(p, 1e-12)
        return (100 - 2 * p + 10 * np.sqrt(p)) - (-20 + 3 * p - 0.1 * p ** 2)

    pg = np.linspace(0.01, 200, 20000)
    vals = bad_excess(pg)

    def bad_d(p):
        h = 1e-7
        return (bad_excess(p + h) - bad_excess(p - h)) / (2 * h)

    xn = 20.0
    for _ in range(50):
        step = bad_excess(xn) / bad_d(xn)
        if not np.isfinite(step):
            break
        xn = xn - step
    xa, xb = 15.0, 25.0
    for _ in range(50):
        f0, f1 = bad_excess(xa), bad_excess(xb)
        if f1 == f0 or not np.isfinite(f1):
            break
        xa, xb = xb, xb - f1 * (xb - xa) / (f1 - f0)
    OUT["no_root"] = {
        "excess_demand": "(100 - 2p + 10 sqrt p) - (-20 + 3p - 0.1 p^2)",
        "min_over_grid": float(vals.min()),
        "argmin_p": float(pg[int(np.argmin(vals))]),
        "is_always_positive": bool(vals.min() > 0),
        "bracketing_methods": "ValueError: f(a) and f(b) must have different signs",
        "newton_returns": float(xn),
        "newton_residual_there": float(bad_excess(xn)),
        "secant_returns": float(xb),
        "secant_residual_there": float(bad_excess(xb)),
        "note": ("the bracketing methods refuse; the open methods return a number, "
                 "and neither number is a root"),
    }

    # ---- systems: Newton vs Broyden ----------------------------------------
    def F(v):
        x, y = v
        return np.array([x ** 2 + y ** 2 - 4.0, np.exp(x) + y - 1.0])

    def J(v):
        x, y = v
        return np.array([[2 * x, 2 * y], [np.exp(x), 1.0]])

    v = np.array([1.0, -1.0])
    n_newton = 0
    for _ in range(100):
        d = np.linalg.solve(J(v), -F(v))
        v = v + d
        n_newton += 1
        if np.max(np.abs(F(v))) < 1e-12:
            break
    root = v.copy()

    w = np.array([1.0, -1.0])
    B = np.eye(2)
    n_broy, n_jac = 0, 0
    for _ in range(200):
        d = np.linalg.solve(B, -F(w))
        w_new = w + d
        yv = F(w_new) - F(w)
        B = B + np.outer(yv - B @ d, d) / (d @ d)
        w = w_new
        n_broy += 1
        if np.max(np.abs(F(w))) < 1e-12:
            break
    OUT["systems"] = {
        "system": "x^2+y^2=4,  e^x+y=1",
        "root": root.tolist(),
        "newton_final_residual": float(np.max(np.abs(F(root)))),
        "newton_iterations": n_newton,
        "newton_jacobians_built": n_newton,
        "broyden_iterations": n_broy,
        "broyden_jacobians_built": 1,
        "broyden_final_residual": float(np.max(np.abs(F(w)))),
    }

    # ---- damped fixed-point iteration ---------------------------------------
    g = lambda x: 4.0 - x ** 2 / 4.0          # fixed point at x = 2*(sqrt(5)-1)
    xfix = 2.0 * (np.sqrt(5.0) - 1.0)
    res = {}
    for damp in (1.0, 0.5, 0.2):
        x = 1.0
        for n in range(1, 501):
            x = (1 - damp) * x + damp * g(x)
            if abs(x - xfix) < 1e-10:
                break
        res[f"damping_{damp}"] = {"iterations": n, "converged": bool(abs(x - xfix) < 1e-10)}
    OUT["damped_fixed_point"] = {"map": "g(x) = 4 - x^2/4", "fixed_point": float(xfix), **res}


# =====================================================================  A2/A3
def run_optimization():
    # golden section on a concave objective
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    f = lambda x: -(x - 2.7) ** 2 + 3.0

    def golden(a, b, tol):
        n = 0
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = f(c), f(d)
        n += 2
        while b - a > tol:
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = f(d)
            n += 1
        return 0.5 * (a + b), n

    rows = []
    for tol in (1e-4, 1e-6, 1e-8):
        _, n = golden(0.0, 10.0, tol)
        rows.append({"tol": tol, "golden_evaluations": n,
                     "theory_log_over_log_phi": round(float(np.log(tol / 10.0) / np.log(phi)), 1)})
    OUT["golden_section"] = {
        "phi": float(phi), "tau": float(1 - phi),
        "rows": rows,
        "grid_search_evaluations_for_same_accuracy": {str(t): int(np.ceil(10.0 / t)) for t in (1e-4, 1e-6, 1e-8)},
        "correction": ("cost is O(log(1/eps)), independent of the grid size M; "
                       "the source's 'O(log M), 24 evaluations at M=1000' is a tolerance count"),
    }

    # the BFGS worked example from QM-4, recomputed exactly
    fq = lambda v: 0.5 * v[0] ** 2 + v[1] ** 2
    gq = lambda v: np.array([v[0], 2 * v[1]])
    H = np.array([[1.0, 0.0], [0.0, 2.0]])
    th0 = np.array([2.0, 2.0])
    g0 = gq(th0)
    B0 = np.eye(2)
    d0 = -np.linalg.solve(B0, g0)
    a0 = 0.5
    th1 = th0 + a0 * d0
    g1 = gq(th1)
    s0, y0 = th1 - th0, g1 - g0
    B1 = B0 + np.outer(y0, y0) / (y0 @ s0) - (B0 @ np.outer(s0, s0) @ B0) / (s0 @ B0 @ s0)
    a_exact = float((g0 @ g0) / (d0 @ H @ d0))
    OUT["bfgs_worked"] = {
        "f": "0.5 x^2 + y^2", "theta0": th0.tolist(), "g0": g0.tolist(),
        "d0": d0.tolist(), "alpha0_used_in_source": a0,
        "alpha0_exact_minimiser": a_exact,
        "theta1": th1.tolist(), "g1": g1.tolist(),
        "s0": s0.tolist(), "y0": y0.tolist(),
        "y0_dot_s0": float(y0 @ s0), "s0_B0_s0": float(s0 @ B0 @ s0),
        "B1": B1.tolist(),
        "secant_residual_exact": float(np.max(np.abs(B1 @ s0 - y0))),
        "H_s0": (H @ s0).tolist(),
        "note": ("the secant condition B1 s0 = y0 holds EXACTLY; the source's "
                 "'approximately' is an artifact of rounding B1 to 2 decimals"),
    }

    # the conjugate-gradient example, with the source's error corrected
    d0c = -g0
    a0c = float((g0 @ g0) / (d0c @ H @ d0c))
    th1c = th0 + a0c * d0c
    g1c = gq(th1c)
    beta_fr = float((g1c @ g1c) / (g0 @ g0))
    d1c = -g1c + beta_fr * d0c
    a1c = float((g1c @ g1c) / (d1c @ H @ d1c))
    th2c = th1c + a1c * d1c
    OUT["cg_worked"] = {
        "alpha0": a0c, "theta1": th1c.tolist(), "g1": g1c.tolist(),
        "beta_fletcher_reeves": beta_fr,
        "d1_correct": d1c.tolist(),
        "d1_as_printed_in_source": [-80 / 81, 40 / 81],
        "alpha1": a1c, "theta2": th2c.tolist(),
        "converged_in_n_steps": 2,
        "note": "source prints d1 = [-80/81, 40/81]; the second component is 20/81",
    }

    # Nelder-Mead vs BFGS on Rosenbrock (derivative-free vs gradient-based)
    from scipy.optimize import minimize
    ros = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
    out = {}
    for meth in ("Nelder-Mead", "BFGS", "Powell"):
        r = minimize(ros, np.array([-1.2, 1.0]), method=meth,
                     options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12}
                     if meth == "Nelder-Mead" else {"maxiter": 20000})
        out[meth] = {"f_evaluations": int(r.nfev), "final_f": float(r.fun),
                     "distance_to_optimum": float(np.max(np.abs(r.x - 1.0)))}
    OUT["derivative_free"] = {"problem": "Rosenbrock from (-1.2, 1)", **out}

    # the cost of a Newton step in N dimensions
    OUT["newton_cost"] = {
        str(N): {"hessian_entries": N * N, "factorisation_flops": int(N ** 3 / 3)}
        for N in (10, 100, 1000)
    }


# ============ large-scale and derivative-free: CG, SGD, simulated annealing
def run_large_scale():
    H = np.array([[1.0, 0.0], [0.0, 2.0]])
    gq = lambda v: np.array([v[0], 2 * v[1]])

    # conjugate gradient: exact in at most N steps on an N-dimensional quadratic
    th = np.array([2.0, 2.0])
    g = gq(th); d = -g
    path = [th.copy()]
    n = 0
    for n in range(1, 5):
        a = float((g @ g) / (d @ H @ d))
        th = th + a * d
        g_new = gq(th)
        beta_fr = float((g_new @ g_new) / (g @ g))
        d = -g_new + beta_fr * d
        g = g_new
        path.append(th.copy())
        if np.max(np.abs(g)) < 1e-14:
            break
    OUT["cg_steps"] = {"path": [p.tolist() for p in path], "steps_to_exact": n,
                       "dimension": 2,
                       "claim": "exact in at most N steps on an N-dimensional quadratic"}

    OUT["optimizer_memory"] = {
        str(N): {"newton_or_bfgs": N * N, "lbfgs_m10": 20 * N, "cg": 3 * N}
        for N in (100, 10_000, 1_000_000)}

    # SGD on the same function split into two per-observation losses:
    #   f = l1 + l2, l1 = 0.5 x^2, l2 = y^2  ->  grad f = [x, 2y], matching BFGS and CG
    grads = {1: lambda v: np.array([v[0], 0.0]),
             2: lambda v: np.array([0.0, 2 * v[1]])}
    th = np.array([2.0, 2.0])
    rows = []
    for i in (2, 1, 1, 2, 1):
        gi = grads[i](th)
        nxt = th - 0.5 * gi
        rows.append({"theta": [round(x, 4) for x in th.tolist()], "drawn": f"l{i}",
                     "grad": [round(x, 4) for x in gi.tolist()],
                     "next": [round(x, 4) for x in nxt.tolist()]})
        th = nxt
    OUT["sgd_trajectory"] = {
        "f": "l1 + l2 with l1 = 0.5 x^2 and l2 = y^2",
        "full_gradient_at_start": gq(np.array([2.0, 2.0])).tolist(),
        "learning_rate": 0.5, "rows": rows,
        "source_defect": ("the source defines f as the SUM but differentiates with 1/M "
                          "averaging, giving [1,2] rather than [2,4] and contradicting its "
                          "own trajectory table"),
    }
    OUT["sgd_dimension"] = {str(N): float(np.sqrt(N))
                            for N in (1, 100, 10_000, 1_000_000)}

    # simulated annealing against a local method, on a multimodal objective
    f = lambda x: -(x ** 2) / 20.0 + 3.0 * np.cos(3.0 * x)
    xs = np.linspace(-12, 12, 200001)
    xstar = float(xs[int(np.argmax(f(xs)))])
    from scipy.optimize import minimize_scalar
    loc = minimize_scalar(lambda x: -f(x), bracket=(7.0, 8.0, 9.0))
    rng = np.random.default_rng(3)
    x = best = 8.0
    for it in range(1, 20001):
        T = 5.0 / np.log(it + 1.0)
        prop = x + rng.normal(0, 1.5)
        if -12 <= prop <= 12 and (f(prop) > f(x)
                                  or rng.random() < np.exp((f(prop) - f(x)) / T)):
            x = prop
        if f(x) > f(best):
            best = x
    OUT["simulated_annealing"] = {
        "objective": "-(x^2)/20 + 3 cos(3x) on [-12, 12]",
        "global_max_x": xstar, "global_max_f": float(f(xstar)),
        "local_from_x0_8": float(loc.x), "local_f": float(-loc.fun),
        "annealing_from_x0_8": float(best), "annealing_f": float(f(best)),
        "iterations": 20000,
        "annealing_found_global": bool(abs(best - xstar) < 0.25),
    }


# =====================================================================  A4
def run_differentiation():
    f = lambda x: x ** ALPHA
    x0 = 2.0
    exact = ALPHA * x0 ** (ALPHA - 1.0)

    fwd, ctr, cplx = [], [], []
    for e in range(1, 17):
        h = 10.0 ** (-e)
        ef = abs(((f(x0 + h) - f(x0)) / h) / exact - 1.0)
        ec = abs(((f(x0 + h) - f(x0 - h)) / (2 * h)) / exact - 1.0)
        ez = abs((np.imag((x0 + 1j * h) ** ALPHA) / h) / exact - 1.0)
        fwd.append((-e, np.log10(max(ef, 1e-18))))
        ctr.append((-e, np.log10(max(ec, 1e-18))))
        cplx.append((-e, np.log10(max(ez, 1e-18))))

    bf, bc = min(fwd, key=lambda t: t[1]), min(ctr, key=lambda t: t[1])
    eps = np.finfo(float).eps
    OUT["differentiation"] = {
        "point": x0, "exact": float(exact),
        "forward_best_log10_error": float(bf[1]), "forward_best_h_exponent": int(-bf[0]),
        "central_best_log10_error": float(bc[1]), "central_best_h_exponent": int(-bc[0]),
        "complex_step_error_at_h_1e_200": float(
            abs((np.imag((x0 + 1j * 1e-200) ** ALPHA) / 1e-200) / exact - 1.0)),
        "optimal_h_forward_sqrt_eps": float(np.sqrt(eps)),
        "optimal_h_central_cbrt_eps": float(eps ** (1 / 3)),
        "forward_truncation_order": 1, "central_truncation_order": 2,
        "complex_step_has_no_subtraction": True,
    }
    OUT["fig_differentiation"] = {
        "forward": " ".join(f"({a},{b:.3f})" for a, b in fwd),
        "central": " ".join(f"({a},{b:.3f})" for a, b in ctr),
        "complex": " ".join(f"({a},{b:.3f})" for a, b in cplx),
    }

    # Jacobian: finite differences vs complex step vs autograd
    def Fv(v):
        return np.array([v[0] ** 2 + v[1] ** 2 - 4.0, np.exp(v[0]) + v[1] - 1.0])

    v0 = np.array([1.0, -1.0])
    Jex = np.array([[2 * v0[0], 2 * v0[1]], [np.exp(v0[0]), 1.0]])
    h = np.sqrt(eps)
    Jfd = np.column_stack([(Fv(v0 + h * e) - Fv(v0)) / h for e in np.eye(2)])
    Jcs = np.column_stack([np.imag(np.array([
        (v0[0] + 1j * 1e-200 * e[0]) ** 2 + (v0[1] + 1j * 1e-200 * e[1]) ** 2 - 4.0,
        np.exp(v0[0] + 1j * 1e-200 * e[0]) + (v0[1] + 1j * 1e-200 * e[1]) - 1.0])) / 1e-200
        for e in np.eye(2)])
    rec = {"finite_difference_max_error": float(np.max(np.abs(Jfd - Jex))),
           "complex_step_max_error": float(np.max(np.abs(Jcs - Jex))),
           "function_evaluations_fd": 3, "function_evaluations_cs": 2}
    try:
        import torch
        vt = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)
        Jt = torch.autograd.functional.jacobian(
            lambda v: torch.stack([v[0] ** 2 + v[1] ** 2 - 4.0, torch.exp(v[0]) + v[1] - 1.0]), vt)
        rec["autograd_max_error"] = float(np.max(np.abs(Jt.detach().numpy() - Jex)))
        rec["torch_version"] = torch.__version__
    except Exception as exc:                                  # pragma: no cover
        rec["autograd_max_error"] = None
        rec["torch_error"] = str(exc)
    OUT["jacobian"] = rec


# =====================================================================  B1
def run_interpolation():
    # Runge's function: equispaced vs Chebyshev nodes
    runge = lambda x: 1.0 / (1.0 + 25.0 * x ** 2)
    xx = np.linspace(-1, 1, 2001)
    rows = []
    for n in (5, 9, 15, 21):
        xe = np.linspace(-1, 1, n)
        ce = np.polyfit(xe, runge(xe), n - 1)
        ee = np.max(np.abs(np.polyval(ce, xx) - runge(xx)))
        k = np.arange(1, n + 1)
        xc = np.sort(np.cos((2 * k - 1) * np.pi / (2 * n)))    # sorted ascending
        cc = npcheb.chebfit(xc, runge(xc), n - 1)
        ec = np.max(np.abs(npcheb.chebval(xx, cc) - runge(xx)))
        rows.append({"n": n, "equispaced_max_error": float(ee),
                     "chebyshev_max_error": float(ec)})
    OUT["runge"] = {
        "function": "1/(1+25 x^2)", "rows": rows,
        "node_formula": "x_i = cos((2i-1) pi / 2n)",
        "nodes_come_out_descending": True,
        "note": "i=1 gives cos(pi/2n) which is near +1; sort ascending before use",
    }
    sel = [r["n"] for r in rows]
    OUT["fig_runge"] = {
        "equispaced": " ".join(f"({n},{np.log10(r['equispaced_max_error']):.3f})"
                               for n, r in zip(sel, rows)),
        "chebyshev": " ".join(f"({n},{np.log10(r['chebyshev_max_error']):.3f})"
                              for n, r in zip(sel, rows)),
    }

    # shape preservation: does the scheme keep a concave function concave?
    kg = np.linspace(0.5, 10.0, 12)
    vg = np.log(kg)                                   # concave, increasing
    fine = np.linspace(kg[0], kg[-1], 1500)
    truth = np.log(fine)

    lin = np.interp(fine, kg, vg)
    spl = CubicSpline(kg, vg, bc_type="natural")(fine)
    pch = PchipInterpolator(kg, vg)(fine)

    def concave(y):
        d2 = np.diff(y, 2)
        return bool(np.all(d2 <= 1e-12)), float(d2.max())

    def monotone(y):
        return bool(np.all(np.diff(y) >= -1e-14))

    res = {}
    for name, y in (("linear", lin), ("cubic_spline", spl), ("pchip", pch)):
        ok_c, worst = concave(y)
        res[name] = {"max_error": float(np.max(np.abs(y - truth))),
                     "preserves_concavity": ok_c,
                     "worst_second_difference": worst,
                     "preserves_monotonicity": monotone(y)}
    OUT["shape_preservation"] = {
        "function": "ln k on 12 nodes over [0.5, 10]",
        **res,
        "correction": ("QM-7's table says linear interpolation does NOT preserve concavity. "
                       "It does -- and that is exactly what licenses golden section inside VFI."),
    }

    # a kinked policy, which is where the spline actually breaks
    kk = np.linspace(0.0, 10.0, 21)
    gg = np.minimum(kk, 4.0)                      # a binding constraint: concave, kinked
    ff = np.linspace(0.0, 10.0, 2000)
    tr = np.minimum(ff, 4.0)
    lin2 = np.interp(ff, kk, gg)
    spl2 = CubicSpline(kk, gg, bc_type="natural")(ff)
    pch2 = PchipInterpolator(kk, gg)(ff)
    res2 = {}
    for name, y in (("linear", lin2), ("cubic_spline", spl2), ("pchip", pch2)):
        d2 = np.diff(y, 2)
        res2[name] = {
            "max_error": float(np.max(np.abs(y - tr))),
            "overshoot_above_the_cap": float(max(y.max() - 4.0, 0.0)),
            "preserves_monotonicity": bool(np.all(np.diff(y) >= -1e-12)),
            "preserves_concavity": bool(np.all(d2 <= 1e-10)),
        }
    OUT["shape_preservation_kinked"] = {
        "function": "min(k, 4) on 21 nodes -- a binding constraint",
        **res2,
        "lesson": ("this is where the cubic spline's O(h^4) stops being an advantage: "
                   "it overshoots a cap that the model says cannot be exceeded"),
    }

    # Gibbs: a discontinuity, polynomial vs Chebyshev
    step = lambda x: np.where(x > 0.0, 1.0, 0.0)
    xs = np.linspace(-1, 1, 41)
    cg = np.polyfit(xs, step(xs), 20)
    OUT["gibbs"] = {
        "max_overshoot_polynomial_degree20": float(np.max(np.polyval(cg, xx)) - 1.0),
        "note": "overshoot does not vanish as the degree rises; it is Gibbs, not bad luck",
    }


# =====================================================================  B2
def run_quadrature():
    # E[g(z)] with z ~ N(0,1), g(z) = exp(z)  ->  exact = exp(0.5)
    exact = np.exp(0.5)
    g = lambda z: np.exp(z)
    rows = []
    for n in (3, 5, 9, 17, 33):
        a, b = -8.0, 8.0
        x = np.linspace(a, b, n)
        w_t = np.full(n, (b - a) / (n - 1))
        w_t[0] *= 0.5
        w_t[-1] *= 0.5
        trap = float(np.sum(w_t * g(x) * norm.pdf(x)))

        if n % 2 == 1:
            hstep = (b - a) / (n - 1)
            wS = np.ones(n)
            wS[1:-1:2] = 4
            wS[2:-1:2] = 2
            simp = float(hstep / 3 * np.sum(wS * g(x) * norm.pdf(x)))
        else:
            simp = None

        xg, wg = np.polynomial.legendre.leggauss(n)
        xg = 0.5 * (b - a) * xg + 0.5 * (a + b)
        wg = 0.5 * (b - a) * wg
        gl = float(np.sum(wg * g(xg) * norm.pdf(xg)))

        xh, wh = np.polynomial.hermite_e.hermegauss(n)
        gh = float(np.sum(wh * g(xh)) / np.sqrt(2 * np.pi))

        rows.append({"n": n,
                     "trapezoid_error": abs(trap - exact),
                     "simpson_error": abs(simp - exact) if simp else None,
                     "gauss_legendre_error": abs(gl - exact),
                     "gauss_hermite_error": abs(gh - exact)})

    rng = np.random.default_rng(0)
    mc = {}
    for n in (100, 10_000, 1_000_000):
        s = g(rng.standard_normal(n))
        mc[str(n)] = {"error": float(abs(s.mean() - exact)),
                      "standard_error": float(s.std(ddof=1) / np.sqrt(n))}
    OUT["quadrature"] = {
        "problem": "E[exp(z)], z ~ N(0,1); exact = exp(1/2)",
        "exact": float(exact), "rows": rows, "monte_carlo": mc,
        "gauss_hermite_exact_for_polynomials_up_to_degree": "2n - 1",
        "monte_carlo_rate": "n^{-1/2}, independent of dimension",
    }
    OUT["fig_quadrature"] = {
        "trapezoid": " ".join(f"({r['n']},{np.log10(max(r['trapezoid_error'],1e-17)):.3f})" for r in rows),
        "gauss_hermite": " ".join(f"({r['n']},{np.log10(max(r['gauss_hermite_error'],1e-17)):.3f})" for r in rows),
    }


# =====================================================================  B3
def tauchen(n, rho, sigma, m=3.0):
    """Tauchen (1986) on log z, zero drift.  Returns (grid, Pi)."""
    sz = sigma / np.sqrt(1.0 - rho ** 2)
    y = np.linspace(-m * sz, m * sz, n)
    step = y[1] - y[0]
    P = np.empty((n, n))
    for i in range(n):
        P[i, 0] = norm.cdf((y[0] - rho * y[i] + step / 2) / sigma)
        P[i, -1] = 1.0 - norm.cdf((y[-1] - rho * y[i] - step / 2) / sigma)
        for j in range(1, n - 1):
            P[i, j] = (norm.cdf((y[j] - rho * y[i] + step / 2) / sigma)
                       - norm.cdf((y[j] - rho * y[i] - step / 2) / sigma))
    return y, P


def rouwenhorst(n, rho, sigma):
    """Rouwenhorst (1995).  Matches rho and the unconditional variance exactly."""
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


def chain_moments(y, P):
    ev = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(ev[0] - 1.0)))
    pi = np.real(ev[1][:, i])
    pi = pi / pi.sum()
    mu = float(pi @ y)
    var = float(pi @ (y - mu) ** 2)
    cov = float(sum(pi[a] * P[a, b] * (y[a] - mu) * (y[b] - mu)
                    for a in range(len(y)) for b in range(len(y))))
    return mu, np.sqrt(var), cov / var


def run_markov():
    rows = []
    for rho in (0.90, 0.95, 0.99):
        tgt_sd = SIG_EPS / np.sqrt(1 - rho ** 2)
        for n in (5, 9, 15):
            yt, Pt = tauchen(n, rho, SIG_EPS)
            yr, Pr = rouwenhorst(n, rho, SIG_EPS)
            _, sdt, rt = chain_moments(yt, Pt)
            _, sdr, rr = chain_moments(yr, Pr)
            rows.append({
                "rho_target": rho, "n": n, "sd_target": float(tgt_sd),
                "tauchen_rho": rt, "tauchen_sd": sdt,
                "rouwenhorst_rho": rr, "rouwenhorst_sd": sdr,
                "tauchen_rho_error_pct": 100 * abs(rt / rho - 1),
                "rouwenhorst_rho_error_pct": 100 * abs(rr / rho - 1),
                "tauchen_sd_error_pct": 100 * abs(sdt / tgt_sd - 1),
                "rouwenhorst_sd_error_pct": 100 * abs(sdr / tgt_sd - 1),
            })
    OUT["markov"] = {
        "sigma_eps": SIG_EPS, "rows": rows,
        "headline": ("Rouwenhorst matches rho and the unconditional sd to machine precision "
                     "at every n and every rho; Tauchen degrades as rho rises"),
    }
    for rho in (0.90, 0.95, 0.99):
        sub = [r for r in rows if r["rho_target"] == rho]
        OUT[f"fig_markov_rho{int(rho*100)}"] = {
            "tauchen": " ".join(f"({r['n']},{r['tauchen_rho_error_pct']:.4f})" for r in sub),
            "rouwenhorst": " ".join(f"({r['n']},{max(r['rouwenhorst_rho_error_pct'],1e-13):.4f})" for r in sub),
        }


# ============ Smolyak sparse grids (B1), and the worked example (B5)
# Construction follows Malin, Krueger & Kubler (2011, JEDC 35, 229-239):
#   m_1 = 1, m_i = 2^(i-1) + 1;  G^i = extrema of the Chebyshev polynomial of
#   degree m_i - 1, so that G^1 subset G^2 subset G^3 ... (the grids NEST).
#   The sparse grid is  Phi(q,d) = union over d <= |i| <= q of G^{i_1} x ... x G^{i_d},
#   with q = d + mu.  mu is the "level"; mu = 0 gives a single point.

def smolyak_nodes_1d(i):
    """G^i: the m_i extrema of a Chebyshev polynomial, ascending. G^1 = {0}."""
    if i == 1:
        return np.array([0.0])
    m = 2 ** (i - 1) + 1
    return -np.cos(np.pi * np.arange(m) / (m - 1))


def smolyak_index_sets(d, mu):
    """All (i_1..i_d), i_j >= 1, with d <= sum i_j <= d + mu."""
    if d == 1:
        for a in range(mu + 1):
            yield (a + 1,)
        return
    for a in range(mu + 1):
        for rest in smolyak_index_sets(d - 1, mu - a):
            yield (a + 1,) + rest


def smolyak_grid(d, mu):
    """The distinct nodes of Phi(d+mu, d), as a sorted list of tuples."""
    from itertools import product
    pts = set()
    for idx in smolyak_index_sets(d, mu):
        grids = [smolyak_nodes_1d(i) for i in idx]
        for combo in product(*grids):
            pts.add(tuple(round(float(c), 12) for c in combo))
    return sorted(pts)


def smolyak_points(d, mu):
    return len(smolyak_grid(d, mu))


def run_smolyak():
    from math import comb

    # --- the 1D ladder, printed exactly as the source states it
    ladder = []
    for i in (1, 2, 3, 4):
        g = smolyak_nodes_1d(i)
        ladder.append({"i": i, "m_i": int(2 ** (i - 1) + 1) if i > 1 else 1,
                       "nodes": [round(float(x), 4) for x in g]})
    nested = all(set(np.round(smolyak_nodes_1d(i), 10)).issubset(
                 set(np.round(smolyak_nodes_1d(i + 1), 10))) for i in (1, 2, 3))

    # --- point counts, including the two rows that check against the source table
    rows = []
    for d in (1, 2, 3, 4, 5, 10, 12):
        rec = {"d": d}
        for mu in (1, 2, 3):
            if d <= 5 or mu <= 2:
                rec[f"smolyak_mu{mu}"] = smolyak_points(d, mu)
        rec["tensor_3_per_dim"] = 3 ** d
        rec["tensor_5_per_dim"] = 5 ** d
        rec["tensor_9_per_dim"] = 9 ** d
        rows.append(rec)

    # Malin-Krueger-Kubler's own table, q = d + 2: columns d, 2^(q-d)+1, |Phi|, (2^(q-d)+1)^d
    src_table = {2: 13, 4: 41, 5: 61, 12: 313}
    check = {str(d): {"source": v, "ours": smolyak_points(d, 2),
                      "match": smolyak_points(d, 2) == v} for d, v in src_table.items()}

    # --- 2D index sets, spelled out, plus the signed combination coefficients
    def idx_report(d, mu):
        q = d + mu
        out = []
        for idx in sorted(smolyak_index_sets(d, mu), key=lambda t: (sum(t), t)):
            a = sum(idx)
            out.append({"i": list(idx), "abs_i": a,
                        "coef": int((-1) ** (q - a) * comb(d - 1, q - a))})
        return out

    # --- the pictures: sparse grids in 2D against the tensor grid
    def coords2d(pts):
        return " ".join(f"({x:.4f},{y:.4f})" for x, y in pts)
    g5 = smolyak_nodes_1d(3)          # 5 nodes per dimension
    tensor5 = [(float(a), float(b)) for a in g5 for b in g5]

    OUT["smolyak"] = {
        "construction": ("Malin-Krueger-Kubler (2011): m_1=1, m_i=2^(i-1)+1; G^i = Chebyshev "
                         "extrema, nested; Phi(q,d) = union_{d<=|i|<=q} G^{i_1}x...xG^{i_d}, q=d+mu"),
        "nodes_1d": ladder,
        "grids_are_nested": bool(nested),
        "rows": rows,
        "source_table_check": check,
        "index_sets_2d_mu1": idx_report(2, 1),
        "index_sets_2d_mu2": idx_report(2, 2),
        "index_sets_3d_mu2": idx_report(3, 2),
        "grid_2d_mu0": [list(p) for p in smolyak_grid(2, 0)],
        "grid_2d_mu1": [list(p) for p in smolyak_grid(2, 1)],
        "error_bound": "C_d * n^{-k} * (log n)^{(d-1)(k+1)} for f in C^k, n = |Phi(q,d)|",
        "note": ("a tensor grid grows like n^d; the sparse grid grows polynomially in d "
                 "at fixed level, at the cost of exactness only for a reduced polynomial set"),
    }
    OUT["fig_smolyak_2d"] = {
        "mu1": coords2d(smolyak_grid(2, 1)),
        "mu2": coords2d(smolyak_grid(2, 2)),
        "mu3": coords2d(smolyak_grid(2, 3)),
        "tensor5": coords2d(tensor5),
        "n_mu1": smolyak_points(2, 1), "n_mu2": smolyak_points(2, 2),
        "n_mu3": smolyak_points(2, 3), "n_tensor5": len(tensor5),
    }


def run_worked_example():
    """The B5 walkthrough: Rouwenhorst + interpolation + golden section + Howard,
    reported step by step so the deck can quote each stage."""
    alpha, beta, delta = ALPHA, BETA, DELTA
    n_z, n_k = 7, 300

    logz, Pi = rouwenhorst(n_z, RHO, SIG_EPS)
    z = np.exp(logz)
    _, sd, rr = chain_moments(logz, Pi)
    kss = ((1 / beta - 1 + delta) / alpha) ** (1 / (alpha - 1))
    kg = np.linspace(0.5 * kss, 1.5 * kss, n_k)

    class M:
        pass
    m = M()
    m.alpha, m.beta, m.delta = alpha, beta, delta
    m.z, m.Pi, m.k = z, Pi, kg
    m.n_k, m.n_z = n_k, n_z
    m.y = z[None, :] * kg[:, None] ** alpha + (1 - delta) * kg[:, None]
    C = m.y[:, :, None] - kg[None, None, :]
    m.U = np.full_like(C, -1e10)
    np.log(C, out=m.U, where=C > 0)

    V0, p0, it0, t0, c0 = vfi_plain(m)
    g_grid = kg[p0]
    _, g_cont, it_c, t_c, nev = vfi_continuous(m, n_h=20)

    E_grid = euler_errors_from_kpolicy(m, g_grid, n_test=1500)
    E_cont = euler_errors_from_kpolicy(m, g_cont, n_test=1500)

    w = slice(120, 156)
    jm = n_z // 2
    OUT["worked_stochastic"] = {
        "n_k": n_k, "n_z": n_z,
        "chain": "Rouwenhorst",
        "chain_implied_rho": rr, "chain_implied_sd": sd,
        "chain_target_sd": float(SIG_EPS / np.sqrt(1 - RHO ** 2)),
        "k_ss": float(kss), "h": float(kg[1] - kg[0]),
        "z_lo": float(z[0]), "z_hi": float(z[-1]),
        "grid_search": {"sweeps": it0, "seconds": round(t0, 2),
                        "comparisons": int(c0),
                        "max_log10_euler": float(np.log10(E_grid.max())),
                        "mean_log10_euler": float(np.log10(E_grid.mean())),
                        "distinct_policy_values_in_36": int(np.unique(np.round(g_grid[w, jm], 10)).size)},
        "continuous": {"policy_updates": it_c, "seconds": round(t_c, 2),
                       "objective_evaluations": int(nev),
                       "max_log10_euler": float(np.log10(E_cont.max())),
                       "mean_log10_euler": float(np.log10(E_cont.mean())),
                       "distinct_policy_values_in_36": int(np.unique(np.round(g_cont[w, jm], 10)).size)},
        "accuracy_gain_decades": float(np.log10(E_grid.max()) - np.log10(E_cont.max())),
    }
    sel = np.arange(0, n_k, 6)
    OUT["fig_worked_policy"] = {
        "low": coords(kg[sel], g_cont[sel, 0]),
        "mid": coords(kg[sel], g_cont[sel, jm]),
        "high": coords(kg[sel], g_cont[sel, -1]),
        "fortyfive": f"({kg[0]:.4f},{kg[0]:.4f}) ({kg[-1]:.4f},{kg[-1]:.4f})",
        "kss": float(kss),
    }
    ww = slice(120, 156)
    OUT["fig_worked_zoom"] = {
        "grid_search": coords(kg[ww], g_grid[ww, jm], "({:.5f},{:.5f})"),
        "continuous": coords(kg[ww], g_cont[ww, jm], "({:.5f},{:.5f})"),
    }


# =====================================================================  B4
class RBC:
    def __init__(self, n_k=500, n_z=7, width=0.5):
        self.alpha, self.beta, self.delta = ALPHA, BETA, DELTA
        logz, self.Pi = tauchen(n_z, RHO, SIG_EPS)
        self.z = np.exp(logz)
        self.kss = ((1 / BETA - 1 + DELTA) / ALPHA) ** (1 / (ALPHA - 1))
        self.k = np.linspace((1 - width) * self.kss, (1 + width) * self.kss, n_k)
        self.n_k, self.n_z = n_k, n_z
        y = self.z[None, :] * self.k[:, None] ** ALPHA + (1 - DELTA) * self.k[:, None]
        self.y = y
        C = y[:, :, None] - self.k[None, None, :]
        self.U = np.full_like(C, -1e10)
        np.log(C, out=self.U, where=C > 0)

    def steady_state(self):
        k = self.kss
        return {"k_ss": float(k), "y_ss": float(k ** ALPHA),
                "c_ss": float(k ** ALPHA - DELTA * k),
                "K_over_Y_quarterly": float(k / k ** ALPHA),
                "r_ss": float(ALPHA * k ** (ALPHA - 1) - DELTA)}


def vfi_plain(m, tol=TOL, maxit=20000):
    V = np.zeros((m.n_k, m.n_z))
    t0 = time.perf_counter()
    it = 0
    while it < maxit:
        EV = V @ m.Pi.T
        M = m.U + m.beta * EV.T[None, :, :]
        Vn, pol = M.max(axis=2), M.argmax(axis=2)
        d = np.max(np.abs(Vn - V))
        V, it = Vn, it + 1
        if d < tol:
            break
    return V, pol, it, time.perf_counter() - t0, m.n_k * m.n_z * m.n_k * it


def vfi_howard(m, n_h=20, tol=TOL, maxit=5000):
    V = np.zeros((m.n_k, m.n_z))
    t0 = time.perf_counter()
    outer, comps = 0, 0
    idx_k, idx_z = np.meshgrid(np.arange(m.n_k), np.arange(m.n_z), indexing="ij")
    while outer < maxit:
        EV = V @ m.Pi.T
        M = m.U + m.beta * EV.T[None, :, :]
        Vn, pol = M.max(axis=2), M.argmax(axis=2)
        comps += m.n_k * m.n_z * m.n_k
        d = np.max(np.abs(Vn - V))
        V = Vn
        outer += 1
        for _ in range(n_h):                       # policy evaluation, O(M)
            EV = V @ m.Pi.T
            V = m.U[idx_k, idx_z, pol] + m.beta * EV[pol, idx_z]
            comps += m.n_k * m.n_z
        if d < tol:
            break
    return V, pol, outer, time.perf_counter() - t0, comps


def vfi_monotone(m, n_h=20, tol=TOL, maxit=5000):
    """Howard, plus a monotone lower bound on the k' search."""
    V = np.zeros((m.n_k, m.n_z))
    pol = np.zeros((m.n_k, m.n_z), dtype=int)
    idx_k, idx_z = np.meshgrid(np.arange(m.n_k), np.arange(m.n_z), indexing="ij")
    t0 = time.perf_counter()
    outer, comps = 0, 0
    while outer < maxit:
        EV = V @ m.Pi.T
        Vn = np.empty_like(V)
        for j in range(m.n_z):
            lo = 0
            for i in range(m.n_k):
                cand = m.U[i, j, lo:] + m.beta * EV[lo:, j]
                a = int(np.argmax(cand))
                comps += cand.size
                pol[i, j] = lo + a
                Vn[i, j] = cand[a]
                lo = lo + a
        d = np.max(np.abs(Vn - V))
        V = Vn
        outer += 1
        for _ in range(n_h):
            EV = V @ m.Pi.T
            V = m.U[idx_k, idx_z, pol] + m.beta * EV[pol, idx_z]
            comps += m.n_k * m.n_z
        if d < tol:
            break
    return V, pol, outer, time.perf_counter() - t0, comps


def vfi_continuous(m, n_h=20, tol=TOL, maxit=3000):
    """Howard, but the inner max is golden section over a CONTINUOUS k',
    with V interpolated linearly.  This is what retires 3.A4's staircase."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    V = np.zeros((m.n_k, m.n_z))
    gk = np.tile(m.k[:, None] * 0.98, (1, m.n_z))
    idx_z = np.arange(m.n_z)
    t0 = time.perf_counter()
    outer, nevals = 0, 0
    while outer < maxit:
        EV = V @ m.Pi.T                                   # (n_k, n_z)
        Vn = np.empty_like(V)
        for j in range(m.n_z):
            lo = np.full(m.n_k, m.k[0])
            hi = np.minimum(m.y[:, j] - 1e-8, m.k[-1])
            a, b = lo.copy(), hi.copy()
            c_ = b - phi * (b - a)
            d_ = a + phi * (b - a)

            def obj(kp):
                cc = m.y[:, j] - kp
                out = np.full_like(kp, -1e10)
                ok = cc > 1e-12
                out[ok] = np.log(cc[ok]) + m.beta * np.interp(kp[ok], m.k, EV[:, j])
                return out

            fc, fd = obj(c_), obj(d_)
            nevals += 2 * m.n_k
            for _ in range(40):
                left = fc > fd
                b = np.where(left, d_, b)
                a = np.where(left, a, c_)
                c_ = b - phi * (b - a)
                d_ = a + phi * (b - a)
                fc, fd = obj(c_), obj(d_)
                nevals += 2 * m.n_k
            gk[:, j] = 0.5 * (a + b)
            Vn[:, j] = obj(gk[:, j])
        d = np.max(np.abs(Vn - V))
        V = Vn
        outer += 1
        for _ in range(n_h):
            EV = V @ m.Pi.T
            cc = np.maximum(m.y - gk, 1e-12)
            EVi = np.empty_like(V)
            for j in range(m.n_z):
                EVi[:, j] = np.interp(gk[:, j], m.k, EV[:, j])
            V = np.log(cc) + m.beta * EVi
        if d < tol:
            break
    return V, gk, outer, time.perf_counter() - t0, nevals


def egm(m, tol=1e-10, maxit=5000):
    """EGM on CASH-ON-HAND: no root-finding anywhere.

    State m = z k^alpha + (1-delta) k, choice k', so m = c + k' inverts in
    closed form.  This is Carroll's actual formulation -- the source deck keeps
    the grid in k and therefore needs a root solve it then claims to avoid.
    """
    kp = m.k.copy()                                   # end-of-period assets
    mgrid = m.y.copy()                                # (n_k, n_z) cash on hand
    mlo, mhi = mgrid.min(), mgrid.max()
    mg = np.linspace(mlo, mhi, m.n_k)
    c = 0.3 * mg[:, None] * np.ones((1, m.n_z))       # guess
    t0 = time.perf_counter()
    it = 0
    while it < maxit:
        # next-period cash on hand for every (k', z')
        mp = m.z[None, :] * kp[:, None] ** m.alpha + (1 - m.delta) * kp[:, None]
        R = m.alpha * m.z[None, :] * kp[:, None] ** (m.alpha - 1) + (1 - m.delta)
        cp = np.empty_like(mp)
        for l in range(m.n_z):
            cp[:, l] = np.interp(mp[:, l], mg, c[:, l])
        rhs = m.beta * ((R / cp) @ m.Pi.T)            # (n_k, n_z)
        c_end = 1.0 / rhs                             # inverse marginal utility
        m_end = c_end + kp[:, None]                   # <- closed form, no solve
        c_new = np.empty_like(c)
        for j in range(m.n_z):
            c_new[:, j] = np.interp(mg, m_end[:, j], c_end[:, j])
            constrained = mg < m_end[0, j]
            c_new[constrained, j] = mg[constrained] - kp[0]
        d = np.max(np.abs(c_new - c))
        c = c_new
        it += 1
        if d < tol:
            break
    return mg, c, it, time.perf_counter() - t0


def euler_errors_from_kpolicy(m, gk, n_test=4000, seed=0):
    rng = np.random.default_rng(seed)
    kt = rng.uniform(m.k[0], m.k[-1], n_test)
    zt = rng.integers(0, m.n_z, n_test)
    E = np.empty(n_test)
    for i in range(n_test):
        j = zt[i]
        kp = np.interp(kt[i], m.k, gk[:, j])
        cc = m.z[j] * kt[i] ** m.alpha + (1 - m.delta) * kt[i] - kp
        if cc <= 0:
            E[i] = np.nan
            continue
        rhs = 0.0
        for l in range(m.n_z):
            kpp = np.interp(kp, m.k, gk[:, l])
            cp = m.z[l] * kp ** m.alpha + (1 - m.delta) * kp - kpp
            R = m.alpha * m.z[l] * kp ** (m.alpha - 1) + (1 - m.delta)
            rhs += m.Pi[j, l] * R / max(cp, 1e-12)
        E[i] = abs(1.0 - (1.0 / (m.beta * rhs)) / cc)
    E = E[~np.isnan(E)]
    return E


def run_acceleration():
    m = RBC(n_k=500, n_z=7)
    OUT["rbc_quarterly"] = {
        "alpha": ALPHA, "beta": BETA, "delta": DELTA, "rho": RHO, "sigma_eps": SIG_EPS,
        "n_k": m.n_k, "n_z": m.n_z, **m.steady_state(),
        "k_grid_lo": float(m.k[0]), "k_grid_hi": float(m.k[-1]),
        "theoretical_sweeps": float(np.log(TOL) / np.log(BETA)),
    }

    table = {}
    V0, p0, it0, t0, c0 = vfi_plain(m)
    g0 = m.k[p0]
    E0 = euler_errors_from_kpolicy(m, g0)
    table["plain_vfi"] = {"sweeps": it0, "seconds": round(t0, 2), "comparisons": int(c0),
                          "max_log10_euler": float(np.log10(E0.max())),
                          "mean_log10_euler": float(np.log10(E0.mean()))}

    for n_h in (10, 20, 50):
        _, ph, ith, th, ch = vfi_howard(m, n_h=n_h)
        gh = m.k[ph]
        Eh = euler_errors_from_kpolicy(m, gh)
        table[f"howard_{n_h}"] = {"policy_updates": ith, "seconds": round(th, 2),
                                  "comparisons": int(ch),
                                  "speedup_vs_plain": round(t0 / th, 1),
                                  "max_log10_euler": float(np.log10(Eh.max()))}

    _, pm, itm, tm, cm = vfi_monotone(m, n_h=20)
    gm = m.k[pm]
    Em = euler_errors_from_kpolicy(m, gm)
    table["howard20_plus_monotone"] = {
        "policy_updates": itm, "seconds": round(tm, 2), "comparisons": int(cm),
        "comparisons_saved_vs_howard20": int(table["howard_20"]["comparisons"] - cm),
        "speedup_vs_plain": round(t0 / tm, 1),
        "max_log10_euler": float(np.log10(Em.max()))}

    # continuous choice: golden section instead of grid search
    _, gc, itc, tc, nev = vfi_continuous(m, n_h=20)
    Ec = euler_errors_from_kpolicy(m, gc)
    w = slice(60, 96)
    table["howard20_plus_continuous_choice"] = {
        "policy_updates": itc, "seconds": round(tc, 2),
        "objective_evaluations": int(nev),
        "max_log10_euler": float(np.log10(Ec.max())),
        "mean_log10_euler": float(np.log10(Ec.mean())),
        "distinct_policy_values_in_36_states": int(np.unique(np.round(gc[w, m.n_z // 2], 10)).size),
    }
    OUT["staircase_retired"] = {
        "window": "36 consecutive grid points, middle z",
        "grid_search_distinct_values": int(np.unique(np.round(m.k[p0][w, m.n_z // 2], 10)).size),
        "continuous_choice_distinct_values": int(np.unique(np.round(gc[w, m.n_z // 2], 10)).size),
        "grid_search_max_log10_euler": float(np.log10(E0.max())),
        "continuous_choice_max_log10_euler": float(np.log10(Ec.max())),
    }
    OUT["fig_staircase_retired"] = {
        "grid_search": coords(m.k[w], m.k[p0][w, m.n_z // 2], "({:.5f},{:.5f})"),
        "continuous": coords(m.k[w], gc[w, m.n_z // 2], "({:.5f},{:.5f})"),
    }

    mg, cpol, ite, te = egm(m)
    kp_of_m = mg[:, None] - cpol
    g_egm = np.empty((m.n_k, m.n_z))
    for j in range(m.n_z):
        g_egm[:, j] = np.interp(m.y[:, j], mg, kp_of_m[:, j])
    Ee = euler_errors_from_kpolicy(m, g_egm)
    table["egm_cash_on_hand"] = {
        "iterations": ite, "seconds": round(te, 2), "root_finds": 0,
        "speedup_vs_plain": round(t0 / te, 1),
        "max_log10_euler": float(np.log10(Ee.max())),
        "mean_log10_euler": float(np.log10(Ee.mean()))}

    OUT["acceleration"] = {
        "note": ("comparisons is the implementation-independent measure; seconds are "
                 "single-core NumPy and the monotone variant carries Python-loop overhead"),
        "table": table,
        "source_claims_unverified": {"overview_frame": "10x-100x", "combining_frame": "50x-200x"},
    }
    ktest = np.linspace(m.k[2], m.k[-3], 400)
    jmid = m.n_z // 2

    def err_curve(gk):
        out = []
        for kv in ktest:
            kp = np.interp(kv, m.k, gk[:, jmid])
            cc = m.z[jmid] * kv ** m.alpha + (1 - m.delta) * kv - kp
            rhs = 0.0
            for l in range(m.n_z):
                kpp = np.interp(kp, m.k, gk[:, l])
                cp = m.z[l] * kp ** m.alpha + (1 - m.delta) * kp - kpp
                R = m.alpha * m.z[l] * kp ** (m.alpha - 1) + (1 - m.delta)
                rhs += m.Pi[jmid, l] * R / max(cp, 1e-12)
            out.append(abs(1.0 - (1.0 / (m.beta * rhs)) / cc))
        return np.array(out)

    e_plain, e_egm = err_curve(g0), err_curve(g_egm)
    OUT["fig_euler_compare"] = {
        "plain_vfi": coords(ktest[::4], np.log10(np.maximum(e_plain[::4], 1e-14))),
        "egm": coords(ktest[::4], np.log10(np.maximum(e_egm[::4], 1e-14))),
        "k_lo": float(ktest[0]), "k_hi": float(ktest[-1]),
    }
    sel = np.arange(0, m.n_k, 10)
    OUT["fig_policy_egm_vs_vfi"] = {
        "vfi": coords(m.k[sel], g0[sel, m.n_z // 2]),
        "egm": coords(m.k[sel], g_egm[sel, m.n_z // 2]),
        "fortyfive": f"({m.k[0]:.4f},{m.k[0]:.4f}) ({m.k[-1]:.4f},{m.k[-1]:.4f})",
    }


ALL_RUNS = (run_rootfinding, run_optimization, run_large_scale,
            run_differentiation, run_smolyak,
            run_interpolation, run_quadrature, run_markov,
            run_acceleration, run_worked_example)

if __name__ == "__main__":
    # With no arguments, regenerate the whole ledger.  With arguments, run only
    # the named sections and MERGE them into the existing JSON -- so that adding
    # one figure does not silently move every measured timing already on a slide.
    import sys
    wanted = sys.argv[1:]
    runs = ALL_RUNS
    merge = False
    if wanted:
        by_name = {fn.__name__.replace("run_", ""): fn for fn in ALL_RUNS}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise SystemExit(f"unknown section(s) {missing}; choose from {sorted(by_name)}")
        runs = tuple(by_name[w] for w in wanted)
        merge = True

    for fn in runs:
        t = time.perf_counter()
        fn()
        print(f"  {fn.__name__:24s} {time.perf_counter()-t:7.1f}s", flush=True)

    path = "tools/figures/s04_numbers.json"
    if merge:
        with open(path) as f:
            base = json.load(f)
        base.update(OUT)
        base.setdefault("_meta", {})["last_partial_update"] = " ".join(wanted)
        OUT.clear()
        OUT.update(base)
    else:
        OUT["_meta"] = {"calibration": "quarterly RBC, Quant_Macro Lab7",
                        "alpha": ALPHA, "beta": BETA, "delta": DELTA,
                        "rho": RHO, "sigma_eps": SIG_EPS, "tol": TOL,
                        "numpy": np.__version__,
                        "generated_by": "tools/figures/s04_generate.py"}
    with open(path, "w") as f:
        json.dump(OUT, f, indent=2)
    print(f"wrote {path}" + (f"  (merged: {' '.join(wanted)})" if merge else ""))
    for k in OUT:
        print(" ", k)
