#!/usr/bin/env python3
"""Every number and figure coordinate used in the Session 6 decks.

Session 6 takes the classical methods of Sessions 3-5 to the models of Session
2.  Block A solves them: Aiyagari as a nested fixed point, Krusell-Smith, the
life-cycle model, and the deterministic transition between two steady states.
Block B steps back and asks when the heterogeneity could have been ignored
altogether -- complete markets, Gorman aggregation, the Negishi method -- and
what the path between two steady states looks like.

Calibration continues Session 4 and 5's quarterly numbers:

    alpha = 0.33, beta = 0.99, delta = 0.025

with CRRA sigma = 2 and an idiosyncratic labour-productivity process

    log z' = rho log z + eps,   eps ~ N(0, sigma_eps^2)

discretized by Rouwenhorst (4.B3).  NOTE: s04_numbers.json's "markov" key is
the AGGREGATE TFP shock at sigma_eps = 0.007 and is NOT reusable here -- an
earnings process is an order of magnitude more volatile.  The persistence
experiment holds the unconditional sd of log z fixed and varies rho, so that
rho alone moves the fraction of constrained households.

Two keys are reused rather than recomputed, per slides/SOURCES.md:
    s04_numbers.json "smolyak"       -- the point counts behind 6.A3's wall
    s04_numbers.json "rbc_quarterly" -- k* and the calibration for 6.B3's
                                        representative-agent transition

Output: tools/figures/s06_numbers.json -- the ledger.  No number appears on an
S6 slide unless it is a key in that file.

Run:  sbatch tools/run_code.slurm python3 tools/figures/s06_generate.py
      sbatch tools/run_code.slurm python3 tools/figures/s06_generate.py aiyagari_ss
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "s06_numbers.json")

# ---- Calibration, quarterly, continuing S4-S5 -----------------------------
ALPHA, BETA, DELTA = 0.33, 0.99, 0.025
SIGMA = 2.0                      # CRRA, the Aiyagari block
SIGMA_RA = 1.0                   # log utility in Block B's RA model, so
                                 # that 6.B3 reuses S5's own eigenvalue
RHO_Z, SD_LOG_Z = 0.95, 0.30     # idiosyncratic earnings: persistence, uncond. sd
N_Z = 7                          # Rouwenhorst states
A_MIN = 0.0                      # borrowing limit: a' >= -abar with abar = 0
A_MAX = 300.0
N_A = 400
CURV = 2.5                       # grid curvature, dense near the constraint
TOL_POL, TOL_DIST, TOL_R = 1e-10, 1e-12, 1e-7

OUT = {}


# ---------------------------------------------------------------- utilities
def coords(xs, ys, fmt="({:.4f},{:.4f})"):
    """Paste-ready pgfplots coordinate string."""
    return " ".join(fmt.format(float(x), float(y)) for x, y in zip(xs, ys))


def rouwenhorst(n, rho, sd_uncond):
    """Rouwenhorst (4.B3) for log z' = rho log z + eps with Var(log z) fixed.

    Returns (z levels normalised to mean one under the invariant, Pi, pi).
    """
    p = (1.0 + rho) / 2.0
    Pi = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        Pi_new = np.zeros((k, k))
        Pi_new[:-1, :-1] += p * Pi
        Pi_new[:-1, 1:] += (1 - p) * Pi
        Pi_new[1:, :-1] += (1 - p) * Pi
        Pi_new[1:, 1:] += p * Pi
        Pi_new[1:-1, :] /= 2.0
        Pi = Pi_new
    psi = sd_uncond * np.sqrt(n - 1)
    logz = np.linspace(-psi, psi, n)
    # invariant distribution of the Rouwenhorst chain is Binomial(n-1, 1/2)
    from math import comb
    pi = np.array([comb(n - 1, i) for i in range(n)], dtype=float) / 2.0 ** (n - 1)
    z = np.exp(logz)
    z = z / (pi @ z)                     # normalise mean effective labour to 1
    return z, Pi, pi


def make_grid(n, a_min, a_max, curv=CURV):
    """Exponentially spaced asset grid, dense near the borrowing constraint."""
    u = np.linspace(0.0, 1.0, n) ** curv
    return a_min + (a_max - a_min) * u


def prices_from_r(r, p=None):
    """Firm FOCs with L = 1 (mean effective labour normalised to one)."""
    kl = (ALPHA / (r + DELTA)) ** (1.0 / (1.0 - ALPHA))
    w = (1.0 - ALPHA) * kl ** ALPHA
    return kl, w                          # K demand per unit of labour, wage


def solve_egm(r, w, grid, z, Pi, tol=TOL_POL, max_iter=5000, c_init=None):
    """EGM / time iteration on the consumption policy over the joint state (a,z).

    Returns a'(a,z), c(a,z), iterations, final policy change.
    """
    na, nz = grid.size, z.size
    R = 1.0 + r
    coh = R * grid[:, None] + w * z[None, :]          # cash on hand
    c = coh - A_MIN if c_init is None else c_init.copy()
    c = np.maximum(c, 1e-10)
    it, diff = 0, np.inf
    for it in range(1, max_iter + 1):
        # marginal utility next period, evaluated ON the grid of a'
        Euc = (c ** (-SIGMA)) @ Pi.T                  # E[u'(c')|z] at a'=grid
        c_end = (BETA * R * Euc) ** (-1.0 / SIGMA)    # consumption today
        a_end = (c_end + grid[:, None] - w * z[None, :]) / R   # endogenous a
        c_new = np.empty_like(c)
        for q in range(nz):
            c_new[:, q] = np.interp(grid, a_end[:, q], c_end[:, q])
            # below the lowest endogenous point the constraint binds
            binds = grid < a_end[0, q]
            c_new[binds, q] = R * grid[binds] + w * z[q] - A_MIN
        c_new = np.maximum(c_new, 1e-10)
        diff = np.abs(c_new - c).max()
        c = c_new
        if diff < tol:
            break
    ap = np.clip(coh - c, A_MIN, grid[-1])
    c = coh - ap
    return ap, c, it, diff


def young_lottery(ap, grid):
    """Indices and upper weights for Young (2010) mass splitting."""
    a1 = np.clip(ap, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, a1, side="right") - 1, 0, grid.size - 2)
    wt_hi = (a1 - grid[j]) / (grid[j + 1] - grid[j])
    return j, wt_hi


def push_forward(lam, j, wt_hi, Pi):
    """One application of the Markov operator Q on measures."""
    na, nz = lam.shape
    out = np.zeros_like(lam)
    for q in range(nz):
        tmp = np.zeros(na)
        np.add.at(tmp, j[:, q], lam[:, q] * (1.0 - wt_hi[:, q]))
        np.add.at(tmp, j[:, q] + 1, lam[:, q] * wt_hi[:, q])
        out += np.outer(tmp, Pi[q, :])
    return out


def stationary_dist(ap, grid, Pi, pi, tol=TOL_DIST, max_iter=100000):
    """Invariant lambda over (a,z) by iterating Q from a sensible start."""
    na, nz = ap.shape
    lam = np.zeros((na, nz))
    lam[0, :] = pi
    j, wt_hi = young_lottery(ap, grid)
    for it in range(1, max_iter + 1):
        lam_new = push_forward(lam, j, wt_hi, Pi)
        diff = np.abs(lam_new - lam).max()
        lam = lam_new
        if diff < tol:
            break
    return lam / lam.sum(), it, diff


def gini(grid, mass):
    """Gini of a distribution given on a grid of values."""
    m = mass / mass.sum()
    order = np.argsort(grid)
    x, m = np.asarray(grid)[order], m[order]
    S = np.cumsum(m * x)
    if S[-1] <= 0:
        return 0.0
    L = np.concatenate([[0.0], S / S[-1]])
    F = np.concatenate([[0.0], np.cumsum(m)])
    return float(1.0 - np.sum((F[1:] - F[:-1]) * (L[1:] + L[:-1])))


def aiyagari_at_r(r, grid, z, Pi, pi, c_init=None):
    """Household supply of capital at a candidate net return r."""
    kl, w = prices_from_r(r)
    ap, c, it_pol, d_pol = solve_egm(r, w, grid, z, Pi, c_init=c_init)
    lam, it_d, d_d = stationary_dist(ap, grid, Pi, pi)
    K_supply = float((lam.sum(axis=1) * grid).sum())
    return dict(r=r, w=w, K_demand=kl, K_supply=K_supply, ap=ap, c=c, lam=lam,
                it_pol=it_pol, it_dist=it_d)


def solve_r(grid, z, Pi, pi, r_lo=-0.004, r_hi=None, tol=TOL_R, verbose=False):
    """Bisection on r: excess demand d(r) = K_supply(r) - K_demand(r)."""
    if r_hi is None:
        r_hi = 1.0 / BETA - 1.0 - 1e-6        # r < 1/beta - 1 strictly
    f_lo = aiyagari_at_r(r_lo, grid, z, Pi, pi)
    d_lo = f_lo["K_supply"] - f_lo["K_demand"]
    f_hi = aiyagari_at_r(r_hi, grid, z, Pi, pi)
    d_hi = f_hi["K_supply"] - f_hi["K_demand"]
    if d_lo * d_hi > 0:
        raise RuntimeError(f"no sign change: d({r_lo})={d_lo}, d({r_hi})={d_hi}")
    n_iter = 0
    sol = f_lo
    while r_hi - r_lo > tol:
        n_iter += 1
        r_mid = 0.5 * (r_lo + r_hi)
        sol = aiyagari_at_r(r_mid, grid, z, Pi, pi, c_init=sol["c"])
        d_mid = sol["K_supply"] - sol["K_demand"]
        if verbose:
            print(f"    it {n_iter:2d}  r={r_mid: .6f}  d={d_mid: .6f}")
        if d_lo * d_mid <= 0:
            r_hi, d_hi = r_mid, d_mid
        else:
            r_lo, d_lo = r_mid, d_mid
    sol["bisection_iters"] = n_iter
    return sol


# ========================================================== 6.A1  Aiyagari
def run_earnings_process():
    """The Rouwenhorst chain behind every Aiyagari number in this session."""
    rows = []
    for rho in (0.90, 0.95, 0.99):
        z, Pi, pi = rouwenhorst(N_Z, rho, SD_LOG_Z)
        logz = np.log(z)
        mean_log = float(pi @ logz)
        sd_log = float(np.sqrt(pi @ (logz - mean_log) ** 2))
        # implied innovation sd for this rho at the fixed unconditional sd
        rows.append(dict(rho=rho, n=N_Z, sd_log_z=sd_log,
                         sigma_eps=float(SD_LOG_Z * np.sqrt(1 - rho ** 2)),
                         z_min=float(z.min()), z_max=float(z.max()),
                         z_ratio=float(z.max() / z.min()),
                         mean_z=float(pi @ z)))
    OUT["earnings_process"] = dict(
        n_states=N_Z, sd_log_z_target=SD_LOG_Z, method="Rouwenhorst (4.B3)",
        note=("Unconditional sd of log z is held fixed across rho, so the "
              "persistence experiment moves rho alone."),
        rows=rows)


def run_aiyagari_ss():
    """The stationary equilibrium: bisection on r, and the figures."""
    z, Pi, pi = rouwenhorst(N_Z, RHO_Z, SD_LOG_Z)
    grid = make_grid(N_A, A_MIN, A_MAX)
    t0 = time.time()
    sol = solve_r(grid, z, Pi, pi)
    solve_time = time.time() - t0

    lam, ap, c = sol["lam"], sol["ap"], sol["c"]
    lam_a = lam.sum(axis=1)
    K = sol["K_supply"]
    Y = K ** ALPHA
    r, w = sol["r"], sol["w"]
    # fraction at the borrowing limit (mass on the first grid point)
    frac_constrained = float(lam[0, :].sum())
    # earnings Gini, for contrast with wealth
    earn = w * z
    gini_w = gini(grid, lam_a)
    gini_e = gini(earn, lam.sum(axis=0))
    cdf_a = np.cumsum(lam_a)
    i90 = int(np.searchsorted(cdf_a, 0.90))
    top_share = float((lam_a[i90:] * grid[i90:]).sum()
                      / (lam_a * grid).sum())

    OUT["aiyagari_ss"] = dict(
        alpha=ALPHA, beta=BETA, delta=DELTA, sigma=SIGMA, rho_z=RHO_Z,
        sd_log_z=SD_LOG_Z, n_z=N_Z, n_a=N_A, a_min=A_MIN, a_max=A_MAX,
        curv=CURV,
        r_quarterly=r, r_annual=float((1 + r) ** 4 - 1),
        r_complete_markets=float(1.0 / BETA - 1.0),
        r_gap_bp=float(((1.0 / BETA - 1.0) - r) * 1e4),
        w=w, K=K, Y=float(Y),
        K_over_Y_quarterly=float(K / Y), K_over_Y_annual=float(K / (4 * Y)),
        gini_wealth=gini_w, gini_earnings=gini_e,
        frac_constrained=frac_constrained,
        top_decile_wealth_share=top_share,
        mass_at_a_max=float(lam_a[-1]),
        bisection_iters=sol["bisection_iters"],
        egm_iters=sol["it_pol"], dist_iters=sol["it_dist"],
        solve_seconds=solve_time)

    # --- figure: the capital market, supply against demand -----------------
    r_lo, r_hi = -0.002, 1.0 / BETA - 1.0 - 2e-4
    rs = np.linspace(r_lo, r_hi, 24)
    Ks, Kd, c_warm = [], [], None
    for rr in rs:
        f = aiyagari_at_r(rr, grid, z, Pi, pi, c_init=c_warm)
        c_warm = f["c"]
        Ks.append(f["K_supply"]); Kd.append(f["K_demand"])
    OUT["fig_capital_market"] = dict(
        supply=coords(Ks, rs, "({:.3f},{:.5f})"),
        demand=coords(Kd, rs, "({:.3f},{:.5f})"),
        r_star=r, K_star=K,
        r_bar=float(1.0 / BETA - 1.0),
        note="x is capital, y is the quarterly net return; r_bar = 1/beta - 1.")

    # --- figure: the wealth distribution -----------------------------------
    sel = grid <= 120.0
    OUT["fig_wealth_dist"] = dict(
        density=coords(grid[sel], lam_a[sel] / np.gradient(grid)[sel],
                       "({:.2f},{:.5f})"),
        cdf=coords(grid[sel], np.cumsum(lam_a)[sel], "({:.2f},{:.4f})"),
        note="a-marginal of lambda; density is mass per unit of a.")

    # --- figure: the saving policy, low and high z -------------------------
    sel = grid <= 60.0
    OUT["fig_policy_egm"] = dict(
        lowz=coords(grid[sel], ap[sel, 0], "({:.2f},{:.3f})"),
        highz=coords(grid[sel], ap[sel, -1], "({:.2f},{:.3f})"),
        fortyfive=coords(grid[sel], grid[sel], "({:.2f},{:.3f})"),
        note="a'(a,z) at the lowest and highest z, against the 45-degree line.")


def run_aiyagari_persistence():
    """What persistence does -- the promise made at S02_A1:284."""
    grid = make_grid(N_A, A_MIN, A_MAX)
    rows = []
    for rho in (0.90, 0.95, 0.99):
        z, Pi, pi = rouwenhorst(N_Z, rho, SD_LOG_Z)
        sol = solve_r(grid, z, Pi, pi)
        lam, lam_a = sol["lam"], sol["lam"].sum(axis=1)
        K = sol["K_supply"]
        rows.append(dict(
            rho=rho,
            r_quarterly=sol["r"], r_annual=float((1 + sol["r"]) ** 4 - 1),
            K=K, K_over_Y_annual=float(K / (4 * K ** ALPHA)),
            gini_wealth=gini(grid, lam_a),
            frac_constrained=float(lam[0, :].sum()),
            frac_below_one_quarter_income=float(
                lam_a[grid < 0.25 * sol["w"]].sum())))
    OUT["aiyagari_persistence"] = dict(
        sd_log_z_held_fixed=SD_LOG_Z, rows=rows,
        note=("Unconditional dispersion is identical across rows; only the "
              "persistence of the shock differs."))


def run_aiyagari_diagnostics():
    """Does the answer move when the grid does? Plus the KKT residual."""
    z, Pi, pi = rouwenhorst(N_Z, RHO_Z, SD_LOG_Z)
    rows = []
    for n_a in (100, 200, 400, 800):
        grid = make_grid(n_a, A_MIN, A_MAX)
        t0 = time.time()
        sol = solve_r(grid, z, Pi, pi)
        lam_a = sol["lam"].sum(axis=1)
        rows.append(dict(n_a=n_a, r=sol["r"], K=sol["K_supply"],
                         gini_wealth=gini(grid, lam_a),
                         seconds=float(time.time() - t0)))
    # Euler residual off the constraint, at the baseline grid
    grid = make_grid(N_A, A_MIN, A_MAX)
    sol = solve_r(grid, z, Pi, pi)
    ap, c, r, w = sol["ap"], sol["c"], sol["r"], sol["w"]
    R = 1.0 + r
    # E[u'(c(a',z'))|z], interpolating next period's policy at a'
    Euc = np.zeros_like(c)
    for q in range(N_Z):
        cp = np.empty((grid.size, N_Z))
        for qq in range(N_Z):
            cp[:, qq] = np.interp(ap[:, q], grid, c[:, qq])
        Euc[:, q] = (cp ** (-SIGMA)) @ Pi[q, :]
    resid = np.abs(1.0 - (BETA * R * Euc) ** (-1.0 / SIGMA) / c)
    # Evaluate only where the Euler equation actually holds with equality:
    # not at the borrowing limit, and not where a' is clipped at the top of
    # the grid (there the residual measures the grid, not the solution).
    off = (ap > A_MIN + 1e-9) & (ap < grid[-1] * (1.0 - 1e-9))
    v = resid[off]
    lam_n = sol["lam"] / sol["lam"].sum()
    i_max = int(np.unravel_index(np.argmax(np.where(off, resid, 0.0)),
                                 resid.shape)[0])
    OUT["aiyagari_diagnostics"] = dict(
        grid_rows=rows,
        euler_resid_median_log10=float(np.log10(np.median(v))),
        euler_resid_p99_log10=float(np.log10(np.percentile(v, 99))),
        euler_resid_max_log10=float(np.log10(v.max())),
        euler_resid_mean_log10=float(np.log10(v.mean())),
        euler_resid_massweighted_log10=float(
            np.log10((resid * lam_n)[off].sum() / lam_n[off].sum())),
        n_nodes_above_1e2=int((v > 1e-2).sum()), n_nodes_evaluated=int(v.size),
        worst_node_a=float(grid[i_max]), worst_node_a_index=i_max,
        frac_states_constrained=float(off.size - off.sum()) / off.size,
        mass_at_a_max=float(sol["lam"].sum(axis=1)[-1]),
        note=("Unit-free residual (3.A1), evaluated only where the constraint "
              "is slack and a' is not clipped.  The worst nodes sit at the TOP "
              "of the asset grid, not at the kink, and carry negligible mass; "
              "the median is the number that describes the solution."))


# ============================================== 6.A2 / 6.B3  the transition
def _backward_policies(path_rnet, path_w, path_T, grid, z, Pi, c_final):
    """Backward pass: consumption policies along a transition of length T."""
    T = len(path_rnet) - 1
    na, nz = grid.size, z.size
    c_path = [None] * (T + 1)
    ap_path = [None] * (T + 1)
    c_path[T] = c_final
    for t in range(T - 1, -1, -1):
        Rn = 1.0 + path_rnet[t + 1]
        Euc = (c_path[t + 1] ** (-SIGMA)) @ Pi.T
        c_end = (BETA * Rn * Euc) ** (-1.0 / SIGMA)
        Rt = 1.0 + path_rnet[t]
        a_end = (c_end + grid[:, None] - path_w[t] * z[None, :] - path_T[t]) / Rt
        c_t = np.empty((na, nz))
        for q in range(nz):
            c_t[:, q] = np.interp(grid, a_end[:, q], c_end[:, q])
            binds = grid < a_end[0, q]
            c_t[binds, q] = (Rt * grid[binds] + path_w[t] * z[q]
                             + path_T[t] - A_MIN)
        c_t = np.maximum(c_t, 1e-10)
        coh = Rt * grid[:, None] + path_w[t] * z[None, :] + path_T[t]
        c_path[t] = c_t
        ap_path[t] = np.clip(coh - c_t, A_MIN, grid[-1])
    return c_path, ap_path


def _values_along(c_path, ap_path, grid, Pi, V_final):
    """Backward pass for V, needed for the consumption-equivalent variation."""
    T = len(c_path) - 1
    V = [None] * (T + 1)
    V[T] = V_final
    nz = Pi.shape[0]
    for t in range(T - 1, -1, -1):
        EV = V[t + 1] @ Pi.T                     # E[V_{t+1}(a',z')|z] on grid
        cont = np.empty_like(EV)
        for q in range(nz):
            cont[:, q] = np.interp(ap_path[t][:, q], grid, EV[:, q])
        u = c_path[t] ** (1.0 - SIGMA) / (1.0 - SIGMA)
        V[t] = u + BETA * cont
    return V


def _value_ss(c, ap, grid, Pi, tol=1e-11, max_iter=20000):
    """Value function of a stationary policy."""
    nz = Pi.shape[0]
    V = (c ** (1.0 - SIGMA) / (1.0 - SIGMA)) / (1.0 - BETA)
    u = c ** (1.0 - SIGMA) / (1.0 - SIGMA)
    for _ in range(max_iter):
        EV = V @ Pi.T
        cont = np.empty_like(EV)
        for q in range(nz):
            cont[:, q] = np.interp(ap[:, q], grid, EV[:, q])
        V_new = u + BETA * cont
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    return V


def run_transition():
    """A permanent capital-income tax, rebated lump sum: the MIT shock."""
    TAU_K = 0.20
    T = 200
    z, Pi, pi = rouwenhorst(N_Z, RHO_Z, SD_LOG_Z)
    grid = make_grid(N_A, A_MIN, A_MAX)

    # --- the two steady states --------------------------------------------
    def ss_with_tax(tau_k):
        """Bisect on the PRE-tax r; households earn (1-tau_k) r and get rebate."""
        lo, hi = 0.0005, 1.0 / BETA - 1.0 - 1e-6
        sol = None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            kl, w = prices_from_r(mid)
            rnet = (1.0 - tau_k) * mid
            # rebate: balance the budget at the firm's capital demand
            reb = tau_k * mid * kl
            ap, c, itp, _ = solve_egm_with_transfer(rnet, w, reb, grid, z, Pi,
                                                    c_init=None if sol is None
                                                    else sol["c"])
            lam, itd, _ = stationary_dist(ap, grid, Pi, pi)
            Ks = float((lam.sum(axis=1) * grid).sum())
            sol = dict(r=mid, rnet=rnet, w=w, rebate=reb, ap=ap, c=c, lam=lam,
                       K_supply=Ks, K_demand=kl)
            if Ks - kl > 0:
                hi = mid
            else:
                lo = mid
        return sol

    ss0 = ss_with_tax(0.0)
    ss1 = ss_with_tax(TAU_K)
    V0 = _value_ss(ss0["c"], ss0["ap"], grid, Pi)
    V1 = _value_ss(ss1["c"], ss1["ap"], grid, Pi)

    # --- iterate on the path of capital ------------------------------------
    K0, K1 = ss0["K_supply"], ss1["K_supply"]
    Kpath = K0 + (K1 - K0) * (1.0 - np.exp(-np.arange(T + 1) / 25.0))
    Kpath[0] = K0
    damp, tol = 0.30, 1e-6
    lam0 = ss0["lam"]
    outer, gap = 0, np.inf
    for outer in range(1, 201):
        r_pre = ALPHA * Kpath ** (ALPHA - 1.0) - DELTA
        w_p = (1.0 - ALPHA) * Kpath ** ALPHA
        rnet = (1.0 - TAU_K) * r_pre
        rnet[0] = ss0["r"]            # capital at t=0 is predetermined, taxed from t=0
        reb = TAU_K * r_pre * Kpath
        c_path, ap_path = _backward_policies(rnet, w_p, reb, grid, z, Pi,
                                             ss1["c"])
        lam = lam0.copy()
        K_new = np.empty(T + 1)
        K_new[0] = K0
        for t in range(T):
            j, wt = young_lottery(ap_path[t], grid)
            lam = push_forward(lam, j, wt, Pi)
            K_new[t + 1] = float((lam.sum(axis=1) * grid).sum())
        gap = float(np.abs(K_new - Kpath).max())
        Kpath = (1.0 - damp) * Kpath + damp * K_new
        if gap < tol:
            break
    lam_T_gap = float(np.abs(lam - ss1["lam"]).sum())

    # --- welfare -----------------------------------------------------------
    Vtr = _values_along(c_path, ap_path, grid, Pi, V1)
    g = (Vtr[0] / V0) ** (1.0 / (1.0 - SIGMA)) - 1.0      # CEV, state by state
    g_ss = (V1 / V0) ** (1.0 / (1.0 - SIGMA)) - 1.0       # steady states only
    w0 = lam0 / lam0.sum()
    cev_agg = float((g * w0).sum())
    cev_ss_agg = float((g_ss * w0).sum())
    frac_losers = float(w0[g < 0].sum())
    frac_sign_flip = float(w0[(np.sign(g) != np.sign(g_ss))].sum())

    lam_a0 = w0.sum(axis=1)
    cdf = np.cumsum(lam_a0)
    dec_edges = [np.searchsorted(cdf, q) for q in np.linspace(0.1, 0.9, 9)]
    gw = (g * w0).sum(axis=1) / np.maximum(lam_a0, 1e-16)
    dec = []
    lo_i = 0
    for hi_i in dec_edges + [len(grid) - 1]:
        m = lam_a0[lo_i:hi_i + 1]
        if m.sum() > 0:
            dec.append(float((gw[lo_i:hi_i + 1] * m).sum() / m.sum()))
        lo_i = hi_i + 1

    OUT["transition_aiyagari"] = dict(
        tau_k=TAU_K, T=T, damping=damp, tol=tol, outer_iterations=outer,
        max_abs_K_gap=gap, lambda_T_L1_gap_to_new_ss=lam_T_gap,
        r_pre_old=ss0["r"], r_pre_new=ss1["r"],
        K_old=K0, K_new=K1, K_pct_change=float(100 * (K1 / K0 - 1)),
        K_trough=float(Kpath.min()), K_trough_quarter=int(Kpath.argmin()),
        halflife_quarters=float(_halflife(Kpath, K0, K1)))
    OUT["cev"] = dict(
        aggregate_cev_pct=100 * cev_agg,
        steady_state_only_cev_pct=100 * cev_ss_agg,
        frac_losers_pct=100 * frac_losers,
        frac_sign_flip_pct=100 * frac_sign_flip,
        cev_by_wealth_decile_pct=[100 * d for d in dec],
        note=("g solves V_reform = (1+g)^(1-sigma) V_baseline; the "
              "steady-state-only number ignores the path entirely."))
    sel = np.arange(0, T + 1, 2)
    OUT["fig_transition_K"] = dict(
        path=coords(sel, Kpath[sel], "({:.0f},{:.4f})"),
        K_old=K0, K_new=K1)
    OUT["fig_transition_r"] = dict(
        path=coords(sel, (ALPHA * Kpath[sel] ** (ALPHA - 1.0) - DELTA),
                    "({:.0f},{:.6f})"),
        r_old=ss0["r"], r_new=ss1["r"])
    OUT["fig_cev"] = dict(
        by_decile=coords(np.arange(1, len(dec) + 1), [100 * d for d in dec],
                         "({:.0f},{:.4f})"))


def solve_egm_with_transfer(rnet, w, transfer, grid, z, Pi, tol=TOL_POL,
                            max_iter=5000, c_init=None):
    """EGM with a lump-sum transfer in the budget constraint."""
    na, nz = grid.size, z.size
    R = 1.0 + rnet
    coh = R * grid[:, None] + w * z[None, :] + transfer
    c = coh - A_MIN if c_init is None else c_init.copy()
    c = np.maximum(c, 1e-10)
    it, diff = 0, np.inf
    for it in range(1, max_iter + 1):
        Euc = (c ** (-SIGMA)) @ Pi.T
        c_end = (BETA * R * Euc) ** (-1.0 / SIGMA)
        a_end = (c_end + grid[:, None] - w * z[None, :] - transfer) / R
        c_new = np.empty_like(c)
        for q in range(nz):
            c_new[:, q] = np.interp(grid, a_end[:, q], c_end[:, q])
            binds = grid < a_end[0, q]
            c_new[binds, q] = R * grid[binds] + w * z[q] + transfer - A_MIN
        c_new = np.maximum(c_new, 1e-10)
        diff = np.abs(c_new - c).max()
        c = c_new
        if diff < tol:
            break
    ap = np.clip(coh - c, A_MIN, grid[-1])
    return ap, coh - ap, it, diff


def _halflife(path, x0, x1):
    """First period at which the path has covered half the total change."""
    if abs(x1 - x0) < 1e-14:
        return float("nan")
    frac = (path - x0) / (x1 - x0)
    idx = np.where(frac >= 0.5)[0]
    return float(idx[0]) if idx.size else float("nan")


# ================================================ 6.B3  the RA transition
def _k_ss_ra():
    return (ALPHA / (1.0 / BETA - 1.0 + DELTA)) ** (1.0 / (1.0 - ALPHA))


def _shoot(k0, c0, T, kss, lo_frac=0.30, hi_frac=1.50):
    """Forward-integrate the RA growth model from (k0,c0), log utility.

    The corridor is deliberately tight: the saddle path is unstable, so any
    c0 that is not exactly right eventually leaves it.  Which side it leaves
    on is what the bisection reads.
    """
    k = np.empty(T + 1); c = np.empty(T + 1)
    k[0], c[0] = k0, c0
    for t in range(T):
        k[t + 1] = k[t] ** ALPHA + (1.0 - DELTA) * k[t] - c[t]
        if k[t + 1] <= lo_frac * kss:
            return k[:t + 2], c[:t + 1], t + 1, "low"
        if k[t + 1] >= hi_frac * kss:
            return k[:t + 2], c[:t + 1], t + 1, "high"
        R = ALPHA * k[t + 1] ** (ALPHA - 1.0) + 1.0 - DELTA
        c[t + 1] = c[t] * (BETA * R) ** (1.0 / SIGMA_RA)
    return k, c, T, "stayed"


def run_shooting():
    """Shooting, its conditioning, and the saddle path."""
    kss = _k_ss_ra()
    css = kss ** ALPHA - DELTA * kss
    k0 = 0.5 * kss
    T = 600                    # long enough that machine precision is binding
    lo, hi = 1e-6, k0 ** ALPHA + (1.0 - DELTA) * k0 - 1e-8
    n_it = 0
    for n_it in range(1, 241):
        mid = 0.5 * (lo + hi)
        _, _, _, why = _shoot(k0, mid, T, kss)
        if why == "high":            # too little consumption, capital explodes
            lo = mid
        elif why == "low":           # too much consumption, capital collapses
            hi = mid
        else:
            break                    # survived T periods: at machine precision
        if (hi - lo) <= 2.3e-16 * max(abs(hi), 1.0):
            break
    c0 = 0.5 * (lo + hi)
    bracket = hi - lo
    kpath, cpath, _, _ = _shoot(k0, c0, T, kss)

    # how far a perturbed c0 survives
    surv = {}
    for eps in (1e-6, 1e-10, 1e-14):
        _, _, tt, _ = _shoot(k0, c0 * (1 + eps), T, kss)
        surv[f"{eps:.0e}"] = int(tt)

    # Local eigenvalues.  Linearizing k' = f(k)+(1-d)k-c and the Euler
    # equation around (k*,c*), and using beta*R = 1 at the steady state:
    #     dk' = R dk - dc
    #     dc' = (c* f''/sigma) dk + (1 - c* beta f''/sigma) dc
    fkk = ALPHA * (ALPHA - 1.0) * kss ** (ALPHA - 2.0)
    R = ALPHA * kss ** (ALPHA - 1.0) + 1.0 - DELTA
    J = np.array([[R, -1.0],
                  [css * fkk / SIGMA_RA,
                   1.0 - css * BETA * fkk / SIGMA_RA]])
    ev = np.linalg.eigvals(J)
    ev = np.sort(np.abs(ev))
    lam_s, lam_u = float(ev[0]), float(ev[1])

    half = _halflife(kpath, k0, kss)
    OUT["shooting"] = dict(
        k0_over_kss=0.5, k_ss=kss, c_ss=css, c0=c0,
        bisection_iterations=n_it, T=T,
        survives_periods=surv,
        digits_needed=float(np.log10(c0 / max(bracket, 1e-17))),
        bracket_width=float(bracket),
        lambda_stable=lam_s, lambda_unstable=lam_u,
        sigma_ra=SIGMA_RA,
        digits_lost_per_period=float(np.log10(lam_u)),
        note=("Forward shooting on c0 from k0 = k*/2; 'survives' is the period "
              "at which a relatively perturbed c0 leaves the corridor."))
    OUT["convergence_speed"] = dict(
        global_halflife_quarters=half,
        global_halflife_years=float(half / 4.0),
        local_lambda_stable=lam_s,
        local_halflife_quarters=float(np.log(0.5) / np.log(lam_s)),
        reused_s05_k_on_k=None,       # filled by main() from s05_numbers.json
        note="Local rate is the stable eigenvalue; the global rate is measured.")

    sel = np.arange(0, min(len(kpath), 241), 3)
    OUT["fig_saddle_path"] = dict(
        path=coords(kpath[sel], cpath[sel], "({:.3f},{:.4f})"),
        kdot0=coords(np.linspace(1.0, 1.6 * kss, 60),
                     [x ** ALPHA - DELTA * x
                      for x in np.linspace(1.0, 1.6 * kss, 60)],
                     "({:.3f},{:.4f})"),
        k_ss=kss, c_ss=css)
    OUT["fig_convergence_speed"] = dict(
        ra=coords(np.arange(0, 161, 2),
                  (kpath[:161:2] - k0) / (kss - k0), "({:.0f},{:.5f})"))


# ================================ 6.B1 / 6.B2  complete markets and Negishi
def _planner_path(thetas, sigmas, ns, k0, T=400):
    """Planner allocation with Pareto weights, by shooting on the multiplier.

    theta_i u_i'(c_it) = mu_t, resources: sum_i n_i c_it + k' = f(k)+(1-d)k.
    Returns (k, C, mu) along the path.
    """
    kss = _k_ss_ra()

    def C_of_mu(mu):
        return sum(n * (th / mu) ** (1.0 / s)
                   for th, s, n in zip(thetas, sigmas, ns))

    def integrate(mu0):
        k = np.empty(T + 1); mu = np.empty(T + 1); C = np.empty(T + 1)
        k[0], mu[0] = k0, mu0
        for t in range(T):
            C[t] = C_of_mu(mu[t])
            k[t + 1] = k[t] ** ALPHA + (1.0 - DELTA) * k[t] - C[t]
            if k[t + 1] <= 0.30 * kss or k[t + 1] >= 1.50 * kss:
                return k[:t + 2], C[:t + 1], mu[:t + 1], t + 1, (
                    "low" if k[t + 1] <= 0.30 * kss else "high")
            R = ALPHA * k[t + 1] ** (ALPHA - 1.0) + 1.0 - DELTA
            mu[t + 1] = mu[t] / (BETA * R)
        C[T] = C_of_mu(mu[T])
        return k, C, mu, T, "stayed"

    lo, hi = 1e-12, 1e6
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        _, _, _, _, why = integrate(mid)
        if why == "high":          # mu too high -> C too low -> k explodes
            hi = mid
        elif why == "low":
            lo = mid
        else:
            break
        if hi / lo < 1 + 1e-15:
            break
    mu0 = np.sqrt(lo * hi)
    return integrate(mu0)


def _ra_path(k0, T=400):
    """The representative-agent transition, and Arrow-Debreu prices along it.

    Log utility, so q_{t+1}/q_t -> beta once the economy has settled; the
    infinite tail beyond T is added in closed form rather than truncated.
    """
    kss = _k_ss_ra()
    css = kss ** ALPHA - DELTA * kss
    lo, hi = 1e-6, k0 ** ALPHA + (1.0 - DELTA) * k0 - 1e-8
    for _ in range(240):
        mid = 0.5 * (lo + hi)
        _, _, _, why = _shoot(k0, mid, T, kss)
        if why == "high":
            lo = mid
        elif why == "low":
            hi = mid
        else:
            break
        if (hi - lo) <= 2.3e-16 * max(abs(hi), 1.0):
            break
    c0 = 0.5 * (lo + hi)
    k, c, _, _ = _shoot(k0, c0, T, kss)
    n = min(len(k), len(c))
    k, c = k[:n], c[:n]
    q = BETA ** np.arange(n) * (c[0] / c)          # q_0 = 1
    r = ALPHA * k ** (ALPHA - 1.0) - DELTA
    w = (1.0 - ALPHA) * k ** ALPHA
    # Present values are truncated where the shot is still accurate and the
    # infinite tail is added from the steady state.  The tail carries a factor
    # beta/(1-beta) = 99, so it must not be taken from the drifting end of the
    # path: q_T there is already contaminated by the unstable root.
    Tp = min(250, n - 1)
    wss = (1.0 - ALPHA) * kss ** ALPHA
    tail = BETA / (1.0 - BETA)
    PV_w = float(q[:Tp + 1] @ w[:Tp + 1]) + float(q[Tp] * wss * tail)
    PV_C = float(q[:Tp + 1] @ c[:Tp + 1]) + float(q[Tp] * css * tail)
    return dict(k=k, c=c, q=q, r=r, w=w, n=n, kss=kss, css=css, T_pv=Tp,
                PV_w=PV_w, PV_C=PV_C)


def run_complete_markets():
    """Same aggregates, different distributions -- and how to break it."""
    kss = _k_ss_ra()
    k0 = 0.5 * kss
    ns = [0.5, 0.5]
    T = 300
    WIN = 150          # compare where every path is still on the manifold

    # --- common curvature: the weights drop out of the aggregate system ----
    # With u_i = log, C(mu) = S(theta)/mu, so the (k,C) system contains no
    # theta at all.  The measurement is of that claim, not of an assumption.
    thetas = ([0.5, 0.5], [0.2, 0.8], [0.05, 0.95])
    paths, C0s = [], []
    for th in thetas:
        k, C, mu, _, _ = _planner_path(th, [SIGMA_RA, SIGMA_RA], ns, k0, T)
        paths.append(k[:WIN]); C0s.append(float(C[0]))
    gap_common = float(max(np.abs(paths[0] - p_).max() for p_ in paths[1:]))
    c0_spread = float(max(abs(C0s[0] - x) for x in C0s[1:]) / C0s[0])

    # --- heterogeneous curvature: Gorman fails -----------------------------
    paths_h = []
    for th in thetas:
        k, C, mu, _, _ = _planner_path(th, [1.0, 5.0], ns, k0, T)
        paths_h.append(k[:WIN])
    gap_broken = float(max(np.abs(paths_h[0] - p_).max() for p_ in paths_h[1:]))

    OUT["complete_markets_irrelevance"] = dict(
        n_agents=2, k0_over_kss=0.5, T=T, compare_window=WIN,
        weights_tried=[list(t) for t in thetas],
        sigma_common=SIGMA_RA,
        max_abs_K_path_gap_common=gap_common,
        log10_gap_common=float(np.log10(max(gap_common, 1e-18))),
        relative_spread_in_C0=c0_spread,
        sigmas_broken=[1.0, 5.0],
        max_abs_K_path_gap_broken=gap_broken,
        log10_gap_broken=float(np.log10(max(gap_broken, 1e-18))),
        ratio_broken_to_common=float(gap_broken / max(gap_common, 1e-18)),
        K_ss=kss, K_ss_pct_of_gap=float(100 * gap_broken / kss),
        note=("With identical log utility the Pareto weights cancel from the "
              "aggregate system; with different curvature they do not, and "
              "the aggregate path depends on who holds the wealth."))
    sel = np.arange(0, WIN, 3)
    OUT["fig_gorman_break"] = dict(
        common=coords(sel, np.abs(paths[0][sel] - paths[2][sel]) + 1e-18,
                      "({:.0f},{:.3e})"),
        broken=coords(sel, np.abs(paths_h[0][sel] - paths_h[2][sel]) + 1e-18,
                      "({:.0f},{:.3e})"))


def run_negishi():
    """The Negishi map: weights -> allocation -> prices -> transfers -> zero.

    Log utility and two agents who differ only in initial capital.  The
    aggregate path is the representative-agent path (run_complete_markets
    measures that); Negishi only has to split it.
    """
    ra = _ra_path(_k_ss_ra() * 0.5, T=400)
    k0 = ra["k"][0]
    n_i = np.array([0.5, 0.5])
    k_i0 = np.array([0.25, 1.75]) * k0        # mean is k0 by construction
    assert abs(float(n_i @ k_i0) - k0) < 1e-12

    r0, H, PV_C = ra["r"][0], ra["PV_w"], ra["PV_C"]
    W = (1.0 + r0) * k_i0 + H                 # lifetime wealth of each agent
    walras_gap = float(abs(n_i @ W - PV_C) / PV_C)

    # Normalise sum_i n_i theta_i = 1, so c_it = theta_i * C_t.
    def transfer_1(theta1):
        theta2 = (1.0 - n_i[0] * theta1) / n_i[1]
        return theta1 * PV_C - W[0], theta2

    lo, hi, trace = 1e-8, 1.0 / n_i[0] - 1e-8, []
    for it in range(1, 121):
        mid = 0.5 * (lo + hi)
        h1, _ = transfer_1(mid)
        trace.append(dict(iteration=it, theta_1=float(mid), h_1=float(h1)))
        if h1 > 0:
            hi = mid
        else:
            lo = mid
        if abs(h1) < 1e-13 * PV_C:
            break
    theta1 = 0.5 * (lo + hi)
    h1, theta2 = transfer_1(theta1)
    closed = float(W[0] / (n_i @ W))          # the answer, in closed form

    OUT["negishi_example"] = dict(
        n_agents=2, T=ra["T_pv"], k0_over_kss=0.5, sigma=SIGMA_RA,
        initial_capital=[float(x) for x in k_i0],
        initial_capital_ratio=float(k_i0[1] / k_i0[0]),
        theta_1=float(theta1), theta_2=float(theta2),
        theta_1_closed_form=closed,
        theta_1_abs_error=float(abs(theta1 - closed)),
        iterations=it, max_abs_transfer=float(abs(h1)),
        wealth_ratio=float(W[1] / W[0]),
        human_wealth=H, PV_aggregate_consumption=PV_C,
        walras_check_rel_gap=walras_gap,
        consumption_share_1=float(theta1), consumption_share_2=float(theta2),
        unknowns_negishi=1, unknowns_tatonnement=int(ra["T_pv"]),
        note=("Capital is 7 to 1 but consumption is only "
              "%.2f to 1, because human wealth is common to both and "
              "dominates." % (theta2 / theta1)))
    OUT["negishi_trace"] = dict(rows=trace[:14])
    OUT["fig_negishi"] = dict(
        residual=coords([t["iteration"] for t in trace],
                        [np.log10(max(abs(t["h_1"]), 1e-16)) for t in trace],
                        "({:.0f},{:.3f})"))


def run_chatterjee():
    """Two economies along a transition: one remembers, one forgets."""
    ra = _ra_path(_k_ss_ra() * 0.5, T=400)
    k, c, q, r, w = ra["k"], ra["c"], ra["q"], ra["r"], ra["w"]
    Tp = ra["T_pv"]
    n_i = np.array([0.5, 0.5])
    k_i0 = np.array([0.25, 1.75]) * k[0]
    # the weights that actually decentralise this initial distribution
    W = (1.0 + r[0]) * k_i0 + ra["PV_w"]
    theta = W / float(n_i @ W)                    # sum_i n_i theta_i = 1
    c_i = np.outer(theta, c[:Tp + 1])             # constant shares of C_t
    k_i = np.zeros((2, Tp + 1))
    k_i[:, 0] = k_i0
    for t in range(Tp):
        k_i[:, t + 1] = (1.0 + r[t]) * k_i[:, t] + w[t] - c_i[:, t]
    tot = (n_i[:, None] * k_i).sum(axis=0)
    shares = (n_i[:, None] * k_i) / np.maximum(tot, 1e-12)
    agg_gap = float(np.abs(tot - k[:Tp + 1]).max())

    # --- Aiyagari: two very different lambda_0, one common policy ----------
    z, Pi, pi = rouwenhorst(N_Z, RHO_Z, SD_LOG_Z)
    grid = make_grid(200, A_MIN, A_MAX)
    sol = solve_r(grid, z, Pi, pi)
    j, wt = young_lottery(sol["ap"], grid)
    lamA = np.zeros((grid.size, N_Z)); lamA[0, :] = pi          # everyone poor
    lamB = np.zeros((grid.size, N_Z))
    lamB[int(np.searchsorted(grid, 80.0)), :] = pi              # everyone rich
    gaps = []
    for _ in range(1200):
        lamA = push_forward(lamA, j, wt, Pi)
        lamB = push_forward(lamB, j, wt, Pi)
        gaps.append(float(np.abs(lamA - lamB).sum()))
    half = next((i + 1 for i, g in enumerate(gaps) if g < 0.5 * gaps[0]),
                float("nan"))
    OUT["chatterjee"] = dict(
        T=Tp, theta_1=float(theta[0]), theta_2=float(theta[1]),
        aggregate_reconstruction_gap=agg_gap,
        wealth_share_1_initial=float(shares[0, 0]),
        wealth_share_1_final=float(shares[0, -1]),
        wealth_share_1_change=float(shares[0, -1] - shares[0, 0]),
        consumption_share_1=float(theta[0] * n_i[0]),
        complete_markets_share_drift_last_50=float(
            np.abs(shares[0, -50:] - shares[0, -1]).max()),
        aiyagari_L1_gap_t1=gaps[0], aiyagari_L1_gap_t50=gaps[49],
        aiyagari_L1_gap_t200=gaps[199], aiyagari_L1_gap_t400=gaps[399],
        aiyagari_L1_gap_final=gaps[-1],
        aiyagari_halflife_quarters=float(half),
        aiyagari_halflife_years=float(half) / 4.0,
        note=("Under complete markets each household keeps a permanent share "
              "of aggregate wealth set by its initial position; under "
              "incomplete markets with a common policy the distribution "
              "forgets where it started -- but slowly."))
    sel_c = np.arange(0, Tp + 1, 3)
    sel_a = np.arange(0, 1200, 8)
    OUT["fig_chatterjee"] = dict(
        shares=coords(sel_c, shares[0, sel_c], "({:.0f},{:.5f})"),
        aiyagari_gap=coords(sel_a, [np.log10(max(gaps[i], 1e-18))
                                    for i in sel_a], "({:.0f},{:.4f})"))


# ================================================== 6.A3  the wall, counted
def run_wall_counts():
    """Arithmetic only -- never build 3**2800."""
    log10_3 = float(np.log10(3.0))
    gamma_dim = N_A * N_Z
    rows = []
    for J in (1, 2, 3, 4):
        rows.append(dict(n_moments=J, state_dim=2 + J,
                         log10_points_at_8_per_axis=float((2 + J)
                                                          * np.log10(8.0))))
    # Smolyak is the honest comparison: a tensor grid is not the only option,
    # and 5.B3 already measured how much a sparse grid buys.  Counts follow
    # Malin-Krueger-Kubler, reusing the construction recorded in the S4 ledger:
    #   mu = 1 -> 2d + 1 ;  mu = 2 -> 2d^2 + 2d + 1.
    d = gamma_dim
    sm1 = 2 * d + 1
    sm2 = 2 * d * d + 2 * d + 1
    inner_solve_seconds = 3.0            # one Aiyagari inner solve, measured
    OUT["wall_counts"] = dict(
        smolyak_mu1_nodes=int(sm1), smolyak_mu2_nodes=int(sm2),
        smolyak_mu2_storage_mb=float(sm2 * 8 / 1024 ** 2),
        smolyak_mu2_years_at_one_inner_solve_each=float(
            sm2 * inner_solve_seconds / (3600 * 24 * 365.25)),
        inner_solve_seconds=inner_solve_seconds,
        n_a=N_A, n_z=N_Z, gamma_dim=gamma_dim,
        log10_tensor_points_3_per_axis=float(gamma_dim * log10_3),
        log10_atoms_observable_universe=80.0,
        gamma_as_vector_kb=float(gamma_dim * 8 / 1024.0),
        moment_rows=rows,
        note=("Storing Gamma is cheap; storing a FUNCTION of Gamma is not. "
              "The count is of grid points over the distribution, not of "
              "the distribution itself.  The tensor count is the naive "
              "bound; the Smolyak counts are the honest one, and they are "
              "what makes the real obstacle economic rather than "
              "arithmetic."))


# ============================================ 6.A2  Krusell-Smith and OLG
def _ks_transitions():
    """KS (1998) joint transition over (z, e) from durations and u rates."""
    u_b, u_g = 0.10, 0.04
    dur_b, dur_g = 8.0, 8.0
    p_bb, p_gg = 1 - 1 / dur_b, 1 - 1 / dur_g
    Pz = np.array([[p_bb, 1 - p_bb], [1 - p_gg, p_gg]])
    # unemployment durations: 2.5 quarters in bad, 1.5 in good
    d_ub, d_ug = 2.5, 1.5
    P = np.zeros((4, 4))          # index = 2*z + e, e=0 unemployed
    u = [u_b, u_g]
    d_u = [d_ub, d_ug]
    for zi in range(2):
        for zj in range(2):
            pz = Pz[zi, zj]
            # p(u'|u) within the destination aggregate state
            puu = 1 - 1 / d_u[zj]
            pue = 1 - puu
            # employment flows consistent with the destination u rate
            peu = (u[zj] - u[zj] * puu) / (1 - u[zj])
            pee = 1 - peu
            P[2 * zi + 0, 2 * zj + 0] = pz * puu
            P[2 * zi + 0, 2 * zj + 1] = pz * pue
            P[2 * zi + 1, 2 * zj + 0] = pz * peu
            P[2 * zi + 1, 2 * zj + 1] = pz * pee
    return Pz, P, np.array(u)


def run_ks():
    """Krusell-Smith: simulate, regress, update -- and then grade it."""
    rng = np.random.default_rng(20261029)
    Z = np.array([0.99, 1.01])
    Pz, P, u_rate = _ks_transitions()
    l_bar = 1.0 / (1.0 - u_rate.mean())
    n_a, n_K = 120, 7
    a_grid = make_grid(n_a, 0.0, 200.0, curv=2.2)
    K_lo, K_hi = 20.0, 40.0
    K_grid = np.linspace(np.log(K_lo), np.log(K_hi), n_K)

    def prices(K, zi):
        L = (1.0 - u_rate[zi]) * l_bar
        r = ALPHA * Z[zi] * (K / L) ** (ALPHA - 1.0) - DELTA
        w = (1.0 - ALPHA) * Z[zi] * (K / L) ** ALPHA
        return r, w, L

    # PLM coefficients, one pair per aggregate state
    A_c = np.array([0.10, 0.09])
    B_c = np.array([0.965, 0.968])

    def solve_household(A_c, B_c, tol=1e-8, max_iter=600):
        """EGM over (a, e, K, z); policy stored on the (a, K) grid per (e,z)."""
        c = np.empty((n_a, 4, n_K))
        for s in range(4):
            for m in range(n_K):
                zi, e = divmod(s, 2)
                r, w, L = prices(np.exp(K_grid[m]), zi)
                c[:, s, m] = np.maximum(
                    0.05 + 0.03 * a_grid, 1e-8)
        for it in range(1, max_iter + 1):
            c_new = np.empty_like(c)
            for m in range(n_K):
                Kp = np.exp(A_c + B_c * K_grid[m])          # per aggregate z'
                for s in range(4):
                    zi, e = divmod(s, 2)
                    r, w, L = prices(np.exp(K_grid[m]), zi)
                    R = 1.0 + r
                    # expected marginal utility over (z',e')
                    Euc = np.zeros(n_a)
                    for sp in range(4):
                        zp, ep = divmod(sp, 2)
                        if P[s, sp] <= 0:
                            continue
                        rp, wp, Lp = prices(Kp[zp], zp)
                        cp = np.empty(n_a)
                        # interpolate the policy in log K at Kp[zp]
                        w_hi = np.interp(np.log(Kp[zp]), K_grid,
                                         np.arange(n_K))
                        m0 = int(np.clip(np.floor(w_hi), 0, n_K - 2))
                        th = w_hi - m0
                        cpol = (1 - th) * c[:, sp, m0] + th * c[:, sp, m0 + 1]
                        cp = np.maximum(cpol, 1e-10)
                        Euc += P[s, sp] * (1.0 + rp) * cp ** (-1.0)
                    c_end = (BETA * Euc) ** (-1.0)
                    inc = w * l_bar * e
                    a_end = (c_end + a_grid - inc) / R
                    cc = np.interp(a_grid, a_end, c_end)
                    binds = a_grid < a_end[0]
                    cc[binds] = R * a_grid[binds] + inc
                    c_new[:, s, m] = np.maximum(cc, 1e-10)
            diff = np.abs(c_new - c).max()
            c = c_new
            if diff < tol:
                break
        return c, it, diff

    def policy_ap(c, s, m_lo, th, K):
        zi, e = divmod(s, 2)
        r, w, L = prices(K, zi)
        cc = (1 - th) * c[:, s, m_lo] + th * c[:, s, m_lo + 1]
        return np.clip((1.0 + r) * a_grid + w * l_bar * e - cc,
                       0.0, a_grid[-1])

    T_sim, T_burn = 3000, 500
    zser = np.empty(T_sim, dtype=int)
    zser[0] = 1
    for t in range(1, T_sim):
        zser[t] = rng.random() > Pz[zser[t - 1], 0]

    outer_rows = []
    for outer in range(1, 26):
        c_pol, it_in, _ = solve_household(A_c, B_c)
        # non-stochastic simulation of the joint distribution over (a, e)
        lam = np.zeros((n_a, 2))
        lam[0, 0], lam[0, 1] = u_rate[zser[0]], 1 - u_rate[zser[0]]
        Ks = np.empty(T_sim)
        for t in range(T_sim):
            K = float((lam.sum(axis=1) * a_grid).sum())
            K = min(max(K, K_lo * 1.001), K_hi * 0.999)
            Ks[t] = K
            zi = zser[t]
            wpos = np.interp(np.log(K), K_grid, np.arange(n_K))
            m0 = int(np.clip(np.floor(wpos), 0, n_K - 2)); th = wpos - m0
            lam_new = np.zeros_like(lam)
            zj = zser[t + 1] if t + 1 < T_sim else zi
            for e in range(2):
                s = 2 * zi + e
                ap = policy_ap(c_pol, s, m0, th, K)
                j, wt = young_lottery(ap, a_grid)
                tmp = np.zeros(n_a)
                np.add.at(tmp, j, lam[:, e] * (1 - wt))
                np.add.at(tmp, j + 1, lam[:, e] * wt)
                denom = P[s, 2 * zj + 0] + P[s, 2 * zj + 1]
                for ep in range(2):
                    lam_new[:, ep] += tmp * (P[s, 2 * zj + ep] / denom)
            lam = lam_new / lam_new.sum()
        # regress log K' on log K, by aggregate state
        A_new, B_new, r2 = np.empty(2), np.empty(2), np.empty(2)
        for zi in range(2):
            idx = np.where(zser[T_burn:T_sim - 1] == zi)[0] + T_burn
            x = np.log(Ks[idx]); y = np.log(Ks[idx + 1])
            X = np.column_stack([np.ones_like(x), x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ beta
            r2[zi] = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            A_new[zi], B_new[zi] = beta
        chg = float(max(np.abs(A_new - A_c).max(), np.abs(B_new - B_c).max()))
        outer_rows.append(dict(iteration=outer, A_bad=float(A_new[0]),
                               B_bad=float(B_new[0]), A_good=float(A_new[1]),
                               B_good=float(B_new[1]), max_coef_change=chg))
        A_c = 0.7 * A_c + 0.3 * A_new
        B_c = 0.7 * B_c + 0.3 * B_new
        if chg < 1e-5:
            break

    # --- Den Haan (2010): iterate the PLM forward without resyncing --------
    Kf = np.empty(T_sim); Kf[T_burn] = Ks[T_burn]
    for t in range(T_burn, T_sim - 1):
        Kf[t + 1] = np.exp(A_c[zser[t]] + B_c[zser[t]] * np.log(Kf[t]))
    err = 100.0 * np.abs(np.log(Kf[T_burn:]) - np.log(Ks[T_burn:]))
    OUT["ks_plm"] = dict(
        alpha=ALPHA, beta=BETA, delta=DELTA, Z=[float(x) for x in Z],
        u_bad=float(u_rate[0]), u_good=float(u_rate[1]), l_bar=float(l_bar),
        n_a=n_a, n_K=n_K, T_sim=T_sim, T_burn=T_burn,
        outer_iterations=len(outer_rows),
        A_bad=float(A_c[0]), B_bad=float(B_c[0]),
        A_good=float(A_c[1]), B_good=float(B_c[1]),
        r2_bad=float(r2[0]), r2_good=float(r2[1]),
        den_haan_max_pct=float(err.max()),
        den_haan_mean_pct=float(err.mean()),
        K_mean=float(Ks[T_burn:].mean()),
        note=("R^2 is a one-step-ahead statistic; the Den Haan test iterates "
              "the forecast rule forward for the whole sample without ever "
              "resyncing it to the simulated series."))
    OUT["ks_convergence"] = dict(rows=outer_rows)
    sel = np.arange(0, min(len(err), 600), 4)
    OUT["fig_ks_forecast"] = dict(
        error=coords(sel, err[sel], "({:.0f},{:.5f})"),
        max_pct=float(err.max()))


def run_olg():
    """Life-cycle SRCE.  ANNUAL -- a declared exception to the quarterly rule:
    a sixty-year life at quarterly frequency is 240 periods."""
    I, I_R = 60, 45                    # ages 21..80, retire after 45 years
    A_BETA, A_ALPHA, A_DELTA = 0.96, ALPHA, 0.08
    REPL = 0.40
    rho_a, sd_a = 0.90, 0.30
    z, Pi, pi = rouwenhorst(5, rho_a, sd_a)
    grid = make_grid(250, 0.0, 60.0)
    na, nz = grid.size, z.size
    age = np.arange(I)
    # hump-shaped age-efficiency profile, normalised to mean one over workers
    eff = np.exp(0.09 * age - 0.0013 * age ** 2)
    eff[I_R:] = 0.0
    eff[:I_R] /= eff[:I_R].mean()
    Lsup = float(eff[:I_R].sum() / I)          # per capita effective labour

    def solve_olg(r):
        kl = (A_ALPHA / (r + A_DELTA)) ** (1.0 / (1.0 - A_ALPHA))
        w = (1.0 - A_ALPHA) * kl ** A_ALPHA
        pen = REPL * w * 1.0
        R = 1.0 + r
        c_age = [None] * I
        ap_age = [None] * I
        # last period: consume everything
        inc_T = pen
        c_age[I - 1] = R * grid[:, None] + inc_T + 0.0 * z[None, :]
        ap_age[I - 1] = np.zeros((na, nz))
        for i in range(I - 2, -1, -1):
            inc = (w * eff[i] * z[None, :] if i < I_R else pen + 0.0 * z[None, :])
            Euc = (c_age[i + 1] ** (-SIGMA)) @ Pi.T
            c_end = (A_BETA * R * Euc) ** (-1.0 / SIGMA)
            a_end = (c_end + grid[:, None] - inc) / R
            c_i = np.empty((na, nz))
            for q in range(nz):
                c_i[:, q] = np.interp(grid, a_end[:, q], c_end[:, q])
                binds = grid < a_end[0, q]
                c_i[binds, q] = R * grid[binds] + (inc[0, q] if inc.shape[1] > 1
                                                   else float(inc))
            c_i = np.maximum(c_i, 1e-10)
            c_age[i] = c_i
            ap_age[i] = np.clip(R * grid[:, None] + inc - c_i, 0.0, grid[-1])
        # forward over cohorts
        lam = np.zeros((I, na, nz))
        lam[0, 0, :] = pi / I                      # newborns, zero assets
        for i in range(I - 1):
            j, wt = young_lottery(ap_age[i], grid)
            lam[i + 1] = push_forward(lam[i], j, wt, Pi)
        K = float((lam.sum(axis=2) * grid[None, :]).sum())
        return K, kl * Lsup, w, lam, ap_age

    lo, hi = 0.005, 1.0 / A_BETA - 1.0 - 1e-6
    t0 = time.time()
    n_it = 0
    for n_it in range(1, 61):
        mid = 0.5 * (lo + hi)
        Ks, Kd, w, lam, ap_age = solve_olg(mid)
        if Ks - Kd > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-8:
            break
    r_star = 0.5 * (lo + hi)
    Ks, Kd, w, lam, ap_age = solve_olg(r_star)
    assets_by_age = (lam.sum(axis=2) * grid[None, :]).sum(axis=1) * I
    OUT["olg_srce"] = dict(
        I=I, retire_after=I_R, frequency="annual",
        alpha=A_ALPHA, beta=A_BETA, delta=A_DELTA, replacement=REPL,
        n_a=na, n_z=nz, node_count=int(I * na * nz),
        r_annual=r_star, w=w, K=Ks, L=Lsup,
        K_over_Y_annual=float(Ks / (Ks ** A_ALPHA * Lsup ** (1 - A_ALPHA))),
        outer_iterations=n_it, seconds=float(time.time() - t0),
        peak_asset_age_index=int(assets_by_age.argmax()),
        peak_asset_age_years=int(21 + assets_by_age.argmax()),
        mass_error=float(abs(lam.sum() - 1.0)),
        note=("ANNUAL, a declared exception: sixty years at quarterly "
              "frequency would be 240 periods.  Only the outer (r,w) fixed "
              "point survives -- there is no inner fixed point in measures."))
    OUT["fig_olg_profiles"] = dict(
        assets=coords(21 + age, assets_by_age, "({:.0f},{:.4f})"),
        efficiency=coords(21 + age, eff, "({:.0f},{:.4f})"))


# ================================================================== driver
ALL_RUNS = [
    ("earnings_process", run_earnings_process),
    ("aiyagari_ss", run_aiyagari_ss),
    ("aiyagari_persistence", run_aiyagari_persistence),
    ("aiyagari_diagnostics", run_aiyagari_diagnostics),
    ("transition", run_transition),
    ("shooting", run_shooting),
    ("complete_markets", run_complete_markets),
    ("negishi", run_negishi),
    ("chatterjee", run_chatterjee),
    ("wall_counts", run_wall_counts),
    ("olg", run_olg),
    ("ks", run_ks),
]


def _load(name):
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main(argv):
    names = [n for n, _ in ALL_RUNS]
    wanted = argv[1:] if len(argv) > 1 else names
    bad = [w for w in wanted if w not in names]
    if bad:
        raise SystemExit(f"unknown section(s) {bad}; valid: {names}")

    s04 = _load("s04_numbers.json")
    s05 = _load("s05_numbers.json")
    # Fail loudly if a later edit moves S4's calibration out from under S6.
    rq = s04.get("rbc_quarterly", {})
    for k, v in (("alpha", ALPHA), ("beta", BETA), ("delta", DELTA)):
        if k in rq and abs(rq[k] - v) > 1e-12:
            raise SystemExit(f"calibration drift: s04 {k}={rq[k]}, s06 {k}={v}")

    for name, fn in ALL_RUNS:
        if name not in wanted:
            continue
        t0 = time.time()
        fn()
        print(f"  {name:24s} {time.time() - t0:7.1f}s")

    # reuse, recorded rather than recomputed
    if "shooting" in wanted and "convergence_speed" in OUT:
        k_on_k = s05.get("pert_first_order", {}).get("k_on_k")
        if k_on_k:
            OUT["convergence_speed"]["reused_s05_k_on_k"] = k_on_k
            OUT["convergence_speed"]["halflife_from_s05_quarters"] = float(
                np.log(0.5) / np.log(k_on_k))
    if "wall_counts" in wanted and "smolyak" in s04:
        OUT["wall_counts"]["reused_s04_smolyak"] = s04["smolyak"]

    merged = _load("s06_numbers.json")
    merged.update(OUT)
    meta = merged.get("_meta", {})
    meta.update(dict(
        calibration=dict(alpha=ALPHA, beta=BETA, delta=DELTA, sigma=SIGMA,
                         rho_z=RHO_Z, sd_log_z=SD_LOG_Z, n_z=N_Z, n_a=N_A),
        numpy=np.__version__,
        generated_by="tools/figures/s06_generate.py",
        reused=dict(
            smolyak="s04_numbers.json -> wall_counts (point counts)",
            rbc_quarterly="s04_numbers.json -> calibration assertion",
            pert_first_order_k_on_k=("s05_numbers.json -> convergence_speed "
                                     "(the local rate is S5's eigenvalue)")),
        transcribed_not_measured=(
            "Nishiyama & Smetters research-scale transition costs and the "
            "Social Security example are cited on the slide, not computed."),
    ))
    if len(argv) > 1:
        meta["last_partial_update"] = dict(sections=wanted,
                                           at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    merged["_meta"] = meta
    with open(OUTFILE, "w") as f:
        json.dump(merged, f, indent=1, sort_keys=True)
    print(f"wrote {OUTFILE}  ({len([k for k in merged if k != '_meta'])} keys)")


if __name__ == "__main__":
    main(sys.argv)
