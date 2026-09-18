#!/usr/bin/env python3
"""Every number and figure coordinate used in the Session 7 decks.

Session 7 stops choosing a basis in advance.  Block A builds the approximator
-- a neuron, a layer, a network -- and asks what approximation theory actually
promises; Block B writes a macro model as a loss function and trains a network
on it.  Block B ends by solving the SAME quarterly RBC model that 3.A4, 4.B5,
5.A6 and 5.B6 solved, so the course closes the arc with ONE MODEL, FIVE
METHODS.

Calibration continues Session 4's quarterly numbers (Quant_Macro labs/Lab7
RBCModel), which is also HW1 Part B Q3's:

    alpha = 0.33, beta = 0.99, delta = 0.025, rho = 0.95, sigma_eps = 0.007

Where a closed form is needed, Block A uses the Brock-Mirman benchmark of
S01_B1:192 (log utility, delta = 1), as Sessions 3 and 5 do.

The RBC model, the Rouwenhorst chain, the capital band and the unit-free Euler
metric are IMPORTED from tools/figures/s05_generate.py rather than copied, so
the fifth row of the methods table is measured on a model that is identical to
5.B6's by construction and not merely by inspection.

Output: tools/figures/s07_numbers.json -- the ledger.  No number appears on an
S7 slide unless it is a key in that file.  The optimizer numbers are NOT
recomputed here: 7.A3 reuses keys "cg_steps", "optimizer_memory",
"sgd_trajectory" and "sgd_dimension" from s04_numbers.json, per
slides/SOURCES.md and the carry-forward obligation recorded in the build plan.

Run:  sbatch tools/run_code.slurm python3 tools/figures/s07_generate.py
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s05_generate as S5           # the S5 RBC model, verbatim

ALPHA, BETA, DELTA = S5.ALPHA, S5.BETA, S5.DELTA
RHO, SIG_EPS = S5.RHO, S5.SIG_EPS
BM_ALPHA, BM_BETA = S5.BM_ALPHA, S5.BM_BETA

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = {}


def coords(xs, ys, fmt="({:.4f},{:.4f})"):
    return " ".join(fmt.format(float(x), float(y)) for x, y in zip(xs, ys))


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)


# ======================================================================  7.A2
# The activation family: the curves, and the saturation arithmetic that makes
# the vanishing gradient a number rather than a worry.
def run_activations():
    x = np.linspace(-3.0, 3.0, 121)
    sig = 1.0 / (1.0 + np.exp(-x))
    fam = {
        "sigmoid":   sig,
        "tanh":      np.tanh(x),
        "relu":      np.maximum(0.0, x),
        "leaky":     np.where(x > 0, x, 0.1 * x),
        "elu":       np.where(x > 0, x, np.expm1(x)),
        "softplus":  np.log1p(np.exp(x)),
        "swish":     x * sig,
        "gelu":      x * 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0))),
    }
    figs = {f"fig_act_{k}": coords(x, v) for k, v in fam.items()}

    # derivative of the logistic at zero, and what L layers of it cost
    dsig = 0.25
    chain = {str(L): dsig ** L for L in range(1, 8)}
    # the largest slope tanh can contribute, for contrast
    dtanh = 1.0

    OUT["activations"] = {
        "domain": [-3.0, 3.0],
        "sigmoid_prime_at_0": dsig,
        "tanh_prime_at_0": dtanh,
        "tanh_from_sigmoid": "tanh(z) = 2*sigmoid(2z) - 1",
        "chain_of_sigmoid_primes": chain,
        "five_layers": dsig ** 5,
        "ranges": {"sigmoid": [0.0, 1.0], "tanh": [-1.0, 1.0],
                   "relu": [0.0, "inf"], "leaky": ["-inf", "inf"],
                   "elu": [-1.0, "inf"], "softplus": [0.0, "inf"],
                   "swish": [-0.2784645, "inf"], "gelu": [-0.1700, "inf"]},
        "leaky_gamma": 0.1,
        "swish_min": float(np.min(np.linspace(-10, 0, 200001) /
                                  (1 + np.exp(-np.linspace(-10, 0, 200001))))),
    }
    OUT.update(figs)


# ----------------------------------------------------------------------------
# The constructed intuition both source decks build: a weighted sum of shifted
# ReLUs fitted to sin(x) by least squares.  With the shifts FIXED this is a
# linear regression on a piecewise-linear basis -- 5.B5's least squares on
# 5.B4's tent-like elements.  Let the shifts be TRAINED and it stops being
# regression: the network chooses where to put the kinks.
def _relu_design(x, b):
    return np.column_stack([np.ones_like(x)] + [np.maximum(0.0, x - bi) for bi in b])


def _fit_fixed(x, y, N, lo, hi):
    b = np.linspace(lo, hi, N + 2)[1:-1] if N > 0 else np.array([])
    A = _relu_design(x, b)
    a, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b, a, A @ a


def _fit_learned(x, y, N, lo, hi, iters=4000, seed=0):
    """Same functional form, but b is trained too -- one hidden ReLU layer."""
    seed_all(seed)
    xt = torch.tensor(x, dtype=torch.float64).view(-1, 1)
    yt = torch.tensor(y, dtype=torch.float64).view(-1, 1)
    b = torch.tensor(np.linspace(lo, hi, N + 2)[1:-1], dtype=torch.float64,
                     requires_grad=True)
    a = torch.zeros(N, dtype=torch.float64, requires_grad=True)
    a0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a, b, a0], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        pred = a0 + (torch.relu(xt - b) * a).sum(1, keepdim=True)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (a0 + (torch.relu(xt - b) * a).sum(1, keepdim=True)).numpy().ravel()
    return b.detach().numpy(), pred


def run_relu_construction():
    lo, hi = -np.pi, np.pi
    x = np.linspace(lo, hi, 801)
    y = np.sin(x)

    rows, figs = [], {}
    for N in (2, 4, 8, 16):
        b, a, yhat = _fit_fixed(x, y, N, lo, hi)
        bl, yl = _fit_learned(x, y, N, lo, hi)
        rows.append({
            "kinks": N,
            "unknowns_fixed": N + 1,          # the a's and the bias
            "unknowns_learned": 2 * N + 1,    # a's, b's and the bias
            "max_err_fixed": float(np.max(np.abs(yhat - y))),
            "rmse_fixed": float(np.sqrt(np.mean((yhat - y) ** 2))),
            "max_err_learned": float(np.max(np.abs(yl - y))),
            "rmse_learned": float(np.sqrt(np.mean((yl - y) ** 2))),
            "knots_fixed": [float(v) for v in b],
            "knots_learned": sorted(float(v) for v in bl),
        })
        if N in (2, 4, 8):
            figs[f"fig_relu_fit_{N}"] = coords(x, yhat)
            figs[f"fig_relu_dev_{N}"] = coords(x, yhat - y)
    figs["fig_relu_target"] = coords(x, y)

    # The same budget on a KINKED target: consumption against cash-on-hand with
    # a borrowing limit.  This is where a fixed smooth basis loses (5.B2's
    # Gibbs frame) and a learned kink wins outright.
    xk = np.linspace(0.0, 4.0, 801)
    yk = np.minimum(xk, 1.0 + 0.25 * (xk - 1.0))       # kink at x = 1
    krows = []
    for N in (2, 4, 8):
        _, _, yhf = _fit_fixed(xk, yk, N, 0.0, 4.0)
        bl, ylk = _fit_learned(xk, yk, N, 0.0, 4.0)
        krows.append({
            "kinks": N,
            "max_err_fixed": float(np.max(np.abs(yhf - yk))),
            "max_err_learned": float(np.max(np.abs(ylk - yk))),
            "nearest_learned_knot_to_true_kink": float(
                min(bl, key=lambda v: abs(v - 1.0))),
        })
    figs["fig_relu_kink_target"] = coords(xk, yk)

    OUT["relu_construction"] = {
        "target": "sin(x) on [-pi, pi]",
        "form": "yhat(x) = a0 + sum_i a_i * ReLU(x - b_i)",
        "fixed_knots_method": "evenly spaced b_i, coefficients by least squares",
        "learned_knots_method": "a_i and b_i both by Adam, 4000 steps, lr 0.05",
        "n_points": int(x.size),
        "rows": rows,
        "kinked_target": "c(x) = min(x, 1 + 0.25(x-1)), kink at x = 1",
        "kinked_rows": krows,
    }
    OUT.update(figs)


# ----------------------------------------------------------------------------
# XOR, represented exactly -- not approximated.  Goodfellow, Bengio & Courville
# section 6.1.  Two ReLU units suffice, and the arithmetic is checkable by hand.
def run_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    tbl = []
    for x1, x2 in X:
        h1 = max(0.0, x1 + x2)
        h2 = max(0.0, x1 + x2 - 1.0)
        yh = h1 - 2.0 * h2
        tbl.append({"x1": x1, "x2": x2, "h1": h1, "h2": h2,
                    "yhat": yh, "xor": float(int(x1) ^ int(x2))})
    OUT["xor"] = {
        "h1": "ReLU(x1 + x2)", "h2": "ReLU(x1 + x2 - 1)", "yhat": "h1 - 2*h2",
        "rows": tbl,
        "exact": all(abs(r["yhat"] - r["xor"]) < 1e-12 for r in tbl),
        "note": "a single linear map cannot do this; one hidden layer does it exactly",
    }


# ----------------------------------------------------------------------------
# Parameter counting: what a network costs against what a tensor basis costs.
def run_param_count():
    d, width, L = 20, 256, 3
    layers = [(d + 1) * width] + [(width + 1) * width] * (L - 1) + [width + 1]
    total = int(sum(layers))
    # C(d+k, k) counts the COMPLETE basis: every monomial of total degree <= k.
    # The TENSOR-PRODUCT basis is a different and much larger object, (k+1)^d.
    # 7.A2 had mislabelled the first as the second; both are recorded here so the
    # slide can show the gap that makes the complete basis worth having.
    poly_complete = {str(k): int(math.comb(d + k, k)) for k in (2, 3, 5)}
    poly_tensor = {str(k): float((k + 1) ** d) for k in (2, 3, 5)}
    grid = {str(n): float(n) ** d for n in (3, 5, 10)}

    # the CNN pipeline of ch.9 fig:cnn_pipeline, counted the same way
    conv1 = 32 * (3 * 3 * 3 + 1)
    conv2 = 64 * (3 * 3 * 32 + 1)
    head = 4096 * 64 + 64
    out = 64 + 1
    cnn_total = conv1 + conv2 + head + out

    # and the 1-D schematic of fig:cnn_vs_fc
    fc_1d, conv_1d = 7 * 5, 3

    OUT["param_count"] = {
        "d": d, "width": width, "hidden_layers": L,
        "per_layer": [int(v) for v in layers], "network_total": total,
        "polynomial_terms_by_degree": poly_complete,
        "complete_poly_terms_by_degree": poly_complete,
        "tensor_poly_terms_by_degree": poly_tensor,
        "_poly_note": "complete = all monomials of total degree <= k, C(d+k,k); "
                      "tensor = (k+1)^d, one degree-k factor per axis",
        "tensor_grid_points": grid,
        "dense_first_layer_on_256x256x3": int(196608 * 256),
        "cnn": {"conv1": conv1, "conv2": conv2, "fc_head": head, "out": out,
                "total": cnn_total,
                "conv_share_pct": round(100.0 * (conv1 + conv2) / cnn_total, 1),
                "head_share_pct": round(100.0 * head / cnn_total, 1)},
        "schematic_1d": {"fully_connected": fc_1d, "convolution_width3": conv_1d,
                         "inputs": 7, "outputs": 5},
    }


# ----------------------------------------------------------------------------
# A network against a Chebyshev expansion, same target, comparable budget.
# 5.B2's Gibbs point, now with the adaptive basis in the race.
def _mlp(nh=(16, 16), act=nn.Tanh):
    layers, prev = [], 1
    for h in nh:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers).double()


def _train_mlp(net, x, y, iters=6000, lr=0.01, seed=0):
    seed_all(seed)
    xt = torch.tensor(x, dtype=torch.float64).view(-1, 1)
    yt = torch.tensor(y, dtype=torch.float64).view(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(xt).numpy().ravel()


def run_nn_vs_cheb():
    from numpy.polynomial import chebyshev as npcheb
    x = np.linspace(-1.0, 1.0, 801)
    targets = {
        "smooth": np.exp(-x) * np.sin(3.0 * x),
        "kinked": np.maximum(0.0, x - 0.3) - 0.5 * np.maximum(0.0, -x - 0.4),
    }
    rows, figs = [], {}
    for name, y in targets.items():
        for deg in (8, 16):
            cf = npcheb.chebfit(x, y, deg)
            yc = npcheb.chebval(x, cf)
            rows.append({"target": name, "method": f"Chebyshev deg {deg}",
                         "params": deg + 1,
                         "max_err": float(np.max(np.abs(yc - y))),
                         "rmse": float(np.sqrt(np.mean((yc - y) ** 2)))})
            if deg == 16:
                figs[f"fig_cheb_{name}"] = coords(x, yc - y)
        seed_all(0)
        net = _mlp((16, 16), act=nn.ReLU)
        npar = sum(p.numel() for p in net.parameters())
        yn = _train_mlp(net, x, y, seed=0)
        rows.append({"target": name, "method": "ReLU net 16-16",
                     "params": int(npar),
                     "max_err": float(np.max(np.abs(yn - y))),
                     "rmse": float(np.sqrt(np.mean((yn - y) ** 2)))})
        figs[f"fig_net_{name}"] = coords(x, yn - y)
        figs[f"fig_target_{name}"] = coords(x, y)

        # A PARAMETER-MATCHED network: 1-5-1 is 16 parameters against the
        # degree-16 Chebyshev expansion's 17 coefficients.  Without this row the
        # comparison above is open to the objection that the network was simply
        # given more to work with.
        seed_all(0)
        small = _mlp((5,), act=nn.ReLU)
        nsmall = sum(p.numel() for p in small.parameters())
        ys = _train_mlp(small, x, y, seed=0)
        rows.append({"target": name, "method": "ReLU net 5 (matched)",
                     "params": int(nsmall),
                     "max_err": float(np.max(np.abs(ys - y))),
                     "rmse": float(np.sqrt(np.mean((ys - y) ** 2)))})
        figs[f"fig_net_small_{name}"] = coords(x, ys - y)
    OUT["nn_vs_cheb"] = {"domain": [-1.0, 1.0], "n_points": int(x.size),
                         "rows": rows,
                         "note": "deviation from the target, not the level"}
    OUT.update(figs)


# ======================================================================  7.A3
# The worked backward pass, by hand and by autograd.  ch.9 section 9.3.2.
def run_backprop_worked():
    x, ytrue = 2.0, 3.0
    w1, b1, v, c = 0.5, -0.1, 1.2, 0.3
    eta = 0.01

    z = w1 * x + b1
    h = max(0.0, z)
    yhat = v * h + c
    loss = (yhat - ytrue) ** 2

    dl_dyhat = 2.0 * (yhat - ytrue)
    dl_dv = dl_dyhat * h
    dl_dc = dl_dyhat
    dl_dh = dl_dyhat * v
    dl_dz = dl_dh * (1.0 if z > 0 else 0.0)
    dl_dw1 = dl_dz * x
    dl_db1 = dl_dz
    w1_new = w1 - eta * dl_dw1

    # the same thing, by the tape
    tw1 = torch.tensor(w1, requires_grad=True, dtype=torch.float64)
    tb1 = torch.tensor(b1, requires_grad=True, dtype=torch.float64)
    tv = torch.tensor(v, requires_grad=True, dtype=torch.float64)
    tc = torch.tensor(c, requires_grad=True, dtype=torch.float64)
    tz = tw1 * x + tb1
    th = torch.relu(tz)
    tl = (tv * th + tc - ytrue) ** 2
    tl.backward()

    OUT["backprop_worked"] = {
        "inputs": {"x": x, "y": ytrue, "w1": w1, "b1": b1, "v": v, "c": c,
                   "eta": eta, "hidden": "ReLU", "loss": "(yhat - y)^2"},
        "forward": {"z": z, "h": h, "yhat": yhat, "loss": loss},
        "backward": {"dl_dyhat": dl_dyhat, "dl_dv": dl_dv, "dl_dc": dl_dc,
                     "dl_dh": dl_dh, "dl_dz": dl_dz, "dl_dw1": dl_dw1,
                     "dl_db1": dl_db1},
        "update": {"w1_new": w1_new},
        "autograd_check": {"dl_dw1": float(tw1.grad), "dl_db1": float(tb1.grad),
                           "dl_dv": float(tv.grad), "dl_dc": float(tc.grad)},
        "agrees": bool(abs(float(tw1.grad) - dl_dw1) < 1e-12
                       and abs(float(tv.grad) - dl_dv) < 1e-12),
        "params": 4,
    }


# ----------------------------------------------------------------------------
# ch.9 section 9.3.4: how much an epoch buys, and when the network is actually
# worth it.  The honest answer is that at n = 2000 it loses to OLS.
def _consumption_dgp(n, d=30, n_signal=13, sig=0.25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = np.zeros(d)
    beta[:n_signal] = rng.normal(size=n_signal)
    lin = X @ beta
    nonlin = 0.5 * X[:, 0] * X[:, 1] + 0.3 * np.tanh(2.0 * X[:, 2])
    y = lin + nonlin + sig * rng.normal(size=n)
    return X, y


def _mlp_tab(d, hidden=(128, 64, 32), p_drop=0.0):
    layers, prev = [], d
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if p_drop > 0:
            layers += [nn.Dropout(p_drop)]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers).double()


def _train_tab(n, seed, wd=0.0, p_drop=0.0, epochs=150, B=256, lr=1e-3):
    X, y = _consumption_dgp(n, seed=seed)
    ntr = int(0.8 * n)
    Xtr, ytr, Xva, yva = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    mu, sd = Xtr.mean(0), Xtr.std(0)          # train statistics only
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    seed_all(seed)
    net = _mlp_tab(X.shape[1], p_drop=p_drop)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr).view(-1, 1)
    Xv = torch.tensor(Xva); yv = torch.tensor(yva).view(-1, 1)
    per_epoch = max(1, ntr // B)
    hist = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(ntr)
        for i in range(per_epoch):
            idx = perm[i * B:(i + 1) * B]
            opt.zero_grad()
            loss = ((net(Xt[idx]) - yt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tr = float(((net(Xt) - yt) ** 2).mean())
            va = float(((net(Xv) - yv) ** 2).mean())
        hist.append((ep + 1, tr, va))
    # OLS on the same split
    A = np.column_stack([np.ones(ntr), Xtr])
    bhat, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    ols = float(np.mean((np.column_stack([np.ones(len(yva)), Xva]) @ bhat - yva) ** 2))
    return hist, ols, per_epoch


def run_epoch_tradeoff():
    rows, figs = [], {}
    for n in (2000, 10000, 50000):
        per_seed = []
        for s in (0, 1, 2):
            hist, ols, per_epoch = _train_tab(n, s)
            best = min(hist, key=lambda r: r[2])
            per_seed.append((hist[0][2], best[0], best[2], hist[-1][1], ols, per_epoch))
        arr = np.array(per_seed, float)
        med = np.median(arr, axis=0)
        rows.append({
            "n": n, "steps_per_epoch": int(med[5]),
            "val_after_epoch1": float(med[0]),
            "best_epoch": int(med[1]), "best_val": float(med[2]),
            "best_steps": int(med[1] * med[5]),
            "final_train": float(med[3]), "ols_val": float(med[4]),
            "net_beats_ols": bool(med[2] < med[4]),
        })
        if n == 2000:
            hist, _, _ = _train_tab(n, 0)
            figs["fig_epoch_train"] = coords([r[0] for r in hist], [r[1] for r in hist])
            figs["fig_epoch_val"] = coords([r[0] for r in hist], [r[2] for r in hist])
    OUT["epoch_tradeoff"] = {
        "dgp": "d=30, 13 of them signal, 17 irrelevant; one interaction and one "
               "tanh term; noise sd 0.25 (variance 0.0625) is the floor",
        "net": "[128, 64, 32] ReLU, AdamW lr 1e-3, batch 256, 150 epochs",
        "split": "80/20, standardised on training statistics only",
        "seeds": [0, 1, 2], "reported": "median over seeds",
        "rows": rows,
    }
    OUT.update(figs)


def run_regularization():
    rows = []
    specs = [("none", 0.0, 0.0), ("weight decay 1e-4", 1e-4, 0.0),
             ("dropout 0.1", 0.0, 0.1), ("dropout 0.2", 0.0, 0.2),
             ("wd 1e-4 + dropout 0.1", 1e-4, 0.1)]
    for name, wd, pd in specs:
        row = {"strategy": name}
        for n in (2000, 50000):
            best = []
            for s in (0, 1, 2):
                hist, _, _ = _train_tab(n, s, wd=wd, p_drop=pd)
                b = min(hist, key=lambda r: r[2])
                best.append((b[0], b[2]))
            a = np.array(best, float)
            row[f"n{n}_best_epoch"] = int(np.median(a[:, 0]))
            row[f"n{n}_best_val"] = float(np.median(a[:, 1]))
        rows.append(row)
    OUT["regularization"] = {
        "note": "same DGP, net and budget as epoch_tradeoff; median over 3 seeds",
        "rows": rows,
    }


# ======================================================================  7.B1
# The quarterly RBC model, solved a fifth way: a policy network trained on the
# Euler residual.  Same Proj object, same band, same unit-free metric as 5.B6.
class PolicyNet(nn.Module):
    """c(k, z) with feasibility built in: c = sigmoid(net) * cash-on-hand."""

    def __init__(self, n_z, hidden=32, layers=2):
        super().__init__()
        seq, prev = [], 1 + n_z
        for _ in range(layers):
            seq += [nn.Linear(prev, hidden), nn.Tanh()]
            prev = hidden
        seq += [nn.Linear(prev, 1)]
        self.f = nn.Sequential(*seq).double()

    def forward(self, kn, zoh):
        return torch.sigmoid(self.f(torch.cat([kn, zoh], 1)))


def run_worked_nn_growth():
    P = S5.Proj(n_coef=7)                       # identical model to 5.B6
    z = torch.tensor(P.z, dtype=torch.float64)
    Pi = torch.tensor(P.Pi, dtype=torch.float64)
    klo, khi, nz = P.klo, P.khi, P.n_z
    a, b, d = P.alpha, P.beta, P.delta

    def norm(k):
        return (2.0 * (k - klo) / (khi - klo) - 1.0)

    def cash(k, j):
        return z[j] * k ** a + (1 - d) * k

    seed_all(0)
    net = PolicyNet(nz, hidden=64, layers=3)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    eye = torch.eye(nz, dtype=torch.float64)

    # Train to a PLATEAU, not to a budget: the comparison against 5.B6 is only
    # honest if the network has stopped improving.  Cosine decay to 1e-5.
    n_batch, iters = 1024, 30000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-5)
    t0 = time.perf_counter()
    hist = []
    for it in range(iters):
        k = torch.rand(n_batch, 1, dtype=torch.float64) * (khi - klo) + klo
        j = torch.randint(0, nz, (n_batch,))
        y = z[j].view(-1, 1) * k ** a + (1 - d) * k
        c = net(norm(k), eye[j]) * y
        kp = torch.clamp(y - c, klo, khi)
        rhs = torch.zeros_like(c)
        for l in range(nz):
            cp = net(norm(kp), eye[l].expand(n_batch, nz)) * \
                 (z[l] * kp ** a + (1 - d) * kp)
            R = a * z[l] * kp ** (a - 1) + (1 - d)
            rhs = rhs + Pi[j, l].view(-1, 1) * R / cp
        resid = 1.0 - c * b * rhs
        loss = (resid ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (it + 1) % 1000 == 0:
            hist.append((it + 1, float(loss.detach())))
    secs = time.perf_counter() - t0

    def cfun(kv, jj):
        with torch.no_grad():
            kt = torch.tensor(np.atleast_1d(kv), dtype=torch.float64).view(-1, 1)
            zo = eye[jj].expand(kt.shape[0], nz)
            y = P.z[jj] * kt ** a + (1 - d) * kt
            return (net(norm(kt), zo) * y).numpy().ravel()

    band_near = (0.9 * P.kss, 1.1 * P.kss)
    Ew = S5.euler_err_from_cpol(cfun, P, n_test=2000, seed=0)
    En = S5.euler_err_from_cpol(cfun, P, n_test=2000, seed=0, band=band_near)

    # WHERE is the worst residual?  A max driven by the edge of the sampling
    # region is a different diagnosis from one spread over the domain, and the
    # slide has to say which.  Sweep k, and also re-measure on the interior 90%.
    span = khi - klo
    inner = (klo + 0.05 * span, khi - 0.05 * span)
    Ei = S5.euler_err_from_cpol(cfun, P, n_test=2000, seed=0, band=inner)
    ks_diag = np.linspace(klo, khi, 120)
    err_k = []
    for kk in ks_diag:
        e = 0.0
        for jj in range(nz):
            c = float(cfun(np.array([kk]), jj)[0])
            y = P.z[jj] * kk ** a + (1 - d) * kk
            kp = min(max(y - c, klo), khi)
            rhs = 0.0
            for l in range(nz):
                cp = float(cfun(np.array([kp]), l)[0])
                R = a * P.z[l] * kp ** (a - 1) + (1 - d)
                rhs += P.Pi[jj, l] * R / max(cp, 1e-12)
            e = max(e, abs(1.0 - c * b * rhs))
        err_k.append(e)
    err_k = np.array(err_k)
    i_worst = int(np.argmax(err_k))
    k_worst = float(ks_diag[i_worst])

    ks = np.linspace(klo, khi, 200)
    mid = nz // 2
    sav = np.array([cash(torch.tensor(k), mid).item() - cfun(np.array([k]), mid)[0]
                    for k in ks])

    OUT["worked_nn_growth"] = {
        "model": "quarterly RBC, Rouwenhorst 7 states, k in [0.5 k*, 1.5 k*] "
                 "-- the same Proj object 5.B6 uses",
        "k_ss": P.kss, "k_lo": klo, "k_hi": khi,
        "architecture": "3 hidden layers of 64, tanh; c = sigmoid(net)*cash",
        "feasibility": "0 < c < cash-on-hand by construction, not by penalty",
        "expectation": "exact over the 7 Rouwenhorst states, not sampled",
        "parameters": int(npar),
        "batch": n_batch, "iterations": iters,
        "optimizer": "Adam, lr 3e-3 cosine-decayed to 1e-5",
        "final_loss": hist[-1][1],
        "loss_last_tenth_ratio": hist[-1][1] / hist[-len(hist) // 10 - 1][1],
        "solve_seconds": round(secs, 3),
        "max_log10_euler_wide": float(np.log10(np.max(Ew))),
        "mean_log10_euler_wide": float(np.log10(np.mean(Ew))),
        "max_log10_euler_near": float(np.log10(np.max(En))),
        "mean_log10_euler_near": float(np.log10(np.mean(En))),
        "max_log10_euler_inner90": float(np.log10(np.max(Ei))),
        "mean_log10_euler_inner90": float(np.log10(np.mean(Ei))),
        "worst_k": k_worst,
        "worst_k_as_fraction_of_kss": k_worst / P.kss,
        "worst_at_boundary": bool(min(k_worst - klo, khi - k_worst) < 0.05 * span),
        "log10_err_at_k_lo": float(np.log10(err_k[0])),
        "log10_err_at_k_ss": float(np.log10(err_k[int(np.argmin(abs(ks_diag - P.kss)))])),
        "log10_err_at_k_hi": float(np.log10(err_k[-1])),
        "loss_history": [{"iter": i, "loss": v} for i, v in hist],
        "device": str(next(net.parameters()).device),
    }
    OUT["fig_nn_saving"] = coords(ks, sav - ks)
    OUT["fig_nn_errk"] = coords(ks_diag, np.log10(err_k))
    OUT["fig_nn_loss"] = coords([h[0] for h in hist],
                                [np.log10(max(h[1], 1e-18)) for h in hist])


def run_five_methods():
    """Append the network to 5.B6's table.  The first four rows are READ from
    the S5 ledger, never recomputed, so the comparison cannot drift."""
    s05 = json.load(open("tools/figures/s05_numbers.json"))
    tm = s05["three_methods"]
    w = OUT["worked_nn_growth"]
    rows = list(tm["rows"]) + [{
        "method": "neural network (7.B1)",
        "numbers_stored": w["parameters"],
        "solve_seconds": w["solve_seconds"],
        "max_log10_euler_wide": w["max_log10_euler_wide"],
        "mean_log10_euler_wide": w["mean_log10_euler_wide"],
        "max_log10_euler_near": w["max_log10_euler_near"],
        "mean_log10_euler_near": w["mean_log10_euler_near"],
    }]
    OUT["five_methods"] = {
        "model": tm["model"], "k_ss": tm["k_ss"],
        "first_four_rows_source": "tools/figures/s05_numbers.json key "
                                  "'three_methods' -- read, not recomputed",
        "rows": rows,
    }


# ======================================================================  7.B3
# Structure as a hard constraint: an input-convex network cannot represent a
# non-convex function, and that is the point.
class ICNN(nn.Module):
    """Amos, Xu & Kolter (2017).  Non-negative W_z keeps the map convex in x."""

    def __init__(self, hidden=32, layers=2):
        super().__init__()
        self.Wy = nn.ModuleList([nn.Linear(1, hidden) for _ in range(layers)]
                                + [nn.Linear(1, 1)])
        self.Wz = nn.ModuleList([nn.Linear(hidden, hidden, bias=False)
                                 for _ in range(layers - 1)]
                                + [nn.Linear(hidden, 1, bias=False)])
        self.double()

    def forward(self, x):
        # Non-negativity by REPARAMETERISATION (softplus), not by clamping -- which
        # is what 7.B3 tells students to do, so the measured numbers must come from
        # the same construction.  A clamped weight at the boundary gets no gradient.
        zc = torch.relu(self.Wy[0](x))
        for i, Wz in enumerate(self.Wz):
            zc = torch.nn.functional.linear(
                zc, torch.nn.functional.softplus(Wz.weight)) + self.Wy[i + 1](x)
            if i < len(self.Wz) - 1:
                zc = torch.relu(zc)
        return zc

    def project(self):
        return None            # nothing to project: the parameterisation is the constraint


def _convexity_violation(xs, ys):
    """Largest negative second difference, scaled -- zero means convex."""
    h = xs[1] - xs[0]
    d2 = (ys[2:] - 2 * ys[1:-1] + ys[:-2]) / h ** 2
    return float(max(0.0, -d2.min()))


def run_icnn():
    """FFNN vs ICNN on a convex target -- ACROSS SEEDS.

    A single seed is not evidence: the first version of this section reported the
    ICNN as nineteen times more accurate, and L07 reproduced the opposite
    ordering from a different initialisation.  What is robust is the convexity
    violation, not the RMSE, and the slide now says only that."""
    x = np.linspace(-1.0, 1.0, 401)
    y = 0.6 * x ** 2 + 0.25 * x + 0.1        # convex target, f'' = 1.2
    ync = np.sin(3.0 * x)                    # not convex
    xt = torch.tensor(x, dtype=torch.float64).view(-1, 1)
    yt = torch.tensor(y, dtype=torch.float64).view(-1, 1)
    ynt = torch.tensor(ync, dtype=torch.float64).view(-1, 1)

    seeds = (0, 1, 2, 3, 4)
    ff_rmse, ic_rmse, ff_viol, ic_viol, nc_rmse, nc_viol = [], [], [], [], [], []
    yff_keep = yic_keep = yic2_keep = None

    for sd in seeds:
        seed_all(sd)
        ff = _mlp((32, 32), act=nn.ReLU)
        yff = _train_mlp(ff, x, y, iters=4000, seed=sd)
        ff_rmse.append(float(np.sqrt(np.mean((yff - y) ** 2))))
        ff_viol.append(_convexity_violation(x, yff))

        seed_all(sd)
        ic = ICNN()
        opt = torch.optim.Adam(ic.parameters(), lr=0.01)
        for _ in range(4000):
            opt.zero_grad()
            ((ic(xt) - yt) ** 2).mean().backward()
            opt.step(); ic.project()
        with torch.no_grad():
            yic = ic(xt).numpy().ravel()
        ic_rmse.append(float(np.sqrt(np.mean((yic - y) ** 2))))
        ic_viol.append(_convexity_violation(x, yic))

        seed_all(sd)
        ic2 = ICNN()
        opt2 = torch.optim.Adam(ic2.parameters(), lr=0.01)
        for _ in range(4000):
            opt2.zero_grad()
            ((ic2(xt) - ynt) ** 2).mean().backward()
            opt2.step(); ic2.project()
        with torch.no_grad():
            yic2 = ic2(xt).numpy().ravel()
        nc_rmse.append(float(np.sqrt(np.mean((yic2 - ync) ** 2))))
        nc_viol.append(_convexity_violation(x, yic2))

        if sd == 0:
            yff_keep, yic_keep, yic2_keep = yff, yic, yic2

    med = lambda a: float(np.median(a))
    OUT["icnn"] = {
        "convex_target": "0.6 x^2 + 0.25 x + 0.1 on [-1, 1], f'' = 1.2 everywhere",
        "nonconvex_target": "sin(3x) on [-1, 1]",
        "seeds": list(seeds),
        "ffnn_params": 1153, "icnn_params": 1186,
        "convex": {
            "ffnn_rmse_median": med(ff_rmse), "icnn_rmse_median": med(ic_rmse),
            "ffnn_rmse_range": [min(ff_rmse), max(ff_rmse)],
            "icnn_rmse_range": [min(ic_rmse), max(ic_rmse)],
            "icnn_more_accurate_in_n_of_5": int(sum(a > b for a, b in zip(ff_rmse, ic_rmse))),
            "ffnn_violation_median": med(ff_viol),
            "ffnn_violation_range": [min(ff_viol), max(ff_viol)],
            "ffnn_violated_every_seed": bool(min(ff_viol) > 0.1),
            "icnn_violation_max": float(max(ic_viol)),
        },
        "nonconvex": {
            "icnn_rmse_median": med(nc_rmse),
            "icnn_violation_max": float(max(nc_viol)),
            "note": "the ICNN cannot fit it, by construction -- the guarantee from the other side",
        },
        "verdict": "ACCURACY IS NOT ROBUST across seeds; the CONVEXITY VIOLATION is. "
                   "Slides must claim only the second.",
    }
    OUT["fig_icnn_convex_dev"] = coords(x, yic_keep - y)
    OUT["fig_ffnn_convex_dev"] = coords(x, yff_keep - y)
    OUT["fig_icnn_nonconvex"] = coords(x, yic2_keep)
    OUT["fig_icnn_nonconvex_target"] = coords(x, ync)


# ----------------------------------------------------------------------------
# The second worked backward pass: the sigmoid classifier of AE Lec.3 T3.  Two
# layers, so the error signal is actually PROPAGATED rather than just formed.
def run_backprop_sigmoid():
    import math as _m
    x, t = 1.0, 0.0
    w1, b1, w2, b2 = 0.5, 0.0, 2.0, 0.0
    eta = 0.1
    sig = lambda v: 1.0 / (1.0 + _m.exp(-v))

    z1 = w1 * x + b1
    h = sig(z1)
    z2 = w2 * h + b2
    yhat = sig(z2)
    L = 0.5 * (yhat - t) ** 2

    d2 = (yhat - t) * yhat * (1.0 - yhat)          # dL/dz2
    dL_dw2, dL_db2 = d2 * h, d2
    dL_dh = d2 * w2
    dh_dz1 = h * (1.0 - h)
    d1 = dL_dh * dh_dz1                             # dL/dz1
    dL_dw1, dL_db1 = d1 * x, d1

    tw1 = torch.tensor(w1, requires_grad=True, dtype=torch.float64)
    tw2 = torch.tensor(w2, requires_grad=True, dtype=torch.float64)
    tb1 = torch.tensor(b1, requires_grad=True, dtype=torch.float64)
    tb2 = torch.tensor(b2, requires_grad=True, dtype=torch.float64)
    th = torch.sigmoid(tw1 * x + tb1)
    ty = torch.sigmoid(tw2 * th + tb2)
    (0.5 * (ty - t) ** 2).backward()

    OUT["backprop_sigmoid"] = {
        "inputs": {"x": x, "t": t, "w1": w1, "b1": b1, "w2": w2, "b2": b2,
                   "eta": eta, "hidden": "sigmoid", "output": "sigmoid",
                   "loss": "0.5 (yhat - t)^2"},
        "forward": {"z1": z1, "h": h, "z2": z2, "yhat": yhat, "loss": L},
        "backward": {"delta2": d2, "dL_dw2": dL_dw2, "dL_db2": dL_db2,
                     "dL_dh": dL_dh, "dh_dz1": dh_dz1, "delta1": d1,
                     "dL_dw1": dL_dw1, "dL_db1": dL_db1},
        "update": {"w2_new": w2 - eta * dL_dw2, "w1_new": w1 - eta * dL_dw1},
        "autograd_check": {"dL_dw1": float(tw1.grad), "dL_dw2": float(tw2.grad)},
        "agrees": bool(abs(float(tw1.grad) - dL_dw1) < 1e-12
                       and abs(float(tw2.grad) - dL_dw2) < 1e-12),
        "saturation_note": "both factors yhat(1-yhat) and h(1-h) are at most 0.25; "
                           "that product is where the vanishing gradient starts",
    }


ALL_RUNS = (run_activations, run_relu_construction, run_xor, run_param_count,
            run_nn_vs_cheb, run_backprop_worked, run_backprop_sigmoid, run_epoch_tradeoff,
            run_regularization, run_worked_nn_growth, run_five_methods,
            run_icnn)


if __name__ == "__main__":
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
        print(f"  {fn.__name__:24s} {time.perf_counter() - t:7.1f}s", flush=True)

    path = "tools/figures/s07_numbers.json"
    if merge:
        base = json.load(open(path))
        base.update(OUT)
        base.setdefault("_meta", {})["last_partial_update"] = " ".join(wanted)
        OUT.clear(); OUT.update(base)
    else:
        OUT["_meta"] = {
            "calibration": "quarterly RBC (Quant_Macro Lab7), with Brock-Mirman "
                           "delta=1 where a closed form is needed",
            "alpha": ALPHA, "beta": BETA, "delta": DELTA,
            "rho": RHO, "sigma_eps": SIG_EPS,
            "bm_alpha": BM_ALPHA, "bm_beta": BM_BETA,
            "numpy": np.__version__, "torch": torch.__version__,
            "device": str(DEV),
            "rbc_model": "imported from tools/figures/s05_generate.py (class Proj "
                         "and euler_err_from_cpol), so 7.B1 and 5.B6 measure the "
                         "same model with the same metric",
            "optimizers": "NOT recomputed here -- 7.A3 reuses keys 'cg_steps', "
                          "'optimizer_memory', 'sgd_trajectory', 'sgd_dimension' "
                          "from tools/figures/s04_numbers.json",
            "generated_by": "tools/figures/s07_generate.py",
        }
    with open(path, "w") as f:
        json.dump(OUT, f, indent=2)
    print(f"wrote {path}  ({len(OUT)} keys)")
