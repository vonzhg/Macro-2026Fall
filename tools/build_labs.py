#!/usr/bin/env python3
"""Generate the Session 3 lab notebooks.

Writing .ipynb by hand is error-prone, so the notebooks are generated from this
file and executed through tools/run_code.slurm.  Edit here, not in the .ipynb.

    python3 tools/build_labs.py            # write labs/L00_*.ipynb and labs/L03_*.ipynb
"""

import json
import os

LABS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labs")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


# =============================================================== L00
L00 = [
md("""
# Lab 0 — Getting Started

**ECON 282E · Foundations of Macroeconomics · UC Riverside, Fall 2026**

Do this **before Session 3 (October 8)**. It installs the tools, checks that they work, and
writes your first two programs. Budget 45 minutes, most of it downloading.

You need nothing from Sessions 1–2 to do it.
"""),

md("""
## Part 0 — Install the tools

One-time steps, in a **Terminal** (macOS/Linux) or **Anaconda Prompt** (Windows).

### Step 1 — An editor
Install [VS Code](https://code.visualstudio.com/), then from its Extensions panel install
**Python** and **Jupyter**, both by Microsoft.

### Step 2 — Python, via Miniconda
Download and run the installer for your platform from
<https://docs.conda.io/en/latest/miniconda.html>.

On **Windows**, tick *"Add Miniconda3 to my PATH"*. Almost every `command not found`
later traces back to that box.

### Step 3 — Create the course environment

```bash
conda create -n econ282e python=3.11 -y
conda activate econ282e
pip install numpy scipy pandas matplotlib jupyter torch tqdm statsmodels
```

`torch` here is the CPU build. **No GPU is needed for this course** until Session 7, and
even then a laptop is enough for the labs.

### Step 4 — Open this notebook and pick the kernel
In VS Code: **File → Open Folder** → the course `labs` folder → open this notebook →
click **Select Kernel** (top right) → **Python Environments** → `econ282e`.

Then run the next cell with **Shift+Enter**.
"""),

code("""
# Run me first: does this environment have what the course needs?
import sys, importlib

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)

ready = True
for label, module in [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                      ("matplotlib", "matplotlib"), ("torch", "torch")]:
    try:
        m = importlib.import_module(module)
        print(f"  OK   {label:12s} {getattr(m, '__version__', '')}")
    except Exception:
        print(f"  MISS {label:12s} (not installed)")
        ready = False

print("\\nSetup looks good." if ready else
      "\\nInstall the missing packages (Part 0, Step 3), then re-run this cell.")
"""),

md("""
`Interpreter:` above should contain `econ282e`. If it does not, you are running a
different Python from the one you installed the packages into — pick the kernel again.
That single mismatch is the most common problem in this course.
"""),

md("""
## Part 1 — Your first Python

Three things that will come up in every lab.
"""),

code("""
# 1. Variables have types, and you do not declare them.
alpha = 0.36          # float
periods = 200         # int
name = "Riverside"    # str
converged = False     # bool

print(type(alpha), type(periods))
print(f"alpha = {alpha}, that is {alpha:.1%} of output")
"""),

code("""
# 2. Computers do not store the real numbers.
print(0.1 + 0.2)
print(0.1 + 0.2 == 0.3)

# So never test floats for equality. Test a distance against a tolerance:
print(abs((0.1 + 0.2) - 0.3) < 1e-12)
"""),

code("""
# 3. Assignment never copies.
a = [1, 2, 3]
b = a              # a second NAME for one list
b[0] = 99
print("a is now", a)

c = a.copy()       # an actual copy
c[0] = 1
print("a is still", a)
"""),

md("""
## Part 2 — Your first macro model

The Solow model, $k_{t+1} = s k_t^{\\alpha} + (1-\\delta)k_t$, with the steady state
$k^{*} = (s/\\delta)^{1/(1-\\alpha)}$. We simulate it and check the simulation against
the closed form — which is the habit the whole of Session 3 is about.
"""),

code("""
import numpy as np
import matplotlib.pyplot as plt

alpha, s, delta, T = 0.33, 0.20, 0.05, 200

k = 0.1
path = [k]
for t in range(T):
    k = s * k**alpha + (1 - delta) * k
    path.append(k)

k_star = (s / delta) ** (1 / (1 - alpha))

plt.figure(figsize=(8, 4))
plt.plot(path, lw=2, label="simulated $k_t$")
plt.axhline(k_star, color="firebrick", ls="--", label=f"closed form $k^*={k_star:.3f}$")
plt.xlabel("period"); plt.ylabel("capital per worker")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

print(f"simulated k_T = {path[-1]:.6f}")
print(f"closed form   = {k_star:.6f}")
print(f"difference    = {abs(path[-1] - k_star):.2e}")
"""),

md("""
**Exercise 1.** Raise the saving rate from `0.20` to `0.30` and re-run. Does $k^*$ move in
the direction you expected? By how much, in percent?

**Exercise 2.** Set `T = 20` instead of `200`. The simulation no longer reaches $k^*$.
How would you *detect* that automatically, without looking at the plot?

(The answer to Exercise 2 is a convergence criterion, and Session 3 spends a long time on
exactly how much it does and does not tell you.)
"""),

code("""
# Exercise 2, one answer: stop when successive values stop moving, and report BOTH
# the criterion and the true error -- because they are not the same number.
k, tol = 0.1, 1e-8
for t in range(10_000):
    k_next = s * k**alpha + (1 - delta) * k
    step = abs(k_next - k)
    k = k_next
    if step < tol:
        break

print(f"stopped after {t+1} periods")
print(f"last step        = {step:.3e}")
print(f"true error       = {abs(k - k_star):.3e}")
"""),

md("""
Notice that the last step and the true error are **not** the same size. Session 3 explains
exactly what the relationship between them is, and why assuming they are equal is the most
common way to report a wrong number.

## You are ready

Bring a working environment to Session 3. The next lab, **L03**, solves the optimal growth
model end to end.
"""),
]


# =============================================================== L03
L03 = [
md("""
# Lab 3 — Python, NumPy, and the Growth Model End to End

**ECON 282E · Session 3 · October 8, 2026**

This lab is Session 3, run by you. By the end you will have solved the optimal growth model
two different ways, graded both against a known answer, and measured what every design
choice cost.

The model throughout is the one from the lectures:

$$v(k,z)=\\max_{k'}\\{\\ln(zk^{\\alpha}-k') + \\beta\\,\\mathbb{E}[v(k',z')]\\},\\qquad \\delta=1,$$

which has the **Brock–Mirman closed form**

$$k'(k,z)=\\alpha\\beta z k^{\\alpha},\\qquad c(k,z)=(1-\\alpha\\beta)zk^{\\alpha}.$$

Full depreciation is not realistic. It is here because we know the answer exactly, and a
method that cannot reproduce a known answer should not be trusted on an unknown one.

**Prerequisite:** L00. If its check cell did not pass, fix that first.
"""),

code("""
import time
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)
print("numpy", np.__version__)
"""),

md("""
---
## Part 1 — Python essentials, in five cells

Only the pieces the rest of the lab needs.
"""),

code("""
# Containers. A dict is how you carry a calibration around.
par = {"alpha": 0.36, "beta": 0.95, "n": 200, "k_lo": 0.05, "k_hi": 0.60}
print(par["alpha"], len(par))

for key, value in par.items():
    print(f"  {key:6s} = {value}")
"""),

code("""
# Functions: default arguments put the calibration in one visible place.
def output(k, z=1.0, alpha=0.36):
    \"\"\"y = z * k**alpha.\"\"\"
    return z * k**alpha

print(output(2.0))
print(output(2.0, alpha=0.40))   # name the argument you are changing
"""),

code("""
# Returning several things at once, and unpacking them.
def summarise(x):
    return x.min(), x.max(), x.mean()

lo, hi, avg = summarise(np.array([1.0, 2.0, 3.0]))
print(lo, hi, avg)
"""),

code("""
# The trap from L00, now in NumPy: a slice is a VIEW, not a copy.
V = np.zeros(5)
part = V[1:3]
part[:] = 7.0
print("V =", V)          # V changed

part = V[1:3].copy()
part[:] = 0.0
print("V =", V)          # V did not change
"""),

md("""
**Why this matters here.** A value-function loop compares the new guess with the old one.
If `V_new` is a *view* of `V`, the comparison is between an array and itself, the distance
is always zero, and the loop stops after one iteration with a completely wrong answer — and
does not raise an error.
"""),

md("""
---
## Part 2 — NumPy and the one idea that makes this fast

The Bellman update asks for *every state against every choice*. Written as a double Python
loop that is unusably slow; written as one array expression it is the same arithmetic.
"""),

code("""
alpha, beta = 0.36, 0.95
N = 500
k = np.linspace(0.05, 0.60, N)
y = k**alpha
V = np.zeros(N)

# --- the loop you would say out loud ---------------------------------------
t0 = time.perf_counter()
Vn_loop = np.empty(N)
for i in range(N):
    best = -1e18
    for j in range(N):
        c = y[i] - k[j]
        if c > 0:
            val = np.log(c) + beta * V[j]
            if val > best:
                best = val
    Vn_loop[i] = best
t_loop = time.perf_counter() - t0

# --- the same thing, broadcast ---------------------------------------------
t0 = time.perf_counter()
C = y[:, None] - k[None, :]                 # (N, N): C[i, j] = y_i - k_j
U = np.full_like(C, -1e18)
np.log(C, out=U, where=C > 0)
Vn_vec = (U + beta * V[None, :]).max(axis=1)
t_vec = time.perf_counter() - t0

print(f"loop      {t_loop:8.4f} s")
print(f"broadcast {t_vec:8.4f} s")
print(f"speed-up  {t_loop/t_vec:8.1f}x")
print(f"max |difference| = {np.max(np.abs(Vn_loop - Vn_vec)):.1e}")
"""),

md("""
**Broadcasting, in one rule.** Reading shapes from the right, two arrays are compatible if
each pair of dimensions is equal or one of them is 1; the size-1 dimension is stretched,
without ever being stored.

```
k[:, None]   ->  (N, 1)     rows    = today's state
k[None, :]   ->  (1, N)     columns = tomorrow's choice
difference   ->  (N, N)
```

**Exercise 3.** Broadcasting almost never raises an error — a wrong axis gives you a
plausible array of the wrong shape. Change `axis=1` to `axis=0` in the cell above and look
at what comes out. What economic object did you just compute, if any?
"""),

md("""
---
## Part 3 — The model as a class

State (parameters, grid, payoff matrix) and behaviour (one Bellman step, solve, the exact
answer) belong together.
"""),

code("""
class GrowthModel:
    \"\"\"u(c) = ln c,  y = z k^alpha,  delta = 1.

    With z omitted the model is deterministic; pass z and Pi for the Markov version.
    \"\"\"

    def __init__(self, alpha=0.36, beta=0.95, n=200, k_lo=0.05, k_hi=0.60,
                 z=None, Pi=None):
        self.alpha, self.beta = alpha, beta
        self.k = np.linspace(k_lo, k_hi, n)
        self.z = np.array([1.0]) if z is None else np.asarray(z)
        self.Pi = np.array([[1.0]]) if Pi is None else np.asarray(Pi)

        y = self.z[None, :] * (self.k**alpha)[:, None]       # (n, nz)
        C = y[:, :, None] - self.k[None, None, :]            # (n, nz, n)
        self.U = np.full_like(C, -1e18)
        np.log(C, out=self.U, where=C > 0)

    # --- one Bellman step --------------------------------------------------
    def bellman(self, V):
        EV = V @ self.Pi.T                       # EV[l, j] = E[V(k_l, z') | z_j]
        M = self.U + self.beta * EV.T[None, :, :]
        return M.max(axis=2), M.argmax(axis=2)

    def solve(self, tol=1e-6, max_iter=20_000):
        V = np.zeros((self.k.size, self.z.size))
        for n in range(1, max_iter + 1):
            Vn, pol = self.bellman(V)
            d = np.max(np.abs(Vn - V))
            V = Vn
            if d < tol:
                return V, self.k[pol], n
        raise RuntimeError("did not converge")

    # --- the truth ---------------------------------------------------------
    def exact_policy(self):
        return self.alpha * self.beta * self.z[None, :] * self.k[:, None]**self.alpha

    def exact_value(self):
        B = self.alpha / (1 - self.alpha * self.beta)
        b = (np.log(1 - self.alpha * self.beta)
             + (1 + self.beta * B) * np.log(self.z)
             + self.beta * B * np.log(self.alpha * self.beta))
        A = np.linalg.solve(np.eye(self.z.size) - self.beta * self.Pi, b)
        return A[None, :] + B * np.log(self.k)[:, None]
"""),

code("""
m = GrowthModel()
t0 = time.perf_counter()
V, g, n = m.solve()
elapsed = time.perf_counter() - t0

pol_err = np.max(np.abs(g / m.exact_policy() - 1))
val_err = np.max(np.abs(V - m.exact_value()))

print(f"iterations           {n}")
print(f"wall time            {elapsed:.3f} s")
print(f"max value error      {val_err:.3e}")
print(f"max rel policy error {pol_err:.3e}")
"""),

md("""
You should get **272 iterations** and a policy that is wrong by about **1%**.

That 1% is not noise and it will not go away with more iterations. The next cell shows why.
"""),

code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(m.k, g[:, 0], color="firebrick", lw=1.5, label="VFI on the grid")
ax[0].plot(m.k, m.exact_policy()[:, 0], "k--", lw=1.2, label=r"$\\alpha\\beta k^{\\alpha}$")
ax[0].set_xlabel("$k$"); ax[0].set_ylabel("$k'$"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title("the whole grid: they agree")

w = slice(60, 96)
ax[1].step(m.k[w], g[w, 0], where="post", color="firebrick", lw=1.5, label="VFI on the grid")
ax[1].plot(m.k[w], m.exact_policy()[w, 0], "k--", lw=1.2, label=r"$\\alpha\\beta k^{\\alpha}$")
ax[1].set_xlabel("$k$"); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].set_title("36 consecutive grid points: a staircase")

plt.tight_layout(); plt.show()

print("distinct policy values across those 36 states:", len(np.unique(g[w, 0])))
"""),

md("""
`argmax` can only return a **grid point**, so the policy is quantised. The error is a
sawtooth of amplitude about half a grid spacing — a property of *how we made the choice*,
not of how long we iterated.

**Exercise 4.** Rerun with `n=800`. Does the policy error fall by a factor of 4, or 2?
Which of the two does the theory predict, and why?
"""),

md("""
---
## Part 4 — How wrong are you, really?

Three diagnostics. The second one is the one people get wrong.
"""),

code("""
# (a) The iteration count belongs to beta, not to the grid.
print(f"{'beta':>6} {'ln(eps)/ln(beta)':>18} {'measured':>10} {'beta/(1-beta)':>15}")
for b in (0.90, 0.95, 0.99):
    _, _, n_b = GrowthModel(beta=b).solve()
    print(f"{b:6.2f} {np.log(1e-6)/np.log(b):18.1f} {n_b:10d} {b/(1-b):15.1f}")
"""),

code("""
# (b) The stopping rule is NOT the error.
#     Iterate to machine precision first, to get the discrete fixed point V^h.
m = GrowthModel()
Vh, _, _ = m.solve(tol=1e-14)

V = np.zeros((m.k.size, m.z.size))
rows = []
for n in range(1, 400):
    Vn, _ = m.bellman(V)
    step = np.max(np.abs(Vn - V))
    V = Vn
    rows.append((n, step, np.max(np.abs(V - Vh))))
    if step < 1e-6:
        break

n, step, true_err = rows[-1]
print(f"stopped at iteration {n}")
print(f"  last step  ||V_n - V_(n-1)||  = {step:.3e}")
print(f"  bound  beta/(1-beta) * step   = {m.beta/(1-m.beta)*step:.3e}")
print(f"  TRUE error ||V_n - V^h||      = {true_err:.3e}")
print(f"  ratio bound/truth             = {m.beta/(1-m.beta)*step/true_err:.6f}")
"""),

md("""
Ask for $10^{-6}$ and you get $1.8\\times10^{-5}$. At $\\beta=0.95$ the factor
$\\beta/(1-\\beta)$ is **19**, and here it is not a pessimistic bound — it is *attained*.
"""),

code("""
ns   = np.array([r[0] for r in rows])
step = np.array([r[1] for r in rows])
err  = np.array([r[2] for r in rows])

plt.figure(figsize=(7, 4))
plt.plot(ns, np.log10(err), color="firebrick", lw=2, label=r"true error $\\|V_n-V^h\\|$")
plt.plot(ns, np.log10(step), color="darkorange", ls="--", lw=2,
         label=r"what you monitor $\\|V_n-V_{n-1}\\|$")
plt.axhline(-6, color="grey", ls=":", label=r"$\\varepsilon=10^{-6}$")
plt.xlabel("iteration $n$"); plt.ylabel(r"$\\log_{10}$ error")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

print("vertical gap between the lines:", np.mean(np.log10(err) - np.log10(step)).round(4))
print("log10(19) =", round(float(np.log10(19)), 4))
"""),

code("""
# (c) Error budgets compose: refining the grid stops helping once the
#     iteration tolerance puts a floor underneath it.
print(f"{'N':>6} {'h':>10} {'value err (1e-6)':>18} {'value err (1e-11)':>19} {'policy err':>12}")
for N in (50, 100, 200, 400, 800):
    loose = GrowthModel(n=N); Vl, gl, _ = loose.solve(tol=1e-6)
    tight = GrowthModel(n=N); Vt, gt, _ = tight.solve(tol=1e-11)
    h = loose.k[1] - loose.k[0]
    print(f"{N:6d} {h:10.2e} "
          f"{np.max(np.abs(Vl-loose.exact_value())):18.3e} "
          f"{np.max(np.abs(Vt-tight.exact_value())):19.3e} "
          f"{np.max(np.abs(gl-loose.exact_policy())):12.3e}")

print(f"\\ntruncation floor beta/(1-beta) * 1e-6 = {0.95/0.05*1e-6:.2e}")
"""),

md("""
Read the last two columns. At `tol=1e-6` the value error stalls around $1.8\\times10^{-5}$
— the floor — while at `tol=1e-11` it keeps falling. **Refining a grid while the tolerance
is loose buys nothing**, and you would never see that without running both.
"""),

md("""
---
## Part 5 — The other road: time iteration

Instead of maximising the value, solve the Euler equation for the policy, one scalar
root-find per state. `brentq` is a black box today — but a licensed one: the residual is
increasing on one side and decreasing on the other, so the root exists and is unique.
"""),

code("""
z = np.array([0.95, 1.05])
Pi = np.array([[0.90, 0.10], [0.10, 0.90]])
ms = GrowthModel(z=z, Pi=Pi)
alpha, beta, k = ms.alpha, ms.beta, ms.k
nz = z.size


def time_iteration(model, tol=1e-8, max_iter=1000):
    k, z, Pi = model.k, model.z, model.Pi
    alpha, beta = model.alpha, model.beta
    y = z[None, :] * (k**alpha)[:, None]
    c = 0.5 * y                                   # guess: save half
    n_roots = 0

    for n in range(1, max_iter + 1):
        c_new = np.empty_like(c)
        for j in range(z.size):
            for i in range(k.size):
                yi = y[i, j]

                def resid(cc, yi=yi, j=j, c=c):
                    kp = yi - cc
                    rhs = sum(Pi[j, mm] * alpha * z[mm] * kp**(alpha - 1)
                              / np.interp(kp, k, c[:, mm]) for mm in range(z.size))
                    return 1.0 / cc - beta * rhs

                c_new[i, j] = brentq(resid, 1e-10, yi - 1e-10, xtol=1e-14, rtol=1e-14)
                n_roots += 1
        d = np.max(np.abs(c_new - c))
        c = c_new
        if d < tol:
            return c, n, n_roots
    raise RuntimeError("did not converge")


t0 = time.perf_counter(); c_ti, n_ti, n_roots = time_iteration(ms); t_ti = time.perf_counter() - t0
t0 = time.perf_counter(); Vs, g_vfi, n_vfi = ms.solve();             t_vfi = time.perf_counter() - t0

c_exact = (1 - alpha * beta) * z[None, :] * k[:, None]**alpha
c_vfi = z[None, :] * (k**alpha)[:, None] - g_vfi

print(f"{'':22s}{'VFI':>12}{'time iteration':>18}")
print(f"{'iterations':22s}{n_vfi:12d}{n_ti:18d}")
print(f"{'wall time (s)':22s}{t_vfi:12.3f}{t_ti:18.3f}")
print(f"{'root-finds':22s}{0:12d}{n_roots:18d}")
print(f"{'max rel policy error':22s}"
      f"{np.max(np.abs(c_vfi/c_exact - 1)):12.2e}{np.max(np.abs(c_ti/c_exact - 1)):18.2e}")
"""),

md("""
Sixteen times fewer iterations, and about a **thousand times** more accurate — at the cost of
6,800 root-finds in pure Python, which is why the wall time goes the other way. Session 4
makes that inner loop cheap.
"""),

md("""
---
## Part 6 — Grade both, on points neither one solved

The **unit-free Euler residual**: at a capital level *off* the solution grid, compute the
consumption the Euler equation asks for and compare it with the one the method reports.

$$\\mathcal{E}(k,z)=\\left|1-\\tilde c(k,z)/c(k,z)\\right|.$$

$\\log_{10}\\mathcal{E}=-3$ is a one-dollar mistake per \\$1,000 of consumption.
"""),

code("""
def euler_errors(model, cpol, k_test):
    k, z, Pi = model.k, model.z, model.Pi
    alpha, beta = model.alpha, model.beta
    E = np.zeros((k_test.size, z.size))
    for j in range(z.size):
        for i, kk in enumerate(k_test):
            cc = np.interp(kk, k, cpol[:, j])
            kp = z[j] * kk**alpha - cc
            rhs = sum(Pi[j, mm] * alpha * z[mm] * kp**(alpha - 1)
                      / np.interp(kp, k, cpol[:, mm]) for mm in range(z.size))
            E[i, j] = abs(1.0 - (1.0 / (beta * rhs)) / cc)
    return E


k_test = np.linspace(0.06, 0.59, 997)        # deliberately between grid points
E_vfi = euler_errors(ms, c_vfi, k_test)
E_ti = euler_errors(ms, c_ti, k_test)

plt.figure(figsize=(8, 4))
plt.plot(k_test, np.log10(E_vfi[:, 0]), lw=0.9, color="firebrick", label="VFI on a grid")
plt.plot(k_test, np.log10(E_ti[:, 0]), lw=0.9, color="darkorange", label="time iteration")
plt.axhline(-3, color="grey", ls="--", label='"acceptable" $=10^{-3}$')
plt.xlabel("$k$ (off the solution grid)"); plt.ylabel(r"$\\log_{10}\\mathcal{E}$")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

for name, E in [("VFI", E_vfi), ("time iteration", E_ti)]:
    print(f"{name:16s} mean log10 E = {np.log10(E.mean()):6.2f}   "
          f"worst = {np.log10(E.max()):6.2f}")
"""),

md("""
**Exercise 5.** Evaluate the residuals *on* the solution grid instead
(`k_test = ms.k[1:-1]`). The numbers get better. Explain why that makes them useless as a
measure of accuracy.
"""),

md("""
---
## Part 7 — Moments, and the filter that makes them comparable

Block B\'s last deck covers the data side of this: acquire, clean, persist, export. The one
step you need for **Homework 1 Part B** is the filter, so here it is on a series we generate
ourselves — no internet required, and no dependency beyond NumPy.
"""),

code("""
def hp_filter(y, lamb=1600.0):
    \"\"\"Hodrick-Prescott filter, written out in four lines of linear algebra.

    Minimises  sum_t (y_t - tau_t)^2 + lamb * sum_t (tau_{t+1} - 2 tau_t + tau_{t-1})^2,
    whose solution is  tau = (I + lamb * K'K)^{-1} y, with K the second-difference
    matrix.  Returns (cycle, trend), in statsmodels' order.
    \"\"\"
    y = np.asarray(y, dtype=float)
    T = y.size
    K = np.zeros((T - 2, T))
    for i in range(T - 2):
        K[i, i:i+3] = [1.0, -2.0, 1.0]
    trend = np.linalg.solve(np.eye(T) + lamb * K.T @ K, y)
    return y - trend, trend


# A simulated log-output series: trend growth + an AR(1) cycle.
rng = np.random.default_rng(0)
T = 240                                   # 60 years, quarterly
eps = rng.normal(0, 0.007, T)
cyc = np.zeros(T)
for t in range(1, T):
    cyc[t] = 0.95 * cyc[t-1] + eps[t]
log_y = np.log(100.0) + 0.005 * np.arange(T) + cyc

cycle, trend = hp_filter(log_y, lamb=1600)   # 1600 = the quarterly convention

print(f"sd of the cyclical component   {cycle.std():.4f}")
print(f"first-order autocorrelation    {np.corrcoef(cycle[1:], cycle[:-1])[0,1]:.4f}")
print(f"(the AR(1) we actually drew had rho = 0.95, sd = {cyc.std():.4f})")

# statsmodels ships the same filter; cross-check ours when it is installed.
try:
    from statsmodels.tsa.filters.hp_filter import hpfilter
    _, t_sm = hpfilter(log_y, lamb=1600)
    print(f"\\nmax |ours - statsmodels| = {np.max(np.abs(trend - t_sm)):.2e}")
except ImportError:
    print("\\n(statsmodels not installed - skipping the cross-check)")
"""),

md("""
Two things to notice, both of which matter for the homework.

1. The filter does **not** return the process you drew. It returns whatever is left after a
   smooth trend is removed, and at `lamb=1600` some of a persistent cycle goes into the trend.
2. So the only fair comparison is **like with like**: filter the model\'s simulated series
   exactly as you filtered the data. That is the whole point of fixing `lamb` by convention.
"""),

code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(log_y, lw=1.0, label="log output", color="grey")
ax[0].plot(trend, lw=2.0, label="HP trend", color="firebrick")
ax[0].set_xlabel("quarter"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(cycle, lw=1.0, color="darkorange", label="HP cycle")
ax[1].axhline(0, ls="--", color="grey")
ax[1].set_xlabel("quarter"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),

md("""
**Exercise 6.** Re-run with `lamb=100` and `lamb=129600`. Which one leaves almost everything
in the cycle, and which one leaves almost nothing? Now explain why the convention has to be
fixed *before* you compare a model to data.
"""),

md("""
---
## Part 8 — A first taste of PyTorch

Everything from Session 4 onwards needs derivatives. Here is why we will not compute them
by hand.
"""),

code("""
import torch

k0, a = 2.0, 0.36
exact = a * k0**(a - 1)

print(f"{'h':>10}{'forward difference':>22}{'relative error':>18}")
for e in range(1, 17):
    h = 10.0**(-e)
    fd = ((k0 + h)**a - k0**a) / h
    print(f"{h:10.0e}{fd:22.15f}{abs(fd/exact - 1):18.2e}")

x = torch.tensor(k0, requires_grad=True, dtype=torch.float64)
y = x**a
y.backward()
print(f"\\nautograd    {float(x.grad):22.15f}{abs(float(x.grad)/exact - 1):18.2e}")
print(f"exact       {exact:22.15f}")
"""),

md("""
Finite differences bottom out near $\\sqrt{\\varepsilon_{\\text{mach}}}\\approx1.5\\times10^{-8}$
and then get **worse**: at $h=10^{-16}$ the numerator underflows to zero and the answer is
100% wrong. Automatic differentiation has no such floor — it applies the chain rule,
exactly.

---
## What to take away

1. Twenty lines take you from the Bellman equation to a policy. Knowing **how wrong** it is
   takes the rest of the lab.
2. The iteration count belongs to $\\beta$; the accuracy belongs to the grid.
3. The quantity you monitor is a factor $\\beta/(1-\\beta)$ below the error you care about.
4. Error budgets compose — refining one past the others buys nothing.
5. Two methods resting on different theorems, agreeing to four digits, is the strongest
   evidence available when there is no closed form. Here there is one, so use it.
6. A model moment and a data moment are comparable only if they were filtered the same way.

**Homework 1, Part A** starts from this lab.
"""),
]



# =============================================================== L04
L04 = [
md("""
# Lab 4 — Numerical Methods, and a Fast RBC Model

**ECON 282E · Session 4 · October 15, 2026**

Session 3 made six crude choices and measured what each one cost. This lab builds the
replacement for each, and measures what it bought.

| You build | It repairs |
|---|---|
| bisection, Newton, secant, Brent | `brentq` as a sealed black box |
| golden section | the grid search for the maximum |
| shape-preserving interpolation | linear interpolation only |
| Gauss--Hermite quadrature | the expectation as a crude sum |
| Tauchen and Rouwenhorst | the transition matrix written by hand |
| Howard and EGM | 1,352 sweeps and 2.4 billion comparisons |

**Prerequisite:** L03. Everything here is NumPy and SciPy; nothing needs the internet.
"""),

code("""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.stats import norm

np.set_printoptions(precision=6, suppress=True)
"""),

md("""
---
## Part 1 — Root-finding, and how methods fail

A market-clearing condition: $Q^d(p) = 100p^{-1/2}$ against $Q^s(p) = e^{p/2}-1$.
"""),

code("""
demand   = lambda p: 100.0 * p**-0.5
supply   = lambda p: np.exp(0.5*p) - 1.0
excess   = lambda p: demand(p) - supply(p)
d_excess = lambda p: -50.0*p**-1.5 - 0.5*np.exp(0.5*p)

p_star = brentq(excess, 0.1, 20.0, xtol=1e-15)
print(f"p* = {p_star:.10f},  residual = {excess(p_star):.2e}")
"""),

code("""
def bisection(f, a, b, tol=1e-12, maxit=200):
    hist = []
    for n in range(maxit):
        m = 0.5*(a+b)
        if f(a)*f(m) <= 0: b = m
        else:              a = m
        hist.append(0.5*(a+b))
        if b-a < tol: break
    return np.array(hist)

def newton(f, fp, x0, tol=1e-12, maxit=200):
    hist, x = [], x0
    for n in range(maxit):
        x = x - f(x)/fp(x)
        hist.append(x)
        if abs(f(x)) < tol: break
    return np.array(hist)

def secant(f, x0, x1, tol=1e-12, maxit=200):
    hist = []
    for n in range(maxit):
        f0, f1 = f(x0), f(x1)
        if f1 == f0: break
        x0, x1 = x1, x1 - f1*(x1-x0)/(f1-f0)
        hist.append(x1)
        if abs(f(x1)) < tol: break
    return np.array(hist)

hists = {"bisection": bisection(excess, 0.1, 20.0),
         "newton":    newton(excess, d_excess, 10.0),
         "secant":    secant(excess, 6.0, 9.0)}

for name, h in hists.items():
    e = np.abs(h - p_star)
    e = e[e > 1e-14]
    if name == "bisection":
        # the bisection error is not monotone, so the 3-point order formula is
        # meaningless here.  Fit the LINEAR rate instead: e_n ~ C * r^n.
        r = np.exp(np.polyfit(np.arange(e.size), np.log(e), 1)[0])
        print(f"{name:10s} {len(h):3d} iterations, error ratio per step {r:.3f} (theory 0.5)")
    else:
        order = np.log(e[-1]/e[-2]) / np.log(e[-2]/e[-3])
        print(f"{name:10s} {len(h):3d} iterations, observed order {order:.2f}")
"""),

code("""
plt.figure(figsize=(7, 4))
for name, h in hists.items():
    plt.plot(np.arange(1, len(h)+1), np.log10(np.maximum(np.abs(h-p_star), 1e-16)),
             marker='o', ms=2.5, label=name)
plt.xlabel("iteration"); plt.ylabel(r"$\log_{10}|x_n - x^*|$")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
"""),

md("""
The three slopes are the three convergence orders: linear, superlinear (golden ratio) and
quadratic. Read them off the plot and check them against the printed numbers.

**Exercise 1.** Now a demand/supply pair with **no** equilibrium:
$Q^d = 100-2p+10\sqrt p$, $Q^s = -20+3p-0.1p^2$. Run all three methods on it. Which ones
tell you something is wrong, and which ones return a number? What is the residual there?
"""),

code("""
bad = lambda p: (100 - 2*p + 10*np.sqrt(np.maximum(p, 1e-12))) - (-20 + 3*p - 0.1*p**2)
pg = np.linspace(0.01, 200, 5000)
print(f"minimum excess demand over the grid: {bad(pg).min():.2f} at p = {pg[bad(pg).argmin()]:.2f}")
try:
    brentq(bad, 1, 50)
except ValueError as e:
    print("brentq refuses:", e)
h = newton(bad, lambda p: (bad(p+1e-7)-bad(p-1e-7))/2e-7, 20.0, maxit=50)
print(f"Newton returns p = {h[-1]:.2f}, where excess demand is {bad(h[-1]):.1f}")
"""),

md("""
---
## Part 2 — Optimization on an interval

Golden section: shrink a bracket by the ratio that lets you recycle one function value.
"""),

code("""
PHI = (np.sqrt(5.0) - 1.0) / 2.0

def golden_section(f, a, b, tol=1e-8):
    n = 0
    c, d = b - PHI*(b-a), a + PHI*(b-a)
    fc, fd = f(c), f(d); n += 2
    while b - a > tol:
        if fc > fd: b, d, fd = d, c, fc; c = b - PHI*(b-a); fc = f(c)
        else:       a, c, fc = c, d, fd; d = a + PHI*(b-a); fd = f(d)
        n += 1
    return 0.5*(a+b), n

f = lambda x: -(x - 2.7)**2 + 3.0
for tol in (1e-4, 1e-6, 1e-8):
    x, n = golden_section(f, 0.0, 10.0, tol)
    print(f"tol {tol:.0e}: golden section {n:3d} evaluations   "
          f"grid search would need {int(np.ceil(10/tol)):,}")
"""),

md("""
Each extra decimal digit costs golden section **five more evaluations** and costs the grid a
factor of ten. And the cost does not depend on the state grid at all.

**Exercise 2.** Verify the ratio. Solve $\tau = (1-\tau)^2$ and check that $1-\tau$ is what
`PHI` holds. Why is that the unique ratio that permits recycling?
"""),

md("""
---
## Part 3 — Interpolation: accuracy is not the criterion

A policy with a binding constraint, $g(k) = \min(k, 4)$.
"""),

code("""
k_nodes = np.linspace(0, 10, 21)
g_nodes = np.minimum(k_nodes, 4.0)
kf = np.linspace(0, 10, 2000)
truth = np.minimum(kf, 4.0)

fits = {"linear":       np.interp(kf, k_nodes, g_nodes),
        "cubic spline": CubicSpline(k_nodes, g_nodes, bc_type='natural')(kf),
        "pchip":        PchipInterpolator(k_nodes, g_nodes)(kf)}

for name, y in fits.items():
    print(f"{name:13s} max error {np.max(np.abs(y-truth)):.4f}   "
          f"overshoot above the cap {max(y.max()-4.0, 0.0):.4f}   "
          f"monotone {bool(np.all(np.diff(y) >= -1e-12))}")
"""),

code("""
plt.figure(figsize=(7, 4))
plt.plot(kf, truth, 'k--', lw=1.2, label='truth: min(k, 4)')
for name, y in fits.items():
    plt.plot(kf, y, lw=1.2, label=name)
plt.axhline(4.0, color='grey', ls=':', lw=1)
plt.xlim(2.5, 7); plt.ylim(3.2, 4.3)
plt.xlabel("$k$"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
"""),

md("""
The cubic spline is the most accurate scheme on smooth data and it is the wrong one here: it
puts the policy **above a cap the model says cannot be exceeded**. Feed that back into a
Bellman loop and the inner maximization is no longer unimodal.

**Exercise 3.** Check the claim the lecture corrected: interpolate $\ln k$ on 12 nodes and
verify that the *linear* interpolant is concave (all second differences $\le 0$). Then check
the cubic spline on `min(k,4)` and show that it is not.
"""),

md("""
### Part 3b — Smolyak: the nodes you keep when $d$ gets large

One dimension at a time is easy. Two continuous states cost $n^2$ points, ten cost $n^{10}$.
Smolyak's sparse grid keeps only the index combinations whose *total* resolution is modest.
"""),

code("""
from itertools import product

def nodes_1d(i):
    \"\"\"G^i: Chebyshev extrema, m_1 = 1 and m_i = 2^(i-1)+1. Nested by construction.\"\"\"
    if i == 1:
        return np.array([0.0])
    m = 2**(i-1) + 1
    return -np.cos(np.pi*np.arange(m)/(m-1))

def index_sets(d, mu):
    \"\"\"All (i_1..i_d) with i_j >= 1 and sum i_j <= d + mu.\"\"\"
    if d == 1:
        for a in range(mu+1):
            yield (a+1,)
        return
    for a in range(mu+1):
        for rest in index_sets(d-1, mu-a):
            yield (a+1,) + rest

def smolyak_grid(d, mu):
    pts = set()
    for idx in index_sets(d, mu):
        for combo in product(*[nodes_1d(i) for i in idx]):
            pts.add(tuple(round(float(c), 12) for c in combo))
    return sorted(pts)

# nestedness is the property the whole construction rests on
for i in (1, 2, 3):
    a, b = set(np.round(nodes_1d(i), 10)), set(np.round(nodes_1d(i+1), 10))
    print(f"G^{i} ({len(a)} nodes) subset of G^{i+1} ({len(b)} nodes): {a <= b}")
"""),

code("""
print(f"{'d':>3} {'mu=1':>6} {'mu=2':>6} {'mu=3':>7} {'5^d':>12} {'9^d':>14}")
for d in (2, 3, 4, 5, 10, 12):
    row = [len(smolyak_grid(d, mu)) if (d <= 5 or mu <= 2) else None for mu in (1, 2, 3)]
    cells = "".join(f"{v:>7,}" if v is not None else f"{'--':>7}" for v in row)
    print(f"{d:>3}{cells} {5**d:>12,} {9**d:>14,}")

# Malin, Krueger & Kubler (2011) print this column; check against it
print()
for d, n in {2: 13, 4: 41, 5: 61, 12: 313}.items():
    print(f"  d={d:>2}, mu=2: ours {len(smolyak_grid(d,2)):>4}   "
          f"paper {n:>4}   {'match' if len(smolyak_grid(d,2))==n else 'MISMATCH'}")
"""),

code("""
sp = np.array(smolyak_grid(2, 2))
g5 = nodes_1d(3)
tensor = np.array([(a, b) for a in g5 for b in g5])

plt.figure(figsize=(4.4, 4.4))
plt.scatter(tensor[:,0], tensor[:,1], s=42, facecolors='none', edgecolors='grey',
            label=f'tensor 5x5 ({len(tensor)})')
plt.scatter(sp[:,0], sp[:,1], s=30, color='C3', label=f'Smolyak mu=2 ({len(sp)})')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
plt.xticks([-1,0,1]); plt.yticks([-1,0,1]); plt.gca().set_aspect('equal')
plt.tight_layout(); plt.show()
"""),

md("""
The sparse grid keeps the **axes and the corners** and throws away the interior cross-terms.
At $d=10$ that is 221 points against 9,765,625 — a factor of 44,000.

**Exercise 3b.** Interpolate $f(x,y)=\\exp(-(x^2+y^2))$ on the Smolyak $\\mu=3$ grid and on
a tensor grid with a comparable number of points, and compare maximum errors on 5,000 random
test points. Which wins, and does the answer change if you replace $f$ with something whose
cross-derivative is large?
"""),

md("""
---
## Part 4 — Quadrature

$\mathbb{E}[e^z]$ for $z \sim N(0,1)$, whose exact value is $e^{1/2}$.
"""),

code("""
exact = np.exp(0.5)
g = lambda z: np.exp(z)

print(f"{'n':>4}  {'trapezoid':>12}  {'Gauss-Hermite':>14}  {'Monte Carlo':>12}")
rng = np.random.default_rng(0)
for n in (3, 5, 9, 17):
    a, b = -8.0, 8.0
    x = np.linspace(a, b, n)
    w = np.full(n, (b-a)/(n-1)); w[0] *= 0.5; w[-1] *= 0.5
    trap = np.sum(w * g(x) * norm.pdf(x))
    xh, wh = np.polynomial.hermite_e.hermegauss(n)
    gh = np.sum(wh * g(xh)) / np.sqrt(2*np.pi)
    mc = g(rng.standard_normal(n)).mean()
    print(f"{n:4d}  {abs(trap-exact):12.2e}  {abs(gh-exact):14.2e}  {abs(mc-exact):12.2e}")
"""),

md("""
Gauss--Hermite knows the weight is a normal density, so it places its nodes where the
probability is. **It is exact at 17 nodes** — a million Monte Carlo draws are not.

**Exercise 4.** Add Simpson's rule to the table. It is a higher-order rule than the trapezoid
rule. Is it more accurate here? Explain what you find.
"""),

md("""
---
## Part 5 — Where $\Pi$ comes from

Two ways to turn $\ln z' = \rho \ln z + \sigma\varepsilon'$ into a finite Markov chain.
"""),

code("""
def tauchen(n, rho, sigma, m=3.0):
    sz = sigma / np.sqrt(1 - rho**2)
    y = np.linspace(-m*sz, m*sz, n)
    step = y[1] - y[0]
    P = np.empty((n, n))
    for i in range(n):
        P[i, 0]  = norm.cdf((y[0]  - rho*y[i] + step/2) / sigma)
        P[i, -1] = 1 - norm.cdf((y[-1] - rho*y[i] - step/2) / sigma)
        for j in range(1, n-1):
            P[i, j] = (norm.cdf((y[j] - rho*y[i] + step/2)/sigma)
                       - norm.cdf((y[j] - rho*y[i] - step/2)/sigma))
    return y, P

def rouwenhorst(n, rho, sigma):
    p = (1 + rho) / 2
    P = np.array([[p, 1-p], [1-p, p]])
    for k in range(3, n+1):
        Z = np.zeros((k, k))
        Z[:-1, :-1] += p*P;     Z[:-1, 1:] += (1-p)*P
        Z[1:,  :-1] += (1-p)*P; Z[1:,  1:] += p*P
        Z[1:-1, :] /= 2.0
        P = Z
    sz = sigma / np.sqrt(1 - rho**2)
    psi = sz * np.sqrt(n - 1)
    return np.linspace(-psi, psi, n), P

def chain_moments(y, P):
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi /= pi.sum()
    mu = pi @ y
    var = pi @ (y - mu)**2
    cov = sum(pi[a]*P[a, b]*(y[a]-mu)*(y[b]-mu) for a in range(len(y)) for b in range(len(y)))
    return np.sqrt(var), cov/var
"""),

code("""
sigma = 0.007
print(f"{'rho':>5} {'n':>3} | {'Tauchen rho':>12} {'sd err %':>9} | "
      f"{'Rouwen. rho':>12} {'sd err %':>9}")
for rho in (0.90, 0.95, 0.99):
    target_sd = sigma / np.sqrt(1 - rho**2)
    for n in (5, 9, 15):
        sdt, rt = chain_moments(*tauchen(n, rho, sigma))
        sdr, rr = chain_moments(*rouwenhorst(n, rho, sigma))
        print(f"{rho:5.2f} {n:3d} | {rt:12.4f} {100*abs(sdt/target_sd-1):9.1f} | "
              f"{rr:12.6f} {100*abs(sdr/target_sd-1):9.1e}")
"""),

md("""
Look at the $\rho = 0.99$, $n = 5$ row: the Tauchen chain reports a persistence of **1.0000**
— a unit root — and an unconditional standard deviation more than a third too small. At
$n=15$ it is still about a fifth too small. Rouwenhorst is exact to machine precision
everywhere, because it matches $\rho$ and $\sigma_z$ *by construction*.

**Use Rouwenhorst**, and report the chain's implied moments next to the targets. Homework 1
Part B asks for exactly this table.
"""),

md("""
---
## Part 6 — Putting it together: a fast RBC solve

Quarterly calibration: $\alpha=0.33$, $\beta=0.99$, $\delta=0.025$, $\rho=0.95$,
$\sigma=0.007$.
"""),

code("""
class RBC:
    def __init__(self, n_k=300, n_z=7, width=0.5):
        self.alpha, self.beta, self.delta = 0.33, 0.99, 0.025
        logz, self.Pi = rouwenhorst(n_z, 0.95, 0.007)
        self.z = np.exp(logz)
        self.kss = ((1/self.beta - 1 + self.delta)/self.alpha)**(1/(self.alpha-1))
        self.k = np.linspace((1-width)*self.kss, (1+width)*self.kss, n_k)
        self.n_k, self.n_z = n_k, n_z
        self.y = self.z[None,:]*self.k[:,None]**self.alpha + (1-self.delta)*self.k[:,None]
        C = self.y[:,:,None] - self.k[None,None,:]
        self.U = np.full_like(C, -1e10)
        np.log(C, out=self.U, where=C > 0)

m = RBC()                      # n_k = 300 here; the lecture used 500, so times differ
print(f"k_ss = {m.kss:.4f},  K/Y = {m.kss/m.kss**m.alpha:.2f} (quarterly)")
"""),

code("""
def vfi(model, n_howard=0, tol=1e-6, maxit=20000):
    V = np.zeros((model.n_k, model.n_z))
    ik, iz = np.meshgrid(np.arange(model.n_k), np.arange(model.n_z), indexing='ij')
    t0, outer = time.perf_counter(), 0
    while outer < maxit:
        EV = V @ model.Pi.T
        M = model.U + model.beta * EV.T[None, :, :]
        Vn, pol = M.max(axis=2), M.argmax(axis=2)
        d = np.max(np.abs(Vn - V)); V = Vn; outer += 1
        for _ in range(n_howard):                    # policy evaluation only
            EV = V @ model.Pi.T
            V = model.U[ik, iz, pol] + model.beta * EV[pol, iz]
        if d < tol: break
    return V, model.k[pol], outer, time.perf_counter() - t0

for n_h in (0, 10, 20, 50):
    _, g, it, el = vfi(m, n_howard=n_h)
    label = "plain VFI" if n_h == 0 else f"Howard n_H={n_h}"
    print(f"{label:16s} {it:5d} policy updates   {el:6.2f} s")
"""),

md("""
Howard's policy improvement replaces most of the maximizations with a lookup. The answer is
identical; only the route to it changes.

Now the endogenous grid method — **on cash on hand**, which is what removes the root-finding.
Putting the grid on $k$ instead reintroduces a solve at every node, and the method loses the
speed race it exists to win.
"""),

code("""
def egm(model, tol=1e-10, maxit=5000):
    kp = model.k.copy()                     # end-of-period assets
    mg = np.linspace(model.y.min(), model.y.max(), model.n_k)   # cash on hand
    c = 0.3 * mg[:, None] * np.ones((1, model.n_z))
    t0, it = time.perf_counter(), 0
    while it < maxit:
        mp = model.z[None,:]*kp[:,None]**model.alpha + (1-model.delta)*kp[:,None]
        R  = model.alpha*model.z[None,:]*kp[:,None]**(model.alpha-1) + (1-model.delta)
        cp = np.empty_like(mp)
        for l in range(model.n_z):
            cp[:, l] = np.interp(mp[:, l], mg, c[:, l])
        rhs   = model.beta * ((R/cp) @ model.Pi.T)
        c_end = 1.0/rhs                      # invert marginal utility
        m_end = c_end + kp[:, None]          # <- closed form: NO root-finding
        c_new = np.empty_like(c)
        for j in range(model.n_z):
            c_new[:, j] = np.interp(mg, m_end[:, j], c_end[:, j])
            tight = mg < m_end[0, j]
            c_new[tight, j] = mg[tight] - kp[0]
        d = np.max(np.abs(c_new - c)); c = c_new; it += 1
        if d < tol: break
    return mg, c, it, time.perf_counter() - t0

mg, cpol, it, el = egm(m)
print(f"EGM: {it} iterations, {el:.2f} s, 0 root solves")
"""),

md("""
**Exercise 5.** Compute off-grid Euler residuals for the plain VFI policy and for the EGM
policy and compare. (L03 Part 6 has the residual function; adapt it to this calibration.)
You should find four to five decades between them.

**Exercise 6.** Time the monotone variant: exploit $g(k,z)$ increasing to start each state's
search where the previous one stopped. Count comparisons *and* wall time. They do not move
in the same direction — explain why.
"""),

md("""
---
## Part 7 — The worked example of deck 4.B5, end to end

Everything above was one tool at a time. This part is the whole toolkit in one solver, and it
is the code behind **deck 4.B5**. Four substitutions relative to the crude Session 3 solver:

| 3.A4 | here |
|---|---|
| $\\Pi$ written by hand | **Rouwenhorst** (Part 5) |
| no interpolation | **linear**, because it keeps $\\hat v$ concave (Part 3) |
| grid search for the max | **golden section** (Part 2) |
| a full sweep every iteration | **Howard**, $n_H = 20$ (Part 6) |

Note the order of the argument, because it is the point of the deck: linear interpolation
keeps $\\hat v$ concave → the objective is unimodal → golden section is *licensed*. Swap in a
cubic spline and the first link breaks, so the third one does too.
"""),

code("""
def rhs(kp, yj, EVj, model, kg):
    \"\"\"Bellman right-hand side, evaluated OFF the grid.\"\"\"
    c = np.maximum(yj - kp, 1e-12)             # clamp acts as a feasibility penalty
    return np.log(c) + model.beta*np.interp(kp, kg, EVj)

def vfi_continuous(model, n_howard=20, tol=1e-6, maxit=20000, n_gs=40):
    kg = model.k
    V  = np.zeros((model.n_k, model.n_z)); Vn = np.empty_like(V)
    g  = np.empty_like(V)
    t0, outer, nev = time.perf_counter(), 0, 0
    while outer < maxit:
        EV = V @ model.Pi.T                    # the whole expectation, one matmul
        for j in range(model.n_z):
            yj, EVj = model.y[:, j], EV[:, j]
            a = np.full(model.n_k, kg[0])
            b = np.minimum(yj - 1e-8, kg[-1])
            c_, d_ = b - PHI*(b-a), a + PHI*(b-a)
            fc = rhs(c_, yj, EVj, model, kg); fd = rhs(d_, yj, EVj, model, kg)
            nev += 2*model.n_k
            for _ in range(n_gs):              # golden section, vectorised over all k
                L = fc > fd
                b = np.where(L, d_, b); a = np.where(L, a, c_)
                c_, d_ = b - PHI*(b-a), a + PHI*(b-a)
                fc = rhs(c_, yj, EVj, model, kg); fd = rhs(d_, yj, EVj, model, kg)
                nev += 2*model.n_k
            g[:, j]  = 0.5*(a + b)
            Vn[:, j] = rhs(g[:, j], yj, EVj, model, kg)
        d = np.max(np.abs(Vn - V))
        V[:] = Vn                              # copy, do NOT rebind: V is Vn otherwise
        outer += 1
        for _ in range(n_howard):              # Howard: same rhs at a frozen policy
            EV = V @ model.Pi.T
            for j in range(model.n_z):
                V[:, j] = rhs(g[:, j], model.y[:, j], EV[:, j], model, kg)
        if d < tol: break
    return V, g, outer, time.perf_counter() - t0, nev
"""),

md("""
`V = Vn` would **rebind**, not copy: the Howard loop would then write into the array the
convergence test reads, `d` would be identically zero, and the solver would stop after one
sweep with a wrong answer. `V[:] = Vn` is the fix. This is a bug worth writing once on
purpose so you recognise it later.
"""),

code("""
_, g_grid, it_g, t_g = vfi(m, n_howard=0)              # the crude Session 3 solver
_, g_cont, it_c, t_c, nev = vfi_continuous(m)

print(f"grid search : {it_g:5d} sweeps          {t_g:6.2f} s   "
      f"{m.n_k*m.n_k*m.n_z*it_g:>14,} comparisons")
print(f"golden sect.: {it_c:5d} policy updates  {t_c:6.2f} s   {nev:>14,} evaluations")
"""),

md("""
Two things to say out loud about that table. The **iteration count** fell because of Howard,
not because of golden section. And the **wall time went up** even though the inner work fell
by a large factor, because golden section runs in a Python loop while the grid search is one
array operation. Report both; a speed claim without a timing is not a result.
"""),

code("""
# Does the continuous policy actually behave? Three checks, in this order.
jm = m.n_z // 2
print("1. ordered in z at the middle of the grid:",
      np.all(np.diff(g_cont[m.n_k//2, :]) > 0))
print("2. increasing in k in every shock state: ",
      np.all(np.diff(g_cont, axis=0) > 0))
sav = g_cont[:, jm] - m.k
cross = m.k[np.argmin(np.abs(sav))]
print(f"3. middle-state 45-degree crossing at {cross:.4f}, analytic k* = {m.kss:.4f} "
      f"({abs(cross-m.kss)/(m.k[1]-m.k[0]):.1f} grid spacings)")
"""),

code("""
# The picture that makes the difference visible: plot SAVING, not the policy.
# g(k) against k is three overlapping straight lines; g(k) - k is not.
sav_g, sav_c = g_grid[:, jm] - m.k, g_cont[:, jm] - m.k
w = slice(120, 156)                                   # 36 consecutive grid points

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for j, lab in ((0, 'z low'), (jm, 'z middle'), (m.n_z-1, 'z high')):
    ax[0].plot(m.k, g_cont[:, j] - m.k, lw=1.3, label=lab)
ax[0].axhline(0, color='grey', ls='--', lw=0.8)
ax[0].axvline(m.kss, color='grey', ls=':', lw=0.8)
ax[0].set_xlabel("$k$"); ax[0].set_ylabel("saving $g(k,z)-k$")
ax[0].legend(frameon=False); ax[0].grid(alpha=0.3)

ax[1].plot(m.k[w], sav_g[w], 's-', ms=3, color='grey', lw=1, label='grid search')
ax[1].plot(m.k[w], sav_c[w], 'o-', ms=2.5, color='C3', lw=1.3, label='golden section')
ax[1].axhline(0, color='grey', ls='--', lw=0.8)
ax[1].set_xlabel("$k$ (36 consecutive grid points)"); ax[1].set_ylabel("saving")
ax[1].legend(frameon=False); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("distinct policy values in those 36 states: "
      f"grid search {len(np.unique(g_grid[w, jm]))}, "
      f"golden section {len(np.unique(g_cont[w, jm]))}")
"""),

md("""
The right-hand panel is the whole argument in one picture. Grid search can only choose
$k' = k + mh$, so in this window **saving takes two values**: one grid spacing, then zero. Its
zero-crossing lands about ten grid spacings before the analytic $k^*$. Golden section gives a
smooth declining line that crosses once, in the right place.

Plotted as $k'$ against $k$ instead, the two are indistinguishable — the differences are
$O(h)$ on an axis that spans 28 units. **Subtract the benchmark before you plot.**

**Exercise 7.** Compute off-grid Euler residuals for both policies. The lecture measured
$10^{-1.30}$ against $10^{-2.52}$ at $n_k=300$ — about 1.2 decades, i.e. the error falls by a
factor of $10^{1.2} \\approx 16$. Do you reproduce it? Then rerun at $n_k = 500$ and check
that *both* improve while the gap stays near 1.2 decades.

**Exercise 8.** Replace `np.interp` with `CubicSpline` inside `rhs` and re-run. The answer may
still look fine. Test the thing that actually broke: check whether $\\hat v$ is concave at
every iteration, and report how often it is not.
"""),

md("""
---
## What to take away

1. Convergence orders are real and measurable: 1, 1.618, 2. Bracketing methods are the ones
   that tell you when you have given them an impossible problem.
2. Golden section costs $O(\log(1/\varepsilon))$ and is **independent of the state grid**.
3. Accuracy order is not the criterion for interpolation. **Shape is.**
4. A quadrature rule that knows the weight function beats one that does not, by decades.
5. **Rouwenhorst, not Tauchen**, whenever $\rho$ is near one — which in macro is always.
6. The largest single gain in this lab came from **reformulating the state**, not from a
   better algorithm.
7. Interpolation is not a display choice inside a Bellman loop; it is a **hypothesis of the
   maximizer**. Linear keeps concavity, concavity gives unimodality, unimodality is what
   licenses golden section.
8. **Plot the deviation, not the level.** Differences between solvers are $O(h)$; a level plot
   on a wide axis hides them completely.
"""),
]


if __name__ == "__main__":
    os.makedirs(LABS, exist_ok=True)
    for name, cells in [("L00_Getting_Started", L00), ("L03_Python_and_VFI", L03),
                        ("L04_Numerical_Methods", L04)]:
        path = os.path.join(LABS, name + ".ipynb")
        with open(path, "w") as f:
            json.dump(notebook(cells), f, indent=1)
        print(f"wrote {path}  ({len(cells)} cells)")
