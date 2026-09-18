# Labs — ECON 282E, Foundations of Macroeconomics

| Lab | Session | What it does |
|---|---|---|
| `L00_Getting_Started.ipynb` | before S3 | Install Python, verify the environment, first program, the Solow model against its closed form. |
| `L03_Python_and_VFI.ipynb` | S3 (Oct 8) | Python and NumPy essentials; the `GrowthModel` class; VFI and time iteration, both graded against Brock–Mirman; convergence and accuracy diagnostics; the HP filter; a first look at autograd. |
| `L04_Numerical_Methods.ipynb` | S4 (Oct 15) | Root-finding with measured convergence orders; golden section; interpolation shape tests and **Smolyak sparse grids**; quadrature; **Tauchen against Rouwenhorst**; Howard and EGM on a quarterly RBC model; and the **4.B5 worked example end to end** — Rouwenhorst + interpolation + golden section + Howard, against the crude Session 3 solver. |
| `L05_Perturbation_and_Projection.ipynb` | S5 (Oct 22) | Log-linearization and the **QZ (generalized Schur) solve**, with Blanchard--Kahn as an eigenvalue count; the **Taylor principle** recovered from a New Keynesian model; second order by residual matching, the risk correction, and **pruning**; why monomials are unusable (the Hilbert matrix at condition $3\times10^{18}$); Chebyshev conditioning and convergence; a **Chebyshev collocation solve of the same quarterly RBC model**; and the closing table of one model solved four ways. |
| `L07_ML_and_Deep_Growth.ipynb` | S7 (Nov 5) | A sum of shifted ReLUs fitted to $\sin x$ by least squares, then with **the kinks trained too** — on a kinked target the learned knot lands on the constraint at $1.0000$; the five-line training loop; **backpropagation by hand, checked against `autograd` to machine precision**; the growth model written as a loss and graded against its **closed form**; the quarterly RBC graded on **off-sample Euler residuals** against Judd's scale; and an **input-convex network** against a plain one on an exactly convex target. |

## Running them

```bash
conda create -n econ282e python=3.11 -y
conda activate econ282e
pip install -r requirements.txt
jupyter lab
```

Labs 0&ndash;5 are **CPU-only** and run in well under a minute on a laptop. **Lab 7 is the
first that trains neural networks**: it still runs on a CPU, in a few minutes, and a GPU only
makes it faster.

## Notes

- Notebooks ship **without stored outputs**. Run them yourself — that is the point, and it
  also means a cell that fails on your machine fails visibly rather than looking correct.
- Before submitting anything built on a lab: **Restart Kernel and Run All**. A notebook run
  out of order can depend on a variable you have since deleted.
- The notebooks are generated from `tools/build_labs.py`. Edit there, not in the `.ipynb`,
  or your change will be overwritten on the next build.
- On a cluster, execute through `tools/run_code.slurm` — never on a login node.
