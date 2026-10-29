# Labs — ECON 282E, Foundations of Macroeconomics

| Lab | Session | What it does |
|---|---|---|
| `L00_Getting_Started.ipynb` | before S3 | Install Python, verify the environment, first program, the Solow model against its closed form. |
| `L03_Python_and_VFI.ipynb` | S3 (Oct 8) | Python and NumPy essentials; the `GrowthModel` class; VFI and time iteration, both graded against Brock–Mirman; convergence and accuracy diagnostics; the HP filter; a first look at autograd. |
| `L04_Numerical_Methods.ipynb` | S4 (Oct 15) | Root-finding with measured convergence orders; golden section; interpolation shape tests and **Smolyak sparse grids**; quadrature; **Tauchen against Rouwenhorst**; Howard and EGM on a quarterly RBC model; and the **4.B4 worked example end to end** — Rouwenhorst + interpolation + golden section + Howard, against the crude Session 3 solver. |
| `L05_Perturbation_and_Projection.ipynb` | S5 (Oct 22) | Linearization and the **Blanchard--Kahn solve in 5.A3's five steps** (roots of $M$, policy $C$, law of motion $K$); the impulse response, explained before the code; why monomials are unusable (the Hilbert matrix); Chebyshev nodes and convergence; a **Chebyshev collocation solve of the same quarterly RBC model**; and the closing table of one model solved four ways. |
| `L06_Aiyagari_End_To_End.ipynb` | S6 (Oct 29) | Session 6 in deck order: the Aiyagari household by **EGM**, by **value function iteration** (on the nodes, interpolated with golden section, Howard steps) and by **time iteration**, graded side by side; by **projection** (Chebyshev collocation), graded against a 16,000-node reference; **Young's lottery** and the invariant distribution two ways; a careless bisection, the equilibrium and five checks; **Krusell–Smith** with Den Haan's test, and its household by VFI; capital differentiated through the lottery (PyTorch, optional); an **MIT shock** in the growth model three ways, then with a distribution, and who gains; the **life-cycle economy**. Part 0 inlines `slides/S06/code/`, the code the decks print. |
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
