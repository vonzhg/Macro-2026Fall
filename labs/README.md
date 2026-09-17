# Labs — ECON 282E, Foundations of Macroeconomics

| Lab | Session | What it does |
|---|---|---|
| `L00_Getting_Started.ipynb` | before S3 | Install Python, verify the environment, first program, the Solow model against its closed form. |
| `L03_Python_and_VFI.ipynb` | S3 (Oct 8) | Python and NumPy essentials; the `GrowthModel` class; VFI and time iteration, both graded against Brock–Mirman; convergence and accuracy diagnostics; the HP filter; a first look at autograd. |
| `L04_Numerical_Methods.ipynb` | S4 (Oct 15) | Root-finding with measured convergence orders; golden section; interpolation shape tests and **Smolyak sparse grids**; quadrature; **Tauchen against Rouwenhorst**; Howard and EGM on a quarterly RBC model; and the **4.B5 worked example end to end** — Rouwenhorst + interpolation + golden section + Howard, against the crude Session 3 solver. |

## Running them

```bash
conda create -n econ282e python=3.11 -y
conda activate econ282e
pip install -r requirements.txt
jupyter lab
```

Everything here is **CPU-only** and runs in well under a minute on a laptop. No GPU is
needed in this course before Session 7.

## Notes

- Notebooks ship **without stored outputs**. Run them yourself — that is the point, and it
  also means a cell that fails on your machine fails visibly rather than looking correct.
- Before submitting anything built on a lab: **Restart Kernel and Run All**. A notebook run
  out of order can depend on a variable you have since deleted.
- The notebooks are generated from `tools/build_labs.py`. Edit there, not in the `.ipynb`,
  or your change will be overwritten on the next build.
- On a cluster, execute through `tools/run_code.slurm` — never on a login node.
