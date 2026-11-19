# Labs — ECON 282E, Foundations of Macroeconomics

| Lab | Session | What it does |
|---|---|---|
| `L00_Getting_Started.ipynb` | before S3 | Install Python, verify the environment, first program, the Solow model against its closed form. |
| `L03_Python_and_VFI.ipynb` | S3 (Oct 8) | Python and NumPy essentials; the `GrowthModel` class; VFI and time iteration, both graded against Brock–Mirman; convergence and accuracy diagnostics; the HP filter; a first look at autograd. |
| `L04_Numerical_Methods.ipynb` | S4 (Oct 15) | Root-finding with measured convergence orders; golden section; interpolation shape tests and **Smolyak sparse grids**; quadrature; **Tauchen against Rouwenhorst**; Howard and EGM on a quarterly RBC model; and the **4.B4 worked example end to end** — Rouwenhorst + interpolation + golden section + Howard, against the crude Session 3 solver. |
| `L05_Perturbation_and_Projection.ipynb` | S5 (Oct 22) | Linearization and the **Blanchard--Kahn solve in 5.A3's five steps** (roots of $M$, policy $C$, law of motion $K$); the impulse response, explained before the code; why monomials are unusable (the Hilbert matrix); Chebyshev nodes and convergence; a **Chebyshev collocation solve of the same quarterly RBC model**; and the closing table of one model solved four ways. |
| `L06_Aiyagari_End_To_End.ipynb` | S6 (Oct 29) | Session 6 in deck order: the Aiyagari household by **EGM**, by **value function iteration** (on the nodes, interpolated with golden section, Howard steps) and by **time iteration**, graded side by side; by **projection** (Chebyshev collocation; the 16,000-node reference is optional); **Young's lottery** and the invariant distribution two ways, then by **simulating a panel** (how it works, what it costs); a careless bisection, the price by **steps along the excess demand**, the equilibrium and five checks; **Krusell–Smith** with Den Haan's test, and its household by VFI; capital differentiated through the lottery (PyTorch, optional); an **MIT shock** in the growth model three ways, then with a distribution, and who gains; the **life-cycle economy**. Part 0 inlines `slides/S06/code/`, the code the decks print. |
| `L07_ML_and_Deep_Growth.ipynb` | S7 (Nov 5) | **Block A:** tensors and autograd; **backpropagation by hand, checked against `autograd`**; the five-line loop landing on OLS. **Block B:** a curve whose law you wrote; kinks fixed by least squares and learned by training ($396\times$ better where the curvature is concentrated); the minimum-norm fit gradient descent picks, and why that is not overfitting. **Block C:** next month's 10-year Treasury yield from five frozen FRED series, split by date --- and it does not beat a random walk. **Block D:** the growth model solved on its Euler residual, three rungs, graded on held-out and off-domain states. Appendices: leakage in a text classifier; one input through four architectures; what a gradient costs. Reads the frozen data in `tools/figures/data/s07/`. |
| `L08_Actor_Critic_Heterogeneous_Agents.ipynb` | S8 (Nov 12) | Part B follows Feng, Han, Sargent and Zhu (8.B1). **Part A:** the answer key, Session 6's equilibrium on a 40-node histogram grid and 6.B2's perfect-foresight path from a tilted distribution. **Part B:** the Aiyagari household by **actor–critic with the whole histogram as a state**: the stationary rule plus a learned correction, the critic with a trailing copy, the actor on the envelope-condition Euler gradient, Young's lottery differentiated (checked against finite differences), a rollout penalty pinning $K$; graded at $\lambda^*$, by where its own law of motion settles, and on paths against perfect foresight, a fixed rule and today's-prices-only. **Part C:** **Krusell–Smith by DeepHAM** (generalised moment, fictitious play, three clocks), graded against 6.B1. Part 0 writes `slides/S06/code` and `slides/S08/code` files to `./l08_code/`; `USE_RECORDED` loads the reference networks from `l08_assets/`. |
| `L09_Words_As_Data.ipynb` | S9 (Nov 19) | Six blocks on the 245 FOMC statements, every vocabulary, weight and vector fitted on 1994–2014 alone. **Block A:** one sentence cleaned step by step, Chinese word boundaries, word pairs; byte-pair merges learned by hand, then WordPiece and unigram; TF-IDF, first on the slides' three documents; topics, with one LDA redraw by hand. **Block B:** word vectors (word2vec's step by hand; the slides' nine points, and whether fitted vectors keep their parallel moves) and a hawk–dove axis that **fails** validation, beside a ten-line rule that passes, read over the whole statement or one sentence. **Block C:** one word in two sentences; attention by hand, with heads, word order and the mask; then **Machine B**, a small character-level transformer: one layer rebuilt by hand, its attention, its next-character distribution, its writing, its first sixty training steps; and what a pretrained encoder returns, from a recorded run. **Block D:** a recorded language-model instrument, 389 replies: provenance, agreement, Cohen's κ. **Block E:** the book's chapter-14 simulations: attenuation, correction, per-class κ, self-consistency, order effects. **Block F:** do the Committee's words improve Session 7's bond-yield forecast, under a protocol hashed before the test window is scored once? Every gate is on procedure or reproduction, never on an outcome. |

## Running them

```bash
conda create -n econ282e python=3.11 -y
conda activate econ282e
pip install -r requirements.txt
jupyter lab
```

Labs 0&ndash;5 are **CPU-only** and run in well under a minute on a laptop; Lab 6 takes about two minutes. **Lab 7 is the
first that trains neural networks**: it still runs on a CPU, in about a minute, and a GPU only
makes it faster. **Lab 8** trains two actor–critics on a CPU in about five minutes; `USE_RECORDED = True` skips the training.
**Lab 9** runs offline on a CPU in a few minutes (block A's topic model takes about one); it needs scikit-learn and torch,
and it runs from `labs/` alone: Session 7's two frozen forecast files are copied into `l09_assets/s07/`, and the pretrained
models the slides discuss are read from one recorded run (`l09_assets/recorded_deck12_examples.json`), never downloaded.

## Notes

- Notebooks ship **without stored outputs**. Run them yourself — that is the point, and it
  also means a cell that fails on your machine fails visibly rather than looking correct.
- Before submitting anything built on a lab: **Restart Kernel and Run All**. A notebook run
  out of order can depend on a variable you have since deleted.
- The notebooks are generated from `tools/build_labs.py`. Edit there, not in the `.ipynb`,
  or your change will be overwritten on the next build.
- On a cluster, execute through `tools/run_code.slurm` — never on a login node.
