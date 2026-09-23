# ECON 282E — Foundations of Macroeconomics

Course site by **Zhigang Feng** — a Ph.D. elective at the University of California, Riverside,
Fall 2026. Ten sessions on the workhorse models of modern macroeconomics, the numerical methods
that solve them, and the machine-learning tools that take over where those methods run out.

🔗 **Live site:** https://vonzhg.github.io/Macro-2026Fall/

The arc is three steps. **On paper** (Sessions 1–2): the neoclassical growth model, then
Bewley–Huggett–Aiyagari, Krusell–Smith, overlapping generations, and optimal fiscal policy with
and without commitment. **On the machine** (Sessions 3–6): numerical dynamic programming, Python
and PyTorch, root-finding, optimization, differentiation, approximation, quadrature, discretizing
stochastic processes, perturbation and projection — ending at the computational wall that
heterogeneous-agent economies run into. **At the frontier** (Sessions 7–10): deep learning,
reinforcement learning, language models, and agentic AI.

Throughout, one discipline: **every computed answer must be benchmarked and verified.** As AI
makes generating answers cheap, verification becomes the scarce skill.

## The companion course

Where this course ends, **[AI for Economic Research: Dynamic Models, Language, and
Agents](https://github.com/vonzhg/AI-ECON-2026)** begins — ten lectures on deep learning for
dynamic models, reinforcement learning with heterogeneous agents, LLMs and text as economic data,
retrieval-augmented generation, and agentic research workflows. Sessions 7–10 here are a
compressed tour of that material; the companion repository develops it in full, with its own labs.

> **Unlisted, not secret.** Every page carries `noindex, nofollow` and `robots.txt` disallows
> crawlers, so the site does not show up in search results — but this repository is public, so
> anyone with the link can read it. Treat the slide PDFs accordingly.

## For students

- **Everything** — syllabus, slides, labs, assignments — is linked from the
  [course home page](https://vonzhg.github.io/Macro-2026Fall/).
- **Slides** are posted session by session, a few days before each class. They are free PDF
  downloads, no password. Each page carries a copyright watermark; please don't redistribute or
  repost them without permission.
- **Labs** ship with **no stored output**, on purpose — you are meant to produce the numbers:

  ```bash
  git clone https://github.com/vonzhg/Macro-2026Fall.git
  cd Macro-2026Fall
  conda create -n econ282e python=3.11 -y
  conda activate econ282e
  pip install -r labs/requirements.txt
  jupyter lab
  ```

  Every notebook runs on a laptop CPU in well under a minute. No GPU is needed before Session 7.

## Structure

```
index.html              Course home — schedule, grading, quick links
syllabus.html           Syllabus in brief; syllabus/syllabus.pdf is authoritative
slides/index.html       Session decks (watermarked PDFs)
slides/S03/, S04/, ...  LaTeX sources, posted with each session
slides/preamble/        Shared Beamer preamble and the course map
labs/                   Jupyter notebooks + requirements.txt
homeworks-exams/HW1/    Homework 1, Parts A and B
tools/                  Build scripts, and the measurement ledgers behind the slides
```

## Reproducing the numbers on the slides

Sessions 3 and 4 are generated from a ledger: **no number appears on a slide unless it is a key in
a JSON file written by code that ran.** Figures are drawn in pgfplots from coordinates the same run
emitted. To regenerate:

```bash
python3 tools/figures/s03_generate.py           # -> tools/figures/s03_numbers.json
python3 tools/figures/s04_generate.py           # -> tools/figures/s04_numbers.json
python3 tools/figures/s04_generate.py smolyak   # or just one section, merged in
python3 tools/build_slides.py S04               # compile + watermark -> slides/pdf/
python3 tools/build_labs.py                     # regenerate the notebooks, clean
python3 tools/check_course.py                   # no out-of-folder paths; figures resolve
```

Where a source turned out to be wrong, the slide says so and shows the corrected number. Several
such corrections are marked on the decks themselves.

## Not in this repository

Answer keys, reference solutions, and the midterm are never published here.

## License

Code and build tooling: MIT (see `LICENSE`). Slides, syllabus and written materials:
© Zhigang Feng, shared for personal study — please do not redistribute without permission.
