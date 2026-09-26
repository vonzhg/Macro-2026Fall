# Data for Homework 2, Part B (question C4)

`fomc/` — every FOMC post-meeting policy statement, 1994-02-04 to 2026-06-17
(245 plain-text files, one per statement, `fomc_YYYYMMDD.txt`). Each file starts
with a two-line provenance header (`# FOMC statement, YYYY-MM-DD` and
`# source: <url>`).

- **Source:** the Federal Reserve Board's website (www.federalreserve.gov),
  fetched 2026-07-09. FOMC statements are works of the U.S. federal government
  and are in the **public domain**.
- **Copied from:** the companion course AI-ECON-2026, `labs/Lec07_LLM_Lab/data/`
  (see `README_fomc.md` for how the corpus was built and what it excludes).
- **Loader:** `fomc_data.py` (standard library only):
  `from fomc_data import load_statements; docs = load_statements("fomc")`.

Validation series are **not** bundled: download them yourself from FRED and
record the download date (e.g. `DFF`, `DFEDTARU`, `VIXCLS`, `DGS10`).
