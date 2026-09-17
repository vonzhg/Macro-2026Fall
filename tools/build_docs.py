#!/usr/bin/env python3
"""Compile the ECON 282E documents: syllabus, homework, exams, project guidelines.

    python3 tools/build_docs.py                          # every document found
    python3 tools/build_docs.py syllabus/syllabus.tex    # one document
    python3 tools/build_docs.py --solutions homeworks-exams/HW1/hw1.tex

A document may define a solutions switch with
    \\newif\\ifsolutions \\solutionsfalse
and wrap answers in \\ifsolutions ... \\fi. With --solutions the file is compiled as
<stem>_solutions.pdf with \\solutionstrue injected; solution PDFs are gitignored.

Runs pdflatex twice in the document's own folder (bibtex is run if a \\bibliography is found).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GLOBS = ["syllabus/*.tex", "homeworks-exams/*/*.tex"]


def compile_doc(tex: str, solutions: bool) -> bool:
    folder, name = os.path.split(os.path.abspath(tex))
    stem = os.path.splitext(name)[0]
    jobname = stem + ("_solutions" if solutions else "")
    if solutions:
        # \solutionstrue must run after the document's own \newif, i.e. at \begin{document}
        src = r"\AtBeginDocument{\solutionstrue}\input{" + name + "}"
    else:
        src = name
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={jobname}", src]
    uses_bib = bool(re.search(r"\\bibliography\{", open(tex, encoding="utf-8").read()))
    proc = subprocess.run(cmd, cwd=folder, capture_output=True)
    if proc.returncode == 0 and uses_bib:
        subprocess.run(["bibtex", jobname], cwd=folder, capture_output=True)
        subprocess.run(cmd, cwd=folder, capture_output=True)
    if proc.returncode == 0:
        proc = subprocess.run(cmd, cwd=folder, capture_output=True)
    log = os.path.join(folder, jobname + ".log")
    logtxt = open(log, encoding="latin-1").read() if os.path.exists(log) else ""
    if proc.returncode != 0:
        errs = [l for l in logtxt.splitlines() if l.startswith("!")]
        print(f"  {os.path.relpath(tex, ROOT):<48} FAILED\n    " + "\n    ".join(errs[:10]))
        return False
    pages = re.findall(r"Output written on .*?\((\d+) pages?", logtxt)
    overfull = len(re.findall(r"Overfull \\hbox", logtxt))
    warn_undef = len(re.findall(r"LaTeX Warning: (Reference|Citation) .* undefined", logtxt))
    print(f"  {os.path.relpath(tex, ROOT):<48} {jobname}.pdf  {pages[-1] if pages else '?'}p  "
          f"overfull={overfull} undefined-refs={warn_undef}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="*")
    ap.add_argument("--solutions", action="store_true")
    args = ap.parse_args()
    docs = args.docs or sorted(p for g in DEFAULT_GLOBS for p in glob.glob(os.path.join(ROOT, g)))
    if not docs:
        sys.exit("no documents found")
    ok = all([compile_doc(d, args.solutions) for d in docs])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
