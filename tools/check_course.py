#!/usr/bin/env python3
"""Portability and hygiene checks for the ECON 282E course folder.

    python3 tools/check_course.py

Fails (exit 1) if any .tex file under slides/, syllabus/ or homeworks-exams/
  * \\input / \\include / \\includegraphics / \\graphicspath points outside this folder,
  * references a figure that cannot be found (extensionless names and case differences
    are resolved the way pdflatex would on a case-sensitive file system -- a case mismatch
    is reported as an error because it breaks on Linux),
  * mentions the private folders (supporting-intro/, plans/) or absolute /u/ or /projects/ paths.
Warns if a deck exceeds the 16-frame budget.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX_GLOBS = ["slides/**/*.tex", "syllabus/**/*.tex", "homeworks-exams/**/*.tex"]
EXTS = ["", ".pdf", ".png", ".jpg", ".jpeg"]
PRIVATE = ["supporting-intro", "plans/", "/u/zfeng2", "/projects/", "/work/"]
FRAME_BUDGET = 16


def strip_comments(s: str) -> str:
    return re.sub(r"(?m)(?<!\\)%.*$", "", s)


def inside_root(path: str) -> bool:
    return os.path.realpath(path).startswith(os.path.realpath(ROOT) + os.sep)


def find_figure(name: str, dirs: list[str]) -> tuple[str | None, str | None]:
    for d in dirs:
        for ext in EXTS:
            cand = os.path.normpath(os.path.join(d, name + ext))
            if os.path.isfile(cand):
                return cand, None
            # case-insensitive match -> report as a portability error
            parent = os.path.dirname(cand)
            if os.path.isdir(parent):
                low = os.path.basename(cand).lower()
                for f in os.listdir(parent):
                    if f.lower() == low:
                        return None, os.path.join(parent, f)
    return None, None


def main() -> int:
    errors, warnings = [], []
    files = sorted({p for g in TEX_GLOBS for p in glob.glob(os.path.join(ROOT, g), recursive=True)})
    for tex in files:
        rel = os.path.relpath(tex, ROOT)
        if os.path.basename(tex).startswith("_"):
            continue
        folder = os.path.dirname(tex)
        src = strip_comments(open(tex, encoding="utf-8").read())
        for bad in PRIVATE:
            if bad in src:
                errors.append(f"{rel}: mentions private/absolute path '{bad}'")
        gdirs = [folder]
        for m in re.finditer(r"\\graphicspath\{((?:\{[^}]*\})+)\}", src):
            for d in re.findall(r"\{([^}]*)\}", m.group(1)):
                full = os.path.join(folder, d)
                if not inside_root(full):
                    errors.append(f"{rel}: \\graphicspath entry '{d}' leaves the course folder")
                gdirs.append(full)
        if "slides" + os.sep in tex and os.path.isdir(os.path.join(folder, "figures")):
            gdirs.append(os.path.join(folder, "figures"))
        for m in re.finditer(r"\\(input|include)\{([^}]*)\}", src):
            target = m.group(2)
            full = os.path.join(folder, target if target.endswith(".tex") else target + ".tex")
            if not inside_root(full):
                errors.append(f"{rel}: \\{m.group(1)}{{{target}}} leaves the course folder")
            elif not os.path.isfile(full) and not os.path.isfile(os.path.join(folder, target)):
                errors.append(f"{rel}: \\{m.group(1)}{{{target}}} not found")
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", src):
            name = m.group(1)
            found, case_hit = find_figure(name, gdirs)
            if found:
                if not inside_root(found):
                    errors.append(f"{rel}: figure '{name}' resolves outside the course folder")
            elif case_hit:
                errors.append(f"{rel}: figure '{name}' only matches with different case: {os.path.relpath(case_hit, ROOT)}")
            else:
                errors.append(f"{rel}: figure '{name}' not found")
        if rel.startswith("slides" + os.sep) and re.match(r"S\d{2}_", os.path.basename(tex)):
            frames = len(re.findall(r"\\begin\{frame\}", src)) + len(re.findall(r"\\makedecktitle", src))
            if frames > FRAME_BUDGET:
                warnings.append(f"{rel}: {frames} frames (budget {FRAME_BUDGET})")

    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\nchecked {len(files)} .tex files: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
