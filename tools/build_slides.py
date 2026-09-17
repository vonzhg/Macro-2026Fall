#!/usr/bin/env python3
"""Compile and publish the ECON 282E slide decks.

Every deck is a .tex file in slides/S01 ... slides/S10 named like S01_A1_Short_Title.tex.
This script runs pdflatex twice in the deck's folder, stamps the copyright watermark on
every page, and writes the published PDF to slides/pdf/ together with a manifest.

    python3 tools/build_slides.py                  # build every deck
    python3 tools/build_slides.py S01_A1 S01_A2    # build decks by name prefix
    python3 tools/build_slides.py S01              # build a whole session
    python3 tools/build_slides.py --check          # report stale/missing PDFs, write nothing
    python3 tools/build_slides.py --no-watermark S01_A1

Needs pdflatex on PATH (on Delta:
export PATH=/projects/bepc/zfeng2/texlive/2026/bin/x86_64-linux:$PATH) and pikepdf.
Adapted from Courses/AI-ECON-2026/tools/build_slides.py; here decks are compiled directly
rather than picked up as pre-built masters, so nothing outside this folder is involved.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDES = os.path.join(ROOT, "slides")
PDF_DIR = os.path.join(SLIDES, "pdf")
MANIFEST = os.path.join(PDF_DIR, "BUILD.json")
PREAMBLE_DIR = os.path.join(SLIDES, "preamble")
DECK_RE = re.compile(r"^S\d{2}_[AB]\d[a-z]?_.+\.tex$")
FRAME_BUDGET = 16


def decks() -> list[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(SLIDES, "S[0-9][0-9]", "*.tex"))):
        if DECK_RE.match(os.path.basename(p)):
            out.append(p)
    return out


def md5_of(paths: list[str]) -> str:
    h = hashlib.md5()
    for p in paths:
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def inputs_of(tex: str) -> list[str]:
    """Deck source plus the shared preamble files and the deck's figures."""
    files = [tex] + sorted(glob.glob(os.path.join(PREAMBLE_DIR, "*.tex")))
    figdir = os.path.join(os.path.dirname(tex), "figures")
    files += sorted(glob.glob(os.path.join(figdir, "*")))
    return files


def count_frames(tex: str) -> int:
    src = open(tex, encoding="utf-8").read()
    src = re.sub(r"(?m)(?<!\\)%.*$", "", src)
    return len(re.findall(r"\\begin\{frame\}", src)) + len(re.findall(r"\\makedecktitle", src))


def compile_deck(tex: str) -> tuple[bool, str]:
    folder, name = os.path.split(tex)
    stem = os.path.splitext(name)[0]
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", name]
    proc = None
    for _ in range(2):
        proc = subprocess.run(cmd, cwd=folder, capture_output=True)
        if proc.returncode != 0:
            break
    log = os.path.join(folder, stem + ".log")
    logtxt = open(log, encoding="latin-1").read() if os.path.exists(log) else ""
    ok = proc is not None and proc.returncode == 0 and os.path.exists(os.path.join(folder, stem + ".pdf"))
    if not ok:
        errs = [l for l in logtxt.splitlines() if l.startswith("!") or ".tex:" in l and "error" in l.lower()]
        return False, "\n".join(errs[:15]) or logtxt[-1500:]
    overfull = len(re.findall(r"Overfull \\[hv]box \((\d+\.\d+)pt", logtxt))
    big = [float(x) for x in re.findall(r"Overfull \\[hv]box \((\d+\.\d+)pt", logtxt) if float(x) > 10]
    undefined = len(re.findall(r"undefined", logtxt, flags=re.I))
    note = f"overfull={overfull}" + (f" (>{10}pt: {len(big)})" if big else "")
    if undefined:
        note += f" undefined-warnings={undefined}"
    return True, note


def build_watermark(workdir: str) -> str:
    shutil.copy(os.path.join(ROOT, "tools", "watermark.tex"), workdir)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=batchmode", "watermark.tex"], cwd=workdir,
                       capture_output=True)
    out = os.path.join(workdir, "watermark.pdf")
    if not os.path.exists(out):
        sys.exit("watermark.pdf was not produced -- is pdflatex on PATH?")
    return out


def stamp(src_pdf: str, overlay_page, dest: str) -> int:
    import pikepdf
    pdf = pikepdf.open(src_pdf)
    for page in pdf.pages:
        page.add_overlay(overlay_page, pikepdf.Rectangle(page.mediabox))
    n = len(pdf.pages)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".pdf")
    os.close(fd)
    pdf.save(tmp)
    pdf.close()
    os.replace(tmp, dest)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("decks", nargs="*", help="deck name prefixes, e.g. S01_A1 or S01")
    ap.add_argument("--check", action="store_true", help="report stale decks; write nothing")
    ap.add_argument("--no-watermark", action="store_true", help="copy PDFs without the watermark")
    args = ap.parse_args()

    targets = decks()
    if args.decks:
        def selected(p: str) -> bool:
            stem = os.path.splitext(os.path.basename(p))[0]
            return any(stem == d or stem.startswith(d + "_") for d in args.decks)
        targets = [t for t in targets if selected(t)]
        if not targets:
            sys.exit(f"no deck matches: {', '.join(args.decks)}")

    manifest = {}
    if os.path.exists(MANIFEST):
        try:
            manifest = json.load(open(MANIFEST, encoding="utf-8"))
        except ValueError:
            manifest = {}

    if args.check:
        stale = 0
        for tex in targets:
            stem = os.path.splitext(os.path.basename(tex))[0]
            rec = manifest.get(stem + ".pdf")
            live = md5_of(inputs_of(tex))
            published = os.path.join(PDF_DIR, stem + ".pdf")
            if not rec or not os.path.exists(published):
                print(f"  {stem:<44} NOT BUILT"); stale += 1
            elif rec.get("inputs_md5") != live:
                print(f"  {stem:<44} STALE"); stale += 1
            else:
                print(f"  {stem:<44} ok ({rec.get('pages')} pages)")
        return 1 if stale else 0

    os.makedirs(PDF_DIR, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="wm-")
    failures = 0
    try:
        overlay = None
        wm = None
        if not args.no_watermark:
            import pikepdf
            wm = pikepdf.open(build_watermark(workdir))
            overlay = wm.pages[0]
        for tex in targets:
            stem = os.path.splitext(os.path.basename(tex))[0]
            frames = count_frames(tex)
            ok, note = compile_deck(tex)
            if not ok:
                failures += 1
                print(f"  {stem:<44} FAILED\n{note}")
                continue
            built = os.path.join(os.path.dirname(tex), stem + ".pdf")
            dest = os.path.join(PDF_DIR, stem + ".pdf")
            if overlay is not None:
                pages = stamp(built, overlay, dest)
            else:
                shutil.copy(built, dest)
                import pikepdf
                pages = len(pikepdf.open(dest).pages)
            manifest[stem + ".pdf"] = {
                "source": os.path.relpath(tex, ROOT),
                "inputs_md5": md5_of(inputs_of(tex)),
                "pages": pages,
                "frames": frames,
                "watermarked": overlay is not None,
            }
            budget = "" if frames <= FRAME_BUDGET else f"  [over {FRAME_BUDGET}-frame budget]"
            print(f"  {stem:<44} {pages:>3}p {frames:>3} frames  {note}{budget}")
        with open(MANIFEST, "w", encoding="utf-8") as fh:
            json.dump(dict(sorted(manifest.items())), fh, indent=2)
            fh.write("\n")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
