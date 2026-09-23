#!/usr/bin/env python3
"""Merge the built deck PDFs into one file, with bookmarks.

The decks in slides/pdf/ are already watermarked by build_slides.py, so the
merged file inherits the watermark.  A PDF outline is added: one top-level
entry per session, one child per deck, so the combined file is navigable.

    python3 tools/combine_slides.py                 # every built deck
    python3 tools/combine_slides.py S06             # one session
    python3 tools/combine_slides.py S05 S06 S07     # several
    python3 tools/combine_slides.py --public        # omit the publish blockers
    python3 tools/combine_slides.py --no-watermark  # clean copy, for review
    python3 tools/combine_slides.py S06 -o six.pdf  # choose the output name

WARNING.  By default the merge includes the three decks that carry unpublished
research (see slides/SOURCES.md).  The output is therefore gitignored and is
for the instructor's own use.  Use --public for a file that is safe to hand
out; it drops those three decks and says so on stdout.

--no-watermark takes the unstamped PDFs that sit next to each .tex instead of
the stamped ones in slides/pdf/.  Those are gitignored build products, so a
merge of them is for review only.  Its name always ends in "_review", which is
deliberately outside the .gitignore exception for "*_public.pdf", so an
unwatermarked file can never become tracked by accident.
"""

import argparse
import json
import os
import re
import sys

import pikepdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDFDIR = os.path.join(ROOT, "slides", "pdf")

# The three decks that may not go into a public artefact (slides/SOURCES.md).
BLOCKERS = {
    "S01_A1_Who_I_Am.pdf",
    "S02_A2_Aiyagari.pdf",
    "S02_B3_Ramsey_Time_Consistency.pdf",
}

SESSION_TITLES = {
    "S01": "Session 1 - Introduction, and the optimal growth model",
    "S02": "Session 2 - Heterogeneity, OLG, and optimal policy",
    "S03": "Session 3 - Computation, and programming basics",
    "S04": "Session 4 - Numerical methods I",
    "S05": "Session 5 - Numerical methods II",
    "S06": "Session 6 - Heterogeneous agents, classically",
    "S07": "Session 7 - Deep learning for macroeconomics",
    "S08": "Session 8 - Reinforcement learning, and HA at the frontier",
    "S09": "Session 9 - Text as economic data",
    "S10": "Session 10 - Agentic AI, and project presentations",
}

DECK_RE = re.compile(r"^(S\d{2})_([AB]\d[a-z]?)_(.+)\.pdf$")


def deck_title(stem_parts):
    """'S06', 'A1', 'Solving_Aiyagari' -> '6.A1  Solving Aiyagari'."""
    sess, deck, rest = stem_parts
    return "%d.%s  %s" % (int(sess[1:]), deck, rest.replace("_", " "))


def source_path(name, unstamped):
    """Where to read a deck from: slides/pdf/ (stamped) or beside its .tex."""
    if not unstamped:
        return os.path.join(PDFDIR, name)
    return os.path.join(ROOT, "slides", DECK_RE.match(name).group(1), name)


def collect(selectors, public):
    names = sorted(n for n in os.listdir(PDFDIR) if DECK_RE.match(n))
    if selectors:
        keep = []
        for n in names:
            stem = n[:-4]
            if any(stem == s or stem.startswith(s) for s in selectors):
                keep.append(n)
        names = keep
    dropped = []
    if public:
        dropped = [n for n in names if n in BLOCKERS]
        names = [n for n in names if n not in BLOCKERS]
    return names, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selectors", nargs="*",
                    help="session or deck prefixes, e.g. S06 or S06_B2")
    ap.add_argument("--public", action="store_true",
                    help="omit the three decks with unpublished research")
    ap.add_argument("--no-watermark", dest="no_watermark", action="store_true",
                    help="merge the unstamped PDFs beside each .tex (review only)")
    ap.add_argument("-o", "--output", default=None, help="output file name")
    args = ap.parse_args(argv)

    names, dropped = collect(args.selectors, args.public)
    if not names:
        sys.exit("no built deck PDFs matched %s" % (args.selectors or "(all)"))

    if args.output:
        out = args.output if os.path.isabs(args.output) \
            else os.path.join(PDFDIR, args.output)
    else:
        tag = "_".join(args.selectors) if args.selectors else "all"
        out = os.path.join(PDFDIR, "ECON282E_%s%s%s.pdf"
                           % (tag, "_public" if args.public else "",
                              "_review" if args.no_watermark else ""))

    missing, stale = [], []
    for n in names:
        path = source_path(n, args.no_watermark)
        if not os.path.exists(path):
            missing.append(n)
            continue
        tex = path[:-4] + ".tex"
        if os.path.exists(tex) and os.path.getmtime(path) < os.path.getmtime(tex):
            stale.append(n)
    if missing:
        sys.exit("not built: %s\n(run tools/build_slides.py first)"
                 % ", ".join(missing))
    if stale:
        print("WARNING: older than their source, rebuild first: %s"
              % ", ".join(stale))

    merged = pikepdf.Pdf.new()
    entries = []                       # (session, deck title, first page index)
    for n in names:
        m = DECK_RE.match(n)
        with pikepdf.open(source_path(n, args.no_watermark)) as src:
            entries.append((m.group(1), deck_title(m.groups()), len(merged.pages)))
            merged.pages.extend(src.pages)

    # one outline entry per session, one child per deck
    with merged.open_outline() as ol:
        current, node = None, None
        for sess, title, page in entries:
            if sess != current:
                current = sess
                node = pikepdf.OutlineItem(
                    SESSION_TITLES.get(sess, sess), page)
                ol.root.append(node)
            node.children.append(pikepdf.OutlineItem(title, page))

    merged.docinfo["/Title"] = ("ECON 282E - Foundations of Macroeconomics "
                                "(UC Riverside, Fall 2026)")
    merged.docinfo["/Author"] = "Zhigang Feng"
    merged.save(out)

    by_session = {}
    for sess, _, _ in entries:
        by_session[sess] = by_session.get(sess, 0) + 1
    print("merged %d decks, %d pages -> %s"
          % (len(names), len(merged.pages), os.path.relpath(out, ROOT)))
    print("  " + "  ".join("%s:%d" % (s, c) for s, c in sorted(by_session.items())))
    if args.no_watermark:
        print("  NO WATERMARK - review copy, not for distribution.")
    if dropped:
        print("  omitted (--public): " + ", ".join(sorted(dropped)))
    else:
        included = [n for n in names if n in BLOCKERS]
        if included:
            print("  WARNING: contains unpublished research (%s)."
                  % ", ".join(sorted(included)))
            print("  This file is gitignored. Use --public to hand it out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
