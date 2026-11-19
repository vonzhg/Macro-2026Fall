#!/usr/bin/env python3
"""Record an LLM stance score for every FOMC statement, once (Sessions 5-6 plan D4(c)).

Instructor tool, not part of the notebook: the notebook only reads what this writes.
Every call is made from an EMPTY working directory with every tool disallowed and one turn, so the
model sees the statement and nothing else -- no repository, no corpus, no panel.

  python3 record_stance.py            # the full recording
  python3 record_stance.py --dry-run  # print the plan and stop

Design (plan v2 section 3):
  variant A  the frozen prompt, on every statement -- this is the signal the forecast uses
  variant B  the same task, reworded
  variant C  variant A with the three options in a different order
  variant D  byte-identical to A, on the same subsample as B and C, so that prompt sensitivity can
             be told apart from sampling noise
Dates are stripped from the text before the model sees it.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, json, os, re, subprocess, sys, tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
FOMC = ASSETS / "fomc"
PROMPTS = HERE / "prompts"
MODEL = "sonnet"
SUBSAMPLE = 50          # for variants B, C and D; stratified by era
WORKERS = 8
DENY = ("Bash Read Write Edit Glob Grep WebFetch WebSearch Task NotebookEdit TodoWrite "
        "BashOutput KillShell SlashCommand Skill")
MONTH = (r"(January|February|March|April|May|June|July|August|September|October|November|December)")


def strip_dates(raw: str) -> str:
    t = "\n".join(l for l in raw.split("\n") if not l.startswith("#")).strip()
    t = re.sub(MONTH + r"\s+\d{1,2},?\s+(19|20)\d\d", "[DATE]", t)
    t = re.sub(MONTH + r"\s+(19|20)\d\d", "[DATE]", t)
    t = re.sub(r"\b(19|20)\d\d\b", "[YEAR]", t)
    t = re.sub(MONTH, "[MONTH]", t)
    return t


def statements() -> list[dict]:
    out = []
    for p in sorted(FOMC.glob("*.txt")):
        d = p.name[5:13]
        out.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "file": p.name,
                    "text": strip_dates(p.read_text(encoding="utf-8"))})
    return out


def subsample(rows: list[dict], n: int) -> list[dict]:
    """Deterministic, stratified by era so every regime is represented."""
    eras = [("1994", "2004"), ("2005", "2014"), ("2015", "2019"), ("2020", "2099")]
    per, picked = n // len(eras), []
    for lo, hi in eras:
        band = [r for r in rows if lo <= r["date"][:4] <= hi]
        if not band:
            continue
        step = max(1, len(band) // per)
        picked += band[::step][:per]
    return sorted(picked, key=lambda r: r["date"])[:n]


def one_call(text: str, template: str) -> dict:
    prompt = template.replace("{TEXT}", text)
    with tempfile.TemporaryDirectory() as empty:
        # The prompt goes on stdin, never as a trailing argument: --disallowedTools is variadic and
        # would swallow it, and the CLI would then exit with "Input must be provided".
        r = subprocess.run(
            ["claude", "-p", "--output-format", "json", "--no-session-persistence",
             "--model", MODEL, "--effort", "low", "--permission-mode", "plan",
             "--disallowedTools", DENY],
            cwd=empty, input=prompt, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return {"error": r.stderr[-400:], "returncode": r.returncode}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"error": "unparseable", "stdout": r.stdout[-400:]}


LABELS = ("HAWKISH", "DOVISH", "NEUTRAL")


def parse_label(result: str | None) -> str | None:
    if not result:
        return None
    up = result.strip().upper()
    hits = [l for l in LABELS if l in up]
    return hits[0] if len(hits) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = statements()
    sub = subsample(rows, SUBSAMPLE)
    subdates = {r["date"] for r in sub}
    templates = {k: (PROMPTS / f"{k}.txt").read_text(encoding="utf-8") for k in "ABCD"}
    phash = {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in templates.items()}

    jobs = [(r, "A") for r in rows] + [(r, v) for v in "BCD" for r in sub]
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    print(f"statements {len(rows)}  subsample {len(sub)}  calls {len(jobs)}  harness {version}")
    print("prompt hashes:", phash, " (A and D are identical by design)")
    if args.dry_run:
        print("dry run: nothing called"); return 0

    raw_path = HERE / "raw_replies.jsonl"
    done = set()
    if raw_path.exists():                       # resume without paying twice
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if not rec.get("error"):
                    done.add((rec["date"], rec["variant"]))
            except json.JSONDecodeError:
                pass
    jobs = [j for j in jobs if (j[0]["date"], j[1]) not in done]
    print(f"already recorded {len(done)}; running {len(jobs)}")

    cost, n_ok, n_bad = 0.0, 0, 0
    with raw_path.open("a", encoding="utf-8") as fh:
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(one_call, r["text"], templates[v]): (r, v) for r, v in jobs}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                r, v = futs[fut]
                try:
                    d = fut.result()
                except Exception as e:                        # noqa: BLE001
                    d = {"error": f"{type(e).__name__}: {e}"}
                rec = {"date": r["date"], "file": r["file"], "variant": v,
                       "prompt_sha256_16": phash[v], "model_requested": MODEL,
                       "harness": version, "recorded_at": datetime.now().isoformat(timespec="seconds"),
                       "label": parse_label(d.get("result")), "result": d.get("result"),
                       "cost_usd": d.get("total_cost_usd"), "duration_ms": d.get("duration_ms"),
                       "model_used": list((d.get("modelUsage") or {}).keys()),
                       "session_id": d.get("session_id"), "error": d.get("error")}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                cost += d.get("total_cost_usd") or 0.0
                n_ok += rec["label"] is not None
                n_bad += rec["label"] is None
                if i % 25 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  parsed {n_ok}  unparsed {n_bad}  ${cost:.2f}", flush=True)
    print(f"done. parsed {n_ok}, unparsed {n_bad}, cost this run ${cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
