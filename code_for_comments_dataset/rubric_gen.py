"""Generate a 1-5 rubric from each judge model.

Each rubric describes what makes a winning forum comment, with 5 score
levels. Output saved as judging/rubrics/RUBRIC_<judge>.csv with two columns
(score, description).

Usage:
    python rubric_gen.py --model all
    python rubric_gen.py --model gpt5nano
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judging_lib import list_judges, query_judge  # noqa: E402

RUBRICS_DIR = HERE / "rubrics"

SYSTEM = "You are a rigorous evaluation expert. You must strictly follow the requested output format."

PROMPT = (
    "You are an expert forum community moderator. Your task is to write clear "
    "and comprehensive guidelines describing the characteristics of a winning "
    "forum comment posted in reply to a forum post.\n"
    "Explain what elements make a comment a winning reply (helpful, clear, "
    "relevant, appropriate tone, addresses the post).\n\n"
    "Format Requirements:\n"
    "Return EXACTLY five sections labeled:\n"
    "Score 1: [Description]\n"
    "Score 2: [Description]\n"
    "Score 3: [Description]\n"
    "Score 4: [Description]\n"
    "Score 5: [Description]\n\n"
    "For EACH section provide a detailed description of criteria, common "
    "pitfalls, and reasoning. Do not include any preamble or trailing text — "
    "only the five sections."
)


def parse_rubric(text: str) -> list[tuple[int, str]]:
    """Split into 5 (score, description) tuples."""
    out: list[tuple[int, str]] = []
    # Use regex to find each "Score N:" header and capture until the next one or end.
    pattern = re.compile(r"Score\s*(\d)\s*:\s*(.*?)(?=Score\s*\d\s*:|$)", re.DOTALL | re.IGNORECASE)
    for m in pattern.finditer(text):
        score = int(m.group(1))
        desc = m.group(2).strip()
        # Strip leading brackets if model echoed [Description] placeholder
        desc = re.sub(r"^\[?\s*Description\s*\]?\s*[:\-]?\s*", "", desc, flags=re.IGNORECASE)
        if 1 <= score <= 5:
            out.append((score, desc))
    # Dedupe — keep first occurrence per score
    seen = set()
    deduped = []
    for s, d in out:
        if s not in seen:
            seen.add(s)
            deduped.append((s, d))
    return sorted(deduped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    aliases = list_judges() if args.model == "all" else [args.model]
    RUBRICS_DIR.mkdir(parents=True, exist_ok=True)
    for alias in aliases:
        path = RUBRICS_DIR / f"RUBRIC_{alias}.csv"
        if path.exists() and not args.force:
            print(f"  [{alias}] exists; --force to regenerate"); continue
        print(f"[{alias}] generating rubric...")
        ans = query_judge(alias, PROMPT, temperature=0.0, system=SYSTEM, max_tokens=2048)
        if ans.startswith("ERROR"):
            print(f"  [{alias}] FAILED: {ans[:200]}")
            continue
        parsed = parse_rubric(ans)
        if len(parsed) != 5:
            print(f"  [{alias}] parse warning: got {len(parsed)} sections, saving raw too")
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["score", "description"])
            for s, d in parsed:
                w.writerow([s, d])
        raw_path = path.with_suffix(".raw.txt")
        raw_path.write_text(ans, encoding="utf-8")
        print(f"  [{alias}] saved {len(parsed)} sections → {path}")


if __name__ == "__main__":
    main()
