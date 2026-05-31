"""Pairwise preference judging (forum-post comments) — 5-sample voting.

Per (human_col, machine_col) pair, ask the judge 5 times:
  NORMAL1, NORMAL2, NORMAL3 — order A=human, B=machine
  SWAP1, SWAP2             — order A=machine, B=human
All five with temperature=0.7 for diversity. Position-bias is captured by
comparing the chosen source across NORMAL vs SWAP, and decision robustness
by checking agreement across the 5 votes.

Usage:
    python preference.py --model gpt5nano --first_n_rows 5
    python preference.py --model all --first_n_rows 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judging_lib import JUDGES, list_judges, query_judge  # noqa: E402


DEFAULT_CSV = HERE.parent / "dataset3.csv"
RESULTS_DIR = HERE / "preference"

SYSTEM = (
    "You are an expert forum community moderator and a careful pairwise judge. "
    "Follow the requested output format strictly."
)

PROMPT_TEMPLATE = (
    "You are an expert forum community moderator. I will provide a forum post topic "
    "and two comments that were posted as replies. Your task is to decide which "
    "comment is the better reply — clearer, more helpful, more relevant to the post, "
    "and more appropriate in tone.\n\n"
    "Forum post topic: {topic}\n\n"
    "Comment 1:\n{text_1}\n\n"
    "Comment 2:\n{text_2}\n\n"
    "Which comment is the better reply? "
    'Answer with ONLY "1" or "2" and no other text.'
)

# 5 samples per pair: 3 NORMAL + 2 SWAP.
SAMPLE_PLAN = [("NORMAL", 1), ("SWAP", 1), ("NORMAL", 2), ("SWAP", 2), ("NORMAL", 3)]
SAMPLING_TEMPERATURE = 0.7


def _parse_choice(answer_text: str) -> str:
    """Return '1', '2', or 'AMBIG'."""
    if not answer_text or answer_text.startswith("ERROR"):
        return answer_text or "ERROR"
    s = answer_text.strip().replace('"', "").replace("'", "").replace(".", "").strip()
    if s == "1":
        return "1"
    if s == "2":
        return "2"
    if "1" in s and "2" not in s:
        return "1"
    if "2" in s and "1" not in s:
        return "2"
    return "AMBIG"


def _resolve_source(choice: str, src_when_1: str, src_when_2: str) -> str:
    if choice == "1":
        return src_when_1
    if choice == "2":
        return src_when_2
    return "Error"


def _build_pair_jobs(df: pd.DataFrame) -> List[Tuple[int, str, str, str]]:
    """Return list of (row_idx, topic, col_a, col_b) human-vs-machine pairs.

    Pairs:
      - (winning_text, losing_text)        — human vs human
      - (winning_text, <machine_col>) for every machine col
      - (losing_text,  <machine_col>) for every machine col
    """
    human_cols = ["winning_text", "losing_text"]
    machine_cols = [
        c for c in df.columns
        if c.startswith("losing_") and any(k in c for k in ("paraphrase", "improve", "generate"))
    ]
    machine_cols.sort()

    jobs: List[Tuple[int, str, str, str]] = []
    for idx, row in df.iterrows():
        topic = str(row.get("topic", "") or "").strip() or str(row.get("theme", "")).strip()
        # Human vs Human (only once)
        if pd.notna(row.get("winning_text")) and pd.notna(row.get("losing_text")):
            if str(row["winning_text"]).strip() and str(row["losing_text"]).strip():
                jobs.append((int(idx), topic, "winning_text", "losing_text"))
        # Human vs Machine
        for h in human_cols:
            for m in machine_cols:
                vh, vm = row.get(h), row.get(m)
                if pd.isna(vh) or pd.isna(vm):
                    continue
                if str(vh).strip() and str(vm).strip():
                    jobs.append((int(idx), topic, h, m))
    return jobs


def _vote_pair(
    model_alias: str,
    topic: str,
    text_a: str,
    text_b: str,
) -> Dict[str, str]:
    """Send the 5 samples for a single (text_a, text_b) pair. Returns dict of
    role -> chosen_source where role is N1/N2/N3/S1/S2."""
    out: Dict[str, str] = {}
    for kind, sample_idx in SAMPLE_PLAN:
        if kind == "NORMAL":
            t1, t2 = text_a, text_b
            src1, src2 = "A", "B"  # placeholder, resolved by caller
        else:
            t1, t2 = text_b, text_a
            src1, src2 = "B", "A"
        prompt = PROMPT_TEMPLATE.format(topic=topic, text_1=t1, text_2=t2)
        ans = query_judge(model_alias, prompt,
                          temperature=SAMPLING_TEMPERATURE,
                          system=SYSTEM,
                          max_tokens=8)
        choice = _parse_choice(ans)
        # Map back to A/B regardless of order.
        if choice in ("1", "2"):
            chosen_ab = src1 if choice == "1" else src2
        else:
            chosen_ab = "Error" if choice == "ERROR" else "Ambig"
        role = ("N" if kind == "NORMAL" else "S") + str(sample_idx)
        out[role] = chosen_ab
        out[role + "_raw"] = ans
    return out


def _run_for_model(
    model_alias: str,
    df: pd.DataFrame,
    output_csv: Path,
    parallelism: int,
) -> None:
    jobs = _build_pair_jobs(df)
    print(f"[{model_alias}] {len(jobs)} pairs × 5 samples = {len(jobs) * 5} API calls")
    if not jobs:
        print(f"[{model_alias}] no jobs — skipping")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    last_save = time.time()
    completed = 0
    total = len(jobs)

    def _do(job):
        idx, topic, col_a, col_b = job
        row = df.loc[idx]
        text_a = str(row[col_a])
        text_b = str(row[col_b])
        votes = _vote_pair(model_alias, topic, text_a, text_b)
        return idx, topic, col_a, col_b, votes

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futs = [pool.submit(_do, j) for j in jobs]
        for fut in as_completed(futs):
            idx, topic, col_a, col_b, votes = fut.result()
            row_id = df.loc[idx].get("id", idx)
            # Resolve chosen_source per vote (A↔col_a, B↔col_b).
            resolved = {}
            for role in ("N1", "N2", "N3", "S1", "S2"):
                ab = votes.get(role, "Error")
                if ab == "A":
                    resolved[role] = col_a
                elif ab == "B":
                    resolved[role] = col_b
                else:
                    resolved[role] = ab  # Error / Ambig

            # Aggregate: position-bias = NORMAL vote vs SWAP vote (any disagreement)
            normal_votes = [resolved[r] for r in ("N1", "N2", "N3")]
            swap_votes = [resolved[r] for r in ("S1", "S2")]
            all_votes = normal_votes + swap_votes
            # Majority decision over the 5
            tally: Dict[str, int] = {}
            for v in all_votes:
                if v in ("Error", "Ambig"):
                    continue
                tally[v] = tally.get(v, 0) + 1
            if tally:
                majority = max(tally.items(), key=lambda kv: kv[1])
                majority_source = majority[0]
                majority_count = majority[1]
            else:
                majority_source = "Error"
                majority_count = 0

            normal_consistent = "Yes" if len(set(normal_votes)) == 1 and "Error" not in normal_votes else "No"
            order_influenced = "No" if set(normal_votes) == set(swap_votes) else "Yes"

            results.append({
                "row_id": row_id,
                "topic": topic,
                "A_source": col_a,
                "B_source": col_b,
                "N1_raw": votes.get("N1_raw", ""),
                "S1_raw": votes.get("S1_raw", ""),
                "N2_raw": votes.get("N2_raw", ""),
                "S2_raw": votes.get("S2_raw", ""),
                "N3_raw": votes.get("N3_raw", ""),
                "chosen_N1": resolved["N1"],
                "chosen_S1": resolved["S1"],
                "chosen_N2": resolved["N2"],
                "chosen_S2": resolved["S2"],
                "chosen_N3": resolved["N3"],
                "majority_source": majority_source,
                "majority_count_of_5": majority_count,
                "normal_consistent": normal_consistent,
                "order_influenced_decision": order_influenced,
            })

            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  [{model_alias}] {completed}/{total} pairs done", flush=True)
            if time.time() - last_save > 30:
                pd.DataFrame(results).to_csv(output_csv, index=False)
                last_save = time.time()

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"[{model_alias}] saved → {output_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all",
                   help="Judge alias or 'all'. Available: " + ", ".join(list_judges()))
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--first_n_rows", type=int, default=None)
    p.add_argument("--parallelism", type=int, default=6)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    print(f"Loaded {len(df)} rows from {args.csv}")

    aliases = list_judges() if args.model == "all" else [args.model]
    for alias in aliases:
        if alias not in JUDGES:
            print(f"WARN: unknown alias '{alias}', skipping", file=sys.stderr)
            continue
        out_csv = RESULTS_DIR / f"preference_{alias}.csv"
        try:
            _run_for_model(alias, df, out_csv, parallelism=args.parallelism)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[{alias}] FAILED top-level: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
