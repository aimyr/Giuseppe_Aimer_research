"""Preference judging — batch mode.

Submits batch jobs to each provider for pairwise preference judging on
dataset3.csv. Uses 5 samples per pair (3 NORMAL + 2 SWAP, temperature=0.7).
Saves batch metadata to judging/batch_jobs/preference_<judge>.json so a
later --fetch run can collect results without re-submitting.

Usage:
    python preference_batch.py --submit --model gpt4omini --first_n_rows 10
    python preference_batch.py --submit --model all --first_n_rows 300
    python preference_batch.py --status --model all
    python preference_batch.py --fetch --model all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judging_lib import JUDGES, list_judges  # noqa: E402
from batch_lib import (  # noqa: E402
    JOBS_DIR,
    check_status,
    fetch_results,
    load_meta,
    save_meta,
    submit_batch,
    supports_batch,
)


DEFAULT_CSV = HERE.parent / "dataset3.csv"
RESULTS_DIR = HERE / "preference"
PREF_JOBS_DIR = JOBS_DIR / "preference"

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

SAMPLE_PLAN = [("N", 1), ("S", 1), ("N", 2), ("S", 2), ("N", 3)]
SAMPLING_TEMPERATURE = 0.7


def build_pair_index(df: pd.DataFrame) -> List[tuple]:
    """List of (row_idx, topic, col_a, col_b)."""
    human_cols = ["winning_text", "losing_text"]
    machine_cols = sorted([
        c for c in df.columns
        if c.startswith("losing_") and any(k in c for k in ("paraphrase", "improve", "generate"))
    ])
    pairs = []
    for idx, row in df.iterrows():
        topic = str(row.get("topic", "") or "").strip() or str(row.get("theme", "")).strip()
        if pd.notna(row.get("winning_text")) and pd.notna(row.get("losing_text")):
            if str(row["winning_text"]).strip() and str(row["losing_text"]).strip():
                pairs.append((int(idx), topic, "winning_text", "losing_text"))
        for h in human_cols:
            for m in machine_cols:
                vh, vm = row.get(h), row.get(m)
                if pd.isna(vh) or pd.isna(vm):
                    continue
                if str(vh).strip() and str(vm).strip():
                    pairs.append((int(idx), topic, h, m))
    return pairs


def build_requests(df: pd.DataFrame) -> list[dict]:
    """Make 5 requests per pair, with custom_ids encoding (idx, A_col, B_col, role)."""
    pairs = build_pair_index(df)
    requests = []
    for idx, topic, col_a, col_b in pairs:
        row = df.loc[idx]
        text_a = str(row[col_a])
        text_b = str(row[col_b])
        for kind, sample_idx in SAMPLE_PLAN:
            if kind == "N":
                t1, t2 = text_a, text_b
            else:
                t1, t2 = text_b, text_a
            # Anthropic batch requires custom_id matching ^[a-zA-Z0-9_-]{1,64}$.
            # Column names contain '_', so use '-' as separator (column names have no '-').
            cid = f"{idx}-{col_a}-{col_b}-{kind}{sample_idx}"
            requests.append({
                "custom_id": cid,
                "prompt": PROMPT_TEMPLATE.format(topic=topic, text_1=t1, text_2=t2),
                "system": SYSTEM,
                "temperature": SAMPLING_TEMPERATURE,
                "max_tokens": 8,
            })
    return requests


# ---------------- submit ----------------

def cmd_submit(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    requests = build_requests(df)
    print(f"Built {len(requests)} requests from {len(df)} rows "
          f"({len(requests) // 5} pairs × 5 samples each)")

    aliases = list_judges() if args.model == "all" else [args.model]
    for alias in aliases:
        if alias not in JUDGES:
            print(f"  [{alias}] unknown judge — skipping")
            continue
        if not supports_batch(alias):
            print(f"  [{alias}] no batch support (provider={JUDGES[alias]['provider']}) — skipping")
            continue
        meta_path = PREF_JOBS_DIR / f"{alias}.json"
        if meta_path.exists() and not args.force:
            print(f"  [{alias}] meta already exists at {meta_path}; skip (use --force to overwrite)")
            continue
        print(f"[{alias}] submitting...")
        try:
            meta = submit_batch(alias, requests)
        except Exception as exc:
            print(f"  [{alias}] SUBMIT FAILED: {type(exc).__name__}: {exc}")
            continue
        meta["task"] = "preference"
        meta["first_n_rows"] = args.first_n_rows
        meta["csv"] = str(args.csv)
        meta["num_requests"] = len(requests)
        save_meta(meta, meta_path)
        print(f"  [{alias}] saved meta → {meta_path}")


# ---------------- status ----------------

def cmd_status(args: argparse.Namespace) -> None:
    aliases = list_judges() if args.model == "all" else [args.model]
    for alias in aliases:
        meta_path = PREF_JOBS_DIR / f"{alias}.json"
        if not meta_path.exists():
            print(f"  [{alias}] no batch meta")
            continue
        meta = load_meta(meta_path)
        try:
            status = check_status(meta)
        except Exception as exc:
            print(f"  [{alias}] status check failed: {exc}")
            continue
        save_meta(meta, meta_path)
        # Per-chunk states
        chunk_states = [ck.get("status") for ck in meta.get("chunks", [])]
        print(f"  [{alias}] overall={status}  chunks={chunk_states}")


# ---------------- fetch + parse ----------------

def _parse_choice(text: str) -> str:
    if not text:
        return "ERROR"
    if text.startswith("BATCH_ERROR") or text.startswith("ERROR"):
        return "ERROR"
    s = text.strip().replace('"', "").replace("'", "").replace(".", "").strip()
    if s == "1": return "1"
    if s == "2": return "2"
    if "1" in s and "2" not in s: return "1"
    if "2" in s and "1" not in s: return "2"
    return "AMBIG"


def cmd_fetch(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    pairs = build_pair_index(df)

    aliases = list_judges() if args.model == "all" else [args.model]
    for alias in aliases:
        meta_path = PREF_JOBS_DIR / f"{alias}.json"
        if not meta_path.exists():
            print(f"  [{alias}] no meta")
            continue
        meta = load_meta(meta_path)
        status = check_status(meta)
        if status != "completed":
            print(f"  [{alias}] not ready (status={status})")
            continue
        print(f"[{alias}] fetching results...")
        results = fetch_results(meta)

        # Aggregate per pair
        out_rows = []
        for idx, topic, col_a, col_b in pairs:
            row_id = df.loc[idx].get("id", idx)
            votes_by_role = {}
            raws_by_role = {}
            for kind, sample_idx in SAMPLE_PLAN:
                role = f"{kind}{sample_idx}"
                cid = f"{idx}-{col_a}-{col_b}-{role}"
                raw = results.get(cid, "")
                raws_by_role[role] = raw
                ch = _parse_choice(raw)
                # Map 1/2 back to actual source
                if kind == "N":
                    src1, src2 = col_a, col_b
                else:
                    src1, src2 = col_b, col_a
                if ch == "1":
                    votes_by_role[role] = src1
                elif ch == "2":
                    votes_by_role[role] = src2
                else:
                    votes_by_role[role] = ch  # ERROR / AMBIG

            normals = [votes_by_role[r] for r in ("N1", "N2", "N3")]
            swaps = [votes_by_role[r] for r in ("S1", "S2")]
            tally: dict[str, int] = {}
            for v in normals + swaps:
                if v in ("ERROR", "AMBIG"):
                    continue
                tally[v] = tally.get(v, 0) + 1
            if tally:
                maj_src, maj_count = max(tally.items(), key=lambda kv: kv[1])
            else:
                maj_src, maj_count = "Error", 0

            normal_consistent = (
                "Yes" if len(set(normals)) == 1 and "ERROR" not in normals
                else "No"
            )
            order_influenced = "No" if set(normals) == set(swaps) else "Yes"

            out_rows.append({
                "row_id": row_id,
                "topic": topic,
                "A_source": col_a,
                "B_source": col_b,
                "N1_raw": raws_by_role.get("N1", ""),
                "S1_raw": raws_by_role.get("S1", ""),
                "N2_raw": raws_by_role.get("N2", ""),
                "S2_raw": raws_by_role.get("S2", ""),
                "N3_raw": raws_by_role.get("N3", ""),
                "chosen_N1": votes_by_role.get("N1"),
                "chosen_S1": votes_by_role.get("S1"),
                "chosen_N2": votes_by_role.get("N2"),
                "chosen_S2": votes_by_role.get("S2"),
                "chosen_N3": votes_by_role.get("N3"),
                "majority_source": maj_src,
                "majority_count_of_5": maj_count,
                "normal_consistent": normal_consistent,
                "order_influenced_decision": order_influenced,
            })

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"preference_{alias}_batch.csv"
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        print(f"  [{alias}] saved → {out_path} ({len(out_rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_mutually_exclusive_group(required=True)
    sub.add_argument("--submit", action="store_true")
    sub.add_argument("--status", action="store_true")
    sub.add_argument("--fetch",  action="store_true")
    p.add_argument("--model", default="all",
                   help="Judge alias or 'all'. Available: " + ", ".join(list_judges()))
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--first_n_rows", type=int, default=None)
    p.add_argument("--force", action="store_true", help="overwrite existing batch meta on submit")
    args = p.parse_args()
    if args.submit:
        cmd_submit(args)
    elif args.status:
        cmd_status(args)
    elif args.fetch:
        cmd_fetch(args)


if __name__ == "__main__":
    main()
