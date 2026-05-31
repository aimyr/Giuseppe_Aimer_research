"""Score one comment 1-5 — task 2.1 (own judgment and researcher-instruction).

Two modes:
  --mode own:      model scores using its own judgment.
  --mode instruct: model uses researcher-supplied criteria (helpful, clear, etc.).

Usage:
    python score_batch.py --mode own      --submit --model all --first_n_rows 300
    python score_batch.py --mode instruct --submit --model all --first_n_rows 300
    python score_batch.py --mode own      --status --model all
    python score_batch.py --mode instruct --fetch  --model all
    python score_batch.py --mode own      --stream --model llama --first_n_rows 300
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judging_lib import JUDGES, list_judges, query_judge  # noqa: E402
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
RESULTS_DIR = HERE / "ranking" / "one_comment_scoring"

SYSTEM = (
    "You are an expert forum community moderator. Your output MUST be exactly "
    "one character: the digit 1, 2, 3, 4, or 5. No words. No punctuation. "
    "No explanation. Just the single digit."
)

PROMPT_OWN = (
    "You are an expert forum community moderator. A user posted a forum topic "
    "and another user replied with a comment. Rate the comment from 1 to 5 based "
    "on your own judgment of how well it serves as a reply to the forum post.\n\n"
    "Forum post topic:\n{topic}\n\n"
    "Comment:\n{text}\n\n"
    "Score (1-5):"
)

PROMPT_INSTRUCT = (
    "You are an expert forum community moderator. A user posted a forum topic "
    "and another user replied with a comment. Rate the comment from 1 to 5 using "
    "these criteria:\n"
    "  - Helpfulness: does it actually address the post?\n"
    "  - Clarity: is it easy to understand?\n"
    "  - Relevance: is it on-topic?\n"
    "  - Tone: is it appropriate for a public forum?\n"
    "Weight all four equally. 1 = poor on all, 5 = excellent on all.\n\n"
    "Forum post topic:\n{topic}\n\n"
    "Comment:\n{text}\n\n"
    "Score (1-5):"
)


def _all_text_cols(df):
    machine = sorted([
        c for c in df.columns
        if c.startswith("losing_") and any(k in c for k in ("paraphrase", "improve", "generate"))
    ])
    return ["winning_text", "losing_text"] + machine


def _build_requests(df, mode):
    cols = _all_text_cols(df)
    tmpl = PROMPT_OWN if mode == "own" else PROMPT_INSTRUCT
    requests = []
    for idx, row in df.iterrows():
        topic = str(row.get("topic", "") or "").strip() or str(row.get("theme", "")).strip()
        for col in cols:
            val = row.get(col)
            if pd.isna(val) or not str(val).strip():
                continue
            text = str(val).strip()
            cid = f"{idx}-{col}"
            requests.append({
                "custom_id": cid,
                "prompt": tmpl.format(topic=topic, text=text),
                "system": SYSTEM,
                "temperature": 0.0,
                "max_tokens": 6,
            })
    return requests


def _parse_score(text):
    if not text or text.startswith(("ERROR", "BATCH_ERROR")):
        return "ERROR"
    s = text.strip().replace('"', "").replace("'", "").replace(".", "").strip()
    for ch in s:
        if ch in "12345":
            return ch
    return "AMBIG"


def _jobs_dir(mode):
    return JOBS_DIR / f"score_{mode}"


def cmd_submit(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    reqs = _build_requests(df, args.mode)
    print(f"Built {len(reqs)} requests for mode={args.mode}")
    aliases = list_judges() if args.model == "all" else [args.model]
    jobs_dir = _jobs_dir(args.mode)
    for alias in aliases:
        if not supports_batch(alias):
            print(f"  [{alias}] no batch — use --stream"); continue
        meta_path = jobs_dir / f"{alias}.json"
        if meta_path.exists() and not args.force:
            print(f"  [{alias}] meta exists; --force to overwrite"); continue
        try:
            meta = submit_batch(alias, reqs)
            meta["task"] = f"score_{args.mode}"
            save_meta(meta, meta_path)
            print(f"  [{alias}] saved meta → {meta_path}")
        except Exception as exc:
            print(f"  [{alias}] SUBMIT FAILED: {type(exc).__name__}: {exc}")


def cmd_status(args):
    aliases = list_judges() if args.model == "all" else [args.model]
    jobs_dir = _jobs_dir(args.mode)
    for alias in aliases:
        mp = jobs_dir / f"{alias}.json"
        if not mp.exists(): print(f"  [{alias}] no meta"); continue
        meta = load_meta(mp)
        try:
            status = check_status(meta)
        except Exception as e:
            print(f"  [{alias}] status check failed: {e}"); continue
        save_meta(meta, mp)
        chunk_states = [ck.get("status") for ck in meta.get("chunks", [])]
        print(f"  [{alias}] overall={status}  chunks={chunk_states}")


def cmd_fetch(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    cols = _all_text_cols(df)
    aliases = list_judges() if args.model == "all" else [args.model]
    jobs_dir = _jobs_dir(args.mode)
    suffix = "" if args.mode == "own" else "_instruct"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for alias in aliases:
        mp = jobs_dir / f"{alias}.json"
        if not mp.exists(): print(f"  [{alias}] no meta"); continue
        meta = load_meta(mp)
        status = check_status(meta)
        if status != "completed":
            print(f"  [{alias}] not ready ({status})"); continue
        res = fetch_results(meta)
        rows = []
        for idx, row in df.iterrows():
            topic = str(row.get("topic", "") or "").strip() or str(row.get("theme", "")).strip()
            row_id = row.get("id", idx)
            for col in cols:
                val = row.get(col)
                if pd.isna(val) or not str(val).strip(): continue
                cid = f"{idx}-{col}"
                raw = res.get(cid, "")
                rows.append({
                    "id": row_id, "topic": topic, "argument_col": col,
                    "raw_answer": raw, "score": _parse_score(raw),
                })
        path = RESULTS_DIR / f"scoring_{alias}{suffix}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  [{alias}] saved → {path} ({len(rows)} rows)")


def cmd_stream(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    requests = _build_requests(df, args.mode)
    print(f"[{args.model}] streaming {len(requests)} requests (score_{args.mode})")

    def _do(r):
        ans = query_judge(args.model, r["prompt"], temperature=0.0,
                          system=r["system"], max_tokens=6)
        return r["custom_id"], ans

    suffix = "" if args.mode == "own" else "_instruct"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"scoring_{args.model}{suffix}.csv"

    by_cid = {r["custom_id"]: r for r in requests}
    raw_by_cid = {}
    last_save = time.time()
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futs = [pool.submit(_do, r) for r in requests]
        for i, fut in enumerate(as_completed(futs)):
            cid, raw = fut.result()
            raw_by_cid[cid] = raw
            if (i + 1) % 100 == 0:
                print(f"  [{args.model}] {i+1}/{len(requests)}", flush=True)
            if time.time() - last_save > 30:
                _save_score(raw_by_cid, df, out_csv)
                last_save = time.time()
    _save_score(raw_by_cid, df, out_csv)
    print(f"[{args.model}] saved → {out_csv}")


def _save_score(raw_by_cid, df, out_csv):
    cols = _all_text_cols(df)
    rows = []
    for idx, row in df.iterrows():
        topic = str(row.get("topic", "") or "").strip() or str(row.get("theme", "")).strip()
        row_id = row.get("id", idx)
        for col in cols:
            val = row.get(col)
            if pd.isna(val) or not str(val).strip(): continue
            cid = f"{idx}-{col}"
            if cid not in raw_by_cid: continue
            raw = raw_by_cid[cid]
            rows.append({
                "id": row_id, "topic": topic, "argument_col": col,
                "raw_answer": raw, "score": _parse_score(raw),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["own", "instruct"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--fetch",  action="store_true")
    g.add_argument("--stream", action="store_true")
    p.add_argument("--model", default="all")
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--first_n_rows", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--parallelism", type=int, default=6)
    args = p.parse_args()
    if args.submit: cmd_submit(args)
    elif args.status: cmd_status(args)
    elif args.fetch: cmd_fetch(args)
    elif args.stream: cmd_stream(args)


if __name__ == "__main__":
    main()
