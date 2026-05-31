"""Score 1-5 with rubric written by another model — task 2.3.

For each (judge_model × rubric_model) pair, the judge rates every comment
using the rubric written by rubric_model. Output saved as
ranking/one_comment_scoring_with_rubric_instructions/rubric_<rubric_model>/
scores_<judge>_<rubric_model>.csv

Usage:
    python score_rubric_batch.py --submit --judge gpt4omini --rubric all --first_n_rows 300
    python score_rubric_batch.py --submit --judge all --rubric all --first_n_rows 300
    python score_rubric_batch.py --status --judge all --rubric all
    python score_rubric_batch.py --fetch  --judge all --rubric all
    python score_rubric_batch.py --stream --judge llama --rubric all --first_n_rows 300
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
RUBRICS_DIR = HERE / "rubrics"
RESULTS_DIR = HERE / "ranking" / "one_comment_scoring_with_rubric_instructions"
JOBS_DIR_R = JOBS_DIR / "score_rubric"

SYSTEM = (
    "You are an expert forum community moderator. Use the rubric to assign a score. "
    "Your output MUST be exactly one character: the digit 1, 2, 3, 4, or 5. "
    "No words. No punctuation. No explanation. Just the single digit."
)

PROMPT = (
    "{rubric_text}\n\n"
    "---\n"
    "Rate the following forum comment using the rubric above. Output ONLY one digit (1-5).\n\n"
    "Forum post topic:\n{topic}\n\n"
    "Comment:\n{text}\n\n"
    "Score (1-5):"
)


def load_rubric(rubric_alias: str) -> str:
    path = RUBRICS_DIR / f"RUBRIC_{rubric_alias}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing rubric: {path}. Run rubric_gen.py first.")
    pieces = ["SCORING RUBRIC:"]
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pieces.append(f"Score {row['score']}: {row['description']}")
    return "\n".join(pieces)


def _all_text_cols(df):
    machine = sorted([
        c for c in df.columns
        if c.startswith("losing_") and any(k in c for k in ("paraphrase", "improve", "generate"))
    ])
    return ["winning_text", "losing_text"] + machine


def _build_requests(df, rubric_text):
    cols = _all_text_cols(df)
    requests = []
    for idx, row in df.iterrows():
        topic = str(row.get("topic", "") or "").strip() or str(row.get("theme", "")).strip()
        for col in cols:
            val = row.get(col)
            if pd.isna(val) or not str(val).strip(): continue
            text = str(val).strip()
            cid = f"{idx}-{col}"
            requests.append({
                "custom_id": cid,
                "prompt": PROMPT.format(rubric_text=rubric_text, topic=topic, text=text),
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


def _pair_jobs_path(judge, rubric):
    return JOBS_DIR_R / f"{judge}__{rubric}.json"


def _result_path(judge, rubric):
    sub = RESULTS_DIR / f"rubric_{rubric}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"scores_{judge}_{rubric}.csv"


def cmd_submit(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    judges = list_judges() if args.judge == "all" else [args.judge]
    rubrics = list_judges() if args.rubric == "all" else [args.rubric]
    for rb in rubrics:
        try:
            rubric_text = load_rubric(rb)
        except FileNotFoundError as e:
            print(f"  rubric {rb}: {e}"); continue
        reqs = _build_requests(df, rubric_text)
        print(f"rubric={rb}: {len(reqs)} requests per judge")
        for ja in judges:
            if not supports_batch(ja):
                print(f"  [{ja} × {rb}] no batch — use --stream"); continue
            mp = _pair_jobs_path(ja, rb)
            if mp.exists() and not args.force:
                print(f"  [{ja} × {rb}] meta exists; --force"); continue
            try:
                meta = submit_batch(ja, reqs)
                meta["task"] = f"score_rubric_{rb}"
                save_meta(meta, mp)
                print(f"  [{ja} × {rb}] saved meta")
            except Exception as exc:
                print(f"  [{ja} × {rb}] SUBMIT FAILED: {type(exc).__name__}: {exc}")


def cmd_status(args):
    judges = list_judges() if args.judge == "all" else [args.judge]
    rubrics = list_judges() if args.rubric == "all" else [args.rubric]
    for ja in judges:
        for rb in rubrics:
            mp = _pair_jobs_path(ja, rb)
            if not mp.exists(): continue
            meta = load_meta(mp)
            try:
                st = check_status(meta)
            except Exception as e:
                print(f"  [{ja} × {rb}] status check failed: {e}"); continue
            save_meta(meta, mp)
            print(f"  [{ja} × {rb}] {st}")


def cmd_fetch(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    cols = _all_text_cols(df)
    judges = list_judges() if args.judge == "all" else [args.judge]
    rubrics = list_judges() if args.rubric == "all" else [args.rubric]
    for ja in judges:
        for rb in rubrics:
            mp = _pair_jobs_path(ja, rb)
            if not mp.exists(): continue
            meta = load_meta(mp)
            st = check_status(meta)
            if st != "completed":
                print(f"  [{ja} × {rb}] not ready ({st})"); continue
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
                        "rubric_source": rb, "judge_model": ja,
                        "raw_answer": raw, "score": _parse_score(raw),
                    })
            out = _result_path(ja, rb)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  [{ja} × {rb}] saved → {out} ({len(rows)} rows)")


def cmd_stream(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    cols = _all_text_cols(df)
    rubrics = list_judges() if args.rubric == "all" else [args.rubric]
    for rb in rubrics:
        try:
            rubric_text = load_rubric(rb)
        except FileNotFoundError as e:
            print(f"  rubric {rb}: {e}"); continue
        reqs = _build_requests(df, rubric_text)
        print(f"[{args.judge} × {rb}] streaming {len(reqs)} requests")

        def _do(r):
            ans = query_judge(args.judge, r["prompt"], temperature=0.0,
                              system=r["system"], max_tokens=6)
            return r["custom_id"], ans

        raw_by_cid = {}
        last_save = time.time()
        out_csv = _result_path(args.judge, rb)

        with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
            futs = [pool.submit(_do, r) for r in reqs]
            for i, fut in enumerate(as_completed(futs)):
                cid, raw = fut.result()
                raw_by_cid[cid] = raw
                if (i + 1) % 100 == 0:
                    print(f"  [{args.judge} × {rb}] {i+1}/{len(reqs)}", flush=True)
                if time.time() - last_save > 30:
                    _save_rows(raw_by_cid, df, args.judge, rb, out_csv)
                    last_save = time.time()
        _save_rows(raw_by_cid, df, args.judge, rb, out_csv)
        print(f"[{args.judge} × {rb}] saved → {out_csv}")


def _save_rows(raw_by_cid, df, judge, rubric, out_csv):
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
                "rubric_source": rubric, "judge_model": judge,
                "raw_answer": raw, "score": _parse_score(raw),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--fetch",  action="store_true")
    g.add_argument("--stream", action="store_true")
    p.add_argument("--judge",  default="all")
    p.add_argument("--rubric", default="all")
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--first_n_rows", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--parallelism", type=int, default=6)
    args = p.parse_args()
    if args.submit: cmd_submit(args)
    elif args.status: cmd_status(args)
    elif args.fetch:  cmd_fetch(args)
    elif args.stream: cmd_stream(args)


if __name__ == "__main__":
    main()
