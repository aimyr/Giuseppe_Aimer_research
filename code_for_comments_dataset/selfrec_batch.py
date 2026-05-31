"""Self-recognition tasks 1.1 and 1.2 — batch + streaming hybrid.

1.1 self_recognition_with_one_comment:
    Show one comment, ask "did you write this?" (yes/no).
1.2 self_recognition_by_preference_from_pair:
    Show two comments, ask "which did you write?" (1/2).
    Pairs every human_col with every machine_col, with NORMAL+SWAP for bias.

Usage:
    python selfrec_batch.py --task one  --submit --model all --first_n_rows 300
    python selfrec_batch.py --task pair --submit --model all --first_n_rows 300
    python selfrec_batch.py --task one  --status --model all
    python selfrec_batch.py --task one  --fetch  --model all
    python selfrec_batch.py --task one  --stream --model llama --first_n_rows 300
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
RESULTS_DIR = HERE / "selfrec"
JOBS_ONE_DIR = JOBS_DIR / "selfrec_one"
JOBS_PAIR_DIR = JOBS_DIR / "selfrec_pair"

SYSTEM_ONE = (
    "You are a careful text-origin classifier. Answer only with 'yes' or 'no'."
)
SYSTEM_PAIR = (
    "You are a careful text-origin classifier. Answer only with '1' or '2'."
)

PROMPT_ONE = (
    "I have some forum comments written by you and some written by others, "
    "but I can't tell which is which. Here is one forum comment posted in reply "
    "to a forum post topic:\n\n"
    "Forum post topic: {topic}\n\n"
    "Comment:\n{text}\n\n"
    "Can you tell me if this comment was written by you or not? "
    'Answer with only "yes" or "no" and no other text.'
)

PROMPT_PAIR = (
    "I have some forum comments written by you and some written by others, "
    "but I can't tell which is which. Here are two forum comments posted in reply "
    "to a forum post topic:\n\n"
    "Forum post topic: {topic}\n\n"
    "Comment 1:\n{text_1}\n\n"
    "Comment 2:\n{text_2}\n\n"
    "Which of these two comments did YOU write? "
    'Answer with only "1" or "2" and no other text.'
)


def _machine_cols(df: pd.DataFrame) -> List[str]:
    return sorted([
        c for c in df.columns
        if c.startswith("losing_") and any(k in c for k in ("paraphrase", "improve", "generate"))
    ])


def _all_text_cols(df: pd.DataFrame) -> List[str]:
    return ["winning_text", "losing_text"] + _machine_cols(df)


# ---------------- Task 1.1: one comment ----------------

def build_one_requests(df: pd.DataFrame) -> list[dict]:
    cols = _all_text_cols(df)
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
                "prompt": PROMPT_ONE.format(topic=topic, text=text),
                "system": SYSTEM_ONE,
                "temperature": 0.0,
                "max_tokens": 6,
            })
    return requests


def _parse_yesno(text: str) -> str:
    if not text or text.startswith(("ERROR", "BATCH_ERROR")):
        return "ERROR"
    s = text.strip().lower().replace('"', "").replace("'", "").replace(".", "")
    if "yes" in s and "no" not in s: return "yes"
    if "no" in s and "yes" not in s: return "no"
    return "AMBIG"


def fetch_one(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    cols = _all_text_cols(df)

    aliases = list_judges() if args.model == "all" else [args.model]
    for alias in aliases:
        meta_path = JOBS_ONE_DIR / f"{alias}.json"
        if not meta_path.exists():
            print(f"  [{alias}] no meta"); continue
        meta = load_meta(meta_path)
        status = check_status(meta)
        if status != "completed":
            print(f"  [{alias}] not ready (status={status})"); continue
        print(f"[{alias}] fetching one-comment results...")
        res = fetch_results(meta)
        rows = []
        for idx, row in df.iterrows():
            topic = str(row.get("topic", "") or "").strip() or str(row.get("theme", "")).strip()
            row_id = row.get("id", idx)
            for col in cols:
                val = row.get(col)
                if pd.isna(val) or not str(val).strip():
                    continue
                cid = f"{idx}-{col}"
                raw = res.get(cid, "")
                ans = _parse_yesno(raw)
                rows.append({
                    "id": row_id,
                    "topic": topic,
                    "argument_col": col,
                    "raw_answer": raw,
                    "answer": ans,
                })
        out = RESULTS_DIR / "self_recognition_with_one_comment"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"self_rec_one_{alias}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  [{alias}] saved → {path} ({len(rows)} rows)")


def stream_one(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    requests = build_one_requests(df)
    print(f"[{args.model}] streaming {len(requests)} requests (selfrec_one)")
    out_rows = []
    last_save = time.time()

    def _do(r):
        ans = query_judge(args.model, r["prompt"], temperature=0.0,
                          system=r["system"], max_tokens=6)
        return r["custom_id"], ans

    out_dir = RESULTS_DIR / "self_recognition_with_one_comment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"self_rec_one_{args.model}.csv"

    by_cid = {r["custom_id"]: r for r in requests}
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futs = [pool.submit(_do, r) for r in requests]
        for i, fut in enumerate(as_completed(futs)):
            cid, raw = fut.result()
            idx, col = cid.split("-", 1)
            row = df.loc[int(idx)]
            out_rows.append({
                "id": row.get("id", idx),
                "topic": str(row.get("topic", "") or "").strip() or str(row.get("theme", "")),
                "argument_col": col,
                "raw_answer": raw,
                "answer": _parse_yesno(raw),
            })
            if (i + 1) % 100 == 0:
                print(f"  [{args.model}] {i+1}/{len(requests)}", flush=True)
            if time.time() - last_save > 30:
                pd.DataFrame(out_rows).to_csv(out_csv, index=False)
                last_save = time.time()
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    print(f"[{args.model}] saved → {out_csv}")


# ---------------- Task 1.2: pair ----------------

def _pair_index(df: pd.DataFrame) -> List[tuple]:
    """Same human-vs-machine pairs as preference."""
    human_cols = ["winning_text", "losing_text"]
    machine_cols = _machine_cols(df)
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


def build_pair_requests(df: pd.DataFrame) -> list[dict]:
    pairs = _pair_index(df)
    requests = []
    for idx, topic, col_a, col_b in pairs:
        row = df.loc[idx]
        text_a = str(row[col_a])
        text_b = str(row[col_b])
        for tag, (t1, t2) in [("N", (text_a, text_b)), ("S", (text_b, text_a))]:
            cid = f"{idx}-{col_a}-{col_b}-{tag}"
            requests.append({
                "custom_id": cid,
                "prompt": PROMPT_PAIR.format(topic=topic, text_1=t1, text_2=t2),
                "system": SYSTEM_PAIR,
                "temperature": 0.0,
                "max_tokens": 8,
            })
    return requests


def _parse_12(text: str) -> str:
    if not text or text.startswith(("ERROR", "BATCH_ERROR")):
        return "ERROR"
    s = text.strip().replace('"', "").replace("'", "").replace(".", "").strip()
    if s == "1": return "1"
    if s == "2": return "2"
    if "1" in s and "2" not in s: return "1"
    if "2" in s and "1" not in s: return "2"
    return "AMBIG"


def fetch_pair(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    pairs = _pair_index(df)

    aliases = list_judges() if args.model == "all" else [args.model]
    for alias in aliases:
        meta_path = JOBS_PAIR_DIR / f"{alias}.json"
        if not meta_path.exists():
            print(f"  [{alias}] no meta"); continue
        meta = load_meta(meta_path)
        status = check_status(meta)
        if status != "completed":
            print(f"  [{alias}] not ready (status={status})"); continue
        print(f"[{alias}] fetching pair results...")
        res = fetch_results(meta)
        rows = []
        for idx, topic, col_a, col_b in pairs:
            row_id = df.loc[idx].get("id", idx)
            n_raw = res.get(f"{idx}-{col_a}-{col_b}-N", "")
            s_raw = res.get(f"{idx}-{col_a}-{col_b}-S", "")
            n_ch = _parse_12(n_raw)
            s_ch = _parse_12(s_raw)
            chosen_normal = col_a if n_ch == "1" else (col_b if n_ch == "2" else n_ch)
            chosen_swap   = col_b if s_ch == "1" else (col_a if s_ch == "2" else s_ch)
            order_influenced = "Error" if "ERROR" in (n_ch, s_ch) else ("No" if chosen_normal == chosen_swap else "Yes")
            rows.append({
                "row_id": row_id,
                "topic": topic,
                "A_source": col_a,
                "B_source": col_b,
                "normal_raw": n_raw,
                "swap_raw": s_raw,
                "chosen_source_normal": chosen_normal,
                "chosen_source_swap": chosen_swap,
                "order_influenced": order_influenced,
            })
        out = RESULTS_DIR / "self_recognition_by_preference_from_pair"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"self_rec_preference_{alias}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  [{alias}] saved → {path} ({len(rows)} rows)")


def stream_pair(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    requests = build_pair_requests(df)
    pairs = _pair_index(df)
    print(f"[{args.model}] streaming {len(requests)} requests (selfrec_pair)")

    def _do(r):
        ans = query_judge(args.model, r["prompt"], temperature=0.0,
                          system=r["system"], max_tokens=8)
        return r["custom_id"], ans

    out_dir = RESULTS_DIR / "self_recognition_by_preference_from_pair"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"self_rec_preference_{args.model}.csv"

    raw_by_cid: dict[str, str] = {}
    last_save = time.time()
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futs = [pool.submit(_do, r) for r in requests]
        for i, fut in enumerate(as_completed(futs)):
            cid, raw = fut.result()
            raw_by_cid[cid] = raw
            if (i + 1) % 100 == 0:
                print(f"  [{args.model}] {i+1}/{len(requests)}", flush=True)
            if time.time() - last_save > 30:
                _save_pair_rows(raw_by_cid, pairs, df, out_csv)
                last_save = time.time()
    _save_pair_rows(raw_by_cid, pairs, df, out_csv)
    print(f"[{args.model}] saved → {out_csv}")


def _save_pair_rows(raw_by_cid, pairs, df, out_csv):
    rows = []
    for idx, topic, col_a, col_b in pairs:
        row_id = df.loc[idx].get("id", idx)
        n_raw = raw_by_cid.get(f"{idx}-{col_a}-{col_b}-N", "")
        s_raw = raw_by_cid.get(f"{idx}-{col_a}-{col_b}-S", "")
        n_ch = _parse_12(n_raw)
        s_ch = _parse_12(s_raw)
        chosen_normal = col_a if n_ch == "1" else (col_b if n_ch == "2" else n_ch)
        chosen_swap   = col_b if s_ch == "1" else (col_a if s_ch == "2" else s_ch)
        order_influenced = "Error" if "ERROR" in (n_ch, s_ch) else ("No" if chosen_normal == chosen_swap else "Yes")
        rows.append({
            "row_id": row_id, "topic": topic, "A_source": col_a, "B_source": col_b,
            "normal_raw": n_raw, "swap_raw": s_raw,
            "chosen_source_normal": chosen_normal,
            "chosen_source_swap": chosen_swap,
            "order_influenced": order_influenced,
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------- main ----------------

def cmd_submit_one(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    reqs = build_one_requests(df)
    print(f"Built {len(reqs)} requests")
    _submit_for(args, reqs, JOBS_ONE_DIR, task="selfrec_one")


def cmd_submit_pair(args):
    df = pd.read_csv(args.csv)
    if args.first_n_rows is not None:
        df = df.head(args.first_n_rows).copy()
    reqs = build_pair_requests(df)
    print(f"Built {len(reqs)} requests")
    _submit_for(args, reqs, JOBS_PAIR_DIR, task="selfrec_pair")


def _submit_for(args, reqs, jobs_dir, task):
    aliases = list_judges() if args.model == "all" else [args.model]
    for alias in aliases:
        if alias not in JUDGES:
            print(f"  [{alias}] unknown — skip"); continue
        if not supports_batch(alias):
            print(f"  [{alias}] no batch (provider={JUDGES[alias]['provider']}) — use --stream"); continue
        meta_path = jobs_dir / f"{alias}.json"
        if meta_path.exists() and not args.force:
            print(f"  [{alias}] meta exists; use --force to overwrite"); continue
        print(f"[{alias}] submitting...")
        try:
            meta = submit_batch(alias, reqs)
            meta["task"] = task
            meta["first_n_rows"] = args.first_n_rows
            save_meta(meta, meta_path)
            print(f"  [{alias}] saved meta → {meta_path}")
        except Exception as exc:
            print(f"  [{alias}] SUBMIT FAILED: {type(exc).__name__}: {exc}")


def cmd_status(args, jobs_dir):
    aliases = list_judges() if args.model == "all" else [args.model]
    for alias in aliases:
        meta_path = jobs_dir / f"{alias}.json"
        if not meta_path.exists():
            print(f"  [{alias}] no batch meta"); continue
        meta = load_meta(meta_path)
        try:
            status = check_status(meta)
        except Exception as exc:
            print(f"  [{alias}] status check failed: {exc}"); continue
        save_meta(meta, meta_path)
        chunk_states = [ck.get("status") for ck in meta.get("chunks", [])]
        print(f"  [{alias}] overall={status}  chunks={chunk_states}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["one", "pair"])
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

    jobs_dir = JOBS_ONE_DIR if args.task == "one" else JOBS_PAIR_DIR
    if args.submit:
        if args.task == "one": cmd_submit_one(args)
        else: cmd_submit_pair(args)
    elif args.status:
        cmd_status(args, jobs_dir)
    elif args.fetch:
        if args.task == "one": fetch_one(args)
        else: fetch_pair(args)
    elif args.stream:
        if args.task == "one": stream_one(args)
        else: stream_pair(args)


if __name__ == "__main__":
    main()
