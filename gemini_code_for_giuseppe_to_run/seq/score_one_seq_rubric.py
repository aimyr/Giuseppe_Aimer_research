# ===========================
# FULL FIXED SCRIPT (Colab-ready)
# Fixes:
# - removes api_version='v1' bug for Gemini Developer API
# - uses enum structured output correctly
# - resumes only rows with valid scores 1..5
# - re-scores Blocked/Error/missing rows
# - stores debug metadata (raw_text, block_reason, finish_reason)
# ===========================

# !pip -q install -U google-genai pandas tqdm

import os
import time
import random
import pandas as pd
from tqdm import tqdm
from enum import Enum

from google import genai
from google.genai import types

os.environ["GEMINI_API_KEY"] = "AIzaSyBDdGcKNAfLlWNDbRQyZsNFAkqKnHK9CIo"
# ===========================
# 0) REMOVE PROXIES
# ===========================
def disable_proxies():
    proxy_keys = [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ]
    found = {k: os.environ.get(k) for k in proxy_keys if os.environ.get(k)}
    if found:
        print("⚠️ Proxy env vars found:")
        for k, v in found.items():
            print(f"  {k}={v}")
    for k in proxy_keys:
        os.environ.pop(k, None)

    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

disable_proxies()


# ===========================
# 1) CONFIG
# ===========================
# Put your key in Colab secrets or env var before running:
# os.environ["GOOGLE_API_KEY"] = "YOUR_NEW_KEY"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not set.\n"
        "Set it with:\n"
        "os.environ['GOOGLE_API_KEY'] = 'YOUR_NEW_KEY'"
    )

MODEL_NAME = "gemini-2.5-flash"

RUBRIC_FILES = [
    "RUBRIC_INT_gemini_1.csv",
    "RUBRIC_INT_gpt4omini_1.csv",
    "RUBRIC_INT_gpt5nano_1.csv",
    "RUBRIC_INT_lama1.csv",
    "RUBRIC_INT_mistral1.csv",
    "RUBRIC_INT_qwen1.csv",
    "RUBRIC_INT_sonnet_1.csv",
]

DATA_CSV = "level1_merged.csv"
OUTPUT_DIR = "rubric_gemini"

MAX_RETRIES = 5
BASE_BACKOFF_SEC = 2.0
JITTER_SEC = 0.8
SUCCESS_DELAY_SEC = 0.35
SAVE_EVERY = 200
REQUEST_TIMEOUT_MS = 90_000
MAX_ARGUMENT_CHARS = 12000
THINKING_BUDGET = 0
# ===========================
# 2) CLIENT
# IMPORTANT:
# - For Gemini Developer API, do NOT force api_version="v1" here.
# - Default beta endpoint supports preview features used below.
# ===========================
client = genai.Client(api_key=GOOGLE_API_KEY)

SYSTEM_INST = (
    "You are an expert research judge.\n"
    "Read the rubric carefully and return only the best score label."
)


class ScoreEnum(Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"


# Optional safety settings.
# Usually not required, but okay to keep explicit.
SAFETY_ALLOW_ALL = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_NONE"),
]


# ===========================
# 3) HELPERS
# ===========================
def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def atomic_to_csv(df: pd.DataFrame, path: str):
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def load_rubric_string(rubric_path: str) -> str:
    df = pd.read_csv(rubric_path)

    desc_col = None
    for col in df.columns:
        if col.lower() != "score":
            desc_col = col
            break

    if not desc_col:
        raise ValueError(f"Could not identify rubric description column in {rubric_path}")

    rubric_text = "SCORING RUBRIC:\n"
    for _, row in df.iterrows():
        text = str(row[desc_col]) if pd.notna(row[desc_col]) else ""
        rubric_text += f"{text}\n\n"

    return rubric_text.strip()


def build_target_columns(df: pd.DataFrame):
    target_columns = ["winning_text", "losing_text"]
    machine_cols = [
        c for c in df.columns
        if any(x in c.lower() for x in ["paraphrase", "improve", "generate"])
    ]
    target_columns.extend(machine_cols)
    return target_columns


def is_valid_numeric_score(x) -> bool:
    try:
        v = int(x)
        return 1 <= v <= 5
    except Exception:
        return False


def load_existing_results(save_path: str):
    """
    Returns:
      old_df
      done_good_keys: keys with valid numeric score 1..5 only
    """
    if not os.path.exists(save_path):
        return None, set()

    old = pd.read_csv(save_path)

    if "score" not in old.columns:
        return old, set()

    old["_score_num"] = pd.to_numeric(old["score"], errors="coerce")
    good_mask = old["_score_num"].between(1, 5, inclusive="both")

    if {"id", "column_name", "rubric_source"}.issubset(old.columns):
        done_good_keys = set(
            zip(
                old.loc[good_mask, "id"],
                old.loc[good_mask, "column_name"],
                old.loc[good_mask, "rubric_source"],
            )
        )
    else:
        done_good_keys = set(
            zip(
                old.loc[good_mask, "id"],
                old.loc[good_mask, "source"],
                old.loc[good_mask, "rubric_source"],
            )
        )

    old.drop(columns=["_score_num"], inplace=True, errors="ignore")
    return old, done_good_keys


def merge_and_dedupe(old_df: pd.DataFrame, new_df: pd.DataFrame, key_cols=("id", "column_name", "rubric_source")):
    """
    Prefer:
    1) numeric score 1..5
    2) non-empty score
    3) newest row
    """
    if old_df is None and (new_df is None or new_df.empty):
        return pd.DataFrame()

    if old_df is None:
        combined = new_df.copy()
    elif new_df is None or new_df.empty:
        combined = old_df.copy()
    else:
        combined = pd.concat([old_df, new_df], ignore_index=True)

    if combined.empty:
        return combined

    for c in key_cols:
        if c not in combined.columns:
            if c == "column_name" and "source" in combined.columns:
                combined["column_name"] = combined["source"]
            else:
                raise ValueError(f"Missing required key column: {c}")

    combined["_order"] = range(len(combined))
    combined["_score_num"] = pd.to_numeric(combined.get("score", pd.Series([None] * len(combined))), errors="coerce")

    score_str = combined.get("score", "").astype(str)
    combined["_quality"] = 0
    combined.loc[score_str.fillna("").str.strip().ne(""), "_quality"] = 1
    combined.loc[combined["_score_num"].between(1, 5, inclusive="both"), "_quality"] = 2

    combined = combined.sort_values(list(key_cols) + ["_quality", "_order"])
    combined = combined.drop_duplicates(subset=list(key_cols), keep="last")
    combined.drop(columns=["_order", "_score_num", "_quality"], inplace=True, errors="ignore")

    return combined


def build_prompt(rubric_text: str, proposition: str, argument: str) -> str:
    arg = argument[:MAX_ARGUMENT_CHARS]
    return (
        f"{rubric_text}\n"
        "---------------------------------------------------\n"
        "TASK: Rate the following research abstract paraphrase using the rubric above.\n"
        "Choose exactly one score from the allowed labels: 1, 2, 3, 4, 5.\n\n"
        f"Proposition: {proposition}\n"
        f"Abstract: {arg}\n"
    )


def safe_response_text(resp) -> str:
    """
    Avoid crashing on response.text when there are no valid text parts.
    """
    try:
        txt = getattr(resp, "text", None)
        return (txt or "").strip()
    except Exception:
        pass

    # manual fallback
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = []
        for part in parts:
            t = getattr(part, "text", None)
            if t:
                chunks.append(t)
        if chunks:
            return "".join(chunks).strip()

    return ""


def get_prompt_block_reason(resp) -> str:
    pf = getattr(resp, "prompt_feedback", None)
    if not pf:
        return ""
    reason = getattr(pf, "block_reason", None)
    msg = getattr(pf, "block_reason_message", None)
    if reason:
        return f"{reason}" + (f" | {msg}" if msg else "")
    return ""


def get_candidate_finish_reason(resp) -> str:
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return ""
    c0 = candidates[0]
    fr = getattr(c0, "finish_reason", None)
    fm = getattr(c0, "finish_message", None)
    if fr:
        return f"{fr}" + (f" | {fm}" if fm else "")
    return ""


def get_safety_details(resp) -> str:
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return ""
    c0 = candidates[0]
    ratings = getattr(c0, "safety_ratings", None) or []
    out = []
    for r in ratings:
        category = getattr(r, "category", None)
        prob = getattr(r, "probability", None)
        blocked = getattr(r, "blocked", None)
        out.append(f"{category}:{prob}:blocked={blocked}")
    return "; ".join(out)


def parse_score_from_response(resp):
    """
    Returns int 1..5 or None
    """
    txt = safe_response_text(resp)
    if txt in {"1", "2", "3", "4", "5"}:
        return int(txt)

    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, Enum):
        val = parsed.value
        if val in {"1", "2", "3", "4", "5"}:
            return int(val)

    if isinstance(parsed, str) and parsed in {"1", "2", "3", "4", "5"}:
        return int(parsed)

    return None


def query_score(proposition: str, argument: str, rubric_text: str):
    """
    Returns dict:
      {
        "score": int 1..5 or diagnostic string,
        "raw_text": str,
        "prompt_block_reason": str,
        "finish_reason": str,
        "safety_details": str,
      }
    """
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INST,
        temperature=0.0,
        candidate_count=1,
        response_mime_type="text/x.enum",
        response_schema=ScoreEnum,
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        safety_settings=SAFETY_ALLOW_ALL,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )

    last_err = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=build_prompt(rubric_text, proposition, argument),
                config=config,
            )

            raw_text = safe_response_text(resp)
            prompt_block_reason = get_prompt_block_reason(resp)
            finish_reason = get_candidate_finish_reason(resp)
            safety_details = get_safety_details(resp)

            score_num = parse_score_from_response(resp)
            if score_num is not None:
                return {
                    "score": score_num,
                    "raw_text": raw_text,
                    "prompt_block_reason": prompt_block_reason,
                    "finish_reason": finish_reason,
                    "safety_details": safety_details,
                }

            if prompt_block_reason:
                return {
                    "score": f"PromptBlocked: {prompt_block_reason}",
                    "raw_text": raw_text,
                    "prompt_block_reason": prompt_block_reason,
                    "finish_reason": finish_reason,
                    "safety_details": safety_details,
                }

            if finish_reason:
                return {
                    "score": f"CandidateFinish: {finish_reason}",
                    "raw_text": raw_text,
                    "prompt_block_reason": prompt_block_reason,
                    "finish_reason": finish_reason,
                    "safety_details": safety_details,
                }

            if not raw_text:
                return {
                    "score": "Blocked_or_Empty",
                    "raw_text": "",
                    "prompt_block_reason": prompt_block_reason,
                    "finish_reason": finish_reason,
                    "safety_details": safety_details,
                }

            return {
                "score": f"Unparsed: {raw_text[:120]}",
                "raw_text": raw_text,
                "prompt_block_reason": prompt_block_reason,
                "finish_reason": finish_reason,
                "safety_details": safety_details,
            }

        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                return {
                    "score": f"Error: {type(e).__name__}: {e}",
                    "raw_text": "",
                    "prompt_block_reason": "",
                    "finish_reason": "",
                    "safety_details": "",
                }

            wait = (BASE_BACKOFF_SEC ** (attempt + 1)) + random.uniform(0, JITTER_SEC)
            print(f"⚠️ API Error (Attempt {attempt+1}/{MAX_RETRIES}): {e} | retry in {wait:.1f}s")
            time.sleep(wait)

    return {
        "score": f"Error: {last_err}",
        "raw_text": "",
        "prompt_block_reason": "",
        "finish_reason": "",
        "safety_details": "",
    }


# ===========================
# 4) CORE LOOP
# ===========================
def process_single_rubric(main_df: pd.DataFrame, rubric_path: str):
    rubric_name = os.path.basename(rubric_path)
    print(f"\n>>> PROCESSING RUBRIC: {rubric_name}")

    rubric_text = load_rubric_string(rubric_path)
    target_columns = build_target_columns(main_df)

    save_name = f"scores_{MODEL_NAME.replace('.', '_')}_{rubric_name}"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    old_df, done_good_keys = load_existing_results(save_path)
    if old_df is not None:
        print(f"🔁 Found existing file: {save_path}")
        print(f"✅ Already valid-scored items (will be skipped): {len(done_good_keys)}")

    results = []
    op_count = 0
    total_ops_est = len(main_df) * len(target_columns)
    pbar = tqdm(total=total_ops_est, desc=rubric_name, unit="call")

    for _, row in main_df.iterrows():
        debate_id = row.get("id")
        theme = str(row.get("theme", "")).strip()

        for col in target_columns:
            pbar.update(1)

            if col not in main_df.columns:
                continue

            cell = row.get(col)
            if pd.isna(cell):
                continue

            arg_text = str(cell).strip()
            if not arg_text:
                continue

            source_type = col.replace("_text", "_human")
            key = (debate_id, col, rubric_name)

            if key in done_good_keys:
                continue

            result = query_score(theme, arg_text, rubric_text)

            results.append({
                "id": debate_id,
                "theme": theme,
                "source": source_type,
                "column_name": col,
                "rubric_source": rubric_name,
                "model_used": MODEL_NAME,
                "argument_preview": (arg_text[:200] + "...") if len(arg_text) > 200 else arg_text,
                "score": result["score"],
                "raw_text": result["raw_text"],
                "prompt_block_reason": result["prompt_block_reason"],
                "finish_reason": result["finish_reason"],
                "safety_details": result["safety_details"],
            })

            op_count += 1
            time.sleep(SUCCESS_DELAY_SEC)

            if op_count % SAVE_EVERY == 0:
                safe_mkdir(OUTPUT_DIR)
                new_df = pd.DataFrame(results)
                combined = merge_and_dedupe(old_df, new_df, key_cols=("id", "column_name", "rubric_source"))
                atomic_to_csv(combined, save_path)
                print(f"💾 Checkpoint saved: {save_path} (+{len(results)} new rows)")

                old_df = combined
                old_df["_score_num"] = pd.to_numeric(old_df["score"], errors="coerce")
                good_mask = old_df["_score_num"].between(1, 5, inclusive="both")
                done_good_keys = set(
                    zip(
                        old_df.loc[good_mask, "id"],
                        old_df.loc[good_mask, "column_name"],
                        old_df.loc[good_mask, "rubric_source"],
                    )
                )
                old_df.drop(columns=["_score_num"], inplace=True, errors="ignore")
                results = []

    pbar.close()

    safe_mkdir(OUTPUT_DIR)
    new_df = pd.DataFrame(results)
    combined = merge_and_dedupe(old_df, new_df, key_cols=("id", "column_name", "rubric_source"))
    atomic_to_csv(combined, save_path)
    print(f"✅ Finished {rubric_name}. Saved to: {save_path}")

    return save_path


# ===========================
# 5) MAIN
# ===========================
def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Data file not found: {DATA_CSV}")

    safe_mkdir(OUTPUT_DIR)

    main_df = pd.read_csv(DATA_CSV)
    print(f"Loaded data: {main_df.shape[0]} rows, {main_df.shape[1]} columns")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Model: {MODEL_NAME}")

    for r_file in RUBRIC_FILES:
        if not os.path.exists(r_file):
            print(f"❌ Missing rubric file, skipping: {r_file}")
            continue
        process_single_rubric(main_df, r_file)

    print("\nAll rubrics processed.")


main()
