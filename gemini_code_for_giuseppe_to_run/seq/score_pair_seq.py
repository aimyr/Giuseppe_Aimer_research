import os
import time
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Optional, List, Dict
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyBDdGcKNAfLlWNDbRQyZsNFAkqKnHK9CIo"

# ---------------------------
# 1. Configuration
# ---------------------------
def configure_gemini(api_key_env: str = "GEMINI_API_KEY") -> None:
    """Configure the Gemini client."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"WARNING: {api_key_env} not set. Ensure you set it before running.")
    else:
        genai.configure(api_key=api_key)

# ---------------------------
# 2. Helper Functions
# ---------------------------
def _extract_text_from_response(response) -> str:
    """Safely extract text from Gemini response."""
    try:
        if response.text:
            return response.text.strip()
    except Exception:
        pass
    try:
        if response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
    except Exception:
        pass
    return ""

def get_chosen_source(answer: str, source_first: str, source_second: str) -> str:
    """
    Maps the model's answer ('1' or '2') to the actual column name.
    """
    # Clean up quotes or periods (e.g., "1." -> "1")
    clean_ans = answer.replace('"', '').replace("'", "").replace(".", "").strip()

    if clean_ans == "1":
        return source_first
    elif clean_ans == "2":
        return source_second

    # Fallback if model is chatty but contains the number clearly
    if "1" in clean_ans and "2" not in clean_ans:
        return source_first
    elif "2" in clean_ans and "1" not in clean_ans:
        return source_second

    return "Error"

# ---------------------------
# 3. Query Function (Ranking / Persuasion)
# ---------------------------
def query_pairwise_ranking_gemini(
    proposition: str,
    argument1: str,
    argument2: str,
    model_name: str,
    temperature: float
) -> str:
    """
    Asks Gemini to choose the more persuasive argument.
    Returns '1' or '2'.
    """
    # Disable safety filters so debate topics don't get blocked
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        candidate_count=1,
    )

    model = genai.GenerativeModel(model_name)

    # Prompt: Persuasion / Ranking
    user_prompt = (
        "You are an expert research evaluator. I will provide research proposition and two abstracts. "
        "Your task is to decide which abstract is more persuasive, logical, and better structured.\n\n"
        f"Proposition: {proposition}\n\n"
        f"Abstract 1:\n{argument1}\n\n"
        f"Abstract 2:\n{argument2}\n\n"
        "Which abstract is more persuasive? "
        "Please answer with ONLY the number \"1\" or \"2\" and no other text."
    )

    # Retry loop
    for attempt in range(3):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            text = _extract_text_from_response(response)

            # Basic validation
            if "1" in text and "2" not in text: return "1"
            if "2" in text and "1" not in text: return "2"
            if text.strip() in ["1", "2"]: return text.strip()

            if text:
                return text[:20] # Return truncated text if ambiguous

            time.sleep(1)

        except Exception as e:
            if attempt == 2: return "Error"
            time.sleep(1)

    return "Error"

# ---------------------------
# 4. Main Logic: Swapped Comparison
# ---------------------------
def process_gemini_ranking_swapped(
    csv_path: str,
    output_path: str,
    num_rows: Optional[int] = None,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0
) -> None:

    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    if num_rows is not None:
        df = df.head(num_rows)

    # 1. Identify Machine Columns Dynamically
    machine_cols = [
        c for c in df.columns
        if any(x in c for x in ['paraphrase', 'improve', 'generate'])
    ]
    machine_cols.sort()

    results = []
    print(f"Processing {len(df)} rows. Comparing Humans against {len(machine_cols)} machine variants.")

    for idx, row in df.iterrows():
        row_id = row.get("id")
        proposition = str(row.get("theme", "")).strip()

        print(f"Processing Row {idx+1}/{len(df)} (ID: {row_id})")

        # 2. Define Comparison Pairs
        comparison_pairs = []

        # A. Human vs Human (Winning vs Losing)
        comparison_pairs.append(('winning_text', 'losing_text'))

        # B. Winning Human vs Machines
        for m_col in machine_cols:
            comparison_pairs.append(('winning_text', m_col))

        # C. Losing Human vs Machines
        for m_col in machine_cols:
            comparison_pairs.append(('losing_text', m_col))

        # 3. Process pairs
        for col_a, col_b in comparison_pairs:
            val_a = row.get(col_a)
            val_b = row.get(col_b)

            if pd.isna(val_a) or pd.isna(val_b):
                continue
            text_a = str(val_a).strip()
            text_b = str(val_b).strip()
            if not text_a or not text_b:
                continue

            # Rename for output (winning_text -> winning_human)
            out_source_a = col_a.replace("_text", "_human")
            out_source_b = col_b.replace("_text", "_human")

            # --- Query 1: Direct Order (A vs B) ---
            # Arg 1 = A, Arg 2 = B
            ans_direct = query_pairwise_ranking_gemini(proposition, text_a, text_b, model_name, temperature)
            chosen_direct = get_chosen_source(ans_direct, out_source_a, out_source_b)

            time.sleep(1.0) # Rate limit safety

            # --- Query 2: Swapped Order (B vs A) ---
            # Arg 1 = B, Arg 2 = A
            ans_swapped = query_pairwise_ranking_gemini(proposition, text_b, text_a, model_name, temperature)
            # If answer is "1", it chose B. If "2", it chose A.
            chosen_swapped = get_chosen_source(ans_swapped, out_source_b, out_source_a)

            time.sleep(1.0)

            # --- Determine Influence ---
            if chosen_direct == "Error" or chosen_swapped == "Error":
                influenced = "Error"
            else:
                # If chosen sources are identical, position didn't matter
                influenced = "No" if chosen_direct == chosen_swapped else "Yes"

            results.append({
                "row_id": row_id,
                "A_source": out_source_a,
                "B_source": out_source_b,
                "first_final_answer": ans_direct,
                "swapped_final_answer": ans_swapped,
                "chosen_source_direct": chosen_direct,
                "chosen_source_reversed": chosen_swapped,
                "order_influenced_decision": influenced
            })

    # Save to CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Done! Results saved to {output_path}")

# ---------------------------
# 5. Execution
# ---------------------------
if __name__ == "__main__":
    # 1. Setup API
    os.environ["GEMINI_API_KEY"] = "AIzaSyBDdGcKNAfLlWNDbRQyZsNFAkqKnHK9CIo"
    configure_gemini()

    # 2. Paths
    CSV_PATH = 'level1_merged.csv'
    OUTPUT_PATH = 'score_pairwise_gemini.csv'


    # 3. Run
    process_gemini_ranking_swapped(
        csv_path=CSV_PATH,
        output_path=OUTPUT_PATH,
        num_rows=None, # Change to None to process the whole dataset
        model_name="gemini-2.5-flash",
        temperature=0.0 # Low temp for deterministic evaluation
    )