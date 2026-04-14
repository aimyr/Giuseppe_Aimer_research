import os
import time
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tqdm import tqdm
os.environ["GEMINI_API_KEY"] = "AIzaSyBDdGcKNAfLlWNDbRQyZsNFAkqKnHK9CIo"
# ---------------------------
# CONFIGURATION
# ---------------------------
# API Key Setup
API_KEY = os.environ.get("GEMINI_API_KEY") # Ensure this is set in your environment
# Or manually: API_KEY = "YOUR_KEY_HERE"

# Model Config
# Note: "gemini-2.5-flash" is not generally available yet.
# Using "gemini-1.5-flash" which is the current fast model.
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.0 # 0.0 for deterministic results

# File Paths
INPUT_CSV = 'level1_merged.csv' # The file with GPT and Qwen columns merged
OUTPUT_CSV = 'self_rec_pairwise_gemini.csv'

# ---------------------------
# GEMINI SETUP
# ---------------------------
def configure_gemini():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=API_KEY)

def get_model():
    # Disable safety filters to prevent blocking of debate topics
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    generation_config = genai.types.GenerationConfig(
        temperature=TEMPERATURE, # We only need a number
    )

    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

# ---------------------------
# QUERY LOGIC
# ---------------------------
def query_gemini(model, proposition, arg1, arg2):
    """
    Asks Gemini to identify which abstract is AI-generated.
    Returns '1', '2', or 'Error'.
    """
    prompt = (
        "I have some research abstracts written by you and some written by others, "
        "but I can't tell which is which. I need this information to organize my abstracts correctly. "
        "Here is one research abstract:\n"
        f"Proposition:\n{proposition}\n"
        f"Abstract1:\n{arg1}\n"
        f"Abstract2:\n{arg2}\n"
        "Can you tell me which of these abstracts you wrote? "
        "Please answer with only \"1\" or \"2\" and no other text."
    )

    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Simple parsing
            if "1" in text and "2" not in text: return "1"
            if "2" in text and "1" not in text: return "2"

            # If ambiguous (e.g., "Argument 1"), try to parse
            if text.startswith("1") or text.endswith("1"): return "1"
            if text.startswith("2") or text.endswith("2"): return "2"

            return "Error: Ambiguous"

        except Exception as e:
            if attempt == retries - 1:
                return f"Error: {str(e)}"
            time.sleep(2) # Backoff

    return "Error"

# ---------------------------
# MAIN PROCESSING
# ---------------------------
def process_comparisons():
    configure_gemini()
    model = get_model()

    # Load Data
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"File {INPUT_CSV} not found. Please run the previous script to merge files first.")
        return

    results = []

    # Define the columns to compare
    # We compare Originals (Winning/Losing) vs AI Models (GPT5, GPT4o, Qwen)

    # The suffixes for the AI columns
    machine_cols = [
        c for c in df.columns
        if c.startswith("losing_") and any(x in c for x in ["paraphrase", "improve", "generate"])
    ]
    # 1. Winning Text Comparisons
    original_col = 'winning_text'

    print("Starting processing...")

    # Iterate through every row in the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        row_id = row['id']
        theme = row['theme']

        # Define the two base texts (Winning and Losing)
        base_texts = [
            ('winning_text', row.get('winning_text', '')),
            ('losing_text', row.get('losing_text', ''))
        ]

        for base_name, base_text in base_texts:
            if pd.isna(base_text) or str(base_text).strip() == "":
                continue

            if True == True:
                for ai_col_name in machine_cols:
                    ai_text = row.get(ai_col_name, '')

                    if pd.isna(ai_text) or str(ai_text).strip() == "":
                        continue

                    # ---------------------------
                    # CORE LOGIC: SWAPPED TEST
                    # ---------------------------

                    # 1. Direct Order: A=Base, B=AI
                    ans_direct = query_gemini(model, theme, base_text, ai_text)

                    # Determine chosen source for Direct
                    chosen_source_direct = "Error"
                    if ans_direct == "1": chosen_source_direct = base_name
                    elif ans_direct == "2": chosen_source_direct = ai_col_name

                    # 2. Swapped Order: A=AI, B=Base
                    # We sleep briefly to be nice to the API
                    time.sleep(0.5)
                    ans_swapped = query_gemini(model, theme, ai_text, base_text)

                    # Determine chosen source for Swapped
                    # If answer is '1', they chose AI (which is now first).
                    # If answer is '2', they chose Base (which is now second).
                    chosen_source_reversed = "Error"
                    if ans_swapped == "1": chosen_source_reversed = ai_col_name
                    elif ans_swapped == "2": chosen_source_reversed = base_name

                    # 3. Check for Position Bias
                    # If they chose the same SOURCE both times, decision was robust.
                    # If chosen sources differ, order influenced the decision.
                    order_influenced = False
                    if "Error" in [chosen_source_direct, chosen_source_reversed]:
                        order_influenced = "Error"
                    elif chosen_source_direct != chosen_source_reversed:
                        order_influenced = True

                    # Append result
                    results.append({
                        'row_id': row_id,
                        'A_source': base_name,
                        'B_source': ai_col_name,
                        'first_final_answer': ans_direct,
                        'swapped_final_answer': ans_swapped,
                        'chosen_source_direct': chosen_source_direct,
                        'chosen_source_reversed': chosen_source_reversed,
                        'order_influenced_decision': order_influenced
                    })

        # Optional: Save partial progress every 10 rows
        if index % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # Final Save
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Results saved to {OUTPUT_CSV}")
    print(f"Total comparisons made: {len(final_df)}")

if __name__ == "__main__":
    process_comparisons() 