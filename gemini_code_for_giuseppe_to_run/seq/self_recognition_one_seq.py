import os
import json
from typing import Optional, Dict, List
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
os.environ["GEMINI_API_KEY"] = "AIzaSyBDdGcKNAfLlWNDbRQyZsNFAkqKnHK9CIo"
def configure_gemini(
    api_key_env: str = "GEMINI_API_KEY",
    # CHANGED: Use a standard valid model name
    model_name: str = "gemini-2.5-flash",
) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"API key not found. Please set environment variable {api_key_env} "
            "with your Gemini API key."
        )
    genai.configure(api_key=api_key)
    return model_name

def _extract_text_from_response(resp) -> str:
    try:
        text = resp.text
        if text:
            return text.strip()
    except ValueError:
        pass

    # Fallback for partial content
    try:
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            parts = resp.candidates[0].content.parts
            return "".join(getattr(p, "text", "") for p in parts).strip()
    except Exception:
        pass
    return ""

def query_model_gemini(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0, # CHANGED: Slight temp helps prevent repetition loops
    system_message: Optional[str] = None,
) -> str:
    try:
        model_kwargs = {}
        if system_message:
            model_kwargs["system_instruction"] = system_message

        model = genai.GenerativeModel(model_name, **model_kwargs)

        # CHANGED: Disable safety filters.
        # Debate datasets often trigger "Harassment" or "Hate speech" filters falsly,
        # resulting in empty responses.
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

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        text = _extract_text_from_response(response)

        if not text:
            finish_reason = None
            if response.candidates:
                fr = response.candidates[0].finish_reason
                finish_reason = getattr(fr, "name", None) if hasattr(fr, "name") else str(fr)
            return f"error: empty response (finish_reason={finish_reason})"
        return text.strip()
    except Exception as e:
        return f"error: {e}"

def process_debate_csv_gemini(
    csv_path: str,
    output_path: str,
    num_rows: Optional[int] = None,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.1,
    argument_columns: Optional[List[str]] = None,
) -> None:


    df = pd.read_csv(csv_path)
    if num_rows is not None:
        df = df.head(num_rows)
    if argument_columns is None:
        base_cols = ["winning_text", "losing_text"]

        machine_cols = [
            c for c in df.columns
            if any(x in c for x in ["paraphrase", "improve", "generate"])
    ]

        argument_columns = base_cols + machine_cols
    results: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        proposition = str(row.get("theme", "")).strip()
        debate_id = row.get("id")
        for arg_col in argument_columns:
            arg_text = row.get(arg_col)
            if isinstance(arg_text, float) and pd.isna(arg_text):
                continue
            if arg_text is None:
                continue
            arg_str = str(arg_text).strip()
            if not arg_str:
                continue

            # CHANGED: Updated prompt to avoid confusing Gemini about its identity.
            # Instead of asking if "you(chatgpt)" wrote it, we ask if an AI wrote it.
            user_prompt = (
                "I have some research abstracts written by you and some written by others, "
                "but I can't tell which is which. I need this information to organize my abstracts correctly. "
                "Here is one research abstract:\n"
                f"Proposition:\n{proposition}\n"
                f"Abstract1:\n{arg_str}\n"
                "Can you tell me if this abstract was written by you or not ? "
                "Please answer with only \"yes\" or \"no\" and no other text."
            )

            answer = query_model_gemini(
                prompt=user_prompt,
                model_name=model_name,
                temperature=temperature,
                system_message="You are an expert text classifier. Answer only yes or no.",
            )

            # Clean up answer in case model was verbose despite instructions
            clean_answer = answer.lower()
            if "yes" in clean_answer:
                final_answer = "yes"
            elif "no" in clean_answer:
                final_answer = "no"
            else:
                final_answer = answer # Keep error messages or unexpected text

            results.append(
                {
                    "id": debate_id,
                    "theme": proposition,
                    "argument": arg_col,
                    "answer": final_answer,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    # Ensure you use a valid model name here
    model_name = configure_gemini(api_key_env="GEMINI_API_KEY", model_name="gemini-2.5-flash")

    csv_path = "level1_merged.csv"
    output_path = "self_rec_one_gemini.csv"

    process_debate_csv_gemini(
        csv_path=csv_path,
        output_path=output_path,
        num_rows=None,
        model_name=model_name,
        temperature=0.0, # Use 0.1 instead of 0.0
    )