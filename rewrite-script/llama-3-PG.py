import os
import time
import re
import pandas as pd
from openai import OpenAI

# --- Config ----------------------------------------------------------------

# Prefer env var; fall back to the hardcoded token if present.
# In shell: export HF_TOKEN="hf_..."
API_KEY = ""
BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

IN_PATH = "datasets/pastel_gender_pair_500.csv"
OUT_PATH = "./datasets/deepseek_PG_no_tendency_500.csv"

SLEEP_S = 0.1
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512
SEED = 42

if not API_KEY:
    raise RuntimeError("Missing HF_TOKEN env var (or API_KEY). Set it to your Hugging Face token.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Prompt ----------------------------------------------------------------

SYSTEM_PROMPT = """
You are generating a parallel dataset. For each input, create two rewrites that preserve the original meaning exactly, but differ in authorial voice.

Voice targets:
FEMALE-author voice tendencies and MALE-author voice tendencies

Hard constraints (must follow):
- Preserve meaning and facts exactly: entities, numbers, dates, locations, claims, intent, sentiment strength, tense, and modality.
- Do not add new information, examples, justifications, or opinions.
- Do not introduce stereotypes or demographic assumptions. Style is purely linguistic, not “gender roles.”
- Keep both rewrites natural and fluent.
- Keep length within ±15% unless impossible.
- If the input is too short to restyle naturally, still try, but do not pad with new content.

Output format (strict):
Return EXACTLY two lines, no extra text:
FEMALE: <rewrite>
MALE: <rewrite>
""".strip()


# --- Parsing helpers --------------------------------------------------------

def parse_two_line_output(text: str) -> tuple[str, str]:
    """
    Extract FEMALE and MALE rewrites from the model response.
    Tolerates minor formatting issues (extra whitespace, different casing).
    Also accepts legacy labels FEMININE/MASCULINE as fallback.
    """
    raw = (text or "").strip()

    # Remove code fences if model adds them
    raw = re.sub(r"^```.*?\n", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\n```$", "", raw.strip())

    female = None
    male = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        m1 = re.match(r"(?i)^(female|female_author|female-author)\s*:\s*(.+)$", line)
        if m1 and female is None:
            female = m1.group(2).strip()
            continue

        m2 = re.match(r"(?i)^(male|male_author|male-author)\s*:\s*(.+)$", line)
        if m2 and male is None:
            male = m2.group(2).strip()
            continue

        # Legacy fallback
        m3 = re.match(r"(?i)^feminine\s*:\s*(.+)$", line)
        if m3 and female is None:
            female = m3.group(1).strip()
            continue

        m4 = re.match(r"(?i)^masculine\s*:\s*(.+)$", line)
        if m4 and male is None:
            male = m4.group(1).strip()
            continue

    # Regex fallback across lines
    if female is None:
        m = re.search(r"(?is)(female|female_author|female-author|feminine)\s*:\s*(.+?)(?:\n|$)", raw)
        if m:
            female = m.group(2).strip()

    if male is None:
        m = re.search(r"(?is)(male|male_author|male-author|masculine)\s*:\s*(.+?)(?:\n|$)", raw)
        if m:
            male = m.group(2).strip()

    if not female or not male:
        raise ValueError(f"Could not parse FEMALE/MALE.\nRAW OUTPUT:\n{raw[:900]}")

    return female, male


def repair_format(raw_output: str) -> tuple[str, str]:
    """
    Repair step: ask the model to reformat its previous output into the exact two-line format.
    """
    repair_messages = [
        {"role": "system", "content": "Reformat text. Follow the output format exactly."},
        {
            "role": "user",
            "content": (
                "Rewrite the content below into EXACTLY two lines, no extra text:\n"
                "FEMALE: <text>\n"
                "MALE: <text>\n\n"
                f"CONTENT:\n<<<{raw_output}>>>"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=repair_messages,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        seed=SEED,
    )
    fixed = resp.choices[0].message.content
    return parse_two_line_output(fixed)


# --- LLM call --------------------------------------------------------------

def generate_author_pair(reference_text: str) -> tuple[str, str]:
    user_prompt = f'Text: """{reference_text}"""'

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
    )

    content = resp.choices[0].message.content or ""
    try:
        return parse_two_line_output(content)
    except Exception:
        return repair_format(content)


# --- Main ------------------------------------------------------------------

def main():
    df = pd.read_csv(IN_PATH)
    #df = df.iloc[411:501, :].reset_index(drop=True)

    # pick the text column
    if "input.y" in df.columns:
        text_col = "input.y"
 #   elif "reference_text" in df.columns:
      # text_col = "reference_text"
    else:
        raise ValueError("Input CSV must contain 'input.y'")

    total = len(df)

    # Ensure columns exist at the end
    female_col = "female_author_text"
    male_col = "male_author_text"
    if female_col not in df.columns:
        df[female_col] = ""
    if male_col not in df.columns:
        df[male_col] = ""

    for i in range(total):
        ref_text = str(df.at[i, text_col])
        print(f"Processing {i + 1}/{total}...", end="\r", flush=True)

        try:
            female, male = generate_author_pair(ref_text)
        except Exception as e:
            print(f"\nError on row {i + 1}: {e}")
            # save partial progress (with whatever has been filled so far)
            df.to_csv(OUT_PATH, index=False)
            print(f"Partial results saved to {OUT_PATH}")
            raise

        df.at[i, female_col] = female
        df.at[i, male_col] = male

        time.sleep(SLEEP_S)

        # Optional: checkpoint every 50 rows (uncomment if you want)
        # if (i + 1) % 50 == 0:
        #     df.to_csv(OUT_PATH, index=False)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nDone. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
