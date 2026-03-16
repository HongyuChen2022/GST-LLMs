import os
import time
import re
import pandas as pd
from openai import OpenAI

# --- Config ----------------------------------------------------------------

# Set in shell:
# export HF_TOKEN="hf_..."
API_KEY = ""
BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

IN_PATH = "datasets/pastel_gender_pair_500.csv"
OUT_PATH = "./datasets/qwen2.5_PGS_with_instructions_500.csv"

SLEEP_S = 0.1
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512
SEED = 42

if not API_KEY:
    raise RuntimeError("Missing HF_TOKEN env var. Set it to your Hugging Face token.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Prompt ----------------------------------------------------------------

SYSTEM_PROMPT = """
You are generating a parallel dataset. For each input, create two rewrites that preserve the original meaning exactly, but differ in linguistic style.

Hard constraints (must follow):
- Preserve meaning and facts exactly: entities, numbers, dates, locations, claims, intent, sentiment strength, tense, and modality.
- Do not add new information, examples, justifications, or opinions.
- Do not introduce stereotypes or demographic assumptions. Style is purely linguistic, not “gender roles.”
- Keep both rewrites natural and fluent.
- Keep length within ±15% unless impossible.
- If the input is too short to restyle naturally, still try, but do not pad with new content.

Style targets:
Feminine Style Tendencies (linguistic):
- More emotional/relational framing and empathy markers (without changing sentiment strength)
- More collaborative tone: inclusive language (we/our/let’s when consistent), hedges (maybe/perhaps/I think)
- More descriptive language: adjectives/adverbs and sensory/aesthetic framing ONLY if already implied (do not invent facts)
- Slightly more complex/flowing sentences (subordinate clauses, narrative flow)

Masculine Style Tendencies (linguistic):
- More fact-focused framing: logic/structure/problem-solution framing (without adding data)
- More direct/assertive phrasing (but keep the same level of certainty as the original)
- More concise language: shorter, to-the-point sentences with minimal elaboration
- More action-oriented verbs and goal-driven framing (without changing intent)

Output format (strict):
Return EXACTLY two lines, no extra text:
FEMININE: <rewrite>
MASCULINE: <rewrite>
""".strip()


# --- Parsing helpers --------------------------------------------------------

def parse_two_line_output(text: str) -> tuple[str, str]:
    """
    Extract FEMININE and MASCULINE rewrites from the model response.
    Tolerates minor formatting issues (extra whitespace, different casing).
    """
    raw = (text or "").strip()

    # Remove code fences if model adds them
    raw = re.sub(r"^```.*?\n", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\n```$", "", raw.strip())

    fem = None
    masc = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        m1 = re.match(r"(?i)^feminine\s*:\s*(.+)$", line)
        if m1 and fem is None:
            fem = m1.group(1).strip()
            continue

        m2 = re.match(r"(?i)^masculine\s*:\s*(.+)$", line)
        if m2 and masc is None:
            masc = m2.group(1).strip()
            continue

    # Fallback: try to find in the whole text with regex across lines
    if fem is None:
        m = re.search(r"(?is)feminine\s*:\s*(.+?)(?:\n|$)", raw)
        if m:
            fem = m.group(1).strip()

    if masc is None:
        m = re.search(r"(?is)masculine\s*:\s*(.+?)(?:\n|$)", raw)
        if m:
            masc = m.group(1).strip()

    if not fem or not masc:
        raise ValueError(f"Could not parse FEMININE/MASCULINE.\nRAW OUTPUT:\n{raw[:900]}")

    return fem, masc


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
                "FEMININE: <text>\n"
                "MASCULINE: <text>\n\n"
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

def generate_style_pair(reference_text: str) -> tuple[str, str]:
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
        # one-shot repair
        return repair_format(content)


# --- Main ------------------------------------------------------------------

def main():
    df = pd.read_csv(IN_PATH)

    # pick the text column
    if "input.y" in df.columns:
        text_col = "input.y"
    #elif "reference_text" in df.columns:
    #text_col = "reference_text"
    #elif "short_text" in df.columns:
      #text_col = "short_text"
    else:
        raise ValueError("Input CSV must contain 'input.y'")

    results = []
    total = len(df)

    for i, row in df.iterrows():
        ref_text = str(row[text_col])
        print(f"Processing {i + 1}/{total}...", end="\r", flush=True)

        try:
            fem, masc = generate_style_pair(ref_text)
        except Exception as e:
            print(f"\nError on row {i + 1}: {e}")
            # save partial
            pd.DataFrame(results).to_csv(OUT_PATH, index=False)
            print(f"Partial results saved to {OUT_PATH}")
            raise

        results.append({
            "reference_text": ref_text,
            "feminine_style": fem,
            "masculine_style": masc,
        })

        time.sleep(SLEEP_S)

    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"\nDone. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
