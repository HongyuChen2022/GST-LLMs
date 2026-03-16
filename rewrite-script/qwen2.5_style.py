import os
import time
import re
import pandas as pd
from openai import OpenAI

# --- Config ----------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN")
BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

IN_PATH = "datasets/pastel_gender_pair_500.csv"
OUT_PATH = "./datasets/qwen2.5_PGS_500.csv"

SLEEP_S = 0.1
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512
SEED = 42

# Checkpoints / resume
CHECKPOINT_EVERY = 25

# Force start (set ONE, or set both None to auto-resume)
START_FROM_PRINT_ROW = 382   # "Processing 382/500" corresponds to index 381
START_FROM_INDEX = None      # 0-based loop index

# Overwrite rows from start point onward (recommended when rerunning a crashed region)
OVERWRITE_FROM_START = True

# Retry behavior
RETRY_N = 6
RETRY_BACKOFF_S = 0.5
ON_FINAL_FAIL = "skip"  # "skip" continues; "raise" stops

if not API_KEY:
    raise RuntimeError("Missing HF_TOKEN env var (or API_KEY). Set it to your Hugging Face token.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Prompt ----------------------------------------------------------------

SYSTEM_PROMPT = """
You are generating a parallel dataset. For each input, create two rewrites that preserve the original meaning exactly, but differ in linguistic style.

Style targets:
Feminine Style Tendencies (linguistic) and Masculine Style Tendencies (linguistic).

Hard constraints (must follow):
- Preserve meaning and facts exactly: entities, numbers, dates, locations, claims, intent, sentiment strength, tense, and modality.
- Do not add new information, examples, justifications, or opinions.
- Do not introduce stereotypes or demographic assumptions. Style is purely linguistic, not “gender roles.”
- Keep both rewrites natural and fluent.
- Keep length within ±15% unless impossible.
- If the input is too short to restyle naturally, still try, but do not pad with new content.

Output format (strict):
Return EXACTLY two lines, no extra text:
FEMININE: <rewrite>
MASCULINE: <rewrite>

IMPORTANT:
- Do NOT output placeholders like "<rewrite>", "<text>", "[rewrite]", "(rewrite)".
- The content after the labels must be the actual rewritten text.
""".strip()

# --- Parsing + validation helpers ------------------------------------------

PLACEHOLDER_PAT = re.compile(
    r"(?i)^\s*(<\s*rewrite\s*>|<\s*text\s*>|\[\s*rewrite\s*\]|\(\s*rewrite\s*\))\s*$"
)

def is_placeholder(s: str) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    if not t:
        return True
    return bool(PLACEHOLDER_PAT.match(t))

def parse_two_line_output(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
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

def validate_pair(fem: str, masc: str):
    if is_placeholder(fem) or is_placeholder(masc):
        raise ValueError("Placeholder returned.")
    if not fem.strip() or not masc.strip():
        raise ValueError("Empty rewrite returned.")

# --- Repair helper ----------------------------------------------------------

def repair_format(raw_output: str) -> tuple[str, str]:
    repair_messages = [
        {
            "role": "system",
            "content": (
                "Reformat into EXACTLY two lines. "
                "Do not output placeholders like <rewrite> or <text>."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return EXACTLY two lines, no extra text:\n"
                "FEMININE: <actual rewrite>\n"
                "MASCULINE: <actual rewrite>\n\n"
                "If the content below is missing labels or has extra text, fix it.\n\n"
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
    fixed = resp.choices[0].message.content or ""
    fem, masc = parse_two_line_output(fixed)
    validate_pair(fem, masc)
    return fem, masc

# --- LLM call with retries --------------------------------------------------

def generate_style_pair(reference_text: str) -> tuple[str, str]:
    user_prompt = f'Text: """{reference_text}"""'

    last_content = ""
    last_err = None

    for attempt in range(1, RETRY_N + 1):
        try:
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
            last_content = content

            fem, masc = parse_two_line_output(content)
            validate_pair(fem, masc)
            return fem, masc

        except Exception as e:
            last_err = e
            time.sleep(RETRY_BACKOFF_S * attempt)

    # deterministic repair as a last resort
    try:
        return repair_format(last_content)
    except Exception as e:
        last_err = e

    if ON_FINAL_FAIL == "skip":
        return "", ""
    raise RuntimeError(f"Failed after retries. Last error: {last_err}")

# --- Resume logic that works with partial OUT_PATH --------------------------

def load_or_build_scaffold(in_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Always returns a full-length output df (same length as input).
    If OUT_PATH exists but is shorter, it copies existing rows into the scaffold by row order.
    """
    total = len(in_df)

    scaffold = pd.DataFrame({
        "reference_text": in_df[text_col].astype(str).tolist(),
        "feminine_style": [""] * total,
        "masculine_style": [""] * total,
    })

    if os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        n = min(len(old), total)

        if "reference_text" in old.columns:
            scaffold.loc[: n - 1, "reference_text"] = old.loc[: n - 1, "reference_text"].astype(str).tolist()
        if "feminine_style" in old.columns:
            scaffold.loc[: n - 1, "feminine_style"] = old.loc[: n - 1, "feminine_style"].astype(str).tolist()
        if "masculine_style" in old.columns:
            scaffold.loc[: n - 1, "masculine_style"] = old.loc[: n - 1, "masculine_style"].astype(str).tolist()

        print(f"[RESUME] Loaded {n} existing rows from {OUT_PATH} into a {total}-row scaffold.")
    else:
        print(f"[INIT] No existing {OUT_PATH}. Starting fresh with a {total}-row scaffold.")

    return scaffold

def first_empty_index(df: pd.DataFrame) -> int:
    for i in range(len(df)):
        f = str(df.at[i, "feminine_style"])
        m = str(df.at[i, "masculine_style"])
        if (not f.strip()) or (not m.strip()) or is_placeholder(f) or is_placeholder(m):
            return i
    return len(df)

# --- Main ------------------------------------------------------------------

def main():
    in_df = pd.read_csv(IN_PATH)
    if "input.y" in in_df.columns:
        text_col = "input.y"
    else:
        raise ValueError("Input CSV must contain 'input.y'")

    total = len(in_df)
    out_df = load_or_build_scaffold(in_df, text_col)

    # Decide start index
    if START_FROM_INDEX is not None and START_FROM_PRINT_ROW is not None:
        raise ValueError("Set only one of START_FROM_INDEX or START_FROM_PRINT_ROW.")

    if START_FROM_INDEX is not None:
        start_idx = int(START_FROM_INDEX)
        print(f"[FORCE] Starting at index i={start_idx} (prints as Processing {start_idx + 1}/{total}).")
    elif START_FROM_PRINT_ROW is not None:
        start_idx = max(0, int(START_FROM_PRINT_ROW) - 1)
        print(f"[FORCE] Starting at Processing {start_idx + 1}/{total} (index i={start_idx}).")
    else:
        start_idx = first_empty_index(out_df)
        print(f"[AUTO] First incomplete row is Processing {start_idx + 1}/{total} (index i={start_idx}).")

    processed_since_ckpt = 0

    for i in range(start_idx, total):
        if not OVERWRITE_FROM_START:
            f_existing = str(out_df.at[i, "feminine_style"])
            m_existing = str(out_df.at[i, "masculine_style"])
            if f_existing.strip() and m_existing.strip() and (not is_placeholder(f_existing)) and (not is_placeholder(m_existing)):
                continue

        ref_text = str(in_df.at[i, text_col])
        print(f"Processing {i + 1}/{total}...", end="\r", flush=True)

        fem, masc = generate_style_pair(ref_text)
        out_df.at[i, "reference_text"] = ref_text
        out_df.at[i, "feminine_style"] = fem
        out_df.at[i, "masculine_style"] = masc

        processed_since_ckpt += 1
        if CHECKPOINT_EVERY and processed_since_ckpt >= CHECKPOINT_EVERY:
            out_df.to_csv(OUT_PATH, index=False)
            processed_since_ckpt = 0

        time.sleep(SLEEP_S)

    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nDone. Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()