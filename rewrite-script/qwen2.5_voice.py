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
OUT_PATH = "./datasets/Qwen2.5_PG_500.csv"

SLEEP_S = 0.1
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512
SEED = 42

# Checkpoints / resume
CHECKPOINT_EVERY = 25

# Force start (set ONE; set both None to auto-resume)
START_FROM_PRINT_ROW = None   # e.g. 382 (matches "Processing 382/500")
START_FROM_INDEX = None       # 0-based i
OVERWRITE_FROM_START = False  # set True to redo rows from the start point onward

# Retry behavior
RETRY_N = 6
RETRY_BACKOFF_S = 0.5
ON_FINAL_FAIL = "skip"  # "skip" continues with empty strings; "raise" stops

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
    """
    Extract FEMALE and MALE rewrites from the model response.
    Tolerates minor formatting issues (extra whitespace, different casing).
    Also accepts legacy labels FEMININE/MASCULINE as fallback.
    """
    raw = (text or "").strip()

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

def validate_pair(female: str, male: str):
    if is_placeholder(female) or is_placeholder(male):
        raise ValueError("Placeholder returned.")
    if not str(female).strip() or not str(male).strip():
        raise ValueError("Empty rewrite returned.")

# --- Repair helper ----------------------------------------------------------

def repair_format(raw_output: str) -> tuple[str, str]:
    """
    Ask the model to reformat into exact 2-line output (no placeholders).
    """
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
    fixed = resp.choices[0].message.content or ""
    female, male = parse_two_line_output(fixed)
    validate_pair(female, male)
    return female, male

# --- LLM call with retries --------------------------------------------------

def generate_author_pair(reference_text: str) -> tuple[str, str]:
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

            female, male = parse_two_line_output(content)
            validate_pair(female, male)
            return female, male

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

# --- Resume logic (works with partial OUT_PATH) -----------------------------

def load_or_build_scaffold(in_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Always returns a full-length output df (same length as input).
    If OUT_PATH exists but is shorter, it copies existing rows into the scaffold by row order.
    """
    total = len(in_df)

    scaffold = pd.DataFrame({
        text_col: in_df[text_col].astype(str).tolist(),  # keep original column too
        "female_author_text": [""] * total,
        "male_author_text": [""] * total,
    })

    if os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        n = min(len(old), total)

        # Preserve original text column if present in old
        if text_col in old.columns:
            scaffold.loc[: n - 1, text_col] = old.loc[: n - 1, text_col].astype(str).tolist()

        if "female_author_text" in old.columns:
            scaffold.loc[: n - 1, "female_author_text"] = old.loc[: n - 1, "female_author_text"].astype(str).tolist()
        if "male_author_text" in old.columns:
            scaffold.loc[: n - 1, "male_author_text"] = old.loc[: n - 1, "male_author_text"].astype(str).tolist()

        # If old uses different column names, try to map them
        if "woman_author_text" in old.columns and "female_author_text" not in old.columns:
            scaffold.loc[: n - 1, "female_author_text"] = old.loc[: n - 1, "woman_author_text"].astype(str).tolist()
        if "man_author_text" in old.columns and "male_author_text" not in old.columns:
            scaffold.loc[: n - 1, "male_author_text"] = old.loc[: n - 1, "man_author_text"].astype(str).tolist()

        print(f"[RESUME] Loaded {n} existing rows from {OUT_PATH} into a {total}-row scaffold.")
    else:
        print(f"[INIT] No existing {OUT_PATH}. Starting fresh with a {total}-row scaffold.")

    return scaffold

def first_incomplete_index(df: pd.DataFrame) -> int:
    for i in range(len(df)):
        f = str(df.at[i, "female_author_text"])
        m = str(df.at[i, "male_author_text"])
        if (not f.strip()) or (not m.strip()) or is_placeholder(f) or is_placeholder(m):
            return i
    return len(df)

# --- Main ------------------------------------------------------------------

def main():
    df_in = pd.read_csv(IN_PATH)

    if "input.y" in df_in.columns:
        text_col = "input.y"
    else:
        raise ValueError("Input CSV must contain 'input.y'")

    total = len(df_in)

    # Always work with a full-length scaffold so indices match input
    df_out = load_or_build_scaffold(df_in, text_col)

    # Decide start index
    if START_FROM_INDEX is not None and START_FROM_PRINT_ROW is not None:
        raise ValueError("Set only one of START_FROM_INDEX or START_FROM_PRINT_ROW.")

    if START_FROM_INDEX is not None:
        start_idx = int(START_FROM_INDEX)
        print(f"[FORCE] Starting at index i={start_idx} (prints Processing {start_idx + 1}/{total}).")
    elif START_FROM_PRINT_ROW is not None:
        start_idx = max(0, int(START_FROM_PRINT_ROW) - 1)
        print(f"[FORCE] Starting at Processing {start_idx + 1}/{total} (index i={start_idx}).")
    else:
        start_idx = first_incomplete_index(df_out)
        print(f"[AUTO] First incomplete row is Processing {start_idx + 1}/{total} (index i={start_idx}).")

    processed_since_ckpt = 0

    for i in range(start_idx, total):
        if not OVERWRITE_FROM_START:
            f_existing = str(df_out.at[i, "female_author_text"])
            m_existing = str(df_out.at[i, "male_author_text"])
            if f_existing.strip() and m_existing.strip() and (not is_placeholder(f_existing)) and (not is_placeholder(m_existing)):
                continue

        ref_text = str(df_in.at[i, text_col])
        print(f"Processing {i + 1}/{total}...", end="\r", flush=True)

        try:
            female, male = generate_author_pair(ref_text)
        except Exception as e:
            print(f"\nError on row {i + 1}: {e}")
            df_out.to_csv(OUT_PATH, index=False)
            print(f"Partial results saved to {OUT_PATH}")
            raise

        df_out.at[i, text_col] = ref_text
        df_out.at[i, "female_author_text"] = female
        df_out.at[i, "male_author_text"] = male

        processed_since_ckpt += 1
        if CHECKPOINT_EVERY and processed_since_ckpt >= CHECKPOINT_EVERY:
            df_out.to_csv(OUT_PATH, index=False)
            processed_since_ckpt = 0

        time.sleep(SLEEP_S)

    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nDone. Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()