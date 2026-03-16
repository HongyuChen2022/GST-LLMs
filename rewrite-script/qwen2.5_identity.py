import os
import time
import re
import pandas as pd
from openai import OpenAI

# --- Config ----------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") 
BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

IN_PATH = "datasets/pastel_gender_pair_500.csv"
OUT_PATH = "./datasets/qwen3_identity_500.csv"

SLEEP_S = 0.1
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512
SEED = 42

RETRY_N = 6
RETRY_BACKOFF_S = 0.5
ON_FINAL_FAIL = "skip"  # "raise" or "skip"

RESUME_FROM_OUT_IF_EXISTS = True
ONLY_FILL_BAD_ROWS = True
CHECKPOINT_EVERY = 25

if not API_KEY:
    raise RuntimeError("Missing HF_TOKEN env var (or API_KEY). Set it to your Hugging Face token.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Prompt ----------------------------------------------------------------

SYSTEM_PROMPT = """
You are generating a parallel dataset. For each input, create two rewrites that preserve the original meaning exactly, but differ in authorial identity.

Identity targets:
author who identifies as a woman and author who identifies as a man

Hard constraints (must follow):
- Preserve meaning and facts exactly: entities, numbers, dates, locations, claims, intent, sentiment strength, tense, and modality.
- Do not add new information, examples, justifications, or opinions.
- Do not introduce stereotypes or demographic assumptions. Style is purely linguistic, not “gender roles.”
- Keep both rewrites natural and fluent.
- Keep length within ±15% unless impossible.
- If the input is too short to restyle naturally, still try, but do not pad with new content.

Language constraint (must follow):
- Output MUST be in English only. Do not output any Chinese (or any non-English) characters.

Output format (strict):
Return EXACTLY two lines, no extra text:
WOMAN: <rewrite>
MAN: <rewrite>

IMPORTANT:
- Do NOT output placeholders like "<rewrite>", "<text>", "[rewrite]", "(rewrite)".
- The content after "WOMAN:" and "MAN:" must be the actual rewritten text.
""".strip()

# --- Parsing + validation helpers ------------------------------------------

PLACEHOLDER_PAT = re.compile(
    r"(?i)^\s*(<\s*rewrite\s*>|<\s*text\s*>|\[\s*rewrite\s*\]|\(\s*rewrite\s*\))\s*$"
)

# CJK Unified Ideographs + common CJK blocks (good enough to catch Chinese output)
CJK_PAT = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F\u3040-\u30FF\uAC00-\uD7AF]")

def is_placeholder(s: str) -> bool:
    if s is None:
        return True
    t = s.strip()
    if not t:
        return True
    return bool(PLACEHOLDER_PAT.match(t))

def contains_cjk(s: str) -> bool:
    return bool(CJK_PAT.search(s or ""))

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _looks_emptyish(s: str) -> bool:
    if not s or not s.strip():
        return True
    stripped = re.sub(r"[\W_]+", "", s, flags=re.UNICODE)
    return len(stripped.strip()) == 0

def parse_two_line_output(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    raw = re.sub(r"^```.*?\n", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\n```$", "", raw.strip())

    woman = None
    man = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        m1 = re.match(r"(?i)^(woman|female_author|woman-author)\s*:\s*(.+)$", line)
        if m1 and woman is None:
            woman = m1.group(2).strip()
            continue

        m2 = re.match(r"(?i)^(man|male_author|man-author)\s*:\s*(.+)$", line)
        if m2 and man is None:
            man = m2.group(2).strip()
            continue

        m3 = re.match(r"(?i)^feminine\s*:\s*(.+)$", line)
        if m3 and woman is None:
            woman = m3.group(1).strip()
            continue

        m4 = re.match(r"(?i)^masculine\s*:\s*(.+)$", line)
        if m4 and man is None:
            man = m4.group(1).strip()
            continue

    if woman is None:
        m = re.search(r"(?is)(woman|female_author|woman-author|feminine)\s*:\s*(.+?)(?:\n|$)", raw)
        if m:
            woman = m.group(2).strip()

    if man is None:
        m = re.search(r"(?is)(man|male_author|man-author|masculine)\s*:\s*(.+?)(?:\n|$)", raw)
        if m:
            man = m.group(2).strip()

    if woman is None or man is None:
        raise ValueError(f"Could not parse WOMAN/MAN.\nRAW OUTPUT:\n{raw[:900]}")

    return woman, man

def validate_pair(reference_text: str, woman: str, man: str):
    if is_placeholder(woman) or is_placeholder(man):
        raise ValueError("Placeholder '<rewrite>' returned.")

    if _looks_emptyish(woman) or _looks_emptyish(man):
        raise ValueError("Empty/meaningless rewrite(s).")

    if contains_cjk(woman) or contains_cjk(man):
        raise ValueError("Non-English (CJK) characters detected.")

    ref = _norm(reference_text)
    w = _norm(woman)
    m = _norm(man)

    if w == ref or m == ref:
        raise ValueError("Verbatim copy returned for at least one rewrite.")

    if len(woman.strip()) < 4 or len(man.strip()) < 4:
        raise ValueError("Rewrite too short (likely failed generation).")

# --- Repair helper ----------------------------------------------------------

def repair_format(raw_output: str) -> tuple[str, str]:
    repair_messages = [
        {
            "role": "system",
            "content": (
                "Return EXACTLY two lines and ENGLISH ONLY. "
                "Do not output placeholders like <rewrite> or <text>. "
                "No Chinese or non-English characters."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return EXACTLY two lines, no extra text:\n"
                "WOMAN: <actual English rewrite>\n"
                "MAN: <actual English rewrite>\n\n"
                "If any line is Chinese/non-English, rewrite it into English while preserving meaning exactly.\n\n"
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
    woman, man = parse_two_line_output(fixed)
    if is_placeholder(woman) or is_placeholder(man) or contains_cjk(woman) or contains_cjk(man):
        raise ValueError("Repair produced placeholder or non-English output.")
    return woman, man

# --- LLM call --------------------------------------------------------------

def generate_author_pair(reference_text: str) -> tuple[str, str]:
    user_prompt = f'Text: """{reference_text}"""'

    last_err = None
    last_content = ""

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

            woman, man = parse_two_line_output(content)
            validate_pair(reference_text, woman, man)
            return woman, man

        except Exception as e:
            last_err = e
            time.sleep(RETRY_BACKOFF_S * attempt)
            continue

    try:
        woman, man = repair_format(last_content)
        validate_pair(reference_text, woman, man)
        return woman, man
    except Exception as e:
        last_err = e

    if ON_FINAL_FAIL == "skip":
        return "", ""
    raise RuntimeError(f"Failed after {RETRY_N} retries. Last error: {last_err}")

# --- Main ------------------------------------------------------------------

def row_needs_fill(df: pd.DataFrame, i: int, woman_col: str, man_col: str) -> bool:
    w = str(df.at[i, woman_col]) if woman_col in df.columns else ""
    m = str(df.at[i, man_col]) if man_col in df.columns else ""
    if not w.strip() or not m.strip():
        return True
    if is_placeholder(w) or is_placeholder(m):
        return True
    if contains_cjk(w) or contains_cjk(m):
        return True
    return False

def main():
    if RESUME_FROM_OUT_IF_EXISTS and os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
    else:
        df = pd.read_csv(IN_PATH)

    if "input.y" in df.columns:
        text_col = "input.y"
    else:
        raise ValueError("Input CSV must contain 'input.y'")

    woman_col = "woman_author_text"
    man_col = "man_author_text"
    if woman_col not in df.columns:
        df[woman_col] = ""
    if man_col not in df.columns:
        df[man_col] = ""

    fail_col = "gen_failed"
    if fail_col not in df.columns:
        df[fail_col] = 0

    total = len(df)

    for i in range(total):
        if ONLY_FILL_BAD_ROWS and not row_needs_fill(df, i, woman_col, man_col):
            continue

        ref_text = str(df.at[i, text_col])
        print(f"Processing {i + 1}/{total}...", end="\r", flush=True)

        try:
            woman, man = generate_author_pair(ref_text)
        except Exception as e:
            print(f"\nError on row {i + 1}: {e}")
            df.to_csv(OUT_PATH, index=False)
            print(f"Partial results saved to {OUT_PATH}")
            raise

        df.at[i, woman_col] = woman
        df.at[i, man_col] = man
        df.at[i, fail_col] = 0 if (woman.strip() and man.strip()) else 1

        time.sleep(SLEEP_S)

        if CHECKPOINT_EVERY and (i + 1) % CHECKPOINT_EVERY == 0:
            df.to_csv(OUT_PATH, index=False)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nDone. Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()