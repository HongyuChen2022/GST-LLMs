import os
import time
import re
import random
import pandas as pd

# --- Config ----------------------------------------------------------------

# Prefer env var; fall back to the hardcoded token if present.
# In shell: export HF_TOKEN="hf_..."
HF_TOKEN = os.getenv("HF_TOKEN") 

MODEL_NAME = "google/gemma-2-27b-it"  # local transformers path / HF repo id

IN_PATH = "datasets/pastel_gender_pair_500.csv"
OUT_PATH = "./datasets/gemma2_voice_500.csv"

SLEEP_S = 0.1
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512
SEED = 42

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

# --- Parsing helpers --------------------------------------------------------

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

    if is_placeholder(female) or is_placeholder(male):
        raise ValueError(f"Placeholder output detected.\nRAW OUTPUT:\n{raw[:900]}")

    return female, male


# --- Local Transformers (Gemma) --------------------------------------------

_TOKENIZER = None
_MODEL = None

def _load_gemma_local():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype="auto",
        device_map="auto",
    )
    _MODEL.eval()


def _gemma_normalize_messages(messages: list[dict]) -> list[dict]:
    """
    Gemma chat template does NOT support role='system'.
    This function folds all system messages into the first user message.
    """
    sys_parts = []
    cleaned = []

    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue

        if role == "system":
            sys_parts.append(content)
        else:
            cleaned.append({"role": role, "content": content})

    if sys_parts:
        sys_text = "\n\n".join(sys_parts).strip()
        if cleaned and cleaned[0]["role"] == "user":
            cleaned[0]["content"] = f"{sys_text}\n\n{cleaned[0]['content']}".strip()
        else:
            cleaned = [{"role": "user", "content": sys_text}] + cleaned

    for m in cleaned:
        if m["role"] not in ("user", "assistant"):
            m["role"] = "user"

    return cleaned


def _gemma_chat(messages: list[dict], max_new_tokens: int) -> str:
    import torch

    _load_gemma_local()

    messages = _gemma_normalize_messages(messages)

    prompt = _TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _TOKENIZER(prompt, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=_TOKENIZER.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return _TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()


def repair_format_local(raw_output: str) -> tuple[str, str]:
    repair_messages = [
        {"role": "system", "content": "Reformat text. Follow the output format exactly. No placeholders."},
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
    fixed = _gemma_chat(repair_messages, max_new_tokens=256)
    return parse_two_line_output(fixed)


def generate_author_pair(reference_text: str) -> tuple[str, str]:
    user_prompt = f'Text: """{reference_text}"""'
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content = _gemma_chat(messages, max_new_tokens=MAX_TOKENS)

    try:
        return parse_two_line_output(content)
    except Exception:
        return repair_format_local(content)


# --- Main ------------------------------------------------------------------

def main():
    df = pd.read_csv(IN_PATH)

    if "input.y" in df.columns:
        text_col = "input.y"
    else:
        raise ValueError("Input CSV must contain 'input.y'")

    total = len(df)

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
            df.to_csv(OUT_PATH, index=False)
            print(f"Partial results saved to {OUT_PATH}")
            raise

        df.at[i, female_col] = female
        df.at[i, male_col] = male

        time.sleep(SLEEP_S)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nDone. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()