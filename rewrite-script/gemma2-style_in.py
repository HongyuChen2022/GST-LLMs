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
OUT_PATH = "./datasets/gemma2_style_in_500.csv"

SLEEP_S = 0.1
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512
SEED = 42

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
    Extract FEMININE and MASCULINE rewrites from the model response.
    Tolerates minor formatting issues (extra whitespace, different casing).
    """
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

    if is_placeholder(fem) or is_placeholder(masc):
        raise ValueError(f"Placeholder output detected.\nRAW OUTPUT:\n{raw[:900]}")

    return fem, masc


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
                "FEMININE: <text>\n"
                "MASCULINE: <text>\n\n"
                f"CONTENT:\n<<<{raw_output}>>>"
            ),
        },
    ]
    fixed = _gemma_chat(repair_messages, max_new_tokens=256)
    return parse_two_line_output(fixed)


def generate_style_pair(reference_text: str) -> tuple[str, str]:
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

    fem_col = "feminine_style"
    masc_col = "masculine_style"
    if fem_col not in df.columns:
        df[fem_col] = ""
    if masc_col not in df.columns:
        df[masc_col] = ""

    for i in range(total):
        ref_text = str(df.at[i, text_col])
        print(f"Processing {i + 1}/{total}...", end="\r", flush=True)

        try:
            fem, masc = generate_style_pair(ref_text)
        except Exception as e:
            print(f"\nError on row {i + 1}: {e}")
            df.to_csv(OUT_PATH, index=False)
            print(f"Partial results saved to {OUT_PATH}")
            raise

        df.at[i, fem_col] = fem
        df.at[i, masc_col] = masc

        time.sleep(SLEEP_S)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nDone. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()