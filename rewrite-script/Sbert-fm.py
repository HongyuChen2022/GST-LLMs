#!/usr/bin/env python3
import os
import glob
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- config ----
DATA_DIR = "./datasets/500"
OUT_DIR = "./results_500/Sbert/m-f"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
NORMALIZE = True

# Only this dataset uses story <-> opp_rewrite_text
STORY_PAIR_DATASETS = {"pastel_gender_pair_500_swapped"}

STORY_COL = "story"
OPP_REWRITE_COL = "opp_rewrite_text"

# Gendered candidates (order matters: first match wins)
FEM_CANDIDATES = [
    "feminine_style",
    "female_author_text",
    "woman_author_text",
    "female_text",
    "woman_text",
    "feminine",
    "female",
    "woman",
]

MASC_CANDIDATES = [
    "masculine_style",
    "male_author_text",
    "man_author_text",
    "male_text",
    "man_text",
    "masculine",
    "male",
    "man",
]

# Optional explicit alias pairs (non-story). Keep if useful.
PAIR_ALIASES = [
    ("feminine", "masculine"),
    ("female", "male"),
    ("woman", "man"),
    ("text_a", "text_b"),
    ("sentence1", "sentence2"),
    ("sent1", "sent2"),
]
# ---------------

os.makedirs(OUT_DIR, exist_ok=True)
model = SentenceTransformer(MODEL_NAME, device="cuda")


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def is_storyish(colname: str) -> bool:
    n = norm_name(colname)
    return n in {norm_name(STORY_COL), norm_name(OPP_REWRITE_COL)}


def is_genderish(colname: str) -> bool:
    n = norm_name(colname)
    return any(tok in n for tok in [
        "female", "male", "woman", "man", "feminine", "masculine", "gender",
        "femaleauthor", "maleauthor", "womanauthor", "manauthor",
    ])


def pick_first_existing(norm_cols: dict[str, str], candidates: list[str]) -> str | None:
    for c in candidates:
        nc = norm_name(c)
        if nc in norm_cols:
            return norm_cols[nc]
    return None


def find_columns(df: pd.DataFrame, dataset_name: str):
    cols = list(df.columns)

    # normalized -> original (keep first occurrence)
    norm_cols = {}
    for c in cols:
        nc = norm_name(c)
        if nc not in norm_cols:
            norm_cols[nc] = c

    # 0) ONLY this dataset uses story <-> opp_rewrite_text
    if dataset_name in STORY_PAIR_DATASETS:
        if norm_name(STORY_COL) not in norm_cols or norm_name(OPP_REWRITE_COL) not in norm_cols:
            raise ValueError(
                f"[{dataset_name}] Expected columns {STORY_COL!r} and {OPP_REWRITE_COL!r} but not found. "
                f"Available columns: {cols}"
            )
        return norm_cols[norm_name(STORY_COL)], norm_cols[norm_name(OPP_REWRITE_COL)]

    # For all other datasets, story/opp should NOT be used as main pair.
    # 1) Try gendered columns first (preferred)
    fem_col = pick_first_existing(norm_cols, FEM_CANDIDATES)
    masc_col = pick_first_existing(norm_cols, MASC_CANDIDATES)
    if fem_col and masc_col:
        # Guard: do not allow story-ish to participate in gendered pairing
        if not (is_storyish(fem_col) or is_storyish(masc_col)):
            return masc_col, fem_col  # keep your "col_a is masc-ish" convention

    # 2) Try explicit alias pairs (but skip anything story-ish or cross-type)
    for a, b in PAIR_ALIASES:
        na, nb = norm_name(a), norm_name(b)
        if na in norm_cols and nb in norm_cols:
            col_a, col_b = norm_cols[na], norm_cols[nb]

            # Never allow story-ish columns here
            if is_storyish(col_a) or is_storyish(col_b):
                continue

            # Never allow gender-ish paired with story-ish (extra safety)
            if (is_genderish(col_a) and is_storyish(col_b)) or (is_genderish(col_b) and is_storyish(col_a)):
                continue

            return col_a, col_b

    # 3) Fallback: choose two most "text-like" columns, but EXCLUDE story/opp entirely
    candidates = []
    for c in cols:
        if is_storyish(c):
            continue  # IMPORTANT: exclude story/opp for all other datasets

        series = df[c]
        if series.dtype == "object" or pd.api.types.is_string_dtype(series):
            s = series.dropna().astype(str)
            if len(s) == 0:
                continue
            avg_len = s.str.len().mean()
            candidates.append((avg_len, c))

    candidates.sort(reverse=True)

    # Try to find a safe pair among top candidates
    for i in range(min(len(candidates), 10)):
        for j in range(i + 1, min(len(candidates), 10)):
            c1 = candidates[i][1]
            c2 = candidates[j][1]

            # extra safety: avoid pairing gender-ish with story-ish (story-ish shouldn't be here anyway)
            if (is_genderish(c1) and is_storyish(c2)) or (is_genderish(c2) and is_storyish(c1)):
                continue

            return c1, c2

    raise ValueError(
        f"[{dataset_name}] Could not detect two safe text columns.\n"
        f"Available columns: {cols}"
    )


def cosine_sim_from_normalized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


summary_rows = []
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

for path in csv_files:
    name = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)

    col_a, col_b = find_columns(df, name)
    print(f"[{name}] using columns: {col_a!r} and {col_b!r}")

    a_texts = df[col_a].fillna("").astype(str).tolist()
    b_texts = df[col_b].fillna("").astype(str).tolist()

    emb_a = model.encode(
        a_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE,
        show_progress_bar=True,
    )
    emb_b = model.encode(
        b_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE,
        show_progress_bar=True,
    )

    if NORMALIZE:
        sims = cosine_sim_from_normalized(emb_a, emb_b)
    else:
        a_norm = np.linalg.norm(emb_a, axis=1)
        b_norm = np.linalg.norm(emb_b, axis=1)
        sims = np.sum(emb_a * emb_b, axis=1) / (a_norm * b_norm + 1e-12)

    df_out = df.copy()
    df_out["sbert_col_a_used"] = col_a
    df_out["sbert_col_b_used"] = col_b
    df_out["sbert_cosine_similarity"] = sims

    out_path = os.path.join(OUT_DIR, f"{name}_with_similarity.csv")
    df_out.to_csv(out_path, index=False)

    std_sim = float(np.std(sims, ddof=1)) if len(sims) > 1 else float("nan")

    summary_rows.append({
        "dataset": name,
        "rows": int(len(df_out)),
        "col_a_used": col_a,
        "col_b_used": col_b,
        "mean_similarity": float(np.mean(sims)) if len(sims) else float("nan"),
        "std_similarity": std_sim,
        "median_similarity": float(np.median(sims)) if len(sims) else float("nan"),
        "min_similarity": float(np.min(sims)) if len(sims) else float("nan"),
        "max_similarity": float(np.max(sims)) if len(sims) else float("nan"),
        "output_file": out_path,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "summary_similarity.csv"), index=False)

print("\nDone. Summary:")
print(summary_df)