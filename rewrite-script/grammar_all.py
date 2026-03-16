#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def pick_positive_index(id2label: dict) -> int:
    """
    Find the index for the "acceptable/grammatical" class.
    CoLA-style models are typically LABEL_0 (unacceptable) / LABEL_1 (acceptable).
    """
    norm = {i: str(lbl).lower() for i, lbl in id2label.items()}
    for i, lbl in norm.items():
        if any(k in lbl for k in ["acceptable", "grammatical", "label_1", "pos", "positive", "yes", "true"]):
            return int(i)
    if len(norm) == 2:
        return 1
    raise ValueError(f"Could not determine positive label from id2label={id2label}")


@torch.inference_mode()
def acceptability_probs(model, tokenizer, texts, device, batch_size=32, max_length=256):
    pos_idx = pick_positive_index(model.config.id2label)
    out = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, pos_idx]
        out.extend(probs.detach().cpu().tolist())

    return out


def iter_csv_files(input_path: Path, recursive: bool):
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Input file is not a CSV: {input_path}")
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(input_path.glob(pattern))
    if not files:
        raise ValueError(f"No CSV files found in: {input_path}")

    for p in files:
        if p.is_file():
            yield p


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # normalize headers (whitespace is a common reason for "missing columns")
    df.columns = df.columns.astype(str).str.strip()
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def score_file(
    csv_path: Path,
    out_path: Path,
    fem_candidates: list[str],
    masc_candidates: list[str],
    model,
    tok,
    device,
    batch_size: int,
    max_length: int,
    on_missing: str,  # "error" or "skip"
):
    df = pd.read_csv(csv_path).copy()
    df.columns = df.columns.astype(str).str.strip()

    fem_col = pick_first_existing_column(df, fem_candidates)
    masc_col = pick_first_existing_column(df, masc_candidates)

    if fem_col is None or masc_col is None:
        msg = (
            f"[{csv_path}] Could not find required columns.\n"
            f"Fem candidates: {fem_candidates}\n"
            f"Masc candidates: {masc_candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
        if on_missing == "skip":
            print("SKIP:", msg)
            return 0
        raise KeyError(msg)

    df = df.dropna(subset=[fem_col, masc_col]).copy()
    df[fem_col] = df[fem_col].astype(str)
    df[masc_col] = df[masc_col].astype(str)

    df["grammar_feminine"] = acceptability_probs(
        model, tok, df[fem_col].tolist(),
        device=device, batch_size=batch_size, max_length=max_length
    )
    df["grammar_masculine"] = acceptability_probs(
        model, tok, df[masc_col].tolist(),
        device=device, batch_size=batch_size, max_length=max_length
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="CSV file or directory of CSVs")
    ap.add_argument("--output_dir", default="./results_grammar", help="Directory for per-file outputs")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories if input_path is a dir")

    ap.add_argument(
        "--fem_candidates",
        nargs="+",
        default=["feminine_style", "female_author_text", "woman_author_text"],
        help="Possible feminine column names; first match is used per file",
    )
    ap.add_argument(
        "--masc_candidates",
        nargs="+",
        default=["masculine_style", "male_author_text", "man_author_text"],
        help="Possible masculine column names; first match is used per file",
    )

    ap.add_argument("--grammar_model", default="textattack/roberta-base-CoLA")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--on_missing", choices=["error", "skip"], default="skip")
    args = ap.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.grammar_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.grammar_model).to(device)
    model.eval()

    pos_idx = pick_positive_index(model.config.id2label)
    print(f"Device: {device}")
    print(f"Positive/acceptable label index: {pos_idx} ({model.config.id2label[pos_idx]})")

    total_rows = 0
    files_scored = 0

    for csv_path in iter_csv_files(input_path, args.recursive):
        out_path = output_dir / f"{csv_path.stem}.grammar_scored.csv"
        n_rows = score_file(
            csv_path, out_path,
            fem_candidates=args.fem_candidates,
            masc_candidates=args.masc_candidates,
            model=model, tok=tok, device=device,
            batch_size=args.batch_size, max_length=args.max_length,
            on_missing=args.on_missing,
        )
        if n_rows == 0 and args.on_missing == "skip":
            continue

        files_scored += 1
        total_rows += n_rows
        print(f"Saved: {out_path} | rows: {n_rows}")

    if input_path.is_dir():
        print(f"Done. Files scored: {files_scored} | total rows scored: {total_rows}")


if __name__ == "__main__":
    main()