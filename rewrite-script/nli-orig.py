#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def pick_entailment_index(id2label: dict) -> int:
    norm = {i: str(lbl).lower() for i, lbl in id2label.items()}
    for i, lbl in norm.items():
        if "entail" in lbl:
            return int(i)
    if len(norm) == 3:
        return 2
    raise ValueError(f"Could not determine entailment label from id2label={id2label}")


@torch.inference_mode()
def entailment_probs(model, tokenizer, premises, hypotheses, device, batch_size=16, max_length=256):
    assert len(premises) == len(hypotheses)
    entail_idx = pick_entailment_index(model.config.id2label)

    out = []
    n = len(premises)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        enc = tokenizer(
            premises[start:end],
            hypotheses[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, entail_idx]
        out.extend(probs.detach().cpu().tolist())
    return out


def iter_csv_files(input_path: Path, recursive: bool):
    if input_path.is_file():
        yield input_path
        return
    pattern = "**/*.csv" if recursive else "*.csv"
    for p in sorted(input_path.glob(pattern)):
        if p.is_file():
            yield p


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # strip whitespace from headers to avoid annoying mismatches
    cols = set(df.columns.astype(str).str.strip())
    # also normalize df.columns in-place to the stripped versions
    df.columns = df.columns.astype(str).str.strip()
    for c in candidates:
        if c in cols:
            return c
    return None


def score_file(csv_path: Path, out_path: Path, fem_candidates, masc_candidates,
               model, tokenizer, device, batch_size, max_length, threshold, on_missing):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    fem_col = pick_first_existing_column(df, fem_candidates)
    masc_col = pick_first_existing_column(df, masc_candidates)

    if fem_col is None or masc_col is None:
        msg = (
            f"[{csv_path}] Missing required column(s): "
            f"{'fem' if fem_col is None else ''} "
            f"{'masc' if masc_col is None else ''}\n"
            f"Fem candidates: {fem_candidates}\n"
            f"Masc candidates: {masc_candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
        if on_missing == "skip":
            print("SKIP:", msg)
            return 0, None
        raise KeyError(msg)

    # drop rows without both texts (keeps all other columns)
    df = df.dropna(subset=[fem_col, masc_col]).copy()
    df[fem_col] = df[fem_col].astype(str)
    df[masc_col] = df[masc_col].astype(str)

    fem = df[fem_col].tolist()
    masc = df[masc_col].tolist()

    e_f2m = entailment_probs(model, tokenizer, premises=fem, hypotheses=masc,
                             device=device, batch_size=batch_size, max_length=max_length)
    e_m2f = entailment_probs(model, tokenizer, premises=masc, hypotheses=fem,
                             device=device, batch_size=batch_size, max_length=max_length)

    df["entail_fem_to_mas"] = e_f2m
    df["entail_mas_to_fem"] = e_m2f
    df["min_entail"] = df[["entail_fem_to_mas", "entail_mas_to_fem"]].min(axis=1)

    if threshold is not None:
        df["meaning_preserved"] = df["min_entail"] >= float(threshold)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df), int(df["meaning_preserved"].sum()) if threshold is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="CSV file or directory of CSVs")
    ap.add_argument("--output_dir", default="./results", help="Where to write <stem>.scored.csv files")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories if input_path is a dir")

    ap.add_argument(
        "--masculine_col",
        nargs="+",
        default=["masculine_style", "male_author_text", "man_author_text"],
        help="List of possible masculine column names; first match is used",
    )
    ap.add_argument(
        "--feminine_col",
        nargs="+",
        default=["feminine_style", "female_author_text", "woman_author_text"],
        help="List of possible feminine column names; first match is used",
    )

    ap.add_argument("--model_name", default="alisawuffles/roberta-large-wanli")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--on_missing", choices=["error", "skip"], default="error")
    args = ap.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    entail_idx = pick_entailment_index(model.config.id2label)
    print(f"Device: {device}")
    print(f"Entailment label index: {entail_idx} ({model.config.id2label[entail_idx]})")

    total_rows = 0
    total_preserved = 0
    counted = False
    files = 0

    for csv_path in iter_csv_files(input_path, args.recursive):
        out_path = output_dir / f"{csv_path.stem}.scored.csv"
        n_rows, preserved = score_file(
            csv_path, out_path,
            fem_candidates=args.fem_candidates,
            masc_candidates=args.masc_candidates,
            model=model, tokenizer=tokenizer, device=device,
            batch_size=args.batch_size, max_length=args.max_length,
            threshold=args.threshold, on_missing=args.on_missing
        )
        if n_rows == 0 and args.on_missing == "skip":
            continue

        files += 1
        total_rows += n_rows
        if args.threshold is not None and preserved is not None:
            total_preserved += preserved
            counted = True
            print(f"Saved: {out_path} | rows: {n_rows} | preserved: {preserved}/{n_rows} @ {args.threshold}")
        else:
            print(f"Saved: {out_path} | rows: {n_rows}")

    if input_path.is_dir():
        if args.threshold is None or not counted:
            print(f"Done. Files scored: {files} | total rows scored: {total_rows}")
        else:
            print(f"Done. Files scored: {files} | preserved total: {total_preserved}/{total_rows} @ {args.threshold}")


if __name__ == "__main__":
    main()