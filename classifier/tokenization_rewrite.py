# tokenization.py
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

def load_df(path, text_col, label_cols=None):
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Missing column: {text_col}")
    texts = df[text_col].astype(str).tolist()

    if label_cols:
        for c in label_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")
        Y = df[label_cols].astype(float).to_numpy()
        row_sums = Y.sum(axis=1, keepdims=True)
        # if rows sum to ~1, treat as probs; else normalize counts to probs
        as_probs = np.allclose(row_sums.mean(), 1.0, atol=1e-3)
        if not as_probs:
            row_sums = np.clip(row_sums, 1.0, None)
            Y = Y / row_sums
        y_major = Y.argmax(axis=1)  # for stratification & hard metrics
    else:
        Y = None
        y_major = None

    return df, texts, Y, y_major

def make_folds(n_samples, y_major, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, va in skf.split(np.zeros(n_samples), y_major):
        folds.append((tr, va))
    return folds

def tokenize_texts(texts, model_name, max_len=128):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    enc = tok(
        texts,
        padding=False,         # pad later in collator
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
    )
    return tok, enc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], default="train",
                    help="train: save folds+labels+encodings; infer: save encodings only")
    ap.add_argument("--data", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_cols", nargs="+", default=None,
                    help="Only needed in --mode train")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--folds_out", default="rewrites/folds.npz")
    ap.add_argument("--enc_out", default="rewrites/encodings.npz")
    ap.add_argument("--labels_out", default="rewrites/labels.npz")
    args = ap.parse_args()

    # Load
    df, texts, Y, y_major = load_df(args.data, args.text_col, args.label_cols if args.mode=="train" else None)

    # Tokenize
    tok, enc = tokenize_texts(texts, args.model_name, max_len=args.max_len)

    # Always save tokenized encodings (lists of lists) + basic tokenizer info
    np.savez(
        args.enc_out,
        input_ids=np.array(enc["input_ids"], dtype=object),
        attention_mask=np.array(enc["attention_mask"], dtype=object),
        model_name=np.array(args.model_name, dtype=object),
        max_len=np.array(args.max_len),
        pad_token_id=np.array(tok.pad_token_id if tok.pad_token_id is not None else 0),
    )
    print(f"Saved encodings -> {args.enc_out}")

    if args.mode == "train":
        if Y is None or y_major is None:
            raise ValueError("In --mode train you must provide --label_cols.")
        folds = make_folds(len(texts), y_major, n_splits=5, seed=42)

        # Save fold indices
        tr_idx = [tr for tr, _ in folds]
        va_idx = [va for _, va in folds]
        np.savez(args.folds_out, train_idx=tr_idx, val_idx=va_idx)
        print(f"Saved folds -> {args.folds_out}")

        # Save labels and class names
        np.savez(
            args.labels_out,
            y_soft=Y,
            y_major=y_major,
            class_names=np.array(args.label_cols, dtype=object),
        )
        print(f"Saved labels -> {args.labels_out}")

    # Report tokenizer info
    print(f"Tokenizer vocab size: {tok.vocab_size}")
    print(f"Mode: {args.mode} | Model: {args.model_name} | max_len: {args.max_len}")

if __name__ == "__main__":
    main()
