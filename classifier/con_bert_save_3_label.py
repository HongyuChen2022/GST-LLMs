import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm.auto import tqdm
from model import SoftBertClassifier  # forward(..., labels, weights) -> {"loss","logits"}

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Dataset & Collator
# -----------------------------
class ListEncodedDataset(Dataset):
    def __init__(self, input_ids, attention_mask, y_soft, weights=None, idx=None):
        self.ids  = input_ids if idx is None else [input_ids[i] for i in idx]
        self.mask = attention_mask if idx is None else [attention_mask[i] for i in idx]
        self.y    = y_soft if idx is None else y_soft[idx]
        self.w    = None if weights is None else (weights if idx is None else weights[idx])

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        item = {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.mask[i], dtype=torch.long),
            "labels": torch.tensor(self.y[i], dtype=torch.float),  # (3,)
        }
        if self.w is not None:
            item["weights"] = torch.tensor(self.w[i], dtype=torch.float)
        return item

def collate_fn(batch, pad_id=0):
    ids  = [b["input_ids"] for b in batch]
    mask = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])          # (B,3)
    weights = torch.tensor([b["weights"] for b in batch]) if "weights" in batch[0] else None

    max_len = max(t.size(0) for t in ids)
    def pad_vec(t):
        if t.size(0) == max_len: return t
        pad_len = max_len - t.size(0)
        return torch.cat([t, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0)

    ids  = torch.stack([pad_vec(t) for t in ids])               # (B,L)
    mask = torch.stack([pad_vec(t) for t in mask])              # (B,L)

    out = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    if weights is not None: out["weights"] = weights
    return out

# -----------------------------
# Training / Prediction
# -----------------------------
def train_one_fold(model, train_loader, val_loader, device, lr=2e-5, weight_decay=0.01,
                   max_epochs=3, warmup_ratio=0.1, patience=2):
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader)) * max_epochs
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in tqdm(train_loader, desc=f"Train ep{epoch}", leave=False):
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)
            out = model(**batch)              # expects {"loss", "logits"}
            loss = out["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            bs = batch["input_ids"].size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs
        train_loss = train_loss_sum / max(train_n, 1)

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                for k in list(batch.keys()):
                    batch[k] = batch[k].to(device)
                out = model(**batch)
                loss = out["loss"]
                bs = batch["input_ids"].size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
        val_loss = val_loss_sum / max(val_n, 1)

        print(f"  epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        # Early stopping on val_loss
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val

def predict_probs(model, loader, device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            logits = out["logits"]           # (B,3)
            all_logits.append(logits.detach().cpu().numpy())
    logits = np.vstack(all_logits)           # (N,3)
    # stable softmax
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
    return P

# -----------------------------
# Utility: ceilings for context
# -----------------------------
def entropy_ceiling(Y, W=None, eps=1e-12):
    H = -(Y * np.log(np.clip(Y, eps, 1.0))).sum(axis=1)
    return np.average(H, weights=W) if W is not None else H.mean()

def oracle_acc_ceiling(Y, W=None):
    m = Y.max(axis=1)
    return np.average(m, weights=W) if W is not None else m.mean()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)

    # data artifacts
    ap.add_argument("--df_path", default="./filter/text_new.csv")
    ap.add_argument("--conf_col", default="conf_avg")
    ap.add_argument("--folds_npz", default="outputs/folds.npz")
    ap.add_argument("--enc_npz", default="outputs/encodings.npz")
    ap.add_argument("--labels_npz", default="outputs/labels.npz")  # contains 3-way y_soft
    ap.add_argument("--out_dir", default="outputs/bert_tri_cv")
    ap.add_argument("--use_gpu", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)

    # ---- Folds ----
    folds = np.load(args.folds_npz, allow_pickle=True)
    train_idx_list = folds["train_idx"]
    val_idx_list   = folds["val_idx"]

    # ---- Encodings ----
    enc = np.load(args.enc_npz, allow_pickle=True)
    input_ids = enc["input_ids"].tolist()
    attention_mask = enc["attention_mask"].tolist()

    # ---- 3-way labels (directly used) ----
    lab = np.load(args.labels_npz, allow_pickle=True)
    Y3 = lab["y_soft"].astype(np.float32)      # (N,3) probs [feminine, masculine, neutral]
    class3 = lab["class_names"].tolist()
    assert len(class3) == 3, "labels_npz must contain 3-way probabilities."

    # ---- Confidence from CSV (aligned row order!) ----
    df = pd.read_csv(args.df_path)
    if args.conf_col not in df.columns:
        raise ValueError(f"Confidence column '{args.conf_col}' not found in {args.df_path}.")
    conf_raw = df[args.conf_col].to_numpy(dtype=float)

    # Handle NaNs / out-of-range gracefully
    conf = np.nan_to_num(conf_raw, nan=np.nanmean(conf_raw))
    # If all zeros or weird, fallback to ones
    if not np.isfinite(conf).all() or np.allclose(conf.mean(), 0.0):
        conf = np.ones_like(conf_raw, dtype=float)

    # Confidence-aware weighting: use confidence only
    conf_norm = conf / max(conf.mean(), 1e-9)
    W3 = conf_norm.astype(np.float32)

    # Report some quick stats
    print(f"Confidence mean (raw): {conf.mean():.3f} | normalized mean: {conf_norm.mean():.3f}")

    # No filtering on S>0: we keep all rows for 3-way training
    y_soft = Y3
    w_all  = W3
    class_names = class3
    num_labels = 3
    label2id = {name: i for i, name in enumerate(class_names)}
    id2label = {i: name for i, name in enumerate(class_names)}

    # ---- Device ----
    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ---- Optional ceilings (weighted by confidence) ----
    ce_ll = entropy_ceiling(y_soft, w_all)
    ce_acc = oracle_acc_ceiling(y_soft, w_all)
    print(f"Ceilings (3-way fem/mas/neu, confidence-weighted): "
          f"entropy log-loss={ce_ll:.4f} | oracle acc={ce_acc:.3f}")

    # ---- Collectors for OOF ----
    all_P, all_Y, all_W = [], [], []
    oof_idx_cat, fold_ids_cat, fold_val_losses = [], [], []

    # ---- Cross-validation ----
    for fold, (tr_idx, va_idx) in enumerate(zip(train_idx_list, val_idx_list), 1):
        print(f"\n===== Fold {fold}/{len(train_idx_list)} =====  (train n={len(tr_idx)}, val n={len(va_idx)})")
        ds_tr = ListEncodedDataset(input_ids, attention_mask, y_soft, weights=w_all, idx=tr_idx)
        ds_va = ListEncodedDataset(input_ids, attention_mask, y_soft, weights=w_all, idx=va_idx)

        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = SoftBertClassifier(args.model_name, num_labels=num_labels, dropout=0.1).to(device)

        best_val = train_one_fold(
            model, dl_tr, dl_va, device,
            lr=args.lr, weight_decay=args.weight_decay,
            max_epochs=args.epochs, warmup_ratio=args.warmup_ratio,
            patience=args.patience
        )
        fold_val_losses.append(best_val)

        # Predict on validation fold
        P_val = predict_probs(model, dl_va, device)
        all_P.append(P_val)             # (n_val, 3)
        all_Y.append(y_soft[va_idx])    # (n_val, 3)
        all_W.append(w_all[va_idx])     # (n_val,)

        # Track which original rows these correspond to
        oof_idx_cat.append(np.asarray(va_idx))
        fold_ids_cat.append(np.full(len(va_idx), fold, dtype=int))

        # -------- Save fold checkpoint (reusable for inference) --------
        fold_dir = os.path.join(args.out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # 1) Weights
        torch.save(model.state_dict(), os.path.join(fold_dir, "pytorch_model.bin"))

        # 2) Minimal config for later reconstruction
        with open(os.path.join(fold_dir, "config.json"), "w") as f:
            json.dump({
                "model_name": args.model_name,
                "num_labels": num_labels,
                "label2id": label2id,
                "id2label": id2label,
                "dropout": 0.1
            }, f, indent=2)

    # ---- Save OOF predictions + bookkeeping ----
    P_oof = np.vstack(all_P)
    Y_oof = np.vstack(all_Y)
    W_oof = np.concatenate(all_W)
    oof_idx = np.concatenate(oof_idx_cat)
    fold_ids = np.concatenate(fold_ids_cat)
    fold_val_losses = np.asarray(fold_val_losses, dtype=float)

    np.savez(
        os.path.join(args.out_dir, "oof_preds_threeway.npz"),
        P_oof=P_oof, Y_oof=Y_oof, W_oof=W_oof,
        oof_idx=oof_idx, fold_ids=fold_ids, fold_val_loss=fold_val_losses,
        class_names=np.array(class_names, dtype=object)
    )
    print(f"\nSaved OOF predictions -> {os.path.join(args.out_dir, 'oof_preds_threeway.npz')}")
    print("Done.")

if __name__ == "__main__":
    main()
