# infer_ensemble_sd_manual.py
# Load encodings + 5 fold models manually, predict, and compute per-text SD across folds.

import os, json, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from model import SoftBertClassifier  # same class used in training
import pandas as pd
# ---------- USER SETTINGS ----------
ENC_NPZ    = "./data/embeddings_500/gemma2_PGS_feminine_in.npz"        # tokenized inputs: input_ids, attention_mask (lists of lists)
MODELS_DIR = "./model_bert_tri"                  # contains: fold_1 ... fold_5 (each has config.json, pytorch_model.bin)
OUT_NPZ    = "./data/preds/500/gemma2_PGS_feminine_in.npz"
BATCH_SIZE = 64
# -----------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load encodings ----
enc = np.load(ENC_NPZ, allow_pickle=True)
input_ids = enc["input_ids"].tolist()
attention_mask = enc["attention_mask"].tolist()
pad_token_id = int(enc["pad_token_id"]) if "pad_token_id" in enc.files else 0

class EncodedDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.ids = input_ids
        self.mask = attention_mask
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.mask[i], dtype=torch.long),
        }

def collate_pad(batch, pad_id=0):
    ids  = [b["input_ids"] for b in batch]
    mask = [b["attention_mask"] for b in batch]
    L = max(t.size(0) for t in ids)
    def pad(t):
        if t.size(0) == L: return t
        return torch.cat([t, torch.full((L - t.size(0),), pad_id, dtype=torch.long)], dim=0)
    return {"input_ids": torch.stack([pad(t) for t in ids]),
            "attention_mask": torch.stack([pad(t) for t in mask])}

@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    all_logits = []
    for batch in tqdm(loader, desc="Predict", leave=False):
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        out = model(input_ids=ids, attention_mask=mask)
        all_logits.append(out["logits"].detach().cpu())
    return torch.vstack(all_logits)  # (N, C)

# ---- Build dataloader ----
ds = EncodedDataset(input_ids, attention_mask)
loader = DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=lambda b: collate_pad(b, pad_id=pad_token_id),
    pin_memory=True
)

# ---- Load each fold and collect logits ----
FOLD_DIRS = [os.path.join(MODELS_DIR, f"fold_{k}") for k in range(1, 6)]
fold_logits = []
for fold_dir in FOLD_DIRS:
    with open(os.path.join(fold_dir, "config.json")) as f:
        cfg = json.load(f)
    model = SoftBertClassifier(
        cfg["model_name"],
        num_labels=int(cfg["num_labels"]),
        dropout=float(cfg.get("dropout", 0.1))
    )
    state = torch.load(os.path.join(fold_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)

    L = predict_logits(model, loader, DEVICE)   # (N, 2)
    fold_logits.append(L)

# ---- Stack across folds ----
# L_stack: (K, N, C)
L_stack = torch.stack(fold_logits, dim=0)

# Per-fold probabilities
P_stack = torch.softmax(L_stack, dim=-1)                 # (K, N, C)

# Ensemble mean probs and per-class SD across folds
P_mean = P_stack.mean(dim=0).cpu().numpy()               # (N, C)
P_std = P_stack.std(dim=0, unbiased=False).cpu().numpy()# (N, C)

# Final predictions from ensemble mean
pred_labels = P_mean.argmax(axis=1)                      # (N,)
pred_conf = P_mean.max(axis=1)                         # (N,)

# SD for the chosen class per text
idx = np.arange(P_mean.shape[0])
chosen_sd = P_std[idx, pred_labels]                      # (N,)

# Optional: disagreement score (max variance across classes)
#disagreement = P_stack.var(dim=0).amax(dim=-1).cpu().numpy()  # (N,)

# ---- Save results ----
Path(OUT_NPZ).parent.mkdir(parents=True, exist_ok=True)
np.savez(
    OUT_NPZ,
    probs=P_mean,              # (N,2)
    probs_std=P_std,           # (N,2) per-class SD
    pred_labels=pred_labels,   # (N,)
    pred_conf=pred_conf,       # (N,)
    chosen_sd=chosen_sd       # (N,)
  #  disagreement=disagreement  # (N,)
)

df = pd.DataFrame({
    "pred_label": pred_labels,
    "pred_conf": pred_conf,
    "chosen_sd": chosen_sd,
    "p_fem": P_mean[:, 0],
    "p_mas": P_mean[:, 1],
    "p_neu": P_mean[:, 2],
    "sd_fem": P_std[:, 0],
    "st_mas": P_std[:, 1],
    "st_neu": P_std[:, 2]
})

csv_path = OUT_NPZ.replace(".npz", ".csv")
df.to_csv(csv_path)

print(f"Saved -> {OUT_NPZ}")
print("probs:", P_mean.shape, "| probs_std:", P_std.shape, "| chosen_sd:", chosen_sd.shape)
print("sample probs[0:3]:\n", P_mean[:3])
print(f"💾 Saved CSV -> {csv_path}")
