import os
import re
import pandas as pd
from script.JS import compute_js_for_csv_pair, distances_to_corners, raw_datasets  # expects p_mas,p_fem,p_neu in both CSVs

REFERENCE_PATH = "./preds/500/reference_text.csv"

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

# Explicit, unambiguous config: each entry says which file is feminine vs masculine

# Build runs: feminine vs ref AND masculine vs ref
rows = []
for d in raw_datasets:
    # 1) between style (fem vs mas)
    _, stats_fm = compute_js_for_csv_pair(d["fem_path"], d["mas_path"])

    # 2) style vs reference
    _, stats_f_ref = compute_js_for_csv_pair(d["fem_path"], REFERENCE_PATH)
    _, stats_m_ref = compute_js_for_csv_pair(d["mas_path"], REFERENCE_PATH)

    rows.append({
        "dataset": d["name"],
        "model": d["source"],

        # mean JS distance
        "mean_js_style_fem_vs_mas": stats_fm["mean"],
        "mean_js_fem_vs_ref": stats_f_ref["mean"],
        "mean_js_mas_vs_ref": stats_m_ref["mean"],

        # convenient combined number
        "mean_js_style_vs_ref_avg": 0.5 * (stats_f_ref["mean"] + stats_m_ref["mean"]),

        # n checks (should match if row order aligns everywhere)
        "n_style_fem_vs_mas": stats_fm["n"],
        "n_fem_vs_ref": stats_f_ref["n"],
        "n_mas_vs_ref": stats_m_ref["n"],
    })

per_dataset = pd.DataFrame(rows).sort_values(["model", "dataset"])
per_dataset.to_csv("js_table_per_dataset.csv", index=False)

# ---- Aggregate per model (approach) ----
per_model = (
    per_dataset
    .groupby("dataset", as_index=False)
    .agg(
        mean_js_style_fem_vs_mas=("mean_js_style_fem_vs_mas", "mean"),
        mean_js_fem_vs_ref=("mean_js_fem_vs_ref", "mean"),
        mean_js_mas_vs_ref=("mean_js_mas_vs_ref", "mean"),
        mean_js_style_vs_ref_avg=("mean_js_style_vs_ref_avg", "mean"),
        num_datasets=("dataset", "count"),
    )
    .sort_values("dataset")
)

per_model.to_csv("js_table_per_model.csv", index=False)

print("Wrote:")
print(" - js_table_per_dataset.csv")
print(" - js_table_per_model.csv")
print("\nPer-model table:")
print(per_model)
