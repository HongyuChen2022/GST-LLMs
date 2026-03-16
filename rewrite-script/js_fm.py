import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

inp = "./results_500/js_per_dataset.csv"
out = "./results_500/js_fm.csv"
out_png = "./plot/500/js_fem_vs_mas.png"

df = pd.read_csv(inp)

APPROACH_MAP = {
    "PASTEL-author-gender": "gender-identity",

    "Llama-3-author-gender": "gender-identity",
    "Llama-3-author-voice": "gender-voice",
    "Llama-3-style": "style",
    "Llama-3-style-instruction": "style-instruction",

    "deepseek-v3-author-gender": "gender-identity",
    "deepseek-v3-author-voice": "gender-voice",
    "deepseek-v3-style": "style",
    "deepseek-v3-style-instruction": "style-instruction",

    "gemma-3-author-gender": "gender-identity",
    "gemma-3-author-voice": "gender-voice",
    "gemma-3-style": "style",
    "gemma-3-style-instruction": "style-instruction",

    "gemma-2-author-gender": "gender-identity",
    "gemma-2-author-voice": "gender-voice",
    "gemma-2-style": "style",
    "gemma-2-style-instruction": "style-instruction",

    "qwen-2.5-author-gender": "gender-identity",
    "qwen-2.5-author-voice": "gender-voice",
    "qwen-2.5-style": "style",
    "qwen-2.5-style-instruction": "style-instruction",
}

df["approach"] = df["dataset"].map(APPROACH_MAP)

# Safety: make sure nothing is missing
missing = df[df["approach"].isna()]["dataset"].unique()
if len(missing) > 0:
    raise ValueError(f"Missing approach mapping for: {list(missing)}")

df.to_csv(out, index=False)
print("saved:", out)



# keep only needed columns
col_model = "model"
col_app = "approach"
col_val = "JS(fem vs mas)"
df = df[[col_model, col_app, col_val]].copy()

# exclude human (case-insensitive)
df = df[df[col_model].astype(str).str.lower().ne("human")]

# make sure values are numeric
df[col_val] = pd.to_numeric(df[col_val], errors="coerce")
df = df.dropna(subset=[col_val, col_model, col_app])

# pivot to matrix (if duplicates exist, take mean)
mat = df.pivot_table(index=col_app, columns=col_model, values=col_val, aggfunc="mean")

# ====== plot heatmap ======
# ---- better proportions ----
n_rows, n_cols = mat.shape
fig_w = max(6, 1.2 + 1.0 * n_cols)     # width scales with models
fig_h = max(3, 0.9 + 0.55 * n_rows)    # height scales with approaches

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 9,
})

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
vals = mat.to_numpy()
im = ax.imshow(mat.values, aspect="auto", cmap="YlGn")
# annotate each cell (FIXED)
for i in range(n_rows):
    for j in range(n_cols):
        v = vals[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
# ticks
ax.set_yticks(range(n_rows))
ax.set_yticklabels(mat.index)

ax.set_xticks(range(n_cols))
ax.set_xticklabels(mat.columns, rotation=0, ha="center")

# model labels on top, but not huge
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
#ax.set_xlabel("model", labelpad=8)
#ax.set_ylabel("approach", labelpad=8)

#ax.set_title("Heatmap of JS(fem vs mas)", pad=18)

# smaller colorbar label/ticks
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Mean JS Distance (F and M)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()
print("saved:", out_png)