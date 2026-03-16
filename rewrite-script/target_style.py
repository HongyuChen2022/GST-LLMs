import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from script.JS import raw_datasets, read_probs, mean_js_between_paths, mean_js_to_corner


PLOT_DIR = "plot/500"
os.makedirs(PLOT_DIR, exist_ok=True)

REFERENCE_PATH = "./preds/500/reference_text.csv"
PROB_COLS = ["p_fem", "p_mas",  "p_neu"]

def normalize_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    X = np.maximum(X, eps)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return X / row_sums


def js_distance_rows(P, Q, eps=1e-12, log_base=2):
    P = normalize_rows(P, eps=eps)
    Q = normalize_rows(Q, eps=eps)
    M = 0.5 * (P + Q)
    log = np.log2 if log_base == 2 else np.log
    kl_pm = np.sum(P * (log(P) - log(M)), axis=1)
    kl_qm = np.sum(Q * (log(Q) - log(M)), axis=1)
    js_div = 0.5 * (kl_pm + kl_qm)
    js_div = np.maximum(js_div, 0.0)
    return np.sqrt(js_div)




# corners follow your PROB_COLS ordering: [p_fem, p_mas, p_neu]
ABS_FEM = np.array([1.0, 0.0, 0.0])
ABS_MAS = np.array([0.0, 1.0, 0.0])

rows = []
for d in raw_datasets:
    fem_path = d["fem_path"]
    mas_path = d["mas_path"]

    mean_js_fem_vs_mas, n_fm = mean_js_between_paths(fem_path, mas_path)

    fem_df = read_probs(fem_path)
    mas_df = read_probs(mas_path)

    mean_js_fem_to_abs_fem = mean_js_to_corner(fem_df, ABS_FEM)
    mean_js_mas_to_abs_mas = mean_js_to_corner(mas_df, ABS_MAS)

    rows.append({
        "dataset": d["name"],
        "model": d["source"],
        "label": d["name"],  # or f'{d["source"]} | {d["name"]}'
        "n": n_fm,
        "JS(fem vs mas)": mean_js_fem_vs_mas,
        "JS(fem vs abs_fem)": mean_js_fem_to_abs_fem,
        "JS(mas vs abs_mas)": mean_js_mas_to_abs_mas,
    })

out = pd.DataFrame(rows).sort_values(["model", "dataset"]).reset_index(drop=True)
out.to_csv("./results_500/js_per_dataset.csv", index=False)
print("Wrote: js_three_groups_per_dataset.csv")

# ---- Heatmap ----
cols = ["JS(fem vs mas)", "JS(fem vs abs_fem)", "JS(mas vs abs_mas)"]
mat = out[cols].to_numpy()
row_labels = out["label"].tolist()
col_labels = ["f↔m", "f↔f'", "m↔m'"]

# not centered at 0 (JS is >= 0)
#vmin = float(np.nanmin(mat)) if np.isfinite(mat).any() else 0.0
#vmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0

vmin = float(np.nanmin(mat))
vmax = float(np.nanmax(mat)) * 1.15

fig_h = max(4, 0.40 * len(row_labels))
fig, ax = plt.subplots(figsize=(8, fig_h))


im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
#ax.set_title("Style Distance Between Rewrites and to Target Styles")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels)
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels)

#ax.text(
 #   0.0, 1.02,
 #   f"ref↔f' = {0.6251226312544116:.2f}   |   ref↔m' = {0.6922472513534745:.2f}",
 #   transform=ax.transAxes, ha="left", va="bottom", fontsize=10
#)


# light grid
ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
ax.grid(which="minor", linewidth=0.5, alpha=0.4)
ax.tick_params(which="minor", bottom=False, left=False)

cbar = fig.colorbar(im, ax=ax, shrink=0.9)
cbar.ax.set_ylabel("Mean JS distance", rotation=90)

fig.tight_layout()
plot_path = os.path.join(PLOT_DIR, "heatmap_js_per_dataset.png")
fig.savefig(plot_path, dpi=200)
plt.close(fig)

print(f"Wrote: {plot_path}")