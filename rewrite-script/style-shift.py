import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from script.JS import raw_datasets

PLOT_DIR = "plot/500"
os.makedirs(PLOT_DIR, exist_ok=True)

REFERENCE_PATH = "./preds/500/reference_text.csv"
PROB_COLS = ["p_fem", "p_mas", "p_neu"]

# Desired within-group row order
APPROACH_ORDER = {
    "gender identity": 0,
    "gender voice": 1,
    "style": 2,
    "style instruction": 3,
}


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def read_probs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in PROB_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[PROB_COLS].copy()


def deltas_vs_ref(rewrite: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """
    Style preservation (to reference):
      Δf = mean(p_fem(rewrite) - p_fem(ref))
      Δm = mean(p_mas(rewrite) - p_mas(ref))
    Assumes row alignment with reference.
    """
    if len(rewrite) != len(ref):
        n = min(len(rewrite), len(ref))
        rewrite = rewrite.iloc[:n]
        ref = ref.iloc[:n]

    return {
        "delta_f": float((rewrite["p_fem"] - ref["p_fem"]).mean()),
        "delta_m": float((rewrite["p_mas"] - ref["p_mas"]).mean()),
        "n": int(len(rewrite)),
    }


def make_row_label(d: dict) -> str:
    # kept for CSV/debugging; plot will use grouped labels
    if d.get("source") == "human":
        return f'human | {d.get("name", "dataset")}'
    return d.get("name", d.get("source", "dataset"))


def parse_model_group(dataset_name: str, source: str) -> str:
    """
    dataset_name examples:
      "Llama-3-style-instruction", "deepseek-v3-author-gender", "qwen-2.5-style", ...
    returns:
      "Llama-3", "deepseek-v3", "qwen-2.5", or "human"
    """
    if str(source).lower() == "human":
        return "human"
    parts = str(dataset_name).split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else str(dataset_name)


def parse_approach_label(dataset_name: str) -> str:
    s = str(dataset_name)
    if "author-gender" in s:
        return "gender identity"
    if "author-voice" in s:
        return "gender voice"
    if "style-instruction" in s:
        return "style instruction"
    if "style" in s:
        return "style"
    return s  # fallback


def plot_heatmap(out: pd.DataFrame, outfile: str) -> None:
    cols = [
        "Δf [feminine rewrites]",
        "Δm [feminine rewrites]",
        "Δf [masculine rewrites]",
        "Δm [masculine rewrites]",
    ]
    mat = out[cols].to_numpy()

    approach_labels = out["approach_label"].tolist()
    groups = out["group"].tolist()
    n_rows = len(approach_labels)

    xticks_small = ["Δf", "Δm", "Δf", "Δm"]

    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    vmin = -vmax if vmax > 0 else -1.0

    # group runs
    runs = []
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            runs.append((groups[start], start, i - 1))
            start = i

    fig_h = max(4, 0.40 * n_rows)

    # key: constrained layout, slightly smaller width
    fig = plt.figure(figsize=(12.0, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.8, 2.4, 8.8],
        wspace=0.02
    )

    ax_model = fig.add_subplot(gs[0, 0])
    ax_app = fig.add_subplot(gs[0, 1])
    ax = fig.add_subplot(gs[0, 2])

    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=12)

    ax.set_xticks(range(len(xticks_small)))
    ax.set_xticklabels(xticks_small, fontsize=11)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)

    ax.axvline(1.5, linewidth=1)

    ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", linewidth=0.25, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.text(0.25, -0.05, "feminine rewrites", transform=ax.transAxes,
            ha="center", va="top", fontsize= 12)
    ax.text(0.75, -0.05, "masculine rewrites", transform=ax.transAxes,
            ha="center", va="top", fontsize= 12)

    for a in (ax_model, ax_app):
        a.set_xlim(0, 1)
        a.set_ylim(ax.get_ylim())
        a.axis("off")

    for i, lab in enumerate(approach_labels):
        ax_app.text(0.98, i, lab, ha="right", va="center", fontsize=13)

    for (g, s, e) in runs:
        y_mid = (s + e) / 2
        ax_model.text(1.2, y_mid, g, ha="right", va="center",
                      fontsize=14, fontweight="bold")

    for (g, s, e) in runs[:-1]:
        y = e + 0.5
        ax.axhline(y, color="black", linewidth=0.8, alpha=0.6)
        ax_app.axhline(y, color="black", linewidth=0.8, alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.set_ylabel("Δ probability", rotation=90, fontsize= 13)

    # key: tight bbox on save
    fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    ref = read_probs(REFERENCE_PATH)

    rows = []
    for d in raw_datasets:
        fem = read_probs(d["fem_path"])
        mas = read_probs(d["mas_path"])

        df_f = deltas_vs_ref(fem, ref)
        df_m = deltas_vs_ref(mas, ref)

        dataset_name = d["name"]
        source = d["source"]

        group = parse_model_group(dataset_name, source)
        approach_label = parse_approach_label(dataset_name)

        rows.append({
            "dataset": dataset_name,
            "model": source,
            "label": make_row_label(d),

            # grouping fields used for plotting
            "group": group,
            "approach_label": approach_label,

            "Δf [feminine rewrites]": df_f["delta_f"],
            "Δm [feminine rewrites]": df_f["delta_m"],
            "Δf [masculine rewrites]": df_m["delta_f"],
            "Δm [masculine rewrites]": df_m["delta_m"],

            "n_rows_used": min(df_f["n"], df_m["n"]),
        })

    out = pd.DataFrame(rows)

    # sort: human first, then group, then approach order
    out["__human_first"] = (out["group"] == "human").astype(int)
    out["__approach_order"] = out["approach_label"].map(APPROACH_ORDER).fillna(999).astype(int)

    out = (
        out.sort_values(
            ["__human_first", "group", "__approach_order", "dataset"],
            ascending=[False, True, True, True]
        )
        .drop(columns=["__human_first", "__approach_order"])
        .reset_index(drop=True)
    )

    out_csv = "./results_500/style_preservation_deltas_per_dataset_multi.csv"
    out.to_csv(out_csv, index=False)

    heatmap_path = os.path.join(PLOT_DIR, "style_preservation_deltas.png")
    plot_heatmap(out, heatmap_path)

    print("Wrote:")
    print(f" - {out_csv}")
    print(f" - {heatmap_path}")


if __name__ == "__main__":
    main()