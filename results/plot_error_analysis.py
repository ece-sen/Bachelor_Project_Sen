"""
plot_error_analysis.py

Generates three publication-ready figures from error analysis JSON files:
  1. Side-by-side pie charts — failure category distribution (MLP vs Hybrid)
  2. Grouped bar chart     — category counts compared side by side
  3. Horizontal bar chart  — top confused database pairs (MLP)

Usage:
    python ./results/plot_error_analysis.py

Outputs:
    figures/error_pie_charts.pdf   (and .png)
    figures/error_bar_comparison.pdf (and .png)
    figures/error_confusion_pairs.pdf (and .png)
"""

from html import parser
import json
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# colour palette
PALETTE = {
    "NEAR_MISS":        "#4C72B0",
    "AMBIGUOUS":        "#55A868",
    "VOCAB_DIVERGENCE": "#C44E52",
    "OTHER":            "#8172B2",
    "FAR_MISS":         "#CCB974",
    "SCHEMA_CONFUSION": "#64B5CD",
}
NICE_NAMES = {
    "NEAR_MISS":        "Near miss",
    "AMBIGUOUS":        "Ambiguous query",
    "VOCAB_DIVERGENCE": "Vocab divergence",
    "OTHER":            "Other",
    "FAR_MISS":         "Far miss",
    "SCHEMA_CONFUSION": "Schema confusion",
}
CATEGORY_ORDER = [
    "NEAR_MISS", "AMBIGUOUS", "VOCAB_DIVERGENCE",
    "OTHER", "FAR_MISS", "SCHEMA_CONFUSION",
]


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, outdir: str, stem: str, dpi: int = 150):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved → {p}")


# Figure 1 — pie charts 
def plot_pies(mlp: dict, hybrid: dict, outdir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=0.35)
 
    datasets = [
        (mlp,    axes[0], "MLP fusion",    mlp["total_failures"]),
        (hybrid, axes[1], "Linear hybrid", hybrid["total_failures"]),
    ]
 
    for data, ax, title, total in datasets:
        counts = [data["category_counts"].get(c, 0) for c in CATEGORY_ORDER]
        colors = [PALETTE[c] for c in CATEGORY_ORDER]
        labels = [
            f"{NICE_NAMES[c]}\n{data['category_counts'].get(c,0)} "
            f"({data['category_counts'].get(c,0)/total*100:.1f}%)"
            for c in CATEGORY_ORDER
        ]
 
        wedges, _, autotexts = ax.pie(
            counts,
            colors=colors,
            startangle=140,
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
            autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
            pctdistance=0.75,
        )

        for autotext in autotexts:
            autotext.set_fontsize(8.5)
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax.set_title(
            f"{title}\n{total} failures / 2,147 queries",
            fontsize=12, fontweight="normal", pad=14,
        )
 
    # shared legend below both axes
    patches = [
        mpatches.Patch(color=PALETTE[c], label=NICE_NAMES[c])
        for c in CATEGORY_ORDER
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.text(
        0.5, -0.13,
        "Note: categories are not mutually exclusive — one failure may belong to multiple categories.",
        ha="center", fontsize=8.5, color="#666666", style="italic",
    )
 
    save(fig, outdir, "error_pie_charts")
    plt.close(fig)
 

# Figure 2 — grouped bar chart 
def plot_bars(mlp: dict, hybrid: dict, outdir: str):
    mlp_total    = mlp["total_failures"]
    hybrid_total = hybrid["total_failures"]

    mlp_pct    = [mlp["category_counts"].get(c, 0)    / mlp_total    * 100 for c in CATEGORY_ORDER]
    hybrid_pct = [hybrid["category_counts"].get(c, 0) / hybrid_total * 100 for c in CATEGORY_ORDER]

    x     = np.arange(len(CATEGORY_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_mlp    = ax.bar(x - width/2, mlp_pct,    width, label="MLP fusion",    color="#4C72B0", alpha=0.88)
    bars_hybrid = ax.bar(x + width/2, hybrid_pct, width, label="Linear hybrid", color="#C44E52", alpha=0.88)

    # value labels on top of each bar
    for bar in list(bars_mlp) + list(bars_hybrid):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}%",
            ha="center", va="bottom", fontsize=8.5, color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([NICE_NAMES[c] for c in CATEGORY_ORDER], fontsize=10)
    ax.set_ylabel("% of failures", fontsize=11)
    ax.set_title(
        "Failure category distribution — MLP fusion vs linear hybrid",
        fontsize=12, fontweight="normal", pad=12,
    )
    ax.set_ylim(0, max(mlp_pct + hybrid_pct) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False)

    fig.text(
        0.5, -0.04,
        "Note: categories are not mutually exclusive — percentages sum to more than 100%.",
        ha="center", fontsize=8.5, color="#666666", style="italic",
    )

    save(fig, outdir, "error_bar_comparison")
    plt.close(fig)


# Figure 3 — confused pairs horizontal bar chart 
def plot_confusion_pairs(mlp: dict, hybrid: dict, outdir: str, top_n: int = 10):

    def _draw(ax, data, title, top_n):
        pairs  = data["confusion_pairs"][:top_n]
        labels = [f"{p['db1']}  ↔  {p['db2']}" for p in pairs]
        counts = [p["count"] for p in pairs]
        labels = labels[::-1]
        counts = counts[::-1]

        colors = ["#C44E52" if c >= 20 else "#4C72B0" if c >= 10 else "#8172B2"
                  for c in counts]
        bars = ax.barh(range(len(labels)), counts, color=colors, alpha=0.85, height=0.6)

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center", ha="left", fontsize=9, color="#333333",
            )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Number of confused queries", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="normal", pad=10)
        ax.set_xlim(0, max(counts) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.55)

    _draw(axes[0], mlp,    "Top 10 confused database pairs — MLP fusion",    top_n)
    _draw(axes[1], hybrid, "Top 10 confused database pairs — Linear hybrid", top_n)

    patches = [
        mpatches.Patch(color="#C44E52", label="≥ 20 queries"),
        mpatches.Patch(color="#4C72B0", label="10–19 queries"),
        mpatches.Patch(color="#8172B2", label="< 10 queries"),
    ]
    fig.legend(handles=patches, fontsize=9, frameon=False,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))

    save(fig, outdir, "error_confusion_pairs")
    plt.close(fig)

# entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp",    default="results/model_results/error_analysis_mlp_fusion.json")
    parser.add_argument("--hybrid", default="results/model_results/error_analysis_best_weights_hybrid.json")
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    print("Loading error analysis files...")
    mlp    = load(args.mlp)
    hybrid = load(args.hybrid)
    print(f"  MLP    : {mlp['total_failures']} failures")
    print(f"  Hybrid : {hybrid['total_failures']} failures")

    print("\nPlotting Figure 1 — pie charts...")
    plot_pies(mlp, hybrid, args.outdir)

    print("Plotting Figure 2 — grouped bar chart...")
    plot_bars(mlp, hybrid, args.outdir)

    print("Plotting Figure 3 — confused pairs...")
    plot_confusion_pairs(mlp, hybrid, args.outdir, top_n=10)

    print("\nDone.")


if __name__ == "__main__":
    main()