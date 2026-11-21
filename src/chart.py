"""Recall vs daily alert budget, HI-Medium: step-4 test (Sept 8-10) and forward test (Sept 11-16). Writes results/recall_vs_budget.png."""
import os, polars as pl, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BUD = [("low", 180), ("high", 300), ("low_s", 723), ("high_s", 1204)]
SER = [("full_r3", "Graph model (this project)", "#2a78d6", "o"), ("base_model", "Account-feature model", "#eb6834", "s"),
       ("best_single", "Best single feature*", "#1baf7a", "^"), ("rule", "Hand-written rule", "#eda100", "D"), ("rand", "Random", "#9a9893", None)]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def rows(tag):
    f = pl.concat([pl.read_csv(f"{R}/{tag}_full.csv"), pl.read_csv(f"{R}/{tag}_baselines.csv")], how="diagonal_relaxed")
    f = f.with_columns(v=pl.col("variant").str.split(":").list.first())
    return f.group_by("v").agg([pl.col(k).mean() for k, _ in BUD])


panels = [("himed_sep10_d8-10", "Held-out test, Sept 8–10"), ("himed_sep16_d11-16", "Forward test, Sept 11–16 (model frozen)")]
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, facecolor=SURF)
for ax, (tag, title) in zip(axes, panels):
    ax.set_facecolor(SURF); f = rows(tag); xs = [k for _, k in BUD]
    for v, lab, col, mk in SER:
        r = f.filter(pl.col("v") == v)
        if r.height == 0: continue
        ys = [r[k][0] for k, _ in BUD]
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=8, mec=SURF, mew=2, label=lab, zorder=3 if v == "full_r3" else 2,
                ls="--" if v == "rand" else "-")
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=10,
                    fontweight="bold" if v == "full_r3" else "normal")
    ax.axvline(300, color=INK2, lw=1, ls=":", zorder=1)
    ax.text(310, 0.97, "6 analysts × 50/day", color=INK2, fontsize=9, va="top")
    ax.text(1190, 0.97, "same alert rate\nas small bank", color=INK2, fontsize=9, va="top", ha="right")
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel("Alerts per day", color=INK2); ax.set_xticks(xs); ax.set_xlim(100, 1400); ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, lw=1); ax.tick_params(colors=INK2)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel("Share of mule accounts caught (recall)", color=INK2)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=5, frameon=False, labelcolor=INK, fontsize=10)
fig.suptitle("Mule-account detection on IBM synthetic AML data (HI-Medium, up to 32M transactions)", color=INK, fontsize=14, x=0.01, ha="left")
fig.text(0.01, 0.01, "*Best single feature chosen on the test days, which flatters it. 3-seed means. Pre-registered; see prereg/.", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(f"{R}/recall_vs_budget.png", dpi=160, facecolor=SURF)
print("wrote", f"{R}/recall_vs_budget.png")

