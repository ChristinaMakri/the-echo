"""Figures for the README, drawn from the published summaries in results/.

    uv run --group figures python scripts/make_figures.py      # writes figures/*.png

Run analyze_study1.py results/study1 and analyze_study2.py on each results/study2 file first,
so the summary CSVs are current.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "figures"

# Reference palette (dataviz skill): categorical slots 1-3, validated as a set.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

MODELS = ["phi4", "qwen2.5_14b", "qwen2.5_32b"]
LABEL = {"phi4": "phi4", "qwen2.5_14b": "qwen2.5:14b", "qwen2.5_32b": "qwen2.5:32b"}
DIMS = ["legitimacy", "in_group", "distribution", "voice", "rule_vs_outcome", "reciprocity"]
DIM_LABEL = {"legitimacy": "legitimacy", "in_group": "in-group", "distribution": "distribution",
             "voice": "voice", "rule_vs_outcome": "rule vs outcome", "reciprocity": "reciprocity"}


def dim_ticks(ax):
    ax.set_xticks(range(len(DIMS)), [DIM_LABEL[d] for d in DIMS], rotation=35, ha="right",
                  rotation_mode="anchor", fontsize=8)
CHANCE = {"legitimacy": 1 / 3, "distribution": 1 / 3}  # the others have 2 options

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(x):
    return float(x) if x not in ("", None) else None


def tidy(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def pct(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig_say_do(rows):
    """Stated weight vs chosen share of the assigned pole, echo skin, Low and High pooled."""
    by = defaultdict(lambda: defaultdict(list))
    att = {}
    for r in rows:
        if r["skin"] != "echo":
            continue
        vset = r["cell"].split("-")[1]
        by[(r["model"], vset, r["dimension"])]["share"].append(num(r["share"]))
        if num(r["stated"]) is not None:
            by[(r["model"], vset, r["dimension"])]["stated"].append(num(r["stated"]))
        if num(r["attractor"]) is not None:
            att[(r["model"], vset, r["dimension"])] = num(r["attractor"])

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for row, vset in enumerate(["hierarchical", "egalitarian"]):
        for col, model in enumerate(MODELS):
            ax = axes[row][col]
            tidy(ax)
            for i, d in enumerate(DIMS):
                v = by[(model, vset, d)]
                chosen = sum(v["share"]) / len(v["share"])
                stated = sum(v["stated"]) / len(v["stated"])
                ax.plot([i, i], [chosen, stated], color=MUTED, linewidth=1.2, zorder=1)
                ax.hlines(CHANCE.get(d, 0.5), i - 0.3, i + 0.3, colors=MUTED, linestyles=(0, (2, 2)),
                          linewidth=1, zorder=1)
                if (model, vset, d) in att:
                    ax.scatter(i, att[(model, vset, d)], marker="x", s=30, color=INK2, linewidth=1.2, zorder=2)
                ax.scatter(i, stated, s=46, facecolor=SURFACE, edgecolor=ORANGE, linewidth=2, zorder=3)
                ax.scatter(i, chosen, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            dim_ticks(ax)
            ax.set_ylim(-0.04, 1.06)
            pct(ax)
            if row == 0:
                ax.set_title(LABEL[model])
            if col == 0:
                ax.set_ylabel(f"{vset} values\nshare on the assigned pole")
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=ORANGE,
               markeredgewidth=2, markersize=7, label="stated (self-reported weight)"),
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="chosen in probes"),
        Line2D([], [], marker="x", linestyle="", color=INK2, markersize=6, label="no-role attractor"),
        Line2D([], [], linestyle=(0, (2, 2)), color=MUTED, label="chance"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Stated weight vs choices in probes, for the assigned pole "
                 "(echo skin, Low and High pooled)", y=0.94, fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig1_stated_vs_chosen.png", dpi=200)
    plt.close(fig)


def fig_position(rows):
    """Low (privileged) vs High (subordinate) with the same hierarchical value text."""
    share = {(r["skin"], r["model"], r["cell"].split("-")[0], r["dimension"]): num(r["share"])
             for r in rows if r["cell"].endswith("hierarchical")}
    combos = [(s, m) for s in ("echo", "abstract") for m in MODELS]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    for k, d in enumerate(DIMS):
        ax = axes[k // 3][k % 3]
        tidy(ax, grid_axis="x")
        for y, (skin, model) in enumerate(combos):
            lo, hi = share[(skin, model, "low", d)], share[(skin, model, "high", d)]
            ax.plot([lo, hi], [y, y], color=MUTED, linewidth=1.5, zorder=1)
            ax.scatter(lo, y, s=48, color=BLUE, marker="o", edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.scatter(hi, y, s=48, color=ORANGE, marker="s", edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.axvline(CHANCE.get(d, 0.5), color=MUTED, linestyle=(0, (2, 2)), linewidth=1)
        ax.set_yticks(range(len(combos)), [f"{LABEL[m]} · {s}" for s, m in combos])
        ax.set_xlim(-0.04, 1.04)
        pct(ax, "x")
        ax.set_title(DIM_LABEL[d].replace("\n", " "))
        ax.invert_yaxis()
    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="Low (privileged)"),
        Line2D([], [], marker="s", linestyle="", color=ORANGE, markersize=7, label="High (subordinate)"),
        Line2D([], [], linestyle=(0, (2, 2)), color=MUTED, label="chance"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Same hierarchical value text, different position: share of choices on the hierarchical pole\n"
                 "(where only a square shows, Low and High coincide)", y=0.95, fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig2_position.png", dpi=200)
    plt.close(fig)


def fig_experience(revisions):
    """Mean share of hierarchical choices over all probes, by memory condition, role and model.
    'No memory' is the baseline before any memory; every experience is shown, not only its
    difference from neutral memory, because neutral memory itself moves some cells."""
    import pyarrow.parquet as pq

    hier = {"legitimacy": "birth", "in_group": "favor_own", "distribution": "contribution",
            "voice": "few_decide", "rule_vs_outcome": "rule", "reciprocity": "conditional"}
    conds = ["baseline", "neutral", "fair", "exploitation", "scarcity"]
    cond_label = ["no memory", "neutral", "fair", "exploitation", "scarcity"]
    roles = [("low", "Low", BLUE, "o"), ("mid", "Mid", AQUA, "D"), ("high", "High", ORANGE, "s")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        rows = pq.read_table(next((RESULTS / "study2").glob(f"echo-{model}-hierarchical-*.parquet"))).to_pylist()
        tidy(ax)
        for k, (role, name, color, marker) in enumerate(roles):
            ys = []
            for c in conds:
                rr = [r for r in rows if r["role"] == role and r["experience"] == c and r["choice"]]
                ys.append(sum(r["choice"] == hier[r["dimension"]] for r in rr) / len(rr))
            # Conditions are categories, not a sequence: markers only, offset by role.
            ax.plot([i + (k - 1) * 0.18 for i in range(len(conds))], ys, linestyle="", marker=marker,
                    markersize=8, color=color, markeredgecolor=SURFACE, label=name)
        ax.axvline(0.5, color=GRID, linewidth=1)
        ax.set_xticks(range(len(conds)), cond_label, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 1)
        pct(ax)
        ax.set_title(f"{LABEL[model]}\ndeclared revisions {revisions[model]}")
    axes[0].set_ylabel("share of hierarchical choices\n(all probes pooled)")
    handles = [Line2D([], [], color=c, marker=mk, linestyle="", markersize=8, label=f"{n}-voice agent")
               for _, n, c, mk in roles]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Choices after each kind of memory (hierarchical starting values; doses pooled)",
                 y=0.92, fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig3_experience.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    s1 = read(RESULTS / "study1" / "summary.csv")
    fig_say_do(s1)
    fig_position(s1)
    # From analyze_study2.py's output (declared revisions / units).
    revisions = {"phi4": "80/180", "qwen2.5_14b": "1/180", "qwen2.5_32b": "2/60 (dose 10 only)"}
    fig_experience(revisions)
    print(f"saved: {sorted(p.name for p in OUT.glob('*.png'))}")


if __name__ == "__main__":
    main()
