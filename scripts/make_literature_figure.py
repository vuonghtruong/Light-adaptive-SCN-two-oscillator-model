#!/usr/bin/env python3
"""Generate Figure 1 directly from the three empirical literature CSV files."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MM = 1 / 25.4
FAMILIES = ["photoperiod", "T-cycle", "T-cycle x photoperiod"]
TITLES = ["Photoperiod", "T-cycle", "Photoperiod x T-cycle"]
LABEL_ORDER = {
    "photoperiod": ["8:16 LD", "12:12 LD", "16:8 LD", "18:6 LD", "20:4 LD", "22:2 LD"],
    "T-cycle": ["T20", "T21", "T22", "T24", "T26", "T28"],
    "T-cycle x photoperiod": ["SP T23", "SP T24", "SP T26", "LP T23", "LP T24", "LP T26"],
}
AGE_COLORS = {"young": "#36AA6F", "adult": "#5B8DB8", "old": "#C23B55"}
REPORTER_COLORS = {
    "Bmal1-ELuc": "#1B9E77",
    "PER2::LUC": "#E67E22",
    "Per1::GFP": "#8073AC",
    "Per1-luc": "#4C78A8",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "data" / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def labels_for(rows: list[dict[str, str]], family: str) -> list[str]:
    present = {r["condition_label"] for r in rows if r["condition_family"] == family}
    return [label for label in LABEL_ORDER[family] if label in present] + sorted(present - set(LABEL_ORDER[family]))


def draw_grouped_bars(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    family: str,
    group_key: str,
    colors: dict[str, str],
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    labels = labels_for(rows, family)
    groups = [g for g in colors if any(r["condition_family"] == family and r.get(group_key) == g for r in rows)]
    x = np.arange(len(labels), dtype=float)
    width = min(0.22, 0.72 / max(1, len(groups)))
    for gi, group in enumerate(groups):
        values, sems, xpos = [], [], []
        for li, label in enumerate(labels):
            match = next(
                (
                    r for r in rows
                    if r["condition_family"] == family
                    and r["condition_label"] == label
                    and r.get(group_key) == group
                ),
                None,
            )
            if match is None:
                continue
            xpos.append(x[li] + (gi - (len(groups) - 1) / 2) * width)
            values.append(float(match["mean_period_h"]))
            sems.append(float(match.get("sem_h", 0) or 0))
        ax.bar(xpos, values, width=width * 0.9, color=colors[group], label=group, zorder=3)
        ax.errorbar(xpos, values, yerr=sems, fmt="none", ecolor="#666666", elinewidth=0.7, capsize=1.5, zorder=4)
    ax.set_xticks(x, labels, rotation=38, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(TITLES[FAMILIES.index(family)], fontweight="bold")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)


def draw_correlation(ax: plt.Axes, rows: list[dict[str, str]], family: str) -> None:
    subset = [r for r in rows if r["condition_family"] == family]
    for reporter, color in REPORTER_COLORS.items():
        reporter_rows = [r for r in subset if r["reporter_scn"] == reporter]
        if not reporter_rows:
            continue
        x = np.asarray([float(r["behavior_mean_period_h"]) for r in reporter_rows])
        y = np.asarray([float(r["scn_mean_period_h"]) for r in reporter_rows])
        xerr = np.asarray([float(r.get("behavior_sem_h", 0) or 0) for r in reporter_rows])
        yerr = np.asarray([float(r.get("scn_sem_h", 0) or 0) for r in reporter_rows])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=3.4, color=color, ecolor=color,
                    elinewidth=0.7, capsize=1.5, label=reporter, zorder=3)
        if len(x) >= 2 and np.ptp(x) > 1e-12:
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, slope * xx + intercept, color=color, linewidth=1.0)
    ax.set_xlabel("Behaviour period (h)")
    ax.set_ylabel("SCN period (h)")
    ax.set_title(f"Correlation: {TITLES[FAMILIES.index(family)].lower()}", fontweight="bold")
    ax.grid(color="#E6E6E6", linewidth=0.6)


def write_counts(behaviour: list[dict[str, str]], scn: list[dict[str, str]], corr: list[dict[str, str]]) -> None:
    rows = []
    for family in FAMILIES:
        rows.append(
            {
                "condition_family": family,
                "behaviour_summary_rows": sum(r["condition_family"] == family for r in behaviour),
                "scn_summary_rows": sum(r["condition_family"] == family for r in scn),
                "paired_correlation_rows": sum(r["condition_family"] == family for r in corr),
            }
        )
    path = ROOT / "tables" / "figure_1_input_counts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_style()
    behaviour = read_csv("tcycle_behavior_summary_for_plots.csv")
    scn = read_csv("tcycle_selected_scn_summary_for_plots.csv")
    corr = read_csv("tcycle_selected_scn_behavior_correlation_pairs.csv")

    fig, axes = plt.subplots(3, 3, figsize=(175 * MM, 170 * MM))
    for col, family in enumerate(FAMILIES):
        draw_grouped_bars(axes[0, col], behaviour, family, "age_category", AGE_COLORS,
                          "Behaviour period (h)", (21, 25.3))
        draw_grouped_bars(axes[1, col], scn, family, "reporter", REPORTER_COLORS,
                          "SCN period (h)", (21, 28.2))
        draw_correlation(axes[2, col], corr, family)
    fig.text(0.012, 0.975, "A", fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.655, "B", fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.335, "C", fontsize=15, fontweight="bold", va="top")
    axes[0, 0].legend(frameon=False, loc="lower left", ncol=1)
    reporter_handles, reporter_labels = axes[1, 0].get_legend_handles_labels()
    axes[1, 2].legend(reporter_handles, reporter_labels, frameon=False, loc="upper right", ncol=2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.08, wspace=0.38, hspace=0.64)

    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "figure_1_literature_summary"
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(stem.with_suffix(f".{suffix}"), facecolor="white", **kwargs)
    plt.close(fig)
    write_counts(behaviour, scn, corr)
    print(f"Wrote {stem}.svg/.pdf/.png", flush=True)


if __name__ == "__main__":
    main()
