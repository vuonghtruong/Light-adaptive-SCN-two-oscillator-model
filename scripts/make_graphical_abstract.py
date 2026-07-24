#!/usr/bin/env python3
"""Generate the fitted systemic-X graphical abstract from project outputs."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import model_core as model  # noqa: E402


MM = 1.0 / 25.4
RED = model.COLORS["D"]
BLUE = model.COLORS["V"]
X_COLOR = "#333333"
GREEN = "#168B4B"
GOLD = "#D99A00"
LIGHT = "#F9D77E"
GRID = "#DEDEDE"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_params() -> dict[str, float]:
    with (ROOT / "tables" / "systemic_x_selected_params.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    params = model.default_params()
    for key in params:
        if row.get(key, "") != "":
            params[key] = float(row[key])
    return params


def load_summary(path: Path) -> dict[str, dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, float | str]] = {}
    for row in rows:
        parsed: dict[str, float | str] = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value) if value != "" else value
            except (TypeError, ValueError):
                parsed[key] = value
        out[row["condition"]] = parsed
    return out


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.055, 1.08, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def activity(rows: list[dict], key: str) -> np.ndarray:
    return (np.asarray([r[key] for r in rows], dtype=float) + 1.0) / 2.0


def add_light_bands(ax: plt.Axes, rows: list[dict], x: np.ndarray) -> None:
    light = np.asarray([r["light"] for r in rows])
    starts = np.flatnonzero((light[1:] > light[:-1]) & (light[1:] == 1)) + 1
    ends = np.flatnonzero((light[1:] < light[:-1]) & (light[1:] == 0)) + 1
    if light[0] == 1:
        starts = np.r_[0, starts]
    if light[-1] == 1:
        ends = np.r_[ends, len(light) - 1]
    for start, end in zip(starts, ends):
        ax.axvspan(x[start], x[end], color=LIGHT, alpha=0.72, lw=0, zorder=0)


def style_trace_axis(ax: plt.Axes) -> None:
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks([0, 1])
    ax.spines[["top", "right"]].set_visible(False)


def plot_t22_branching(fig: plt.Figure, spec, params: dict[str, float]) -> None:
    top = spec.subgridspec(2, 2, width_ratios=[0.30, 0.70], hspace=0.16, wspace=0.07)
    schedule_column = top[:, 0].subgridspec(3, 1, height_ratios=[0.5, 1.0, 0.5], hspace=0)
    ax_schedule = fig.add_subplot(schedule_column[1])
    ax_invivo = fig.add_subplot(top[0, 1])
    ax_exvivo = fig.add_subplot(top[1, 1], sharex=ax_invivo)

    invivo = model.simulate("T22", params, ex_vivo_dd=False)
    exvivo = model.simulate("T22", params, ex_vivo_dd=True)
    schedule = [r for r in invivo if -3 * 24 <= r["rel_h"] <= 0]
    inv_dd = [r for r in invivo if 0 <= r["rel_h"] <= 7 * 24]
    exv_dd = [r for r in exvivo if 0 <= r["rel_h"] <= 7 * 24]

    sx = np.asarray([r["rel_h"] / 24 for r in schedule])
    add_light_bands(ax_schedule, schedule, sx)
    for key, color, width in (("D", RED, 1.8), ("V", BLUE, 1.8), ("X", X_COLOR, 1.3)):
        ax_schedule.plot(sx, activity(schedule, key), color=color, lw=width, zorder=3)
    ax_schedule.set_xlim(-3, 0)
    ax_schedule.set_xticks([-3, -2, -1, 0])
    ax_schedule.set_xlabel("T22 schedule day")
    ax_schedule.set_ylabel("Normalized activity")
    ax_schedule.set_title("Shared light entrainment", fontweight="bold", pad=4)
    style_trace_axis(ax_schedule)
    ax_schedule.text(-0.13, 1.58, "A", transform=ax_schedule.transAxes, fontsize=15, fontweight="bold", va="top")

    for ax, rows, title, include_x in (
        (ax_invivo, inv_dd, "In vivo DD: systemic coupling retained", True),
        (ax_exvivo, exv_dd, "In vitro explant DD: X disconnected", False),
    ):
        xx = np.asarray([r["rel_h"] / 24 for r in rows])
        for key, color, width in (("D", RED, 1.9), ("V", BLUE, 1.9)):
            ax.plot(xx, activity(rows, key), color=color, lw=width, zorder=3)
        if include_x:
            ax.plot(xx, activity(rows, "X"), color=X_COLOR, lw=1.3, zorder=3)
        ax.axvline(0, color="#666666", lw=0.9, ls=(0, (3, 3)))
        ax.set_xlim(0, 7)
        ax.set_title(title, loc="left", fontweight="bold", pad=2)
        style_trace_axis(ax)
    ax_invivo.tick_params(labelbottom=False)
    ax_exvivo.set_xticks([0, 2, 4, 6])
    ax_exvivo.set_xlabel("DD day")
    ax_exvivo.set_ylabel("")
    ax_invivo.set_ylabel("")

    # A small fork makes the intervention explicit without adding a block of prose.
    ax_schedule.annotate(
        "",
        xy=(1.05, 1.22), xycoords="axes fraction",
        xytext=(0.975, 0.52), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555555"),
        annotation_clip=False,
    )
    ax_schedule.annotate(
        "",
        xy=(1.05, -0.22), xycoords="axes fraction",
        xytext=(0.975, 0.52), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555555"),
        annotation_clip=False,
    )
    handles = [
        Line2D([0], [0], color=RED, lw=1.6, label="D"),
        Line2D([0], [0], color=BLUE, lw=1.6, label="V"),
        Line2D([0], [0], color=X_COLOR, lw=1.3, label="X"),
        Rectangle((0, 0), 1, 1, facecolor=LIGHT, edgecolor="none", label="Light"),
    ]
    ax_invivo.legend(handles=handles, frameon=False, ncol=4, fontsize=7, loc="upper right")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str,
          width: float = 1.5, style: str = "-|>", rad: float = 0.0, alpha: float = 1.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=width,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def node(ax: plt.Axes, xy: tuple[float, float], radius: float, color: str, text: str) -> None:
    ax.scatter([xy[0]], [xy[1]], s=(radius * 370) ** 2, facecolor=color, edgecolor="black", lw=1.3, zorder=3)
    ax.text(*xy, text, ha="center", va="center", fontweight="bold", fontsize=10, zorder=4)


def plot_architecture(ax: plt.Axes, params: dict[str, float]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    scn_box = FancyBboxPatch(
        (0.08, 0.36), 0.52, 0.43,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor="#F8F8F8", edgecolor="#888888", lw=1.0, linestyle=(0, (3, 2)), zorder=0,
    )
    ax.add_patch(scn_box)
    ax.text(0.11, 0.75, "SCN", fontweight="bold", fontsize=8, color="#555555")

    d = (0.23, 0.58)
    v = (0.53, 0.58)
    x = (0.75, 0.22)
    light_xy = (0.84, 0.82)
    node(ax, d, 0.09, "#E56B6F", "D")
    node(ax, v, 0.09, "#4F86E8", "V")
    node(ax, x, 0.075, "#888888", "X")

    # Attractive D -> V and repulsive V -| D coupling.
    arrow(ax, (0.32, 0.54), (0.44, 0.54), RED, 1.8, "-|>")
    ax.text(
        0.38, 0.40, r"$K_{VD}$ attractive", color=RED, ha="center", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.88), zorder=5,
    )
    arrow(ax, (0.44, 0.64), (0.32, 0.64), BLUE, 1.8, "-[")
    ax.text(
        0.38, 0.72, r"$K_{DV}$ repulsive", color=BLUE, ha="center", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.88), zorder=5,
    )

    # Direct light input reaches both oscillators; line width encodes the fitted gain difference.
    ax.scatter([light_xy[0]], [light_xy[1]], s=430, facecolor=LIGHT, edgecolor=GOLD, lw=1.2, zorder=3)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        p0 = (light_xy[0] + 0.055 * np.cos(angle), light_xy[1] + 0.055 * np.sin(angle))
        p1 = (light_xy[0] + 0.075 * np.cos(angle), light_xy[1] + 0.075 * np.sin(angle))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GOLD, lw=1.1)
    arrow(ax, (0.79, 0.78), (0.58, 0.64), GOLD, 3.0, "-|>", rad=0.05)
    arrow(ax, (0.80, 0.84), (0.28, 0.67), GOLD, 1.0, "-|>", rad=0.11)
    ax.text(
        0.70, 0.67, "V input > D input", color="#8A6500", fontsize=8, ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.80), zorder=5,
    )
    ax.text(0.84, 0.91, "Light", ha="center", fontweight="bold")

    # Systemic oscillator feedback and SCN output to X.
    arrow(ax, (0.69, 0.28), (0.29, 0.49), X_COLOR, 1.25, "-|>", rad=-0.12)
    arrow(ax, (0.72, 0.29), (0.56, 0.49), X_COLOR, 1.25, "-|>", rad=0.08)
    arrow(ax, (0.54, 0.49), (0.70, 0.25), "#777777", 1.1, "-|>", rad=0.18)
    ax.text(0.55, 0.26, "systemic feedback", color=X_COLOR, fontsize=8, ha="center")

    memory = FancyBboxPatch(
        (0.05, 0.12), 0.24, 0.12,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="#EAF5EB", edgecolor=GREEN, lw=1.2,
    )
    ax.add_patch(memory)
    ax.text(0.17, 0.18, r"history $M_D\rightarrow\tau_D$", ha="center", va="center", color=GREEN, fontsize=8)
    arrow(ax, (0.20, 0.25), (0.22, 0.48), GREEN, 1.6, "-|>")
    ax.text(0.50, 0.95, "Fitted D-V-X architecture", ha="center", fontweight="bold", fontsize=10)
    ax.text(0.75, 0.10, "no direct photic phase input", ha="center", fontsize=7, color="#555555")
    panel_label(ax, "B")


def style_period_axis(ax: plt.Axes, title: str, conditions: list[str]) -> None:
    ax.set_title(title, fontweight="bold", pad=5)
    ax.set_ylabel("Period (h)")
    ax.set_xticks(np.arange(len(conditions)), conditions)
    ax.set_ylim(22.4, 25.15)
    ax.set_yticks([22.5, 23.5, 24.5])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_aftereffects(ax_photo: plt.Axes, ax_tcycle: plt.Axes,
                      inv: dict[str, dict], exv: dict[str, dict]) -> None:
    marker_specs = [
        ("Behaviour in vivo", "behaviour_period_h", "o", "#6B6B6B"),
        ("D ex vivo", "SCN_D_period_h", "s", RED),
        ("V ex vivo", "SCN_V_period_h", "^", BLUE),
    ]
    for ax, conditions, title in (
        (ax_photo, ["LP", "LD", "SP"], "Photoperiod alignment"),
        (ax_tcycle, ["T22", "LD", "T26"], "T-cycle compensation"),
    ):
        xx = np.arange(len(conditions))
        for label, key, marker, color in marker_specs:
            source = inv if key == "behaviour_period_h" else exv
            yy = [float(source[c][key]) for c in conditions]
            ax.plot(xx, yy, color=color, lw=1.3, marker=marker, ms=4.2, label=label, zorder=3)
        style_period_axis(ax, title, conditions)
    ax_photo.legend(
        handles=[Line2D([0], [0], color=c, marker=m, lw=1.3, ms=4, label=l) for l, _, m, c in marker_specs],
        frameon=False, fontsize=7, loc="lower right",
    )
    panel_label(ax_photo, "C")


def main() -> None:
    configure_style()
    params = load_params()
    inv = load_summary(ROOT / "tables" / "final_in_vivo_summary.csv")
    exv = load_summary(ROOT / "tables" / "final_ex_vivo_SCN_summary.csv")

    fig = plt.figure(figsize=(175 * MM, 140 * MM), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.76, 1.34], hspace=0.25)
    plot_t22_branching(fig, outer[0], params)
    lower = outer[1].subgridspec(1, 2, width_ratios=[1.72, 0.78], wspace=0.20)
    ax_arch = fig.add_subplot(lower[0])
    evidence = lower[1].subgridspec(2, 1, hspace=0.42)
    ax_photo = fig.add_subplot(evidence[0])
    ax_tcycle = fig.add_subplot(evidence[1])

    plot_architecture(ax_arch, params)
    plot_aftereffects(ax_photo, ax_tcycle, inv, exv)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.075)
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_6_graphical_abstract"
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    print(f"Wrote {stem}.svg/.pdf/.png", flush=True)


if __name__ == "__main__":
    main()
