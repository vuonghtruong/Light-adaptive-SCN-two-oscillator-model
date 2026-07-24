#!/usr/bin/env python3
"""Regenerate the final systemic-X model figures and supporting CSV tables.

The script reads only the selected parameter CSV and reruns every simulation.
Ex-vivo summaries intentionally exclude behavioural and systemic-X period
readouts: only dorsal SCN, ventral SCN, and the D-V phase gap are reported.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_dd_relaxation as relax  # noqa: E402
import model_core as model  # noqa: E402


MM = 1 / 25.4
ORDER = model.ORDER
NEW_CONDITIONS = {"18:6 LD": (24.0, 18.0), "T23": (23.0, 11.5), "T25": (25.0, 12.5)}
BEHAVIOUR_SEM = model.TARGET_BEHAVIOUR_SEM
COL = {
    "behaviour": "#666666",
    "D": "#D73027",
    "V": "#2C7BB6",
    "X": "#222222",
    "target": "#111111",
    "light": "#F6C85F",
    **{k: model.COLORS[k] for k in ORDER},
    "18:6 LD": "#A50F15",
    "T23": "#5E81AC",
    "T25": "#C06C84",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_params() -> dict[str, float]:
    row = read_one(ROOT / "tables" / "systemic_x_selected_params.csv")
    params = model.default_params()
    for key in params:
        if key in row and row[key] != "":
            params[key] = float(row[key])
    return params


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(out / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.15) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(length=3, width=0.7)
    if grid:
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)


def summarize_condition(condition: str, params: dict, ex_vivo: bool) -> dict:
    trace = model.simulate(condition, params, ex_vivo)
    dd = [r for r in trace if r["stage"] == "DD"]
    gap_n = min(len(dd), int(7 * 24 / model.DT) + 1)
    gap_readout = dd[:gap_n]
    return {
        "condition": condition,
        "context": "ex_vivo" if ex_vivo else "in_vivo",
        "behaviour_period_h": model.lomb_scargle_period(dd, "SCN"),
        "SCN_D_period_h": model.lomb_scargle_period(dd, "D"),
        "SCN_V_period_h": model.lomb_scargle_period(dd, "V"),
        "systemic_X_period_h": model.lomb_scargle_period(dd, "X"),
        "release_gap_h": dd[0]["gap_h"],
        "mean_gap_h": float(np.mean([r["gap_h"] for r in gap_readout])),
        "mean_tau_D_h": float(np.mean([r["tau_D_h"] for r in gap_readout])),
    }


def simulation_summaries(params: dict) -> tuple[list[dict], list[dict]]:
    return model.summarize(params, False), model.summarize(params, True)


def draw_period_bars(
    ax: plt.Axes,
    rows: list[dict],
    series: list[tuple[str, str, str]],
    targets: dict[str, float] | None,
    target_sem: dict[str, float] | None,
    title: str,
) -> None:
    x = np.arange(len(rows))
    width = 0.76 / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for off, (key, label, color) in zip(offsets, series):
        ax.bar(x + off, [float(r[key]) for r in rows], width * 0.92, color=color, label=label, zorder=2)
    if targets:
        vals = np.array([targets[r["condition"]] for r in rows])
        errs = np.array([target_sem[r["condition"]] for r in rows]) if target_sem else None
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=COL["target"], elinewidth=0.8, capsize=2, zorder=4)
        for xx, value in zip(x, vals):
            ax.hlines(value, xx - 0.42, xx + 0.42, color=COL["target"], linestyle=(0, (3, 2)), linewidth=0.9, zorder=3)
    ax.set_xticks(x, [r["condition"] for r in rows])
    ax.set_ylabel("Period (h)")
    ax.set_title(title, fontweight="bold")
    style_axis(ax)


def doubleplot_axis(ax: plt.Axes, trace: list[dict], condition: str, ex_vivo: bool, show_ylabels: bool) -> None:
    starts = [(-14 + i) * 24 for i in range(14)] + [i * 24 for i in range(7)]
    yrows = np.arange(20, -1, -1)
    amp = 0.72
    for idx, (start, base) in enumerate(zip(starts, yrows)):
        rows = [r for r in trace if start <= r["rel_h"] <= start + 48]
        if not rows:
            continue
        xx = np.array([r["rel_h"] - start for r in rows])
        light = np.array([r["light"] for r in rows])
        ax.fill_between(xx, base, base + amp * light, color=COL["light"], alpha=0.35, linewidth=0)
        keys = []
        if not ex_vivo:
            keys.append(("X", COL["X"], 0.12))
        keys.extend([("D", COL["D"], 0.30), ("V", COL["V"], 0.30)])
        for key, color, alpha in keys:
            values = np.maximum(0.0, np.array([r[key] for r in rows]))
            ax.fill_between(xx, base, base + amp * values, color=color, alpha=alpha, linewidth=0)
            ax.plot(xx, base + amp * values, color=color, linewidth=0.55)
        if not ex_vivo:
            for w0 in (0, 24):
                win = [r for r in rows if start + w0 <= r["rel_h"] < start + w0 + 24]
                if win:
                    peak = max(win, key=lambda r: r["SCN"])
                    ax.plot(peak["rel_h"] - start, base + amp * 1.05, "o", color="black", ms=1.6)
    ax.axhline(6.5, color="#555555", linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_xlim(0, 48)
    ax.set_ylim(-0.4, 21.2)
    ax.set_title(condition, fontweight="bold", pad=2)
    ax.set_xticks([])
    if show_ylabels:
        ax.set_yticks([20, 7, 6, 0], ["S1", "S14", "DD1", "DD7"])
    else:
        ax.set_yticks([])
    ax.tick_params(axis="y", length=0, pad=1, labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure1(params: dict, inv: list[dict], exv: list[dict]) -> None:
    fig = plt.figure(figsize=(175 * MM, 175 * MM))
    gs = GridSpec(
        3,
        15,
        figure=fig,
        height_ratios=[1.0, 1.72, 1.72],
        left=0.07,
        right=0.995,
        bottom=0.035,
        top=0.92,
        hspace=0.42,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, 0:6])
    ax_b = fig.add_subplot(gs[0, 6:11])
    ax_c = fig.add_subplot(gs[0, 11:15])
    draw_period_bars(
        ax_a,
        inv,
        [
            ("behaviour_period_h", "Behaviour", COL["behaviour"]),
            ("SCN_D_period_h", "D", COL["D"]),
            ("SCN_V_period_h", "V", COL["V"]),
            ("systemic_X_period_h", "X", COL["X"]),
        ],
        model.TARGET_BEHAVIOUR,
        BEHAVIOUR_SEM,
        "In vivo period fit",
    )
    ax_a.set_ylim(22.4, 25.0)
    panel_label(ax_a, "A", -0.15, 1.18)
    draw_period_bars(
        ax_b,
        exv,
        [("SCN_D_period_h", "D", COL["D"]), ("SCN_V_period_h", "V", COL["V"])],
        model.TARGET_EXVIVO_SCN_PERIOD,
        None,
        "Ex vivo SCN periods",
    )
    ax_b.set_ylim(22.0, 25.2)
    ax_b.set_ylabel("")
    panel_label(ax_b, "B", -0.18, 1.18)
    x = np.arange(len(exv))
    targets = np.array([model.TARGET_GAP[r["condition"]] for r in exv])
    sem = np.array([model.TARGET_GAP_SEM[r["condition"]] for r in exv])
    sim = np.array([float(r["mean_gap_h"]) for r in exv])
    ax_c.errorbar(x - 0.09, targets, yerr=sem, fmt="o", ms=3.5, color="white", mec="black", ecolor="black", capsize=2, label="Target")
    for i, r in enumerate(exv):
        ax_c.plot(i + 0.09, sim[i], "o", ms=4, color=COL[r["condition"]])
    ax_c.axhline(0, color="#888888", linewidth=0.7)
    ax_c.set_xticks(x, [r["condition"] for r in exv])
    ax_c.set_ylabel("D-V gap (h)")
    ax_c.set_title("Ex vivo phase-gap fit", fontweight="bold")
    ax_c.set_ylim(-3.0, 3.2)
    style_axis(ax_c)
    panel_label(ax_c, "C", -0.25, 1.18)
    legend = [
        Patch(facecolor=COL["behaviour"], label="Behaviour"),
        Patch(facecolor=COL["D"], label="D"),
        Patch(facecolor=COL["V"], label="V"),
        Patch(facecolor=COL["X"], label="X"),
        Line2D([0], [0], color="black", linestyle=(0, (3, 2)), label="Target"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.50, 0.995), ncol=5, frameon=False)
    axes_d, axes_e = [], []
    for i, cond in enumerate(ORDER):
        ax = fig.add_subplot(gs[1, i * 3 : (i + 1) * 3])
        doubleplot_axis(ax, model.simulate(cond, params, False), cond, False, i == 0)
        axes_d.append(ax)
        ax = fig.add_subplot(gs[2, i * 3 : (i + 1) * 3])
        doubleplot_axis(ax, model.simulate(cond, params, True), cond, True, i == 0)
        axes_e.append(ax)
    fig.text(0.015, 0.655, "D", fontsize=15, fontweight="bold")
    fig.text(0.045, 0.655, "In vivo", fontsize=9, fontweight="bold")
    fig.text(0.015, 0.333, "E", fontsize=15, fontweight="bold")
    fig.text(0.045, 0.333, "Ex vivo", fontsize=9, fontweight="bold")
    save_figure(fig, "figure_2_model_fit_and_trajectories")


def data_fit_sse(params: dict) -> float:
    inv, exv = simulation_summaries(params)
    score = 0.0
    for i, c in enumerate(ORDER):
        score += ((float(inv[i]["behaviour_period_h"]) - model.TARGET_BEHAVIOUR[c]) / BEHAVIOUR_SEM[c]) ** 2
        sigma_p = 0.25 if c in ("SP", "LP", "LD") else 0.30
        score += ((float(exv[i]["SCN_D_period_h"]) - model.TARGET_EXVIVO_SCN_PERIOD[c]) / sigma_p) ** 2
        score += ((float(exv[i]["SCN_V_period_h"]) - model.TARGET_EXVIVO_SCN_PERIOD[c]) / sigma_p) ** 2
        score += ((float(exv[i]["mean_gap_h"]) - model.TARGET_GAP[c]) / model.TARGET_GAP_SEM[c]) ** 2
    return score


def landscape_score_task(task: tuple[dict, float, float]) -> tuple[float, float, float]:
    base, gphi, gp = task
    varied = dict(base)
    varied["gphi_d"] = gphi
    varied["gp_d"] = gp
    return gphi, gp, data_fit_sse(varied)


def parameter_landscape(params: dict, n: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    gx = np.linspace(params["gphi_d"] - 1.25, params["gphi_d"] + 1.25, n)
    gy = np.linspace(params["gp_d"] - 0.35, params["gp_d"] + 0.35, n)
    z = np.empty((n, n))
    rows = []
    tasks = [(dict(params), float(gphi), float(gp)) for gp in gy for gphi in gx]
    workers = min(8, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (gphi, gp, score) in enumerate(pool.map(landscape_score_task, tasks, chunksize=2), start=1):
            ix = int(np.argmin(np.abs(gx - gphi)))
            iy = int(np.argmin(np.abs(gy - gp)))
            z[iy, ix] = score
            rows.append({"gphi_d": gphi, "gp_d": gp, "weighted_sse": score})
            if done % 25 == 0 or done == len(tasks):
                print(f"  parameter landscape: {done}/{len(tasks)}", flush=True)
    return gx, gy, z, rows


def reduced_phase_metrics(exv: list[dict], params: dict) -> tuple[list[dict], dict[str, np.ndarray]]:
    k_eff = params["kvd"] - params["kdv"]
    phi = np.linspace(-math.pi, math.pi, 501)
    metrics, flows = [], {}
    for row in exv:
        cond = row["condition"]
        tau_d = float(row["mean_tau_D_h"])
        delta = 2 * math.pi / tau_d - 2 * math.pi / params["tau_v"]
        ratio = delta / k_eff if abs(k_eff) > 1e-12 else math.nan
        flow = delta - k_eff * np.sin(phi)
        flows[cond] = flow
        metrics.append(
            {
                "condition": cond,
                "tau_D_h": tau_d,
                "tau_V_h": params["tau_v"],
                "delta_omega_rad_h": delta,
                "K_effective_rad_h": k_eff,
                "fixed_point_ratio": ratio,
                "phase_locked_possible": abs(ratio) <= 1,
            }
        )
    return metrics, {"phi": phi, **flows}


def long_dd_rows(params: dict) -> tuple[list[dict], list[dict]]:
    tau_rows, period_rows = [], []
    for cond in ORDER:
        print(f"  long-DD relaxation: {cond}", flush=True)
        trace = relax.simulate_long(cond, params, False, 60)
        dd = [r for r in trace if r["stage"] == "DD"]
        for day in np.arange(0, 60.01, 0.5):
            chunk = [r for r in dd if day * 24 <= r["rel_h"] < min(60 * 24, (day + 0.5) * 24)]
            if chunk:
                tau_rows.append(
                    {
                        "condition": cond,
                        "DD_day": day,
                        "tau_D_h": float(np.mean([r["tau_D_h"] for r in chunk])),
                        "tau_X_h": float(np.mean([r["tau_X_h"] for r in chunk])),
                    }
                )
        for day in np.arange(0, 55.01, 0.5):
            period_rows.append(
                {
                    "condition": cond,
                    "DD_day": day,
                    "behaviour_period_h": relax.rolling_period(trace, "phi_SCN", float(day), 5.0),
                }
            )
    return tau_rows, period_rows


def figure2(params: dict, exv: list[dict]) -> None:
    gx, gy, z, landscape_rows = parameter_landscape(params)
    metrics, flows = reduced_phase_metrics(exv, params)
    tau_rows, period_rows = long_dd_rows(params)
    write_csv(ROOT / "tables" / "final_parameter_landscape.csv", landscape_rows)
    write_csv(ROOT / "tables" / "final_fixed_point_metrics.csv", metrics)
    flow_rows = []
    for i, phi in enumerate(flows["phi"]):
        flow_rows.append({"phi_rad": phi, **{c: flows[c][i] for c in ORDER}})
    write_csv(ROOT / "tables" / "final_phase_gap_flow.csv", flow_rows)
    write_csv(ROOT / "tables" / "final_long_DD_tau.csv", tau_rows)
    write_csv(ROOT / "tables" / "final_long_DD_behaviour.csv", period_rows)

    fig = plt.figure(figsize=(175 * MM, 138 * MM), layout="constrained")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], wspace=0.25, hspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    norm = mpl.colors.LogNorm(vmin=max(np.nanmin(z), 1e-3), vmax=np.nanpercentile(z, 98))
    im = ax_a.pcolormesh(gx, gy, z, shading="auto", cmap="viridis", norm=norm)
    ax_a.plot(params["gphi_d"], params["gp_d"], "o", color="white", mec="black", ms=5, zorder=3)
    ax_a.set_xlabel(r"Phase gain, $g_{\phi,D}$")
    ax_a.set_ylabel(r"Photoperiod gain, $g_{P,D}$")
    ax_a.set_title("Parameter landscape", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.05, pad=0.03)
    cbar.set_label("Weighted SSE")
    panel_label(ax_a, "A")
    ax_b = fig.add_subplot(gs[0, 1])
    vals = [m["fixed_point_ratio"] for m in metrics]
    ax_b.bar(np.arange(5), vals, color=[COL[c] for c in ORDER], width=0.55)
    ax_b.axhline(1, color="black", linestyle=(0, (3, 2)), linewidth=0.8)
    ax_b.axhline(-1, color="black", linestyle=(0, (3, 2)), linewidth=0.8)
    lim = max(1.2, 1.12 * max(abs(v) for v in vals))
    ax_b.set_ylim(-lim, lim)
    ax_b.set_xticks(np.arange(5), ORDER)
    ax_b.set_ylabel(r"Fixed-point ratio, $\Delta\omega/K_{eff}$")
    ax_b.set_title("Phase-lock criterion", fontweight="bold")
    style_axis(ax_b)
    panel_label(ax_b, "B")
    ax_c = fig.add_subplot(gs[1, 0])
    for c in ORDER:
        ax_c.plot(flows["phi"], flows[c], color=COL[c], label=c)
    ax_c.axhline(0, color="black", linestyle=(0, (3, 2)), linewidth=0.8)
    ax_c.set_xlim(-math.pi, math.pi)
    ax_c.set_xticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"])
    ymax = max(float(np.max(np.abs(flows[c]))) for c in ORDER) * 1.15
    ax_c.set_ylim(-ymax, ymax)
    ax_c.set_xlabel(r"D-V phase gap, $\phi$ (rad)")
    ax_c.set_ylabel(r"$d\phi/dt$ (rad h$^{-1}$)")
    ax_c.set_title("Phase-gap flow", fontweight="bold")
    ax_c.legend(frameon=False, ncol=3, loc="upper center")
    style_axis(ax_c)
    panel_label(ax_c, "C")
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.12)
    ax_d1 = fig.add_subplot(sub[0])
    ax_d2 = fig.add_subplot(sub[1], sharex=ax_d1)
    for c in ORDER:
        rt = [r for r in tau_rows if r["condition"] == c]
        rp = [r for r in period_rows if r["condition"] == c]
        ax_d1.plot([r["DD_day"] for r in rt], [r["tau_D_h"] for r in rt], color=COL[c], linestyle="-")
        ax_d1.plot([r["DD_day"] for r in rt], [r["tau_X_h"] for r in rt], color=COL[c], linestyle=(0, (3, 2)))
        ax_d2.plot([r["DD_day"] for r in rp], [r["behaviour_period_h"] for r in rp], color=COL[c])
    ax_d1.axhline(params["tau_d0"], color="#888888", linewidth=0.6)
    ax_d1.axhline(params["tau_x0"], color="#888888", linewidth=0.6, linestyle=(0, (3, 2)))
    ax_d2.axhline(23.7, color="black", linewidth=0.8, linestyle=(0, (3, 2)))
    ax_d1.set_ylabel("Intrinsic period (h)")
    ax_d2.set_ylabel("Behaviour (h)")
    ax_d2.set_xlabel("DD day")
    ax_d1.set_title("Long-DD relaxation", fontweight="bold")
    ax_d1.tick_params(labelbottom=False)
    style_axis(ax_d1)
    style_axis(ax_d2)
    ax_d1.legend(
        [Line2D([0], [0], color="black"), Line2D([0], [0], color="black", linestyle=(0, (3, 2)))],
        [r"$\tau_D$", r"$\tau_X$"],
        frameon=False,
        ncol=2,
        loc="upper right",
    )
    panel_label(ax_d1, "D", -0.20, 1.20)
    save_figure(fig, "figure_3_landscape_stability_relaxation")


def bifurcation_rows(exv: list[dict], params: dict) -> list[dict]:
    rows = []
    k_grid = np.linspace(0, max(0.05, params["kvd"] * 2.5), 500)
    for summary in exv:
        cond = summary["condition"]
        delta = 2 * math.pi / float(summary["mean_tau_D_h"]) - 2 * math.pi / params["tau_v"]
        for kvd in k_grid:
            keff = kvd - params["kdv"]
            ratio = delta / keff if abs(keff) > 1e-12 else math.nan
            if math.isfinite(ratio) and abs(ratio) <= 1:
                a = math.asin(ratio)
                candidates = [a, math.pi - a]
                candidates = [((p + math.pi) % (2 * math.pi)) - math.pi for p in candidates]
                for phi in candidates:
                    derivative = -keff * math.cos(phi)
                    rows.append(
                        {
                            "condition": cond,
                            "K_VD_rad_h": kvd,
                            "phi_star_rad": phi,
                            "stability": "stable" if derivative < 0 else "unstable",
                            "delta_omega_rad_h": delta,
                            "K_DV_rad_h": params["kdv"],
                        }
                    )
    return rows


def novel_predictions(params: dict) -> list[dict]:
    rows = []
    original = dict(model.CONDITIONS)
    try:
        model.CONDITIONS.update(NEW_CONDITIONS)
        for cond in NEW_CONDITIONS:
            inv = summarize_condition(cond, params, False)
            exv = summarize_condition(cond, params, True)
            rows.append(
                {
                    "condition": cond,
                    "cycle_h": NEW_CONDITIONS[cond][0],
                    "photoperiod_h": NEW_CONDITIONS[cond][1],
                    "in_vivo_behaviour_h": inv["behaviour_period_h"],
                    "in_vivo_D_h": inv["SCN_D_period_h"],
                    "in_vivo_V_h": inv["SCN_V_period_h"],
                    "in_vivo_X_h": inv["systemic_X_period_h"],
                    "ex_vivo_D_h": exv["SCN_D_period_h"],
                    "ex_vivo_V_h": exv["SCN_V_period_h"],
                    "ex_vivo_mean_tau_D_h": exv["mean_tau_D_h"],
                    "ex_vivo_mean_gap_h": exv["mean_gap_h"],
                    "ex_vivo_release_gap_h": exv["release_gap_h"],
                }
            )
    finally:
        model.CONDITIONS.clear()
        model.CONDITIONS.update(original)
    return rows


def figure3(params: dict, exv: list[dict]) -> None:
    pred = novel_predictions(params)
    bif_summaries = list(exv) + [
        {"condition": r["condition"], "mean_tau_D_h": r["ex_vivo_mean_tau_D_h"]}
        for r in pred
    ]
    bif = bifurcation_rows(bif_summaries, params)
    write_csv(ROOT / "tables" / "final_bifurcation_branches.csv", bif)
    write_csv(ROOT / "tables" / "final_novel_condition_predictions.csv", pred)
    legacy_prediction_table = ROOT / "tables" / "final_T23_T25_predictions.csv"
    if legacy_prediction_table.exists():
        legacy_prediction_table.unlink()
    fig = plt.figure(figsize=(175 * MM, 145 * MM))
    gs = GridSpec(
        3, 12, figure=fig, height_ratios=[1, 1, 1.05],
        left=0.075, right=0.985, bottom=0.09, top=0.95,
        hspace=0.48, wspace=0.32,
    )
    bif_by = {c: [r for r in bif if r["condition"] == c] for c in [*ORDER, *NEW_CONDITIONS]}
    summary_by = {r["condition"]: r for r in bif_summaries}

    def draw_bifurcation(ax, cond: str, show_ylabel: bool = False) -> None:
        summary = summary_by[cond]
        delta = 2 * math.pi / float(summary["mean_tau_D_h"]) - 2 * math.pi / params["tau_v"]
        lo, hi = sorted((params["kdv"] - abs(delta), params["kdv"] + abs(delta)))
        xmax = max(0.05, params["kvd"] * 2.5)
        shade_lo, shade_hi = max(0.0, lo), min(xmax, hi)
        if shade_hi > shade_lo:
            ax.axvspan(shade_lo, shade_hi, color="#D9D9D9", alpha=0.75, label="No fixed point")
        for stability, ls in (("stable", "-"), ("unstable", (0, (3, 2)))):
            rr = sorted(
                (r for r in bif_by[cond] if r["stability"] == stability),
                key=lambda r: float(r["K_VD_rad_h"]),
            )
            # The valid branches exist on two disconnected K_VD domains.
            # Plotting them as one line creates a false segment through the
            # no-fixed-point interval, so each side is drawn separately.
            for segment in (
                [r for r in rr if float(r["K_VD_rad_h"]) <= lo],
                [r for r in rr if float(r["K_VD_rad_h"]) >= hi],
            ):
                if segment:
                    ax.plot(
                        [r["K_VD_rad_h"] for r in segment],
                        [r["phi_star_rad"] for r in segment],
                        color=COL[cond], linestyle=ls, linewidth=0.9,
                    )
        ax.axvline(params["kvd"], color="black", linestyle=(0, (2, 2)), linewidth=0.8)
        ax.set_ylim(-math.pi, math.pi)
        ax.set_yticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"] if show_ylabel else [])
        ax.set_xlim(0, xmax)
        ax.set_title(cond.replace(" LD", ""), fontweight="bold", pad=2)
        ax.set_xlabel(r"$K_{VD}$")
        if show_ylabel:
            ax.set_ylabel(r"$\phi^*$ (rad)")
        style_axis(ax, grid=False)

    bif_rows = [
        ["LD", "SP", "LP", "18:6 LD"],
        ["T22", "T23", "T25", "T26"],
    ]
    bif_axes = {}
    for row, conditions in enumerate(bif_rows):
        for col, cond in enumerate(conditions):
            ax = fig.add_subplot(gs[row, col * 3:(col + 1) * 3])
            draw_bifurcation(ax, cond, show_ylabel=col == 0)
            bif_axes[cond] = ax
    panel_label(bif_axes["LD"], "A", -0.28, 1.20)

    labels = [r["condition"].replace(" LD", "") for r in pred]
    x = np.arange(len(pred))
    width = 0.18
    prediction_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], wspace=0.58)
    ax_b1 = fig.add_subplot(prediction_gs[0, 0])
    for j, (key, label, color) in enumerate(
        [
            ("in_vivo_behaviour_h", "Behaviour", COL["behaviour"]),
            ("in_vivo_D_h", "D", COL["D"]),
            ("in_vivo_V_h", "V", COL["V"]),
            ("in_vivo_X_h", "X", COL["X"]),
        ]
    ):
        ax_b1.bar(x + (j - 1.5) * width, [r[key] for r in pred], width, color=color, label=label)
    ax_b1.set_xticks(x, labels)
    ax_b1.set_ylabel("In vivo period (h)")
    ax_b1.set_title("In vivo prediction", fontweight="bold", pad=2)
    inv_values = [r[k] for r in pred for k in ("in_vivo_behaviour_h", "in_vivo_D_h", "in_vivo_V_h", "in_vivo_X_h")]
    ax_b1.set_ylim(min(inv_values) - 0.35, max(inv_values) + 0.35)
    ax_b1.legend(frameon=False, ncol=2, loc="best", handlelength=1.2, columnspacing=0.8)
    style_axis(ax_b1)
    panel_label(ax_b1, "B", -0.22, 1.18)

    ax_b2 = fig.add_subplot(prediction_gs[0, 1])
    ax_b2.bar(x - width / 2, [r["ex_vivo_D_h"] for r in pred], width, color=COL["D"], label="D")
    ax_b2.bar(x + width / 2, [r["ex_vivo_V_h"] for r in pred], width, color=COL["V"], label="V")
    ax_b2.set_xticks(x, labels)
    ax_b2.set_ylabel("Ex vivo SCN period (h)")
    ax_b2.set_title("Ex vivo period", fontweight="bold", pad=2)
    exv_values = [r[k] for r in pred for k in ("ex_vivo_D_h", "ex_vivo_V_h")]
    ax_b2.set_ylim(min(exv_values) - 0.35, max(exv_values) + 0.35)
    ax_b2.legend(frameon=False, ncol=2, loc="best")
    style_axis(ax_b2)
    panel_label(ax_b2, "C", -0.22, 1.18)

    ax_gap = fig.add_subplot(prediction_gs[0, 2])
    gapvals = [r["ex_vivo_mean_gap_h"] for r in pred]
    ax_gap.axhline(0, color="#888888", linewidth=0.7, zorder=0)
    ax_gap.scatter(x, gapvals, color=[COL[r["condition"]] for r in pred], s=28, zorder=3)
    ax_gap.plot(x, gapvals, color="#777777", linewidth=0.7, zorder=1)
    ax_gap.set_xticks(x, labels)
    ax_gap.set_ylabel("Ex vivo D-V gap (h)")
    ax_gap.set_title("Ex vivo phase gap", fontweight="bold", pad=2)
    gapvals = [r["ex_vivo_mean_gap_h"] for r in pred]
    margin = max(0.5, 0.25 * (max(gapvals) - min(gapvals) + 1e-9))
    ax_gap.set_ylim(min(-0.2, min(gapvals) - margin), max(0.2, max(gapvals) + margin))
    style_axis(ax_gap)
    panel_label(ax_gap, "D", -0.22, 1.18)
    save_figure(fig, "figure_4_bifurcation_and_predictions")


def sensitivity_analysis(params: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Compare mutually exclusive light-history adaptation mechanisms."""

    def summarize_architecture(condition: str, mechanism: str, gain: float, periods: bool = True) -> dict:
        varied = dict(params)
        if mechanism == "adaptive period":
            varied["g_d"] = params["g_d"] * gain
            trace = model.simulate(condition, varied, True, adaptation_mode="period")
        else:
            trace = model.simulate(
                condition, varied, True, adaptation_mode="coupling", coupling_memory_gain=gain
            )
        dd = [r for r in trace if r["stage"] == "DD"]
        gap_n = min(len(dd), int(7 * 24 / model.DT) + 1)
        readout = dd[:gap_n]
        d_period = model.lomb_scargle_period(dd, "D") if periods else math.nan
        v_period = model.lomb_scargle_period(dd, "V") if periods else math.nan
        return {
            "condition": condition,
            "D_period_h": d_period,
            "V_period_h": v_period,
            "mean_DV_period_h": 0.5 * (d_period + v_period),
            "mean_gap_h": float(np.mean([r["gap_h"] for r in readout])),
            "mean_tau_D_h": float(np.mean([r["tau_D_h"] for r in readout])),
            "mean_K_effective_rad_h": float(np.mean([r["K_effective_rad_h"] for r in readout])),
        }

    # Fit one coupling-memory gain to the same ex vivo gap targets. Intrinsic
    # periods are fixed in this branch, so all schedule dependence in the
    # reduced restoring term comes from M_D(t).
    gain_grid = np.linspace(-0.50, 0.50, 251)
    gain_fit_rows = []
    gap_sem = {"SP": 0.18, "LP": 0.21, "LD": 0.15, "T22": 0.44, "T26": 0.30}
    for index, gain in enumerate(gain_grid):
        score = 0.0
        for condition in ORDER:
            result = summarize_architecture(condition, "adaptive coupling", float(gain), periods=False)
            score += ((result["mean_gap_h"] - model.TARGET_GAP[condition]) / gap_sem[condition]) ** 2
        gain_fit_rows.append({"coupling_memory_gain_rad_h": float(gain), "weighted_gap_SSE": score})
        if (index + 1) % 20 == 0 or index + 1 == len(gain_grid):
            print(f"  coupling-memory fit: {index + 1}/{len(gain_grid)}", flush=True)
    selected_gain = min(gain_fit_rows, key=lambda r: r["weighted_gap_SSE"])["coupling_memory_gain_rad_h"]
    write_csv(ROOT / "tables" / "final_adaptive_coupling_gain_fit.csv", gain_fit_rows)

    scale_grid = np.linspace(0.0, 2.0, 41)
    rows = []
    mechanisms = ("adaptive period", "adaptive coupling")
    total = len(mechanisms) * len(scale_grid)
    done = 0
    for mechanism in mechanisms:
        for scale in scale_grid:
            actual_gain = float(scale) if mechanism == "adaptive period" else float(selected_gain * scale)
            for condition in ORDER:
                result = summarize_architecture(condition, mechanism, actual_gain, periods=True)
                delta_omega = 2 * math.pi / result["mean_tau_D_h"] - 2 * math.pi / params["tau_v"]
                k_eff = result["mean_K_effective_rad_h"]
                ratio = abs(delta_omega / k_eff) if abs(k_eff) > 1e-12 else math.inf
                rows.append(
                    {
                        "mechanism": mechanism,
                        "adaptive_gain_scale": float(scale),
                        "adaptive_gain_value": actual_gain if mechanism == "adaptive coupling" else params["g_d"] * actual_gain,
                        "selected_coupling_memory_gain_rad_h": selected_gain,
                        "condition": condition,
                        **result,
                        "mean_gap_rad": result["mean_gap_h"] * (2 * math.pi) / 24.0,
                        "delta_omega_rad_h": delta_omega,
                        "K_effective_rad_h": k_eff,
                        "locking_ratio": ratio,
                        "phase_lock_possible": ratio <= 1.0,
                    }
                )
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  adaptive-mechanism scan: {done}/{total}", flush=True)

    baseline = {
        (mechanism, condition): next(
            r for r in rows
            if r["mechanism"] == mechanism and r["condition"] == condition
            and abs(r["adaptive_gain_scale"] - 1.0) < 1e-9
        )
        for mechanism in mechanisms for condition in ORDER
    }
    for row in rows:
        base = baseline[(row["mechanism"], row["condition"])]
        row["fractional_change"] = row["adaptive_gain_scale"] - 1.0
        row["percent_change"] = 100.0 * row["fractional_change"]
        row["D_period_change_h"] = row["D_period_h"] - base["D_period_h"]
        row["V_period_change_h"] = row["V_period_h"] - base["V_period_h"]
        row["mean_DV_period_change_h"] = row["mean_DV_period_h"] - base["mean_DV_period_h"]
        row["gap_change_h"] = row["mean_gap_h"] - base["mean_gap_h"]
        row["baseline_phase_lock_possible"] = base["phase_lock_possible"]

    summary = []
    for mechanism in mechanisms:
        for magnitude in (0.05, 0.50, 1.00):
            selected = [
                r for r in rows if r["mechanism"] == mechanism
                and abs(abs(r["fractional_change"]) - magnitude) < 1e-9
            ]
            summary.append(
                {
                    "mechanism": mechanism,
                    "perturbation_percent": 100.0 * magnitude,
                    "mean_abs_gap_change_h": float(np.mean([abs(r["gap_change_h"]) for r in selected])),
                    "max_abs_gap_change_h": float(np.max([abs(r["gap_change_h"]) for r in selected])),
                    "mean_abs_D_period_change_h": float(np.mean([abs(r["D_period_change_h"]) for r in selected])),
                    "mean_abs_V_period_change_h": float(np.mean([abs(r["V_period_change_h"]) for r in selected])),
                    "locking_status_changes": sum(
                        bool(r["phase_lock_possible"]) != bool(r["baseline_phase_lock_possible"])
                        for r in selected
                    ),
                    "lock_to_no_lock": sum(
                        bool(r["baseline_phase_lock_possible"]) and not bool(r["phase_lock_possible"])
                        for r in selected
                    ),
                    "no_lock_to_lock": sum(
                        not bool(r["baseline_phase_lock_possible"]) and bool(r["phase_lock_possible"])
                        for r in selected
                    ),
                    "comparisons": len(selected),
                }
            )

    derivatives = []
    for mechanism, driver in (("adaptive period", "delta_omega_rad_h"), ("adaptive coupling", "K_effective_rad_h")):
        for condition in ORDER:
            base = baseline[(mechanism, condition)]
            minus = next(r for r in rows if r["mechanism"] == mechanism and r["condition"] == condition and abs(r["adaptive_gain_scale"] - 0.95) < 1e-9)
            plus = next(r for r in rows if r["mechanism"] == mechanism and r["condition"] == condition and abs(r["adaptive_gain_scale"] - 1.05) < 1e-9)
            driver_span = plus[driver] - minus[driver]
            numerical_gap = (plus["mean_gap_rad"] - minus["mean_gap_rad"]) / driver_span if abs(driver_span) > 1e-12 else math.nan
            numerical_period = (plus["mean_DV_period_h"] - minus["mean_DV_period_h"]) / driver_span if abs(driver_span) > 1e-12 else math.nan
            delta = base["delta_omega_rad_h"]
            k_eff = base["K_effective_rad_h"]
            ratio = abs(delta / k_eff) if abs(k_eff) > 1e-12 else math.inf
            root = math.sqrt(max(0.0, k_eff ** 2 - delta ** 2))
            kvd = params["kdv"] + k_eff
            omega_locked = 2 * math.pi / params["tau_v"] + kvd * delta / k_eff if abs(k_eff) > 1e-12 else math.nan
            analytic_available = k_eff > 0 and ratio < 1 and root > 0 and omega_locked > 0
            analytic_gap = analytic_period = math.nan
            if analytic_available and mechanism == "adaptive period":
                analytic_gap = 1.0 / root
                analytic_period = -((2 * math.pi) / omega_locked ** 2) * (kvd / k_eff)
            elif analytic_available:
                analytic_gap = -delta / (k_eff * root)
                analytic_period = ((2 * math.pi) / omega_locked ** 2) * (params["kdv"] * delta / k_eff ** 2)
            derivatives.append(
                {
                    "condition": condition,
                    "mechanism": mechanism,
                    "driver": "Delta omega" if mechanism == "adaptive period" else "K effective",
                    "baseline_delta_omega_rad_h": delta,
                    "baseline_K_effective_rad_h": k_eff,
                    "baseline_locking_ratio": ratio,
                    "numerical_dphi_d_driver_h": numerical_gap,
                    "reduced_analytic_dphi_d_driver_h": analytic_gap,
                    "numerical_dperiod_d_driver_h2_per_rad": numerical_period,
                    "reduced_analytic_dperiod_d_driver_h2_per_rad": analytic_period,
                    "reduced_derivative_defined": analytic_available,
                }
            )
    return rows, summary, derivatives


def figure4(params: dict) -> tuple[list[dict], list[dict], list[dict]]:
    rows, summary, derivatives = sensitivity_analysis(params)
    write_csv(ROOT / "tables" / "final_parameter_sensitivity.csv", rows)
    write_csv(ROOT / "tables" / "final_parameter_sensitivity_summary.csv", summary)
    write_csv(ROOT / "tables" / "final_driver_sensitivity_derivatives.csv", derivatives)

    fig, axes = plt.subplots(4, 2, figsize=(175 * MM, 165 * MM))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.965, wspace=0.31, hspace=0.54)
    intrinsic = [r for r in rows if r["mechanism"] == "adaptive period"]
    coupling = [r for r in rows if r["mechanism"] == "adaptive coupling"]

    def draw_condition_lines(ax, data, xkey, ykey):
        for condition in ORDER:
            rr = sorted((r for r in data if r["condition"] == condition), key=lambda r: r[xkey])
            ax.plot([r[xkey] for r in rr], [r[ykey] for r in rr], color=COL[condition], linewidth=1.0, label=condition)
            base = min(rr, key=lambda r: abs(r["fractional_change"]))
            ax.scatter(base[xkey], base[ykey], s=16, color=COL[condition], edgecolor="black", linewidth=0.35, zorder=3)

    draw_condition_lines(axes[0, 0], intrinsic, "adaptive_gain_scale", "delta_omega_rad_h")
    axes[0, 0].set_xlabel(r"Period-memory gain ($g_D/g_{D,fit}$)")
    axes[0, 0].set_ylabel(r"$\Delta\omega$ (rad h$^{-1}$)")
    axes[0, 0].set_title("Intrinsic period to frequency mismatch", fontweight="bold")

    draw_condition_lines(axes[0, 1], coupling, "adaptive_gain_scale", "K_effective_rad_h")
    axes[0, 1].axhline(0, color="#777777", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[0, 1].set_xlabel(r"Coupling-memory gain ($g_K/g_{K,fit}$)")
    axes[0, 1].set_ylabel(r"$K_{eff}$ (rad h$^{-1}$)")
    axes[0, 1].set_title("Coupling to effective restoring term", fontweight="bold")

    draw_condition_lines(axes[1, 0], intrinsic, "delta_omega_rad_h", "mean_gap_h")
    axes[1, 0].set_xlabel(r"$\Delta\omega$ (rad h$^{-1}$)")
    axes[1, 0].set_ylabel("D-V phase gap (h)")
    axes[1, 0].set_title(r"Phase gap versus $\Delta\omega$", fontweight="bold")

    draw_condition_lines(axes[1, 1], coupling, "K_effective_rad_h", "mean_gap_h")
    axes[1, 1].axvline(0, color="#777777", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[1, 1].set_xlabel(r"$K_{eff}$ (rad h$^{-1}$)")
    axes[1, 1].set_ylabel("D-V phase gap (h)")
    axes[1, 1].set_title(r"Phase gap versus $K_{eff}$", fontweight="bold")

    draw_condition_lines(axes[2, 0], intrinsic, "delta_omega_rad_h", "mean_DV_period_h")
    axes[2, 0].set_xlabel(r"$\Delta\omega$ (rad h$^{-1}$)")
    axes[2, 0].set_ylabel("Mean D/V period (h)")
    axes[2, 0].set_title(r"SCN period versus $\Delta\omega$", fontweight="bold")

    draw_condition_lines(axes[2, 1], coupling, "K_effective_rad_h", "mean_DV_period_h")
    axes[2, 1].axvline(0, color="#777777", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[2, 1].set_xlabel(r"$K_{eff}$ (rad h$^{-1}$)")
    axes[2, 1].set_ylabel("Mean D/V period (h)")
    axes[2, 1].set_title(r"SCN period versus $K_{eff}$", fontweight="bold")

    draw_condition_lines(axes[3, 0], intrinsic, "delta_omega_rad_h", "locking_ratio")
    axes[3, 0].set_xlabel(r"$\Delta\omega$ (rad h$^{-1}$)")
    axes[3, 0].set_ylabel(r"$|\Delta\omega/K_{eff}|$")
    axes[3, 0].set_title(r"Locking margin versus $\Delta\omega$", fontweight="bold")

    draw_condition_lines(axes[3, 1], coupling, "K_effective_rad_h", "locking_ratio")
    axes[3, 1].axvline(0, color="#777777", linewidth=0.7, linestyle=(0, (2, 2)))
    axes[3, 1].set_xlabel(r"$K_{eff}$ (rad h$^{-1}$)")
    axes[3, 1].set_ylabel(r"$|\Delta\omega/K_{eff}|$")
    axes[3, 1].set_title(r"Locking margin versus $K_{eff}$", fontweight="bold")

    for ax in axes[3, :]:
        ax.axhline(1, color="black", linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_yscale("log")
        ax.set_ylim(0.02, 100)
    for ax in axes.flat:
        style_axis(ax)
    axes[3, 0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(1.05, -0.43), handlelength=1.8)
    for ax, label in zip(axes.flat, "ABCDEFGH"):
        panel_label(ax, label, -0.18, 1.16)
    save_figure(fig, "figure_5_parameter_sensitivity")
    return rows, summary, derivatives


def main() -> None:
    configure_style()
    params = selected_params()
    print("Simulating fitted conditions...", flush=True)
    inv, exv = simulation_summaries(params)
    write_csv(ROOT / "tables" / "final_in_vivo_summary.csv", inv)
    exv_clean = [
        {
            "condition": r["condition"],
            "SCN_D_period_h": r["SCN_D_period_h"],
            "SCN_V_period_h": r["SCN_V_period_h"],
            "release_gap_h": r["release_gap_h"],
            "mean_gap_h": r["mean_gap_h"],
            "mean_tau_D_h": r["mean_tau_D_h"],
            "target_exvivo_scn_period_h": r["target_exvivo_scn_period_h"],
            "target_gap_h": r["target_gap_h"],
        }
        for r in exv
    ]
    write_csv(ROOT / "tables" / "final_ex_vivo_SCN_summary.csv", exv_clean)
    print("Drawing Figure 2...", flush=True)
    figure1(params, inv, exv)
    print("Drawing Figure 3...", flush=True)
    figure2(params, exv)
    print("Drawing Figure 4...", flush=True)
    figure3(params, exv)
    print("Drawing Figure 5...", flush=True)
    figure4(params)
    print(f"Done. Figures: {ROOT / 'figures'}", flush=True)


if __name__ == "__main__":
    main()
