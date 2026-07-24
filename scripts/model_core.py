#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from astropy.timeseries import LombScargle

TWOPI = 2 * math.pi
DT = 0.25
MM_TO_PX = 96 / 25.4

CONDITIONS = {
    "SP": (24.0, 8.0),
    "LP": (24.0, 16.0),
    "LD": (24.0, 12.0),
    "T22": (22.0, 11.0),
    "T26": (26.0, 13.0),
}
ORDER = ["SP", "LP", "LD", "T22", "T26"]
TARGET_BEHAVIOUR = {"SP": 23.97, "LP": 23.59, "LD": 23.70, "T22": 22.83, "T26": 24.50}
TARGET_EXVIVO_SCN_PERIOD = {
    "SP": TARGET_BEHAVIOUR["SP"],
    "LP": TARGET_BEHAVIOUR["LP"],
    "LD": TARGET_BEHAVIOUR["LD"],
    "T22": TARGET_BEHAVIOUR["T26"],
    "T26": TARGET_BEHAVIOUR["T22"],
}
TARGET_GAP = {"SP": 0.49, "LP": 1.08, "LD": 0.58, "T22": -2.08, "T26": 2.43}
TARGET_GAP_SEM = {"SP": 0.18, "LP": 0.21, "LD": 0.15, "T22": 0.44, "T26": 0.30}
TARGET_BEHAVIOUR_SEM = {"SP": 0.16, "LP": 0.16, "LD": 0.10, "T22": 0.10, "T26": 0.10}


def _load_empirical_targets() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "model_empirical_targets.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        condition = row["condition"]
        CONDITIONS[condition] = (float(row["cycle_h"]), float(row["light_h"]))
        TARGET_BEHAVIOUR[condition] = float(row["behavior_mean_h"])
        TARGET_BEHAVIOUR_SEM[condition] = float(row["behavior_sem_h"])
        TARGET_EXVIVO_SCN_PERIOD[condition] = float(row["exvivo_scn_target_h"])
        TARGET_GAP[condition] = float(row["gap_mean_h"])
        TARGET_GAP_SEM[condition] = float(row["gap_sem_h"])


_load_empirical_targets()

COLORS = {
    "D": "#D73027",
    "V": "#2C7BB6",
    "X": "#555555",
    "SCN": "#777777",
    "light": "#F6C85F",
    "target": "#555555",
    "SP": "#1B9E77",
    "LP": "#7B3294",
    "LD": "#4D4D4D",
    "T22": "#E6AB02",
    "T26": "#A6761D",
}


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def wrap_pi(x: float) -> float:
    return ((x + math.pi) % TWOPI) - math.pi


def phase_gap_h(phi_d: float, phi_v: float) -> float:
    gap = wrap_pi(phi_d - phi_v) / TWOPI * 24
    if gap > 6:
        gap -= 12
    elif gap < -6:
        gap += 12
    return gap


def light_state(t: float, cycle: float, photo: float, schedule_days: int = 30) -> tuple[float, float, str]:
    if t >= schedule_days * 24:
        return 0.0, 0.0, "DD"
    ph = t % cycle
    return (1.0 if ph < photo else 0.0), TWOPI * ph / cycle, "schedule"


def default_params() -> dict:
    return {
        "tau_d0": 23.8,
        "tau_v": 25.1,
        "tau_x0": 23.8,
        "g_d": 6.0,
        "g_x": 3.5,
        "gphi_d": 1.0,
        "gp_d": 1.0,
        "gphi_x": -1.0,
        "gp_x": 1.0,
        "mem_tau_d": 240.0,
        "mem_tau_x": 240.0,
        "kvd": 0.06,
        "kdv": 0.012,
        "kx_d": 0.07,
        "kx_v": 0.045,
        "ksx": 0.08,
        "phase_bias": -0.2,
        "light_amp": 0.10,
        "v_light_boost": 8.0,
    }


def simulate(
    condition: str,
    params: dict,
    ex_vivo_dd: bool = False,
    adaptation_mode: str = "period",
    coupling_memory_gain: float = 0.0,
) -> list[dict]:
    """Simulate the fitted model or an isolated adaptive-coupling alternative.

    ``period`` retains the fitted adaptive dorsal intrinsic period and fixed
    coupling. ``coupling`` fixes both intrinsic periods and lets the same
    light-history memory alter K_eff = K_VD - K_DV.
    """
    if adaptation_mode not in {"period", "coupling"}:
        raise ValueError("adaptation_mode must be 'period' or 'coupling'")
    cycle, photo = CONDITIONS[condition]
    schedule_days = 30
    dd_days = 14
    total = (schedule_days + dd_days) * 24
    schedule_end = schedule_days * 24
    phi_d = 0.35
    phi_v = 0.0
    phi_x = -0.2
    mem_d = 0.0
    mem_x = 0.0
    rows = []
    for step in range(int(total / DT) + 1):
        t = step * DT
        light, theta, stage = light_state(t, cycle, photo, schedule_days)
        phi_scn = math.atan2(
            0.8 * math.sin(phi_d) + 0.2 * math.sin(phi_v),
            0.8 * math.cos(phi_d) + 0.2 * math.cos(phi_v),
        )
        if stage == "schedule":
            s_d = 0.5 * (1.0 + math.sin(phi_d))
            s_x = 0.5 * (1.0 + math.sin(phi_x))
            drive_d = params["gphi_d"] * light * math.sin(theta - phi_d) + params["gp_d"] * (s_d - light)
            drive_x = params["gphi_x"] * light * math.sin(theta - phi_x) + params["gp_x"] * (s_x - light)
        else:
            drive_d = 0.0
            drive_x = 0.0
        mem_d += DT * ((drive_d - mem_d) / params["mem_tau_d"])
        mem_x += DT * ((drive_x - mem_x) / params["mem_tau_x"])
        if adaptation_mode == "period":
            tau_d = params["tau_d0"] - params["g_d"] * math.tanh(mem_d)
            k_eff = params["kvd"] - params["kdv"]
        else:
            tau_d = params["tau_d0"]
            k_eff = params["kvd"] - params["kdv"] + coupling_memory_gain * math.tanh(mem_d)
        kvd_current = params["kdv"] + k_eff
        tau_v = params["tau_v"]
        tau_x = params["tau_x0"] - params["g_x"] * math.tanh(mem_x)
        tau_d = min(34.0, max(16.0, tau_d))
        tau_x = min(34.0, max(16.0, tau_x))
        rows.append(
            {
                "t_h": t,
                "rel_h": t - schedule_end,
                "stage": stage,
                "light": light,
                "D": -math.sin(phi_d),
                "V": -math.sin(phi_v),
                "X": -math.sin(phi_x),
                "SCN": -math.sin(phi_scn),
                "phi_D": phi_d,
                "phi_V": phi_v,
                "phi_X": phi_x,
                "phi_SCN": phi_scn,
                "gap_h": phase_gap_h(phi_d, phi_v),
                "mem_D": mem_d,
                "mem_X": mem_x,
                "tau_D_h": tau_d,
                "tau_X_h": tau_x,
                "K_effective_rad_h": k_eff,
            }
        )
        x_to_d = 0.0 if ex_vivo_dd and stage == "DD" else params["kx_d"] * math.sin(phi_x - phi_d)
        x_to_v = 0.0 if ex_vivo_dd and stage == "DD" else params["kx_v"] * math.sin(phi_x - phi_v)
        d_d = (
            TWOPI / tau_d
            - params["kdv"] * math.sin(phi_v - phi_d + params["phase_bias"])
            + x_to_d
            + params["light_amp"] * light * math.sin(theta - phi_d)
        )
        d_v = (
            TWOPI / tau_v
            + kvd_current * math.sin(phi_d - phi_v - params["phase_bias"])
            + x_to_v
            + params["v_light_boost"] * params["light_amp"] * light * math.sin(theta - phi_v)
        )
        d_x = TWOPI / tau_x + params["ksx"] * math.sin(phi_scn - phi_x)
        phi_d = (phi_d + DT * d_d) % TWOPI
        phi_v = (phi_v + DT * d_v) % TWOPI
        phi_x = (phi_x + DT * d_x) % TWOPI
    return rows


def lomb_scargle_period(
    rows: list[dict],
    key: str,
    *,
    start_h: float = 0.0,
    end_h: float = 14.0 * 24.0,
    minimum_period_h: float = 18.0,
    maximum_period_h: float = 30.0,
    samples_per_peak: int = 20,
) -> float:
    """Return the dominant Lomb-Scargle period for a DD activity trace."""
    selected = [r for r in rows if r["stage"] == "DD" and start_h <= r["rel_h"] <= end_h]
    if len(selected) < 3:
        return float("nan")
    times = np.asarray([r["rel_h"] for r in selected], dtype=float)
    values = np.asarray([r[key] for r in selected], dtype=float)
    if not np.all(np.isfinite(values)) or np.ptp(values) <= 1e-12:
        return float("nan")
    frequency, power = LombScargle(times, values, center_data=True, fit_mean=True).autopower(
        minimum_frequency=1.0 / maximum_period_h,
        maximum_frequency=1.0 / minimum_period_h,
        samples_per_peak=samples_per_peak,
    )
    return float(1.0 / frequency[int(np.nanargmax(power))])


def summarize(params: dict, ex_vivo_dd: bool = False) -> list[dict]:
    rows = []
    for condition in ORDER:
        trace = simulate(condition, params, ex_vivo_dd)
        dd = [r for r in trace if r["stage"] == "DD"]
        sched = [r for r in trace if r["stage"] == "schedule"]
        gap_n = min(len(dd), int(7 * 24 / DT) + 1)
        rows.append(
            {
                "condition": condition,
                "context": "ex_vivo_release" if ex_vivo_dd else "in_vivo",
                "behaviour_period_h": lomb_scargle_period(dd, "SCN"),
                "SCN_D_period_h": lomb_scargle_period(dd, "D"),
                "SCN_V_period_h": lomb_scargle_period(dd, "V"),
                "systemic_X_period_h": lomb_scargle_period(dd, "X"),
                "release_gap_h": dd[0]["gap_h"],
                "mean_gap_h": sum(r["gap_h"] for r in dd[:gap_n]) / gap_n,
                "mean_tau_D_h": sum(r["tau_D_h"] for r in dd[:gap_n]) / gap_n,
                "mean_tau_X_h": sum(r["tau_X_h"] for r in dd[:gap_n]) / gap_n,
                "schedule_mem_D": sum(r["mem_D"] for r in sched[-int(7 * 24 / DT):]) / int(7 * 24 / DT),
                "schedule_mem_X": sum(r["mem_X"] for r in sched[-int(7 * 24 / DT):]) / int(7 * 24 / DT),
                "target_behaviour_h": TARGET_BEHAVIOUR[condition] if not ex_vivo_dd else "",
                "target_exvivo_scn_period_h": TARGET_EXVIVO_SCN_PERIOD[condition] if ex_vivo_dd else "",
                "target_gap_h": TARGET_GAP[condition] if ex_vivo_dd else "",
            }
        )
    return rows


def score_params(params: dict) -> float:
    inv = summarize(params, False)
    exv = summarize(params, True)
    inv_by = {r["condition"]: r for r in inv}
    exv_by = {r["condition"]: r for r in exv}
    score = 0.0
    for r in inv:
        sigma = 0.16 if r["condition"] in ["SP", "LP", "LD"] else 0.25
        score += 2.0 * ((r["behaviour_period_h"] - r["target_behaviour_h"]) / sigma) ** 2
    for r in exv:
        for key in ["SCN_D_period_h", "SCN_V_period_h"]:
            sigma = 0.22 if r["condition"] in ["SP", "LP", "LD"] else 0.20
            weight = 1.8 if r["condition"] in ["SP", "LP", "LD"] else 3.0
            score += weight * ((r[key] - r["target_exvivo_scn_period_h"]) / sigma) ** 2
        gap_sigma = TARGET_GAP_SEM[r["condition"]]
        score += 30.0 * ((r["mean_gap_h"] - r["target_gap_h"]) / gap_sigma) ** 2
        gap_abs_err = abs(r["mean_gap_h"] - r["target_gap_h"])
        gap_tol = 0.45 if r["condition"] in ["SP", "LP", "LD"] else 0.70
        if gap_abs_err > gap_tol:
            score += 5000 + 3000 * (gap_abs_err - gap_tol) ** 2
    # Photoperiod aftereffects must remain visible while fitting T-cycles:
    # LP should be shorter than LD, and SP should be longer than LD.
    if inv_by["LP"]["behaviour_period_h"] >= inv_by["LD"]["behaviour_period_h"] - 0.05:
        score += 700 + 500 * (inv_by["LP"]["behaviour_period_h"] - inv_by["LD"]["behaviour_period_h"] + 0.05) ** 2
    if inv_by["SP"]["behaviour_period_h"] <= inv_by["LD"]["behaviour_period_h"] + 0.10:
        score += 700 + 500 * (inv_by["LD"]["behaviour_period_h"] + 0.10 - inv_by["SP"]["behaviour_period_h"]) ** 2
    for key in ["SCN_D_period_h", "SCN_V_period_h"]:
        if exv_by["LP"][key] >= exv_by["LD"][key] - 0.05:
            score += 500 + 300 * (exv_by["LP"][key] - exv_by["LD"][key] + 0.05) ** 2
        if exv_by["SP"][key] <= exv_by["LD"][key] + 0.10:
            score += 500 + 300 * (exv_by["LD"][key] + 0.10 - exv_by["SP"][key]) ** 2
    if not (exv_by["LP"]["mean_tau_X_h"] < exv_by["LD"]["mean_tau_X_h"] < exv_by["SP"]["mean_tau_X_h"]):
        score += 5000
    if exv_by["LP"]["mean_tau_X_h"] - exv_by["LP"]["mean_tau_D_h"] > 0.45:
        score += 2500 + 1000 * (exv_by["LP"]["mean_tau_X_h"] - exv_by["LP"]["mean_tau_D_h"] - 0.45) ** 2
    # Desired in-vivo T-cycle behavioural direction.
    if inv_by["T22"]["behaviour_period_h"] >= inv_by["LD"]["behaviour_period_h"] - 0.15:
        score += 900
    if inv_by["T26"]["behaviour_period_h"] <= inv_by["LD"]["behaviour_period_h"] + 0.35:
        score += 900
    # Desired ex-vivo phase organization.
    if exv_by["SP"]["mean_gap_h"] <= 0.05 or exv_by["SP"]["mean_gap_h"] > 0.85:
        score += 300
    if exv_by["LP"]["mean_gap_h"] <= exv_by["SP"]["mean_gap_h"] + 0.25:
        score += 250
    if exv_by["T22"]["mean_gap_h"] >= -1.20:
        score += 900 + 500 * (exv_by["T22"]["mean_gap_h"] + 1.20) ** 2
    if exv_by["T26"]["mean_gap_h"] <= exv_by["SP"]["mean_gap_h"] + 0.50:
        score += 300
    # Ex vivo SCN period direction: T22 should lengthen D and V relative to LD,
    # whereas T26 should shorten D and V relative to LD.
    for key in ["SCN_D_period_h", "SCN_V_period_h"]:
        if exv_by["T22"][key] <= exv_by["LD"][key] + 0.20:
            score += 5000 + 1000 * (exv_by["LD"][key] + 0.20 - exv_by["T22"][key]) ** 2
        if exv_by["T26"][key] >= exv_by["LD"][key] - 0.20:
            score += 5000 + 1000 * (exv_by["T26"][key] - exv_by["LD"][key] + 0.20) ** 2
    # Key requested memory relationship:
    # SP/LP/LD: X and D memories same sign; T22/T26: opposite signs.
    # This corresponds to shared photoperiod gain (g_P) but opposite
    # phase-of-light gain (g_phi) between D and X.
    if params["g_d"] <= 0 or params["g_x"] <= 0:
        score += 5000
    if abs(params["gp_x"] - params["gp_d"]) > 0.35:
        score += 5000 + 1000 * (abs(params["gp_x"] - params["gp_d"]) - 0.35) ** 2
    if params["gphi_d"] * params["gphi_x"] >= 0:
        score += 5000
    for cond in ["SP", "LP", "LD"]:
        if exv_by[cond]["schedule_mem_D"] * exv_by[cond]["schedule_mem_X"] <= 0:
            score += 5000
    for cond in ["T22", "T26"]:
        if exv_by[cond]["schedule_mem_D"] * exv_by[cond]["schedule_mem_X"] >= 0:
            score += 700
    return score


def fit_params() -> tuple[dict, list[dict]]:
    base = default_params()
    records = []
    baseline_grid = [
        (23.8, 25.1, 24.2),
    ]
    memory_gain_grid = [(-1.5, 1.0, 1.5, 1.0), (1.0, -1.5, -1.0, -1.5), (1.5, 1.5, -1.5, 1.5)]
    phase_bias_grid = [-0.525]
    v_light_boost_grid = [8.0, 16.0, 24.0, 32.0]
    for tau_d0, tau_v, tau_x0 in baseline_grid:
        for gphi_d, gp_d, gphi_x, gp_x in memory_gain_grid:
                for phase_bias in phase_bias_grid:
                    for g_d in [5.0, 7.0, 9.0, 11.0, 14.0]:
                        for g_x in [5.0, 7.0, 10.0, 14.0]:
                            for v_light_boost in v_light_boost_grid:
                                for kvd, kdv in [(0.04, 0.006)]:
                                    for kx_d, kx_v, ksx in [(0.04, 0.03, 0.05)]:
                                        p = dict(base)
                                        p.update(
                                            {
                                                "tau_d0": tau_d0,
                                                "tau_v": tau_v,
                                                "tau_x0": tau_x0,
                                                "gphi_d": gphi_d,
                                                "gp_d": gp_d,
                                                "gphi_x": gphi_x,
                                                "gp_x": gp_x,
                                                "phase_bias": phase_bias,
                                                "g_d": g_d,
                                                "g_x": g_x,
                                                "kvd": kvd,
                                                "kdv": kdv,
                                                "kx_d": kx_d,
                                                "kx_v": kx_v,
                                                "ksx": ksx,
                                                "v_light_boost": v_light_boost,
                                            }
                                        )
                                        s = score_params(p)
                                        records.append({"score": round(s, 4), **p})
    records.sort(key=lambda r: r["score"])
    return records[0], records


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SVG:
    def __init__(self, path: Path, width_mm: float = 175, height_mm: float = 130):
        self.path = path
        self.w = width_mm * MM_TO_PX
        self.h = height_mm * MM_TO_PX
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" viewBox="0 0 {self.w:.2f} {self.h:.2f}">',
            '<style>text{font-family:Helvetica,Arial,sans-serif;font-size:9px;fill:#000}.panel{font-size:18px;font-weight:700}.title{font-weight:700}.tiny{font-size:7px}</style>',
        ]

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, anchor="middle", klass=""):
        attrs = f'x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}"'
        if klass:
            attrs += f' class="{klass}"'
        self.add(f"<text {attrs}>{s}</text>")

    def line(self, x1, y1, x2, y2, color="#000", width=1, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, fill, opacity=1.0):
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" opacity="{opacity}"/>')

    def circle(self, x, y, r, fill, stroke="none"):
        self.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}"/>')

    def path(self, pts, color, width=1.2):
        if not pts:
            return
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def save(self):
        self.parts.append("</svg>")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.parts), encoding="utf-8")


def ymap(v, y, h, ymin, ymax):
    return y + h - (v - ymin) / (ymax - ymin) * h


def draw_bar_panel(svg: SVG, rows: list[dict], x: float, y: float, w: float, h: float, title: str, target_key: str):
    svg.text(x + w / 2, y - 10, title, "middle", "title")
    ymin, ymax = 21.5, 26.0
    svg.line(x, y + h, x + w, y + h, "#333", 1)
    svg.line(x, y, x, y + h, "#333", 1)
    for tick in [22, 24, 26]:
        yy = ymap(tick, y, h, ymin, ymax)
        svg.line(x, yy, x + w, yy, "#e5e5e5", 1)
        svg.text(x - 8, yy + 3, str(tick), "end", "tiny")
    bw = 7
    for i, cond in enumerate(ORDER):
        r = next(rr for rr in rows if rr["condition"] == cond)
        cx = x + 24 + i * 38
        vals = [("behaviour_period_h", COLORS["SCN"], -9), ("SCN_D_period_h", COLORS["D"], 0), ("SCN_V_period_h", COLORS["V"], 9)]
        for key, color, dx in vals:
            yy = ymap(r[key], y, h, ymin, ymax)
            svg.rect(cx + dx - bw / 2, yy, bw, y + h - yy, color, 0.85)
        target = r.get(target_key, "")
        if target != "":
            ty = ymap(float(target), y, h, ymin, ymax)
            svg.line(cx - 15, ty, cx + 15, ty, COLORS["target"], 1, "3 2")
        svg.text(cx, y + h + 14, cond, "middle", "tiny")
    svg.text(x - 26, y + h / 2, "period (h)", "middle", "tiny")


def draw_memory_panel(svg: SVG, rows: list[dict], x: float, y: float, w: float, h: float):
    svg.text(x + w / 2, y - 10, "Schedule memory: SCN D vs X", "middle", "title")
    ymin, ymax = -0.35, 0.20
    svg.line(x, ymap(0, y, h, ymin, ymax), x + w, ymap(0, y, h, ymin, ymax), "#999", 1)
    svg.line(x, y, x, y + h, "#333", 1)
    svg.line(x, y + h, x + w, y + h, "#333", 1)
    for tick in [-0.3, 0.0, 0.2]:
        yy = ymap(tick, y, h, ymin, ymax)
        svg.line(x, yy, x + w, yy, "#e5e5e5", 0.8)
        svg.text(x - 8, yy + 3, f"{tick:.1f}", "end", "tiny")
    for i, cond in enumerate(ORDER):
        r = next(rr for rr in rows if rr["condition"] == cond)
        cx = x + 25 + i * 38
        for key, color, dx in [("schedule_mem_D", COLORS["D"], -5), ("schedule_mem_X", COLORS["X"], 5)]:
            val = r[key]
            yy = ymap(max(0, val), y, h, ymin, ymax)
            y0 = ymap(0, y, h, ymin, ymax)
            y1 = ymap(val, y, h, ymin, ymax)
            svg.rect(cx + dx - 4, min(y0, y1), 8, abs(y1 - y0), color, 0.85)
        svg.text(cx, y + h + 14, cond, "middle", "tiny")
    svg.text(x - 22, y + h / 2, "memory", "middle", "tiny")


def draw_gap_panel(svg: SVG, rows: list[dict], x: float, y: float, w: float, h: float):
    svg.text(x + w / 2, y - 10, "Ex vivo DD D-V gap", "middle", "title")
    ymin, ymax = -3.2, 3.2
    svg.line(x, ymap(0, y, h, ymin, ymax), x + w, ymap(0, y, h, ymin, ymax), "#bbb", 1)
    svg.line(x, y, x, y + h, "#333", 1)
    svg.line(x, y + h, x + w, y + h, "#333", 1)
    for tick in [-3, 0, 3]:
        yy = ymap(tick, y, h, ymin, ymax)
        svg.line(x, yy, x + w, yy, "#e5e5e5", 0.8)
        svg.text(x - 8, yy + 3, str(tick), "end", "tiny")
    for i, cond in enumerate(ORDER):
        r = next(rr for rr in rows if rr["condition"] == cond)
        cx = x + 25 + i * 38
        svg.circle(cx, ymap(r["mean_gap_h"], y, h, ymin, ymax), 3.5, COLORS[cond])
        svg.line(cx - 10, ymap(r["target_gap_h"], y, h, ymin, ymax), cx + 10, ymap(r["target_gap_h"], y, h, ymin, ymax), "#555", 1, "3 2")
        svg.text(cx, y + h + 14, cond, "middle", "tiny")
    svg.text(x - 22, y + h / 2, "gap (h)", "middle", "tiny")


def draw_figure(path: Path, inv: list[dict], exv: list[dict]):
    svg = SVG(path)
    svg.text(16, 24, "A", "middle", "panel")
    draw_bar_panel(svg, inv, 42, 42, 190, 110, "In vivo periods", "target_behaviour_h")
    svg.text(258, 24, "B", "middle", "panel")
    draw_memory_panel(svg, exv, 282, 42, 190, 110)
    svg.text(16, 205, "C", "middle", "panel")
    draw_gap_panel(svg, exv, 42, 220, 190, 105)
    svg.text(300, 205, "D", "middle", "panel")
    draw_bar_panel(svg, exv, 326, 220, 190, 105, "Ex vivo periods", "target_exvivo_scn_period_h")
    # legend
    y = 455
    x = 175
    for label, color in [("Behaviour/SCN", COLORS["SCN"]), ("D", COLORS["D"]), ("V", COLORS["V"]), ("X memory", COLORS["X"])]:
        svg.rect(x, y - 8, 9, 9, color, 0.85)
        svg.text(x + 14, y, label, "start", "tiny")
        x += 85
    svg.save()


def add_poly(svg: SVG, pts: list[tuple[float, float]], fill: str, opacity: float = 0.35, stroke: str = "none", width: float = 0.7) -> None:
    if not pts:
        return
    p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    svg.add(f'<polygon points="{p}" fill="{fill}" opacity="{opacity}" stroke="{stroke}" stroke-width="{width}"/>')


def add_line_path(svg: SVG, pts: list[tuple[float, float]], color: str, width: float = 0.7, opacity: float = 1.0) -> None:
    if not pts:
        return
    d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    svg.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>')


def light_segments(rows: list[dict]) -> list[tuple[float, float]]:
    segs = []
    start = None
    last = None
    for r in rows:
        if r["light"] > 0:
            if start is None:
                start = r["rel_h"]
            last = r["rel_h"]
        elif start is not None:
            segs.append((start, last + DT))
            start = None
    if start is not None and last is not None:
        segs.append((start, last + DT))
    return segs


def draw_condition_doubleplot(svg: SVG, trace: list[dict], condition: str, x: float, y: float, w: float, h: float, show_x: bool) -> None:
    row_count = 21
    row_h = h / row_count
    amp = row_h * 0.70
    xscale = w / 48.0
    svg.text(x + w / 2, y - 9, condition, "middle", "title")
    segs = light_segments(trace)
    row_starts = [(-14 + i) * 24 for i in range(14)] + [i * 24 for i in range(7)]
    row_labels = [f"S{i}" for i in range(1, 15)] + [f"DD{i}" for i in range(1, 8)]
    for idx, start_h in enumerate(row_starts):
        base = y + idx * row_h + row_h * 0.83
        if idx == 14:
            svg.line(x - 2, base - row_h * 0.95, x + w, base - row_h * 0.95, "#666", 0.8, "3 2")
        if idx in [0, 13, 14, 20]:
            svg.text(x - 5, base + 2, row_labels[idx], "end", "tiny")
        svg.line(x, base, x + w, base, "#d8d8d8", 0.35)
        for a, b in segs:
            aa = max(a, start_h)
            bb = min(b, start_h + 48)
            if bb > aa:
                svg.rect(x + (aa - start_h) * xscale, base - amp, (bb - aa) * xscale, amp, COLORS["light"], 0.38)
        rows = [r for r in trace if start_h <= r["rel_h"] <= start_h + 48]
        for key, color, alpha, lw in [
            ("D", COLORS["D"], 0.28, 0.70),
            ("V", COLORS["V"], 0.28, 0.70),
            ("X", COLORS["X"], 0.16, 0.55),
        ]:
            if key == "X" and not show_x:
                continue
            top = []
            line = []
            for r in rows:
                xx = x + (r["rel_h"] - start_h) * xscale
                val = max(0.0, r[key])
                yy = base - amp * val
                top.append((xx, yy))
                line.append((xx, yy))
            if top:
                poly = [(top[0][0], base)] + top + [(top[-1][0], base)]
                add_poly(svg, poly, color, alpha)
                add_line_path(svg, line, color, lw)
        for win_start in [start_h, start_h + 24]:
            win = [r for r in rows if win_start <= r["rel_h"] < win_start + 24]
            if win:
                peak = max(win, key=lambda r: r["SCN"])
                svg.circle(x + (peak["rel_h"] - start_h) * xscale, base - amp * 1.04, 1.45, "#111")


def draw_doubleplot_figure(path: Path, params: dict, ex_vivo_dd: bool) -> None:
    svg = SVG(path, 175, 125)
    title = "Ex vivo release double plots" if ex_vivo_dd else "In vivo double plots"
    svg.text(svg.w / 2, 18, title, "middle", "title")
    col_w = 112
    gap = 18
    start_x = 42
    top_y = 44
    plot_h = 370
    for i, cond in enumerate(ORDER):
        trace = simulate(cond, params, ex_vivo_dd)
        draw_condition_doubleplot(svg, trace, cond, start_x + i * (col_w + gap), top_y, col_w, plot_h, show_x=not ex_vivo_dd)
    y = 455
    x = 165
    items = [("Light", COLORS["light"]), ("D", COLORS["D"]), ("V", COLORS["V"])]
    if not ex_vivo_dd:
        items.append(("X", COLORS["X"]))
    for label, color in items:
        svg.rect(x, y - 8, 10, 10, color, 0.65 if label != "Light" else 0.38)
        svg.text(x + 14, y, label, "start", "tiny")
        x += 58
    svg.circle(x + 4, y - 4, 2.0, "#111")
    svg.text(x + 12, y, "Behaviour peak", "start", "tiny")
    svg.save()



