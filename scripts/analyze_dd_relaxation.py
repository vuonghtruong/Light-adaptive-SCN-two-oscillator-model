#!/usr/bin/env python3
from __future__ import annotations

import math

import model_core as viz

TWOPI = 2.0 * math.pi
DT = viz.DT
WINDOW_DAYS = 5.0
DD_DAYS = 60


def simulate_long(condition: str, params: dict, ex_vivo_dd: bool = False, dd_days: int = DD_DAYS) -> list[dict]:
    cycle, photo = viz.CONDITIONS[condition]
    schedule_days = 30
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
        light, theta, stage = viz.light_state(t, cycle, photo, schedule_days)
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
        tau_d = params["tau_d0"] - params["g_d"] * math.tanh(mem_d)
        tau_x = params["tau_x0"] - params["g_x"] * math.tanh(mem_x)
        tau_d = min(34.0, max(16.0, tau_d))
        tau_x = min(34.0, max(16.0, tau_x))
        rows.append(
            {
                "t_h": t,
                "dd_day": (t - schedule_end) / 24.0,
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
                "gap_h": viz.phase_gap_h(phi_d, phi_v),
                "mem_D": mem_d,
                "mem_X": mem_x,
                "tau_D_h": tau_d,
                "tau_X_h": tau_x,
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
            TWOPI / params["tau_v"]
            + params["kvd"] * math.sin(phi_d - phi_v - params["phase_bias"])
            + x_to_v
            + params["v_light_boost"] * params["light_amp"] * light * math.sin(theta - phi_v)
        )
        d_x = TWOPI / tau_x + params["ksx"] * math.sin(phi_scn - phi_x)
        phi_d = (phi_d + DT * d_d) % TWOPI
        phi_v = (phi_v + DT * d_v) % TWOPI
        phi_x = (phi_x + DT * d_x) % TWOPI
    return rows


def rolling_period(rows: list[dict], key: str, start_day: float, window_days: float = WINDOW_DAYS) -> float:
    start_h = start_day * 24.0
    end_h = (start_day + window_days) * 24.0
    activity_key = {"phi_D": "D", "phi_V": "V", "phi_X": "X", "phi_SCN": "SCN"}.get(key, key)
    return viz.lomb_scargle_period(rows, activity_key, start_h=start_h, end_h=end_h)
