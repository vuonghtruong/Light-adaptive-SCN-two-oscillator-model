#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import model_core as reporting_model

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This fitting script requires numba. Install with: pip install numba") from exc

TWOPI = 2.0 * math.pi
DT = 0.25
ORDER = ["SP", "LP", "LD", "T22", "T26"]
CYCLES = tuple(reporting_model.CONDITIONS[c][0] for c in ORDER)
PHOTOPERIODS = tuple(reporting_model.CONDITIONS[c][1] for c in ORDER)
TARGET_BEHAVIOUR = tuple(reporting_model.TARGET_BEHAVIOUR[c] for c in ORDER)
TARGET_EXVIVO_SCN = tuple(reporting_model.TARGET_EXVIVO_SCN_PERIOD[c] for c in ORDER)
TARGET_GAP = tuple(reporting_model.TARGET_GAP[c] for c in ORDER)
TARGET_GAP_SEM = (0.18, 0.21, 0.15, 0.44, 0.30)

PARAM_NAMES = [
    "tau_d0",
    "tau_v",
    "tau_x0",
    "g_d",
    "g_x",
    "gphi_d",
    "gp_d",
    "gphi_x",
    "gp_x",
    "kvd",
    "kdv",
    "kx_d",
    "kx_v",
    "ksx",
    "phase_bias",
    "light_amp",
    "v_light_boost",
]

PARAM_BOUNDS = np.asarray(
    [
        (21.5, 24.5),      # tau_d0
        (23.0, 25.8),      # tau_v
        (20.5, 28.0),      # tau_x0
        (1.0, 40.0),       # g_d: D period response to memory
        (1.0, 40.0),       # g_x: X period response to memory, same photoperiod polarity as D
        (-5.0, 5.0),       # gphi_d: phase-of-light memory gain for D
        (-6.0, 6.0),       # gp_d: photoperiod mismatch memory gain for D
        (-5.0, 5.0),       # gphi_x: phase-of-light memory gain for X
        (-6.0, 6.0),       # gp_x: photoperiod mismatch memory gain for X
        (0.010, 0.075),    # K_VD, attractive D-to-V coupling
        (0.002, 0.050),    # K_DV, repulsive V-to-D coupling
        (0.000, 0.50),    # K_XD
        (0.000, 0.30),    # K_XV
        (0.000, 0.50),    # K_SX
        (0.0, 0.0),       # phase_bias fixed to zero
        (0.005, 0.5),     # light_amp
        (1.0, 60.0),      # v_light_boost
    ],
    dtype=np.float64,
)

# Mechanistically calibrated candidate retained from the architecture screen.
# It is evaluated by exactly the same simulations and final diagnostic score as
# every newly sampled candidate. Including it makes the published fit
# reproducible while still allowing a better Latin/local candidate to replace
# it when the full search finds one.
CALIBRATED_CANDIDATE = np.asarray(
    [
        23.959188627026233, 24.032896157458122, 23.424059198396847,
        7.994471106801864, 33.61079695293627,
        -1.8901263880436137, 0.0582106248676755,
        1.912620030092253, -0.37865675723483405,
        0.016, 0.012424951558233189,
        0.3985972506714376, 0.19778097531322766,
        0.3654192872470772, 0.0,
        0.3394456257150647, 18.350755904385736,
    ],
    dtype=np.float64,
)


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@njit(cache=True)
def wrap_pi(x: float) -> float:
    return ((x + math.pi) % TWOPI) - math.pi


@njit(cache=True)
def phase_gap_h(phi_d: float, phi_v: float) -> float:
    gap = wrap_pi(phi_d - phi_v) / TWOPI * 24.0
    if gap > 6.0:
        gap -= 12.0
    elif gap < -6.0:
        gap += 12.0
    return gap


@njit(cache=True)
def phase_slope_period_surrogate(n: int, sx: float, sy: float, sxx: float, sxy: float) -> float:
    den = n * sxx - sx * sx
    if n < 3 or den <= 0.0:
        return math.nan
    slope = (n * sxy - sx * sy) / den
    if slope <= 0.0:
        return math.nan
    return TWOPI / slope


@njit(cache=True)
def simulate_summary(condition_idx: int, p: tuple, ex_vivo_dd: bool) -> tuple:
    tau_d0, tau_v, tau_x0, g_d, g_x, gphi_d, gp_d, gphi_x, gp_x, kvd, kdv, kx_d, kx_v, ksx, phase_bias, light_amp, v_light_boost = p
    cycle = CYCLES[condition_idx]
    photo = PHOTOPERIODS[condition_idx]
    schedule_days = 30.0
    schedule_end = schedule_days * 24.0
    total = (30.0 + 14.0) * 24.0
    steps = int(total / DT) + 1
    phi_d = 0.35
    phi_v = 0.0
    phi_x = -0.2
    mem_d = 0.0
    mem_x = 0.0
    release_gap = 0.0
    release_set = False
    sched_mem_d_sum = 0.0
    sched_mem_x_sum = 0.0
    sched_mem_n = 0
    mean_gap_sum = 0.0
    mean_tau_d_sum = 0.0
    mean_tau_x_sum = 0.0
    mean_n = 0

    n_d = n_v = n_x = n_s = 0
    sx_d = sy_d = sxx_d = sxy_d = 0.0
    sx_v = sy_v = sxx_v = sxy_v = 0.0
    sx_x = sy_x = sxx_x = sxy_x = 0.0
    sx_s = sy_s = sxx_s = sxy_s = 0.0
    prev_d = prev_v = prev_x = prev_s = 0.0
    off_d = off_v = off_x = off_s = 0.0
    have_prev = False

    for step in range(steps):
        t = step * DT
        stage_dd = t >= schedule_end
        rel_h = t - schedule_end
        ph = t % cycle
        light = 0.0
        theta = 0.0
        if not stage_dd:
            theta = TWOPI * ph / cycle
            if ph < photo:
                light = 1.0
        phi_s = math.atan2(0.8 * math.sin(phi_d) + 0.2 * math.sin(phi_v), 0.8 * math.cos(phi_d) + 0.2 * math.cos(phi_v))
        if not stage_dd:
            s_d = 0.5 * (1.0 + math.sin(phi_d))
            s_x = 0.5 * (1.0 + math.sin(phi_x))
            drive_d = gphi_d * light * math.sin(theta - phi_d) + gp_d * (s_d - light)
            drive_x = gphi_x * light * math.sin(theta - phi_x) + gp_x * (s_x - light)
        else:
            drive_d = 0.0
            drive_x = 0.0
        mem_d += DT * ((drive_d - mem_d) / 240.0)
        mem_x += DT * ((drive_x - mem_x) / 240.0)
        tau_d = tau_d0 - g_d * math.tanh(mem_d)
        tau_x = tau_x0 - g_x * math.tanh(mem_x)
        if tau_d < 16.0:
            tau_d = 16.0
        elif tau_d > 34.0:
            tau_d = 34.0
        if tau_x < 16.0:
            tau_x = 16.0
        elif tau_x > 34.0:
            tau_x = 34.0

        if not stage_dd and rel_h >= -7.0 * 24.0:
            sched_mem_d_sum += mem_d
            sched_mem_x_sum += mem_x
            sched_mem_n += 1
        if stage_dd:
            if not release_set:
                release_gap = phase_gap_h(phi_d, phi_v)
                release_set = True
            if rel_h <= 7.0 * 24.0:
                mean_gap_sum += phase_gap_h(phi_d, phi_v)
                mean_tau_d_sum += tau_d
                mean_tau_x_sum += tau_x
                mean_n += 1
            # Fast Numba-compatible screening surrogate over the complete
            # 14-day DD interval. Finalist selection and every reported period
            # are recomputed with Astropy Lomb-Scargle below.
            if 0.0 <= rel_h <= 14.0 * 24.0:
                if have_prev:
                    jd = phi_d - prev_d
                    jv = phi_v - prev_v
                    jx = phi_x - prev_x
                    js = phi_s - prev_s
                    if jd > math.pi:
                        off_d -= TWOPI
                    elif jd < -math.pi:
                        off_d += TWOPI
                    if jv > math.pi:
                        off_v -= TWOPI
                    elif jv < -math.pi:
                        off_v += TWOPI
                    if jx > math.pi:
                        off_x -= TWOPI
                    elif jx < -math.pi:
                        off_x += TWOPI
                    if js > math.pi:
                        off_s -= TWOPI
                    elif js < -math.pi:
                        off_s += TWOPI
                yd = phi_d + off_d
                yv = phi_v + off_v
                yx = phi_x + off_x
                ys = phi_s + off_s
                sx_d += rel_h; sy_d += yd; sxx_d += rel_h * rel_h; sxy_d += rel_h * yd; n_d += 1
                sx_v += rel_h; sy_v += yv; sxx_v += rel_h * rel_h; sxy_v += rel_h * yv; n_v += 1
                sx_x += rel_h; sy_x += yx; sxx_x += rel_h * rel_h; sxy_x += rel_h * yx; n_x += 1
                sx_s += rel_h; sy_s += ys; sxx_s += rel_h * rel_h; sxy_s += rel_h * ys; n_s += 1
                prev_d = phi_d; prev_v = phi_v; prev_x = phi_x; prev_s = phi_s
                have_prev = True

        x_to_d = 0.0
        x_to_v = 0.0
        if not (ex_vivo_dd and stage_dd):
            x_to_d = kx_d * math.sin(phi_x - phi_d)
            x_to_v = kx_v * math.sin(phi_x - phi_v)
        d_d = TWOPI / tau_d - kdv * math.sin(phi_v - phi_d + phase_bias) + x_to_d + light_amp * light * math.sin(theta - phi_d)
        d_v = TWOPI / tau_v + kvd * math.sin(phi_d - phi_v - phase_bias) + x_to_v + v_light_boost * light_amp * light * math.sin(theta - phi_v)
        d_x = TWOPI / tau_x + ksx * math.sin(phi_s - phi_x)
        phi_d = (phi_d + DT * d_d) % TWOPI
        phi_v = (phi_v + DT * d_v) % TWOPI
        phi_x = (phi_x + DT * d_x) % TWOPI

    if mean_n < 1:
        mean_n = 1
    if sched_mem_n < 1:
        sched_mem_n = 1
    return (
        phase_slope_period_surrogate(n_s, sx_s, sy_s, sxx_s, sxy_s),
        phase_slope_period_surrogate(n_d, sx_d, sy_d, sxx_d, sxy_d),
        phase_slope_period_surrogate(n_v, sx_v, sy_v, sxx_v, sxy_v),
        phase_slope_period_surrogate(n_x, sx_x, sy_x, sxx_x, sxy_x),
        release_gap,
        mean_gap_sum / mean_n,
        mean_tau_d_sum / mean_n,
        mean_tau_x_sum / mean_n,
        sched_mem_d_sum / sched_mem_n,
        sched_mem_x_sum / sched_mem_n,
    )


@njit(cache=True)
def model_score(p: tuple) -> float:
    kvd = p[9]
    kdv = p[10]
    g_d = p[3]
    g_x = p[4]
    gphi_d = p[5]
    gp_d = p[6]
    gphi_x = p[7]
    gp_x = p[8]
    inv = [simulate_summary(i, p, False) for i in range(5)]
    exv = [simulate_summary(i, p, True) for i in range(5)]
    score = 0.0
    # Architecture hypothesis: X should share the photoperiod/day-length memory
    # branch with D but oppose the phase-of-light branch that dominates
    # non-24 h T-cycle adaptation.
    if g_d <= 0.0 or g_x <= 0.0:
        score += 1000000.0
    # D and X are required to share the observable photoperiod response,
    # rather than numerically identical g_P coefficients. Their subjective-day
    # phases differ, so opposite coefficient signs can yield the same
    # LP < LD < SP period ordering.
    if gp_d <= 0.0 or gp_x >= 0.0:
        score += 1000000.0
    if gphi_d * gphi_x >= 0.0:
        score += 1000000.0 + 100000.0 * (gphi_d * gphi_x) ** 2
    gphi_mag_diff = abs(gphi_x) - abs(gphi_d)
    score += 1200.0 * gphi_mag_diff * gphi_mag_diff
    for i in range(5):
        sigma_b = 0.08 if i < 3 else 0.18
        b_weight = 600.0 if i < 3 else 500.0
        score += b_weight * ((inv[i][0] - TARGET_BEHAVIOUR[i]) / sigma_b) ** 2
        b_abs_err = abs(inv[i][0] - TARGET_BEHAVIOUR[i])
        b_tol = 0.14 if i < 3 else 0.25
        if b_abs_err > b_tol:
            score += 3000000.0 + 1200000.0 * (b_abs_err - b_tol) ** 2
        if i < 3:
            # For standard photoperiod histories, removing systemic feedback
            # during DD should not create a large in vivo/ex vivo SCN-period
            # split. The systemic counteraction hypothesis is T-cycle specific.
            sigma_s = 0.08
            weight = 1400.0
            score += weight * ((exv[i][1] - inv[i][1]) / sigma_s) ** 2
            score += weight * ((exv[i][2] - inv[i][2]) / sigma_s) ** 2
            phot_split = max(abs(exv[i][1] - inv[i][1]), abs(exv[i][2] - inv[i][2]))
            if phot_split > 0.22:
                score += 9000000.0 + 4000000.0 * (phot_split - 0.22) ** 2
            score += 900.0 * ((inv[i][3] - inv[i][0]) / 0.08) ** 2
        else:
            sigma_s = 0.16
            weight = 700.0
            score += weight * ((exv[i][1] - TARGET_EXVIVO_SCN[i]) / sigma_s) ** 2
            score += weight * ((exv[i][2] - TARGET_EXVIVO_SCN[i]) / sigma_s) ** 2
            if i == 3:
                # T22: in vivo behaviour is short, but ex vivo SCN should relax
                # toward the opposite/long-period T-cycle aftereffect.
                for key_idx in (1, 2):
                    split = exv[i][key_idx] - inv[i][key_idx]
                    if split < 0.70:
                        score += 2500000.0 + 1200000.0 * (0.70 - split) ** 2
            elif i == 4:
                # T26: in vivo behaviour is long, but ex vivo SCN should relax
                # toward the opposite/short-period T-cycle aftereffect.
                for key_idx in (1, 2):
                    split = inv[i][key_idx] - exv[i][key_idx]
                    if split < 0.70:
                        score += 2500000.0 + 1200000.0 * (0.70 - split) ** 2
        gap_sigma = TARGET_GAP_SEM[i]
        gap_weight = 140.0
        score += gap_weight * ((exv[i][5] - TARGET_GAP[i]) / gap_sigma) ** 2
        score += gap_weight * ((exv[i][4] - TARGET_GAP[i]) / gap_sigma) ** 2
        gap_abs_err = max(abs(exv[i][5] - TARGET_GAP[i]), abs(exv[i][4] - TARGET_GAP[i]))
        gap_tol = 0.45 if i < 3 else 0.70
        if gap_abs_err > gap_tol:
            score += 200000.0 + 150000.0 * (gap_abs_err - gap_tol) ** 2

    if not (inv[1][0] < inv[2][0] < inv[0][0]):
        score += 1000000.0
    if inv[2][0] - inv[1][0] < 0.09:
        score += 1000000.0 + 600000.0 * (0.09 - (inv[2][0] - inv[1][0])) ** 2
    if inv[0][0] - inv[2][0] < 0.18:
        score += 1000000.0 + 600000.0 * (0.18 - (inv[0][0] - inv[2][0])) ** 2
    if not (inv[3][0] < inv[2][0] < inv[4][0]):
        score += 1000000.0
    if not (inv[1][3] < inv[2][3] < inv[0][3]):
        score += 1000000.0
    score += 5000.0 * ((inv[0][0] - inv[2][0]) - 0.27) ** 2
    score += 5000.0 * ((inv[2][0] - inv[1][0]) - 0.11) ** 2
    # Ex vivo photoperiod should keep the same LP < LD < SP direction.
    # Without this explicit direction term, the broad parameter bounds can
    # find good T-cycle fits by flipping the photoperiod aftereffect.
    for key_idx in (0, 1, 2):
        lp = exv[1][key_idx]
        ld = exv[2][key_idx]
        sp = exv[0][key_idx]
        if not (lp < ld < sp):
            score += 1000000.0
        if ld - lp < 0.08:
            score += 3500.0 + 3000.0 * (0.08 - (ld - lp)) ** 2
        if sp - ld < 0.12:
            score += 3500.0 + 3000.0 * (0.12 - (sp - ld)) ** 2
    if not (exv[1][6] < exv[2][6] < exv[0][6]):
        score += 2500.0
    # X should follow the SCN photoperiod output. The systemic counteracting
    # role is reserved for T-cycles, not SP/LP/LD.
    if not (exv[1][7] < exv[2][7] < exv[0][7]):
        score += 1000000.0
    if exv[1][7] - exv[1][6] > 0.45:
        score += 100000.0 + 80000.0 * (exv[1][7] - exv[1][6] - 0.45) ** 2
    if abs((exv[0][7] - exv[2][7]) - (exv[0][6] - exv[2][6])) > 0.75:
        score += 30000.0
    # The systemic oscillator should express the T22 shortening more strongly
    # than the SCN readout; otherwise X cannot rescue in-vivo behaviour.
    if inv[3][3] >= inv[2][3] - 0.50:
        score += 5000.0 + 3000.0 * (inv[3][3] - inv[2][3] + 0.50) ** 2
    if inv[4][3] <= inv[2][3] + 0.50:
        score += 5000.0 + 3000.0 * (inv[2][3] + 0.50 - inv[4][3]) ** 2
    if exv[3][5] >= -1.20:
        score += 6000000.0 + 2000000.0 * (exv[3][5] + 1.20) ** 2
    # Phase organization during ex vivo DD:
    # SP, LP, LD, and T26 should have D leading V; only T22 should have V leading D.
    for gap_idx in (0, 1, 2, 4):
        if exv[gap_idx][5] <= 0.0:
            score += 1000000.0 + 100000.0 * exv[gap_idx][5] ** 2
    if exv[4][5] < 1.40:
        score += 3500000.0 + 1600000.0 * (1.40 - exv[4][5]) ** 2
    if exv[3][5] >= 0.0:
        score += 8000000.0 + 2500000.0 * exv[3][5] ** 2
    # In the T22 ex-vivo state, ventral SCN should run faster than dorsal SCN
    # so that the V lead can persist rather than collapsing immediately in DD.
    if exv[3][2] >= exv[3][1]:
        score += 8000000.0 + 2500000.0 * (exv[3][2] - exv[3][1]) ** 2
    elif exv[3][1] - exv[3][2] < 0.18:
        score += 1500000.0 + 800000.0 * (0.18 - (exv[3][1] - exv[3][2])) ** 2
    for key_idx in (1, 2):
        if exv[3][key_idx] <= exv[2][key_idx] + 0.20:
            score += 5000.0 + 1000.0 * (exv[2][key_idx] + 0.20 - exv[3][key_idx]) ** 2
        if exv[4][key_idx] >= exv[2][key_idx] - 0.20:
            score += 1000000.0 + 100000.0 * (exv[4][key_idx] - exv[2][key_idx] + 0.20) ** 2
    if exv[3][8] * exv[3][9] >= 0.0:
        score += 700.0
    if exv[4][8] * exv[4][9] >= 0.0:
        score += 700.0
    # Avoid the previous over-synchronized solution: K_VD can be modest, but
    # should not dominate K_DV so strongly that all DD phase gaps collapse.
    if kvd - kdv > 0.045:
        score += 500000.0 + 500000.0 * (kvd - kdv - 0.045) ** 2
    return score


@njit(parallel=True, cache=True)
def score_batch(params: np.ndarray) -> np.ndarray:
    scores = np.empty(params.shape[0], dtype=np.float64)
    for i in prange(params.shape[0]):
        scores[i] = model_score(params[i])
    return scores


def unit_to_params(unit_params: np.ndarray) -> np.ndarray:
    lo = PARAM_BOUNDS[:, 0]
    hi = PARAM_BOUNDS[:, 1]
    params = lo + np.clip(unit_params, 0.0, 1.0) * (hi - lo)
    return enforce_memory_architecture(params)


def enforce_memory_architecture(params: np.ndarray) -> np.ndarray:
    """Encode the intended D/X memory logic in sampled candidates.

    Photoperiod adaptation is shared at the response level: D and X are
    required to produce the same LP < LD < SP period ordering. Because their
    phases enter S_j-I differently, the useful coefficient convention is
    positive gp_d and negative gp_x. T-cycle counteraction uses opposite
    phase-sensitive gains.
    """
    out = np.array(params, dtype=np.float64, copy=True)
    one_dim = out.ndim == 1
    if one_dim:
        out = out.reshape(1, -1)
    # Same period-response polarity, so same-sign photoperiod memory produces
    # the same LP/LD/SP direction in D and X.
    out[:, 4] = np.abs(out[:, 4])
    out[:, 3] = np.abs(out[:, 3])
    out[:, 6] = np.abs(out[:, 6])
    out[:, 8] = -np.abs(out[:, 8])
    out[:, 5] = -np.abs(out[:, 5])
    out[:, 7] = np.abs(out[:, 7])
    if one_dim:
        return out[0]
    return out


def params_to_unit(params: np.ndarray) -> np.ndarray:
    lo = PARAM_BOUNDS[:, 0]
    hi = PARAM_BOUNDS[:, 1]
    width = hi - lo
    unit = np.full_like(params, 0.5, dtype=np.float64)
    variable = width > 0.0
    unit[variable] = (params[variable] - lo[variable]) / width[variable]
    return np.clip(unit, 0.0, 1.0)


def latin_hypercube(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_dim = len(PARAM_NAMES)
    unit = np.empty((n_samples, n_dim), dtype=np.float64)
    for j in range(n_dim):
        strata = (np.arange(n_samples, dtype=np.float64) + rng.random(n_samples)) / n_samples
        rng.shuffle(strata)
        unit[:, j] = strata
    return unit_to_params(unit)


def score_candidates(params_array: np.ndarray, batch_size: int, label: str) -> np.ndarray:
    scores = np.empty(params_array.shape[0], dtype=np.float64)
    start = time.time()
    for lo in range(0, params_array.shape[0], batch_size):
        hi = min(lo + batch_size, params_array.shape[0])
        scores[lo:hi] = score_batch(np.asarray(params_array[lo:hi], dtype=np.float64))
        done = hi
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (params_array.shape[0] - done) / rate if rate > 0 else 0.0
        print(
            f"[{label}] scored {done}/{params_array.shape[0]} "
            f"({100.0 * done / params_array.shape[0]:.1f}%), "
            f"elapsed {elapsed:.1f}s, eta {remaining:.1f}s",
            flush=True,
        )
    return scores


def local_gradient_descent(
    start_params: np.ndarray,
    start_score: float,
    *,
    start_index: int,
    n_iter: int,
    batch_size: int,
    step0: float,
    eps: float,
) -> tuple[np.ndarray, float, list[dict]]:
    x = params_to_unit(start_params)
    best_score = float(start_score)
    step = step0
    history: list[dict] = []
    n_dim = len(PARAM_NAMES)
    for iteration in range(1, n_iter + 1):
        probes = np.empty((2 * n_dim, n_dim), dtype=np.float64)
        for j in range(n_dim):
            xp = x.copy()
            xm = x.copy()
            xp[j] = min(1.0, xp[j] + eps)
            xm[j] = max(0.0, xm[j] - eps)
            probes[2 * j] = xp
            probes[2 * j + 1] = xm
        probe_scores = score_batch(unit_to_params(probes))
        grad = np.empty(n_dim, dtype=np.float64)
        for j in range(n_dim):
            denom = max(eps, abs(probes[2 * j, j] - probes[2 * j + 1, j]) / 2.0)
            grad[j] = (probe_scores[2 * j] - probe_scores[2 * j + 1]) / (2.0 * denom)
        grad_norm = float(np.linalg.norm(grad))
        if not np.isfinite(grad_norm) or grad_norm <= 1e-12:
            print(f"[local {start_index}] iter {iteration}: gradient vanished; stop", flush=True)
            break
        accepted = False
        trial_step = step
        trial_score = best_score
        trial_x = x
        for _ in range(8):
            candidate_x = np.clip(x - trial_step * grad / grad_norm, 0.0, 1.0)
            candidate_score = float(score_batch(unit_to_params(candidate_x.reshape(1, -1)))[0])
            if candidate_score < best_score:
                trial_score = candidate_score
                trial_x = candidate_x
                accepted = True
                break
            trial_step *= 0.5
        if accepted:
            x = trial_x
            best_score = trial_score
            step = min(trial_step * 1.2, 0.20)
            status = "accepted"
        else:
            step *= 0.5
            status = "rejected"
        history.append({"local_start": start_index, "iteration": iteration, "score": best_score, "step": step, "status": status})
        print(
            f"[local {start_index}] iter {iteration}/{n_iter}: "
            f"score={best_score:.3f}, step={step:.5f}, {status}",
            flush=True,
        )
        if step < 1e-4:
            break
    return unit_to_params(x), best_score, history


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(params: tuple, ex_vivo: bool, inv_reference: list[tuple] | None = None) -> list[dict]:
    param_dict = {name: float(params[i]) for i, name in enumerate(PARAM_NAMES)}
    param_dict["mem_tau_d"] = 240.0
    param_dict["mem_tau_x"] = 240.0
    return reporting_model.summarize(param_dict, ex_vivo)


def direction_rows(inv: list[dict], exv: list[dict]) -> list[dict]:
    ib = {r["condition"]: r for r in inv}
    eb = {r["condition"]: r for r in exv}
    return [
        {"check": "in vivo photoperiod: LP < LD < SP", "value": f"{ib['LP']['behaviour_period_h']:.3f} < {ib['LD']['behaviour_period_h']:.3f} < {ib['SP']['behaviour_period_h']:.3f}", "passed": ib["LP"]["behaviour_period_h"] < ib["LD"]["behaviour_period_h"] < ib["SP"]["behaviour_period_h"]},
        {"check": "in vivo T-cycle: T22 < LD < T26", "value": f"{ib['T22']['behaviour_period_h']:.3f} < {ib['LD']['behaviour_period_h']:.3f} < {ib['T26']['behaviour_period_h']:.3f}", "passed": ib["T22"]["behaviour_period_h"] < ib["LD"]["behaviour_period_h"] < ib["T26"]["behaviour_period_h"]},
        {"check": "ex vivo photoperiod D: LP < LD < SP", "value": f"{eb['LP']['SCN_D_period_h']:.3f} < {eb['LD']['SCN_D_period_h']:.3f} < {eb['SP']['SCN_D_period_h']:.3f}", "passed": eb["LP"]["SCN_D_period_h"] < eb["LD"]["SCN_D_period_h"] < eb["SP"]["SCN_D_period_h"]},
        {"check": "ex vivo photoperiod V: LP < LD < SP", "value": f"{eb['LP']['SCN_V_period_h']:.3f} < {eb['LD']['SCN_V_period_h']:.3f} < {eb['SP']['SCN_V_period_h']:.3f}", "passed": eb["LP"]["SCN_V_period_h"] < eb["LD"]["SCN_V_period_h"] < eb["SP"]["SCN_V_period_h"]},
        {"check": "ex vivo D: T22 > LD > T26", "value": f"{eb['T22']['SCN_D_period_h']:.3f} > {eb['LD']['SCN_D_period_h']:.3f} > {eb['T26']['SCN_D_period_h']:.3f}", "passed": eb["T22"]["SCN_D_period_h"] > eb["LD"]["SCN_D_period_h"] > eb["T26"]["SCN_D_period_h"]},
        {"check": "ex vivo V: T22 > LD > T26", "value": f"{eb['T22']['SCN_V_period_h']:.3f} > {eb['LD']['SCN_V_period_h']:.3f} > {eb['T26']['SCN_V_period_h']:.3f}", "passed": eb["T22"]["SCN_V_period_h"] > eb["LD"]["SCN_V_period_h"] > eb["T26"]["SCN_V_period_h"]},
        {"check": "ex vivo DD phase sign: SP D leads V", "value": f"{eb['SP']['mean_gap_h']:.3f} h", "passed": eb["SP"]["mean_gap_h"] > 0.0},
        {"check": "ex vivo DD phase sign: LP D leads V", "value": f"{eb['LP']['mean_gap_h']:.3f} h", "passed": eb["LP"]["mean_gap_h"] > 0.0},
        {"check": "ex vivo DD phase sign: LD D leads V", "value": f"{eb['LD']['mean_gap_h']:.3f} h", "passed": eb["LD"]["mean_gap_h"] > 0.0},
        {"check": "ex vivo DD phase sign: T26 D leads V", "value": f"{eb['T26']['mean_gap_h']:.3f} h", "passed": eb["T26"]["mean_gap_h"] > 0.0},
        {"check": "ex vivo T22: V period shorter than D", "value": f"V {eb['T22']['SCN_V_period_h']:.3f} h < D {eb['T22']['SCN_D_period_h']:.3f} h", "passed": eb["T22"]["SCN_V_period_h"] < eb["T22"]["SCN_D_period_h"]},
        {"check": "ex vivo DD T22 V-leads-D gap", "value": f"{eb['T22']['mean_gap_h']:.3f} h; target {eb['T22']['target_gap_h']:.3f} h", "passed": eb["T22"]["mean_gap_h"] < -1.2},
    ]


def diagnostic_compromise_score(row: dict) -> float:
    """Select the final reported model from the ranked optimizer candidates.

    The raw fitting score is intentionally large and contains several hard
    penalties. For reporting, we choose the candidate that best satisfies the
    biological diagnostics requested for this model family: behavioural fit,
    modest photoperiod in-vivo/ex-vivo split, correct D-V phase signs, and
    T22 V running shorter than D.
    """
    params = {name: float(row[name]) for name in PARAM_NAMES}
    params["mem_tau_d"] = 240.0
    params["mem_tau_x"] = 240.0
    inv_rows = reporting_model.summarize(params, False)
    exv_rows = reporting_model.summarize(params, True)

    def as_tuple(r: dict) -> tuple:
        return (
            r["behaviour_period_h"], r["SCN_D_period_h"], r["SCN_V_period_h"],
            r["systemic_X_period_h"], r["release_gap_h"], r["mean_gap_h"],
            r["mean_tau_D_h"], r["mean_tau_X_h"], r["schedule_mem_D"], r["schedule_mem_X"],
        )

    inv = [as_tuple(r) for r in inv_rows]
    exv = [as_tuple(r) for r in exv_rows]
    behaviour_error = sum(abs(inv[i][0] - TARGET_BEHAVIOUR[i]) for i in range(5))
    phot_split = max(max(abs(exv[i][1] - inv[i][1]), abs(exv[i][2] - inv[i][2])) for i in range(3))
    t22_d_minus_v = exv[3][1] - exv[3][2]
    sign_fail = sum([exv[0][5] <= 0.0, exv[1][5] <= 0.0, exv[2][5] <= 0.0, exv[4][5] <= 0.0, exv[3][5] >= 0.0])
    # SP may approach synchrony; retain the empirical gap targets for the
    # remaining schedules. Explicit contrast terms prevent LP and LD from
    # collapsing onto the same in-vivo period.
    gap_error = sum(abs(exv[i][5] - TARGET_GAP[i]) for i in range(1, 5))
    contrast_error = abs((inv[0][0] - inv[2][0]) - 0.27) + abs((inv[2][0] - inv[1][0]) - 0.11)
    x_match_error = sum(abs(inv[i][3] - inv[i][0]) for i in range(3))
    return (
        10.0 * behaviour_error
        + 8.0 * phot_split
        + 100.0 * max(0.0, 0.05 - t22_d_minus_v)
        + 20.0 * (1.0 if t22_d_minus_v <= 0.0 else 0.0)
        + 8.0 * max(0.0, exv[3][5] + 0.60)
        + 4.0 * sign_fail
        + 25.0 * contrast_error
        + 5.0 * x_match_error
        + gap_error
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=0, help="Numba worker threads; 0 uses numba default.")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--latin-samples", type=int, default=65536)
    parser.add_argument("--local-starts", type=int, default=32)
    parser.add_argument("--local-iters", type=int, default=55)
    parser.add_argument("--local-step", type=float, default=0.08)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    args = parser.parse_args()
    if args.threads > 0:
        set_num_threads(args.threads)
    root = root_dir()
    print("[fit] method: Latin-hypercube global search + finite-difference local gradient descent", flush=True)
    print(f"[fit] latin samples: {args.latin_samples}", flush=True)
    print(f"[fit] local starts: {args.local_starts}; local iterations/start: {args.local_iters}", flush=True)
    print(f"[fit] numba threads: {get_num_threads()}", flush=True)
    print("[fit] compiling numba kernels...", flush=True)
    _ = score_batch(latin_hypercube(2, args.seed))
    batch_size = max(1, args.batch_size)
    print("[latin] generating samples...", flush=True)
    latin_params = latin_hypercube(max(1, args.latin_samples), args.seed)
    latin_scores = score_candidates(latin_params, batch_size, "latin")
    order = np.argsort(latin_scores)
    local_starts = min(max(1, args.local_starts), latin_params.shape[0])
    local_rows = []
    local_history = []
    for rank, idx in enumerate(order[:local_starts], start=1):
        print(f"[local {rank}] starting from Latin rank {rank}; score={latin_scores[idx]:.3f}", flush=True)
        p_local, s_local, hist = local_gradient_descent(
            latin_params[idx],
            float(latin_scores[idx]),
            start_index=rank,
            n_iter=max(0, args.local_iters),
            batch_size=batch_size,
            step0=args.local_step,
            eps=args.fd_eps,
        )
        local_rows.append({"score": s_local, "stage": "local_gradient", **{name: p_local[j] for j, name in enumerate(PARAM_NAMES)}})
        local_history.extend(hist)
    ranked = []
    keep_latin = min(args.top_n, latin_params.shape[0])
    for idx in order[:keep_latin]:
        ranked.append({"score": float(latin_scores[idx]), "stage": "latin", **{name: latin_params[idx, j] for j, name in enumerate(PARAM_NAMES)}})
    ranked.extend(local_rows)
    ranked.sort(key=lambda r: r["score"])
    calibrated_score = float(model_score(tuple(CALIBRATED_CANDIDATE)))
    calibrated_row = {
        "score": calibrated_score,
        "stage": "calibrated_architecture_candidate",
        **{name: CALIBRATED_CANDIDATE[j] for j, name in enumerate(PARAM_NAMES)},
    }
    selection_pool = ranked[: args.top_n] + [calibrated_row]
    print(f"[selection] recomputing {len(selection_pool)} finalists with 14-day Astropy Lomb-Scargle...", flush=True)
    for row in selection_pool:
        row["diagnostic_compromise_score"] = diagnostic_compromise_score(row)
        row["lomb_scargle_selection_score"] = row["diagnostic_compromise_score"]
    best = min(selection_pool, key=lambda r: float(r.get("diagnostic_compromise_score", math.inf)))
    best_tuple = tuple(float(best[name]) for name in PARAM_NAMES)
    report_rows = sorted(selection_pool, key=lambda r: float(r.get("diagnostic_compromise_score", math.inf)))
    write_csv(root / "tables" / "systemic_x_fit_ranked_numba.csv", report_rows)
    write_csv(
        root / "tables" / "systemic_x_selected_params.csv",
        [{name: best[name] for name in PARAM_NAMES} | {
            "score": best["score"],
            "diagnostic_compromise_score": best.get("diagnostic_compromise_score", ""),
            "lomb_scargle_selection_score": best.get("lomb_scargle_selection_score", ""),
        }],
    )
    if local_history:
        write_csv(root / "tables" / "systemic_x_local_gradient_history.csv", local_history)
    write_csv(
        root / "tables" / "systemic_x_search_settings.csv",
        [
            {"setting": "method", "value": "Numba phase-slope surrogate screening followed by exact 14-day Astropy Lomb-Scargle finalist selection"},
            {"setting": "seed", "value": args.seed},
            {"setting": "latin_samples", "value": args.latin_samples},
            {"setting": "local_starts", "value": local_starts},
            {"setting": "local_iters", "value": args.local_iters},
            {"setting": "local_step", "value": args.local_step},
            {"setting": "fd_eps", "value": args.fd_eps},
            {"setting": "batch_size", "value": batch_size},
            {"setting": "numba_threads", "value": get_num_threads()},
            {"setting": "calibrated_candidate_included", "value": True},
            {"setting": "period_estimator", "value": "Astropy LombScargle, DD days 0-14, 18-30 h search range, 20 samples per peak"},
            {"setting": "phase_gap_window", "value": "arithmetic mean of wrapped D-V gap over DD days 0-7"},
            {"setting": "memory_release_rule", "value": "no reset; M_D and M_X retain schedule-learned values and decay identically in DD for every condition"},
            {"setting": "final_selection", "value": "minimum Lomb-Scargle diagnostic score among surrogate top-N plus calibrated architecture candidate"},
        ],
    )
    bounds_rows = []
    for i, name in enumerate(PARAM_NAMES):
        bounds_rows.append({"parameter": name, "lower": PARAM_BOUNDS[i, 0], "upper": PARAM_BOUNDS[i, 1]})
    write_csv(root / "tables" / "systemic_x_parameter_bounds.csv", bounds_rows)
    print(f"[fit] selected score: {best['score']:.3f} ({best.get('stage', 'unknown')})", flush=True)
    print(root / "tables" / "systemic_x_selected_params.csv")
    print(root / "tables" / "systemic_x_fit_ranked_numba.csv")


if __name__ == "__main__":
    main()
