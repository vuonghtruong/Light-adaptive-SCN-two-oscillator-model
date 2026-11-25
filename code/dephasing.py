"""Dephasing analysis of ex vivo SCN explants (D-V synchronization index in culture).

Applies the Kuramoto synchronization index SI(t) = |(e^{i phi_D} + e^{i phi_V}) / 2| = |cos(delta/2)|
and the quadratic decay SI(t) = exp(-D_omega * t^2 / 2) of uncoupled oscillators with normally distributed
periods (Myung et al., 2012, Eqs. 9-11) over DD days 0 to 7 for both the period-adaptive and coupling-adaptive architectures,
with intact ex vivo coupling and under simulated pharmacological decoupling (K_VD = K_DV = 0).

Outputs (paths from the repository root; run as python code/dephasing.py):
  - tables/Table_S8_dephasing.csv (exact values for Table S8)
  - figure_S7/*.csv (data of every graph of Figure S7, which figure_S7/FigureS7.nb plots in Mathematica)
"""
import csv
import math
import os
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

TWO_PI = 2.0 * math.pi

# Selected parameters from Table S2A
P_PER = dict(
    tau_d0=24.3542813, tau_v=24.6780351, tau_x0=22.4605033,
    g_d=4.38464917, g_x=38.4798241,
    gphi_d=-4.96349524, gp_d=0.433721606, gphi_x=4.49667677, gp_x=-1.58857513,
    kvd=0.0198082491, kdv=0.0177020595, kxd=0.17056853, kxv=0.133853986, ksx=0.12940891,
    A=0.414848277, bv=4.61430224,
)

P_COUP = dict(
    tau_d0=24.1094513, tau_v=24.3863637, tau_x0=22.2699766,
    g_k=0.341608633, g_x=16.0345339,
    gphi_d=-1.3971757, gp_d=2.9934759, gphi_x=3.94857559, gp_x=-4.99698896,
    kvd0=0.0684068543, kdv=0.0432272838, kxd=0.116291296, kxv=0.209928435, ksx=0.198279622,
    A=0.0643597321, bv=13.6219793,
)

ORDER = ["SP", "LD", "LP", "18:6", "T22", "T23", "T25", "T26"]
DISPLAY = {"SP": "SP", "LD": "LD", "LP": "LP", "18:6": "18:6", "T22": "T22", "T23": "T23", "T25": "T25", "T26": "T26"}
SCHEDULES = {
    "SP": (24.0, 8.0),
    "LD": (24.0, 12.0),
    "LP": (24.0, 16.0),
    "18:6": (24.0, 18.0),
    "T22": (22.0, 11.0),
    "T23": (23.0, 11.5),
    "T25": (25.0, 12.5),
    "T26": (26.0, 13.0),
}

# Exact Okabe-Ito palette matching Figures 3, 4, S1, S2, S5, S6
COLORS = {
    "SP": "#0072B2",
    "LD": "#525252",
    "LP": "#D55E00",
    "18:6": "#7B3294",
    "T22": "#009E73",
    "T23": "#56B4E9",
    "T25": "#E69F00",
    "T26": "#CC79A7",
}


def simulate(p, cycle, photo, mode="period", explant=True, decouple=False, dt=1.0 / 64.0, dd_days=7.0):
    d, v, x, md, mx = 0.35, 0.0, -0.2, 0.0, 0.0
    release = int(round(720.0 / dt))
    total = int(round((720.0 + 24.0 * dd_days) / dt))
    stride = int(round(0.25 / dt))
    rows = []
    sin, cos, tanh, atan2 = math.sin, math.cos, math.tanh, math.atan2
    for i in range(total + 1):
        t = i * dt
        scn = atan2(0.8 * sin(d) + 0.2 * sin(v), 0.8 * cos(d) + 0.2 * cos(v))
        light = 1.0 if (t < 720.0 and (t % cycle) < photo) else 0.0
        theta = TWO_PI * (t % cycle) / cycle
        tau_d = min(34.0, max(16.0, p["tau_d0"] - p["g_d"] * tanh(md))) if mode == "period" else p["tau_d0"]
        tau_x = min(34.0, max(16.0, p["tau_x0"] - p["g_x"] * tanh(mx)))
        kvd = p["kvd"] if mode == "period" else (p["kvd0"] + p["g_k"] * tanh(md))
        kdv = p["kdv"]
        if explant and decouple and i >= release:
            kvd = 0.0
            kdv = 0.0
        if i >= release and (i - release) % stride == 0:
            rows.append((t - 720.0, d, v, tau_d, p["tau_v"]))
        s = 0.0 if (explant and i >= release) else 1.0
        dd = TWO_PI / tau_d - kdv * sin(v - d) + s * p["kxd"] * sin(x - d) + p["A"] * light * sin(theta - d)
        dv = TWO_PI / p["tau_v"] + kvd * sin(d - v) + s * p["kxv"] * sin(x - v) + p["bv"] * p["A"] * light * sin(theta - v)
        dx = TWO_PI / tau_x + p["ksx"] * sin(scn - x)
        if t < 720.0:
            in_d = p["gphi_d"] * light * sin(theta - d) + p["gp_d"] * (0.5 * (1.0 + sin(d)) - light)
            in_x = p["gphi_x"] * light * sin(theta - x) + p["gp_x"] * (0.5 * (1.0 + sin(x)) - light)
        else:
            in_d = 0.0
            in_x = 0.0
        d += dt * dd
        v += dt * dv
        x += dt * dx
        md += dt * (in_d - md) / 240.0
        mx += dt * (in_x - mx) / 240.0
    return np.array(rows)


def fit_myung2012(t_h, si):
    td = t_h / 24.0
    td2 = td ** 2
    res_g = minimize_scalar(
        lambda D: float(np.mean((si - np.exp(-0.5 * D * td2)) ** 2)),
        bounds=(0.0, 5.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    D_fit = float(res_g.x)
    pred_g = np.exp(-0.5 * D_fit * td2)
    ss_tot = float(np.sum((si - np.mean(si)) ** 2))
    r2_g = 1.0 - float(np.sum((si - pred_g) ** 2)) / max(ss_tot, 1e-15)
    rmse_g = float(np.sqrt(np.mean((si - pred_g) ** 2)))
    return dict(D=D_fit, pred_g=pred_g, r2_g=r2_g, rmse_g=rmse_g)


def run_analysis(out_dir, table_path=None):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    csv_rows = []
    for name in ORDER:
        cyc, ph = SCHEDULES[name]
        p_in = simulate(P_PER, cyc, ph, "period", explant=True, decouple=False)
        p_dc = simulate(P_PER, cyc, ph, "period", explant=True, decouple=True)
        c_in = simulate(P_COUP, cyc, ph, "coupling", explant=True, decouple=False)
        c_dc = simulate(P_COUP, cyc, ph, "coupling", explant=True, decouple=True)

        td = p_in[:, 0] / 24.0
        gap_pin = p_in[:, 1] - p_in[:, 2]
        gap_pdc = p_dc[:, 1] - p_dc[:, 2]
        gap_cin = c_in[:, 1] - c_in[:, 2]
        gap_cdc = c_dc[:, 1] - c_dc[:, 2]

        si_pin = np.abs(np.cos(gap_pin / 2.0))
        si_pdc = np.abs(np.cos(gap_pdc / 2.0))
        si_cin = np.abs(np.cos(gap_cin / 2.0))
        si_cdc = np.abs(np.cos(gap_cdc / 2.0))
        si_cdc_shift = np.abs(np.cos((gap_cdc - gap_cdc[0]) / 2.0))

        fp_in = fit_myung2012(p_in[:, 0], si_pin)
        m4 = p_in[:, 0] <= 96.0 + 1e-9
        fp_in4 = fit_myung2012(p_in[m4, 0], si_pin[m4])
        fp_dc = fit_myung2012(p_dc[:, 0], si_pdc)
        fc_in = fit_myung2012(c_in[:, 0], si_cin)
        fc_dc = fit_myung2012(c_dc[:, 0], si_cdc)
        fc_dc_shift = fit_myung2012(c_dc[:, 0], si_cdc_shift)

        # Analytic two-point frequency variance (average |Delta omega| over DD days 0-7)
        dw_avg = float(np.mean(np.abs(TWO_PI / p_in[:, 3] - TWO_PI / p_in[:, 4])))
        D_unc_per = ((dw_avg * 24.0) / 2.0) ** 2
        dw_coup = abs(TWO_PI / P_COUP["tau_d0"] - TWO_PI / P_COUP["tau_v"])
        D_unc_coup = ((dw_coup * 24.0) / 2.0) ** 2

        results[name] = dict(
            td=td, si_pin=si_pin, si_pdc=si_pdc, si_cin=si_cin, si_cdc=si_cdc,
            fp_in=fp_in, fp_in4=fp_in4, fp_dc=fp_dc, fc_in=fc_in, fc_dc=fc_dc, fc_dc_shift=fc_dc_shift,
            D_unc_per=D_unc_per, D_unc_coup=D_unc_coup,
        )
        csv_rows.append(dict(
            Schedule=DISPLAY[name],
            Per_SI_d7=f"{si_pin[-1]:.4f}",
            Per_D_intact=f"{fp_in['D']:.5f}",
            Per_D_uncoupled=f"{D_unc_per:.5f}",
            Per_R2_Gauss=f"{fp_in['r2_g']:.3f}",
            Per_D_decoupled=f"{fp_dc['D']:.5f}",
            Coup_SI_d7=f"{si_cin[-1]:.4f}",
            Coup_D_intact=f"{fc_in['D']:.5f}",
            Coup_D_decoupled=f"{fc_dc['D']:.5f}",
        ))

    # Summary verification diagnostics
    x_unc = np.array([results[k]["D_unc_per"] for k in ORDER])
    y_fit = np.array([results[k]["fp_in"]["D"] for k in ORDER])
    r_corr = float(np.corrcoef(x_unc, y_fit)[0, 1])
    rho_corr = float(spearmanr(x_unc, y_fit).statistic)
    ratios = [results[k]["fp_dc"]["D"] / results[k]["fp_in"]["D"] for k in ORDER if results[k]["fp_in"]["D"] > 1e-4]
    si0_per = [results[k]["si_pin"][0] for k in ORDER]
    si0_coup = [results[k]["si_cin"][0] for k in ORDER]
    sh_coup = [results[k]["fc_dc_shift"]["D"] for k in ORDER]
    print(f"Period-adaptive SI(0) range: {min(si0_per):.5f} to {max(si0_per):.5f}")
    print(f"Coupling-adaptive SI(0) range: {min(si0_coup):.5f} to {max(si0_coup):.5f}")
    print(f"T22 Gaussian R2: 0-4d={results['T22']['fp_in4']['r2_g']:.3f}, 0-7d={results['T22']['fp_in']['r2_g']:.3f}")
    print(f"Intact vs analytic variance: Pearson r={r_corr:.6f}, Spearman rho={rho_corr:.4f}")
    print(f"Decoupled/intact D_omega ratio (period-adaptive, non-T23): min={min(ratios):.3f}, max={max(ratios):.3f}, mean={np.mean(ratios):.3f}")
    print(f"Coupling-adaptive fixed variance={results['LD']['D_unc_coup']:.5f}, shifted decoupled D_omega={min(sh_coup):.5f}..{max(sh_coup):.5f} rad^2/d^2")

    csv_path = table_path or os.path.join(out_dir, "table_S8_dephasing.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Wrote {csv_path}")

    # ------------------------------------------------------------------ Figure S7 data (plotted in Mathematica)
    fig_dir = os.path.join(out_dir, "figure_S7")
    os.makedirs(fig_dir, exist_ok=True)
    names = [DISPLAY[k] for k in ORDER]
    for tag, key_si, key_fit in (("S7A_period_adaptive_SI", "si_pin", "fp_in"), ("S7B_coupling_adaptive_SI", "si_cin", "fc_in")):
        with open(os.path.join(fig_dir, tag + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["day"] + [n + suffix for n in names for suffix in ("_sim", "_fit")])
            td = results[ORDER[0]]["td"]
            for k in range(len(td)):
                row = [f"{td[k]:.6f}"]
                for name in ORDER:
                    r = results[name]
                    row += [f"{r[key_si][k]:.6f}", f"{r[key_fit]['pred_g'][k]:.6f}"]
                w.writerow(row)
    with open(os.path.join(fig_dir, "S7C_dispersion_period_adaptive.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "analytic_variance", "D_intact", "D_uncoupled"])
        for name in ORDER:
            r = results[name]
            w.writerow([DISPLAY[name], f"{r['D_unc_per']:.6f}", f"{r['fp_in']['D']:.6f}", f"{r['fp_dc']['D']:.6f}"])
    with open(os.path.join(fig_dir, "S7D_dispersion_by_schedule.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "family", "period_intact", "period_uncoupled", "coupling_intact", "coupling_uncoupled",
                    "fit_drawn_coupling"])
        for name in ORDER:
            r = results[name]
            fam = "T-cycle" if name.startswith("T") else "Photoperiod"
            w.writerow([DISPLAY[name], fam, f"{r['fp_in']['D']:.6g}", f"{r['fp_dc']['D']:.6g}",
                        f"{r['fc_in']['D']:.6g}", f"{r['fc_dc']['D']:.6g}", int(r["fc_in"]["D"] > 0.001)])
    with open(os.path.join(fig_dir, "S7_constants.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        w.writerow(["coupling_adaptive_fixed_variance", f"{results['LD']['D_unc_coup']:.6f}"])
        w.writerow(["pearson_r_intact", f"{r_corr:.4f}"])
        w.writerow(["spearman_rho_intact", f"{rho_corr:.4f}"])
    print(f"Wrote Figure S7 data to {fig_dir}")
    return results, csv_rows


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
    run_analysis(root, table_path=os.path.join(root, "tables", "Table_S8_dephasing.csv"))
