# Figure S7 and Table S8: dephasing in culture

Run `python code/dephasing.py` from the repository root to reproduce Table S8
and all five plotting CSVs. The same calculation is included in `reproduce.py`.
The supplied Mathematica 13.2 notebook (`FigureS7.nb`) and script
(`FigureS7.wl`) render the four-panel figure from these CSVs. Python does not
rerender the supplied Mathematica PDF/PNG. `Figure_S7.svg` is a vector
conversion of that PDF for website use; its text is outlined.

The synchronization index is the equal-weight two-oscillator resultant,
`SI = abs(cos((phi_D - phi_V)/2))`, not the 0.8-D/0.2-V behavioral readout.
"Intact" in panels C/D means intact D-V coupling **within the explant**:
X-to-SCN feedback is removed in all these culture simulations. "Uncoupled"
means that both D-V coupling terms are additionally set to zero at release.

Fits use `SI(t) = exp(-D_omega * t^2 / 2)` over DD0-7 days with time in days,
so `D_omega` has units rad^2/day^2. This is a descriptive Gaussian-decay fit
to a deterministic two-population model, not a direct estimate of a neuronal
frequency distribution. Two uncoupled oscillators have a cosine resultant,
and the adaptive intrinsic frequencies continue to change in DD. The panel C
"analytic variance" is the squared half-difference of the average absolute
intrinsic angular-frequency mismatch over the sampled window. It is not the
time-average of the instantaneous variance or an exact prediction of the
full coupled trajectory. The eight schedule points are model conditions,
not independent animals.

Table S8 distinguishes this analytic quantity (`Per_D_uncoupled`) from the
fitted value after simulated decoupling (`Per_D_decoupled`). In the Figure S7
CSVs, columns named `D_uncoupled` denote the latter fitted decoupling value.
The Gaussian approximation is poor for nearly synchronized T23
(`Per_R2_Gauss = -2.650`) and less accurate over seven days after T22
(`R2 = 0.722`, versus 0.919 over four days). Do not describe all schedules as
well fitted by this decay law.

The contribution intentionally preserves the revision's rounded Table S2A
parameters and simultaneous forward-Euler update at 1/64 h. The earlier
`architecture_model.py` uses full-precision selected parameters and updates
memory before phases. A review across both architectures and eight schedules
found a maximum SI difference below 0.00034 over seven days. Table S8 and the
five supplied CSVs reproduce exactly with `dephasing.py`; minor last-digit
differences from the earlier integrator are numerical, not new fitted models.
See `validation/dephasing_review_2026-11-26.json` for the comparison.
