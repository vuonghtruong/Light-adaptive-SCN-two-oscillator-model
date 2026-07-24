# Python code for two-oscillator SCN model with adaptive intrinsic period

The set of codes provided here simulate a light-history-dependent phase model and generate figures for the paper by Vuong & Myung. The model describes dynamics of two SCN phase oscillators representing dorsal (`D`) and ventral (`V`) subregions and a third systemic phase oscillator (`X`).

The current set contains simulations from (1) 5 Zeitgeber conditions, (2) summary of parameters from literature, (3) freerunning conditions _in vivo_ and _ex vivo_, and post-hoc analyses such as (4) Lomb–Scargle period estimation, parameter search and sensitivity analysis. workflow. As outputs, six figures are exported as SVG, PDF, and PNG.

`run_all.py` automatically runs through all scripts and produce outputs.
