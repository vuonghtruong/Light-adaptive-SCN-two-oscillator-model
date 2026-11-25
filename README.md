# Light-adaptive SCN aftereffects model

Code, numerical inputs, figures and CSV tables for the model by Vuong and
Myung (JBR-26-0241). The model contains dorsal (`D`) and ventral (`V`) SCN
phase oscillators coupled to a systemic phase oscillator (`X`), with
light-history-dependent adaptation.

The [preprint](https://www.biorxiv.org/content/10.64898/2026.08.09.743784v1.full)
describes the study; this repository contains the subsequent revised analyses
and publication figures.

## Reproduce the results

Use Python 3.13 and install the pinned dependencies:

```sh
python -m pip install -r requirements-lock.txt
python reproduce.py
```

`reproduce.py` is the single entry point. It selects parameters from all 12
archived finalist pools (three seeds for each of four adaptive architectures),
reruns the forward simulations and sensitivity analyses, and regenerates four
main figures, Figures S1-S6, the technical appendix, the graphical abstract,
and 32 CSV tables. A full rebuild took about nine minutes on the reference
machine; runtime varies. No network connection is needed after installation.

The original candidate-discovery searches and literature extraction are
bundled inputs. They are not repeated by this command. The original search
procedure remains available in `code/fit_protocol.py`.

Optional modes use the same entry point:

```sh
python reproduce.py --cached
python reproduce.py --verify
```

`--cached` reuses forward data only when generator/parameter fingerprints and
content hashes match, while rebuilding figures, tables and numerical checks.
`--verify` additionally rebuilds in an isolated copy containing only source,
dependencies and the 19 inputs listed in `inputs.json`, then compares generated
outputs. No existing outputs are copied into that verification run.

## Package contents

- [Download all 11 publication SVGs](downloads/SCN_all_figures_SVG.zip) for website upload: Figures 1-4, S1-S6 and the graphical abstract. The ZIP contains the same current SVG exports as the figure folders, with flat filenames.
- `code/`: model, analysis, rendering and validation routines.
- `data/`: literature and fitting inputs, plus generated numerical results.
- `figures/main/`: Figures 1-4 as PDF, SVG and PNG.
- `figures/supplement/`: Figures S1-S6 and the technical appendix PDF.
- `figures/graphical_abstract/`: editable SVG, PDF and 600-dpi PNG.
- `tables/`: parameters, targets, residuals, descriptive statistics and sensitivity results.
- `validation/`: numerical checks, extraction provenance and dated reproduction records.
- `inputs.json`: required starting files and their checksums.
- `MANIFEST.sha256`: checksums for the delivered files, before rerunning.

Manuscript files, reviewer responses and private review documents are not
included. Keep `.gitattributes`: it prevents Git from changing line endings in
checksummed files.

## Analysis and display conventions

Primary simulations use the selected adaptive-D fit; architecture comparisons
use their separately selected fits. Selection uses the same penalized objective,
including T22/T26 ex vivo residuals. Figure objective displays show `J/10^6`;
numerical files retain the raw objective. Discrete fitting estimates and refined
display estimates are retained in explicitly named columns.

The 64-member initial-phase ensembles vary initial D/V/X phases with fixed
parameters. They measure sensitivity to initial state, not experimental SEM or
parameter uncertainty. Table and figure exports retain the fixed-initial-state
reference separately from the ensemble.

Behavioral target error bars in Figures 2C, 3F, S1E and S3B use the reported
SEMs: SP 0.04 h, LP 0.05 h, LD 0.10 h, T22 0.10 h and T26 0.02 h.
These are distinct from the fitting scales and penalty thresholds.
Figure 2D/E uses solid D and dashed V curves in its regional-period column;
the gray dashed line in the ratio column marks one.

The literature synthesis contains 31 paired condition means. Correlations are
descriptive, not animal-level estimates. Reporter, study and protocol differ
across datasets. The observations retain original uncertainty types and
conversion provenance where available; plotted errors use SEM. For current
source assignments and extraction corrections, see the literature validation
records and `tables/Table_S1_literature_provenance.csv`.

## Current figures and validation

Figures use 175 mm width, stronger axes and curves, and a consistent condition
palette across model panels. The literature figure retains distinct age and
reporter encodings. PDF/SVG retain vector elements; PNGs use 600 dpi. Figure 2
retains the equations and model connections from the author's Inkscape edits,
with enlarged panel A and four columns in each of D/E. The graphical abstract
uses current model trajectories and endpoints.

Figure 2A and the graphical abstract explicitly show `M_X -> tau_X`. X has no
direct additive photic phase-forcing term, but its memory acquisition retains
explicit schedule variables. Their interpretation as proxies for SCN-regulated
systemic cues does not make the implemented law exclusively SCN-mediated.

The November 2026 literature metadata update corrects Myung (2012) Bmal1
behavioral sample size, converts four reported regional SDs to SEM, and corrects
Tackenberg (2020) ex vivo animal counts and exposure metadata. It does not change
the 31 paired means, their correlations, selected parameters or model outputs.
See `validation/literature_metadata_revision_2026-11-25.json` for source evidence
and current checks. Earlier dated validation files document their historical
release scope; they are not fresh rebuild claims for later exports.

The reference environment is Python 3.13.5 on Windows. Arial is preferred,
with DejaVu Sans as fallback. Numerical reproducibility does not require Arial,
but byte-identical figure comparisons require matching fonts and package
versions. Matplotlib runs without a display; other operating systems have not
been independently tested.
