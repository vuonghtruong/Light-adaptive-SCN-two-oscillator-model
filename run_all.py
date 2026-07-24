#!/usr/bin/env python3
"""Generate all journal-share tables and six figures from packaged data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIGURE_STEMS = [
    "figure_1_literature_summary",
    "figure_2_model_fit_and_trajectories",
    "figure_3_landscape_stability_relaxation",
    "figure_4_bifurcation_and_predictions",
    "figure_5_parameter_sensitivity",
    "figure_6_graphical_abstract",
]


def write_runtime_versions(refit: bool, quick: bool) -> None:
    fit_script = ROOT / "scripts" / "fit_model.py"
    fit_hash = hashlib.sha256(fit_script.read_bytes()).hexdigest()
    rows = [
        {"item": "python", "version_or_value": platform.python_version()},
        {"item": "python_implementation", "version_or_value": platform.python_implementation()},
        {"item": "platform", "version_or_value": platform.platform()},
    ]
    for package in ("numpy", "numba", "llvmlite", "astropy", "matplotlib"):
        rows.append({"item": package, "version_or_value": importlib.metadata.version(package)})
    rows.extend(
        [
            {"item": "optimizer", "version_or_value": "seeded Latin-hypercube global search + finite-difference local-gradient descent"},
            {"item": "optimizer_script", "version_or_value": "scripts/fit_model.py"},
            {"item": "optimizer_script_sha256", "version_or_value": fit_hash},
            {"item": "random_seed", "version_or_value": "1729"},
            {"item": "default_latin_samples", "version_or_value": "65536"},
            {"item": "default_local_starts", "version_or_value": "32"},
            {"item": "default_local_iterations", "version_or_value": "55"},
            {"item": "default_batch_size", "version_or_value": "1024"},
            {"item": "period_estimator", "version_or_value": "Astropy LombScargle; DD days 0-14; 18-30 h search range; 20 samples per peak"},
            {"item": "run_mode", "version_or_value": "refit" if refit else "archived fitted parameters"},
            {"item": "quick_mode", "version_or_value": str(bool(quick))},
        ]
    )
    with (ROOT / "tables" / "runtime_software_and_optimizer_versions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "version_or_value"])
        writer.writeheader()
        writer.writerows(rows)


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def clear_outputs() -> None:
    for directory in (ROOT / "tables", ROOT / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()


def write_manifest() -> None:
    rows = []
    for number, stem in enumerate(FIGURE_STEMS, start=1):
        for suffix in ("svg", "pdf", "png"):
            path = ROOT / "figures" / f"{stem}.{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "figure": number,
                    "stem": stem,
                    "format": suffix,
                    "relative_path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                }
            )
    with (ROOT / "tables" / "generated_figure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true", help="Rerun parameter fitting before figure generation.")
    parser.add_argument("--quick", action="store_true", help="Use a short deterministic refit for pipeline testing.")
    parser.add_argument("--threads", type=int, default=0, help="Numba threads used only with --refit.")
    args = parser.parse_args()

    clear_outputs()
    if args.refit:
        if args.quick:
            latin, starts, iterations, top_n, batch = 2048, 4, 10, 50, 512
        else:
            latin, starts, iterations, top_n, batch = 65536, 32, 55, 200, 1024
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "fit_model.py"),
                "--seed", "1729",
                "--latin-samples", str(latin),
                "--local-starts", str(starts),
                "--local-iters", str(iterations),
                "--top-n", str(top_n),
                "--batch-size", str(batch),
                "--threads", str(args.threads),
            ]
        )
    else:
        shutil.copyfile(
            ROOT / "data" / "selected_model_parameters.csv",
            ROOT / "tables" / "systemic_x_selected_params.csv",
        )

    write_runtime_versions(args.refit, args.quick)

    run([sys.executable, str(ROOT / "scripts" / "make_literature_figure.py")])
    run([sys.executable, str(ROOT / "scripts" / "make_model_figures.py")])
    run([sys.executable, str(ROOT / "scripts" / "make_graphical_abstract.py")])
    write_manifest()
    print(f"\nGenerated six figures in {ROOT / 'figures'}", flush=True)
    print(f"Generated supporting tables in {ROOT / 'tables'}", flush=True)


if __name__ == "__main__":
    main()
