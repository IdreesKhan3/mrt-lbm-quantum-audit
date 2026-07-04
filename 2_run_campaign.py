#!/usr/bin/env python3
"""Run the 18-case MRT channel verification campaign via 1_mrt_guo_channel.py."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SOLVER_FILE = "1_mrt_guo_channel.py"


# --- Case definitions ---

def nu_from_tau(tau):
    """Kinematic viscosity from relaxation time."""
    return (tau - 0.5) / 3.0


def force_from_umax(tau, umax, ny):
    """Body force for target centerline u_max."""
    nu = nu_from_tau(tau)
    return 8.0 * nu * umax / (ny * ny)


def make_case(case_id, group, nx, ny, nz, tau, umax, timesteps, sample_every, initial_condition):
    """Build one campaign case dict."""
    return {
        "case_id": case_id,
        "group": group,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "tau": tau,
        "target_umax": umax,
        "nu": nu_from_tau(tau),
        "force_x": force_from_umax(tau, umax, ny),
        "timesteps": timesteps,
        "sample_every": sample_every,
        "initial_condition": initial_condition,
    }


def build_cases(include_performance=False):
    """Return the full A--E (optional F) case list."""
    cases = []

    cases.append(make_case(
        "A1_baseline_rest", "A_baseline",
        64, 32, 16, 0.8, 0.02, 20000, 200, "rest"
    ))

    cases += [
        make_case("B1_grid16", "B_grid_convergence", 32, 16, 8, 0.8, 0.02, 10000, 200, "parabolic"),
        make_case("B2_grid32", "B_grid_convergence", 64, 32, 16, 0.8, 0.02, 10000, 200, "parabolic"),
        make_case("B3_grid48", "B_grid_convergence", 96, 48, 24, 0.8, 0.02, 15000, 300, "parabolic"),
        make_case("B4_grid64", "B_grid_convergence", 128, 64, 32, 0.8, 0.02, 20000, 400, "parabolic"),
    ]

    cases += [
        make_case("C1_tau055", "C_tau_sweep", 64, 32, 16, 0.55, 0.02, 10000, 200, "parabolic"),
        make_case("C2_tau060", "C_tau_sweep", 64, 32, 16, 0.60, 0.02, 10000, 200, "parabolic"),
        make_case("C3_tau080", "C_tau_sweep", 64, 32, 16, 0.80, 0.02, 10000, 200, "parabolic"),
        make_case("C4_tau100", "C_tau_sweep", 64, 32, 16, 1.00, 0.02, 10000, 200, "parabolic"),
        make_case("C5_tau120", "C_tau_sweep", 64, 32, 16, 1.20, 0.02, 10000, 200, "parabolic"),
        make_case("C6_tau160", "C_tau_sweep", 64, 32, 16, 1.60, 0.02, 10000, 200, "parabolic"),
    ]

    cases += [
        make_case("D1_umax0005", "D_force_sweep", 64, 32, 16, 0.8, 0.005, 10000, 200, "parabolic"),
        make_case("D2_umax0010", "D_force_sweep", 64, 32, 16, 0.8, 0.010, 10000, 200, "parabolic"),
        make_case("D3_umax0020", "D_force_sweep", 64, 32, 16, 0.8, 0.020, 10000, 200, "parabolic"),
        make_case("D4_umax0040", "D_force_sweep", 64, 32, 16, 0.8, 0.040, 10000, 200, "parabolic"),
        make_case("D5_umax0080", "D_force_sweep", 64, 32, 16, 0.8, 0.080, 10000, 200, "parabolic"),
    ]

    cases += [
        make_case("E1_rest", "E_initial_condition", 64, 32, 16, 0.8, 0.02, 20000, 200, "rest"),
        make_case("E2_parabolic", "E_initial_condition", 64, 32, 16, 0.8, 0.02, 10000, 200, "parabolic"),
    ]

    if include_performance:
        cases += [
            make_case("F1_perf_grid16", "F_performance_optional", 32, 16, 8, 0.8, 0.02, 5000, 500, "parabolic"),
            make_case("F2_perf_grid32", "F_performance_optional", 64, 32, 16, 0.8, 0.02, 5000, 500, "parabolic"),
            make_case("F3_perf_grid48", "F_performance_optional", 96, 48, 24, 0.8, 0.02, 5000, 500, "parabolic"),
            make_case("F4_perf_grid64", "F_performance_optional", 128, 64, 32, 0.8, 0.02, 5000, 500, "parabolic"),
        ]

    return cases


# --- Execution ---

def result_exists(outdir):
    """True if case folder already has .json and .npz outputs."""
    outdir = Path(outdir)
    return any(outdir.glob("*.json")) and any(outdir.glob("*.npz"))


def write_case_file(case, outdir):
    """Write case_input.json into the case directory."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(outdir) / "case_input.json", "w") as f:
        json.dump(case, f, indent=2)


def run_case(case, solver_file, results_dir, overwrite=False):
    """Invoke the channel solver for one case."""
    outdir = Path(results_dir) / case["case_id"]

    if result_exists(outdir) and not overwrite:
        print(f"skip {case['case_id']}")
        return

    write_case_file(case, outdir)

    cmd = [
        sys.executable,
        solver_file,
        "--nx", str(case["nx"]),
        "--ny", str(case["ny"]),
        "--nz", str(case["nz"]),
        "--tau", str(case["tau"]),
        "--force-x", f"{case['force_x']:.12g}",
        "--timesteps", str(case["timesteps"]),
        "--sample-every", str(case["sample_every"]),
        "--initial-condition", case["initial_condition"],
        "--outdir", str(outdir),
    ]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def print_cases(cases):
    """Print a one-line summary per case."""
    for c in cases:
        print(
            c["case_id"],
            c["group"],
            f"{c['nx']}x{c['ny']}x{c['nz']}",
            "tau=", c["tau"],
            "nu=", f"{c['nu']:.8g}",
            "force_x=", f"{c['force_x']:.8g}",
            "steps=", c["timesteps"],
            "init=", c["initial_condition"],
        )


# --- CLI ---

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run MRT channel verification campaign."
    )
    parser.add_argument("--solver", default=SOLVER_FILE)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--only-case", default=None)
    parser.add_argument("--only-group", default=None)
    parser.add_argument("--include-performance", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = build_cases(args.include_performance)

    if args.only_case is not None:
        cases = [c for c in cases if c["case_id"] == args.only_case]

    if args.only_group is not None:
        cases = [c for c in cases if c["group"] == args.only_group]

    print_cases(cases)

    if args.dry_run:
        return

    for case in cases:
        run_case(case, args.solver, args.results_dir, args.overwrite)


if __name__ == "__main__":
    main()
