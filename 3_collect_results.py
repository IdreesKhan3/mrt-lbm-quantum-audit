#!/usr/bin/env python3
"""Aggregate per-case solver outputs into paper_tables/*.csv."""

import argparse
import csv
import json
from pathlib import Path


TABLE_DIR = "paper_tables"


FIELDS = [
    "case_id",
    "group",
    "nx",
    "ny",
    "nz",
    "tau",
    "nu",
    "target_umax",
    "force_x",
    "timesteps",
    "sample_every",
    "initial_condition",
    "l2_profile_error",
    "max_profile_error",
    "wall_slip_extrapolated",
    "near_wall_node_ux_max",
    "wall_normal_mean_max",
    "max_mass_relative_error",
    "final_centerline_u",
    "final_max_u",
    "runtime_seconds",
    "mlups",
    "status",
    "outdir",
]


ALIASES = {
    "case_id": ["case_id", "case", "name"],
    "group": ["group"],
    "nx": ["nx", "Nx", "N_x"],
    "ny": ["ny", "Ny", "N_y"],
    "nz": ["nz", "Nz", "N_z"],
    "tau": ["tau"],
    "nu": ["nu", "viscosity", "kinematic_viscosity"],
    "target_umax": ["target_umax", "u_max_target", "umax_target", "target_u_max"],
    "force_x": ["force_x", "fx", "body_force_x", "force"],
    "timesteps": ["timesteps", "steps", "nsteps", "max_steps"],
    "sample_every": ["sample_every", "output_every", "report_every"],
    "initial_condition": ["initial_condition", "init"],
    "l2_profile_error": [
        "l2_profile_error",
        "L2_profile_error",
        "relative_l2_profile_error",
        "rel_l2_profile_error",
        "l2_error",
        "L2_error",
    ],
    "max_profile_error": [
        "max_profile_error",
        "maximum_profile_error",
        "linf_profile_error",
        "Linf_profile_error",
        "max_error",
    ],
    "wall_slip_extrapolated": [
        "wall_slip_extrapolated",
        "extrapolated_wall_slip",
        "wall_slip",
    ],
    "near_wall_node_ux_max": [
        "near_wall_node_ux_max",
        "near_wall_ux_max",
        "max_near_wall_ux",
    ],
    "wall_normal_mean_max": [
        "wall_normal_mean_max",
        "max_wall_normal_mean",
        "max_abs_uy_mean",
        "wall_normal_leakage",
    ],
    "max_mass_relative_error": [
        "max_mass_relative_error",
        "mass_relative_error_max",
        "max_rel_mass_error",
        "mass_error",
    ],
    "final_centerline_u": [
        "final_centerline_u",
        "centerline_u_final",
        "final_centerline_ux",
        "centerline_velocity",
    ],
    "final_max_u": [
        "final_max_u",
        "final_max_ux",
        "max_u_final",
        "max_velocity",
        "umax_final",
    ],
    "runtime_seconds": [
        "runtime_seconds",
        "runtime",
        "elapsed_seconds",
        "wall_time_seconds",
    ],
    "status": ["status"],
}


# --- I/O helpers ---

def load_json(path):
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def first_value(data, names, default=""):
    for name in names:
        if name in data:
            return data[name]
    return default


def flatten_dict(d, prefix=""):
    out = {}

    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k

        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[k] = v
            out[key] = v

    return out


# --- Collection ---

def find_case_dirs(results_dir):
    """Find result folders that contain case_input.json."""
    results_dir = Path(results_dir)
    dirs = []

    for p in results_dir.rglob("*"):
        if p.is_dir() and (p / "case_input.json").exists():
            dirs.append(p)

    return sorted(dirs)


def find_summary_json(case_dir):
    names_to_skip = {
        "case_input.json",
        "campaign_run_record.json",
        "campaign_error.json",
    }

    candidates = []

    for p in Path(case_dir).glob("*.json"):
        if p.name not in names_to_skip:
            candidates.append(p)

    if not candidates:
        return None

    preferred = [
        "ref_mrt_channel_summary.json",
        "summary.json",
        "results_summary.json",
        "output_summary.json",
    ]

    for name in preferred:
        p = Path(case_dir) / name
        if p.exists():
            return p

    return candidates[0]


def nu_from_tau(tau):
    try:
        return (float(tau) - 0.5) / 3.0
    except Exception:
        return ""


def force_from_umax(tau, umax, ny):
    try:
        nu = nu_from_tau(tau)
        return 8.0 * float(nu) * float(umax) / (float(ny) * float(ny))
    except Exception:
        return ""


def mlups(nx, ny, nz, timesteps, runtime_seconds):
    try:
        nodes = float(nx) * float(ny) * float(nz)
        return nodes * float(timesteps) / (1.0e6 * float(runtime_seconds))
    except Exception:
        return ""


def collect_case(case_dir):
    """Merge case_input.json and solver summary into one table row."""
    case_dir = Path(case_dir)

    data = {}

    case_input = case_dir / "case_input.json"
    if case_input.exists():
        data.update(flatten_dict(load_json(case_input)))

    summary_json = find_summary_json(case_dir)
    if summary_json is not None:
        data.update(flatten_dict(load_json(summary_json)))

    run_record = case_dir / "campaign_run_record.json"
    if run_record.exists():
        data.update(flatten_dict(load_json(run_record)))

    error_record = case_dir / "campaign_error.json"
    if error_record.exists():
        data.update(flatten_dict(load_json(error_record)))

    row = {}

    for field in FIELDS:
        if field == "outdir":
            row[field] = str(case_dir)
        else:
            row[field] = first_value(data, ALIASES.get(field, [field]))

    if row["case_id"] == "":
        row["case_id"] = case_dir.name

    if row["group"] == "":
        row["group"] = group_from_case_id(row["case_id"])

    if row["nu"] == "":
        row["nu"] = nu_from_tau(row["tau"])

    if row["force_x"] == "":
        row["force_x"] = force_from_umax(row["tau"], row["target_umax"], row["ny"])

    if row["runtime_seconds"] == "":
        row["runtime_seconds"] = first_value(data, ["runtime_seconds_from_campaign"])

    if row["mlups"] == "":
        row["mlups"] = mlups(
            row["nx"],
            row["ny"],
            row["nz"],
            row["timesteps"],
            row["runtime_seconds"],
        )

    if row["status"] == "":
        row["status"] = "completed" if summary_json is not None else "missing_summary"

    return row


def group_from_case_id(case_id):
    if case_id.startswith("A"):
        return "A_baseline"
    if case_id.startswith("B"):
        return "B_grid_convergence"
    if case_id.startswith("C"):
        return "C_tau_sweep"
    if case_id.startswith("D"):
        return "D_force_sweep"
    if case_id.startswith("E"):
        return "E_initial_condition"
    if case_id.startswith("F"):
        return "F_performance_optional"
    return ""


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def filter_group(rows, group):
    return [r for r in rows if r.get("group") == group]


# --- CLI ---

def main():
    """Parse arguments and write summary CSV tables."""
    parser = argparse.ArgumentParser(
        description="Collect campaign results into CSV tables."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tables-dir", default=TABLE_DIR)
    args = parser.parse_args()

    case_dirs = find_case_dirs(args.results_dir)

    rows = []
    for case_dir in case_dirs:
        rows.append(collect_case(case_dir))

    rows = sorted(rows, key=lambda r: r["case_id"])

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_csv(tables_dir / "table_all_cases.csv", rows)
    write_csv(tables_dir / "table_baseline.csv", filter_group(rows, "A_baseline"))
    write_csv(tables_dir / "table_grid_convergence.csv", filter_group(rows, "B_grid_convergence"))
    write_csv(tables_dir / "table_tau_sweep.csv", filter_group(rows, "C_tau_sweep"))
    write_csv(tables_dir / "table_force_sweep.csv", filter_group(rows, "D_force_sweep"))
    write_csv(tables_dir / "table_initial_condition.csv", filter_group(rows, "E_initial_condition"))
    write_csv(tables_dir / "table_performance.csv", filter_group(rows, "F_performance_optional"))

    print(f"found {len(rows)} cases")
    print(f"wrote {tables_dir / 'table_all_cases.csv'}")
    print(f"wrote {tables_dir / 'table_baseline.csv'}")
    print(f"wrote {tables_dir / 'table_grid_convergence.csv'}")
    print(f"wrote {tables_dir / 'table_tau_sweep.csv'}")
    print(f"wrote {tables_dir / 'table_force_sweep.csv'}")
    print(f"wrote {tables_dir / 'table_initial_condition.csv'}")
    print(f"wrote {tables_dir / 'table_performance.csv'}")


if __name__ == "__main__":
    main()
