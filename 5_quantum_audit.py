#!/usr/bin/env python3
"""Build quantum operator-audit and qubit-estimate tables for the paper."""

import argparse
import csv
import math
from pathlib import Path


TABLE_DIR = "paper_tables"


OPERATOR_ROWS = [
    {
        "lbm_step": "Population storage",
        "classical_operation": "store f_i(x,y,z), i=0,...,18",
        "quantum_interpretation": "position-velocity register",
        "difficulty": "medium",
        "main_issue": "state preparation and memory layout",
    },
    {
        "lbm_step": "Macroscopic fields",
        "classical_operation": "rho=sum_i f_i, u=(sum_i c_i f_i + F/2)/rho",
        "quantum_interpretation": "state-dependent reduction and arithmetic",
        "difficulty": "high",
        "main_issue": "nonlinear normalization and measurement cost",
    },
    {
        "lbm_step": "Moment transform",
        "classical_operation": "m = M f",
        "quantum_interpretation": "local linear map or block encoding",
        "difficulty": "medium",
        "main_issue": "non-unitary dense local transform",
    },
    {
        "lbm_step": "MRT relaxation",
        "classical_operation": "m = m - S(m - m_eq)",
        "quantum_interpretation": "non-unitary damping or quantum channel",
        "difficulty": "high",
        "main_issue": "dissipation is not naturally unitary",
    },
    {
        "lbm_step": "Equilibrium moments",
        "classical_operation": "m_eq = m_eq(rho,u)",
        "quantum_interpretation": "nonlinear state-dependent operation",
        "difficulty": "high",
        "main_issue": "requires nonlinear arithmetic from encoded flow fields",
    },
    {
        "lbm_step": "Body-force forcing",
        "classical_operation": "compute F_i, transform M F_i, apply (I-S/2)M F_i",
        "quantum_interpretation": "source-term operation depending on u and F",
        "difficulty": "high",
        "main_issue": "velocity-dependent forcing and non-unitary source update",
    },
    {
        "lbm_step": "Inverse transform",
        "classical_operation": "f = M^{-1} m",
        "quantum_interpretation": "local linear map or block encoding",
        "difficulty": "medium",
        "main_issue": "inverse non-unitary local map",
    },
    {
        "lbm_step": "Streaming",
        "classical_operation": "shift populations to neighboring nodes",
        "quantum_interpretation": "reversible permutation",
        "difficulty": "low",
        "main_issue": "mostly index arithmetic",
    },
    {
        "lbm_step": "Bounce-back wall",
        "classical_operation": "reflect wall populations using opposite directions",
        "quantum_interpretation": "boundary-controlled swap",
        "difficulty": "low-medium",
        "main_issue": "requires boundary masks and conditional logic",
    },
    {
        "lbm_step": "Diagnostics",
        "classical_operation": "compute profiles, errors, wall slip, mass error",
        "quantum_interpretation": "sampling or tomography bottleneck",
        "difficulty": "high",
        "main_issue": "extracting classical fields from quantum states",
    },
]


# --- Tables ---

DEFAULT_GRIDS = [
    {"case_id": "B1_grid16", "nx": 32, "ny": 16, "nz": 8},
    {"case_id": "B2_grid32", "nx": 64, "ny": 32, "nz": 16},
    {"case_id": "B3_grid48", "nx": 96, "ny": 48, "nz": 24},
    {"case_id": "B4_grid64", "nx": 128, "ny": 64, "nz": 32},
]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []

    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_md(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")

        for row in rows:
            values = []
            for field in fields:
                value = str(row.get(field, ""))
                value = value.replace("|", "/")
                values.append(value)
            f.write("| " + " | ".join(values) + " |\n")


def intval(value, default=None):
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def ceil_log2(n):
    n = int(n)
    if n <= 1:
        return 0
    return int(math.ceil(math.log2(n)))


def unique_grids(rows):
    grids = []
    seen = set()

    for row in rows:
        nx = intval(row.get("nx"))
        ny = intval(row.get("ny"))
        nz = intval(row.get("nz"))

        if nx is None or ny is None or nz is None:
            continue

        key = (nx, ny, nz)
        if key in seen:
            continue

        seen.add(key)

        grids.append({
            "case_id": row.get("case_id", ""),
            "nx": nx,
            "ny": ny,
            "nz": nz,
        })

    if not grids:
        grids = DEFAULT_GRIDS

    return sorted(grids, key=lambda r: (r["nx"], r["ny"], r["nz"]))


def make_qubit_rows(grids):
    """Lower-bound qubit counts from grid sizes (position--velocity encoding)."""
    rows = []

    for g in grids:
        nx = g["nx"]
        ny = g["ny"]
        nz = g["nz"]

        qx = ceil_log2(nx)
        qy = ceil_log2(ny)
        qz = ceil_log2(nz)
        qv = ceil_log2(19)
        qmin = qx + qy + qz + qv

        rows.append({
            "case_id": g.get("case_id", ""),
            "grid": f"{nx}x{ny}x{nz}",
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "q_x": qx,
            "q_y": qy,
            "q_z": qz,
            "velocity_qubits": qv,
            "minimum_logical_qubits": qmin,
            "fluid_nodes": nx * ny * nz,
            "population_values": nx * ny * nz * 19,
        })

    return rows


# --- CLI ---

def main():
    """Parse arguments and write quantum audit tables."""
    parser = argparse.ArgumentParser(
        description="Export quantum audit tables (CSV and Markdown)."
    )
    parser.add_argument("--tables-dir", default=TABLE_DIR)
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    rows = read_csv(tables_dir / "table_all_cases.csv")
    grids = unique_grids(rows)
    qubit_rows = make_qubit_rows(grids)

    operator_fields = [
        "lbm_step",
        "classical_operation",
        "quantum_interpretation",
        "difficulty",
        "main_issue",
    ]

    qubit_fields = [
        "case_id",
        "grid",
        "nx",
        "ny",
        "nz",
        "q_x",
        "q_y",
        "q_z",
        "velocity_qubits",
        "minimum_logical_qubits",
        "fluid_nodes",
        "population_values",
    ]

    write_csv(
        tables_dir / "table_quantum_operator_audit.csv",
        OPERATOR_ROWS,
        operator_fields,
    )

    write_csv(
        tables_dir / "table_qubit_estimates.csv",
        qubit_rows,
        qubit_fields,
    )

    write_md(
        tables_dir / "table_quantum_operator_audit.md",
        OPERATOR_ROWS,
        operator_fields,
    )

    write_md(
        tables_dir / "table_qubit_estimates.md",
        qubit_rows,
        qubit_fields,
    )

    print(f"wrote {tables_dir / 'table_quantum_operator_audit.csv'}")
    print(f"wrote {tables_dir / 'table_qubit_estimates.csv'}")
    print(f"wrote {tables_dir / 'table_quantum_operator_audit.md'}")
    print(f"wrote {tables_dir / 'table_qubit_estimates.md'}")


if __name__ == "__main__":
    main()
