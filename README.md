# D3Q19 MRT Channel Benchmark and Quantum-Operator Audit

Reproducible Python workflow for a forced three-dimensional D3Q19 multiple-relaxation-time (MRT) lattice-Boltzmann channel benchmark, including classical verification cases and a quantum-operator resource audit.

The solver targets body-force-driven Poiseuille flow with periodic streamwise and spanwise boundaries, halfway bounce-back channel walls, and MRT collision with body-force forcing in moment space.

## Contents

| Script | Role |
|--------|------|
| `1_mrt_guo_channel.py` | Single-case channel solver |
| `2_run_campaign.py` | 18-case verification campaign |
| `3_collect_results.py` | Aggregate case outputs into CSV tables |
| `4_make_figures.py` | Build PDF figures from tables |
| `5_quantum_audit.py` | Quantum-operator and qubit-estimate tables |

Paths below are relative to the repository root.

## Requirements

- Python 3.10+
- `numpy`, `jax`, `matplotlib` (see `requirements.txt`)

```bash
python3 -m pip install -r requirements.txt
export MPLBACKEND=Agg   # headless figure export
```

## Quick start

```bash
git clone https://github.com/IdreesKhan3/mrt-lbm-quantum-audit.git
cd mrt-lbm-quantum-audit

python3 2_run_campaign.py --overwrite
python3 3_collect_results.py
python3 4_make_figures.py
python3 5_quantum_audit.py
```

Output directories (created next to the scripts):

```text
results/        per-case simulation outputs
paper_tables/   CSV summary tables
figures/        PDF figures
```

## Verification campaign (18 cases)

| Group | Cases | Purpose |
|-------|-------|---------|
| A | 1 | Baseline (`A1_baseline_rest`) |
| B | 4 | Grid refinement (`B1_grid16` … `B4_grid64`) |
| C | 6 | Relaxation-time sweep (`C1_tau055` … `C6_tau160`) |
| D | 5 | Forcing-strength / target \(u_{\max}\) sweep |
| E | 2 | Initialization: rest vs parabolic |

```bash
python3 2_run_campaign.py --dry-run
python3 2_run_campaign.py --only-case A1_baseline_rest --overwrite
```

## Single-case run

```bash
python3 1_mrt_guo_channel.py \
  --nx 64 --ny 32 --nz 16 \
  --tau 0.8 \
  --force-x 1.5625e-5 \
  --timesteps 20000 \
  --sample-every 200 \
  --initial-condition rest \
  --outdir results/A1_baseline_rest
```

Each `results/<case_id>/` folder should contain:

```text
case_input.json
ref_mrt_channel_results.npz
ref_mrt_channel_summary.json
```

## Tables and figures

```bash
python3 3_collect_results.py
python3 4_make_figures.py
python3 5_quantum_audit.py
```

Tables: `paper_tables/table_*.csv` (including `table_all_cases.csv`, `table_quantum_operator_audit.csv`, `table_qubit_estimates.csv`).

Figures: `figures/*.pdf` (`algorithm_schematic`, `baseline_profile`, `grid_convergence`, `centerline_history`, `tau_sensitivity`, `force_sensitivity`, `wall_diagnostics`, `grid_profiles`).

## Runtime

- Small grids: minutes on CPU.
- `B4_grid64` (128×64×32, 20 000 steps): can take hours; JAX compiles per grid size.
- GPU `jaxlib` is used when available; otherwise CPU.

## Custom paths

```bash
python3 2_run_campaign.py --results-dir my_results --overwrite
python3 3_collect_results.py --results-dir my_results --tables-dir my_tables
python3 4_make_figures.py --tables-dir my_tables --figures-dir my_figures
```

## Citation

```text
Muhammad Idrees Khan and Hua-Dong Yao,
"A Reproducible D3Q19 MRT Lattice-Boltzmann Benchmark and Quantum-Operator Audit
for Forced Wall-Bounded Flow Simulations," 2026.
```

## License

MIT License. See `LICENSE`.
