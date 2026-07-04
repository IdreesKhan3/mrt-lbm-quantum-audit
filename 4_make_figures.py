#!/usr/bin/env python3
"""Generate PDF figures from paper_tables and case .npz files."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FIG_DIR = "figures"
FIG_DPI = 600

SINGLE_COL = (3.5, 2.65)  # single-column size (in)
GRID_PROFILES = (3.5, 4.35)  # taller canvas: y extends to 64 for finest grid
WIDE_COL = (5.5, 2.0)

# Colorblind-safe palette
C_NUM = "#2166ac"
C_EXACT = "#b2182b"
C_GRID = ["#2166ac", "#4393c3", "#92c5de", "#1b7837"]
C_LINES = ["#2166ac", "#d6604d", "#4daf4a", "#984ea3", "#ff7f00"]
# Figure 3 grid profiles: blue, orange, green, purple
C_PROFILES = [C_LINES[0], C_LINES[4], C_LINES[2], C_LINES[3]]


# --- Plot style ---

def setup_style():
    """Set matplotlib rcParams for paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "lines.markersize": 5.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.2,
            "ytick.minor.size": 2.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.borderpad": 0.3,
            "legend.handlelength": 1.6,
            "axes.grid": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIG_DPI,
            "figure.dpi": FIG_DPI,
            "figure.constrained_layout.use": False,
        }
    )


def style_axes(ax, *, logx=False, logy=False, grid=True):
    """Ticks, grid, and spine styling."""
    ax.tick_params(which="both", top=True, right=True)
    if grid:
        which = "both" if (logx or logy) else "major"
        ax.grid(True, which=which, alpha=0.28, linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def format_log_axis(ax, axis="both"):
    formatter = ticker.LogFormatterSciNotation(labelOnlyBase=False)
    if axis in ("x", "both") and ax.get_xscale() == "log":
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    if axis in ("y", "both") and ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def format_power_axis(ax, axis="y", mantissa_decimals=3):
    """Mathtext $m \\times 10^{n}$ tick labels (matches log-figure style on linear axes)."""

    def _fmt(value, _pos):
        if value == 0:
            return r"$0$"
        exponent = int(np.floor(np.log10(np.abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"${mantissa:.{mantissa_decimals}f} \times 10^{{{exponent}}}$"

    formatter = ticker.FuncFormatter(_fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


# --- Data helpers ---

def short_case_label(row):
    case_id = sval(row, "case_id")
    group = sval(row, "group")

    if group == "B_grid_convergence":
        ny = int(fval(row, "ny"))
        return rf"$N_y={ny}$"

    if group == "C_tau_sweep":
        return rf"$\tau={fval(row, 'tau'):.2g}$"

    if group == "D_force_sweep":
        return rf"$u_{{\max}}={fval(row, 'target_umax'):.3g}$"

    if group == "E_initial_condition":
        init = sval(row, "initial_condition")
        return "Rest" if init == "rest" else "Parabolic"

    if group == "A_baseline":
        return "Baseline"

    return case_id.split("_")[0]


def centerline_label(case_id):
    mapping = {
        "E1_rest": "Rest initialization",
        "E2_parabolic": "Parabolic initialization",
        "A1_baseline_rest": "Baseline (rest)",
    }
    return mapping.get(case_id, case_id.replace("_", " "))


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []

    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def fval(row, key, default=np.nan):
    try:
        value = row.get(key, "")
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def sval(row, key, default=""):
    value = row.get(key, default)
    if value is None:
        return default
    return str(value)


def case_rows(rows, group):
    return [r for r in rows if sval(r, "group") == group]


def case_row(rows, case_id):
    for r in rows:
        if sval(r, "case_id") == case_id:
            return r
    return None


def resolve_outdir(row):
    """Resolve case output directory, with fallbacks for relocated results."""
    outdir = Path(sval(row, "outdir"))
    case_id = sval(row, "case_id")
    script_dir = Path(__file__).resolve().parent

    candidates = [
        outdir,
        outdir.resolve(),
        script_dir / outdir,
        script_dir / "results" / case_id,
        script_dir / "results" / outdir.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return outdir


def find_npz(row):
    outdir = resolve_outdir(row)
    if not outdir.exists():
        return None

    preferred = [
        "ref_mrt_channel_results.npz",
        "results.npz",
        "output.npz",
    ]

    for name in preferred:
        p = outdir / name
        if p.exists():
            return p

    files = sorted(outdir.glob("*.npz"))
    if files:
        return files[0]

    return None


def get_array(npz, names):
    keys = list(npz.keys())
    low = {k.lower(): k for k in keys}

    for name in names:
        if name in npz:
            return np.asarray(npz[name])

    for name in names:
        k = low.get(name.lower())
        if k is not None:
            return np.asarray(npz[k])

    return None


def reduce_to_profile(arr, ny):
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        if arr.shape[0] == ny:
            return np.mean(arr, axis=1)
        if arr.shape[1] == ny:
            return np.mean(arr, axis=0)

    if arr.ndim == 3:
        for axis in range(3):
            if arr.shape[axis] == ny:
                return np.mean(np.moveaxis(arr, axis, 0), axis=tuple(range(1, 3)))

    if arr.ndim == 4:
        if arr.shape[0] in [2, 3]:
            return reduce_to_profile(arr[0], ny)
        if arr.shape[-1] in [2, 3]:
            return reduce_to_profile(arr[..., 0], ny)

    return None


def get_y(npz, row, n):
    arr = get_array(npz, [
        "y",
        "y_profile",
        "y_nodes",
        "y_coords",
        "wall_normal_coordinate",
    ])

    if arr is not None:
        arr = np.asarray(arr, dtype=float).ravel()
        if len(arr) == n:
            return arr

    ny = int(fval(row, "ny", n))
    if ny == n:
        return np.arange(n) + 0.5

    return np.arange(n)


def get_profile(npz, row):
    ny = int(fval(row, "ny", 0))

    num = get_array(npz, [
        "ux_profile",
        "u_profile",
        "numerical_profile",
        "ux_mean_y",
        "profile_ux",
        "u_num",
        "ux",
        "u_x",
        "velocity_profile",
    ])

    exact = get_array(npz, [
        "ux_analytical",
        "u_analytical",
        "analytical_profile",
        "exact_profile",
        "ux_exact",
        "u_exact",
        "poiseuille_profile",
    ])

    if num is None:
        return None, None, None

    num = reduce_to_profile(num, ny)
    if num is None:
        return None, None, None

    num = np.asarray(num, dtype=float).ravel()
    y = get_y(npz, row, len(num))

    if exact is not None:
        exact = reduce_to_profile(exact, len(num))
        if exact is not None:
            exact = np.asarray(exact, dtype=float).ravel()

    if exact is None or len(exact) != len(num):
        tau = fval(row, "tau")
        force_x = fval(row, "force_x")
        ny = int(fval(row, "ny", len(num)))

        if np.isfinite(tau) and np.isfinite(force_x):
            nu = (tau - 0.5) / 3.0
            yp = np.arange(ny) + 0.5
            exact = force_x * yp * (ny - yp) / (2.0 * nu)
            if len(exact) != len(num):
                exact = None

    return y, num, exact


def get_history(npz):
    t = get_array(npz, [
        "time",
        "times",
        "timesteps",
        "steps",
        "step_history",
        "sample_steps",
        "history_steps",
    ])

    u = get_array(npz, [
        "centerline_u",
        "centerline_history",
        "centerline_ux",
        "u_centerline",
        "centerline_velocity",
        "final_centerline_history",
    ])

    if u is None:
        return None, None

    u = np.asarray(u, dtype=float).ravel()

    if t is None:
        t = np.arange(len(u))
    else:
        t = np.asarray(t, dtype=float).ravel()

    n = min(len(t), len(u))
    return t[:n], u[:n]


def savefig(path, tight=True):
    """Save current figure as PDF and close."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout(pad=0.4)
    plt.savefig(
        path,
        format="pdf",
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close()


# --- Figures ---

def fig1_algorithm(fig_dir):
    """Timestep schematic (algorithm_schematic.pdf)."""
    steps = [
        (r"$f_i$", 0.70),
        (r"$\rho$, $\mathbf{u}$", 0.95),
        (r"$\mathbf{m}=M\mathbf{f}$", 1.15),
        (r"$\mathbf{m}^{\mathrm{eq}}$", 1.05),
        ("MRT collision", 1.25),
        ("Guo forcing", 1.15),
        (r"$f_i^{\,\mathrm{post}}$", 1.05),
        ("Streaming", 1.15),
        ("Bounce-back", 1.20),
    ]

    gap = 0.28
    box_h = 0.78
    x0 = 0.25
    y_box = 1.85
    face = "#f7f9fc"
    edge = "#2f2f2f"
    box_fs = 10.5
    eq_fs = 11.0

    total_w = x0 + sum(w for _, w in steps) + gap * (len(steps) - 1) + 0.25
    fig_h = 3.15
    fig, ax = plt.subplots(figsize=(total_w * 0.95, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, 2.55)
    ax.axis("off")

    x = x0
    for i, (label, box_w) in enumerate(steps):
        x_center = x + box_w / 2

        ax.add_patch(
            FancyBboxPatch(
                (x, y_box - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=0.8,
                edgecolor=edge,
                facecolor=face,
                zorder=2,
            )
        )
        ax.text(
            x_center,
            y_box,
            label,
            ha="center",
            va="center",
            fontsize=box_fs,
            zorder=3,
        )

        if i < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + box_w + 0.02, y_box),
                    (x + box_w + gap - 0.02, y_box),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=0.85,
                    color=edge,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=1,
                )
            )

        x += box_w + gap

    eq_x = x0 + (total_w - x0 - 0.25) / 2
    ax.text(
        eq_x,
        0.95,
        r"$\mathbf{u} = \left(\sum_i \mathbf{c}_i f_i + \mathbf{F}/2\right)/\rho$",
        ha="center",
        va="center",
        fontsize=eq_fs,
    )
    ax.text(
        eq_x,
        0.38,
        r"$\mathbf{m}^{\mathrm{post}} = \mathbf{m} - S(\mathbf{m}-\mathbf{m}^{\mathrm{eq}})"
        r" + \left(I-S/2\right)M\boldsymbol{\Phi}$",
        ha="center",
        va="center",
        fontsize=eq_fs,
    )

    savefig(Path(fig_dir) / "algorithm_schematic.pdf", tight=False)


def fig2_baseline(rows, fig_dir):
    """Baseline profile vs analytic (baseline_profile.pdf)."""
    row = case_row(rows, "A1_baseline_rest")
    if row is None:
        print("missing A1_baseline_rest")
        return

    npz_path = find_npz(row)
    if npz_path is None:
        print("missing npz for A1_baseline_rest")
        return

    with np.load(npz_path, allow_pickle=True) as npz:
        y, num, exact = get_profile(npz, row)

    if y is None or num is None:
        print("missing baseline profile")
        return

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.plot(
        num,
        y,
        "o",
        color=C_NUM,
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="MRT-LBM",
        zorder=3,
    )
    if exact is not None:
        ax.plot(exact, y, "-", color=C_EXACT, linewidth=1.3, label="Analytical", zorder=2)

    ax.set_xlabel(r"Streamwise velocity, $u_x$")
    ax.set_ylabel(r"Wall-normal coordinate, $y$")
    ax.legend(loc="best")
    style_axes(ax)

    savefig(Path(fig_dir) / "baseline_profile.pdf")


def fig3_grid(rows, fig_dir):
    """Grid convergence of L2 error (grid_convergence.pdf)."""
    data = []

    for row in case_rows(rows, "B_grid_convergence"):
        ny = fval(row, "ny")
        err = fval(row, "l2_profile_error")
        if np.isfinite(ny) and np.isfinite(err) and err > 0:
            data.append((ny, err, sval(row, "case_id")))

    if not data:
        print("missing grid convergence data")
        return

    data.sort()
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    if len(x) >= 2:
        p_ref = 2.0
        c_ref = y[0] * (x[0] ** p_ref)
        x_ref = np.geomspace(x.min(), x.max(), num=40)
        y_ref = c_ref * x_ref ** (-p_ref)
        ax.loglog(
            x_ref,
            y_ref,
            "--",
            color="#444444",
            linewidth=1.35,
            dashes=(5, 3),
            label=rf"$N_y^{{-{p_ref:.0f}}}$ reference",
            zorder=2,
        )

    ax.loglog(
        x,
        y,
        "o",
        color=C_NUM,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linestyle="none",
        label=r"$E_{L_2}$",
        zorder=4,
    )

    ax.set_xlabel(r"Wall-normal resolution, $N_y$")
    ax.set_ylabel(r"Relative $L_2$ profile error, $E_{L_2}$")
    format_log_axis(ax)
    ax.legend(loc="best")
    style_axes(ax)

    savefig(Path(fig_dir) / "grid_convergence.pdf")


def fig4_centerline(rows, fig_dir):
    """Centerline history for E cases (centerline_history.pdf)."""
    wanted = ["E1_rest", "E2_parabolic"]
    plotted = False

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    for idx, case_id in enumerate(wanted):
        row = case_row(rows, case_id)
        if row is None:
            continue

        npz_path = find_npz(row)
        if npz_path is None:
            continue

        with np.load(npz_path, allow_pickle=True) as npz:
            t, u = get_history(npz)

        if t is None or u is None:
            continue

        ax.plot(
            t,
            u,
            "-",
            color=C_LINES[idx % len(C_LINES)],
            linewidth=1.2,
            label=centerline_label(case_id),
        )
        plotted = True

    if not plotted:
        row = case_row(rows, "A1_baseline_rest")
        if row is not None:
            npz_path = find_npz(row)
            if npz_path is not None:
                with np.load(npz_path, allow_pickle=True) as npz:
                    t, u = get_history(npz)

                if t is not None and u is not None:
                    ax.plot(
                        t,
                        u,
                        "-",
                        color=C_NUM,
                        linewidth=1.2,
                        label=centerline_label("A1_baseline_rest"),
                    )
                    plotted = True

    if not plotted:
        plt.close()
        print("missing centerline history")
        return

    ax.set_xlabel(r"Timestep index, $n$")
    ax.set_ylabel(r"Centerline velocity, $u_x$")
    ax.legend(loc="best")
    style_axes(ax)

    savefig(Path(fig_dir) / "centerline_history.pdf")


def fig5_tau(rows, fig_dir):
    data = []

    for row in case_rows(rows, "C_tau_sweep"):
        tau = fval(row, "tau")
        err = fval(row, "l2_profile_error")
        if np.isfinite(tau) and np.isfinite(err) and err > 0:
            data.append((tau, err, sval(row, "case_id")))

    if not data:
        print("missing tau sweep data")
        return

    data.sort()
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.semilogy(
        x,
        y,
        "o-",
        color=C_NUM,
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
    )

    ax.set_xlabel(r"Relaxation time, $\tau$")
    ax.set_ylabel(r"Relative $L_2$ profile error, $E_{L_2}$")
    format_log_axis(ax, axis="y")
    style_axes(ax)

    savefig(Path(fig_dir) / "tau_sensitivity.pdf")


def fig6_force(rows, fig_dir):
    data = []

    for row in case_rows(rows, "D_force_sweep"):
        umax = fval(row, "target_umax")
        err = fval(row, "l2_profile_error")
        if np.isfinite(umax) and np.isfinite(err) and err > 0:
            data.append((umax, err, sval(row, "case_id")))

    if not data:
        print("missing force sweep data")
        return

    data.sort()
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.semilogx(
        x,
        y,
        "o-",
        color=C_NUM,
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
    )

    ymid = float(np.mean(y))
    span = max(float(y.max() - y.min()), ymid * 5e-4)
    ylo = ymid - 3.0 * span
    yhi = ymid + 3.0 * span
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(np.linspace(ylo, yhi, 4))
    format_power_axis(ax, axis="y", mantissa_decimals=3)

    ax.set_xlabel(r"Target maximum velocity, $u_{\max}$")
    ax.set_ylabel(r"Relative $L_2$ profile error, $E_{L_2}$")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"{v:g}" for v in x])
    ax.minorticks_off()
    style_axes(ax)

    savefig(Path(fig_dir) / "force_sensitivity.pdf")


def fig7_wall_metrics(rows, fig_dir):
    data = []

    group_order = [
        "A_baseline",
        "B_grid_convergence",
        "C_tau_sweep",
        "D_force_sweep",
        "E_initial_condition",
    ]
    order = {g: i for i, g in enumerate(group_order)}

    for row in rows:
        case_id = sval(row, "case_id")
        slip = abs(fval(row, "wall_slip_extrapolated"))
        leakage = abs(fval(row, "wall_normal_mean_max"))
        mass = abs(fval(row, "max_mass_relative_error"))

        if np.isfinite(slip) or np.isfinite(leakage) or np.isfinite(mass):
            data.append((
                order.get(sval(row, "group"), 99),
                case_id,
                short_case_label(row),
                slip,
                leakage,
                mass,
            ))

    if not data:
        print("missing wall metric data")
        return

    data.sort(key=lambda d: (d[0], d[1]))
    labels = [d[2] for d in data]
    x = np.arange(len(labels))

    fig_w = max(5.0, 0.28 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 2.85))

    slip = np.array([d[3] for d in data], dtype=float)
    leakage = np.array([d[4] for d in data], dtype=float)
    mass = np.array([d[5] for d in data], dtype=float)

    if np.isfinite(slip).any():
        ax.semilogy(
            x,
            slip,
            "o-",
            color=C_LINES[0],
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=r"Wall slip, $E_{\mathrm{slip}}$",
        )

    if np.isfinite(leakage).any():
        ax.semilogy(
            x,
            leakage,
            "s-",
            color=C_LINES[1],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=r"Wall-normal leakage",
        )

    if np.isfinite(mass).any():
        ax.semilogy(
            x,
            mass,
            "^-",
            color=C_LINES[2],
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=r"Mass error, $E_m$",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Diagnostic magnitude")
    format_log_axis(ax, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3)
    style_axes(ax)

    savefig(Path(fig_dir) / "wall_diagnostics.pdf")


def fig_profiles_all_grids(rows, fig_dir):
    grid_rows = case_rows(rows, "B_grid_convergence")
    plotted = False

    fig, ax = plt.subplots(figsize=GRID_PROFILES)

    for idx, row in enumerate(sorted(grid_rows, key=lambda r: fval(r, "ny"))):
        npz_path = find_npz(row)
        if npz_path is None:
            continue

        with np.load(npz_path, allow_pickle=True) as npz:
            y, num, exact = get_profile(npz, row)

        if y is None or num is None:
            continue

        color = C_PROFILES[idx % len(C_PROFILES)]
        ax.plot(
            num,
            y,
            "-o",
            color=color,
            linewidth=1.1,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=0.85,
            label=short_case_label(row),
        )
        plotted = True

    if not plotted:
        plt.close()
        print("missing grid profiles")
        return

    ax.set_xlabel(r"Streamwise velocity, $u_x$")
    ax.set_ylabel(r"Wall-normal coordinate, $y$")
    ax.set_ylim(bottom=0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    style_axes(ax)
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.15, right=0.98)

    savefig(Path(fig_dir) / "grid_profiles.pdf", tight=False)


# --- CLI ---

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build paper figures from collected tables."
    )
    parser.add_argument("--tables-dir", default="paper_tables")
    parser.add_argument("--figures-dir", default=FIG_DIR)
    args = parser.parse_args()

    setup_style()

    rows = read_csv(Path(args.tables_dir) / "table_all_cases.csv")

    if not rows:
        raise SystemExit("no rows found; run python 3_collect_results.py first")

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    fig1_algorithm(args.figures_dir)
    fig2_baseline(rows, args.figures_dir)
    fig3_grid(rows, args.figures_dir)
    fig4_centerline(rows, args.figures_dir)
    fig5_tau(rows, args.figures_dir)
    fig6_force(rows, args.figures_dir)
    fig7_wall_metrics(rows, args.figures_dir)
    fig_profiles_all_grids(rows, args.figures_dir)

    print(f"wrote figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
