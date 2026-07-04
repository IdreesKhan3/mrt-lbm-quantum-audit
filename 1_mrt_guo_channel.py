#!/usr/bin/env python3
"""D3Q19 MRT-LBM channel solver: periodic x/z, no-slip y walls, Guo forcing in moment space."""

from __future__ import annotations

__version__ = "2.0.1-channel-wallaudit"

import argparse
import json
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# --- Lattice ---

Q = 19

W = jnp.array([1 / 18] * 6 + [1 / 36] * 12 + [1 / 3], dtype=jnp.float64)  # 6 faces, 12 edges, rest

DIRX_NP = np.array(
    [1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0],
    dtype=np.int32,
)
DIRY_NP = np.array(
    [0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 0],
    dtype=np.int32,
)
DIRZ_NP = np.array(
    [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 0],
    dtype=np.int32,
)

Dirx = jnp.asarray(DIRX_NP, dtype=jnp.int32)
Diry = jnp.asarray(DIRY_NP, dtype=jnp.int32)
Dirz = jnp.asarray(DIRZ_NP, dtype=jnp.int32)

NODE_VELOCITIES = jnp.array([Dirx, Diry, Dirz], dtype=jnp.float64)

OPP_NP = np.array(
    [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 18],
    dtype=np.int32,
)

rho0 = 1.0
cs2 = 1.0 / 3.0
cs4 = cs2 * cs2

# Conserved moments (m0, jx, jy, jz): forcing added as m + phi_m, not via (I - S/2).
CONSERVED_MOMENT_MASK = jnp.array(
    [True] + [False] * 2 + [True] + [False] + [True] + [False] + [True]
    + [False] * 11
)


# --- MRT ---

def build_M_matrix_D3Q19() -> jnp.ndarray:
    """Build 19x19 D3Q19 moment matrix M (f -> m)."""
    e_alpha = jnp.stack([Dirx, Diry, Dirz], axis=1)
    M = jnp.zeros((19, 19), dtype=jnp.float64)

    for alpha in range(19):
        ex, ey, ez = e_alpha[alpha]
        e_norm_sq = ex * ex + ey * ey + ez * ez

        M = M.at[0, alpha].set(1.0)
        M = M.at[1, alpha].set(19 * e_norm_sq - 30.0)
        M = M.at[2, alpha].set(
            0.5 * (21 * e_norm_sq**2 - 53 * e_norm_sq + 24.0)
        )

        M = M.at[3, alpha].set(ex)
        M = M.at[4, alpha].set((5 * e_norm_sq - 9.0) * ex)

        M = M.at[5, alpha].set(ey)
        M = M.at[6, alpha].set((5 * e_norm_sq - 9.0) * ey)

        M = M.at[7, alpha].set(ez)
        M = M.at[8, alpha].set((5 * e_norm_sq - 9.0) * ez)

        M = M.at[9, alpha].set(3 * ex * ex - e_norm_sq)
        M = M.at[10, alpha].set(
            (3 * e_norm_sq - 5.0) * (3 * ex * ex - e_norm_sq)
        )

        M = M.at[11, alpha].set(ey * ey - ez * ez)
        M = M.at[12, alpha].set(
            (3 * e_norm_sq - 5.0) * (ey * ey - ez * ez)
        )

        M = M.at[13, alpha].set(ex * ey)
        M = M.at[14, alpha].set(ey * ez)
        M = M.at[15, alpha].set(ex * ez)

        M = M.at[16, alpha].set((ey * ey - ez * ez) * ex)
        M = M.at[17, alpha].set((ez * ez - ex * ex) * ey)
        M = M.at[18, alpha].set((ex * ex - ey * ey) * ez)

    return M


def make_relaxation_vector(tau: float) -> jnp.ndarray:
    """Diagonal MRT rates S; shear modes use nu = (tau - 0.5) / 3."""
    nu = (tau - 0.5) / 3.0
    if nu <= 0.0:
        raise ValueError("tau must be > 0.5 for positive viscosity.")

    s9 = 1.0 / (3.0 * nu + 0.5)

    return jnp.array(
        [
            0.0,
            1.19,
            1.4,
            0.0,
            1.2,
            0.0,
            1.2,
            0.0,
            1.2,
            s9,
            1.4,
            s9,
            1.4,
            s9,
            s9,
            s9,
            1.98,
            1.98,
            1.98,
        ],
        dtype=jnp.float64,
    )


@jax.jit
def get_equilibrium(u: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    """D3Q19 equilibrium populations from (rho, u)."""
    eu = jnp.einsum("dQ,ijkd->ijkQ", NODE_VELOCITIES, u)
    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)

    return rho[..., None] * W * (
        1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * u_sq
    )


def analytical_poiseuille_profile(
    ny: int,
    tau: float,
    force_x: float,
    rho_ref: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic u_x(y) at cell centers; walls at y=0,H, centers at y=0.5,...,ny-0.5."""
    nu = (tau - 0.5) / 3.0
    H = float(ny)
    y = np.arange(ny, dtype=np.float64) + 0.5
    ux = (force_x / (2.0 * rho_ref * nu)) * y * (H - y)
    return y, ux


def initialize_channel(
    nx: int,
    ny: int,
    nz: int,
    tau: float,
    force_x: float,
    initial_condition: str = "rest",
) -> jnp.ndarray:
    """Equilibrium populations; initial_condition is 'rest' or 'parabolic'."""
    rho_init = jnp.ones((nx, ny, nz), dtype=jnp.float64)

    if initial_condition == "rest":
        u_init = jnp.zeros((nx, ny, nz, 3), dtype=jnp.float64)

    elif initial_condition == "parabolic":
        _, ux_np = analytical_poiseuille_profile(ny, tau, force_x)
        ux_y = jnp.asarray(ux_np, dtype=jnp.float64)
        ux = jnp.broadcast_to(ux_y[None, :, None], (nx, ny, nz))
        uy = jnp.zeros_like(ux)
        uz = jnp.zeros_like(ux)
        u_init = jnp.stack([ux, uy, uz], axis=-1)

    else:
        raise ValueError(
            "initial_condition must be either 'rest' or 'parabolic'."
        )

    return get_equilibrium(u_init, rho_init)


# --- Streaming ---

def _roll_xz(a: jnp.ndarray, cx: int, cz: int) -> jnp.ndarray:
    """Periodic shift in x and z."""
    out = a
    if cx != 0:
        out = jnp.roll(out, shift=cx, axis=0)
    if cz != 0:
        out = jnp.roll(out, shift=cz, axis=2)
    return out


def stream_bounceback_y(f_post: jnp.ndarray) -> jnp.ndarray:
    """Stream with periodic x/z and halfway bounce-back on y walls."""
    f_streamed = jnp.zeros_like(f_post)

    for i in range(Q):
        cx = int(DIRX_NP[i])
        cy = int(DIRY_NP[i])
        cz = int(DIRZ_NP[i])
        opp = int(OPP_NP[i])

        shifted_xz = _roll_xz(f_post[..., i], cx, cz)

        if cy == 0:
            f_streamed = f_streamed.at[..., i].set(shifted_xz)

        elif cy == 1:
            f_streamed = f_streamed.at[:, 1:, :, i].set(
                shifted_xz[:, :-1, :]
            )
            f_streamed = f_streamed.at[:, 0, :, i].set(
                f_post[:, 0, :, opp]
            )

        elif cy == -1:
            f_streamed = f_streamed.at[:, :-1, :, i].set(
                shifted_xz[:, 1:, :]
            )
            f_streamed = f_streamed.at[:, -1, :, i].set(
                f_post[:, -1, :, opp]
            )

        else:
            raise RuntimeError("D3Q19 direction has invalid y-component.")

    return f_streamed


# --- Solver ---

def run_ref_mrt_channel(
    nx: int = 64,
    ny: int = 32,
    nz: int = 16,
    tau: float = 0.8,
    force_x: float = 1.0e-6,
    timesteps: int = 10000,
    sample_every: int = 100,
    initial_condition: str = "rest",
    quiet: bool = True,
    return_fields: bool = False,
) -> dict:
    """Run channel benchmark; return histories, profiles, and error metrics."""
    if nx <= 2 or ny <= 2 or nz <= 2:
        raise ValueError("nx, ny, nz must all be > 2.")
    if tau <= 0.5:
        raise ValueError("tau must be > 0.5.")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive.")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive.")

    nu = (tau - 0.5) / 3.0

    M = build_M_matrix_D3Q19()
    invM = jnp.linalg.inv(M)
    s_vec = make_relaxation_vector(tau)
    force = jnp.array([force_x, 0.0, 0.0], dtype=jnp.float64)

    dv0 = initialize_channel(
        nx=nx,
        ny=ny,
        nz=nz,
        tau=tau,
        force_x=force_x,
        initial_condition=initial_condition,
    )

    @jax.jit
    def update_local(dv_prev: jnp.ndarray):
        rho = jnp.sum(dv_prev, axis=-1)
        rho_safe = jnp.where(rho < 1e-15, 1.0, rho)

        momentum = jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, dv_prev)
        u = (momentum + 0.5 * force[None, None, None, :]) / rho_safe[..., None]

        jx = rho * u[..., 0]
        jy = rho * u[..., 1]
        jz = rho * u[..., 2]
        j_sq = jx**2 + jy**2 + jz**2
        delta_rho = rho - rho0

        m = jnp.einsum("ab,ijkb->ijka", M, dv_prev)

        m_eq = jnp.zeros_like(m)
        m_eq = m_eq.at[..., 0].set(delta_rho)

        m_eq = m_eq.at[..., 1].set(
            -11.0 * delta_rho + 19.0 * j_sq / rho0
        )
        m_eq = m_eq.at[..., 2].set(
            (-475.0 / 63.0) * j_sq / rho0
        )

        m_eq = m_eq.at[..., 3].set(jx)
        m_eq = m_eq.at[..., 4].set(-(2.0 / 3.0) * jx)

        m_eq = m_eq.at[..., 5].set(jy)
        m_eq = m_eq.at[..., 6].set(-(2.0 / 3.0) * jy)

        m_eq = m_eq.at[..., 7].set(jz)
        m_eq = m_eq.at[..., 8].set(-(2.0 / 3.0) * jz)

        m_eq = m_eq.at[..., 9].set(
            (2.0 * jx**2 - jy**2 - jz**2) / rho0
        )
        m_eq = m_eq.at[..., 10].set(0.0)

        m_eq = m_eq.at[..., 11].set((jy**2 - jz**2) / rho0)
        m_eq = m_eq.at[..., 12].set(0.0)

        m_eq = m_eq.at[..., 13].set(jx * jy / rho0)
        m_eq = m_eq.at[..., 14].set(jy * jz / rho0)
        m_eq = m_eq.at[..., 15].set(jx * jz / rho0)

        m_eq = m_eq.at[..., 16:19].set(0.0)

        F_field = jnp.broadcast_to(force[None, None, None, :], u.shape)
        u_dot_F = jnp.sum(u * F_field, axis=-1)
        c_dot_F = jnp.einsum("dQ,ijkd->ijkQ", NODE_VELOCITIES, F_field)
        c_dot_u = jnp.einsum("dQ,ijkd->ijkQ", NODE_VELOCITIES, u)
        phi_i = W * (
            (1.0 / cs2) * (c_dot_F - u_dot_F[..., None])
            + (1.0 / cs4) * c_dot_u * c_dot_F
        )

        phi_m = jnp.einsum("ab,ijkb->ijka", M, phi_i)

        dm = m - m_eq
        m_relaxed = m - s_vec[None, None, None, :] * dm
        m_with_force = (
            m_relaxed + (1.0 - 0.5 * s_vec[None, None, None, :]) * phi_m
        )
        m_post = jnp.where(CONSERVED_MOMENT_MASK, m + phi_m, m_with_force)
        f_post = jnp.einsum("ab,ijkb->ijka", invM, m_post)

        f_streamed = stream_bounceback_y(f_post)

        rho_s = jnp.sum(f_streamed, axis=-1)
        rho_s_safe = jnp.where(rho_s < 1e-15, 1.0, rho_s)

        mom_s = jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, f_streamed)
        u_s = (mom_s + 0.5 * force[None, None, None, :]) / rho_s_safe[..., None]

        return f_streamed, rho_s, u_s

    dv = dv0

    time_hist = []
    ke_hist = []
    mass_hist = []
    centerline_hist = []
    max_u_hist = []

    initial_mass = float(jax.device_get(jnp.sum(dv0)))
    print_interval = max(1, timesteps // 20)

    for step in range(1, timesteps + 1):
        dv, rho, u = update_local(dv)

        if step % sample_every == 0 or step == 1 or step == timesteps:
            rho_np = np.asarray(jax.device_get(rho))
            u_np = np.asarray(jax.device_get(u))

            ke = 0.5 * np.mean(np.sum(u_np**2, axis=-1))
            mass = np.sum(rho_np)
            ux_mean_y = np.mean(u_np[..., 0], axis=(0, 2))

            center_idx = ny // 2
            centerline_u = float(ux_mean_y[center_idx])
            max_u = float(np.max(ux_mean_y))

            time_hist.append(step)
            ke_hist.append(float(ke))
            mass_hist.append(float(mass))
            centerline_hist.append(centerline_u)
            max_u_hist.append(max_u)

        if not quiet and (step % print_interval == 0 or step == timesteps):
            print(f"  classical MRT channel step {step}/{timesteps}")

    rho_final = np.asarray(jax.device_get(rho))
    u_final = np.asarray(jax.device_get(u))

    ux_mean_y = np.mean(u_final[..., 0], axis=(0, 2))
    uy_mean_y = np.mean(u_final[..., 1], axis=(0, 2))
    uz_mean_y = np.mean(u_final[..., 2], axis=(0, 2))
    rho_mean_y = np.mean(rho_final, axis=(0, 2))

    y_analytic, ux_analytic = analytical_poiseuille_profile(
        ny=ny,
        tau=tau,
        force_x=force_x,
        rho_ref=rho0,
    )

    l2_profile_error = np.sqrt(
        np.mean((ux_mean_y - ux_analytic) ** 2)
    ) / max(np.sqrt(np.mean(ux_analytic**2)), 1e-30)

    max_profile_error = np.max(np.abs(ux_mean_y - ux_analytic))

    # Extrapolate u_x to walls at y=0 and y=H (fluid nodes at y=0.5, ny-0.5).
    wall_ux_bottom_extrapolated = 1.5 * ux_mean_y[0] - 0.5 * ux_mean_y[1]
    wall_ux_top_extrapolated = 1.5 * ux_mean_y[-1] - 0.5 * ux_mean_y[-2]

    wall_slip_extrapolated = max(
        abs(float(wall_ux_bottom_extrapolated)),
        abs(float(wall_ux_top_extrapolated)),
    )

    near_wall_node_ux_max = max(
        abs(float(ux_mean_y[0])),
        abs(float(ux_mean_y[-1])),
    )

    wall_normal_mean_max = max(
        abs(float(uy_mean_y[0])),
        abs(float(uy_mean_y[-1])),
    )

    result = {
        "params": {
            "solver": "classical_D3Q19_MRT_channel",
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "tau": tau,
            "nu": nu,
            "force_x": force_x,
            "timesteps": timesteps,
            "sample_every": sample_every,
            "initial_condition": initial_condition,
            "boundary": "halfway_bounce_back_y_walls",
            "periodic_directions": ["x", "z"],
        },
        "time": np.asarray(time_hist, dtype=np.float64),
        "kinetic_energy": np.asarray(ke_hist, dtype=np.float64),
        "mass": np.asarray(mass_hist, dtype=np.float64),
        "mass_relative_error": (
            np.asarray(mass_hist, dtype=np.float64) - float(initial_mass)
        )
        / max(abs(float(initial_mass)), 1e-30),
        "centerline_u": np.asarray(centerline_hist, dtype=np.float64),
        "max_u": np.asarray(max_u_hist, dtype=np.float64),
        "y": y_analytic,
        "ux_mean_y": ux_mean_y,
        "uy_mean_y": uy_mean_y,
        "uz_mean_y": uz_mean_y,
        "rho_mean_y": rho_mean_y,
        "ux_analytic_y": ux_analytic,
        "l2_profile_error": float(l2_profile_error),
        "max_profile_error": float(max_profile_error),
        "wall_ux_bottom_extrapolated": float(wall_ux_bottom_extrapolated),
        "wall_ux_top_extrapolated": float(wall_ux_top_extrapolated),
        "wall_slip_extrapolated": float(wall_slip_extrapolated),
        "near_wall_node_ux_max": float(near_wall_node_ux_max),
        "wall_normal_mean_max": float(wall_normal_mean_max),
        "final_mass": float(np.sum(rho_final)),
        "initial_mass": float(initial_mass),
    }

    if return_fields:
        result["rho_final"] = rho_final
        result["u_final"] = u_final
        result["f_final"] = np.asarray(jax.device_get(dv))

    return result


def run_ref_mrt(*args, **kwargs):
    """Alias for run_ref_mrt_channel."""
    return run_ref_mrt_channel(*args, **kwargs)


# --- CLI ---

def _json_safe_summary(result: dict) -> dict:
    """Compact dict for JSON logging."""
    p = result["params"]
    return {
        "solver": p["solver"],
        "nx": p["nx"],
        "ny": p["ny"],
        "nz": p["nz"],
        "tau": p["tau"],
        "nu": p["nu"],
        "force_x": p["force_x"],
        "timesteps": p["timesteps"],
        "initial_condition": p["initial_condition"],
        "l2_profile_error": result["l2_profile_error"],
        "max_profile_error": result["max_profile_error"],
        "wall_slip_extrapolated": result["wall_slip_extrapolated"],
        "near_wall_node_ux_max": result["near_wall_node_ux_max"],
        "wall_normal_mean_max": result["wall_normal_mean_max"],
        "max_mass_relative_error": float(
            np.max(np.abs(result["mass_relative_error"]))
        ),
        "final_centerline_u": float(result["centerline_u"][-1]),
        "final_max_u": float(result["max_u"][-1]),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="D3Q19 MRT channel-flow reference solver."
    )

    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--force-x", type=float, default=1.0e-6)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument(
        "--initial-condition",
        choices=["rest", "parabolic"],
        default="rest",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Output directory for .npz and .json files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    result = run_ref_mrt_channel(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        force_x=args.force_x,
        timesteps=args.timesteps,
        sample_every=args.sample_every,
        initial_condition=args.initial_condition,
        quiet=args.quiet,
        return_fields=False,
    )

    runtime = time.time() - t0
    summary = _json_safe_summary(result)
    summary["runtime_seconds"] = runtime

    npz_path = args.outdir / "ref_mrt_channel_results.npz"
    json_path = args.outdir / "ref_mrt_channel_summary.json"

    np.savez(
        npz_path,
        time=result["time"],
        kinetic_energy=result["kinetic_energy"],
        mass=result["mass"],
        mass_relative_error=result["mass_relative_error"],
        centerline_u=result["centerline_u"],
        max_u=result["max_u"],
        y=result["y"],
        ux_mean_y=result["ux_mean_y"],
        uy_mean_y=result["uy_mean_y"],
        uz_mean_y=result["uz_mean_y"],
        rho_mean_y=result["rho_mean_y"],
        ux_analytic_y=result["ux_analytic_y"],
        l2_profile_error=np.asarray(result["l2_profile_error"]),
        max_profile_error=np.asarray(result["max_profile_error"]),
        wall_ux_bottom_extrapolated=np.asarray(
            result["wall_ux_bottom_extrapolated"]
        ),
        wall_ux_top_extrapolated=np.asarray(
            result["wall_ux_top_extrapolated"]
        ),
        wall_slip_extrapolated=np.asarray(result["wall_slip_extrapolated"]),
        near_wall_node_ux_max=np.asarray(result["near_wall_node_ux_max"]),
        wall_normal_mean_max=np.asarray(result["wall_normal_mean_max"]),
    )

    json_path.write_text(json.dumps(summary, indent=2))

    print("\nClassical MRT channel-flow run complete.")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {npz_path}")
    print(f"Saved: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
