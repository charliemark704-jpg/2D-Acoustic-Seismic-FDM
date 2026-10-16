"""
Utility, Stability Diagnostics & Visualization Module
===================================================

Provides diagnostic tools for evaluating Courant-Friedrichs-Lewy (CFL)
stability limits, Points Per Wavelength (PPW) numerical grid dispersion
thresholds, and publication-quality plotting routines for 2D velocity
models, wavefield snapshots, and synthetic surface seismograms.

Author: Dairo Olamide Sule
Institution: Department of Physics, University of Ibadan, Nigeria
Supervisor: Dr. Aderemi
"""

import matplotlib.pyplot as plt
import numpy as np


def calculate_cfl_dt(
    h: float,
    c_max: float,
    safety_factor: float = 0.8
) -> tuple[float, float]:
    """Calculates the maximum allowable time step dt based on the
    Courant-Friedrichs-Lewy (CFL) condition for a 2nd-order time,
    4th-order space explicit finite-difference scheme.

    CFL Condition (2D, 2nd-order time, 4th-order space using the staggered
    first-derivative weights 9/8 and -1/24 that the solver applies twice):

    dt <= 6 / (7 * c_max * sqrt(1/dx^2 + 1/dz^2))
        = (6 / (7 * sqrt(2))) * (h / c_max)      for dx = dz = h
        ≈ 0.6061 * (h / c_max)

    (The factor 6/7 = 1 / (9/8 + 1/24) is the reciprocal of the sum of the
    absolute stencil weights.)

    Parameters
    ----------
    h : float
        Spatial grid spacing increment (meters, assuming dx = dz = h).
    c_max : float
        Maximum compressional P-wave velocity in the medium (m/s).
    safety_factor : float
        Safety factor multiplier (typically 0.7 - 0.9) to guarantee
        numerical stability.

    Returns
    -------
    dt : float
        Selected stable time step increment (seconds).
    cfl_limit : float
        Theoretical maximum CFL stability limit (seconds).
    """

    # Courant constant for the implemented 4th-order stencil: 6 / (7 sqrt 2)
    cfl_constant = 6.0 / (7.0 * np.sqrt(2.0))  # ≈ 0.6061

    cfl_limit = cfl_constant * (h / c_max)
    dt = cfl_limit * safety_factor

    return dt, cfl_limit


def check_grid_dispersion(
    h: float,
    c_min: float,
    f0: float
) -> dict:
    """Evaluates grid sampling density (Points Per Wavelength, PPW)
    to prevent numerical grid dispersion.

    Formula:

    PPW = lambda_min / h = c_min / (f_max * h)

    where f_max ≈ 2.5 * f0 is the upper significant frequency limit
    of the Ricker wavelet.

    Target: PPW >= 5.0 for 4th-order spatial stencils
    (Levander, 1988; Dablain, 1986).

    Parameters
    ----------
    h : float
        Spatial grid sampling increment (meters).
    c_min : float
        Minimum P-wave velocity in the model (m/s).
    f0 : float
        Central dominant frequency of the Ricker wavelet (Hz).

    Returns
    -------
    diagnostics : dict
        Dictionary containing f_max, lambda_min, ppw, and dispersion
        status boolean.
    """

    f_max = 2.5 * f0
    lambda_min = c_min / f_max
    ppw = lambda_min / h
    is_dispersion_free = ppw >= 5.0

    return {
        "f_max": f_max,
        "lambda_min": lambda_min,
        "ppw": ppw,
        "is_dispersion_free": is_dispersion_free,
    }


def plot_velocity_model(
    c_grid: np.ndarray,
    dx: float,
    dz: float,
    layer_depths: list = (480.0, 1280.0),
    src_x: float = 1400.0,
    src_z: float = 240.0,
    receiver_j: int = 5,
    save_path: str = None,
) -> None:
    """Generates a publication-quality plot of the 2D subsurface
    3-layered velocity model c(x, z).
    """

    nx, nz = c_grid.shape
    extent = [0, nx * dx, nz * dz, 0]

    plt.figure(figsize=(10, 6), dpi=150)

    im = plt.imshow(
        c_grid.T,
        cmap="viridis",
        extent=extent,
        aspect="auto"
    )

    # Label each layer with its velocity (read from the model itself)
    bounds = [0.0] + [float(d) for d in layer_depths] + [nz * dz]
    for k in range(len(bounds) - 1):
        z_mid = 0.5 * (bounds[k] + bounds[k + 1])
        vel = float(c_grid[nx // 2, min(int(z_mid / dz), nz - 1)])
        plt.text(
            nx * dx * 0.02, z_mid, f"Layer {k + 1}:  {vel:.0f} m/s",
            color="white", fontsize=10, fontweight="bold", va="center",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2.5),
        )

    # Layer boundary interface lines
    for depth in layer_depths:
        plt.axhline(
            y=depth,
            color="white",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9
        )

    # Source marker
    plt.scatter(
        [src_x],
        [src_z],
        color="gold",
        marker="*",
        s=220,
        edgecolors="black",
        zorder=5,
        label="Source (12 Hz Ricker)",
    )

    # Receiver array line near surface
    rec_x = np.arange(10, nx - 10) * dx
    rec_z = np.full_like(rec_x, receiver_j * dz)

    plt.scatter(
        rec_x[::10],
        rec_z[::10],
        color="red",
        marker="v",
        s=30,
        edgecolors="black",
        zorder=4,
        label="Receivers",
    )

    plt.title(
        "2D Subsurface Velocity Model $c(x, z)$ — 3-Layered Media\n"
        "Department of Physics, University of Ibadan",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    plt.xlabel(
        "Horizontal Distance $X$ (m)",
        fontsize=11
    )

    plt.ylabel(
        "Depth $Z$ (m)",
        fontsize=11
    )

    plt.colorbar(
        im,
        label="P-Wave Velocity (m/s)"
    )

    plt.legend(
        loc="lower right",
        framealpha=0.9
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"[INFO] Velocity model plot saved to: {save_path}"
        )

    plt.close()


def plot_wavefield_snapshot(
    p_wavefield: np.ndarray,
    dx: float,
    dz: float,
    time_ms: float,
    c_grid: np.ndarray = None,
    layer_depths: list = (480.0, 1280.0),
    src_x: float = 1400.0,
    src_z: float = 240.0,
    save_path: str = None,
) -> None:
    """Plots a 2D spatial wavefield amplitude snapshot at a specific time t.
    """

    nx, nz = p_wavefield.shape
    extent = [0, nx * dx, nz * dz, 0]

    vmax = np.percentile(
        np.abs(p_wavefield),
        99.5
    )

    if vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(10, 6), dpi=150)

    im = plt.imshow(
        p_wavefield.T,
        cmap="seismic",
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    # Layer boundary overlays
    for depth in layer_depths:
        plt.axhline(
            y=depth,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7
        )

    # Source location
    plt.scatter(
        [src_x],
        [src_z],
        color="gold",
        marker="*",
        s=180,
        edgecolors="black",
        zorder=5,
    )

    plt.title(
        f"2D Acoustic Wavefield Snapshot at $t = {time_ms:.1f}$ ms\n"
        "(2nd-Order Time / 4th-Order Space FDM)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    plt.xlabel(
        "Horizontal Offset $X$ (m)",
        fontsize=11
    )

    plt.ylabel(
        "Depth $Z$ (m)",
        fontsize=11
    )

    plt.colorbar(
        im,
        label="Acoustic Wavefield Pressure $p(x, z, t)$"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"[INFO] Wavefield snapshot ({time_ms:.0f} ms) saved to: {save_path}"
        )

    plt.close()


def plot_seismogram(
    seismogram: np.ndarray,
    dt: float,
    dx: float,
    t_max: float,
    receiver_z: float = 40.0,
    save_path: str = None,
) -> None:
    """Plots a synthetic surface seismogram (shot gather)
    displaying reflected and refracted arrivals.
    """

    nt, nx = seismogram.shape
    extent = [
        0,
        (nx - 1) * dx,
        (nt - 1) * dt * 1000,
        0
    ]

    vmax = np.percentile(
        np.abs(seismogram),
        99.0
    )

    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(
        figsize=(10, 7),
        dpi=150
    )

    # Image density gather background
    im = ax.imshow(
        seismogram,
        cmap="PuOr",
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto"
    )

    ax.set_title(
        f"Synthetic Surface Seismogram Shot Gather (Receivers $Z = {receiver_z:.0f}$ m)\n"
        "2D Acoustic Wave Propagation in 3-Layered Media",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Receiver Distance / Offset $X$ (m)",
        fontsize=11
    )

    ax.set_ylabel(
        "Recording Time $t$ (ms)",
        fontsize=11
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Pressure Amplitude $p(x, z=z_{rec}, t)$"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"[INFO] Synthetic seismogram plot saved to: {save_path}"
        )

    plt.close()


def make_snapshot_schedule(
    n_snap: int,
    dt: float,
    t_first: float,
    t_last: float,
    power: float = 1.5
) -> list:
    """Time-step indices for snapshots whose spacing INCREASES with time.

    t_k = t_first + (t_last - t_first) * (k / (n_snap - 1)) ** power,
    k = 0 .. n_snap-1. With power > 1 the gap between successive snapshots
    grows (dense early, when the wavefronts are compact; sparse late).

    Parameters
    ----------
    n_snap : int      Number of snapshots requested.
    dt : float        Time step (s).
    t_first, t_last : float   First / last snapshot time (s).
    power : float     Spacing exponent (1 = uniform, >1 = increasing gaps).

    Returns
    -------
    steps : list of int   Sorted, unique time-step indices.
    """

    k = np.arange(n_snap) / max(n_snap - 1, 1)
    t = t_first + (t_last - t_first) * k ** power
    steps = np.unique(np.round(t / dt).astype(int))
    return [int(s) for s in steps if s > 0]


def plot_snapshot_grid(
    panels: list,
    dx: float,
    dz: float,
    ncols: int = 3,
    layer_depths: list = (480.0, 1280.0),
    src_x: float = 1400.0,
    src_z: float = 240.0,
    suptitle: str = None,
    save_path: str = None,
) -> None:
    """Multi-panel figure of wavefield snapshots.

    Parameters
    ----------
    panels : list of (title, 2D array)
        One entry per panel, filled row by row. Each panel is scaled
        independently to +/- its own 99.5th-percentile amplitude, so
        differences in absolute amplitude are stated in the panel titles.
    """

    n = len(panels)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.6 * ncols, 3.5 * nrows),
        dpi=130,
        squeeze=False,
    )

    for idx, ax in enumerate(axes.ravel()):
        if idx >= n:
            ax.axis("off")
            continue

        title, field = panels[idx]
        field = np.nan_to_num(np.asarray(field, dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
        nx, nz = field.shape
        vmax = np.percentile(np.abs(field), 99.5)
        if vmax == 0:
            vmax = 1.0

        ax.imshow(
            field.T,
            cmap="seismic",
            extent=[0, nx * dx, nz * dz, 0],
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        for depth in layer_depths:
            ax.axhline(depth, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.scatter([src_x], [src_z], color="gold", marker="*", s=80,
                   edgecolors="black", linewidths=0.6, zorder=5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Depth Z (m)", fontsize=8)
        if idx >= n - ncols or idx + ncols >= n:
            ax.set_xlabel("X (m)", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.97 if suptitle else 1))

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Snapshot grid saved to: {save_path}")

    plt.close(fig)