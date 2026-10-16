"""
2D Finite-Difference Acoustic Wave Engine Module
===============================================

Solves the 2D variable-density acoustic wave equation in divergence form:

(b / c^2) * d^2p/dt^2 - [ d/dx(b * dp/dx) + d/dz(b * dp/dz) ] = s(x, z, t)

using an explicit 2nd-order temporal and 4th-order spatial finite-difference
scheme. All fields (p, c, rho, b) live on the standard collocated
grid nodes (i, j); only the buoyancy is averaged onto half-nodes so that the
divergence form (b * grad p) is differenced conservatively across interfaces.

Spatial derivative weights (9/8, -1/24) are derived from truncated Taylor
series expansions (Dablain, 1986; Levander, 1988).

Stability: for this stencil the 2D CFL limit is
    dt <= 6 / (7 * c_max * sqrt(1/dx^2 + 1/dz^2))  ->  0.6061 * h / c_max
(see utils.calculate_cfl_dt).

Absorbing boundaries (left, right and bottom edges; the top edge is the
receiver / free-surface side and is left un-damped):

  * boundary="damping" (default): sponge layer that adds a velocity-
    proportional damping term, p_tt + gamma(x) p_t, with a smooth quadratic
    ramp (Israeli & Orszag, 1981; Kosloff & Kosloff, 1986).
  * boundary="cerjan": Cerjan Gaussian taper (Cerjan et al., 1985), which
    multiplies the wavefield by exp(-(alpha*d)^2) each step.

The three outermost node rows/columns are never updated and stay at p = 0.

Author: Dairo Olamide Sule
Institution: Department of Physics, University of Ibadan, Nigeria
Supervisor: Dr. Aderemi
"""

import time
import numpy as np
from numba import jit

from .source import apply_soft_source_injection


@jit(nopython=True, fastmath=True)
def update_wavefield_stencil_4th_order(
    p_next, p_curr, p_prev,
    c_grid, b_grid, b_x_half, b_z_half,
    flux_x, flux_z,
    dt, dx, dz, nx, nz
):
    """
    Numba JIT-accelerated 2D 4th-order spatial update kernel for the
    divergence-form operator  L p = d/dx(b dp/dx) + d/dz(b dp/dz).

    Both differentiation stages use the 4th-order staggered weights
    w1 = 9/8 and w2 = -1/24 (Taylor-series derived; Dablain, 1986;
    Levander, 1988):

    Stage 1 - flux at half-nodes (i+1/2, j):
        F_x(i+1/2, j) = b_{i+1/2, j} *
            [ w1 (p_{i+1} - p_i) + w2 (p_{i+2} - p_{i-1}) ] / dx

    Stage 2 - divergence back at the nodes (i, j):
        d/dx(b dp/dx)|_i =
            [ w1 (F_{i+1/2} - F_{i-1/2}) + w2 (F_{i+3/2} - F_{i-3/2}) ] / dx

    (Using only the two-point difference F_{i+1/2} - F_{i-1/2} in stage 2
    would reduce the whole operator to 2nd-order accuracy.)

    The same is done along z. Explicit time update:

        p^{n+1} = 2 p^n - p^{n-1} + (c^2 dt^2 / b) * L p^n

    The stencil reaches 3 nodes, so a 3-node halo is left untouched (p = 0).
    `flux_x` and `flux_z` are work arrays of shape (nx, nz).
    """

    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    dt2 = dt * dt

    w1 = 9.0 / 8.0
    w2 = -1.0 / 24.0

    # Stage 1a: x-flux at (i+1/2, j) for i = 1 .. nx-3
    for i in range(1, nx - 2):
        for j in range(nz):
            flux_x[i, j] = b_x_half[i, j] * inv_dx * (
                w1 * (p_curr[i + 1, j] - p_curr[i, j])
                + w2 * (p_curr[i + 2, j] - p_curr[i - 1, j])
            )

    # Stage 1b: z-flux at (i, j+1/2) for j = 1 .. nz-3
    for i in range(nx):
        for j in range(1, nz - 2):
            flux_z[i, j] = b_z_half[i, j] * inv_dz * (
                w1 * (p_curr[i, j + 1] - p_curr[i, j])
                + w2 * (p_curr[i, j + 2] - p_curr[i, j - 1])
            )

    # Stage 2 + explicit time update on interior nodes (3-node halo excluded)
    for i in range(3, nx - 3):
        for j in range(3, nz - 3):

            div_x = inv_dx * (
                w1 * (flux_x[i, j] - flux_x[i - 1, j])
                + w2 * (flux_x[i + 1, j] - flux_x[i - 2, j])
            )
            div_z = inv_dz * (
                w1 * (flux_z[i, j] - flux_z[i, j - 1])
                + w2 * (flux_z[i, j + 1] - flux_z[i, j - 2])
            )

            c2_dt2_over_b = (c_grid[i, j] * c_grid[i, j] * dt2) / b_grid[i, j]

            p_next[i, j] = (
                2.0 * p_curr[i, j]
                - p_prev[i, j]
                + c2_dt2_over_b * (div_x + div_z)
            )


def create_cerjan_sponge_mask(
    nx: int,
    nz: int,
    npad: int = 35,
    alpha: float = 0.003
) -> np.ndarray:
    """
    Constructs a 2D Gaussian sponge damping matrix (Cerjan et al., 1985).

    NOTE: with dt ~ 1 ms and a 12 Hz source, larger alpha (e.g. the classic
    0.015) makes the layer reflect strongly rather than absorb; alpha ~ 0.003
    was the best value tested for npad = 35. The default "damping" boundary
    of run_2d_fdm_simulation performs considerably better.

        w(i) = exp( -(alpha * (npad - i))^2 ),   i = 0 .. npad-1

    where i is the node index counted from the outer edge inward. The factor
    is applied once per time step on the left, right and bottom edges (the
    top edge is left un-damped for the free surface / receiver side).
    """

    damp = np.ones((nx, nz), dtype=np.float32)

    for i in range(npad):
        factor = np.exp(-((alpha * (npad - i)) ** 2))
        damp[i, :] *= factor             # left
        damp[nx - 1 - i, :] *= factor    # right
        damp[:, nz - 1 - i] *= factor    # bottom

    return damp


def apply_cerjan_sponge(
    p_wavefield: np.ndarray,
    damp_mask: np.ndarray
) -> None:
    """Applies boundary damping in-place."""

    p_wavefield *= damp_mask


def create_damping_profile(
    nx: int,
    nz: int,
    npad: int = 35,
    a_max: float = 0.05
) -> np.ndarray:
    """
    Quadratic damping profile a(i, j) = gamma(x) * dt / 2 for the
    velocity-proportional sponge layer.

        a(d) = a_max * ((npad - d) / npad)^2,   d = 0 .. npad-1  (from the edge)

    Applied on the left, right and bottom edges (max at overlaps/corners).
    The damped update is
        p^{n+1} = [ 2 p^n - (1 - a) p^{n-1} + dt^2 (...) ] / (1 + a),
    which the solver obtains from the undamped result as
        p^{n+1} = ( p_undamped^{n+1} + a * p^{n-1} ) / (1 + a).
    """

    a = np.zeros((nx, nz), dtype=np.float32)

    for i in range(npad):
        val = a_max * ((npad - i) / npad) ** 2
        a[i, :] = np.maximum(a[i, :], val)             # left
        a[nx - 1 - i, :] = np.maximum(a[nx - 1 - i, :], val)  # right
        a[:, nz - 1 - i] = np.maximum(a[:, nz - 1 - i], val)  # bottom

    return a


def apply_velocity_damping(
    p_next: np.ndarray,
    p_prev: np.ndarray,
    a_profile: np.ndarray
) -> None:
    """Applies the velocity-proportional sponge damping in-place."""

    p_next[:, :] = (p_next + a_profile * p_prev) / (1.0 + a_profile)


def run_2d_fdm_simulation(
    c_grid: np.ndarray,
    rho_grid: np.ndarray,
    b_grid: np.ndarray,
    b_x_half: np.ndarray,
    b_z_half: np.ndarray,
    wavelet: np.ndarray,
    dt: float,
    dx: float,
    dz: float,
    src_i: int,
    src_j: int,
    receiver_j: int = 5,
    npad: int = 35,
    snapshot_interval: int = 150,
    boundary: str = "damping",
    damping_amax: float = 0.05,
    sponge_alpha: float = 0.003,
    snapshot_steps=None,
    source_sigma: float = 1.5,
    history: list = None,
    abort_amplitude: float = None,
    top_rows_zero: int = 0,
) -> tuple[dict, np.ndarray]:
    """
    Executes the full 2D finite-difference time-stepping simulation loop.

    Parameters
    ----------
    c_grid, rho_grid, b_grid : np.ndarray
        2D velocity, density, and buoyancy matrices (nx, nz).
    b_x_half, b_z_half : np.ndarray
        Buoyancy averages at half-grid nodes.
    wavelet : np.ndarray
        1D Ricker wavelet source array (one sample per time step).
    dt, dx, dz : float
        Temporal and spatial grid sampling increments.
    src_i, src_j : int
        Source node indices (X, Z).
    receiver_j : int
        Depth index (Z-node) of the surface receiver line.
    npad : int
        Number of boundary nodes in the absorbing frame.
    snapshot_interval : int
        Time-step interval for saving 2D wavefield snapshots.
    boundary : {"damping", "cerjan"}
        Absorbing-layer type (see module docstring).
    damping_amax : float
        Peak value of a = gamma*dt/2 at the outer edge ("damping" only).
    sponge_alpha : float
        Cerjan coefficient ("cerjan" only).
    snapshot_steps : iterable of int, optional
        Explicit time-step indices at which to save snapshots (allows
        non-uniform, e.g. increasing, time spacing). If given,
        `snapshot_interval` is ignored.
    source_sigma : float
        Gaussian half-width of the source injection, in grid nodes.
    history : list, optional
        If a list is passed, max|p| of every time step is appended to it
        (used to plot amplitude growth in the CFL-violation tests).
    top_rows_zero : int
        If larger than the built-in 3-row halo, this many top node rows are
        forced to p = 0 after every step. It moves the pressure-release top
        boundary to a chosen physical depth, which is needed to compare runs
        on grids of different spacing (see frequency_sensitivity_test.py).
    abort_amplitude : float, optional
        If given, the run stops as soon as max|p| exceeds this value (or is
        NaN/Inf); the wavefield at that step is stored as a final snapshot and
        the seismogram is truncated to the steps actually computed.

    Returns
    -------
    snapshots : dict
        {time_step: wavefield_matrix} snapshots.
    seismogram : np.ndarray
        Synthetic surface seismogram, shape (nt_steps, nx).
    """

    nx, nz = c_grid.shape
    nt_steps = len(wavelet)

    # Time levels p^{n-1}, p^n, p^{n+1}
    p_prev = np.zeros((nx, nz), dtype=np.float32)
    p_curr = np.zeros((nx, nz), dtype=np.float32)
    p_next = np.zeros((nx, nz), dtype=np.float32)

    # Work arrays for the half-node fluxes b * grad(p)
    flux_x = np.zeros((nx, nz), dtype=np.float32)
    flux_z = np.zeros((nx, nz), dtype=np.float32)

    if boundary not in ("damping", "cerjan"):
        raise ValueError("boundary must be 'damping' or 'cerjan'.")

    if boundary == "cerjan":
        damp_mask = create_cerjan_sponge_mask(nx, nz, npad=npad, alpha=sponge_alpha)
    else:
        a_profile = create_damping_profile(nx, nz, npad=npad, a_max=damping_amax)

    seismogram = np.zeros((nt_steps, nx), dtype=np.float32)
    snapshots = {}
    snap_set = None if snapshot_steps is None else {int(k) for k in snapshot_steps}
    track_amplitude = (history is not None) or (abort_amplitude is not None)
    steps_done = nt_steps

    print(
        f"Executing 2D Finite-Difference wavefield updates "
        f"({nt_steps} time steps)..."
    )
    start_time = time.time()

    c_src = float(c_grid[src_i, src_j])
    b_src = float(b_grid[src_i, src_j])

    for it in range(nt_steps):

        # 1. 4th-order spatial divergence & 2nd-order time update
        update_wavefield_stencil_4th_order(
            p_next, p_curr, p_prev,
            c_grid, b_grid, b_x_half, b_z_half,
            flux_x, flux_z,
            dt, dx, dz, nx, nz
        )

        # 2. Ricker wavelet soft source
        apply_soft_source_injection(
            p_next, wavelet[it], src_i, src_j, c_src, b_src, dt, sigma=source_sigma
        )

        # 3. Absorbing boundary layer
        if boundary == "cerjan":
            apply_cerjan_sponge(p_next, damp_mask)
        else:
            apply_velocity_damping(p_next, p_prev, a_profile)

        # 3b. Optional deeper pressure-release top boundary
        if top_rows_zero > 3:
            p_next[:, :top_rows_zero] = 0.0

        # 4. Surface seismogram along Z = receiver_j
        seismogram[it, :] = p_next[:, receiver_j]

        # 5. Snapshots (explicit schedule, or a regular interval)
        if snap_set is not None:
            save_now = it in snap_set
        else:
            save_now = it > 0 and (it % snapshot_interval == 0 or it == nt_steps - 1)
        if save_now:
            snapshots[it] = p_next.copy()

        # 5b. Amplitude tracking / early abort (CFL-violation experiments)
        if track_amplitude:
            pmax = float(np.abs(p_next).max())
            if history is not None:
                history.append(pmax)
            if abort_amplitude is not None and (
                not np.isfinite(pmax) or pmax > abort_amplitude
            ):
                snapshots[it] = p_next.copy()
                steps_done = it + 1
                print(
                    f"[ABORT] Amplitude exceeded {abort_amplitude:.1e} at "
                    f"step {it} (t = {it * dt * 1000.0:.1f} ms): unstable."
                )
                break

        # 6. Cycle time levels by swapping references (no array copies):
        #    p^{n-1} <- p^n,  p^n <- p^{n+1},  old p^{n-1} is reused as p_next
        p_prev, p_curr, p_next = p_curr, p_next, p_prev

    elapsed = time.time() - start_time

    seismogram = seismogram[:steps_done]

    if not np.all(np.isfinite(p_curr)):
        print(
            "[WARNING] Wavefield contains NaN/Inf: the simulation went "
            "unstable (check dt against the CFL limit)."
        )

    print(f"[INFO] Simulation completed in {elapsed:.2f} seconds!")

    return snapshots, seismogram