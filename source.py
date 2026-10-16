"""
Seismic Source Wavelet Generation & Spatial Injection Module
===========================================================

Generates time-series functions for a band-limited Ricker wavelet
(the second derivative of a Gaussian pulse) and handles additive
soft-source spatial injection onto the computational finite-difference
grid mesh.

Author: Dairo Olamide Sule
Institution: Department of Physics, University of Ibadan, Nigeria
Supervisor: Dr. Aderemi
"""

import numpy as np


def generate_ricker_wavelet(
    f0: float = 12.0,
    dt: float = 0.001,
    t_max: float = 0.85
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a zero-phase, band-limited Ricker wavelet source time-series.

    The Ricker wavelet is mathematically defined as the second derivative
    of a Gaussian function (Ricker, 1953):

    s(t) = (1 - 2 * pi^2 * f0^2 * (t - t0)^2)
           * exp(-pi^2 * f0^2 * (t - t0)^2)

    Parameters
    ----------
    f0 : float
        Dominant central peak frequency of the wavelet (Hz).
    dt : float
        Temporal sampling interval (seconds).
    t_max : float
        Total simulation time duration (seconds).

    Returns
    -------
    time_vec : np.ndarray
        1D array of discrete time points [0, dt, 2*dt, ..., t_max].
    wavelet : np.ndarray
        1D array of Ricker wavelet amplitude values at each time step.
    """

    # Localized time delay ensuring smooth, causal onset at t = 0
    t0 = 1.0 / f0

    time_vec = np.arange(0.0, t_max, dt, dtype=np.float32)

    # Dimensionless tau time parameter
    tau = np.pi * f0 * (time_vec - t0)

    # Ricker wavelet formula
    wavelet = (1.0 - 2.0 * (tau**2)) * np.exp(-(tau**2))

    return time_vec, wavelet.astype(np.float32)


def apply_soft_source_injection(
    p_next: np.ndarray,
    source_val: float,
    src_i: int,
    src_j: int,
    c_val: float,
    b_val: float,
    dt: float,
    sigma: float = 1.5,
) -> None:
    """Injects a point source additively ("soft source") with Gaussian spatial smoothing.

    Unlike a "hard source" that overwrites grid node values and creates
    artificial rigid boundary reflections, an additive "soft source"
    adds wave energy as a force term to p^{n+1}, allowing returning
    subsurface wavefield reflections to pass through the source coordinate
    undisturbed (Kelly et al., 1976).

    Spatial Gaussian smoothing distributes the point source across a
    localized cluster of neighboring grid nodes, preventing spatial
    Dirac delta singularities and suppressing high-frequency
    grid-dispersion noise.

    Parameters
    ----------
    p_next : np.ndarray
        (nx, nz) Future wavefield pressure matrix p^{n+1} to be modified
        in-place.
    source_val : float
        Scalar Ricker wavelet amplitude value s(n * dt) at the current
        time step.
    src_i : int
        Horizontal grid index (X-coordinate) of the source center.
    src_j : int
        Vertical grid index (Z-coordinate / depth) of the source center.
    c_val : float
        P-wave velocity c(x, z) at the source coordinate node (m/s).
    b_val : float
        Buoyancy b(x, z) = 1 / rho(x, z) at the source coordinate node.
    dt : float
        Temporal grid step size (seconds).
    sigma : float
        Spatial Gaussian filter standard deviation (in grid units).
    """

    if abs(source_val) < 1e-8:
        return

    nx, nz = p_next.shape

    radius = int(np.ceil(3.0 * sigma))

    # Scaling multiplier from the discretized wave equation: (c^2 * dt^2 / b)
    factor = (c_val ** 2 * dt ** 2) / b_val

    # Normalised 2D Gaussian weights over the local window, so the total
    # injected strength is independent of sigma.
    offsets = np.arange(-radius, radius + 1)
    di, dj = np.meshgrid(offsets, offsets, indexing="ij")
    weights = np.exp(-(di ** 2 + dj ** 2) / (2.0 * sigma ** 2))
    weights /= weights.sum()

    i0, i1 = src_i - radius, src_i + radius + 1
    j0, j1 = src_j - radius, src_j + radius + 1

    # Clip the window to the grid (safe when the source is near an edge)
    wi0, wj0 = max(0, -i0), max(0, -j0)
    wi1 = weights.shape[0] - max(0, i1 - nx)
    wj1 = weights.shape[1] - max(0, j1 - nz)

    p_next[max(i0, 0):min(i1, nx), max(j0, 0):min(j1, nz)] += (
        factor * source_val * weights[wi0:wi1, wj0:wj1]
    ).astype(p_next.dtype)