"""
Velocity Model Construction & Material Property Averaging Module
===============================================================

Constructs 2D spatial matrices for P-wave velocity c(x, z) and mass density
rho(x, z) for a 3-layered geological Earth model.

Also computes the buoyancy parameter b(x, z) = 1 / rho(x, z) and averages it
onto half-grid nodes to satisfy flux continuity across sharp layer boundaries.

Note on the averaging: the arithmetic mean of buoyancy,

    b_{i+1/2} = 0.5 * (b_i + b_{i+1}),

is identical to the HARMONIC mean of density,

    rho_{i+1/2} = 2 * rho_i * rho_{i+1} / (rho_i + rho_{i+1})

(Eq. 2.8 / 1.11 in the dissertation).

Author: Dairo Olamide Sule
Institution: Department of Physics, University of Ibadan, Nigeria
Supervisor: Dr. Aderemi
"""

import numpy as np


def build_3layered_earth_model(
    nx: int = 350,
    nz: int = 350,
    dx: float = 8.0,
    dz: float = 8.0,
    layer_depths: list = (480.0, 1280.0),
    velocities: list = (1500.0, 2600.0, 4000.0),
    densities: list = (1800.0, 2200.0, 2700.0),
):
    """Constructs a 2D 3-layered subsurface Earth model.

    Parameters
    ----------
    nx : int
        Number of grid nodes along horizontal X-axis.
    nz : int
        Number of grid nodes along vertical Z-axis (depth).
    dx : float
        Horizontal grid spacing increment (meters).
    dz : float
        Vertical grid spacing increment (meters).
    layer_depths : sequence of 2 floats
        Depth boundaries separating layers [Z1, Z2] in meters.
    velocities : sequence of 3 floats
        Compressional P-wave velocities [c1, c2, c3] in m/s.
    densities : sequence of 3 floats
        Mass densities [rho1, rho2, rho3] in kg/m^3.

    Returns
    -------
    c_grid : np.ndarray (nx, nz)
        P-wave velocity matrix c(x, z) in m/s.
    rho_grid : np.ndarray (nx, nz)
        Mass density matrix rho(x, z) in kg/m^3.
    b_grid : np.ndarray (nx, nz)
        Buoyancy matrix b(x, z) = 1 / rho(x, z) in m^3/kg.
    b_x_half : np.ndarray (nx, nz)
        Buoyancy at horizontal half-grid nodes (i+1/2, j).
    b_z_half : np.ndarray (nx, nz)
        Buoyancy at vertical half-grid nodes (i, j+1/2).
    """

    if len(layer_depths) != 2 or len(velocities) != 3 or len(densities) != 3:
        raise ValueError(
            "Expected 2 layer depths, 3 velocities and 3 densities "
            "for a 3-layered model."
        )

    c_grid = np.zeros((nx, nz), dtype=np.float32)
    rho_grid = np.zeros((nx, nz), dtype=np.float32)

    # Convert physical layer depth limits (meters) to grid indices along z
    k1 = int(round(layer_depths[0] / dz))
    k2 = int(round(layer_depths[1] / dz))

    # Keep interface indices strictly inside the grid and ordered
    k1 = max(1, min(k1, nz - 2))
    k2 = max(k1 + 1, min(k2, nz - 1))

    # Layer 1: Shallow Weathered Topsoil (0 <= Z < k1)
    c_grid[:, :k1] = velocities[0]
    rho_grid[:, :k1] = densities[0]

    # Layer 2: Sedimentary Sand-Shale Strata (k1 <= Z < k2)
    c_grid[:, k1:k2] = velocities[1]
    rho_grid[:, k1:k2] = densities[1]

    # Layer 3: Deep Crystalline Basement Rock (Z >= k2)
    c_grid[:, k2:] = velocities[2]
    rho_grid[:, k2:] = densities[2]

    # Scalar buoyancy field b(x, z) = 1 / rho(x, z)
    b_grid = (1.0 / rho_grid).astype(np.float32)

    # Half-node buoyancy (arithmetic mean of b == harmonic mean of rho).
    # Preserves flux continuity across sharp interfaces
    # (Kelly et al., 1976; Moczo et al., 2002; Hustedt et al., 2004).
    b_x_half = np.zeros((nx, nz), dtype=np.float32)
    b_x_half[:-1, :] = 0.5 * (b_grid[:-1, :] + b_grid[1:, :])
    b_x_half[-1, :] = b_grid[-1, :]  # Boundary padding

    b_z_half = np.zeros((nx, nz), dtype=np.float32)
    b_z_half[:, :-1] = 0.5 * (b_grid[:, :-1] + b_grid[:, 1:])
    b_z_half[:, -1] = b_grid[:, -1]  # Boundary padding

    return c_grid, rho_grid, b_grid, b_x_half, b_z_half