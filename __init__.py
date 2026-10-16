"""
UI Seismic FDM Project Package
==============================

A 2D Acoustic Seismic Forward Modeling Engine using the Standard-Grid
Finite-Difference Method (2nd-Order Time / 4th-Order Space) for
3-Layered Media.

Package layout (src/):
    velocity_model.py : layered c(x,z), rho(x,z), buoyancy b and half-node b
    source.py         : Ricker wavelet and additive Gaussian-smoothed soft source
    fd_solver.py      : Numba-accelerated 4th-order stencil, absorbing layers, time loop
    utils.py          : CFL / points-per-wavelength diagnostics and plotting

`main.py` (project root) drives the workflow with `from src import ...`, so
everything it needs is re-exported here.

Author: Dairo Olamide Sule (Matriculation Number: 236555)
Institution: Department of Physics, Faculty of Science, University of Ibadan, Nigeria
Supervisor: Dr. Aderemi
"""

__title__ = "UI_Seismic_FDM"
__version__ = "1.1.0"
__author__ = "Dairo Olamide Sule"
__institution__ = "University of Ibadan, Nigeria"
__supervisor__ = "Dr. Aderemi"

from .velocity_model import build_3layered_earth_model

from .source import (
    generate_ricker_wavelet,
    apply_soft_source_injection,
)

from .fd_solver import (
    run_2d_fdm_simulation,
    create_cerjan_sponge_mask,
    apply_cerjan_sponge,
    create_damping_profile,
    apply_velocity_damping,
)

from .utils import (
    calculate_cfl_dt,
    check_grid_dispersion,
    plot_velocity_model,
    plot_wavefield_snapshot,
    plot_seismogram,
    make_snapshot_schedule,
    plot_snapshot_grid,
)

__all__ = [
    "build_3layered_earth_model",
    "generate_ricker_wavelet",
    "apply_soft_source_injection",
    "run_2d_fdm_simulation",
    "create_cerjan_sponge_mask",
    "apply_cerjan_sponge",
    "create_damping_profile",
    "apply_velocity_damping",
    "calculate_cfl_dt",
    "check_grid_dispersion",
    "plot_velocity_model",
    "plot_wavefield_snapshot",
    "plot_seismogram",
    "make_snapshot_schedule",
    "plot_snapshot_grid",
]