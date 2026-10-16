"""
Main Simulation Driver Script
=============================

2D Acoustic Seismic Wave Propagation Simulation in 3-Layered Media
Standard-Grid Finite-Difference Method (2nd-Order Time / 4th-Order Space)

Project Title:
NUMERICAL SIMULATION OF 2D ACOUSTIC SEISMIC WAVE PROPAGATION IN
LAYERED MEDIA USING THE STANDARD-GRID FINITE-DIFFERENCE METHOD

Author     : Dairo Olamide Sule (Matriculation No.: 236555)
Institution: Department of Physics, Faculty of Science, University of Ibadan, Nigeria
Supervisor : Dr. Aderemi

Run from the project root:   python main.py
Companion experiment scripts (Section 5.3 of the proposal):
    python cfl_violation_test.py
    python frequency_sensitivity_test.py
"""

import os
import sys

import numpy as np


# Ensure project root directory is at head of sys.path BEFORE package imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src import (
    build_3layered_earth_model,
    generate_ricker_wavelet,
    run_2d_fdm_simulation,
    calculate_cfl_dt,
    check_grid_dispersion,
    plot_velocity_model,
    plot_wavefield_snapshot,
    plot_seismogram,
    make_snapshot_schedule,
    plot_snapshot_grid,
)


def main():
    print("=" * 72)
    print("2D ACOUSTIC SEISMIC WAVE PROPAGATION IN 3-LAYERED MEDIA")
    print(
        "Standard-Grid Finite-Difference Method "
        "(2nd-Order Time / 4th-Order Space)"
    )
    print(
        "Department of Physics, Faculty of Science, "
        "University of Ibadan, Nigeria"
    )
    print(
        "Author: Dairo Olamide Sule (Matric: 236555) | "
        "Supervisor: Dr. Aderemi"
    )
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 1. Project Directory Configuration
    # -------------------------------------------------------------------------

    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    snapshots_dir = os.path.join(figures_dir, "snapshots")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")

    for d in (figures_dir, snapshots_dir, outputs_dir):
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Grid & Geological Earth Model Parameters
    # -------------------------------------------------------------------------

    nx, nz = 350, 350  # 350 x 350 grid nodes
    dx, dz = 8.0, 8.0  # Spatial grid resolution h = 8.0 meters

    # 3-Layer Subsurface Parameters (Southwestern Nigerian Sedimentary Profile)
    #   Layer 1 : weathered topsoil / overburden   0    -  480 m  (480 m thick)
    #   Layer 2 : sedimentary sand-shale           480  - 1280 m  (800 m thick)
    #   Layer 3 : crystalline basement             1280 - 2800 m  (1520 m thick)
    layer_depths = [480.0, 1280.0]

    velocities = [1500.0, 2600.0, 4000.0]  # Layer P-wave velocities (m/s)

    densities = [1800.0, 2200.0, 2700.0]  # Layer mass densities (kg/m^3)

    # Source Location: centre of the model, INSIDE LAYER 1 (Z = 240 m)
    src_x, src_z = 1400.0, 240.0

    src_i = int(round(src_x / dx))  # Source horizontal node index (i = 175)
    src_j = int(round(src_z / dz))  # Source depth node index (j = 30)

    # Surface Receiver Line Depth (j = 5 => Z = 40 m)
    receiver_j = 5

    # Ricker Wavelet Source Parameters
    f0 = 12.0  # Dominant central peak frequency (Hz)
    t_max = 0.85  # Simulation recording duration (seconds)

    # Absorbing boundary: frame width (nodes) and layer type
    # "damping" = velocity-proportional sponge (recommended); "cerjan" = Cerjan taper
    npad = 35
    boundary = "damping"

    # CFL safety factor (< 1.0 = stable; > 1.0 violates the limit, see
    # cfl_violation_test.py)
    safety_factor = 0.8

    # Snapshots: number and time window. Spacing INCREASES with time.
    n_snapshots = 15
    t_first_snapshot = 0.040  # seconds
    snapshot_power = 1.5      # >1 => increasing gaps between snapshots

    # -------------------------------------------------------------------------
    # 3. Numerical Diagnostics: CFL Stability & Grid Dispersion Verification
    # -------------------------------------------------------------------------

    c_max = max(velocities)
    c_min = min(velocities)

    dt, cfl_limit = calculate_cfl_dt(dx, c_max, safety_factor=safety_factor)

    disp_diag = check_grid_dispersion(dx, c_min, f0)

    print("\n--- NUMERICAL DIAGNOSTICS & BOUNDARY CHECKS ---")
    print(f"Grid Size : {nx} x {nz} nodes ({nx*dx:.1f}m x {nz*dz:.1f}m)")
    print(f"Spatial Step Size (h) : {dx:.1f} m (dx = dz)")
    print(
        f"P-Wave Velocities : L1={velocities[0]}m/s | "
        f"L2={velocities[1]}m/s | L3={velocities[2]}m/s"
    )
    print(
        f"Layer Thicknesses : L1={layer_depths[0]:.0f}m | "
        f"L2={layer_depths[1]-layer_depths[0]:.0f}m | "
        f"L3={nz*dz-layer_depths[1]:.0f}m"
    )
    print(f"Source Position : X={src_x:.0f} m, Z={src_z:.0f} m (Layer 1)")
    print(f"CFL Time Step Limit : {cfl_limit*1000:.3f} ms")
    print(f"Selected Time Step dt : {dt*1000:.3f} ms")
    print(f"Max Frequency (f_max) : {disp_diag['f_max']:.1f} Hz")
    print(f"Min Wavelength (l_min): {disp_diag['lambda_min']:.2f} m")
    print(
        f"Points Per Wavelength : {disp_diag['ppw']:.2f} PPW "
        f"(Required PPW >= 5.0)"
    )
    print(
        f"Grid Dispersion Status: "
        f"{'PASSED (Dispersion-Free)' if disp_diag['is_dispersion_free'] else 'WARNING (Dispersive)'}"
    )
    print("-" * 72)

    # -------------------------------------------------------------------------
    # 4. Build 3-Layered Earth Model Matrices
    # -------------------------------------------------------------------------

    print("\n[STEP 1/4] Constructing 2D 3-layered velocity and density matrices...")

    c_grid, rho_grid, b_grid, b_x_half, b_z_half = build_3layered_earth_model(
        nx=nx, nz=nz, dx=dx, dz=dz,
        layer_depths=layer_depths,
        velocities=velocities,
        densities=densities,
    )

    plot_velocity_model(
        c_grid, dx, dz,
        layer_depths=layer_depths,
        src_x=src_x, src_z=src_z,
        receiver_j=receiver_j,
        save_path=os.path.join(figures_dir, "3layer_velocity_model.png"),
    )

    # -------------------------------------------------------------------------
    # 5. Generate Ricker Wavelet Source Time-Series
    # -------------------------------------------------------------------------

    print(f"[STEP 2/4] Generating {f0:.0f} Hz Ricker wavelet source time-series...")

    time_vec, wavelet = generate_ricker_wavelet(f0=f0, dt=dt, t_max=t_max)

    print(
        f"[INFO] Wavelet generated: {len(wavelet)} time steps "
        f"({t_max*1000:.0f} ms total duration)"
    )

    # Snapshot schedule with increasing time gaps
    snapshot_steps = make_snapshot_schedule(
        n_snapshots, dt,
        t_first=t_first_snapshot,
        t_last=(len(wavelet) - 1) * dt,
        power=snapshot_power,
    )
    snap_times_ms = [k * dt * 1000.0 for k in snapshot_steps]
    print(
        f"[INFO] {len(snapshot_steps)} snapshots at t (ms) = "
        + ", ".join(f"{t:.0f}" for t in snap_times_ms)
    )

    # -------------------------------------------------------------------------
    # 6. Execute 2D Finite-Difference Forward Wave Engine
    # -------------------------------------------------------------------------

    print("[STEP 3/4] Executing 2D explicit time-marching finite-difference solver...")

    snapshots, seismogram = run_2d_fdm_simulation(
        c_grid=c_grid, rho_grid=rho_grid, b_grid=b_grid,
        b_x_half=b_x_half, b_z_half=b_z_half,
        wavelet=wavelet, dt=dt, dx=dx, dz=dz,
        src_i=src_i, src_j=src_j,
        receiver_j=receiver_j,
        npad=npad,
        boundary=boundary,
        snapshot_steps=snapshot_steps,
    )

    # -------------------------------------------------------------------------
    # 7. Render and Save Wavefield Snapshots & Synthetic Surface Seismogram
    # -------------------------------------------------------------------------

    print("\n[STEP 4/4] Rendering publication-quality figures and exporting data...")

    panels = []
    for n, (it_step, p_snap) in enumerate(sorted(snapshots.items()), start=1):
        snap_time_ms = it_step * dt * 1000.0

        plot_wavefield_snapshot(
            p_snap, dx, dz,
            time_ms=snap_time_ms,
            c_grid=c_grid,
            layer_depths=layer_depths,
            src_x=src_x, src_z=src_z,
            save_path=os.path.join(
                snapshots_dir,
                f"snapshot_{n:02d}_t{snap_time_ms:05.0f}ms.png",
            ),
        )
        panels.append((f"({n}) t = {snap_time_ms:.0f} ms", p_snap))

    # One overview figure containing all snapshots (each panel self-scaled)
    plot_snapshot_grid(
        panels, dx, dz, ncols=5,
        layer_depths=layer_depths, src_x=src_x, src_z=src_z,
        suptitle=(
            f"Wavefield evolution, {f0:.0f} Hz Ricker source in Layer 1 "
            f"(snapshot spacing increases with time)"
        ),
        save_path=os.path.join(figures_dir, "wavefield_snapshots_overview.png"),
    )

    # Synthetic surface shot gather
    plot_seismogram(
        seismogram, dt, dx, t_max,
        receiver_z=receiver_j * dz,
        save_path=os.path.join(figures_dir, "synthetic_seismogram_shot_gather.png"),
    )

    seis_data_path = os.path.join(outputs_dir, "synthetic_seismogram.npy")
    np.save(seis_data_path, seismogram)
    print(f"[INFO] Numerical seismogram array saved to: {seis_data_path}")

    print("\n" + "=" * 72)
    print("ALL SIMULATION STEPS & FIGURE GENERATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    main()