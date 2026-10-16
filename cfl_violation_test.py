"""
CFL-Violation Experiment  (Proposal Section 5.3: "Violating the CFL Condition")
==============================================================================

Runs the SAME model and source as main.py with time steps that are a chosen
fraction/multiple of the theoretical CFL limit

    dt_CFL = 6 / (7 * c_max * sqrt(1/dx^2 + 1/dz^2))  = 0.6061 * h / c_max,

and shows how the explicit scheme goes from stable (dt < dt_CFL) to unstable
(dt > dt_CFL): exponentially growing, checkerboard-like noise that swamps the
physical wavefield.

Outputs (figures/cfl_violation/ and outputs/):
    * cfl_snapshots_grid.png        rows = dt/dt_CFL, columns = 4 times
    * cfl_amplitude_growth.png      max|p| versus time (log scale)
    * cfl_checkerboard_zoom.png     node-scale view of the unstable mode
    * cfl_unstable_<factor>.png     last snapshot of every violating run
    * cfl_violation_summary.csv     numbers used in the report

Usage (from the project root):   python cfl_violation_test.py
Edit CFL_FACTORS below to test other values.
"""

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import (
    build_3layered_earth_model,
    generate_ricker_wavelet,
    run_2d_fdm_simulation,
    calculate_cfl_dt,
    plot_wavefield_snapshot,
    plot_snapshot_grid,
)

# --------------------------------------------------------------------------
# Experiment settings (same physical set-up as main.py)
# --------------------------------------------------------------------------
NX, NZ = 350, 350
DX = DZ = 8.0
LAYER_DEPTHS = [480.0, 1280.0]
VELOCITIES = [1500.0, 2600.0, 4000.0]
DENSITIES = [1800.0, 2200.0, 2700.0]
SRC_X, SRC_Z = 1400.0, 240.0
RECEIVER_J = 5
F0 = 12.0
T_MAX = 0.85
NPAD = 35

# dt = factor * dt_CFL.   <1 stable, >1 violates the CFL condition.
CFL_FACTORS = [0.80, 0.99, 1.02, 1.10, 1.50]

ABORT_AMPLITUDE = 1.0e10      # stop a run once |p| exceeds this (float32 safety)
SNAP_EVERY = 10               # store a snapshot every N steps, pick panels later
PANEL_FRACTIONS = [0.25, 0.50, 0.75, 1.00]   # of the (possibly aborted) run


def theoretical_growth_per_step(factor: float) -> float:
    """Largest |mu| of the leapfrog amplification factor for dt = factor*dt_CFL.

    The most unstable (Nyquist-like) mode satisfies
        mu + 1/mu = 2 - 4 factor^2   ->  |mu| = 1 for factor <= 1.
    """

    q = 4.0 * factor ** 2 - 2.0
    if q <= 2.0:
        return 1.0
    return 0.5 * (q + np.sqrt(q * q - 4.0))


def run_case(factor, c_grid, rho_grid, b_grid, b_x_half, b_z_half):
    c_max = float(c_grid.max())
    dt, dt_cfl = calculate_cfl_dt(DX, c_max, safety_factor=factor)
    _, wavelet = generate_ricker_wavelet(f0=F0, dt=dt, t_max=T_MAX)
    nt = len(wavelet)

    steps = sorted(set(range(SNAP_EVERY, nt, SNAP_EVERY)) | {nt - 1})
    history = []

    snaps, _ = run_2d_fdm_simulation(
        c_grid=c_grid, rho_grid=rho_grid, b_grid=b_grid,
        b_x_half=b_x_half, b_z_half=b_z_half,
        wavelet=wavelet, dt=dt, dx=DX, dz=DZ,
        src_i=int(round(SRC_X / DX)), src_j=int(round(SRC_Z / DZ)),
        receiver_j=RECEIVER_J, npad=NPAD,
        snapshot_steps=steps,
        history=history,
        abort_amplitude=ABORT_AMPLITUDE,
    )
    return dt, dt_cfl, np.array(history), snaps


def main():
    fig_dir = os.path.join(PROJECT_ROOT, "figures", "cfl_violation")
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("CFL-VIOLATION EXPERIMENT")
    print("=" * 72)

    c_grid, rho_grid, b_grid, b_x_half, b_z_half = build_3layered_earth_model(
        NX, NZ, DX, DZ, LAYER_DEPTHS, VELOCITIES, DENSITIES
    )

    panels, rows, zoom_panels = [], [], []
    growth_curves = {}

    for factor in CFL_FACTORS:
        print(f"\n--- dt = {factor:.2f} x dt_CFL ---")
        dt, dt_cfl, hist, snaps = run_case(
            factor, c_grid, rho_grid, b_grid, b_x_half, b_z_half
        )

        n_done = len(hist)
        t_ms = np.arange(n_done) * dt * 1000.0
        growth_curves[factor] = (t_ms, hist)
        aborted = hist.max() > ABORT_AMPLITUDE or not np.isfinite(hist[-1])

        # --- measured growth per step (fit where growth is established) ---
        measured = np.nan
        if aborted:
            # points well above the physical wavefield amplitude (~1e4)
            sel = np.where((hist > 1e5) & np.isfinite(hist))[0]
            if len(sel) >= 3:
                slope = np.polyfit(sel, np.log(hist[sel]), 1)[0]
                measured = float(np.exp(slope))

        theory = theoretical_growth_per_step(factor)

        # --- choose 4 panels at fractions of the run ---
        keys = np.array(sorted(snaps.keys()))
        last = keys[-1]
        for frac in PANEL_FRACTIONS:
            k = int(keys[np.argmin(np.abs(keys - frac * last))])
            field = snaps[k]
            amp = float(np.nanmax(np.abs(field)))
            tag = "  [ABORTED]" if (aborted and k == last) else ""
            panels.append((
                f"dt = {factor:.2f} x CFL | t = {k * dt * 1000:.0f} ms | "
                f"max|p| = {amp:.1e}{tag}",
                field,
            ))

        # last snapshot of each violating run as a stand-alone figure
        if factor > 1.0:
            zoom_panels.append((factor, dt, last, snaps[last]))
            plot_wavefield_snapshot(
                snaps[last], DX, DZ, time_ms=last * dt * 1000.0,
                layer_depths=LAYER_DEPTHS, src_x=SRC_X, src_z=SRC_Z,
                save_path=os.path.join(fig_dir, f"cfl_unstable_{factor:.2f}.png"),
            )

        rows.append({
            "dt_over_dt_CFL": factor,
            "dt_ms": round(dt * 1000, 4),
            "status": "UNSTABLE" if aborted else "stable",
            "abort_time_ms": round((n_done - 1) * dt * 1000, 1) if aborted else "",
            "final_max_abs_p": f"{hist[-1]:.3e}",
            "theory_growth_per_step": round(theory, 3),
            "measured_growth_per_step": round(measured, 3) if np.isfinite(measured) else "",
        })
        print(
            f"status={rows[-1]['status']}  final max|p|={hist[-1]:.2e}  "
            f"theory growth/step={theory:.3f}  measured={measured:.3f}"
        )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_snapshot_grid(
        panels, DX, DZ, ncols=len(PANEL_FRACTIONS),
        layer_depths=LAYER_DEPTHS, src_x=SRC_X, src_z=SRC_Z,
        suptitle="Effect of the time step on stability (each panel scaled to its own maximum)",
        save_path=os.path.join(fig_dir, "cfl_snapshots_grid.png"),
    )

    # Node-scale zoom: shows the checkerboard nature of the instability
    fig, axes = plt.subplots(1, len(zoom_panels), figsize=(4.6 * len(zoom_panels), 4.4),
                             dpi=150, squeeze=False)
    for ax, (factor, dt, last, field) in zip(axes[0], zoom_panels):
        field = np.nan_to_num(field)
        ci, cj = np.unravel_index(np.argmax(np.abs(field)), field.shape)
        h = 12
        i0, j0 = max(ci - h, 0), max(cj - h, 0)
        crop = field[i0:i0 + 2 * h, j0:j0 + 2 * h]
        vm = np.abs(crop).max() or 1.0
        ax.imshow(crop.T, cmap="seismic", vmin=-vm, vmax=vm, interpolation="nearest",
                  extent=[i0 * DX, (i0 + crop.shape[0]) * DX,
                          (j0 + crop.shape[1]) * DZ, j0 * DZ])
        ax.set_title(f"dt = {factor:.2f} x CFL, t = {last * dt * 1000:.0f} ms\n"
                     f"zoom on {2 * h}x{2 * h} nodes (1 pixel = 1 node)", fontsize=9)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Depth Z (m)")
    fig.suptitle("Node-scale structure of the unstable mode (alternating sign between neighbouring nodes)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    zoom_path = os.path.join(fig_dir, "cfl_checkerboard_zoom.png")
    fig.savefig(zoom_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Checkerboard zoom saved to: {zoom_path}")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for factor, (t_ms, hist) in growth_curves.items():
        ax.semilogy(t_ms, np.maximum(hist, 1e-12), linewidth=1.8,
                    label=f"dt = {factor:.2f} x dt_CFL")
    ax.axhline(ABORT_AMPLITUDE, color="k", linestyle=":", linewidth=1)
    ax.text(5, ABORT_AMPLITUDE * 1.5, "run aborted above this amplitude", fontsize=8)
    ax.set_xlabel("Simulation time t (ms)")
    ax.set_ylabel("max |p(x, z, t)|  (arbitrary units)")
    ax.set_title("Amplitude growth: stable vs CFL-violating time steps")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    growth_path = os.path.join(fig_dir, "cfl_amplitude_growth.png")
    fig.savefig(growth_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Amplitude-growth plot saved to: {growth_path}")

    csv_path = os.path.join(out_dir, "cfl_violation_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] Summary table saved to: {csv_path}")

    print("\n" + "=" * 72)
    print(f"{'dt/dt_CFL':>10} {'status':>9} {'abort t (ms)':>13} "
          f"{'theory mu':>10} {'measured mu':>12}")
    for r in rows:
        print(f"{r['dt_over_dt_CFL']:>10.2f} {r['status']:>9} "
              f"{str(r['abort_time_ms']):>13} {r['theory_growth_per_step']:>10} "
              f"{str(r['measured_growth_per_step']):>12}")
    print("=" * 72)


if __name__ == "__main__":
    main()