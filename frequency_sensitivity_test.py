"""
Source-Frequency Sensitivity Experiment
(Proposal Section 5.3: "Varying Frequencies")
=======================================================================

Question: what happens when the source frequency is raised while the grid
(h = 8 m) and time step are kept fixed?

For a 4th-order scheme the grid must supply about G >= 5 points per shortest
wavelength, G = c_min / (f_max * h) with f_max = 2.5 * f0 (Eq. 2.7). Raising
f0 shortens the wavelength, G falls below 5, and numerical GRID DISPERSION
appears: high-frequency components travel slower than the true wave speed,
producing lagging "ripples" behind the wavefront and distorted arrivals.

Method
------
1. For each f0 in F0_LIST run the standard model on the h = 8 m grid
   (same dt for every frequency, so only the source changes).
2. Run the same source on a refined h = 8/3 m grid (same physical model, source
   position and receiver line) as a nearly dispersion-free reference.
3. Compare the seismograms of the h = 8 m run with the reference using a
   gain-corrected relative misfit
        E = || a*S_8 - S_4 || / || S_4 ||,   a = <S_8, S_4> / <S_8, S_8>,
   (the gain a removes the arbitrary source amplitude scaling so that E
   measures waveform distortion only), over receivers outside the sponge.

Outputs (figures/frequency_sensitivity/ and outputs/):
    * freq_snapshots_grid.png      rows = f0, columns = snapshot times
    * freq_trace_comparison.png    h=8 m vs h=8/3 m trace at ~800 m offset
    * freq_error_vs_ppw.png        waveform misfit versus points per wavelength
    * freq_snapshot_f<f0>Hz.png    one stand-alone snapshot per frequency
    * frequency_sensitivity_summary.csv

The h = 8/3 m reference keeps G >= 5.6 even at 40 Hz, so it is adequate for
every frequency tested. Refining by an ODD integer keeps the sharp interfaces at
exactly the same physical depth on both grids (interfaces sit half-way between
nodes), and the pressure-release top boundary is placed at the same depth
(z = 16 m) through `top_rows_zero`.

Usage (from the project root):   python frequency_sensitivity_test.py
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
    check_grid_dispersion,
    plot_wavefield_snapshot,
    plot_snapshot_grid,
)

# --------------------------------------------------------------------------
# Settings (same physical set-up as main.py)
# --------------------------------------------------------------------------
F0_LIST = [8.0, 12.0, 20.0, 30.0, 40.0]        # Hz
SNAPSHOT_TIMES_MS = [250.0, 450.0, 650.0]

NX, NZ = 350, 350
DX = DZ = 8.0
LAYER_DEPTHS = [480.0, 1280.0]
VELOCITIES = [1500.0, 2600.0, 4000.0]
DENSITIES = [1800.0, 2200.0, 2700.0]
SRC_X, SRC_Z = 1400.0, 240.0
RECEIVER_J = 5
T_MAX = 0.85
NPAD = 35
SAFETY = 0.8

REF_REFINE = 3                 # reference grid: h/3 (2.667 m); keeps the interfaces
                               # at exactly the same physical depth as the h = 8 m grid
TRACE_OFFSET_M = 800.0         # offset of the trace shown in the comparison


def build(refine):
    """Model on a grid refined by an integer factor (same physical size)."""
    nx, nz, dx, dz = NX * refine, NZ * refine, DX / refine, DZ / refine
    # The model switches layer half-way between two nodes, i.e. at
    # (k - 1/2) * h with k = round(depth / h). To put the refined-grid interface
    # at exactly the same physical depth as on the h = 8 m grid, shift the
    # nominal depth by (h_ref - h)/2 (exact for odd `refine`).
    depths = [d + 0.5 * (dz - DZ) for d in LAYER_DEPTHS]
    grids = build_3layered_earth_model(
        nx, nz, dx, dz, depths, VELOCITIES, DENSITIES
    )
    return nx, nz, dx, dz, grids


def simulate(f0, refine, snapshot_times_ms=None):
    nx, nz, dx, dz, (c, rho, b, bxh, bzh) = build(refine)
    dt, _ = calculate_cfl_dt(dx, max(VELOCITIES), safety_factor=SAFETY)
    _, wavelet = generate_ricker_wavelet(f0=f0, dt=dt, t_max=T_MAX)

    steps = None
    if snapshot_times_ms is not None:
        steps = [int(round(t / 1000.0 / dt)) for t in snapshot_times_ms]

    snaps, seis = run_2d_fdm_simulation(
        c_grid=c, rho_grid=rho, b_grid=b, b_x_half=bxh, b_z_half=bzh,
        wavelet=wavelet, dt=dt, dx=dx, dz=dz,
        src_i=int(round(SRC_X / dx)), src_j=int(round(SRC_Z / dz)),
        receiver_j=RECEIVER_J * refine,
        npad=NPAD * refine,
        snapshot_steps=steps,
        source_sigma=1.5 * refine,              # same PHYSICAL source width
        damping_amax=0.05 / refine,             # same physical damping (dt/refine)
        top_rows_zero=2 * refine + 1,           # pressure-release top at z = 16 m on every grid
    )
    return dt, snaps, seis


def gain_corrected_misfit(s_coarse, s_ref):
    a = float(np.sum(s_coarse * s_ref) / np.sum(s_coarse * s_coarse))
    err = float(np.linalg.norm(a * s_coarse - s_ref) / np.linalg.norm(s_ref))
    return err, a


def main():
    fig_dir = os.path.join(PROJECT_ROOT, "figures", "frequency_sensitivity")
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("SOURCE-FREQUENCY SENSITIVITY EXPERIMENT")
    print("=" * 72)

    c_min = min(VELOCITIES)
    src_i = int(round(SRC_X / DX))
    recv_ok = slice(NPAD + 5, NX - NPAD - 5)          # receivers outside sponge
    trace_i = src_i + int(round(TRACE_OFFSET_M / DX))

    panels, rows, traces = [], [], {}

    for f0 in F0_LIST:
        diag = check_grid_dispersion(DX, c_min, f0)
        print(f"\n--- f0 = {f0:.0f} Hz | f_max = {diag['f_max']:.0f} Hz | "
              f"lambda_min = {diag['lambda_min']:.1f} m | PPW = {diag['ppw']:.2f} ---")

        dt, snaps, seis = simulate(f0, 1, SNAPSHOT_TIMES_MS)
        dt_ref, _, seis_ref = simulate(f0, REF_REFINE)

        nt = seis.shape[0]
        # reference sample m is recorded at (m+1)*dt/R; m = R*it + R-1 coincides with
        # coarse sample it, recorded at (it+1)*dt
        ref_sub = seis_ref[REF_REFINE - 1::REF_REFINE][:nt, ::REF_REFINE]
        n = min(nt, ref_sub.shape[0])

        err, gain = gain_corrected_misfit(
            seis[:n, recv_ok], ref_sub[:n, recv_ok]
        )
        print(f"waveform misfit vs h=8/3 m reference: {100 * err:.1f} %")

        traces[f0] = (
            np.arange(n) * dt * 1000.0,
            gain * seis[:n, trace_i],
            ref_sub[:n, trace_i],
            err,
        )

        # snapshot panels (row = f0)
        for k, t_ms in zip(sorted(snaps), SNAPSHOT_TIMES_MS):
            panels.append((
                f"f0 = {f0:.0f} Hz (PPW = {diag['ppw']:.1f}) | t = {k * dt * 1000:.0f} ms",
                snaps[k],
            ))

        # stand-alone snapshot at ~450 ms
        k_mid = sorted(snaps)[len(snaps) // 2]
        plot_wavefield_snapshot(
            snaps[k_mid], DX, DZ, time_ms=k_mid * dt * 1000.0,
            layer_depths=LAYER_DEPTHS, src_x=SRC_X, src_z=SRC_Z,
            save_path=os.path.join(fig_dir, f"freq_snapshot_f{f0:.0f}Hz.png"),
        )

        rows.append({
            "f0_Hz": f0,
            "f_max_Hz": diag["f_max"],
            "lambda_min_m": round(diag["lambda_min"], 2),
            "PPW_at_fmax": round(diag["ppw"], 2),
            "PPW_at_f0": round(c_min / (f0 * DX), 2),
            "dispersion_free_PPW>=5": bool(diag["is_dispersion_free"]),
            "waveform_misfit_percent": round(100 * err, 2),
        })

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_snapshot_grid(
        panels, DX, DZ, ncols=len(SNAPSHOT_TIMES_MS),
        layer_depths=LAYER_DEPTHS, src_x=SRC_X, src_z=SRC_Z,
        suptitle="Wavefield snapshots versus source frequency "
                 "(h = 8 m, each panel scaled to its own maximum)",
        save_path=os.path.join(fig_dir, "freq_snapshots_grid.png"),
    )

    fig, axes = plt.subplots(len(F0_LIST), 1, figsize=(9, 2.3 * len(F0_LIST)),
                             dpi=150, sharex=True)
    for ax, f0 in zip(np.atleast_1d(axes), F0_LIST):
        t_ms, coarse, ref, err = traces[f0]
        scale = np.abs(ref).max() or 1.0
        ax.plot(t_ms, ref / scale, "k-", linewidth=1.6, label="reference (h = 8/3 m)")
        ax.plot(t_ms, coarse / scale, "r--", linewidth=1.3, label="h = 8 m")
        ax.set_ylabel("norm. p", fontsize=8)
        ax.set_title(f"f0 = {f0:.0f} Hz | misfit = {100 * err:.1f} %", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel(f"Time (ms), receiver at ~{TRACE_OFFSET_M:.0f} m offset")
    fig.suptitle("Effect of source frequency on the recorded trace", fontsize=11, fontweight="bold")
    fig.tight_layout()
    tr_path = os.path.join(fig_dir, "freq_trace_comparison.png")
    fig.savefig(tr_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Trace comparison saved to: {tr_path}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ppw = [r["PPW_at_fmax"] for r in rows]
    mis = [r["waveform_misfit_percent"] for r in rows]
    ax.plot(ppw, mis, "o-", color="tab:red")
    for r in rows:
        ax.annotate(f"{r['f0_Hz']:.0f} Hz", (r["PPW_at_fmax"], r["waveform_misfit_percent"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.axvline(5.0, color="k", linestyle=":", label="PPW = 5 (4th-order criterion)")
    ax.set_xlabel("Points per shortest wavelength, G = c_min / (2.5 f0 h)")
    ax.set_ylabel("Waveform misfit vs reference (%)")
    ax.set_title("Grid dispersion error grows as G falls")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ep = os.path.join(fig_dir, "freq_error_vs_ppw.png")
    fig.savefig(ep, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Error-vs-PPW plot saved to: {ep}")

    csv_path = os.path.join(out_dir, "frequency_sensitivity_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] Summary table saved to: {csv_path}")

    print("\n" + "=" * 72)
    print(f"{'f0 (Hz)':>8} {'PPW(fmax)':>10} {'PPW>=5?':>8} {'misfit %':>9}")
    for r in rows:
        print(f"{r['f0_Hz']:>8.0f} {r['PPW_at_fmax']:>10.2f} "
              f"{'yes' if r['dispersion_free_PPW>=5'] else 'NO':>8} "
              f"{r['waveform_misfit_percent']:>9.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()