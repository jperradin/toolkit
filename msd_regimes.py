#!/usr/bin/env python3
"""Si MSD regime analysis for an NVT isochore directory.

Reads the per-temperature MSD(t) time series written by the generated inputs
(build_tree.sh):

    msd_<T>K.dat    MSD every step (1.6 fs) over the whole production

a LAMMPS ``fix ave/time`` scalar file (TimeStep, c_msd1[4]) referenced to t=0 at
the start of production, covering all regimes in one trace.

On a log-log MSD-vs-t plot a liquid shows three regimes:

    ballistic   MSD ~ t^2    free flight before the first collision (sub-ps)
    caging      MSD ~ t^a, a<1   transient plateau; atom rattles in the cage
                             formed by its neighbours (beta relaxation). Deep /
                             long at low T, shallow or absent at high T.
    diffusive   MSD ~ t^1    cage breaks; Fickian diffusion. D = MSD/(6 t).

The local slope d(log MSD)/d(log t) is used to label the regimes and to read
off the self-diffusion coefficient D from the diffusive tail.

Examples:
    ./msd_regimes.py ../sample-1/rho-210
    ./msd_regimes.py ../sample-1/rho-210 --csv D_of_T.csv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

DT_PS = 0.0016                 # timestep (ps)
SLOPE_BALLISTIC = 1.6          # local slope above this => ballistic (~2)
SLOPE_DIFFUSIVE = (0.85, 1.2)  # local slope band counted as diffusive (~1)
SMOOTH = 5                     # log-slope smoothing window (points)

# A^2/ps -> cm^2/s : (1e-8 cm)^2 / (1e-12 s) = 1e-4
A2PS_TO_CM2S = 1.0e-4


def read_msd_file(path):
    """Read a LAMMPS fix ave/time scalar file -> (step, value) arrays."""
    step, val = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) >= 2:
                try:
                    step.append(float(p[0]))
                    val.append(float(p[1]))
                except ValueError:
                    continue
    return np.array(step), np.array(val)


def load_temperature(directory, T):
    """Read the MSD file for one temperature -> t(ps), MSD(A^2)."""
    path = os.path.join(directory, f"msd_{T}K.dat")
    if not os.path.isfile(path):
        return None, None
    step, msd = read_msd_file(path)
    if step.size == 0:
        return None, None
    order = np.argsort(step)
    step, msd = step[order], msd[order]
    t = step * DT_PS
    good = (t > 0) & (msd > 0)
    return t[good], msd[good]


def local_slope(t, msd):
    """Smoothed d(log MSD)/d(log t)."""
    lt, lm = np.log(t), np.log(msd)
    slope = np.gradient(lm, lt)
    if SMOOTH > 1 and slope.size >= SMOOTH:
        k = np.ones(SMOOTH) / SMOOTH
        slope = np.convolve(slope, k, mode="same")
    return slope


def analyze(t, msd):
    """Identify regimes and fit D from the diffusive tail. Returns a dict."""
    slope = local_slope(t, msd)

    ballistic = slope > SLOPE_BALLISTIC
    diffusive = (slope >= SLOPE_DIFFUSIVE[0]) & (slope <= SLOPE_DIFFUSIVE[1])

    # ballistic end: last point of the leading ballistic run
    t_ball = t[ballistic].max() if ballistic.any() else float("nan")

    # caging: minimum local slope between ballistic end and diffusive onset
    caging_slope = float(np.nanmin(slope)) if slope.size else float("nan")
    t_cage = float(t[np.nanargmin(slope)]) if slope.size else float("nan")

    # diffusive onset: first sustained diffusive point in the back half
    half = t.size // 2
    diff_tail = diffusive.copy()
    diff_tail[:half] = False
    t_diff = t[diff_tail].min() if diff_tail.any() else float("nan")

    # D from the diffusive tail: fit MSD = 6 D t (through the last decade in t)
    D = float("nan")
    mask = t >= (t.max() / 10.0)
    if mask.sum() >= 3:
        b = np.polyfit(t[mask], msd[mask], 1)[0]   # A^2/ps
        D = b / 6.0 * A2PS_TO_CM2S                  # cm^2/s
    diffusive_reached = np.isfinite(t_diff)

    return {
        "t_ball_ps": t_ball,
        "t_cage_ps": t_cage,
        "caging_slope": caging_slope,
        "t_diff_ps": t_diff,
        "diffusive": diffusive_reached,
        "D_cm2_s": D,
        "msd_end": float(msd[-1]),
        "t_end_ps": float(t[-1]),
    }


def discover_temperatures(directory):
    Ts = set()
    for path in glob.glob(os.path.join(directory, "msd_*K.dat")):
        m = re.search(r"msd_(\d+)K\.dat$", os.path.basename(path))
        if m:
            Ts.add(int(m.group(1)))
    return sorted(Ts, reverse=True)


def plot_one(t, msd, info, T, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(t, msd, lw=1.3, label=f"{T} K")
    # slope guides anchored to the data
    i0 = max(t.size // 50, 1)
    ax.loglog(t, msd[i0] * (t / t[i0]) ** 2, "k--", lw=0.7, label=r"$t^2$ ballistic")
    if np.isfinite(info["D_cm2_s"]):
        ax.loglog(t, msd[-1] * (t / t[-1]), "r--", lw=0.7, label=r"$t^1$ diffusive")
    for tb, c, lab in ((info["t_ball_ps"], "tab:blue", "ballistic end"),
                       (info["t_diff_ps"], "tab:red", "diffusive onset")):
        if np.isfinite(tb):
            ax.axvline(tb, color=c, ls=":", lw=0.8)
    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"Si MSD ($\AA^2$)")
    d = info["D_cm2_s"]
    dtxt = f"D = {d:.2e} cm$^2$/s" if np.isfinite(d) else "no diffusive regime"
    state = "diffusive" if info["diffusive"] else "ARRESTED (cage not broken)"
    ax.set_title(f"T = {T} K   {state}\n{dtxt}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"msd_{T}K.png"), dpi=120)
    plt.close(fig)


def plot_overlay(curves, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, len(curves))))
    for T, (t, msd) in sorted(curves.items()):
        ax.loglog(t, msd, lw=1.1, label=f"{T} K")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"Si MSD ($\AA^2$)")
    ax.set_title("Si MSD vs time, all temperatures")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "msd_all.png"), dpi=120)
    plt.close(fig)


_HDR = (f"{'T(K)':>6} {'t_ball/ps':>10} {'t_cage/ps':>10} {'min_slope':>10} "
        f"{'t_diff/ps':>10} {'D(cm2/s)':>11} {'regime':>22}")


def fmt(T, info):
    def g(x, w, p=2):
        return f"{'n/a':>{w}}" if not np.isfinite(x) else f"{x:>{w}.{p}g}"
    regime = "diffusive" if info["diffusive"] else "ARRESTED/caged"
    return (f"{T:>6d} {g(info['t_ball_ps'],10)} {g(info['t_cage_ps'],10)} "
            f"{g(info['caging_slope'],10)} {g(info['t_diff_ps'],10)} "
            f"{g(info['D_cm2_s'],11)} {regime:>22}")


def main(argv=None):
    global DT_PS
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", help="rho-XXX directory holding msd_<T>K_*.dat files")
    p.add_argument("--csv", help="write the per-T regime/D table to this CSV")
    p.add_argument("--no-plot", action="store_true", help="skip PNG output")
    p.add_argument("--dt", type=float, default=DT_PS, help=f"timestep ps (def {DT_PS})")
    args = p.parse_args(argv)
    DT_PS = args.dt

    Ts = discover_temperatures(args.directory)
    if not Ts:
        raise SystemExit(
            f"No msd_<T>K.dat files in {args.directory}. These are produced by "
            "runs generated with the updated build_tree.sh (compute msd average no "
            "+ per-step MSD logging). The rho-210_test run predates that and has none.")

    out_dir = os.path.join(args.directory, "msd_plots")
    print(_HDR)
    rows, curves = [], {}
    for T in Ts:
        t, msd = load_temperature(args.directory, T)
        if t is None or t.size < 4:
            print(f"{T:>6d}   (no/short data)")
            continue
        info = analyze(t, msd)
        print(fmt(T, info))
        rows.append((T, info))
        curves[T] = (t, msd)
        if not args.no_plot:
            plot_one(t, msd, info, T, out_dir)

    if curves and not args.no_plot:
        plot_overlay(curves, out_dir)
        print(f"# wrote plots -> {out_dir}")

    if args.csv and rows:
        keys = ["t_ball_ps", "t_cage_ps", "caging_slope", "t_diff_ps",
                "diffusive", "D_cm2_s", "msd_end", "t_end_ps"]
        with open(args.csv, "w") as f:
            f.write("T," + ",".join(keys) + "\n")
            for T, info in rows:
                f.write(f"{T}," + ",".join(
                    f"{info[k]:.6g}" if isinstance(info[k], float) else str(info[k])
                    for k in keys) + "\n")
        print(f"# wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
