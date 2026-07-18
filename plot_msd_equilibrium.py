#!/usr/bin/env python3
"""Plot Si MSD(t) of the 01-equilibrium runs, all temperatures overlaid.

Reads sample-N/<T>K/01-equilibrium/msd_<T>K.dat (fix ave/time scalar file:
TimeStep  c_msd1[4], every step, 1.6 fs) and draws one log-log MSD-vs-t figure
with t^2 (ballistic) and t^1 (diffusive) guides.

Examples:
    ./toolkit/plot_msd_equilibrium.py                 # sample-1 -> sample-1/msd_equilibrium.png
    ./toolkit/plot_msd_equilibrium.py sample-1 --out msd.png
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

DT_PS = 0.0016     # timestep (ps)
NPLOT = 1500       # log-spaced points kept per curve (files are 3.1M rows)


def load_msd(path):
    """Return (t_ps, msd_A2), log-decimated to ~NPLOT points."""
    d = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                    usecols=[0, 1]).to_numpy()
    t = (d[:, 0] - d[0, 0]) * DT_PS
    m = d[:, 1]
    g = (t > 0) & (m > 0)
    t, m = t[g], m[g]
    if t.size > NPLOT:  # ponytail: nearest-index log resample, no interpolation
        idx = np.unique(np.geomspace(1, t.size - 1, NPLOT).astype(int))
        t, m = t[idx], m[idx]
    return t, m


def discover(sample_dir):
    """{T: msd file} from <sample>/<T>K/01-equilibrium/msd_<T>K.dat."""
    out = {}
    for f in glob.glob(os.path.join(sample_dir, "*K", "01-equilibrium", "msd_*K.dat")):
        m = re.search(r"msd_(\d+)K\.dat$", f)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def main(argv=None):
    global DT_PS
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample", nargs="?", default="sample-1",
                   help="sample directory (default: sample-1)")
    p.add_argument("--out", help="output PNG (default: <sample>/msd_equilibrium.png)")
    p.add_argument("--dt", type=float, default=DT_PS,
                   help=f"timestep in ps (default {DT_PS})")
    args = p.parse_args(argv)
    DT_PS = args.dt

    files = discover(args.sample)
    if not files:
        raise SystemExit(f"no <T>K/01-equilibrium/msd_<T>K.dat under {args.sample}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    hot = None                                 # hottest curve anchors the guides
    for (T, f), c in zip(files.items(), colors):
        t, msd = load_msd(f)
        ax.loglog(t, msd, lw=1.2, color=c, label=f"{T} K")
        hot = (t, msd)
        print(f"{T:>5d} K  t_end={t[-1]:8.1f} ps  MSD_end={msd[-1]:10.3g} A^2")

    t, msd = hot
    tb = np.geomspace(t[0], t[0] * 300, 50)    # ballistic guide through 1st point
    ax.loglog(tb, msd[0] * (tb / t[0]) ** 2, "k--", lw=0.7, label=r"$t^2$ ballistic")
    td = np.geomspace(t[-1] / 1e3, t[-1], 50)  # diffusive guide through last point
    ax.loglog(td, msd[-1] * (td / t[-1]), "r--", lw=0.7, label=r"$t^1$ diffusive")

    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"Si MSD ($\AA^2$)")
    ax.set_title(f"Si MSD, 01-equilibrium (NPT, P=0)  {os.path.basename(args.sample)}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    out = args.out or os.path.join(args.sample, "msd_equilibrium.png")
    fig.savefig(out, dpi=130)
    print(f"# wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
