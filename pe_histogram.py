#!/usr/bin/env python3
"""Potential-energy histograms per temperature -- one-state vs two-state test.

A single homogeneous liquid gives a UNIMODAL (near-Gaussian) P(PE) distribution.
Near a liquid-liquid critical point the system fluctuates between two structural
states (HDL/LDL), so P(PE) broadens and can become BIMODAL (two peaks) -- the
energy-distribution signature of the LLCP / two-state behaviour.

Reads the per-step PE series written by the 01-equilibrium runs (fix ave/time
scalar file: TimeStep  v_PEv, every step, 1.6 fs):

    sample-N/<T>K/01-equilibrium/pe_<T>K.dat

For each temperature this plots the PE histogram (centred on <PE>), overlays a
Gaussian of the same mean/width (so any departure from one-state behaviour
stands out), prints the fluctuation sigma(PE) and a bimodality coefficient
BC = (skew^2 + 1)/kurtosis ; BC > 0.555 (uniform limit) flags a candidate
two-state / bimodal distribution.

usage: ./toolkit/pe_histogram.py                      # sample-1, all T
       ./toolkit/pe_histogram.py sample-1 --per-atom
       ./toolkit/pe_histogram.py sample-1/300K/01-equilibrium   # single dir
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

NATOMS = 8064


def bimodality_coeff(x):
    """BC = (skew^2 + 1)/kurtosis (Pearson, non-excess). >0.555 => candidate."""
    g = skew(x)
    k = kurtosis(x, fisher=True) + 3.0          # non-excess kurtosis
    return (g * g + 1.0) / k if k > 0 else float("nan")


def discover(path):
    """{T: pe file}. `path` = sample root (walk <T>K/01-equilibrium) or a
    directory holding pe_<T>K.dat files directly."""
    hits = glob.glob(os.path.join(path, "*K", "01-equilibrium", "pe_*K.dat")) \
        or glob.glob(os.path.join(path, "pe_*K.dat"))
    out = {}
    for f in hits:
        m = re.search(r"pe_(\d+)K\.dat$", f)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items(), reverse=True))


def load_pe(path):
    """Per-step PE series (eV) from a fix ave/time scalar file."""
    return pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                       usecols=[1]).to_numpy().ravel()


def analyse(path, per_atom, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = discover(path)
    if not files:
        print(f"# {path}: no pe_<T>K.dat found -- skipped"); return

    n = len(files)
    nc = min(3, n); nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.8 * nr), squeeze=False)
    label = os.path.basename(os.path.abspath(path))
    unit = "eV/atom" if per_atom else "eV"
    print(f"# {label}: PE distribution per T  (sigma = fluctuation, "
          f"BC>0.555 = bimodal candidate)")
    print(f"{'T(K)':>6} {'<PE>('+unit+')':>14} {'sigma':>9} {'skew':>7} "
          f"{'BC':>6} {'Nsamp':>9}  flag")
    for i, (T, pef) in enumerate(files.items()):
        ax = axes[i // nc][i % nc]
        pe = load_pe(pef) / (NATOMS if per_atom else 1.0)
        mu = pe.mean(); sig = pe.std()
        d = pe - mu
        BC = bimodality_coeff(pe)
        nb = bins if isinstance(bins, int) else \
            int(np.clip(np.sqrt(pe.size), 50, 400))   # 'auto': scale with Nsamp
        # PE is quantized (6 sig figs in the ave/time file, ~0.1 eV at -78000).
        # Bins narrower than / unaligned with that grid grow comb artifacts, so
        # snap the bin edges to the grid with an integer-quantum bin width.
        u = np.unique(d)
        res = np.diff(u).min() if u.size > 1 else 0.0
        if res > 0:
            k = max(int(np.ceil((d.max() - d.min()) / nb / res)), 1)
            nb = np.arange(d.min() - res / 2, d.max() + k * res, k * res)
        ax.hist(d, bins=nb, density=True, color="C0", alpha=0.7)
        if sig > 0:                              # same-width Gaussian reference
            xs = np.linspace(d.min(), d.max(), 200)
            ax.plot(xs, np.exp(-xs**2 / (2 * sig**2)) / (sig * np.sqrt(2*np.pi)),
                    "r-", lw=1.0)
        flag = "<- bimodal?" if (not np.isnan(BC) and BC > 0.555) else ""
        ax.set_title(f"T={T}K  BC={BC:.2f}{(' '+flag) if flag else ''}",
                     fontsize=8)
        ax.set_xlabel(f"PE-<PE> ({unit})", fontsize=7)
        ax.tick_params(labelsize=6)
        print(f"{T:>6d} {mu:>14.4f} {sig:>9.4f} {skew(pe):>7.3f} "
              f"{BC:>6.3f} {pe.size:>9d}  {flag}")
    for j in range(n, nr * nc):
        axes[j // nc][j % nc].axis("off")
    fig.suptitle(f"PE distributions  {label}  (red = same-sigma Gaussian)")
    fig.tight_layout()
    out = os.path.join(os.path.abspath(path), "pe_hist.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"# plot -> {out}")


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*", default=["sample-1"],
                   help="sample root(s) or dir(s) holding pe_<T>K.dat "
                        "(default: sample-1)")
    p.add_argument("--per-atom", action="store_true",
                   help="plot PE per atom instead of total")
    p.add_argument("--bins", default="auto",
                   help="histogram bins: integer, or 'auto' = sqrt(Nsamp) "
                        "clamped [50,400] (default auto)")
    a = p.parse_args(argv)
    bins = int(a.bins) if a.bins.isdigit() else "auto"
    for d in a.dirs:
        analyse(d, a.per_atom, bins)


if __name__ == "__main__":
    main(sys.argv[1:])
