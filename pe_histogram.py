#!/usr/bin/env python3
"""Potential-energy histograms per state point -- one-state vs two-state test.

A single homogeneous liquid gives a UNIMODAL (near-Gaussian) P(PE) distribution.
Near a liquid-liquid critical point the system fluctuates between two structural
states (HDL/LDL), so P(PE) broadens and can become BIMODAL (two peaks) -- the
energy-distribution signature of the LLCP / two-state behaviour.

For each production block (per temperature) of an isochore this plots the PE
histogram (centred on <PE>), overlays a Gaussian of the same mean/width (so any
departure from one-state behaviour stands out), prints the fluctuation sigma(PE)
and a bimodality coefficient BC = (skew^2 + 1)/kurtosis ; BC > 0.555 (uniform
limit) flags a candidate two-state / bimodal distribution.

PE is read from the log thermo (PotEng, every 100 steps); use longer/colder runs
for cleaner statistics. NVT fixes V, so a true HDL/LDL density split cannot
occur -- a bimodal PE here means the whole box flips structural state (a strong
near-critical hint); pair with NPT density histograms for confirmation.

usage: ./toolkit/pe_histogram.py sample-1/rho-260 [sample-1/rho-275 ...]
       ./toolkit/pe_histogram.py sample-1/rho-260 --per-atom
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import skew, kurtosis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_equilibration import (resolve_logs, parse_log,      # noqa: E402
                                  pair_production, mean_sem)

NATOMS = 27216


def bimodality_coeff(x):
    """BC = (skew^2 + 1)/kurtosis (Pearson, non-excess). >0.555 => candidate."""
    g = skew(x)
    k = kurtosis(x, fisher=True) + 3.0          # non-excess kurtosis
    return (g * g + 1.0) / k if k > 0 else float("nan")


def analyse_dir(path, per_atom, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        logs = resolve_logs(path)
    except SystemExit:
        print(f"# {path}: no log (not run yet) -- skipped"); return
    prods = [b for b, _ in pair_production(parse_log(logs))]
    if not prods:
        print(f"# {path}: no production blocks -- skipped"); return
    prods = sorted(prods, key=lambda b: -b.tset)

    n = len(prods)
    nc = min(4, n); nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), squeeze=False)
    label = os.path.basename(os.path.abspath(path))
    unit = "eV/atom" if per_atom else "eV"
    print(f"# {label}: PE distribution per T  (sigma = fluctuation, "
          f"BC>0.555 = bimodal candidate)")
    print(f"{'T(K)':>6} {'<PE>('+unit+')':>14} {'sigma':>9} {'skew':>7} "
          f"{'BC':>6} {'Nsamp':>9} {'res':>8}  flag")
    for i, b in enumerate(prods):
        ax = axes[i // nc][i % nc]
        # prefer the per-step pe_<T>K.dat (1.6 fs) for fine resolution; else the
        # log thermo PE (every 100 steps).
        pef = os.path.join(os.path.abspath(path), f"pe_{int(b.tset)}K.dat")
        if os.path.isfile(pef):
            pe = np.loadtxt(pef, comments="#", usecols=1)
            res = "perstep"
        else:
            pe = b.pe
            res = "log/100"
        pe = pe / (NATOMS if per_atom else 1.0)
        mu = pe.mean(); sig = pe.std()
        d = pe - mu
        BC = bimodality_coeff(pe)
        nb = bins if isinstance(bins, int) else \
            int(np.clip(np.sqrt(pe.size), 50, 400))   # 'auto': scale with Nsamp
        ax.hist(d, bins=nb, density=True, color="C0", alpha=0.7)
        if sig > 0:                              # same-width Gaussian reference
            xs = np.linspace(d.min(), d.max(), 200)
            ax.plot(xs, np.exp(-xs**2 / (2 * sig**2)) / (sig * np.sqrt(2*np.pi)),
                    "r-", lw=1.0)
        flag = "<- bimodal?" if (not np.isnan(BC) and BC > 0.555) else ""
        ax.set_title(f"T={b.tset:.0f}K  BC={BC:.2f}{(' '+flag) if flag else ''}",
                     fontsize=8)
        ax.set_xlabel(f"PE-<PE> ({unit})", fontsize=7)
        ax.tick_params(labelsize=6)
        print(f"{b.tset:>6.0f} {mu:>14.4f} {sig:>9.4f} {skew(pe):>7.3f} "
              f"{BC:>6.3f} {pe.size:>9d} {res:>8}  {flag}")
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
    p.add_argument("dirs", nargs="+", help="rho-XXX director(ies)")
    p.add_argument("--per-atom", action="store_true",
                   help="plot PE per atom instead of total")
    p.add_argument("--bins", default="auto",
                   help="histogram bins: integer, or 'auto' = sqrt(Nsamp) "
                        "clamped [50,400] (default auto)")
    a = p.parse_args(argv)
    bins = int(a.bins) if a.bins.isdigit() else "auto"
    for d in a.dirs:
        analyse_dir(d, a.per_atom, bins)


if __name__ == "__main__":
    main(sys.argv[1:])
