#!/usr/bin/env python3
"""Diffusive-onset time tau_diff and diffusion coefficient D, mapped on the
isochore / isotherm planes.

For each per-step Si MSD file (msd_<T>K.dat: TimeStep  MSD[A^2]) the local
log-log slope beta(t) = d ln(MSD)/d ln(t) is computed.  The system is diffusive
where beta ~ 1.  tau_diff is the earliest time after which beta stays ~1 to the
end of the run (entry into the diffusive regime, i.e. structural relaxation /
end of caging).  D is the long-time slope, MSD = 6 D t.

tau_diff diverges on cooling / on approaching an arrest or a critical slowdown,
so mapping log(tau_diff) on the P-T isochores and P-rho isotherms shows where
the dynamics freeze relative to the (sought) LLCP.

NOTE: the MSD is the standard single-origin MSD (`compute msd ... com yes`, default
`average no`, dumped every step). At high T the liquid has no caging plateau, so the
reported tau_diff there is the ballistic->diffusive knee (~0.03-0.05 ps), not a
structural-relaxation time -- treat the high-T end qualitatively. A genuine caging
dip (beta < BETA_LO) appears only on cooling.

Outputs: diffusion_map.csv and PNGs in <sample>/diffusion/.

usage: ./toolkit/diffusion_map.py [sample-dir ...]   (default: all sample-*/)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

DT_PS = 0.0016                      # ps/step
BETA_LO, BETA_HI = 0.85, 1.15       # diffusive band for beta
NQ = 400                            # log-spaced resample points

# Glass-transition locus Tg(P) for this silica system (Tg in K at P in GPa).
# Non-monotonic: a pressure-induced Tg minimum near 20 GPa (mirrors the silica
# diffusivity maximum / melting-curve anomaly). State points BELOW this line in
# the P-T plane are glassy / cannot be equilibrated.
TG_P_GPA = (0, 5, 10, 20, 30)
TG_K = (2280, 1980, 1650, 1637, 1800)


def overlay_tg(ax):
    """Draw Tg(P) on a (T on x, P on y) axis; left of it (lower T) = glassy."""
    ax.plot(TG_K, TG_P_GPA, "r--o", ms=4, lw=1.3, zorder=4, label=r"$T_g(P)$")
    ax.legend(fontsize=7, loc="lower right")


def load_cavitated(sample_dir, thr=1.0):
    """Set of (round(rho,2), int(T)) flagged as cavitated (holes) in
    structure/void_scan.csv (empty-cell fraction > thr %). Empty if absent."""
    import csv
    f = os.path.join(sample_dir, "structure", "void_scan.csv")
    cav = set()
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            if float(r["empty_pct_mean"]) > thr:
                cav.add((round(float(r["rho"]), 2), int(float(r["T"]))))
    return cav


def tau_and_D(msd_file):
    """Return (tau_diff_ps, D_cm2s, msd_end) from a per-step MSD file."""
    df = pd.read_csv(msd_file, sep=r"\s+", comment="#", header=None,
                     usecols=[0, 1]).to_numpy()
    step = df[:, 0] - df[0, 0]
    t = step * DT_PS                                 # ps
    m = df[:, 1]
    g = (t > 0) & (m > 0)
    t, m = t[g], m[g]
    if t.size < 20:
        return np.nan, np.nan, (m[-1] if m.size else np.nan)
    lt, lm = np.log(t), np.log(m)
    ltq = np.linspace(lt.min(), lt.max(), NQ)
    lmq = uniform_filter1d(np.interp(ltq, lt, lm), 5, mode="nearest")
    beta = np.gradient(lmq, ltq)
    # diffusive onset = first time beta enters [BETA_LO,BETA_HI] AND the mean
    # slope from there to the (edge-trimmed) end is ~1, i.e. it STAYS diffusive.
    # This skips the ballistic descent (beta:2->1) and the caging dip and picks
    # the genuine entry into the long-time linear regime.
    trim = max(beta.size - 3, 1)            # drop noisy one-sided endpoint
    lo = max(int(0.02 * beta.size), 2)      # skip the first ballistic points
    onset = None
    for k in range(lo, trim):
        if BETA_LO <= beta[k] <= BETA_HI and 0.9 <= beta[k:trim].mean() <= 1.1:
            onset = k
            break
    tau = float(np.exp(ltq[onset])) if onset is not None else np.nan
    # D from the diffusive tail: MSD = 6 D t
    tail = t >= (tau if np.isfinite(tau) else t[int(0.7 * t.size)])
    if tail.sum() >= 5:
        slope = np.polyfit(t[tail], m[tail], 1)[0]   # A^2/ps
        D = slope / 6.0 * 1e-4                        # -> cm^2/s
    else:
        D = np.nan
    return tau, D, float(m[-1])


def load_PT(sample_dir):
    """rho,T -> P[GPa] from isochore-PT.dat (for placing the map points)."""
    P = {}
    for f in sorted(glob.glob(os.path.join(sample_dir, "rho-*", "isochore-PT.dat"))):
        if "_test" in f:
            continue
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            P[(round(float(c[3]), 2), int(float(c[0])))] = float(c[2]) * 1e-4
    return P


def analyse(sample_dir):
    PT = load_PT(sample_dir)
    rows = []                       # (rho, T, P_GPa, tau_ps, D, msd_end)
    for f in sorted(glob.glob(os.path.join(sample_dir, "rho-*", "msd_*K.dat"))):
        if "_test" in f:
            continue
        lab = os.path.basename(os.path.dirname(f))
        rho = int(lab.replace("rho-", "")) / 100.0
        T = int(os.path.basename(f).replace("msd_", "").replace("K.dat", ""))
        tau, D, mend = tau_and_D(f)
        P = PT.get((round(rho, 2), T), np.nan)
        rows.append((rho, T, P, tau, D, mend))
    rows.sort()
    return rows


def write_and_plot(rows, sample_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(sample_dir, "diffusion")
    os.makedirs(out, exist_ok=True)
    arr = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows], float)
    rho, T, P, tau, D = arr.T

    # CSV
    with open(os.path.join(out, "diffusion_map.csv"), "w") as fh:
        fh.write("rho,T,P_GPa,tau_diff_ps,D_cm2_s,msd_end_A2\n")
        for r in rows:
            fh.write("%.2f,%d,%.4f,%.6g,%.6g,%.4g\n" % r)

    cmap = plt.cm.viridis
    # exclude cavitated (holes) state points from the plots only; the CSV above
    # keeps every point. These are phase-separated (liquid-vapor), so their P is
    # not a single-phase value and they clutter the maps.
    cav = load_cavitated(sample_dir)
    keep = ~np.array([(round(rr, 2), int(tt)) in cav for rr, tt in zip(rho, T)])
    rho, T, P, tau, D = rho[keep], T[keep], P[keep], tau[keep], D[keep]
    rhos = sorted(set(rho))
    logtau = np.log10(np.where(np.isfinite(tau) & (tau > 0), tau, np.nan))

    # 1) tau_diff(T) per isochore (Arrhenius axis) + 2) D(T)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    _vir = plt.cm.viridis(np.linspace(0, 1, len(rhos)))
    ax[0].set_prop_cycle(color=_vir); ax[1].set_prop_cycle(color=_vir)
    for rr in rhos:
        m = (rho == rr) & np.isfinite(tau)
        if m.sum() >= 2:
            o = np.argsort(T[m])
            ax[0].semilogy(1000.0 / T[m][o], tau[m][o], "-o", ms=3, lw=0.8,
                           label=f"{rr:.2f}")
        m = (rho == rr) & np.isfinite(D)
        if m.sum() >= 2:
            o = np.argsort(T[m])
            ax[1].semilogy(1000.0 / T[m][o], D[m][o], "-o", ms=3, lw=0.8,
                           label=f"{rr:.2f}")
    ax[0].set_xlabel("1000/T (1/K)"); ax[0].set_ylabel(r"$\tau_{diff}$ (ps)")
    ax[0].set_title("Diffusive-onset time"); ax[0].legend(fontsize=5, ncol=2,
                                                          title=r"$\rho$")
    ax[1].set_xlabel("1000/T (1/K)"); ax[1].set_ylabel(r"$D$ (cm$^2$/s)")
    ax[1].set_title("Si diffusion coefficient")
    fig.tight_layout(); fig.savefig(f"{out}/tau_D_vs_T.png", dpi=130)
    plt.close(fig)

    # 3) P-T isochores coloured by log tau_diff ; 4) P-rho isotherms
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for rr in rhos:                                   # isochore lines
        m = rho == rr
        o = np.argsort(T[m])
        ax[0].plot(T[m][o], P[m][o], "-", color="0.8", lw=0.6, zorder=1)
    sc = ax[0].scatter(T, P, c=logtau, cmap=cmap, s=28, zorder=2)
    ax[0].set_xlabel("T (K)"); ax[0].set_ylabel("P (GPa)")
    ax[0].set_title("Isochores, coloured by log$_{10}\\,\\tau_{diff}$[ps]")
    overlay_tg(ax[0])               # glass-transition line; below it = glassy
    # focus y on the data; Tg(P) is only a guide for the eye, let it clip
    pad = 0.05 * (np.nanmax(P) - np.nanmin(P))
    ax[0].set_ylim(np.nanmin(P) - pad, np.nanmax(P) + pad)
    fig.colorbar(sc, ax=ax[0], label=r"log$_{10}\,\tau_{diff}$ (ps)")
    for TT in sorted(set(T)):                          # isotherm lines
        m = T == TT
        o = np.argsort(rho[m])
        ax[1].plot(rho[m][o], P[m][o], "-", color="0.8", lw=0.6, zorder=1)
    sc = ax[1].scatter(rho, P, c=logtau, cmap=cmap, s=28, zorder=2)
    ax[1].set_xlabel(r"$\rho$ (g/cc)"); ax[1].set_ylabel("P (GPa)")
    ax[1].set_title("Isotherms, coloured by log$_{10}\\,\\tau_{diff}$[ps]")
    fig.colorbar(sc, ax=ax[1], label=r"log$_{10}\,\tau_{diff}$ (ps)")
    fig.tight_layout(); fig.savefig(f"{out}/tau_map_PT_Prho.png", dpi=130)
    plt.close(fig)

    # 5) T(P) variant: same isochore data, axes swapped (T on y, P on x)
    fig, ax = plt.subplots(figsize=(7, 5))
    for rr in rhos:
        m = rho == rr
        o = np.argsort(T[m])                          # follow the T path
        ax.plot(P[m][o], T[m][o], "-", color="0.8", lw=0.6, zorder=1)
    sc = ax.scatter(P, T, c=logtau, cmap=cmap, s=28, zorder=2)
    ax.plot(TG_P_GPA, TG_K, "r--o", ms=4, lw=1.3, zorder=4, label=r"$T_g(P)$")
    ax.legend(fontsize=7, loc="best")
    ax.set_xlabel("P (GPa)"); ax.set_ylabel("T (K)")
    ax.set_title("Isochores T(P), coloured by log$_{10}\\,\\tau_{diff}$[ps]")
    pad = 0.05 * (np.nanmax(P) - np.nanmin(P))
    ax.set_xlim(np.nanmin(P) - pad, np.nanmax(P) + pad)
    fig.colorbar(sc, ax=ax, label=r"log$_{10}\,\tau_{diff}$ (ps)")
    fig.tight_layout(); fig.savefig(f"{out}/tau_map_TP.png", dpi=130)
    plt.close(fig)
    print(f"# wrote {out}/diffusion_map.csv , tau_D_vs_T.png , "
          f"tau_map_PT_Prho.png , tau_map_TP.png")


def main(argv):
    sdirs = argv[1:] or sorted(glob.glob("sample-*"))
    for sd in [s for s in sdirs if os.path.isdir(s)]:
        rows = analyse(sd)
        if not rows:
            print(f"# {sd}: no msd_*K.dat found"); continue
        print(f"# {sd}: tau_diff (ps) / D (cm^2/s) per (rho,T)")
        print(f"{'rho':>5} {'T':>6} {'P(GPa)':>8} {'tau_diff(ps)':>13} "
              f"{'D(cm2/s)':>11}")
        for rho, T, P, tau, D, _ in rows:
            ts = f"{tau:13.4g}" if np.isfinite(tau) else f"{'arrested':>13}"
            Ds = f"{D:11.4g}" if np.isfinite(D) else f"{'-':>11}"
            print(f"{rho:>5.2f} {T:>6d} {P:>8.3f} {ts} {Ds}")
        write_and_plot(rows, sd)


if __name__ == "__main__":
    main(sys.argv)
