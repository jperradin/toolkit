#!/usr/bin/env python3
"""Aggregate the per-(rho,T) rreve structural outputs into a grid summary +
diagnostic maps, for the cavitation-vs-LLCP structural question.

Reads <sample>/structure/<rho-XXX>/<T>K/*.dat (written by structure_analysis.py)
and <sample>/rho-XXX/isochore-PT.dat (for P), extracts per state point:
  - mean O-Si-O angle (intra-tetra, ideal 109.47) and Si-O-Si angle (linkage)
  - SiO4 / SiO5 / SiO6 fractions (coordination state)
  - tetrahedricity 4-fold proportion
  - corner / edge / face sharing connectivity
  - Si-O bond length (O-Si g(r) first-peak position)
Writes structure/structure_summary.csv and structure/maps_structure.png +
structure/gr_bad_overlays.png.

usage: ./toolkit/structure_aggregate.py [sample-dir]   (default sample-1)
"""
import glob
import os
import sys

import numpy as np

GPA = 1e-4


def _row(path):
    """First non-comment numeric row as float list."""
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        return [float(x) for x in ln.split()]
    return []


def named(path):
    """Single-row .dat -> {column-name: value}, using the '#'-comment header.
    Header forms: "# ['SiO3', 'SiO4', ...]" or "# corner-sharing edge-sharing ...".
    Robust to a data row with fewer columns than the header (missing -> absent)."""
    hdr, vals = None, None
    for ln in open(path):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            h = s.lstrip("#").strip().strip("[]")
            hdr = [t.strip().strip("'\"") for t in
                   (h.split(",") if "," in h else h.split())]
        else:
            vals = [float(x) for x in s.split()]
            break
    if hdr is None or vals is None:
        return {}
    return {k: v for k, v in zip(hdr, vals)}


def load_P(sd):
    P = {}
    for f in glob.glob(os.path.join(sd, "rho-*", "isochore-PT.dat")):
        for ln in open(f):
            if ln.startswith("#") or not ln.strip():
                continue
            c = ln.split()
            P[(round(float(c[3]), 2), int(float(c[0])))] = float(c[2]) * GPA
    return P


def gr_first_peak(pdf_file, col=1, rmax=2.3):
    """Position of the g(r) first peak in column `col` (O-Si) within r<rmax."""
    d = np.loadtxt(pdf_file, comments="#")
    r, g = d[:, 0], d[:, col]
    m = r < rmax
    if not m.any():
        return np.nan
    return float(r[m][np.argmax(g[m])])


def collect(sd):
    Pmap = load_P(sd)
    rows = []
    for td in sorted(glob.glob(os.path.join(sd, "structure", "rho-*", "*K"))):
        rho = int(os.path.basename(os.path.dirname(td)).replace("rho-", "")) / 100.0
        T = int(os.path.basename(td).replace("K", ""))
        try:
            ang = _row(os.path.join(td, "mean_angles.dat"))      # OOO OSiO SiOSi SiSiSi
            sio = named(os.path.join(td, "structural_units-SiO.dat"))
            tet = _row(os.path.join(td, "tetrahedricity_proportions.dat"))  # 4-fold
            con = named(os.path.join(td, "connectivities.dat"))
            rSiO = gr_first_peak(os.path.join(td, "pair_distribution_function.dat"))
        except (OSError, IndexError):
            continue
        P = Pmap.get((round(rho, 2), T), np.nan)
        rows.append(dict(rho=rho, T=T, P=P,
                         oSiO=ang[1], SiOSi=ang[2],
                         fSiO3=sio.get("SiO3", 0.0), fSiO4=sio.get("SiO4", 0.0),
                         fSiO5=sio.get("SiO5", 0.0), fSiO6=sio.get("SiO6", 0.0),
                         tetra4=tet[0] if tet else np.nan,
                         corner=con.get("corner-sharing", 0.0),
                         edge=con.get("edge-sharing", 0.0),
                         face=con.get("face-sharing", 0.0),
                         rSiO=rSiO))
    return rows


def write_csv(rows, out):
    keys = ["rho", "T", "P", "oSiO", "SiOSi", "fSiO3", "fSiO4", "fSiO5", "fSiO6",
            "tetra4", "corner", "edge", "face", "rSiO"]
    with open(out, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in sorted(rows, key=lambda x: (x["rho"], x["T"])):
            fh.write(",".join(f"{r[k]:.5g}" for k in keys) + "\n")


def maps(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho = np.array([r["rho"] for r in rows])
    T = np.array([r["T"] for r in rows])
    P = np.array([r["P"] for r in rows])
    over = np.array([r["fSiO5"] + r["fSiO6"] for r in rows])   # overcoordination
    edgeface = np.array([r["edge"] + r["face"] for r in rows])

    panels = [("O-Si-O angle (deg)", np.array([r["oSiO"] for r in rows]), "viridis"),
              ("Si-O-Si angle (deg)", np.array([r["SiOSi"] for r in rows]), "viridis"),
              ("SiO5+SiO6 fraction (overcoord.)", over, "magma"),
              ("edge+face sharing / Si", edgeface, "magma")]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for a, (title, c, cm) in zip(ax.ravel(), panels):
        sc = a.scatter(rho, T, c=c, cmap=cm, s=42)
        a.set_xlabel(r"$\rho$ (g/cc)"); a.set_ylabel("T (K)"); a.set_title(title)
        fig.colorbar(sc, ax=a)
    fig.suptitle("Structure across the isochore grid (rreve)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def overlays(sd, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pdf(rho_tag, T, col=1):
        f = os.path.join(sd, "structure", rho_tag, f"{T}K",
                         "pair_distribution_function.dat")
        d = np.loadtxt(f, comments="#")
        return d[:, 0], d[:, col]

    def bad(rho_tag, T, col=2):   # O-Si-O column
        f = os.path.join(sd, "structure", rho_tag, f"{T}K",
                         "bond_angular_distribution.dat")
        d = np.loadtxt(f, comments="#")
        return d[:, 0], d[:, col]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # (a) O-Si g(r) across rho at fixed T=2500K
    T0 = 2500
    rhos = ["rho-190", "rho-210", "rho-240", "rho-265", "rho-290", "rho-310"]
    vir = plt.cm.viridis(np.linspace(0, 1, len(rhos)))
    for rt, col in zip(rhos, vir):
        try:
            r, g = pdf(rt, T0)
            ax[0, 0].plot(r, g, lw=1, color=col, label=rt.replace("rho-", "ρ"))
        except OSError:
            pass
    ax[0, 0].set_xlim(0, 6); ax[0, 0].set_xlabel("r (Å)"); ax[0, 0].set_ylabel("g(r) O–Si")
    ax[0, 0].set_title(f"O–Si g(r) vs ρ at {T0} K"); ax[0, 0].legend(fontsize=7)
    # (b) O-Si-O angle across rho at fixed T
    for rt, col in zip(rhos, vir):
        try:
            a, p = bad(rt, T0)
            ax[0, 1].plot(a, p, lw=1, color=col, label=rt.replace("rho-", "ρ"))
        except OSError:
            pass
    ax[0, 1].axvline(109.47, color="k", ls=":", lw=0.8)
    ax[0, 1].set_xlim(60, 180); ax[0, 1].set_xlabel("angle (deg)")
    ax[0, 1].set_title(f"O–Si–O angle vs ρ at {T0} K"); ax[0, 1].legend(fontsize=7)
    # (c) O-Si g(r) across T at the spinodal density rho-200
    rt = "rho-200"
    Ts = [4500, 3500, 3000, 2600, 2400]
    vir2 = plt.cm.plasma(np.linspace(0, 1, len(Ts)))
    for T, col in zip(Ts, vir2):
        try:
            r, g = pdf(rt, T)
            ax[1, 0].plot(r, g, lw=1, color=col, label=f"{T} K")
        except OSError:
            pass
    ax[1, 0].set_xlim(0, 6); ax[1, 0].set_xlabel("r (Å)"); ax[1, 0].set_ylabel("g(r) O–Si")
    ax[1, 0].set_title("O–Si g(r) vs T at ρ2.00 (spinodal)"); ax[1, 0].legend(fontsize=7)
    # (d) O-Si-O angle across T at rho-200
    for T, col in zip(Ts, vir2):
        try:
            a, p = bad(rt, T)
            ax[1, 1].plot(a, p, lw=1, color=col, label=f"{T} K")
        except OSError:
            pass
    ax[1, 1].axvline(109.47, color="k", ls=":", lw=0.8)
    ax[1, 1].set_xlim(60, 180); ax[1, 1].set_xlabel("angle (deg)")
    ax[1, 1].set_title("O–Si–O angle vs T at ρ2.00 (spinodal)"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main(argv):
    sd = (argv[1] if len(argv) > 1 else "sample-1")
    rows = collect(sd)
    if not rows:
        raise SystemExit("no structure outputs found")
    outdir = os.path.join(sd, "structure")
    write_csv(rows, os.path.join(outdir, "structure_summary.csv"))
    maps(rows, os.path.join(outdir, "maps_structure.png"))
    overlays(sd, os.path.join(outdir, "gr_bad_overlays.png"))
    print(f"# {len(rows)} state points -> {outdir}/structure_summary.csv ,"
          f" maps_structure.png , gr_bad_overlays.png")


if __name__ == "__main__":
    main(sys.argv)
