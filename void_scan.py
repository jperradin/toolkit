#!/usr/bin/env python3
"""Scan every (rho,T) isochore point for cavitation (voids/holes).

For each pos_<T>K.data, bin the last NFRAMES frames into ~CELL-A cubic cells and
measure density inhomogeneity:
  - empty% : fraction of cells with zero atoms (a void larger than one cell)
  - infl   : occupancy std / sqrt(mean). A homogeneous (incompressible) liquid is
             SUB-Poisson (infl < 1); voids/phase separation make it SUPER-Poisson
             (infl > 1) with empty cells appearing.
A point is flagged CAVITATED when mean empty% over the frames exceeds EMPTY_THR.

Writes structure/void_scan.csv and prints the flagged (rho,T) set.

usage: ./toolkit/void_scan.py [sample-dir]   (default sample-1)
"""
import glob
import os
import sys

import numpy as np

NAT = 8064
NFRAMES = 5
CELL = 5.0          # target cell edge (A)
EMPTY_THR = 1.0     # % empty cells to flag as cavitated (controls ~0%)


def box_L(rho_dir):
    for bf in sorted(glob.glob(os.path.join(rho_dir, "box_*.data"))):
        for ln in open(bf):
            if "xlo xhi" in ln:
                a, b = ln.split()[:2]
                return float(b) - float(a)
    raise RuntimeError(f"no box in {rho_dir}")


def frames_xyz(pos, nframes):
    lines = open(pos).read().split("\n")
    fsz = NAT + 2
    nfit = len(lines) // fsz
    for fi in range(max(0, nfit - nframes), nfit):
        blk = lines[fi * fsz + 2: fi * fsz + 2 + NAT]
        xyz = np.array([[float(v) for v in ln.split()[1:4]]
                        for ln in blk if len(ln.split()) >= 4])
        if xyz.shape[0] == NAT:
            yield xyz


def void_metrics(pos, L):
    n = max(int(L // CELL), 6)
    emptys, infls = [], []
    for xyz in frames_xyz(pos, NFRAMES):
        idx = np.clip((np.mod(xyz, L) / (L / n)).astype(int), 0, n - 1)
        occ = np.zeros((n, n, n), int)
        np.add.at(occ, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
        occ = occ.ravel().astype(float)
        emptys.append((occ == 0).mean() * 100.0)
        infls.append(occ.std() / np.sqrt(occ.mean()))
    return (float(np.mean(emptys)), float(np.max(emptys)),
            float(np.mean(infls)), n)


def main(argv):
    sd = argv[1] if len(argv) > 1 else "sample-1"
    rows = []
    for rd in sorted(glob.glob(os.path.join(sd, "rho-*"))):
        rho = int(os.path.basename(rd).replace("rho-", "")) / 100.0
        L = box_L(rd)
        # P lookup
        Pm = {}
        pf = os.path.join(rd, "isochore-PT.dat")
        if os.path.exists(pf):
            for ln in open(pf):
                if not ln.startswith("#") and ln.strip():
                    c = ln.split(); Pm[int(float(c[0]))] = float(c[2]) * 1e-4
        for pos in sorted(glob.glob(os.path.join(rd, "pos_*K.data"))):
            T = int(os.path.basename(pos).replace("pos_", "").replace("K.data", ""))
            em, emx, infl, n = void_metrics(pos, L)
            rows.append((rho, T, Pm.get(T, float("nan")), em, emx, infl, n))

    out = os.path.join(sd, "structure", "void_scan.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("rho,T,P_GPa,empty_pct_mean,empty_pct_max,inflation,ncells\n")
        for r in rows:
            fh.write("%.2f,%d,%.4f,%.3f,%.3f,%.3f,%d\n" % r)

    flagged = [r for r in rows if r[3] > EMPTY_THR]
    flagged.sort(key=lambda r: -r[3])
    print(f"# void scan: {len(rows)} points, CELL~{CELL}A, last {NFRAMES} frames, "
          f"flag empty%>{EMPTY_THR}")
    print(f"# CAVITATED (holes): {len(flagged)} state points")
    print(f"{'rho':>5}{'T':>6}{'P_GPa':>8}{'empty%':>8}{'emptymax%':>10}{'inflation':>10}")
    for rho, T, P, em, emx, infl, n in flagged:
        print(f"{rho:>5.2f}{T:>6d}{P:>8.3f}{em:>8.2f}{emx:>10.2f}{infl:>10.2f}")
    print(f"# full table -> {out}")


if __name__ == "__main__":
    main(sys.argv)
