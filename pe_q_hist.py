#!/usr/bin/env python3
"""2D histogram of PE vs tetrahedral order q for one compression point.

Aging vs two-state diagnostic (near-LLCP test): genuine two-state coexistence
gives a BIMODAL 2D distribution (two islands + saddle) in the (PE, q) plane;
aging gives a single ridge that drifts with time.

Per B-part frame of a compression point (pos<n>B.data, xyz dump, 251 frames
over 0.5 ns) this computes the Errington-Debenedetti tetrahedral order
    q_i = 1 - 3/8 * sum_{j<k} (cos theta_jik + 1/3)^2
over the 4 nearest O neighbours of every Si, averages it over the box, and
pairs it with PotEng and Lx read from the range log at the same timestep.

usage: ./toolkit/pe_q_hist.py sample-1/2000K/02-compression 8
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

NAT = 8064


def range_log(cp_dir, n):
    """Path of the 5 GPa range log holding point n, and its lstart."""
    k = 1 if n <= 21 else (n - 22) // 20 + 2
    lstart = 1 if k == 1 else 20 * (k - 1) + 2
    return os.path.join(cp_dir, f"log.lammps-{lstart}-{20 * k + 1}"), lstart


def b_block(log_file, nth):
    """{step: (PotEng, Lx)} of the nth (1-based) B block (first Step=312500)."""
    seen = 0
    cols = None
    out = {}
    with open(log_file) as fh:
        for line in fh:
            p = line.split()
            if p and p[0] == "Step":
                cols = p
                first = True
                continue
            if cols is None or len(p) != len(cols):
                continue
            try:
                row = [float(x) for x in p]
            except ValueError:
                continue
            if first:
                first = False
                if int(row[0]) == 312500:
                    seen += 1
                if seen != nth or int(row[0]) != 312500:
                    cols = None if seen >= nth else cols
                    if seen >= nth:
                        break
                    continue
            if seen == nth:
                out[int(row[0])] = (row[cols.index("PotEng")],
                                    row[cols.index("Lx")])
    return out


def frames(pos_file):
    """Yield (step, types, xyz) per frame of a LAMMPS xyz dump."""
    with open(pos_file) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            n = int(line)
            step = int(fh.readline().split()[-1])     # "Atoms. Timestep: N"
            typ = np.empty(n, int)
            pos = np.empty((n, 3))
            for k in range(n):
                p = fh.readline().split()
                typ[k] = int(p[0])
                pos[k] = (p[1], p[2], p[3])
            yield step, typ, pos


def q_tet(typ, pos, L):
    """Mean Errington-Debenedetti q over all Si (4 nearest O, PBC)."""
    o = np.mod(pos[typ == 1], L)
    si = np.mod(pos[typ == 2], L)
    tree = cKDTree(o, boxsize=L)
    _, idx = tree.query(si, k=4)
    v = o[idx] - si[:, None, :]                       # (Nsi, 4, 3)
    v -= L * np.round(v / L)                          # minimum image
    v /= np.linalg.norm(v, axis=2)[:, :, None]
    q = np.full(si.shape[0], 1.0)
    for j in range(3):
        for k in range(j + 1, 4):
            cos = np.einsum("ij,ij->i", v[:, j], v[:, k])
            q -= 3.0 / 8.0 * (cos + 1.0 / 3.0) ** 2
    return float(q.mean())


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    cp_dir, n = argv[1].rstrip("/"), int(argv[2])
    pos_file = os.path.join(cp_dir, "B-part", f"pos{n}B.data")
    if not os.path.isfile(pos_file):
        raise SystemExit(f"no {pos_file} (frames not synced from cluster?)")
    log_file, lstart = range_log(cp_dir, n)
    thermo = b_block(log_file, n - lstart + 1)
    if not thermo:
        raise SystemExit(f"no B block for point {n} in {log_file}")

    t_ps, pe, qm = [], [], []
    for step, typ, pos in frames(pos_file):
        if step not in thermo:
            continue
        E, L = thermo[step]
        t_ps.append((step - 312500) * 0.0016)
        pe.append(E)
        qm.append(q_tet(typ, pos, L))
    t_ps, pe, qm = map(np.array, (t_ps, pe, qm))
    print(f"# {len(pe)} frames  P_target={(n - 1) * 0.25:.2f} GPa  "
          f"corr(PE,q)={np.corrcoef(pe, qm)[0, 1]:+.3f}")
    print(f"# <q>={qm.mean():.4f} +- {qm.std():.4f}   "
          f"<PE>={pe.mean():.1f} +- {pe.std():.1f} eV")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    h = ax[0].hist2d(qm, pe, bins=25, cmap="viridis")
    fig.colorbar(h[3], ax=ax[0], label="frames")
    ax[0].set_xlabel(r"$\langle q_{tet}\rangle$ per frame")
    ax[0].set_ylabel("PE (eV)")
    ax[0].set_title("2D histogram (two-state = two islands)")
    ax[0].locator_params(axis="x", nbins=6)
    sc = ax[1].scatter(qm, pe, c=t_ps, cmap="plasma", s=14)
    fig.colorbar(sc, ax=ax[1], label="t (ps)")
    ax[1].set_xlabel(r"$\langle q_{tet}\rangle$ per frame")
    ax[1].set_title("time-coloured (aging = drifting ridge)")
    label = f"{os.path.basename(os.path.dirname(cp_dir))} pt{n} " \
            f"({(n - 1) * 0.25:.2f} GPa)"
    fig.suptitle(f"PE vs tetrahedral order  {label}")
    fig.tight_layout()
    out = os.path.join(cp_dir, f"pe_q_pt{n}.png")
    fig.savefig(out, dpi=130)
    print(f"# wrote {out}")


if __name__ == "__main__":
    main(sys.argv)
