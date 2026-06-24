#!/usr/bin/env python3
"""SHIK/Wolf potential: energy derivatives + per-config virial and hypervirial.

Functional form (Sundararaman-Huang-Ispas-Kob; see the user's thesis eqn 1.7):

    V(r) = q_a q_b k_e [ 1/r - 1/Rc + (r-Rc)/Rc^2 ]        (Wolf, r < Rc=10 A)
         + A exp(-B r) - C/r^6 + D/r^24                      (short range, r < 8 A)

This module provides u(r), u'(r), u''(r) per type-pair and computes, for a single
configuration, the virial pressure P and the hypervirial X needed for the NVT
fluctuation route to the isothermal compressibility (Allen & Tildesley eqns
2.83-2.87):

    w(r) = r u'(r)            (pair virial function)
    x(r) = r w'(r) = r u'(r) + r^2 u''(r)   (hypervirial integrand)
    W = sum_{i<j} (-r u'(r))  -> P = rho kB T + W/(3V)        [virial pressure]
    X = sum_{i<j} x(r)                                         [extensive]

The virial-P calculation is validated against the LAMMPS-reported pressure
(see validate() / __main__): a correct match fixes the sign/prefactor chain so
the hypervirial X (same chain) can be trusted.

Units: distances A, energies eV, charges e. k_e = 14.399645 eV.A.
Pressure returned in bar (LAMMPS metal units): 1 eV/A^3 = 1.6021766e6 bar.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

KE = 14.399645            # e^2/(4 pi eps0) in eV.A
EV_A3_TO_BAR = 1.6021766e6
KB = 8.617333e-5          # eV/K
RSR = 8.0                 # short-range cutoff (A)
RC = 10.0                 # Wolf cutoff (A)

# charges (eV multiples): type 1 = O, type 2 = Si
Q = {1: -0.88775, 2: 1.7755}
# short-range params per unordered type pair: (A[eV], B[1/A], C[eV.A^6], D[eV.A^24])
SR = {
    (1, 1): (1120.528996, 2.892741833, 26.13207696, 16800.0),
    (1, 2): (23107.84764, 5.097856735, 139.6947857, 66.0),
    (2, 2): (2797.979165, 4.407320018, 0.0, 3423204.0),
}


def _sr(r, p):
    """short-range u, u', u'' for params p=(A,B,C,D)."""
    A, B, C, D = p
    e = A * np.exp(-B * r)
    u = e - C / r**6 + D / r**24
    du = -B * e + 6 * C / r**7 - 24 * D / r**25
    d2 = B * B * e - 42 * C / r**8 + 600 * D / r**26
    return u, du, d2


def _wolf(r, qq):
    """Wolf shifted-force u, u', u'' for charge product qq (in e^2)."""
    k = KE * qq
    u = k * (1.0 / r - 1.0 / RC + (r - RC) / RC**2)
    du = k * (-1.0 / r**2 + 1.0 / RC**2)
    d2 = k * (2.0 / r**3)
    return u, du, d2


def pair_terms(r, ta, tb):
    """u'(r) and u''(r) for a pair of types ta,tb at separations r (array)."""
    key = (min(ta, tb), max(ta, tb))
    qq = Q[ta] * Q[tb]
    du = np.zeros_like(r)
    d2 = np.zeros_like(r)
    inw = r < RC
    if inw.any():
        _, dw, d2w = _wolf(r[inw], qq)
        du[inw] += dw; d2[inw] += d2w
    ins = r < RSR
    if ins.any():
        _, ds, d2s = _sr(r[ins], SR[key])
        du[ins] += ds; d2[ins] += d2s
    return du, d2


def config_virial_hypervirial(pos, types, box, T):
    """Return (P_bar, X_extensive, n_pairs) for one configuration.

    pos: (N,3) A, types: (N,) int {1,2}, box: (3,) A (orthorhombic), T: K."""
    N = pos.size // 3
    V = float(np.prod(box))
    tree = cKDTree(pos % box, boxsize=box)
    pairs = tree.query_pairs(RC, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[i] - pos[j]
    d -= box * np.round(d / box)                 # minimum image
    r = np.sqrt((d * d).sum(axis=1))
    ta, tb = types[i], types[j]

    W = 0.0       # virial      sum_{i<j}(-r u')
    X = 0.0       # hypervirial  sum_{i<j}(r u' + r^2 u'')
    Bsum = 0.0    # Born term    sum_{i<j}(r^2 u'' - 2 r u')
    for a in (1, 2):
        for b in (1, 2):
            if b < a:
                continue
            m = ((ta == a) & (tb == b)) | ((ta == b) & (tb == a))
            if not m.any():
                continue
            rr = r[m]
            du, d2 = pair_terms(rr, a, b)
            W += float(np.sum(-rr * du))
            X += float(np.sum(rr * du + rr * rr * d2))
            Bsum += float(np.sum(rr * rr * d2 - 2.0 * rr * du))
    P = (N * KB * T / V + W / (3.0 * V)) * EV_A3_TO_BAR
    # configurational virial pressure and affine bulk modulus, in eV/A^3
    Pc = W / (3.0 * V)
    Born = (N / V) * KB * T + Bsum / (9.0 * V)
    return P, X, len(pairs), Pc, Born, V


# --------------------------------------------------------------------------- #
def read_lammps_data(path):
    """Parse a write_data file (atom_style charge): coords, types, cubic box."""
    lines = open(path).read().splitlines()
    box = np.zeros(3)
    natoms = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.endswith("atoms"):
            natoms = int(ln.split()[0])
        for k, tag in enumerate(("xlo xhi", "ylo yhi", "zlo zhi")):
            if ln.endswith(tag):
                lo, hi = float(ln.split()[0]), float(ln.split()[1])
                box[k] = hi - lo
        if ln.strip() == "Atoms" or ln.startswith("Atoms"):
            i += 2                                  # skip blank line
            pos = np.zeros((natoms, 3)); typ = np.zeros(natoms, int)
            for n in range(natoms):
                p = lines[i + n].split()
                typ[n] = int(p[1])
                pos[n] = [float(p[3]), float(p[4]), float(p[5])]
            return pos, typ, box
        i += 1
    raise SystemExit(f"no Atoms section in {path}")


def validate(datafile, P_lammps_bar, T):
    """Compute virial P from a config and compare to LAMMPS."""
    pos, typ, box = read_lammps_data(datafile)
    P, X, npair, _Pc, _Born, _V = config_virial_hypervirial(pos, typ, box, T)
    print(f"# validate {datafile}")
    print(f"#   N={pos.shape[0]}  box={box[0]:.4f} A  T={T} K  pairs<{RC}A={npair}")
    print(f"#   P_python = {P:12.1f} bar = {P/1e4:7.3f} GPa")
    print(f"#   P_lammps = {P_lammps_bar:12.1f} bar = {P_lammps_bar/1e4:7.3f} GPa")
    diff = (P - P_lammps_bar) / P_lammps_bar * 100
    print(f"#   diff     = {diff:+.2f} %   (hypervirial X = {X:.4e} eV)")
    return P, X


if __name__ == "__main__":
    # usage: shik.py <data file> <P_lammps_bar> <T_K>
    validate(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
