#!/usr/bin/env python3
"""kappa_T from fluctuations, both routes, cross-checked against the EOS.

For a probe directory sample-N/rho-LAB/kappa-TK/ produced by make_kappa_probes.sh:

  OPTION 1 (NPT volume fluctuations, Allen & Tildesley eqn 2.92):
      kappa_T = Var(V) / (<V> kB T)
    Clean and unambiguous; needs the per-step volume series vol.dat.

  OPTION 2 (NVT configurational-pressure fluctuations + SHIK hypervirial):
      K_T = <Born> - (V/kB T) Var(P_c) ,   kappa_T = 1/K_T
    Born = rho_N kB T + (1/9V) sum[r^2 u'' - 2 r u']  (affine modulus, from the
    dumped configs via shik.py); Var(P_c) from the per-step configurational
    pressure pc.dat. The Born term is exact (SHIK virial validated to 0.45%);
    the variance term REQUIRES per-step sampling -- coarse sampling
    underestimates Var and overestimates K_T.

Both are printed next to the EOS bulk modulus K_T = rho (dP/drho)_T from the
isochore grid (toolkit/llcp_analysis.py), which is the ground-truth cross-check.

usage: ./toolkit/kappa_fluctuations.py sample-1/rho-215/kappa-3000K [...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shik import KB, config_virial_hypervirial          # noqa: E402

EV_A3_TO_GPA = 160.21766
BAR_TO_GPA = 1e-4


def npt_kappa(probe, T):
    """Option 1: kappa_T [1/GPa] from per-step volume fluctuations."""
    f = os.path.join(probe, "vol.dat")
    if not os.path.isfile(f):
        return None
    V = np.loadtxt(f, comments="#", usecols=1)            # A^3, every step
    if V.size < 100:
        return None
    kappa = V.var() / (V.mean() * KB * T)                 # A^3/eV
    return kappa / EV_A3_TO_GPA                            # -> 1/GPa


def read_dump_frames(path):
    """Yield (types, pos) per frame of a LAMMPS custom dump (id type x y z)."""
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if "ITEM: NUMBER OF ATOMS" in line:
                n = int(fh.readline())
            elif "ITEM: BOX BOUNDS" in line:
                lo, hi = map(float, fh.readline().split()[:2]); L = hi - lo
                fh.readline(); fh.readline()
            elif line.startswith("ITEM: ATOMS"):
                typ = np.empty(n, int); pos = np.empty((n, 3))
                for k in range(n):
                    p = fh.readline().split()
                    typ[k] = int(p[1]); pos[k] = (p[2], p[3], p[4])
                yield typ, pos, L


def nvt_kappa(probe, T):
    """Option 2: kappa_T [1/GPa] from Born (configs) - fluct (per-step P_c)."""
    fp = os.path.join(probe, "pc.dat")
    fd = os.path.join(probe, "conf.dump")
    if not (os.path.isfile(fp) and os.path.isfile(fd)):
        return None
    Pc = np.loadtxt(fp, comments="#", usecols=1) * BAR_TO_GPA   # GPa, every step
    born = []
    V = None
    for typ, pos, L in read_dump_frames(fd):
        box = np.array([L, L, L])
        _P, _X, _n, _Pc, B, V = config_virial_hypervirial(pos, typ, box, T)
        born.append(B * EV_A3_TO_GPA)                     # GPa
    if not born or V is None:
        return None
    born = np.array(born)
    fluct = (V / (KB * T)) * (Pc.var() / EV_A3_TO_GPA**2) * EV_A3_TO_GPA
    K_T = born.mean() - fluct                             # GPa
    return (1.0 / K_T if K_T > 0 else float("nan"),
            born.mean(), fluct, Pc.std())


def eos_KT(sample_dir, lab, T):
    """Ground-truth K_T = rho (dP/drho)_T from neighbouring isochores [GPa]."""
    import glob
    pts = []
    for f in sorted(glob.glob(os.path.join(sample_dir, "rho-*", "isochore-PT.dat"))):
        if "_test" in f:
            continue
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            if int(float(c[0])) == int(T):
                pts.append((float(c[3]), float(c[2]) * BAR_TO_GPA))  # rho, P GPa
    if len(pts) < 3:
        return None
    pts.sort()
    rho = np.array([p[0] for p in pts]); P = np.array([p[1] for p in pts])
    rho0 = lab / 100.0
    j = int(np.argmin(abs(rho - rho0)))
    if j == 0 or j == rho.size - 1:
        return None
    dPdr = (P[j + 1] - P[j - 1]) / (rho[j + 1] - rho[j - 1])
    return rho[j] * dPdr                                  # GPa


def main(argv):
    print(f"{'probe':>26} {'T':>5} {'k_NPT':>9} {'k_NVT':>9} {'k_EOS':>9} "
          f"{'<Born>':>8} {'fluct':>8} {'sd(Pc)':>7}  [1/GPa, GPa]")
    for probe in argv[1:]:
        probe = probe.rstrip("/")
        T = float(os.path.basename(probe).replace("kappa-", "").replace("K", ""))
        lab = int(os.path.basename(os.path.dirname(probe)).replace("rho-", ""))
        sd = os.path.dirname(os.path.dirname(probe))
        k1 = npt_kappa(probe, T)
        nv = nvt_kappa(probe, T)
        ke = eos_KT(sd, lab, T)
        k2 = nv[0] if nv else None
        s = f"{probe:>26} {T:>5.0f}"
        s += f" {k1:>9.4f}" if k1 else f" {'-':>9}"
        s += f" {k2:>9.4f}" if k2 else f" {'-':>9}"
        s += f" {1/ke:>9.4f}" if ke else f" {'-':>9}"
        if nv:
            s += f" {nv[1]:>8.1f} {nv[2]:>8.1f} {nv[3]:>7.3f}"
        print(s)
    print("# k_NPT (opt.1) and k_NVT (opt.2) should agree with k_EOS; "
          "if k_NVT > k_EOS, Var(P_c) is undersampled (need per-step pc.dat).")


if __name__ == "__main__":
    main(sys.argv)
