#!/usr/bin/env python3
"""Validate the NVT hypervirial bulk-modulus formula against the EOS.

For a set of NVT configurations (an xyz trajectory at fixed box L, temperature
T) compute, per frame:
    P_c    = (1/3V) sum_{i<j} (-r u'(r))         configurational virial pressure
    Born   = rho_N kB T + (1/9V) sum [r^2 u'' - 2 r u']   affine bulk modulus
and assemble the isothermal bulk modulus (Squire-Holt-Hoover / Born + stress
fluctuations):
    K_T = <Born> - (V/kB T) Var(P_c)
Compared to the EOS value K_T = rho (dP/drho)_T from neighbouring isochores.

usage: validate_kappa_nvt.py <xyz traj> <L_A> <T_K> <P_series_file> <K_eos_GPa>
       (P_series_file: one configurational+total P per dumped frame, bar; or '-'
        to use per-frame computed P_c for the variance)
"""
import sys

import numpy as np

from shik import KB, config_virial_hypervirial

EV_A3_TO_GPA = 160.21766


def frame_terms(pos, types, L, T):
    box = np.array([L, L, L])
    _P, _X, _np, Pc, Born, V = config_virial_hypervirial(pos, types, box, T)
    return Pc, Born, V                                   # eV/A^3, eV/A^3, A^3


def read_xyz(path, maxframes=None):
    frames = []
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            n = int(line)
            fh.readline()                                 # comment
            typ = np.empty(n, int); pos = np.empty((n, 3))
            for k in range(n):
                p = fh.readline().split()
                typ[k] = int(p[0])
                pos[k] = (float(p[1]), float(p[2]), float(p[3]))
            frames.append((typ, pos))
            if maxframes and len(frames) >= maxframes:
                break
    return frames


def main(argv):
    traj, L, T, K_eos = argv[1], float(argv[2]), float(argv[3]), float(argv[5])
    frames = read_xyz(traj)
    Pc = np.empty(len(frames)); Born = np.empty(len(frames))
    for f, (typ, pos) in enumerate(frames):
        Pc[f], Born[f], V = frame_terms(pos, typ, L, T)
    kBT = KB * T
    var_Pc = float(Pc.var())                              # (eV/A^3)^2
    fluct = V / kBT * var_Pc                               # eV/A^3
    K_T = float(Born.mean()) - fluct                      # eV/A^3
    print(f"# frames={len(frames)}  L={L} A  T={T} K  V={V:.1f} A^3")
    print(f"#   <P_c>   = {Pc.mean()*EV_A3_TO_GPA:8.3f} GPa "
          f"(std {Pc.std()*EV_A3_TO_GPA:.3f})")
    print(f"#   <Born>  = {Born.mean()*EV_A3_TO_GPA:8.3f} GPa")
    print(f"#   fluct   = {fluct*EV_A3_TO_GPA:8.3f} GPa  (V/kBT * Var(P_c))")
    print(f"#   K_T     = {K_T*EV_A3_TO_GPA:8.3f} GPa   (Born - fluct)")
    print(f"#   K_eos   = {K_eos:8.3f} GPa   -> kappa_T = {1/K_eos:.4f} /GPa")
    print(f"#   ratio K_T/K_eos = {K_T*EV_A3_TO_GPA/K_eos:.3f}")


if __name__ == "__main__":
    main(sys.argv)
