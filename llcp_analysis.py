#!/usr/bin/env python3
"""LLCP search from the isochore campaign.

Reads every sample-N/rho-XXX/isochore-PT.dat (cols: Tset <T> <P_bar> <rho> <PE>
<CN>) and looks for a liquid-liquid critical point in two equivalent ways:

  1. Isochores P(T): a LLCP makes neighbouring isochores CONVERGE / CROSS in the
     P-T plane (Lascaris 2014/2016).
  2. Isotherms P(rho): below Tc a van-der-Waals-like flattening/loop appears,
     i.e. (dP/drho)_T drops toward zero (or negative => mechanically unstable).
     The critical T is where the minimum of (dP/drho)_T first touches ~0.

Outputs: isochores + isotherms PNGs, and a per-isotherm table of the minimum
slope (dP/drho) with the rho where it occurs -> the LLCP estimate.

usage: ./toolkit/llcp_analysis.py [sample-dir ...]   (default: all sample-*/)
"""
import glob
import os
import sys

import numpy as np

GPA = 1e-4  # bar -> GPa

# Glass-transition locus Tg(P) for this silica system (Tg in K at P in GPa).
# Pressure-induced Tg minimum near 20 GPa; below this line the liquid is glassy.
TG_P_GPA = (0, 5, 10, 20, 30)
TG_K = (2280, 1980, 1650, 1637, 1800)


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


def load_sample(sd):
    """Return dict rho(g/cc) -> (T[], P_GPa[]) for one sample dir."""
    data = {}
    for f in sorted(glob.glob(os.path.join(sd, "rho-*", "isochore-PT.dat"))):
        if "_test" in f:
            continue
        rows = []
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            rows.append((float(p[0]), float(p[2]) * GPA, float(p[3])))
        if not rows:
            continue
        rows.sort(key=lambda r: -r[0])         # descending T
        T = np.array([r[0] for r in rows])
        P = np.array([r[1] for r in rows])
        rho = float(np.median([r[2] for r in rows]))
        data[rho] = (T, P)
    return data


def isotherms(data):
    """Invert to T -> (rho_sorted[], P[]) using the common T ladder."""
    allT = sorted({t for (T, _) in data.values() for t in T}, reverse=True)
    out = {}
    for T0 in allT:
        pts = []
        for rho, (T, P) in data.items():
            m = np.isclose(T, T0)
            if m.any():
                pts.append((rho, float(P[m][0])))
        if len(pts) >= 4:
            pts.sort()
            out[T0] = (np.array([p[0] for p in pts]),
                       np.array([p[1] for p in pts]))
    return out


def widom_locus(iso):
    """Widom line = (T, P) of the kappa_T maximum on each isotherm
    (kappa_T = 1/(rho*(dP/drho)_T); interior point, dP/drho>0). As T -> Tc this
    locus tracks the response-function ridge emanating from a (possible) LLCP."""
    Tw, Pw = [], []
    for T0 in sorted(iso, reverse=True):
        r, P = iso[T0]
        if r.size < 3:
            continue
        dPdr = np.gradient(P, r)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = 1.0 / (r * dPdr)
        ok = np.ones(r.size, bool); ok[0] = ok[-1] = False
        ok &= dPdr > 0
        if ok.any():
            idx = np.where(ok)[0]
            j = idx[int(np.argmax(kappa[idx]))]
            Tw.append(T0); Pw.append(float(P[j]))
    o = np.argsort(Tw)
    return np.array(Tw)[o], np.array(Pw)[o]


def analyse(data, label, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # Drop cavitated (holes) state points: they are phase-separated (liquid-vapor),
    # not single-phase liquid, so their P is not an EOS value -- remove from all
    # isochore/isotherm analysis and plots. A density fully cavitated is dropped.
    cav = load_cavitated(os.path.dirname(outdir))
    if cav:
        filt = {}
        for rho, (T, P) in data.items():
            keep = [(t, p) for t, p in zip(T, P)
                    if (round(rho, 2), int(t)) not in cav]
            if keep:
                tt, pp = zip(*keep)
                filt[rho] = (np.array(tt), np.array(pp))
        data = filt
    iso = isotherms(data)

    # ---- isochores P(T) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, len(data))))
    for rho in sorted(data):
        T, P = data[rho]
        ax.plot(T, P, "-o", ms=3, lw=0.8, label=f"{rho:.2f}")
    Tw, Pw = widom_locus(iso)                           # Widom line (kappa_T max)
    # if Tw.size:                                       # disabled: kappa_T-max
    #     ax.plot(Tw, Pw, "k-s", ms=4, lw=1.6, zorder=5,  # is edge-pinned here,
    #             label=r"Widom line ($\kappa_T$ max)")    # misleading to draw
    ax.plot(TG_K, TG_P_GPA, "r--", lw=1.4, zorder=4)   # glass-transition line
    ax.annotate(r"$T_g(P)$", (TG_K[0], TG_P_GPA[0]), color="r", fontsize=8,
                xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("T (K)"); ax.set_ylabel("P (GPa)")
    ax.set_title(f"Isochores  {label}")
    ax.legend(fontsize=6, ncol=2, title=r"$\rho$ (g/cc)")
    # focus y on the data; Tg(P) is only a guide for the eye, let it clip
    allP = np.concatenate([P for (_, P) in data.values()])
    pad = 0.05 * (allP.max() - allP.min())
    ax.set_ylim(allP.min() - pad, allP.max() + pad)
    fig.tight_layout(); fig.savefig(f"{outdir}/isochores.png", dpi=130)
    plt.close(fig)

    # ---- isochores variant T(P): same data, axes swapped (T on y, P on x) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, len(data))))
    for rho in sorted(data):
        T, P = data[rho]                       # rows are ordered in T already
        ax.plot(P, T, "-o", ms=3, lw=0.8, label=f"{rho:.2f}")
    # if Tw.size:                                       # disabled (see isochores)
    #     ax.plot(Pw, Tw, "k-s", ms=4, lw=1.6, zorder=5,
    #             label=r"Widom line ($\kappa_T$ max)")
    ax.plot(TG_P_GPA, TG_K, "r--", lw=1.4, zorder=4)   # glass-transition guide
    ax.annotate(r"$T_g(P)$", (TG_P_GPA[0], TG_K[0]), color="r", fontsize=8,
                xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("P (GPa)"); ax.set_ylabel("T (K)")
    ax.set_title(f"Isochores T(P)  {label}")
    ax.legend(fontsize=6, ncol=2, title=r"$\rho$ (g/cc)")
    ax.set_xlim(allP.min() - pad, allP.max() + pad)    # focus x on the data
    fig.tight_layout(); fig.savefig(f"{outdir}/isochores_TP.png", dpi=130)
    plt.close(fig)

    # ---- isotherms P(rho) + slope diagnostic ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    _vir = plt.cm.viridis(np.linspace(0, 1, len(iso)))
    ax[0].set_prop_cycle(color=_vir); ax[1].set_prop_cycle(color=_vir)
    print(f"\n# {label}: minimum INTERIOR (dP/drho)_T per isotherm.")
    print(f"#   LLCP requires (dP/drho)_T <= 0 (van-der-Waals loop) at some rho.")
    print(f"{'T(K)':>6} {'min dP/drho':>12} {'@rho':>7} "
          f"{'P@min(GPa)':>12}  flag")
    crit = []
    for T0 in sorted(iso, reverse=True):
        r, P = iso[T0]
        ax[0].plot(r, P, "-o", ms=3, lw=0.8, label=f"{T0:.0f}")
        dPdr = np.gradient(P, r)
        ax[1].plot(r, dPdr, "-o", ms=3, lw=0.8, label=f"{T0:.0f}")
        # ignore the two boundary points (one-sided gradient is unreliable)
        if dPdr.size > 4:
            interior = dPdr[1:-1]
            jj = int(np.argmin(interior)) + 1
        else:
            jj = int(np.argmin(dPdr))
        smin = float(dPdr[jj]); Pmin = float(P[jj])
        # A loop (dP/drho<=0) is a LLCP signature ONLY at positive P. At P<0 it is
        # the liquid-vapor (cavitation) spinodal -- a different instability that
        # must not be counted as a LLCP (Lascaris/Sciortino 2016).
        if smin <= 0:
            flag = "<-- LLCP loop" if Pmin > 0 else "<-- L-V spinodal (P<0)"
        else:
            flag = "soft" if smin < 1 else ""
        print(f"{T0:>6.0f} {smin:>12.2f} {r[jj]:>7.2f} {Pmin:>12.2f}  {flag}")
        crit.append((T0, smin, r[jj], Pmin))
    ax[0].set_xlabel(r"$\rho$ (g/cc)"); ax[0].set_ylabel("P (GPa)")
    ax[0].set_title("Isotherms"); ax[0].legend(fontsize=6, ncol=2, title="T (K)")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel(r"$\rho$ (g/cc)"); ax[1].set_ylabel(r"$(dP/d\rho)_T$")
    ax[1].set_title("Mechanical stability"); ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{outdir}/isotherms.png", dpi=130)
    plt.close(fig)

    # ---- density anomaly: P minimum vs T along each isochore (TMD locus) ----
    print(f"\n# {label}: density anomaly -- isochores with a P-minimum vs T "
          f"(TMD locus, LLCP precursor):")
    tmd = []
    for rho in sorted(data):
        T, P = data[rho]
        order = np.argsort(T)
        Ts, Ps = T[order], P[order]
        if Ps.size >= 3:
            k = int(np.argmin(Ps))
            if 0 < k < Ps.size - 1:        # interior minimum
                tmd.append((rho, Ts[k], Ps[k]))
    if tmd:
        for rho, Tm, Pm in tmd:
            print(f"   rho={rho:.2f}: P-min at T~{Tm:.0f} K, P~{Pm:.3f} GPa")
    else:
        print("   none (no interior P(T) minimum in range)")

    # ---- isochore crossings (direct LLCP signature) ----
    crossings = _crossings(data)
    npos = sum(1 for x in crossings if x[3] > 0)
    print(f"\n# {label}: isochore crossings in P-T plane "
          f"(direct LLCP signature): {len(crossings)} ({npos} at P>0)")
    for (r1, r2, Tx, Px) in crossings:
        tag = "" if Px > 0 else "  [L-V spinodal, P<0 -- not a LLCP]"
        print(f"   {r1:.2f} x {r2:.2f}  at T~{Tx:.0f} K, P~{Px:.3f} GPa{tag}")

    # ---- verdict ----
    # Only loops/crossings ABOVE the liquid-vapor spinodal (P>0) count as a LLCP;
    # subcritical isotherms and crossings at P<0 are cavitation, not polyamorphism.
    sub = [c for c in crit if c[1] <= 0 and c[3] > 0]
    xpos = [x for x in crossings if x[3] > 0]
    spin = [c for c in crit if c[1] <= 0 and c[3] <= 0]
    if sub or xpos:
        Tc = max(c[0] for c in sub) if sub else xpos[0][2]
        print(f"\n# {label}: LLCP candidate near T~{Tc:.0f} K "
              f"(subcritical isotherm and/or isochore crossing at P>0).")
    else:
        smin_all = min(crit, key=lambda c: c[1])
        msg = (f"\n# {label}: NO LLCP in the sampled window "
               f"(no P>0 van-der-Waals loop, no P>0 isochore crossing). "
               f"Softest point dP/drho_min={smin_all[1]:.1f} at "
               f"T={smin_all[0]:.0f} K, rho={smin_all[2]:.2f} g/cc.")
        if spin:
            msg += (f" The dP/drho<=0 loops found are all at P<0 (lowest "
                    f"P~{min(c[3] for c in spin):.2f} GPa) = liquid-vapor "
                    f"spinodal/cavitation, NOT a LLCP.")
        if tmd:
            msg += (" A density anomaly (TMD) IS present -> LLCP, if it exists, "
                    "lies below the lowest equilibrated T.")
        print(msg)
    print(f"# plots -> {outdir}/isochores.png , {outdir}/isotherms.png")
    response_functions(data, label, outdir)
    gamma_v_analysis(data, label, outdir)


def gamma_v_analysis(data, label, outdir):
    """Thermal pressure coefficient gamma_V = (dP/dT)_V along each isochore.

    gamma_V = 0  -> P-minimum vs T = the TMD / density-anomaly locus.
    LLCP signature (Anara/Saika-Voivod): the gamma_V(T) curves of different
    isochores INTERSECT near the critical point, because there (dP/dT)_V of the
    coexisting low- and high-density branches coincide. Crossings of adjacent
    isochores are flagged; with no crossing the LLCP is outside the window.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # gamma_V(rho,T) on the common T grid
    rhos = sorted(data)
    gv = {}
    for rho in rhos:
        T, P = data[rho]
        o = np.argsort(T)
        if o.size >= 3:
            gv[rho] = (T[o], np.gradient(P[o], T[o]) * 1e3)   # MPa/K
    allT = sorted({int(t) for (Ts, _) in gv.values() for t in Ts}, reverse=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, len(rhos))))
    # (a) gamma_V vs T per isochore
    for rho in rhos:
        if rho in gv:
            ax[0].plot(gv[rho][0], gv[rho][1], "-o", ms=3, lw=0.8,
                       label=f"{rho:.2f}")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("T (K)")
    ax[0].set_ylabel(r"$\gamma_V=(\partial P/\partial T)_V$ (MPa/K)")
    ax[0].set_title("isochores"); ax[0].legend(fontsize=5, ncol=2, title=r"$\rho$")
    # (b) gamma_V vs rho per isotherm  -> shows the min at rho~2.6 / TMD dome
    import matplotlib.cm as cm
    for k, T0 in enumerate(allT):
        rr = [r for r in rhos if r in gv and T0 in gv[r][0]]
        vv = [float(gv[r][1][list(gv[r][0]).index(T0)]) for r in rr]
        ax[1].plot(rr, vv, "-o", ms=3, lw=0.8, color=cm.viridis(k / len(allT)),
                   label=f"{T0}")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel(r"$\rho$ (g/cc)"); ax[1].set_ylabel(r"$\gamma_V$ (MPa/K)")
    ax[1].set_title("isotherms (gamma_V<0 = density-anomaly dome)")
    ax[1].legend(fontsize=5, ncol=2, title="T(K)")
    fig.tight_layout(); fig.savefig(f"{outdir}/gamma_v.png", dpi=130)
    plt.close(fig)

    # TMD / density-anomaly dome: gamma_V=0 crossings along each isotherm, and
    # the gamma_V-minimum locus (strongest anomaly). gamma_V<0 = negative thermal
    # expansion (the LLCP precursor region; LLCP sits below this dome).
    print(f"\n# {label}: gamma_V=(dP/dT)_V density-anomaly dome "
          f"(gamma_V<0 region; LLCP lies below it):")
    apexT = None
    for T0 in allT:
        rr = np.array([r for r in rhos if r in gv and T0 in gv[r][0]])
        vv = np.array([float(gv[r][1][list(gv[r][0]).index(T0)]) for r in rr])
        s = np.sign(vv)
        edges = []
        for i in range(s.size - 1):
            if s[i] and s[i + 1] and s[i] != s[i + 1]:
                f = vv[i] / (vv[i] - vv[i + 1])
                edges.append(rr[i] + f * (rr[i + 1] - rr[i]))
        if (vv < 0).any():
            apexT = T0 if apexT is None else max(apexT, T0)
            jmin = int(np.argmin(vv))
            edge_s = " , ".join(f"{e:.2f}" for e in edges) if edges else "open"
            print(f"   T={T0:>4d} K: gamma_V<0 (anomaly), edges rho=[{edge_s}], "
                  f"min at rho~{rr[jmin]:.2f} ({vv[jmin]:.2f} MPa/K)")
    if apexT is not None:
        print(f"#   anomaly-dome apex ~{apexT} K (highest T with gamma_V<0); "
              f"LLCP is below this. Anomaly strongest near rho~2.55-2.65.")
    else:
        print("   no gamma_V<0 in window (no density anomaly sampled).")
    print(f"# plot -> {outdir}/gamma_v.png")


def response_functions(data, label, outdir):
    """Isothermal compressibility kappa_T from the EOS grid (no fluctuations
    needed): kappa_T = 1/(rho * (dP/drho)_T). Its maximum vs rho at each T is a
    point on the Widom line; as T -> T_c the maximum diverges, so 1/kappa_max
    extrapolates linearly to 0 at T_c. NVT gives no volume fluctuations, so the
    EOS derivative is the correct (and only) route here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iso = isotherms(data)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].set_prop_cycle(color=plt.cm.viridis(np.linspace(0, 1, len(iso))))

    print(f"\n# {label}: isothermal compressibility kappa_T = "
          f"1/(rho*(dP/drho)_T)  [1/GPa], from the EOS grid.")
    print(f"{'T(K)':>6} {'kappa_max':>10} {'@rho':>7} {'P(GPa)':>9}")
    widom = []                                    # (T, kappa_max, rho, P)
    for T0 in sorted(iso, reverse=True):
        r, P = iso[T0]
        dPdr = np.gradient(P, r)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = 1.0 / (r * dPdr)
        ax[0].plot(r, kappa, "-o", ms=3, lw=0.8, label=f"{T0:.0f}")
        # interior maximum where dP/drho > 0 (stable, finite kappa)
        ok = np.ones(r.size, bool); ok[0] = ok[-1] = False
        ok &= dPdr > 0
        if ok.any():
            idx = np.where(ok)[0]
            j = idx[int(np.argmax(kappa[idx]))]
            widom.append((T0, float(kappa[j]), float(r[j]), float(P[j])))
            print(f"{T0:>6.0f} {kappa[j]:>10.4f} {r[j]:>7.2f} {P[j]:>9.2f}")
    ax[0].set_xlabel(r"$\rho$ (g/cc)"); ax[0].set_ylabel(r"$\kappa_T$ (1/GPa)")
    ax[0].set_title("Compressibility"); ax[0].legend(fontsize=6, ncol=2,
                                                     title="T (K)")

    # ---- Widom line / extrapolation: 1/kappa_max vs T -> 0 at T_c ----
    rho_lo = sorted(data)[1]            # lowest INTERIOR density (edge excluded)
    print(f"\n# {label}: WIDOM-LINE EXTRAPOLATION (kappa_T-max locus):")
    if len(widom) >= 3:
        Tw = np.array([w[0] for w in widom])
        inv = np.array([1.0 / w[1] for w in widom])
        rho_at = np.array([w[2] for w in widom])
        ax[1].plot(Tw, inv, "o-", lw=0.8, label=r"$1/\kappa_{max}$")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xlabel("T (K)"); ax[1].set_ylabel(r"1/$\kappa_{max}$ (GPa)")
        ax[1].set_title("Widom-line check")

        # guard 1: kappa-max must sit OFF the density boundary (interior ridge)
        order = np.argsort(Tw)
        boundary = np.isclose(rho_at, rho_lo)
        # guard 2: 1/kappa_max must fall monotonically toward low T (approach Tc)
        inv_lowT = inv[order]
        falling = np.all(np.diff(inv_lowT) >= -1e-9)  # increasing with T index
        # i.e. inv decreases as T decreases <=> increases with T

        if boundary.mean() > 0.5:
            ax[1].set_title("Widom ridge at density boundary")
            print(f"#   kappa_T-max pinned at the low-density edge "
                  f"(rho={rho_lo:.2f}) at almost every T -> the Widom ridge "
                  f"lies at rho < {rho_lo:.2f} g/cc, OFF this grid. "
                  f"NO reliable LLCP extrapolation. Extend the density grid "
                  f"to lower rho (e.g. 1.9-2.1) to bracket the kappa_T maximum.")
        elif not falling:
            kpk = Tw[int(np.argmin(inv))]
            print(f"#   1/kappa_max is non-monotonic (kappa_T-max peaks at "
                  f"T~{kpk:.0f} K, then decreases) -> no approach to a "
                  f"divergence in-window. LLCP is below the explored T; need "
                  f"lower-T data before extrapolating.")
        else:
            sel = order
            a, b = np.polyfit(Tw[sel], inv[sel], 1)
            Tc = -b / a if a != 0 else float("nan")
            ra, rb = np.polyfit(Tw[sel], rho_at[sel], 1)
            Pw = np.array([w[3] for w in widom])
            pa, pb = np.polyfit(Tw[sel], Pw[sel], 1)
            tt = np.linspace(min(Tc, Tw.min()), Tw.max(), 50)
            ax[1].plot(tt, a * tt + b, "r--", lw=0.8, label=f"Tc={Tc:.0f} K")
            ax[1].set_title("Widom-line extrapolation")
            print(f"#   1/kappa_max -> 0 at  T_c ~ {Tc:.0f} K, "
                  f"rho_c ~ {ra*Tc+rb:.2f} g/cc, P_c ~ {pa*Tc+pb:.2f} GPa "
                  f"(linear extrapolation -- verify with lower-T / more samples)")
        ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir}/kappa.png", dpi=130)
    plt.close(fig)
    print(f"# plot -> {outdir}/kappa.png")


def _crossings(data):
    """Find pairs of adjacent isochores whose P(T) curves cross."""
    rhos = sorted(data)
    out = []
    for a, b in zip(rhos[:-1], rhos[1:]):
        Ta, Pa = data[a]; Tb, Pb = data[b]
        Tcommon = sorted(set(Ta).intersection(set(Tb)))
        if len(Tcommon) < 2:
            continue
        pa = np.array([Pa[list(Ta).index(t)] for t in Tcommon])
        pb = np.array([Pb[list(Tb).index(t)] for t in Tcommon])
        d = pa - pb                         # higher rho should have higher P
        s = np.sign(d)
        for i in range(len(s) - 1):
            if s[i] != 0 and s[i + 1] != 0 and s[i] != s[i + 1]:
                # linear interpolate the crossing T
                t0, t1 = Tcommon[i], Tcommon[i + 1]
                f = d[i] / (d[i] - d[i + 1])
                Tx = t0 + f * (t1 - t0)
                Px = pa[i] + f * (pa[i + 1] - pa[i])
                out.append((a, b, Tx, Px))
    return out


def main(argv):
    sdirs = argv[1:] or sorted(glob.glob("sample-*"))
    sdirs = [s for s in sdirs if os.path.isdir(s)]
    if not sdirs:
        raise SystemExit("no sample-* dirs found")
    for sd in sdirs:
        data = load_sample(sd)
        if len(data) < 4:
            print(f"# {sd}: only {len(data)} usable isochores, skipping")
            continue
        analyse(data, os.path.basename(sd), os.path.join(sd, "llcp"))


if __name__ == "__main__":
    main(sys.argv)
