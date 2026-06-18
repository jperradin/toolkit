#!/usr/bin/env python3
"""Per-temperature equilibration check for an NVT isochore run.

Given a ``rho-XXX`` directory (or its ``log.lammps``), this splits the log into
the per-temperature equilibration and production runs of the cooling ladder and
judges, for each temperature, whether the production block is equilibrated.

Isochore = fixed volume, descending T, <P>(T) measured. The SLOW variable is
the pressure (volume is constant), so equilibration is judged on P:

  drift      linear P drift across production, as a fraction of std(P)
  half       |<P>_1st_half - <P>_2nd_half| / std(P)
  cont       |<P>_equil_tail - <P>_production| / std(P)   (equil -> prod jump)
  N_eff      decorrelated samples of P (autocorrelation-corrected)

plus two crystallization / arrest flags from the diagnostics dumped by the
generated inputs (build_tree.sh):

  dCN        change in mean Si-O coordination across the block (a jump => the
             liquid ordered / crystallized; the Lascaris LLCP order parameter)
  MSD        Si mean-squared displacement still growing (liquid mobile) vs flat

Verdict per T: OK / WARN / NOT-EQ / CRYSTAL? / INCOMPLETE.

Examples:
    ./detect_equilibration.py ../sample-1/rho-210
    ./detect_equilibration.py ../sample-1/rho-210_test/log.lammps
    ./detect_equilibration.py ../sample-1/rho-210 --plot --csv eq.csv
"""

import argparse
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Thresholds (physical scale: fractions of the P fluctuation width std(P))    #
# --------------------------------------------------------------------------- #
EQ_DRIFT_FRAC_MAX = 1.0    # |total P drift| / std(P): NOT-EQ above this
EQ_SHIFT_FRAC_MAX = 0.5    # half-half and equil->prod shift / std(P)
EQ_MIN_NEFF = 20.0         # need at least this many decorrelated P samples
WARN_FRAC = 0.5            # fraction of a NOT-EQ threshold that already WARNs
CRYSTAL_DCN = 0.05         # |dCN| within one block flagged as ordering/crystal

DT_PS = 0.0016             # timestep (ps); used only for ns axis / drift rates

# Thermo column names emitted by the generated isochore inputs.
_C_STEP = "Step"
_C_TEMP = "Temp"
_C_PE = "PotEng"
_C_PRESS = "Press"
_C_MSD = "c_msd1[4]"
_C_CN = "c_cnSiO"

_NVT_RE = re.compile(r"\bnvt\s+temp\s+(\S+)\s+(\S+)")
_AVG_RE = re.compile(r"\bave/time\b")


# --------------------------------------------------------------------------- #
# Autocorrelation (Sokal) -- same method as thermo_analyzer.py                 #
# --------------------------------------------------------------------------- #
def integrated_autocorr_time(x):
    """tau_int = 1 + 2 sum rho(k), initial-positive-sequence + Sokal window."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return 1.0
    xc = x - x.mean()
    var = float(np.dot(xc, xc) / n)
    if var == 0.0:
        return 1.0
    acf = np.correlate(xc, xc, mode="full")[n - 1:] / (var * n)
    tau = 1.0
    for k in range(1, n):
        rho = acf[k]
        if rho <= 0.0:
            break
        tau += 2.0 * rho
        if k >= 6.0 * tau:
            break
    return max(tau, 1.0)


def mean_sem(x):
    """Mean and autocorrelation-corrected SEM; also tau and N_eff."""
    x = np.asarray(x, dtype=float)
    tau = integrated_autocorr_time(x)
    n_eff = max(x.size / tau, 1.0)
    sem = float(x.std()) / np.sqrt(n_eff)
    return float(x.mean()), sem, tau, n_eff


# --------------------------------------------------------------------------- #
# Log parsing                                                                  #
# --------------------------------------------------------------------------- #
class Block:
    """One thermo table (one ``run``) with the columns we need."""

    def __init__(self, cols, arr, tset, is_prod, complete):
        self.tset = tset            # target temperature (K)
        self.is_prod = is_prod      # production block (had a fix ave/time)?
        self.complete = complete    # did the run finish (Loop time seen)?
        idx = {c: cols.index(c) for c in cols}
        self.step = arr[:, idx[_C_STEP]]
        self.temp = arr[:, idx[_C_TEMP]]
        self.press = arr[:, idx[_C_PRESS]] / 1000.0          # bar -> kbar
        self.pe = arr[:, idx[_C_PE]]                          # eV
        self.msd = arr[:, idx[_C_MSD]] if _C_MSD in idx else None
        self.cn = arr[:, idx[_C_CN]] if _C_CN in idx else None
        self.t_ns = (self.step - self.step[0]) * DT_PS / 1000.0


def parse_log(log_file):
    """Split a log into ordered Blocks, tagging production runs and target T."""
    with open(log_file) as fh:
        lines = fh.readlines()

    blocks = []
    cols = None
    rows = None
    cur_tset = None
    pending_prod = False  # a fix ave/time was declared since the last run

    def flush(complete):
        nonlocal rows, pending_prod
        if rows:
            arr = np.array(rows, dtype=float)
            blocks.append(Block(cols, arr, cur_tset, pending_prod, complete))
            pending_prod = False
        rows = None

    for line in lines:
        parts = line.split()
        if parts and parts[0] == _C_STEP and _C_TEMP in parts:
            flush(complete=False)        # new header => previous block had no Loop time
            cols = parts
            rows = []
            continue
        if rows is not None and len(parts) == len(cols):
            try:
                rows.append([float(p) for p in parts])
                continue
            except ValueError:
                pass
        # non-data line: harvest control commands the log echoes
        if rows is not None and rows:
            flush(complete=False)
        m = _NVT_RE.search(line)
        if m:
            cur_tset = float(m.group(2))   # target T (ramp end / setpoint)
        if _AVG_RE.search(line):
            pending_prod = True
        if "Loop time" in line and blocks:
            blocks[-1].complete = True
    flush(complete=False)
    return blocks


def pair_production(blocks):
    """Yield (prod_block, equil_block_or_None) for each production block.

    The equilibration partner is the nearest preceding non-production block
    with the same target temperature.
    """
    for i, b in enumerate(blocks):
        if not b.is_prod:
            continue
        eq = None
        for j in range(i - 1, -1, -1):
            if not blocks[j].is_prod and blocks[j].tset == b.tset:
                eq = blocks[j]
                break
        yield b, eq


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #
def assess(prod, equil):
    """Return a dict of equilibration metrics + verdict for one temperature."""
    P = prod.press
    meanP, semP, tau, n_eff = mean_sem(P)
    stdP = float(P.std())

    slope = float(np.polyfit(prod.t_ns, P, 1)[0])          # kbar/ns
    drift = slope * (prod.t_ns[-1] - prod.t_ns[0])         # total kbar
    drift_frac = abs(drift) / stdP if stdP > 0 else 0.0

    half = P.size // 2
    half_frac = (abs(P[:half].mean() - P[half:].mean()) / stdP
                 if stdP > 0 else 0.0)

    if equil is not None and equil.press.size >= 2 and stdP > 0:
        eh = equil.press.size // 2
        cont_frac = abs(float(equil.press[eh:].mean()) - meanP) / stdP
    else:
        cont_frac = float("nan")

    # crystallization / arrest: coordination change across the production block
    if prod.cn is not None and prod.cn.size >= 20:
        k = max(prod.cn.size // 10, 1)
        dcn = float(prod.cn[-k:].mean() - prod.cn[:k].mean())
    else:
        dcn = float("nan")

    # mobility: is Si MSD still growing in the 2nd half? (informational)
    msd_end = float(prod.msd[-1]) if prod.msd is not None else float("nan")
    if prod.msd is not None and prod.msd.size >= 4:
        h = prod.msd.size // 2
        msd_slope = float(np.polyfit(prod.t_ns[h:], prod.msd[h:], 1)[0])
    else:
        msd_slope = float("nan")

    # verdict
    fails = []
    if not prod.complete:
        verdict = "INCOMPLETE"
    elif not np.isnan(dcn) and abs(dcn) > CRYSTAL_DCN:
        verdict = "CRYSTAL?"
    else:
        if drift_frac >= EQ_DRIFT_FRAC_MAX:
            fails.append("drift")
        if half_frac >= EQ_SHIFT_FRAC_MAX:
            fails.append("half")
        if not np.isnan(cont_frac) and cont_frac >= EQ_SHIFT_FRAC_MAX:
            fails.append("cont")
        if n_eff < EQ_MIN_NEFF:
            fails.append("Neff")
        if fails:
            verdict = "NOT-EQ"
        elif (drift_frac >= WARN_FRAC * EQ_DRIFT_FRAC_MAX
              or half_frac >= WARN_FRAC * EQ_SHIFT_FRAC_MAX
              or (not np.isnan(cont_frac)
                  and cont_frac >= WARN_FRAC * EQ_SHIFT_FRAC_MAX)):
            verdict = "WARN"
        else:
            verdict = "OK"

    return {
        "T": prod.tset,
        "meanP": meanP,
        "semP": semP,
        "stdP": stdP,
        "slope": slope,
        "drift_frac": drift_frac,
        "half_frac": half_frac,
        "cont_frac": cont_frac,
        "tau": tau,
        "N_eff": n_eff,
        "dcn": dcn,
        "msd_end": msd_end,
        "msd_slope": msd_slope,
        "complete": prod.complete,
        "len_ns": float(prod.t_ns[-1]),
        "verdict": verdict,
        "fails": fails,
    }


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
_HDR = (f"{'T(K)':>6} {'<P>kbar':>9} {'+-SEM':>7} {'drift/s':>8} "
        f"{'half/s':>7} {'cont/s':>7} {'N_eff':>7} {'dCN':>8} "
        f"{'MSDend':>8} {'ns':>5}  verdict")


def fmt_row(r):
    def g(x, w, p="6.2f"):
        return f"{'  n/a':>{w}}" if (isinstance(x, float) and np.isnan(x)) \
            else f"{x:>{w}.{p.split('.')[1][0]}f}"
    note = ("  [" + ",".join(r["fails"]) + "]") if r["fails"] else ""
    return (f"{r['T']:>6.0f} {r['meanP']:>9.2f} {r['semP']:>7.3f} "
            f"{r['drift_frac']:>8.2f} {r['half_frac']:>7.2f} "
            f"{g(r['cont_frac'],7)} {r['N_eff']:>7.0f} "
            f"{g(r['dcn'],8,'8.4f')} {r['msd_end']:>8.0f} "
            f"{r['len_ns']:>5.1f}  {r['verdict']}{note}")


def make_plots(results, blocks, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    prods = [b for b, _ in pair_production(blocks)]
    for prod, r in zip(prods, results):
        fig, ax = plt.subplots(2, 2, figsize=(9, 6))
        fig.suptitle(f"T = {r['T']:.0f} K   verdict: {r['verdict']}")
        ax[0, 0].plot(prod.t_ns, prod.press); ax[0, 0].set_ylabel("P (kbar)")
        ax[0, 0].axhline(r["meanP"], color="r", lw=0.8, ls="--")
        ax[0, 1].plot(prod.t_ns, prod.pe); ax[0, 1].set_ylabel("PE (eV)")
        if prod.cn is not None:
            ax[1, 0].plot(prod.t_ns, prod.cn)
        ax[1, 0].set_ylabel("CN Si-O"); ax[1, 0].set_xlabel("t (ns)")
        if prod.msd is not None:
            # MSD log-log vs time in ps (smallest unit) to resolve the regimes:
            #   ballistic  MSD ~ t^2   (short time, free flight)
            #   caging     MSD ~ flat  (transitional plateau, low T)
            #   diffusive  MSD ~ t^1   (long time)
            tps = prod.t_ns * 1000.0
            m = (tps > 0) & (prod.msd > 0)
            tp, mp = tps[m], prod.msd[m]
            ax[1, 1].loglog(tp, mp, lw=1.0)
            # slope guides: t^2 anchored at the start, t^1 anchored at the end
            ax[1, 1].loglog(tp, mp[0] * (tp / tp[0]) ** 2, "k--", lw=0.7,
                            label=r"$t^2$ ballistic")
            ax[1, 1].loglog(tp, mp[-1] * (tp / tp[-1]), "r--", lw=0.7,
                            label=r"$t^1$ diffusive")
            ax[1, 1].legend(fontsize=7, loc="upper left")
        ax[1, 1].set_ylabel("Si MSD (A^2)"); ax[1, 1].set_xlabel("t (ps)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"eq_{r['T']:.0f}K.png"), dpi=110)
        plt.close(fig)
    return out_dir


def write_csv(results, path):
    keys = ["T", "meanP", "semP", "stdP", "slope", "drift_frac", "half_frac",
            "cont_frac", "tau", "N_eff", "dcn", "msd_end", "msd_slope",
            "len_ns", "complete", "verdict"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            f.write(",".join(
                f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                for k in keys) + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def resolve_log(path):
    if os.path.isdir(path):
        cand = os.path.join(path, "log.lammps")
        if os.path.isfile(cand):
            return cand
        raise SystemExit(f"No log.lammps in directory {path}")
    if os.path.isfile(path):
        return path
    raise SystemExit(f"Not found: {path}")


def main(argv=None):
    global DT_PS
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="rho-XXX directory or a log.lammps file")
    p.add_argument("--csv", help="also write the per-T table to this CSV file")
    p.add_argument("--plot", action="store_true",
                   help="write P/PE/CN/MSD-vs-time PNGs per T to <dir>/eq_plots")
    p.add_argument("--dt", type=float, default=DT_PS,
                   help=f"timestep in ps for the ns axis (default {DT_PS})")
    args = p.parse_args(argv)
    DT_PS = args.dt

    log_file = resolve_log(args.path)
    blocks = parse_log(log_file)
    prods = list(pair_production(blocks))
    if not prods:
        raise SystemExit(
            "No production blocks found (looked for runs preceded by a "
            "'fix ... ave/time'). Is this an isochore log?")

    results = [assess(prod, eq) for prod, eq in prods]

    label = os.path.basename(os.path.dirname(os.path.abspath(log_file))) \
        or os.path.basename(os.path.abspath(log_file))
    print(f"# Equilibration check: {label}  ({log_file})")
    print(f"# slow variable = pressure; *_/s = fraction of std(P)={'':s}"
          f" thresholds drift<{EQ_DRIFT_FRAC_MAX} half/cont<{EQ_SHIFT_FRAC_MAX}"
          f" N_eff>{EQ_MIN_NEFF:.0f} |dCN|<{CRYSTAL_DCN}")
    print(_HDR)
    for r in results:
        print(fmt_row(r))

    n_bad = sum(1 for r in results if r["verdict"] in ("NOT-EQ", "INCOMPLETE"))
    n_warn = sum(1 for r in results if r["verdict"] in ("WARN", "CRYSTAL?"))
    print(f"# {len(results)} temperatures: "
          f"{len(results) - n_bad - n_warn} OK, {n_warn} WARN/CRYSTAL?, "
          f"{n_bad} NOT-EQ/INCOMPLETE")

    if args.csv:
        write_csv(results, args.csv)
        print(f"# wrote {args.csv}")
    if args.plot:
        out = os.path.join(os.path.dirname(os.path.abspath(log_file)), "eq_plots")
        make_plots(results, blocks, out)
        print(f"# wrote plots -> {out}")

    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
