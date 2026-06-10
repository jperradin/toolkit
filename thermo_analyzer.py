#!/usr/bin/env python3
"""Thermodynamic data analyzer for LAMMPS output.

A simulation at each density is run in two parts that share the same log:

    part A : steps 0 -> 31250   (block whose Step counter restarts at 0)
    part B : steps 31250 -> 62500

A new density point starts every time the Step counter resets to 0. For each
point both the A and B blocks are written verbatim to ``thermo/`` as
``thermo-<n>A.dat`` / ``thermo-<n>B.dat``; only the B blocks are averaged into
the summary/per-property files in ``analysis/``.

System type, atom count and the A/B split are auto-detected from the log unless
overridden on the command line.

Examples:
    ./thermo_analyzer.py out-load-npt-96k-1a.o4499287
    ./thermo_analyzer.py log.lammps --system SiO2 --n-atoms 96000
    ./thermo_analyzer.py --list
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Property tables                                                             #
# --------------------------------------------------------------------------- #

# Density conversion factors keyed by (system, n_atoms): density = factor / V_cm3.
DENSITY_FACTORS = {
    "SiO2": {
        1008: 3.352307451024e-20,
        3024: 1.0056922353072e-19,
        8064: 2.6818459608192e-19,
        15120: 5.028461176536e-19,
        27216: 9.0512301177648e-19,
        96000: 3.19267376288e-18,
        1056000: 3.511941139168e-17,
    },
    "NSx": {
        1080: 1.005571132125e-19,
        3000: 3.62005607565e-20,
        1350: 4.51799557146e-20,
        13500: 4.51799557146e-19,
    },
}

ANGSTROM_TO_CM = 1.0e-8
PRESSURE_SCALE = 0.0001  # bar -> GPa
RESET_STEP = 0  # Step value that marks the start of a new density point (part A)
DEFAULT_STEPS_PER_PART = 31250  # fallback run length per A/B part

# Metal units (eV, Angstrom, bar, K) constants for fluctuation analysis.
KB_METAL = 8.617333262e-5  # Boltzmann constant, eV/K
EVA3_TO_GPA = 160.21766208  # eV/Angstrom^3 -> GPa


def available_systems():
    return sorted(DENSITY_FACTORS)


def available_atom_counts(system):
    return sorted(DENSITY_FACTORS[system])


def density_factor(system, n_atoms):
    """Look up the density conversion factor for ``(system, n_atoms)``."""
    if system not in DENSITY_FACTORS:
        raise ValueError(
            f"System '{system}' not implemented. "
            f"Known: {', '.join(available_systems())}."
        )
    table = DENSITY_FACTORS[system]
    if n_atoms not in table:
        raise ValueError(
            f"Atom count {n_atoms} for system '{system}' not implemented. "
            f"Known: {', '.join(map(str, sorted(table)))}."
        )
    return table[n_atoms]


def box_to_density(box_length, factor):
    """Convert a cubic box edge length (angstrom) to a density (g/cm^3)."""
    return factor / (box_length * ANGSTROM_TO_CM) ** 3


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #

# Thermo columns we keep.
_NEEDED = ("Press", "Temp", "Volume", "Lx", "KinEng", "PotEng", "TotEng")
_ATOMS_RE = re.compile(r"\b(\d+)\s+atoms\b")
_STEPS_RE = re.compile(r"\bfor\s+(\d+)\s+steps\b")


@dataclass
class Block:
    """One thermo table (one ``run``), with per-step samples.

    ``raw_first_step`` / ``raw_last_step`` are the untrimmed Step bounds, used to
    decide whether the run reached its target end (a crashed run stops short).
    """

    step: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    volume: np.ndarray
    box: np.ndarray
    ekin: np.ndarray
    epot: np.ndarray
    etot: np.ndarray
    density: np.ndarray = field(default=None)
    raw_first_step: int = 0
    raw_last_step: int = 0
    target_end: int = 0

    @property
    def start_step(self):
        return int(self.step[0])

    @property
    def is_complete(self):
        """True if the run reached its target end step (did not crash)."""
        return self.raw_last_step >= self.target_end

    def means(self):
        return {
            "pressure": float(np.mean(self.pressure)),
            "temperature": float(np.mean(self.temperature)),
            "volume": float(np.mean(self.volume)),
            "density": float(np.mean(self.density)),
            "box": float(np.mean(self.box)),
            "ekin": float(np.mean(self.ekin)),
            "epot": float(np.mean(self.epot)),
            "etot": float(np.mean(self.etot)),
        }


@dataclass
class Point:
    """A density point: its A block and B block (either may be missing)."""

    index: int
    a: Block = None
    b: Block = None


def detect_n_atoms(lines):
    for line in lines:
        m = _ATOMS_RE.search(line)
        if m:
            return int(m.group(1))
    return None


def detect_steps_per_part(lines):
    """Return the per-run step count from a 'Loop time ... for N steps' line.

    Uses the most frequent value across completed runs; None if none found.
    """
    counts = {}
    for line in lines:
        if "Loop time" not in line:
            continue
        m = _STEPS_RE.search(line)
        if m:
            n = int(m.group(1))
            counts[n] = counts.get(n, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def detect_system(lines):
    text = "\n".join(lines[:200]).lower()
    if "group gna" in text or " na " in text or "sodium" in text:
        return "NSx"
    if "group gsi" in text or "group go" in text or "silic" in text:
        return "SiO2"
    return None


def _find_header(lines):
    for i, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0] == "Step":
            return i, parts
    return None, None


def _is_data_row(parts, n_cols):
    if len(parts) != n_cols:
        return False
    try:
        float(parts[0])
    except ValueError:
        return False
    return True


def parse_log(
    log_file, system=None, n_atoms=None, start=None, end=None, steps_per_part=None
):
    """Parse ``log_file`` into Blocks grouped into density Points.

    Returns (points, meta) where meta carries the resolved
    system/n_atoms/factor/steps_per_part.
    """
    with open(log_file) as fh:
        lines = fh.readlines()

    if n_atoms is None:
        n_atoms = detect_n_atoms(lines)
        if n_atoms is None:
            raise ValueError("Could not auto-detect atom count; pass --n-atoms.")
    if system is None:
        system = detect_system(lines)
        if system is None:
            raise ValueError("Could not auto-detect system; pass --system.")
    if steps_per_part is None:
        steps_per_part = detect_steps_per_part(lines) or DEFAULT_STEPS_PER_PART

    factor = density_factor(system, n_atoms)

    header_idx, columns = _find_header(lines)
    if header_idx is None:
        raise ValueError("No thermo table ('Step ...') found in log.")
    idx = {name: columns.index(name) for name in _NEEDED}
    n_cols = len(columns)

    blocks = []
    rows = None

    def flush():
        if not rows:
            return
        arr = np.array(rows, dtype=float)
        # Untrimmed bounds decide completeness (a crashed run stops short).
        raw_first = int(arr[0, 0])
        raw_last = int(arr[-1, 0])
        steps = arr[:, 0]
        if start is not None or end is not None:
            lo = -np.inf if start is None else start
            hi = np.inf if end is None else end
            mask = (steps >= lo) & (steps <= hi)
            if not mask.any():
                return
            arr = arr[mask]
            steps = arr[:, 0]
        block = Block(
            step=steps,
            pressure=arr[:, idx["Press"]] * PRESSURE_SCALE,
            temperature=arr[:, idx["Temp"]],
            volume=arr[:, idx["Volume"]],
            box=arr[:, idx["Lx"]],
            ekin=arr[:, idx["KinEng"]],
            epot=arr[:, idx["PotEng"]],
            etot=arr[:, idx["TotEng"]],
            raw_first_step=raw_first,
            raw_last_step=raw_last,
            target_end=raw_first + steps_per_part,
        )
        block.density = box_to_density(block.box, factor)
        blocks.append(block)

    for line in lines[header_idx:]:
        parts = line.split()
        if parts and parts[0] == "Step":
            flush()
            rows = []
            continue
        if rows is not None and _is_data_row(parts, n_cols):
            rows.append([float(p) for p in parts])
        elif rows:
            flush()
            rows = None
    flush()

    points = _group_points(blocks)
    meta = {
        "system": system,
        "n_atoms": n_atoms,
        "factor": factor,
        "steps_per_part": steps_per_part,
    }
    return points, meta


def report_incomplete(points):
    """Return a list of crashed/short blocks as (point_index, part, last, target)."""
    bad = []
    for pt in points:
        for part, block in (("A", pt.a), ("B", pt.b)):
            if block is not None and not block.is_complete:
                bad.append((pt.index, part, block.raw_last_step, block.target_end))
    return bad


def _group_points(blocks):
    """Group blocks into Points: a Step reset (==RESET_STEP) opens a new point.

    The opening block is part A; the next block is part B.
    """
    points = []
    current = None
    for block in blocks:
        if block.start_step == RESET_STEP:
            current = Point(index=len(points) + 1, a=block)
            points.append(current)
        elif current is not None and current.b is None:
            current.b = block
        else:
            # Orphan B-style block with no preceding A: start a fresh point.
            current = Point(index=len(points) + 1, b=block)
            points.append(current)
    return points


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #

_BLOCK_HEADER = (
    "# Step\tPressure\tTemperature\tVolume\tDensity\tLbox\tEkin\tEpot\tEtot\n"
)
_SUMMARY_KEYS = (
    "pressure", "temperature", "volume", "density", "box", "ekin", "epot", "etot",
)
_PROPERTY_FILES = {
    "pressure": "pressure",
    "temperature": "temperature",
    "volume": "volume",
    "boxes": "box",
    "outputs": "density",
    "ekin": "ekin",
    "epot": "epot",
    "etot": "etot",
}


def count_summary_lines(analysis_dir):
    """Number of B-average rows already in analysis_dir/thermo (0 if absent).

    This is the count of valid density points recorded so far; --append
    continues block numbering from here so indices stay aligned with the
    averaged analysis rows.
    """
    path = os.path.join(analysis_dir, "thermo")
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _write_block(block, path):
    with open(path, "w") as out:
        out.write(_BLOCK_HEADER)
        for i in range(len(block.step)):
            out.write(
                f"{block.step[i]:.2f}\t{block.pressure[i]:.6f}\t"
                f"{block.temperature[i]:.6f}\t{block.volume[i]:.6f}\t"
                f"{block.density[i]:.3f}\t{block.box[i]:.6f}\t{block.ekin[i]:.6f}\t"
                f"{block.epot[i]:.6f}\t{block.etot[i]:.6f}\n"
            )


def write_block_files(points, block_dir, keep_incomplete=False, index_offset=0):
    """Write thermo-<n>A.dat / thermo-<n>B.dat for every point into block_dir.

    Incomplete (crashed) blocks are skipped unless ``keep_incomplete`` is set.
    ``index_offset`` shifts the <n> numbering (used by --append to continue past
    existing files instead of overwriting them).
    """
    os.makedirs(block_dir, exist_ok=True)
    for pt in points:
        n = pt.index + index_offset
        for part, block in (("A", pt.a), ("B", pt.b)):
            if block is None:
                continue
            if not block.is_complete and not keep_incomplete:
                continue
            _write_block(block, os.path.join(block_dir, f"thermo-{n}{part}.dat"))


def _format_summary(m):
    return [
        f"{m[k]:.3f}" if k == "density" else f"{m[k]:.6f}" for k in _SUMMARY_KEYS
    ]


# Fixed column width for the aligned fluctuations / equilibration tables. Fixed
# (not adaptive) so --append rows line up with rows written earlier.
_COL_W = 13


def _aligned_line(cells, header=False):
    """Render cells as fixed-width right-justified columns.

    Header lines start with '# ', data lines with two spaces, so both share the
    same column positions.
    """
    body = "  ".join(f"{c:>{_COL_W}}" for c in cells)
    return ("# " if header else "  ") + body + "\n"


def write_summary(points, analysis_dir, append=False, echo=False, keep_incomplete=False):
    """Average the B block of each point into summary/per-property files.

    Points whose B block is missing or incomplete (crashed) are skipped unless
    ``keep_incomplete`` is set. With ``append`` the rows are added to the
    existing files (summary and per-property) instead of overwriting them.
    """
    os.makedirs(analysis_dir, exist_ok=True)
    means = [
        pt.b.means()
        for pt in points
        if pt.b is not None and (pt.b.is_complete or keep_incomplete)
    ]

    if echo:
        for m in means:
            print("\t".join(_format_summary(m)))

    file_mode = "a" if append else "w"

    with open(os.path.join(analysis_dir, "thermo"), file_mode) as f:
        for m in means:
            f.write("\t".join(_format_summary(m)) + "\n")

    for filename, key in _PROPERTY_FILES.items():
        with open(os.path.join(analysis_dir, filename), file_mode) as f:
            for m in means:
                value = m[key]
                if filename == "outputs":
                    f.write(f"dens{value:.3f}\n")
                else:
                    f.write(f"{value:.6f}\n")
    return len(means)


# --------------------------------------------------------------------------- #
# Fluctuation analysis (NPT, metal units)                                     #
# --------------------------------------------------------------------------- #

# Output columns for the fluctuations table, in order.
_FLUCT_COLUMNS = (
    "density",      # g/cm^3
    "T",            # K
    "P",            # GPa
    "B_T",          # isothermal bulk modulus, GPa
    "B_S",          # adiabatic bulk modulus, GPa
    "kappa_T",      # isothermal compressibility, 1/GPa
    "alpha_P",      # thermal expansion, 1/K
    "dPdT_V",       # thermal pressure coeff, GPa/K
    "Cp_atom",      # isobaric heat capacity, k_B/atom
    "Cv_atom",      # isochoric heat capacity, k_B/atom
    "gamma",        # Gruneisen parameter
    "sound",        # speed of sound, m/s
)


def compute_fluctuations(block, n_atoms):
    """Derive NPT response functions from one B block's fluctuations.

    Metal units in (eV, Angstrom, bar->GPa, K); returns a dict keyed by
    ``_FLUCT_COLUMNS``. Population variance (ddof=0) is used.
    """
    T = float(np.mean(block.temperature))      # K
    V = float(np.mean(block.volume))           # A^3
    rho = float(np.mean(block.density))        # g/cm^3
    P = float(np.mean(block.pressure))         # GPa

    # Enthalpy for NPT fluctuations must use the CONSTANT target pressure, not
    # the instantaneous virial pressure (whose barostat fluctuations would
    # otherwise dominate Var(H)). Use the block-mean pressure as P_ext.
    P_ext = P / EVA3_TO_GPA                                    # GPa -> eV/A^3
    H = block.etot + P_ext * block.volume                     # eV

    var_V = float(np.var(block.volume))                        # A^6
    var_H = float(np.var(H))                                   # eV^2
    cov_VH = float(np.mean(block.volume * H) - V * np.mean(H))  # A^3 * eV

    kbt = KB_METAL * T
    # Isothermal compressibility kappa_T [A^3/eV] = Var(V)/(kB T V).
    kappa_T_evA3 = var_V / (kbt * V)
    B_T_evA3 = 1.0 / kappa_T_evA3                              # eV/A^3
    B_T = B_T_evA3 * EVA3_TO_GPA                               # GPa
    kappa_T = 1.0 / B_T                                        # 1/GPa

    Cp_total = var_H / (KB_METAL * T * T)                      # eV/K
    alpha_P = cov_VH / (KB_METAL * T * T * V)                  # 1/K
    # Cv = Cp - T V alpha_P^2 / kappa_T, with 1/kappa_T = B_T_evA3.
    Cv_total = Cp_total - T * V * alpha_P ** 2 * B_T_evA3      # eV/K
    dPdT_V = alpha_P * B_T                                     # GPa/K

    if Cv_total > 0:
        B_S = B_T * Cp_total / Cv_total                        # GPa
        gamma = alpha_P * B_T_evA3 * V / Cv_total
    else:
        B_S = float("nan")
        gamma = float("nan")
    sound = 1000.0 * np.sqrt(B_S / rho) if (B_S > 0 and rho > 0) else float("nan")

    return {
        "density": rho,
        "T": T,
        "P": P,
        "B_T": B_T,
        "B_S": B_S,
        "kappa_T": kappa_T,
        "alpha_P": alpha_P,
        "dPdT_V": dPdT_V,
        "Cp_atom": Cp_total / (n_atoms * KB_METAL),
        "Cv_atom": Cv_total / (n_atoms * KB_METAL),
        "gamma": gamma,
        "sound": sound,
    }


def write_fluctuations(
    points, analysis_dir, n_atoms, append=False, keep_incomplete=False
):
    """Write per-B-block fluctuation response functions to analysis_dir/fluctuations.

    One row per valid density point, aligned with the ``thermo`` summary. With
    ``append`` rows are added; the header is written only when the file is new.
    """
    os.makedirs(analysis_dir, exist_ok=True)
    rows = [
        compute_fluctuations(pt.b, n_atoms)
        for pt in points
        if pt.b is not None and (pt.b.is_complete or keep_incomplete)
    ]

    path = os.path.join(analysis_dir, "fluctuations")
    write_header = not (append and os.path.isfile(path))
    with open(path, "a" if append else "w") as f:
        if write_header:
            f.write(_aligned_line(_FLUCT_COLUMNS, header=True))
        for r in rows:
            f.write(_aligned_line([f"{r[c]:.6g}" for c in _FLUCT_COLUMNS]))
    return len(rows)


# --------------------------------------------------------------------------- #
# Equilibration check (NPT, slow variable = volume)                           #
# --------------------------------------------------------------------------- #

# Equilibration is judged on the PHYSICAL scale (fraction of the fluctuation
# width std(V)), not statistical SEM: with thousands of samples a SEM z-test
# flags physically negligible drift, so it is not used for the verdict.
EQ_DRIFT_FRAC_MAX = 1.0    # |total V drift| must stay below this * std(V)
EQ_SHIFT_FRAC_MAX = 0.5    # half-to-half / A->B mean shift, as a fraction of std(V)
EQ_MIN_NEFF = 20.0         # require at least this many decorrelated samples

_EQ_COLUMNS = (
    "density",       # g/cm^3
    "T",             # K
    "drift_frac",    # |total V drift| / std(V)        (< 1 good)
    "half_frac",     # |<V>_1st - <V>_2nd| / std(V)    (< 0.5 good)
    "cont_frac",     # |<V>_A_tail - <V>_B| / std(V)   (A->B continuity; < 0.5 good)
    "tau_int",       # integrated autocorr time of V (samples)
    "N_eff",         # effective independent samples of V in B block
    "status",        # PASS / WARN
)


def integrated_autocorr_time(x):
    """Integrated autocorrelation time tau_int = 1 + 2*sum rho(k) (Sokal window).

    Returns samples; >= 1. N_eff = N / tau_int, SEM = std / sqrt(N_eff).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return 1.0
    xc = x - x.mean()
    var = float(np.dot(xc, xc) / n)
    if var == 0.0:
        return 1.0
    acf = np.correlate(xc, xc, mode="full")[n - 1:] / (var * n)  # normalized
    tau = 1.0
    for k in range(1, n):
        rho = acf[k]
        if rho <= 0.0:  # initial-positive-sequence cutoff
            break
        tau += 2.0 * rho
        if k >= 6.0 * tau:  # Sokal automatic window
            break
    return max(tau, 1.0)


def _mean_sem(x):
    """Mean and autocorrelation-corrected SEM of a series."""
    x = np.asarray(x, dtype=float)
    n = x.size
    tau = integrated_autocorr_time(x)
    n_eff = max(n / tau, 1.0)
    sem = float(x.std()) / np.sqrt(n_eff)
    return float(x.mean()), sem, tau, n_eff


def compute_equilibration(point):
    """Assess whether a density point's B (production) block is equilibrated.

    Uses volume (the slow NPT variable): drift, first/second-half agreement,
    A->B continuity, and effective sample count. Returns a dict keyed by
    ``_EQ_COLUMNS`` with a PASS/WARN ``status``.
    """
    b = point.b
    V = b.volume
    mV, _, tau, n_eff = _mean_sem(V)
    std_V = float(V.std())

    # Linear drift over the block, relative to the fluctuation width.
    t = b.step.astype(float)
    slope = float(np.polyfit(t, V, 1)[0])
    total_drift = slope * (t[-1] - t[0])
    drift_frac = abs(total_drift) / std_V if std_V > 0 else 0.0

    # First vs second half of B, relative to std(V).
    half = V.size // 2
    half_frac = (
        abs(float(V[:half].mean()) - float(V[half:].mean())) / std_V
        if std_V > 0 else 0.0
    )

    # A->B continuity: A's tail (2nd half) should match B's mean.
    if point.a is not None and point.a.volume.size >= 2 and std_V > 0:
        ah = point.a.volume.size // 2
        cont_frac = abs(float(point.a.volume[ah:].mean()) - mV) / std_V
    else:
        cont_frac = float("nan")

    drift_ok = drift_frac < EQ_DRIFT_FRAC_MAX
    half_ok = half_frac < EQ_SHIFT_FRAC_MAX
    tau_ok = n_eff >= EQ_MIN_NEFF
    cont_ok = np.isnan(cont_frac) or cont_frac < EQ_SHIFT_FRAC_MAX
    status = "PASS" if (drift_ok and half_ok and tau_ok and cont_ok) else "WARN"

    return {
        "density": float(np.mean(b.density)),
        "T": float(np.mean(b.temperature)),
        "drift_frac": drift_frac,
        "half_frac": half_frac,
        "cont_frac": cont_frac,
        "tau_int": tau,
        "N_eff": n_eff,
        "status": status,
    }


def write_equilibration(points, analysis_dir, append=False, keep_incomplete=False):
    """Write per-point equilibration diagnostics to analysis_dir/equilibration.

    One row per valid density point, aligned with the other tables. Returns
    (n_rows, n_warn).
    """
    os.makedirs(analysis_dir, exist_ok=True)
    rows = [
        compute_equilibration(pt)
        for pt in points
        if pt.b is not None and (pt.b.is_complete or keep_incomplete)
    ]

    path = os.path.join(analysis_dir, "equilibration")
    write_header = not (append and os.path.isfile(path))
    with open(path, "a" if append else "w") as f:
        if write_header:
            f.write(_aligned_line(_EQ_COLUMNS, header=True))
        for r in rows:
            cells = [
                r[c] if c == "status" else f"{r[c]:.6g}" for c in _EQ_COLUMNS
            ]
            f.write(_aligned_line(cells))
    n_warn = sum(1 for r in rows if r["status"] == "WARN")
    return len(rows), n_warn


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _list_systems():
    print("Tabulated systems and atom counts:")
    for system in available_systems():
        counts = ", ".join(map(str, available_atom_counts(system)))
        print(f"  {system}: {counts}")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("log_file", nargs="?", help="LAMMPS log file to process")
    p.add_argument("--system", help="system type (auto-detected if omitted)")
    p.add_argument(
        "--n-atoms", type=int, help="atom count (auto-detected if omitted)"
    )
    p.add_argument("--start", type=float, help="lower step bound to trim each block")
    p.add_argument("--end", type=float, help="upper step bound to trim each block")
    p.add_argument(
        "--steps-per-part", type=int,
        help="run length per A/B part used for the crash check "
        "(auto-detected from the log if omitted)",
    )
    p.add_argument(
        "--keep-incomplete", action="store_true",
        help="process crashed/short blocks instead of skipping them",
    )
    p.add_argument(
        "--analysis-dir",
        default="./analysis",
        help="output dir for averaged B summary/property files (default: ./analysis)",
    )
    p.add_argument(
        "--block-dir",
        default="./thermo",
        help="output dir for per-block A/B .dat files (default: ./thermo)",
    )
    p.add_argument(
        "--append", action="store_true",
        help="append this log's results to existing analysis/ and thermo/ output "
        "(continues block numbering, does not overwrite) -- use when rerunning a "
        "crashed tail in a separate log",
    )
    p.add_argument(
        "--fluctuations", action="store_true",
        help="also compute NPT response functions (bulk modulus, Cp/Cv, thermal "
        "expansion, Gruneisen, sound speed) from each B block's fluctuations "
        "(metal units) -> analysis/fluctuations",
    )
    p.add_argument(
        "--equilibration", action="store_true",
        help="check each B block for equilibration (V drift, half agreement, "
        "A->B continuity, autocorrelation/N_eff) -> analysis/equilibration",
    )
    p.add_argument(
        "--quiet", action="store_true", help="do not echo B-block means to stdout"
    )
    p.add_argument(
        "--list", action="store_true",
        help="list tabulated systems and atom counts, then exit",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _list_systems()
        return 0
    if not args.log_file:
        parser.error("log_file is required (or use --list)")

    points, meta = parse_log(
        args.log_file,
        system=args.system,
        n_atoms=args.n_atoms,
        start=args.start,
        end=args.end,
        steps_per_part=args.steps_per_part,
    )
    if not points:
        print("No thermo blocks found; nothing written.", file=sys.stderr)
        return 1

    bad = report_incomplete(points)
    if bad:
        verb = "kept" if args.keep_incomplete else "skipped"
        print(
            f"Crashed/short block(s) {verb} "
            f"(part end < target, run length {meta['steps_per_part']}):",
            file=sys.stderr,
        )
        for index, part, last, target in bad:
            print(
                f"  point {index}{part}: reached step {last} < {target}",
                file=sys.stderr,
            )

    offset = count_summary_lines(args.analysis_dir) if args.append else 0

    write_block_files(
        points, args.block_dir,
        keep_incomplete=args.keep_incomplete, index_offset=offset,
    )
    n_avg = write_summary(
        points, args.analysis_dir, append=args.append, echo=not args.quiet,
        keep_incomplete=args.keep_incomplete,
    )

    fluct_note = ""
    if args.fluctuations:
        n_fluct = write_fluctuations(
            points, args.analysis_dir, meta["n_atoms"],
            append=args.append, keep_incomplete=args.keep_incomplete,
        )
        fluct_note = f", {n_fluct} fluctuation row(s) -> {args.analysis_dir}/fluctuations"

    eq_note = ""
    if args.equilibration:
        n_eq, n_warn = write_equilibration(
            points, args.analysis_dir,
            append=args.append, keep_incomplete=args.keep_incomplete,
        )
        eq_note = (
            f", {n_eq} equilibration row(s) ({n_warn} WARN) "
            f"-> {args.analysis_dir}/equilibration"
        )

    action = f"appended after index {offset}" if args.append else "wrote"
    print(
        f"Processed {len(points)} density point(s) from {args.log_file} "
        f"[system={meta['system']}, n_atoms={meta['n_atoms']}, "
        f"steps/part={meta['steps_per_part']}]: "
        f"{action} blocks -> {args.block_dir}, {n_avg} B-average(s) -> {args.analysis_dir}"
        f"{fluct_note}{eq_note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
