#!/usr/bin/env python3
"""Average thermo_analyzer.py analysis outputs over repeat samples (e.g. 1a..1e).

Each repeat runs the same compression protocol, so analysis rows line up by
index (row k = the same compression step in every sample). For every analysis
file shared by the samples this averages the numeric columns row-by-row and
writes, into the output directory:

    <file>        sample mean   (same format/columns as the per-sample file)
    <file>.err    standard error of the mean = std / sqrt(N)

The sample-to-sample std is a *real* error bar (independent runs), better than a
single run's internal fluctuation noise.

Alignment: rows are matched by index. If samples disagree on row count the
common minimum is used and a warning is printed (check those files). The
equilibration ``status`` column becomes ``n_pass`` (how many samples PASS that
point).

Examples:
    ./average_samples.py                       # base=., samples 1a..1e -> average/
    ./average_samples.py --base /path/to/run --samples 1a 1b 1c
    ./average_samples.py --out averaged
"""

import argparse
import os
import sys

import numpy as np

DEFAULT_SAMPLES = ["1a", "1b", "1c", "1d", "1e"]

# Multi-column tables (averaged column-wise, formatting preserved per file).
TABLE_FILES = ["thermo", "fluctuations", "equilibration"]
# Single-value-per-line property files.
SCALAR_FILES = [
    "pressure", "temperature", "volume", "boxes",
    "ekin", "epot", "etot", "outputs",
]

_COL_W = 13  # fixed width for aligned tables (matches thermo_analyzer.py)


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #

def load(path):
    """Return (header_cells_or_None, list_of_split_data_rows)."""
    header = None
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header = s.lstrip("#").split()
                continue
            rows.append(line.split())
    return header, rows


def to_float(tok):
    """Parse a cell to float; handles the 'dens<value>' prefix; NaN otherwise."""
    try:
        return float(tok)
    except ValueError:
        if tok.startswith("dens"):
            try:
                return float(tok[4:])
            except ValueError:
                return float("nan")
        return float("nan")


def column_kinds(sample_row):
    """Classify each column from a sample data row: 'status' | 'dens' | 'num'."""
    kinds = []
    for tok in sample_row:
        if tok in ("PASS", "WARN"):
            kinds.append("status")
        elif tok.startswith("dens"):
            kinds.append("dens")
        else:
            kinds.append("num")
    return kinds


def average_table(datasets):
    """Average a list of datasets (each a list of split rows) column-wise.

    Returns (kinds, mean_rows, sem_rows, npass_rows, nrow, counts) where
    mean/sem rows hold floats (NaN for status columns) and npass_rows holds the
    PASS count per status column (or None).
    """
    counts = [len(d) for d in datasets]
    nrow = min(counts)
    ncol = len(datasets[0][0])
    kinds = column_kinds(datasets[0][0])
    n_samples = len(datasets)

    mean_rows, sem_rows, npass_rows = [], [], []
    for r in range(nrow):
        mrow, erow, prow = [], [], []
        for c in range(ncol):
            if kinds[c] == "status":
                p = sum(1 for d in datasets if d[r][c] == "PASS")
                mrow.append(float("nan"))
                erow.append(float("nan"))
                prow.append(p)
            else:
                vals = np.array([to_float(d[r][c]) for d in datasets])
                valid = vals[~np.isnan(vals)]
                m = float(np.mean(valid)) if valid.size else float("nan")
                std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
                sem = std / np.sqrt(valid.size) if valid.size else float("nan")
                mrow.append(m)
                erow.append(sem)
                prow.append(None)
        mean_rows.append(mrow)
        sem_rows.append(erow)
        npass_rows.append(prow)
    return kinds, mean_rows, sem_rows, npass_rows, nrow, counts


# --------------------------------------------------------------------------- #
# Formatting (per file, to stay drop-in compatible)                          #
# --------------------------------------------------------------------------- #

def _fmt_thermo_cell(c, kind, v):
    # thermo columns: pressure temperature volume density box ekin epot etot
    return f"{v:.3f}" if c == 3 else f"{v:.6f}"


def write_thermo(path, kinds, mean_rows):
    with open(path, "w") as f:
        for row in mean_rows:
            f.write("\t".join(_fmt_thermo_cell(c, kinds[c], v)
                              for c, v in enumerate(row)) + "\n")


def write_thermo_err(path, kinds, sem_rows):
    with open(path, "w") as f:
        for row in sem_rows:
            f.write("\t".join(_fmt_thermo_cell(c, kinds[c], v)
                              for c, v in enumerate(row)) + "\n")


def _aligned(cells, header=False):
    body = "  ".join(f"{c:>{_COL_W}}" for c in cells)
    return ("# " if header else "  ") + body + "\n"


def write_aligned(path, header, kinds, rows, npass_rows, n_samples, err=False):
    """Write fluctuations/equilibration mean or err in aligned columns."""
    out_header = None
    if header is not None:
        out_header = [
            f"n_pass/{n_samples}" if (i < len(kinds) and kinds[i] == "status")
            else name
            for i, name in enumerate(header)
        ]
    with open(path, "w") as f:
        if out_header is not None:
            f.write(_aligned(out_header, header=True))
        for r, row in enumerate(rows):
            cells = []
            for c, v in enumerate(row):
                if kinds[c] == "status":
                    cells.append("-" if err else str(npass_rows[r][c]))
                else:
                    cells.append(f"{v:.6g}")
            f.write(_aligned(cells))


def write_scalar(path, name, kinds, rows, err=False):
    """Write a single-column property file (mean or err)."""
    is_dens = (name == "outputs")
    with open(path, "w") as f:
        for row in rows:
            v = row[0]
            if is_dens and not err:
                f.write(f"dens{v:.3f}\n")
            else:
                f.write(f"{v:.6f}\n")


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def process_file(name, paths, out_dir, n_samples):
    """Average one analysis file across samples; return a status string."""
    datasets = []
    for p in paths:
        _, rows = load(p)
        datasets.append(rows)
    header, _ = load(paths[0])

    if any(len(d) == 0 for d in datasets):
        return f"{name}: skipped (empty in some sample)"

    kinds, mean_rows, sem_rows, npass_rows, nrow, counts = average_table(datasets)

    out = os.path.join(out_dir, name)
    err_out = out + ".err"
    if name == "thermo":
        write_thermo(out, kinds, mean_rows)
        write_thermo_err(err_out, kinds, sem_rows)
    elif name in ("fluctuations", "equilibration"):
        write_aligned(out, header, kinds, mean_rows, npass_rows, n_samples)
        write_aligned(err_out, header, kinds, sem_rows, npass_rows, n_samples,
                      err=True)
    else:  # scalar property files
        write_scalar(out, name, kinds, mean_rows)
        write_scalar(err_out, name, kinds, sem_rows, err=True)

    note = ""
    if len(set(counts)) > 1:
        note = f"  [WARNING: row counts {counts} -> truncated to {nrow}]"
    return f"{name}: {nrow} rows x {n_samples} samples{note}"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", default=".",
                   help="directory containing the sample dirs (default: .)")
    p.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES,
                   help="sample dir names (default: 1a 1b 1c 1d 1e)")
    p.add_argument("--analysis-subdir", default="analysis",
                   help="analysis dir inside each sample (default: analysis)")
    p.add_argument("--out", default="average/analysis",
                   help="output dir for averaged files (default: average/analysis)")
    args = p.parse_args(argv)

    out_dir = os.path.join(args.base, args.out) if not os.path.isabs(args.out) \
        else args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"Averaging {len(args.samples)} samples: {', '.join(args.samples)}")
    print(f"Output -> {out_dir}\n")

    any_done = False
    for name in TABLE_FILES + SCALAR_FILES:
        paths = [
            os.path.join(args.base, s, args.analysis_subdir, name)
            for s in args.samples
        ]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            print(f"{name}: skipped (missing in "
                  f"{len(missing)}/{len(paths)} samples)")
            continue
        try:
            print(process_file(name, paths, out_dir, len(args.samples)))
            any_done = True
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{name}: ERROR {e}", file=sys.stderr)

    if not any_done:
        print("Nothing averaged (no shared files found).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
