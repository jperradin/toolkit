"""
average_export.py
-----------------
Averages N independent sample exports (produced by export.py) into a single
averaged_export/ directory.

Expected input layout (configurable via SAMPLE_GLOB):

    ./8a/analysis/export/
    ./8b/analysis/export/
    ...
    ./8e/analysis/export/

Two file types are handled:

1. Standard files  (<var>-<property>.dat  +  --errors.dat  +  --std.dat)
   Columns: x-axis | one column per connectivity type ...
   Output columns per connectivity type:
       mean, std (ddof=1), sem=std/sqrt(N), mean_err, combined=sqrt(sem²+mean_err²)

2. Concentration files  (concentrations/<name>.dat)
   Columns: concentration | value | std | error   (one file per connectivity type,
   no companion --errors / --std files).
   Output columns: concentration | mean | std | sem | mean_err | combined_err

Output is written to ./averaged_export/ mirroring the input structure.
"""

import os
import glob
import numpy as np
from natsort import natsorted

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Glob pattern relative to the working directory that finds each sample's
# export folder.  '*' matches 8a, 8b, 8c, …
SAMPLE_GLOB  = "*/analysis/export"
OUTPUT_DIR   = "./averaged_export"

# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def parse_dat_file(filepath):
    """
    Parse a .dat file produced by export.py.

    Returns:
        headers : list of str   – column names in order (index 0 = x-axis variable)
        data    : np.ndarray    – shape (n_cols, n_rows), data_matrix[col, row]
    """
    headers = []
    rows    = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                # e.g.  "# 1 density"  or  "# 2 HD"
                parts = line.lstrip("# ").split()
                if len(parts) >= 2:
                    headers.append(" ".join(parts[1:]))  # skip the index number
            else:
                values = line.split()
                if values:
                    rows.append([float(v) for v in values])

    if not rows:
        return headers, np.empty((len(headers), 0))

    data = np.array(rows).T   # shape: (n_cols, n_rows)
    return headers, data


def write_dat_file(filepath, headers, data, description=None):
    """
    Write a .dat file in the same format as export.py produces.

    headers     : list of str, len == data.shape[0]
    data        : np.ndarray, shape (n_cols, n_rows)
    description : optional str written as a comment block before the column headers
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        if description:
            for line in description.splitlines():
                f.write(f"# {line}\n")
            f.write("#\n")
        for i, h in enumerate(headers):
            f.write(f"# {i + 1} {h}\n")
        for j in range(data.shape[1]):
            f.write("\t".join(f"{data[col, j]:^14.6g}" for col in range(data.shape[0])))
            f.write("\n")


# ---------------------------------------------------------------------------
# Core averaging logic
# ---------------------------------------------------------------------------

def check_shapes(paths, label):
    """
    Verify all files in a group have the same shape.
    Raises ValueError with a clear message if they differ.
    """
    shapes = []
    for p in paths:
        _, d = parse_dat_file(p)
        shapes.append(d.shape)
    if len(set(shapes)) > 1:
        msg = f"Shape mismatch in {label}:\n"
        for p, s in zip(paths, shapes):
            msg += f"  {p}: {s}\n"
        raise ValueError(msg)


def is_concentration_file(rel_path):
    """
    Return True for files inside the concentrations/ subdirectory.
    These have a different column layout and no companion --errors / --std files.
    """
    parts = rel_path.replace("\\", "/").split("/")
    return "concentrations" in parts


def average_concentration_file(paths):
    """
    Average a concentration-property-connectivity.dat file across N samples.

    All samples follow the same thermo-mechanical pathway, so row i in
    sample A corresponds exactly to row i in sample B, etc.
    Averaging is done directly row by row — no alignment needed.

    Input columns:  concentration | value | std | error
    Output columns: concentration | mean | std | sem | mean_err | combined_err
    """
    N       = len(paths)
    samples = [parse_dat_file(p)[1] for p in paths]  # each: (4, n_rows)

    check_shapes(paths, paths[0])

    # x-axis: take from sample 0 (identical across all samples)
    x_col = samples[0][0]                              # (n_rows,)
    vals  = np.array([d[1] for d in samples])          # (N, n_rows): value column
    errs  = np.array([d[3] for d in samples])          # (N, n_rows): error column

    mean     = vals.mean(axis=0)
    std      = vals.std(axis=0, ddof=1)
    sem      = std / np.sqrt(N)
    mean_err = errs.mean(axis=0)
    combined = np.sqrt(sem**2 + mean_err**2)

    out_data    = np.vstack([x_col, mean, std, sem, mean_err, combined])
    out_headers = ["concentration", "mean", "std", "sem", "mean_err", "combined_err"]
    return out_headers, out_data


def average_file_group(base_paths, errors_paths, std_paths):
    """
    Average a group of standard export files across N samples.

    All samples follow the same thermo-mechanical pathway, so row i in
    sample A corresponds exactly to row i in sample B, etc.
    Averaging is done directly row by row — no alignment needed.

    Returns a dict with four keys, each mapping to (headers, data):
        'mean'   – mean of values across N samples,   shape (1 + n_ct, n_rows)
        'std'    – sample std (ddof=1),                shape (1 + n_ct, n_rows)
        'sem'    – std / sqrt(N),                      shape (1 + n_ct, n_rows)
        'errors' – sqrt(SEM² + mean_per-point_err²),  shape (1 + n_ct, n_rows)

    All four matrices share the same x-axis column 0 and the same headers:
        [variable_name, connectivity_type_1, connectivity_type_2, ...]
    """
    check_shapes(base_paths,   base_paths[0])
    check_shapes(errors_paths, errors_paths[0])

    sample_headers = []
    sample_base    = []
    sample_errors  = []

    for bp, ep, _sp in zip(base_paths, errors_paths, std_paths):
        h, d  = parse_dat_file(bp)
        sample_headers.append(h)
        sample_base.append(d)
        _, de = parse_dat_file(ep)
        sample_errors.append(de)

    headers_in         = sample_headers[0]
    variable_name      = headers_in[0]
    connectivity_types = headers_in[1:]
    N                  = len(base_paths)
    n_ct               = len(connectivity_types)
    n_rows             = sample_base[0].shape[1]

    # x-axis: take from sample 0 (identical across all samples)
    x_col = sample_base[0][0]   # (n_rows,)

    mat_mean   = np.zeros((1 + n_ct, n_rows))
    mat_std    = np.zeros((1 + n_ct, n_rows))
    mat_sem    = np.zeros((1 + n_ct, n_rows))
    mat_errors = np.zeros((1 + n_ct, n_rows))
    for m in (mat_mean, mat_std, mat_sem, mat_errors):
        m[0] = x_col

    for ci in range(n_ct):
        col = ci + 1   # col 0 is the x-axis

        vals     = np.array([d[col] for d in sample_base])    # (N, n_rows)
        errs     = np.array([d[col] for d in sample_errors])  # (N, n_rows)

        mean     = vals.mean(axis=0)
        std      = vals.std(axis=0, ddof=1)
        sem      = std / np.sqrt(N)
        mean_err = errs.mean(axis=0)
        combined = np.sqrt(sem**2 + mean_err**2)

        mat_mean  [col] = mean
        mat_std   [col] = std
        mat_sem   [col] = sem
        mat_errors[col] = combined

    headers = [variable_name] + list(connectivity_types)

    return {
        'mean':   (headers, mat_mean),
        'std':    (headers, mat_std),
        'sem':    (headers, mat_sem),
        'errors': (headers, mat_errors),
    }


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def find_sample_dirs():
    """Return natsorted list of export directories matching SAMPLE_GLOB."""
    dirs = natsorted(glob.glob(SAMPLE_GLOB))
    if not dirs:
        raise FileNotFoundError(
            f"No sample directories found matching pattern '{SAMPLE_GLOB}'. "
            "Check SAMPLE_GLOB in the configuration."
        )
    print(f"Found {len(dirs)} sample(s):")
    for d in dirs:
        print(f"  {d}")
    return dirs


def collect_base_files(sample_dirs):
    """
    Return a dict  { relative_filename: [path_in_sample_0, path_in_sample_1, …] }
    for every base .dat file (excluding --errors and --std companions)
    present in ALL samples.
    """
    # Gather files present in every sample
    sets = []
    for sd in sample_dirs:
        files = set()
        for root, _, fnames in os.walk(sd):
            for fname in fnames:
                if fname.endswith(".dat") and "--" not in fname:
                    rel = os.path.relpath(os.path.join(root, fname), sd)
                    files.add(rel)
        sets.append(files)

    common = sets[0].intersection(*sets[1:])

    result = {}
    for rel in sorted(common):
        result[rel] = [os.path.join(sd, rel) for sd in sample_dirs]

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sample_dirs = find_sample_dirs()
    N           = len(sample_dirs)
    print(f"\nAveraging over {N} samples.")

    base_files = collect_base_files(sample_dirs)
    print(f"Base files to average: {len(base_files)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "concentrations"), exist_ok=True)

    # Descriptions written at the top of each output file type
    desc = {
        'mean': (
            f"Average over N={N} independent samples."
            f"\n  mean(s) = ( s_1 + s_2 + ... + s_N ) / N"
        ),
        'std': (
            f"Sample standard deviation over N={N} independent samples (ddof=1)."
            f"\n  std(s) = sqrt( sum( (s_i - mean)^2 ) / (N-1) )"
        ),
        'sem': (
            f"Standard error of the mean over N={N} independent samples."
            f"\n  SEM(s) = std(s) / sqrt(N)"
        ),
        'errors': (
            f"Combined error over N={N} independent samples."
            f"\n  err(s) = sqrt( SEM(s)^2 + mean(err_i)^2 )"
            f"\n  where err_i is the per-point measurement error of sample i."
        ),
        'concentration': (
            f"Averaged concentration file over N={N} independent samples."
            f"\n  mean(s)     = ( s_1 + ... + s_N ) / N"
            f"\n  std(s)      = sqrt( sum( (s_i - mean)^2 ) / (N-1) )"
            f"\n  SEM(s)      = std(s) / sqrt(N)"
            f"\n  mean_err(s) = mean of per-point measurement errors err_i"
            f"\n  combined    = sqrt( SEM^2 + mean_err^2 )"
        ),
    }

    for rel_path, abs_paths in base_files.items():

        # --- concentration files: different layout, no companion files -------
        if is_concentration_file(rel_path):
            try:
                headers, data = average_concentration_file(abs_paths)
            except Exception as e:
                print(f"  Error averaging {rel_path}: {e}")
                continue
            out_path = os.path.join(OUTPUT_DIR, rel_path)
            write_dat_file(out_path, headers, data, description=desc['concentration'])
            print(f"  Exported: {out_path}")
            continue

        # --- standard files: require --errors and --std companions -----------
        errors_paths = [p.replace(".dat", "--errors.dat") for p in abs_paths]
        std_paths    = [p.replace(".dat", "--std.dat")    for p in abs_paths]

        # Skip if any companion file is missing in any sample
        missing = [
            p for p in errors_paths + std_paths
            if not os.path.exists(p)
        ]
        if missing:
            print(f"  Skipping {rel_path} — missing companion files:")
            for m in missing:
                print(f"    {m}")
            continue

        try:
            results = average_file_group(abs_paths, errors_paths, std_paths)
        except Exception as e:
            print(f"  Error averaging {rel_path}: {e}")
            continue

        # Write the four output files mirroring the input naming convention:
        #   base.dat            → mean values
        #   base--std.dat       → sample std (ddof=1)
        #   base--SEM.dat       → standard error of the mean
        #   base--errors.dat    → combined error sqrt(SEM² + mean_per-point_err²)
        base_out = os.path.join(OUTPUT_DIR, rel_path)
        for key, suffix in [('mean',   ''),
                             ('std',    '--std.dat'),
                             ('sem',    '--SEM.dat'),
                             ('errors', '--errors.dat')]:
            out_path = base_out if suffix == '' else base_out.replace(".dat", suffix)
            headers, data = results[key]
            write_dat_file(out_path, headers, data, description=desc[key])
        print(f"  Exported: {base_out} (+ --std, --SEM, --errors)")

    print("\nDone.")


if __name__ == "__main__":
    main()
