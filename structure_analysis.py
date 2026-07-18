#!/usr/bin/env python3
"""Comprehensive structural analysis of every (rho, T) isochore state point with
the `rreve` package (https://github.com/jperradin/rreve).

For each sample-N/rho-XXX/pos_<T>K.data trajectory it runs rreve's real-space
analyzers -- PDF g(r), bond-angle distribution, structural units, connectivity,
polyhedricity, tetrahedricity -- on the last NFRAMES production frames. The
neutron structure factor (FFT) is excluded by request.

The LAMMPS pos dumps are type-coded (1->O, 2->Si) and carry no box, so each frame
is rewritten as extended-XYZ with a `Lattice="..."` header taken from the
isochore box file (one L per density). Outputs land in
<sample>/structure/<rho-XXX>/<T>K/ (one rreve project per state point).

usage: ./toolkit/structure_analysis.py [sample-dir ...] [--nframes N]
       (default: all sample-*/, last 10 frames)
"""
import glob
import os
import sys

# rreve.main calls os.get_terminal_size() unconditionally for its progress bar,
# which raises OSError when stdout is not a tty (piped/batch). Provide a fallback.
_real_gts = os.get_terminal_size
def _safe_gts(fd=1):
    try:
        return _real_gts(fd)
    except OSError:
        return os.terminal_size((120, 40))
os.get_terminal_size = _safe_gts

NAT = 8064
NFRAMES = 10
TYPEMAP = {"1": "O", "2": "Si"}
CUTOFFS = [("Si", "Si", 3.50), ("Si", "O", 2.30), ("O", "O", 3.05)]


def box_L(rho_dir):
    """Edge length L (A) from the first box_*.data in the isochore dir."""
    for bf in sorted(glob.glob(os.path.join(rho_dir, "box_*.data"))):
        for ln in open(bf):
            if "xlo xhi" in ln:
                a, b = ln.split()[:2]
                return float(b) - float(a)
    raise RuntimeError(f"no box file with 'xlo xhi' in {rho_dir}")


def extract_xyz(pos, L, out_xyz, nframes):
    """Write the last `nframes` frames of a type-coded pos dump as extended-XYZ."""
    lines = open(pos).read().split("\n")
    fsz = NAT + 2
    nfit = len(lines) // fsz
    start = max(0, nfit - nframes)
    lat = f'Lattice="{L} 0.0 0.0 0.0 {L} 0.0 0.0 0.0 {L}"'
    with open(out_xyz, "w") as o:
        for fi in range(start, nfit):
            blk = lines[fi * fsz:(fi + 1) * fsz]
            o.write(f"{NAT}\n{lat}\n")
            for ln in blk[2:2 + NAT]:
                p = ln.split()
                if len(p) >= 4:
                    o.write(f"{TYPEMAP[p[0]]} {p[1]} {p[2]} {p[3]}\n")
    return nfit - start


def run_one(pos, tag, T, L, outroot):
    from rreve import SettingsBuilder, main
    import rreve.config.settings as c

    tmp = os.path.join(outroot, "_xyz", f"{tag}_{T}K.xyz")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    nf = extract_xyz(pos, L, tmp, NFRAMES)

    cg = c.GeneralSettings(
        project_name=f"{tag}/{T}K",
        export_directory=outroot,
        file_location=tmp,
        range_of_frames=(0, -1),
        apply_pbc=True,
        verbose=False,
        save_logs=True,
        save_performance=False,
        decorate_input_file=False,
        cutoffs=[c.Cutoff(*cu) for cu in CUTOFFS],
        coordination_mode="different_type",
    )
    cl = c.LatticeSettings(apply_custom_lattice=False)   # read Lattice from xyz
    ca = c.AnalysisSettings(with_all=True)               # poly/tetra default Si/O
    ca.exclude_analyzer("neutron_structure_factor_fft")
    settings = (SettingsBuilder().with_general(cg).with_lattice(cl)
                .with_analysis(ca).build())
    main(settings)
    return nf


def main_cli(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    global NFRAMES
    if "--nframes" in argv:
        NFRAMES = int(argv[argv.index("--nframes") + 1])
    sdirs = args or sorted(glob.glob("sample-*"))
    sdirs = [s for s in sdirs if os.path.isdir(s)]
    for sd in sdirs:
        outroot = os.path.join(sd, "structure")
        rho_dirs = sorted(glob.glob(os.path.join(sd, "rho-*")))
        for rd in rho_dirs:
            tag = os.path.basename(rd)
            try:
                L = box_L(rd)
            except RuntimeError as e:
                print(f"# skip {tag}: {e}"); continue
            for pos in sorted(glob.glob(os.path.join(rd, "pos_*K.data"))):
                T = int(os.path.basename(pos).replace("pos_", "").replace("K.data", ""))
                nf = run_one(pos, tag, T, L, outroot)
                print(f"# {sd}/{tag} {T}K  L={L:.2f}A  frames={nf}  done")


if __name__ == "__main__":
    main_cli(sys.argv)
