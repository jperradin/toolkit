#!/usr/bin/env python3
"""Convert a LAMMPS xyz dump of an NPT run to extended-XYZ with per-frame box.

Each frame of the dump carries its timestep in the comment line
("Atoms. Timestep: N"). The cubic box edge at that step is read from the run's
log (thermo Lx column), so no separate boxes file is needed. Output frames get
a proper extended-XYZ header:

    Lattice="L 0 0 0 L 0 0 0 L" Properties=species:S:1:pos:R:3

Types are mapped 1->O, 2->Si, 3->Na.

usage: ./toolkit/convert_to_xyz-npt.py pos_2000K.data [pos_...data ...]
           [--log log.lammps] [--out out.xyz]
       (default log: log.lammps next to each pos file; default out: <pos>.xyz)
"""
import argparse
import os
import sys

TYPEMAP = {"1": "O", "2": "Si", "3": "Na"}


def lx_by_step(log_file):
    """{step: Lx} from every thermo table of a LAMMPS log."""
    out = {}
    cols = None
    ilx = istep = None
    with open(log_file) as fh:
        for line in fh:
            p = line.split()
            if p and p[0] == "Step":
                cols = p
                istep, ilx = p.index("Step"), p.index("Lx")
                continue
            if cols is None or len(p) != len(cols):
                continue
            try:
                out[int(float(p[istep]))] = float(p[ilx])
            except ValueError:
                continue
    if not out:
        raise SystemExit(f"no thermo table with Lx in {log_file}")
    return out


def find_heads(data):
    """[(natoms, step, count_line_start, atoms_start)] per frame header.

    Headers are located by searching the raw bytes for the literal
    "Atoms. Timestep:" -- far cheaper than a line-anchored regex over
    millions of atom lines.
    """
    heads = []
    j = 0
    while True:
        c = data.find(b"Atoms. Timestep:", j)
        if c == -1:
            return heads
        line_start = data.rfind(b"\n", 0, c) + 1
        count_start = data.rfind(b"\n", 0, max(line_start - 1, 0)) + 1
        eol = data.find(b"\n", c)
        heads.append((int(data[count_start:line_start]),
                      int(data[c + 16:eol]), count_start, eol + 1))
        j = eol


def map_types(data):
    """Map type digits to species at atom-line starts, whole buffer at once.

    Safe because only atom lines start with '<digit> ': count lines have no
    trailing token and comment lines start with ' Atoms.'.
    """
    for t, sym in TYPEMAP.items():
        data = data.replace(b"\n%s " % t.encode(), b"\n%s " % sym.encode())
    return data


def convert(pos_file, log_file, out_file):
    lx = lx_by_step(log_file)
    with open(pos_file, "rb") as f:
        data = f.read()
    if data and not data.endswith(b"\n"):
        data += b"\n"
    data = map_types(data)
    heads = find_heads(data)
    parts = []
    nframes = nskip = 0
    for i, (n, step, _, astart) in enumerate(heads):
        end = heads[i + 1][2] if i + 1 < len(heads) else len(data)
        L = lx.get(step)
        if L is None:
            nskip += 1
            continue
        parts.append(f'{n}\nLattice="{L} 0.0 0.0 0.0 {L} 0.0 0.0 0.0 {L}" '
                     f'Properties=species:S:1:pos:R:3 Timestep={step}\n'
                     .encode())
        parts.append(data[astart:end])
        nframes += 1
    with open(out_file, "wb") as out:
        out.write(b"".join(parts))
    skip = f", {nskip} frame(s) skipped (step not in log)" if nskip else ""
    print(f"{pos_file}: {nframes} frame(s) -> {out_file}{skip}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pos", nargs="+", help="xyz dump file(s) (pos_*.data)")
    p.add_argument("--log", help="log file with thermo Lx "
                                 "(default: log.lammps next to each pos file)")
    p.add_argument("--out", help="output xyz (single pos file only; "
                                 "default: <pos>.xyz)")
    args = p.parse_args(argv)
    if args.out and len(args.pos) > 1:
        p.error("--out only valid with a single pos file")
    for pos in args.pos:
        log = args.log or os.path.join(os.path.dirname(os.path.abspath(pos)),
                                       "log.lammps")
        out = args.out or os.path.splitext(pos)[0] + ".xyz"
        convert(pos, log, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
