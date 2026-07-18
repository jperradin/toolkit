#!/usr/bin/env python3
"""Log-driven conversion of LAMMPS xyz dumps to extended-XYZ (NPT, cubic box).

Reverse of convert_to_xyz-npt.py: instead of matching frame timesteps against
the log (ambiguous when a log chains many runs whose step ranges overlap, e.g.
the 141-pt compression where EVERY point's B block spans 312500-625000), this
walks the log itself:

    dump <id> <grp> xyz <rate> <file>     opens an xyz dump (echoed, expanded)
    Step ... Lx ...                       thermo table of the following run
    undump <id>                           closes it

Within each thermo table, rows whose Step is a multiple of the dump rate are
the dumped frames, in order; their Lx values are assigned frame-by-frame to
that dump file. Each dump file present on disk is then rewritten as
<file>.xyz with per-frame Lattice headers (types 1->O, 2->Si, 3->Na).

usage: ./toolkit/convert_to_xyz-from-log.py <log> [<log> ...] [--dry-run]
       ./toolkit/convert_to_xyz-from-log.py sample-1/2000K/02-compression/log.lammps-1-21
"""
import argparse
import os
import re
import sys

TYPEMAP = {"1": "O", "2": "Si", "3": "Na"}
_DUMP_RE = re.compile(r"^dump\s+(\S+)\s+\S+\s+xyz\s+(\d+)\s+(\S+)\s*$")
_UNDUMP_RE = re.compile(r"^undump\s+(\S+)")


def scan_log(log_file):
    """{dump_path: [Lx per frame, in order]} from one log."""
    boxes = {}
    active = {}          # dump id -> (path, rate)
    cols = None
    istep = ilx = None
    with open(log_file) as fh:
        for line in fh:
            p = line.split()
            if p and p[0] == "Step" and "Lx" in p:
                cols = p
                istep, ilx = p.index("Step"), p.index("Lx")
                continue
            if cols is not None and len(p) == len(cols):
                try:
                    step = int(float(p[istep]))
                    lx = float(p[ilx])
                except ValueError:
                    cols = None
                else:
                    for path, rate in active.values():
                        if step % rate == 0:
                            boxes.setdefault(path, []).append(lx)
                    continue
            elif cols is not None and p:
                cols = None          # non-matching line ends the table
            m = _DUMP_RE.match(line)
            if m and "$" not in m.group(3):    # fully-substituted echo only
                active[m.group(1)] = (m.group(3), int(m.group(2)))
                continue
            m = _UNDUMP_RE.match(line)
            if m:
                active.pop(m.group(1), None)
    return boxes


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


def convert(pos_file, lx_list, out_file):
    """Rewrite an xyz dump as extended-XYZ using the per-frame Lx list."""
    with open(pos_file, "rb") as f:
        data = f.read()
    if data and not data.endswith(b"\n"):
        data += b"\n"
    data = map_types(data)
    heads = find_heads(data)
    parts = []
    k = 0
    note = ""
    for i, (n, step, _, astart) in enumerate(heads):
        if k >= len(lx_list):
            note = "more frames than log rows"
            break
        end = heads[i + 1][2] if i + 1 < len(heads) else len(data)
        L = lx_list[k]
        parts.append(f'{n}\nLattice="{L} 0.0 0.0 0.0 {L} 0.0 0.0 0.0 {L}" '
                     f'Properties=species:S:1:pos:R:3 '
                     f'Timestep={step}\n'.encode())
        parts.append(data[astart:end])
        k += 1
    with open(out_file, "wb") as out:
        out.write(b"".join(parts))
    if not note and k != len(lx_list):
        note = f"file has {k} frames, log has {len(lx_list)}"
    return k, note


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="LAMMPS log file(s)")
    p.add_argument("--dry-run", action="store_true",
                   help="only list the dump files and frame counts found")
    args = p.parse_args(argv)

    n_done = 0
    for log in args.logs:
        boxes = scan_log(log)
        if not boxes:
            print(f"# {log}: no xyz dumps with a thermo Lx table found")
            continue
        base = os.path.dirname(os.path.abspath(log))
        for path, lx_list in boxes.items():
            f = os.path.normpath(os.path.join(base, path))
            tag = f"{path} ({len(lx_list)} frames in log)"
            if not os.path.isfile(f):
                print(f"# {tag}: file not on disk -- skipped")
                continue
            if args.dry_run:
                print(f"# {tag}: would convert")
                continue
            out = os.path.splitext(f)[0] + ".xyz"
            k, note = convert(f, lx_list, out)
            note = f"  [{note}]" if note else ""
            print(f"{tag}: {k} frame(s) -> {out}{note}")
            n_done += 1
    return 0 if (n_done or args.dry_run) else 1


if __name__ == "__main__":
    sys.exit(main())
