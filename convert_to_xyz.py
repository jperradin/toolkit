# This script convert the lammps .data files to .xyz files
# with Lattice information in the comment line

import numpy as np
from natsort import natsorted
import sys
import os

rootdir = os.getcwd()
n_atoms = np.int32(sys.argv[1])
boxes_path = sys.argv[2]

boxes = []
with open(boxes_path, "r") as inp:
    lines = inp.readlines()
    for line in lines:
        boxes.append(float(line.strip()))
boxes = np.array(boxes)

expected_files = len(boxes)

counter_file = 0
for subdir, dirs, files in os.walk(rootdir):
    dirs.sort()
    files = natsorted(files)
    for file in files:
        if file.split(".")[1] == "data":
            with open(file, "r") as inp:
                with open(file.split(".")[0] + ".xyz", "w") as out:
                    counter_line = 0
                    lines = out.readlines()
                    for line in lines:
                        try:
                            if (
                                line.split(".")[0] == " Atoms"
                                or line.split(".")[0] == "Atoms"
                            ):
                                out.write(
                                    f'Lattice="{boxes[counter_file]} 0.0 0.0 0.0 {boxes[counter_file]} 0.0 0.0 0.0 {boxes[counter_file]}"\n'
                                )
                                counter_line += 1
                            elif line.split()[0] == f"{n_atoms}":
                                out.write(line)
                            elif line.split()[0] == "1":
                                parts = line.split()
                                out.write(f"O {parts[1]} {parts[2]} {parts[3]}\n")
                            elif line.split()[0] == "2":
                                parts = line.split()
                                out.write(f"Si {parts[1]} {parts[2]} {parts[3]}\n")
                            elif line.split()[0] == "3":
                                parts = line.split()
                                out.write(f"Na {parts[1]} {parts[2]} {parts[3]}\n")
                        except:
                            pass

            counter_file += 1

print(f"{expected_files} expected files, converted {counter_file} files !")
