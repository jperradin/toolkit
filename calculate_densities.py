import numpy as np
from tqdm import tqdm
import sys

system = sys.argv[1]
n_atoms = np.int32(sys.argv[2])


def calculate_factor(system, n_atoms):
    # implemenent new factor each time
    if system == "SiO2":
        if n_atoms == 1008:
            return 3.352307451024e-20
        elif n_atoms == 3024:
            return 1.0056922353072e-19
        elif n_atoms == 8064:
            return 2.6818459608192e-19
        elif n_atoms == 15120:
            return 5.028461176536e-19
        elif n_atoms == 27216:
            return 9.0512301177648e-19
        elif n_atoms == 96000:
            return 3.19267376288e-18
        elif n_atoms == 1056000:
            return 3.511941139168e-17
        else:
            raise ValueError(f"system {system} - {n_atoms} not implemented.")

    elif system == "NSx":
        if n_atoms == 1080:
            return 1.005571132125e-19
        elif n_atoms == 3000:
            return 3.62005607565e-20
        elif n_atoms == 1350:
            return 4.51799557146e-20
        elif n_atoms == 13500:
            return 4.51799557146e-19
        else:
            raise ValueError(f"system {system} - {n_atoms} not implemented.")

    else:
        raise ValueError(f"system {system} not implemented.")


factor = calculate_factor(system, n_atoms)

densities = []
with open("./analysis/boxes", "r") as file:
    for li, l in enumerate(file):
        box = np.float64(l.strip())
        volume = box**3  # volume in angstrom
        volume *= 0.00000001**3  # conversion to cm
        density = factor / volume
        densities.append(density)

with open("./analysis/outputs", "w") as f:
    for dens in densities:
        print(f"dens{dens:1.3f}")
        f.write(f"dens{dens:1.3f}\n")

