#!/usr/bin/env python3
"""
Thermodynamic Data Analyzer for LAMMPS Output

This script processes LAMMPS log files to extract thermodynamic data,
calculate averages, and compute densities for molecular dynamics simulations.
"""

import numpy as np
import os
import sys
import argparse
from tqdm import tqdm


def ensure_directories():
    """Create required output directories if they don't exist."""
    for directory in ["./thermo", "./analysis"]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def calculate_density_factor(system, n_atoms):
    """
    Calculate the density conversion factor for a specific system and atom count.
    
    Parameters:
        system (str): The type of system (e.g., "SiO2", "NSx")
        n_atoms (int): Number of atoms in the system
    
    Returns:
        float: The density conversion factor
    """
    # System-specific conversion factors
    factors = {
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
        }
    }
    
    if system not in factors:
        raise ValueError(f"System '{system}' not implemented.")
    
    if n_atoms not in factors[system]:
        raise ValueError(f"Atom count {n_atoms} for system '{system}' not implemented.")
    
    return factors[system][n_atoms]


def calculate_densities(system, n_atoms):
    """
    Calculate densities from box dimensions.
    
    Parameters:
        system (str): The type of system
        n_atoms (int): Number of atoms in the system
    
    Returns:
        list: Calculated densities
    """
    factor = calculate_density_factor(system, n_atoms)
    densities = []
    
    with open("./analysis/boxes", "r") as file:
        for line in file:
            box = float(line.strip())
            volume = box**3  # volume in angstrom
            volume *= 0.00000001**3  # conversion to cm³
            density = factor / volume
            densities.append(density)
    
    return densities
            

def parse_lammps_log_column(log_file):
    """
    Parse LAMMPS log file and extract thermodynamic data.
    
    Parameters:
        log_file (str): Path to the LAMMPS log file
    """
    with open(log_file, "r") as inp:
        lines = inp.readlines()
        for line in lines:
            try:
                if line.split()[0] == "Step":
                    header = line.split()
                    break
            except:
                pass
            
    return header

def parse_lammps_log(log_file, columns, start_t="0", end_t="10000", system='SiO2', n_atoms=1008):
    """
    Parse LAMMPS log file and extract thermodynamic data.
    
    Parameters:
        log_file (str): Path to the LAMMPS log file
        start_t (str): Starting timestep
        end_t (str): Ending timestep
    
    Returns:
        tuple: Lists of extracted data (pressure, temperature, volume, box, ekin, epot, etot)
    """
    pressure = []
    temperature = []
    volume = []
    box = []
    dens = []
    ekin = []
    epot = []
    etot = []
    time = []
    
    # get indices
    idx_pressure = columns.index("Press")
    idx_temperature = columns.index("Temp")
    idx_volume = columns.index("Volume")
    idx_box = columns.index("Lx")
    idx_ekin = columns.index("KinEng")
    idx_epot = columns.index("PotEng")
    idx_etot = columns.index("TotEng")
    idx_time = columns.index("Time")
    
    # get factor for density calculation
    factor = calculate_density_factor(system, n_atoms)

    start = 1
    checkpoint = False
    
    with open(log_file, "r") as inp:
        for li, line in enumerate(inp):
            try:
                if line.split()[0] == start_t:
                    checkpoint = True
                    block_pressure = []
                    block_temperature = []
                    block_volume = []
                    block_box = []
                    block_epot = []
                    block_ekin = []
                    block_etot = []
                    block_time = []
            except:
                pass
                
            if checkpoint and line[0] != "#":
                try:
                    values = line.split()
                    block_pressure.append(float(values[idx_pressure]) * 0.0001)
                    block_temperature.append(float(values[idx_temperature]))
                    block_volume.append(float(values[idx_volume]))
                    block_box.append(float(values[idx_box]))
                    block_ekin.append(float(values[idx_ekin]))
                    block_epot.append(float(values[idx_epot]))
                    block_etot.append(float(values[idx_etot]))
                    block_time.append(float(values[idx_time]))
                except:
                    pass
                    
            try:
                if line.split()[0] == end_t:
                    checkpoint = False
                    
                    # Convert to numpy arrays
                    block_pressure = np.array(block_pressure)
                    block_temperature = np.array(block_temperature)
                    block_volume = np.array(block_volume)
                    block_box = np.array(block_box)
                    block_ekin = np.array(block_ekin)
                    block_epot = np.array(block_epot)
                    block_etot = np.array(block_etot)
                    
                    # Append means to result lists
                    pressure.append(np.mean(block_pressure))
                    temperature.append(np.mean(block_temperature))
                    volume.append(np.mean(block_volume))
                    box.append(np.mean(block_box))
                    v = np.mean(block_box)**3
                    v *= 0.00000001**3
                    dens.append(factor / v)
                    
                    ekin.append(np.mean(block_ekin))
                    epot.append(np.mean(block_epot))
                    etot.append(np.mean(block_etot))
                    
                    # Write block data to file
                    with open(f"thermo/thermo-{start}B.dat", "w") as out:
                        out.write(
                            "# Time\tPressure\tTemperature\tVolume\tDensity\tLbox\tEkin\tEpot\tEtot\n"
                        )
                        for i in range(len(block_pressure)):
                            out.write(
                                f"{block_time[i]:2.2f}\t{block_pressure[i]:2.6f}\t{block_temperature[i]:2.6f}\t"
                                f"{block_volume[i]:2.6f}\t{dens[i]:2.3f}\t{block_box[i]:2.6f}\t{block_ekin[i]:2.6f}\t"
                                f"{block_epot[i]:2.6f}\t{block_etot[i]:2.6f}\n"
                            )
                    
                    start += 1
            except:
                pass
    
    return pressure, temperature, volume, dens, box, ekin, epot, etot


def write_analysis_files(pressure, temperature, volume, dens, box, ekin, epot, etot):
    """
    Write extracted thermodynamic data to analysis files.
    
    Parameters:
        pressure, temperature, volume, box, ekin, epot, etot: Lists of extracted data
    """
    # Print to console
    for i in range(len(pressure)):
        print(
            f"{pressure[i]:2.6f}\t{temperature[i]:2.6f}\t{volume[i]:2.6f}\t{dens[i]:2.3f}\t"
            f"{box[i]:2.6f}\t{ekin[i]:2.6f}\t{epot[i]:2.6f}\t{etot[i]:2.6f}\t"
        )

    # Write combined data
    with open("./analysis/thermo", "w") as f:
        for i in range(len(pressure)):
            f.write(
                f"{pressure[i]:2.6f}\t{temperature[i]:2.6f}\t{volume[i]:2.6f}\t{dens[i]:2.3f}\t"
                f"{box[i]:2.6f}\t{ekin[i]:2.6f}\t{epot[i]:2.6f}\t{etot[i]:2.6f}\n"
            )

    # Write individual property files
    properties = {
        "pressure": pressure,
        "temperature": temperature,
        "volume": volume,
        "boxes": box,
        "outputs": dens,    
        "ekin": ekin,
        "epot": epot,
        "etot": etot
    }
    
    for prop_name, prop_data in properties.items():
        with open(f"./analysis/{prop_name}", "w") as f:
            for value in prop_data:
                if prop_name == "outputs":
                    f.write(f"dens{value:1.3f}\n")
                f.write(f"{value:2.6f}\n")


def main():
    """Main function to process LAMMPS data and calculate densities."""
    parser = argparse.ArgumentParser(description="Process LAMMPS thermodynamic data and calculate densities.")
    parser.add_argument("log_file", help="LAMMPS log file to process")
    parser.add_argument("--start", default="0", help="Starting timestep for data extraction")
    parser.add_argument("--end", default="10000", help="Ending timestep for data extraction")
    parser.add_argument("--system", help="System type (e.g., 'SiO2', 'NSx')")
    parser.add_argument("--n_atoms", type=int, help="Number of atoms in the system")
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    ensure_directories()
    
    # Parse LAMMPS log file
    header = parse_lammps_log_column(args.log_file)
    
    # Process LAMMPS log file
    pressure, temperature, volume, dens, box, ekin, epot, etot = parse_lammps_log(
        args.log_file, header, args.start, args.end, args.system, args.n_atoms
    )
    
    # Write thermodynamic data to analysis files
    write_analysis_files(pressure, temperature, volume, dens, box, ekin, epot, etot)
    
    print("processed log_lammps file")


if __name__ == "__main__":
    main()