# File that read thermodynamic data in log.lammps for multiple runs
# Specify timesteps to start and to end parsing process

import numpy as np
import os
import sys

if not os.path.exists("./thermo"):
    os.makedirs("./thermo")

if not os.path.exists("./analysis"):
    os.makedirs("./analysis")

start_t = "15650"
end_t = "31250"

with open(sys.argv[1], "r") as inp:
    checkpoint = False
    pressure = []
    temperature = []
    volume = []
    box = []
    ekin = []
    epot = []
    etot = []

    start = 1

    for li, l in enumerate(inp):
        try:
            if l.split()[0] == start_t:  # skipping one value but np
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
        if checkpoint and l[0] != "#":
            block_pressure.append(float(l.split()[6]) * 0.0001)
            block_temperature.append(float(l.split()[1]))
            block_volume.append(float(l.split()[5]))
            block_box.append(float(l.split()[11]))
            block_ekin.append(float(l.split()[2]))
            block_epot.append(float(l.split()[3]))
            block_etot.append(float(l.split()[4]))
            block_time.append(float(l.split()[-2]))
        try:
            if l.split()[0] == end_t:
                checkpoint = False
                checkpoint_c = False
                block_pressure = np.array(block_pressure)
                block_temperature = np.array(block_temperature)
                block_volume = np.array(block_volume)
                pressure.append(np.mean(block_pressure))
                temperature.append(np.mean(block_temperature))
                volume.append(np.mean(block_volume))
                box.append(np.mean(block_box))
                ekin.append(np.mean(block_ekin))
                epot.append(np.mean(block_epot))
                etot.append(np.mean(block_etot))
                with open(f"thermo/thermo-{start}B.dat", "w") as out:
                    out.write(
                        "# Time\tPressure\tTemperature\tVolume\tLbox\tEkin\tEpot\tEtot\n"
                    )
                    for i in range(len(block_pressure)):
                        out.write(
                            f"{block_time[i]:2.2f}\t{block_pressure[i]:2.6f}\t{block_temperature[i]:2.6f}\t{block_volume[i]:2.6f}\t{block_box[i]:2.6f}\t{block_ekin[i]:2.6f}\t{block_epot[i]:2.6f}\t{block_etot[i]:2.6f}\n"
                        )
                out.close()
                start += 1
        except:
            pass

for i in range(len(pressure)):
    print(
        f"{pressure[i]:2.6f}\t{temperature[i]:2.6f}\t{volume[i]:2.6f}\t{box[i]:2.6f}\t{ekin[i]:2.6f}\t{epot[i]:2.6f}\t{etot[i]:2.6f}\t"
    )


# Write data to ./analysis/thermo.dat and other files
with open("./analysis/thermo", "w") as f:
    for i in range(len(pressure)):
        f.write(
            f"{pressure[i]:2.6f}\t{temperature[i]:2.6f}\t{volume[i]:2.6f}\t{box[i]:2.6f}\t{ekin[i]:2.6f}\t{epot[i]:2.6f}\t{etot[i]:2.6f}\n"
        )

with open("./analysis/pressure", "w") as f:
    for i in range(len(pressure)):
        f.write(f"{pressure[i]:2.6f}\n")

with open("./analysis/temperature", "w") as f:
    for i in range(len(pressure)):
        f.write(f"{temperature[i]:2.6f}\n")

with open("./analysis/volume", "w") as f:
    for i in range(len(pressure)):
        f.write(f"{volume[i]:2.6f}\n")

with open("./analysis/boxes", "w") as f:
    for i in range(len(pressure)):
        f.write(f"{box[i]:2.6f}\n")

with open("./analysis/ekin", "w") as f:
    for i in range(len(pressure)):
        f.write(f"{ekin[i]:2.6f}\n")

with open("./analysis/epot", "w") as f:
    for i in range(len(pressure)):
        f.write(f"{epot[i]:2.6f}\n")

with open("./analysis/etot", "w") as f:
    for i in range(len(pressure)):
        f.write(f"{etot[i]:2.6f}\n")
