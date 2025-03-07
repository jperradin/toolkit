import nexus
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sys


def process_input(input, output, pressure, temperature, n_atoms, n_Si, n_O):
    settings = nexus.settings.Settings(extension="SiOz")

    settings.quiet.set_value(True)

    # Set output file name
    settings.project_name.set_value(output)

    # Set various parameters
    settings.extension.set_value("SiOz")
    settings.export_directory.set_value(f"./")
    settings.path_to_xyz_file.set_value(input)

    settings.number_of_atoms.set_value(n_atoms)
    settings.header.set_value(2)
    # settings.range_of_frames.set_value([10,11])
    settings.structure.set_value(
        [
            {"element": "Si", "alias": 2, "number": n_Si},
            {"element": "O", "alias": 1, "number": n_O},
        ]
    )

    settings.temperature.set_value(temperature)
    settings.pressure.set_value(pressure)

    settings.overwrite_results.set_value(True)

    # Set cluster parameter 'bond' criteria
    settings.cluster_settings.set_cluster_parameter("connectivity", ["O", "Si", "O"])
    settings.cluster_settings.set_cluster_parameter("criteria", "bond")
    settings.cluster_settings.set_cluster_parameter("polyhedra", [[3, 3]])

    # Run the main function
    nexus.main(settings)


# -----------
system = {
    1008: {"n_Si": 336, "n_O": 672},
    3024: {"n_Si": 1008, "n_O": 2016},
    8064: {"n_Si": 2688, "n_O": 5376},
    15120: {"n_Si": 5040, "n_O": 10080},
    27216: {"n_Si": 9072, "n_O": 18144},
    96000: {"n_Si": 32000, "n_O": 64000},
    1056000: {"n_Si": 352000, "n_O": 704000},
}

# ----
n_atoms = np.int32(sys.argv[1])
n_Si = system[n_atoms]["n_Si"]
n_O = system[n_atoms]["n_O"]


# ----


# Load inputs, output names, and pressures from files
directory = "./"
inputs = []
pressures = []
outputs = []
temperatures = []


with open(os.path.join(directory, "inputs")) as f:
    data = f.readlines()
    for i, l in enumerate(data):
        inputs.append(l.strip())
f.close()

with open(os.path.join(directory, "outputs")) as f:
    data = f.readlines()
    for i, l in enumerate(data):
        outputs.append(l.strip())
f.close()

with open(os.path.join(directory, "pressure")) as f:
    data = f.readlines()
    for i, l in enumerate(data):
        pressures.append(float(l.strip()))
f.close()

with open(os.path.join(directory, "temperature")) as f:
    data = f.readlines()
    for i, l in enumerate(data):
        temperatures.append(float(l.strip()))
f.close()

# check data
li = len(inputs)
lo = len(outputs)
lp = len(pressures)
lt = len(temperatures)

if li == lo == lp == lt:
    pass
else:
    raise ValueError(
        "Inputs, outputs, pressures, temperatures, don't have the same size."
    )

# Initialize a progress bar
progress_bar = tqdm(
    enumerate(inputs),
    total=len(inputs),
    desc="",
    colour="#510e4c",
    unit="file",
    leave=False,
)

# Fancy color bar
color_gradient = nexus.utils.generate_color_gradient(len(inputs))

# Initialize settings
settings = nexus.settings.Settings(extension="SiOz")

# Enable print clusters positions
settings.print_clusters_positions.disable_warnings = (
    True  # Disable warnings if false, it ask to user to confirm the action (every loop)
)
settings.print_clusters_positions.set_value(False)

# give ntasks of slurm job in the submission script
n_workers = np.int32(sys.argv[2])
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    # Parallel processing of the inputs
    futures = []
    for i, input in progress_bar:
        if i >= 0:
            output = outputs[i]
            pressure = pressures[i]
            temperature = temperatures[i]
            progress_bar.set_description(
                f"Processing ... \u279c {str(input).split('/')[-1]}"
            )
            progress_bar.colour = "#%02x%02x%02x" % color_gradient[i]
            future = executor.submit(
                process_input, input, output, pressure, temperature, n_atoms, n_Si, n_O
            )
            futures.append(future)

    for future in tqdm(
        futures,
        total=len(futures),
        desc="Waiting for the results",
        colour="#510e4c",
        unit="file",
        leave=False,
    ):
        future.result()


print("\n\n\t\tAll inputs have been processed successfully.")
print(
    f"\n\t\tResults are saved here \u279c {settings.export_directory.get_value()}\n\n"
)
