# Import necessary modules
from nexus import SettingsBuilder, main
import nexus.config.settings as c
import numpy as np
from tqdm import tqdm


outputs = np.loadtxt("../sio6-sio6/analysis/outputs", dtype=np.str_)

for input, density in tqdm(enumerate(outputs)):
    print(input + 1, density)
    input += 1
    path = f"../sio6-sio6/B-part/positions_xyz/pos{input}B.xyz"

    # General settings
    config_general = c.GeneralSettings(
        project_name=density,  # Project name
        export_directory="VHD",  # Export directory
        file_location=path,  # File location
        range_of_frames=(0, -1),  # Range of frames
        apply_pbc=True,  # Apply periodic boundary conditions
        verbose=True,  # Verbose mode (if True, print title, progress bars, etc.)
        save_logs=True,  # Save logs    (save logs to export_directory/logs.txt)
        save_performance=True,  # Save performance (save performance data to export_directory/performance...json)
    )

    # Lattice settings
    config_lattice = c.LatticeSettings(
        apply_custom_lattice=False,  # If False, read lattice from trajectory file
    )

    # Clustering settings
    config_clustering = c.ClusteringSettings(
        criterion="distance",
        node_types=["Si"],
        node_masses=[28.0855],
        connectivity=["Si", "Si"],
        cutoffs=[
            c.Cutoff(type1="Si", type2="Si", distance=3.50)
        ],  # cutoff distance in reduced units
        with_coordination_number=True,
        coordination_mode="Si",  # "all_types" or "same_type" or "different_type" or "<node_type>"
        coordination_range=[5, 7],
        # if all below are False, calculate A_z-B_z cluster connectivity with z = coordination range
        with_pairwise=False,  # if with_coordination_number is True, calculate pairwise coordination number ie 4-4, 5-5, 6-6 ...
        with_mixing=False,  # if with_coordination_number is True, calculate mixing coordination number ie 4-5, 5-6, 4-6 ...
        with_alternating=False,  # if with_coordination_number is True, calculate alternating coordination number ie 4-5, 5-6 ...
        with_connectivity_name="VHD",
    )

    # Analysis settings
    config_analysis = c.AnalysisSettings(
        with_all=True,
    )
    config_analysis.overwrite = False

    # Build Settings object
    settings = (
        SettingsBuilder()
        .with_general(config_general)  # General settings \
        .with_lattice(config_lattice)  # Lattice settings \
        .with_clustering(config_clustering)  # Clustering settings \
        .with_analysis(config_analysis)  # Analysis settings \
        .build()  # Don't forget to build the settings object
    )

    if input >= 0:
        # Run the main function to process the trajectory
        main(settings)
