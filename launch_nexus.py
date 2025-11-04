from nexus import SettingsBuilder, main
import nexus.config.settings as c
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import os
import sys
import traceback


def run_single_analysis(task):
    """Process a single file with explicit error handling"""
    path, name = task
    
    # IMPORTANT: Print immediately to confirm the process started
    print(f"[WORKER {os.getpid()}] Starting: {name}", flush=True)
    
    try:
        # Test if modules are accessible
        print(f"[WORKER {os.getpid()}] Modules loaded OK", flush=True)
        
        # Lattice settings
        config_lattice = c.LatticeSettings(
            apply_custom_lattice=False,
        )

        # First analysis: coordination [4, 6]
        config_clustering = c.ClusteringSettings(
            criterion="bond",
            node_types=["Si", "O"],
            node_masses=[28.0855, 15.9994],
            connectivity=["Si", "O", "Si"],
            cutoffs=[
                c.Cutoff(type1="Si", type2="O", distance=2.30),
            ],
            with_coordination_number=True,
            coordination_mode="O",
            coordination_range=[4, 6],
            with_alternating=True,
            with_printed_unwrapped_clusters=False,
            print_mode="connectivity",
        )

        config_analysis = c.AnalysisSettings(
            with_all=True,
        )

        config_general = c.GeneralSettings(
            project_name=name,
            export_directory="./",
            file_location=path,
            range_of_frames=(0, -1),
            apply_pbc=True,
            verbose=False,
            save_logs=True,
            save_performance=True,
        )

        settings = (
            SettingsBuilder()
            .with_general(config_general)
            .with_lattice(config_lattice)
            .with_clustering(config_clustering)
            .with_analysis(config_analysis)
            .build()
        )

        print(f"[WORKER {os.getpid()}] Running analysis 1/3 for {name}", flush=True)
        main(settings)

        # Second analysis: coordination [6, 6]
        config_analysis.overwrite = False
        
        config_clustering = c.ClusteringSettings(
            criterion="bond",
            node_types=["Si", "O"],
            node_masses=[28.0855, 15.9994],
            connectivity=["Si", "O", "Si"],
            cutoffs=[
                c.Cutoff(type1="Si", type2="O", distance=2.30),
            ],
            with_coordination_number=True,
            coordination_mode="O",
            coordination_range=[6, 6],
            with_pairwise=False,
            with_printed_unwrapped_clusters=False,
            print_mode="connectivity",
            with_number_of_shared=True,
            shared_mode="O",
            shared_threshold=2,
            with_connectivity_name="Stishovite",
        )

        settings = (
            SettingsBuilder()
            .with_general(config_general)
            .with_lattice(config_lattice)
            .with_clustering(config_clustering)
            .with_analysis(config_analysis)
            .build()
        )

        print(f"[WORKER {os.getpid()}] Running analysis 2/3 for {name}", flush=True)
        main(settings)

        # Third analysis: O-Si-O connectivity
        config_clustering = c.ClusteringSettings(
            criterion="bond",
            node_types=["O", "Si"],
            node_masses=[15.9994, 28.0855],
            connectivity=["O", "Si", "O"],
            cutoffs=[
                c.Cutoff(type1="O", type2="Si", distance=2.30),
            ],
            with_coordination_number=True,
            coordination_mode="Si",
            coordination_range=[2, 3],
            with_alternating=True,
            with_printed_unwrapped_clusters=False,
        )

        settings = (
            SettingsBuilder()
            .with_general(config_general)
            .with_lattice(config_lattice)
            .with_clustering(config_clustering)
            .with_analysis(config_analysis)
            .build()
        )

        print(f"[WORKER {os.getpid()}] Running analysis 3/3 for {name}", flush=True)
        main(settings)
        
        print(f"[WORKER {os.getpid()}] COMPLETED: {name}", flush=True)
        return {"status": "success", "name": name, "pid": os.getpid()}

    except Exception as e:
        # Capture full traceback
        tb_str = traceback.format_exc()
        error_msg = f"[WORKER {os.getpid()}] ERROR in {name}:\n{tb_str}"
        print(error_msg, flush=True)
        # Return error info instead of raising (so we can see it)
        return {"status": "error", "name": name, "error": str(e), "traceback": tb_str}


if __name__ == "__main__":
    print(f"[MAIN] Starting script with PID {os.getpid()}", flush=True)
    print(f"[MAIN] Python version: {sys.version}", flush=True)
    
    # Load file paths and output names
    try:
        files, outputs = np.loadtxt(
            "nexus_inputs",
            dtype="<U200",
            unpack=True,
            comments="#",
        )
    except IOError as e:
        print(f"[MAIN] Error reading input files: {e}", flush=True)
        exit(1)

    # Create task list
    if files.shape != outputs.shape:
        print("[MAIN] Error: inputs and outputs have different sizes", flush=True)
        exit(1)

    tasks = list(zip(files, outputs))
    
    if not tasks:
        print("[MAIN] No tasks found in input files.", flush=True)
        exit(1)

    # Get number of workers from SLURM
    n_workers = int(os.environ.get('SLURM_NTASKS', 1))
    print(f"[MAIN] SLURM_NTASKS={n_workers}", flush=True)
    print(f"[MAIN] Processing {len(tasks)} files with {n_workers} workers\n", flush=True)

    # Test with just 2 tasks first for debugging
    # test_tasks = tasks[:2]
    # print(f"[MAIN] TESTING with first {len(test_tasks)} tasks only", flush=True)

    results = []
    errors = []
    
    try:
        # Use ProcessPoolExecutor with explicit error checking
        # with ProcessPoolExecutor(max_workers=min(n_workers, len(test_tasks))) as executor:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as executor:
            print(f"[MAIN] ProcessPoolExecutor created", flush=True)
            
            # Submit all tasks and store futures
            futures = []
            for i, task in enumerate(tasks):
                print(f"[MAIN] Submitting task {i}: {task[1]}", flush=True)
                future = executor.submit(run_single_analysis, task)
                futures.append((future, task))
            
            print(f"[MAIN] All tasks submitted, waiting for results...", flush=True)
            
            # Explicitly get results and check for exceptions
            for i, (future, task) in enumerate(futures):
                try:
                    print(f"[MAIN] Getting result for task {i}", flush=True)
                    result = future.result(timeout=30000)  # 5 min timeout per task
                    results.append(result)
                    
                    if result['status'] == 'error':
                        errors.append(result)
                        print(f"[MAIN] Task {i} failed: {result['name']}", flush=True)
                    else:
                        print(f"[MAIN] Task {i} succeeded: {result['name']}", flush=True)
                        
                except Exception as e:
                    error_msg = f"[MAIN] Exception getting result for task {i} ({task[1]}): {e}\n{traceback.format_exc()}"
                    print(error_msg, flush=True)
                    errors.append({"name": task[1], "error": str(e)})

    except Exception as e:
        print(f"[MAIN] ProcessPoolExecutor failed: {e}\n{traceback.format_exc()}", flush=True)
        exit(1)

    # Print summary
    print(f"\n[MAIN] ===== SUMMARY =====", flush=True)
    print(f"[MAIN] Total tasks: {len(tasks)}", flush=True)
    print(f"[MAIN] Successful: {len([r for r in results if r['status'] == 'success'])}", flush=True)
    print(f"[MAIN] Failed: {len(errors)}", flush=True)
    
    if errors:
        print(f"\n[MAIN] Errors encountered:", flush=True)
        for err in errors:
            print(f"  - {err['name']}: {err.get('error', 'Unknown error')}", flush=True)
        exit(1)
    else:
        print(f"[MAIN] All test tasks completed successfully!", flush=True)

