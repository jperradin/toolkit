#!/bin/bash
#SBATCH --nodes=1               # Number of nodes
#SBATCH --constraint=GENOA      # GENOA partition
#SBATCH --account=c1916879       # Account
#SBATCH --ntasks-per-node=192   # Reserve 192 CPUs
#SBATCH --cpus-per-task=1       # 1 CPU per task = 120 total
#SBATCH --hint=nomultithread    # Disable hyperthreading
#SBATCH --job-name=nexus        # Job name
#SBATCH --output=nexus-%j.out   # Output file
#SBATCH --error=nexus-%j.err    # Error file
#SBATCH --time=01:15:00         # Time limit

# Clean environment
module purge

# Load modules
module load PrgEnv-cray/8.6.0
module load cray-python/3.11.7
source /lus/home/CT9/ccm0504/jperradin/venv/bin/activate

export PATH=/opt/hpe/hpc/hmpt/hmpt-2.32/lib:$PATH

# Print job info for debugging
echo "Running on: $SLURM_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NNODES=$SLURM_NNODES"

# Run Python with multiprocessing
python launch_nexus-HPC.py
