#!/bin/bash
#SBATCH --job-name nexus
#SBATCH --output nexus-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=1
#SBATCH --partition l2c_stat
#SBATCH --account l2c_stat

module purge
module load cv-standard
module load gcc/4.9.3
module load openmpi/psm2/2.0.1

module list

echo "Running on: $SLURM_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"

python launch_nexus.py 1056000 $SLURM_NTASKS
