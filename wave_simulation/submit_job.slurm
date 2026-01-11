#!/bin/bash
#SBATCH --job-name=wave_sim
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=wave_sim_%j.out
#SBATCH --error=wave_sim_%j.err

# Load required modules (adjust based on your cluster)
module load openmpi 2>/dev/null || echo "OpenMPI module not found, using system default"
module load gcc 2>/dev/null || echo "GCC module not found, using system default"

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "=========================================="

# Change to the job submission directory
cd $SLURM_SUBMIT_DIR

# Create output directory
OUTPUT_DIR="./output_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

#Compile the code if executable doesn't exist
if [ ! -f ./wave_simulation ]; then
    echo "Compiling wave_simulation..."
    make clean
    make
fi

# Run the simulation
# 2 nodes x 16 cores = 32 MPI processes
echo "Starting simulation with $SLURM_NTASKS MPI processes..."
mpirun -np $SLURM_NTASKS ./wave_simulation \
    --nx 512 \
    --ny 512 \
    --gamma 0.1 \
    --dt 0.0001 \
    --t_end 2.0 \
    --save_interval 100 \
    --output $OUTPUT_DIR

echo "=========================================="
echo "End time: $(date)"
echo "Output files saved to: $OUTPUT_DIR"
echo "=========================================="

# Count output files
NUM_FILES=$(ls -1 $OUTPUT_DIR/*.bin 2>/dev/null | wc -l)
echo "Generated $NUM_FILES output files"
