# Dampened Wave Equation Simulation

MPI-parallel simulation of the dampened wave equation on a rectangular drumhead.

## Mathematical Model

The simulation solves:

```
∂²u/∂t² + γ(∂u/∂t) = c²(∂²u/∂x² + ∂²u/∂y²)
```

Where:
- `u` - displacement of the drumhead
- `γ` - damping coefficient (default: 0.1)
- `c` - wave speed (default: 1.0)

Boundary conditions: Fixed edges (u = 0)

## Files

- `src/wave_simulation.cpp` - Main MPI simulation code
- `Makefile` - Build configuration
- `submit_job.slurm` - SLURM job submission script
- `visualize.py` - Python visualization script

## Building

```bash
# Load modules (cluster-specific)
module load openmpi gcc

# Compile
make
```

## Running on HPC

```bash
# Submit job to SLURM (2 nodes, 16 cores each)
sbatch submit_job.slurm

# Check job status
squeue -u $USER
```

## Command Line Options

```
--nx N           Grid points in x direction (default: 512)
--ny N           Grid points in y direction (default: 512)
--gamma G        Damping coefficient (default: 0.1)
--dt DT          Time step (default: 0.0001)
--t_end T        End time (default: 2.0)
--save_interval  Save every N steps (default: 100)
--output DIR     Output directory (default: ./output)
```

## Visualization

After simulation completes:

```bash
# Install dependencies
pip install numpy matplotlib

# Create MP4 animation
python visualize.py output_JOBID/ -o wave_animation.mp4

# Create GIF instead
python visualize.py output_JOBID/ -o wave_animation.gif

# With snapshots grid
python visualize.py output_JOBID/ -o wave_animation.mp4 --snapshots

# Custom options
python visualize.py output_JOBID/ \
    --fps 60 \
    --dpi 200 \
    --colormap plasma \
    --skip 2
```

## Local Testing

For local testing without SLURM:

```bash
# Compile
make

# Run with 4 processes
mpirun -np 4 ./wave_simulation --nx 256 --ny 256 --t_end 1.0 --output ./test_output

# Visualize
python visualize.py ./test_output -o test.gif
```
