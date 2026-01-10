# Dampened Wave Equation Simulation

A high-performance, MPI-parallel simulation of the dampened wave equation modeling a rectangular drumhead reverberating over time. Designed for execution on HPC clusters using SLURM job scheduling.

## Table of Contents

- [Overview](#overview)
- [Mathematical Model](#mathematical-model)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Building the Simulation](#building-the-simulation)
- [Running the Simulation](#running-the-simulation)
  - [HPC Cluster (SLURM)](#hpc-cluster-slurm)
  - [Local Execution](#local-execution)
- [Command Line Reference](#command-line-reference)
- [Output Format](#output-format)
- [Visualization](#visualization)
  - [Creating Animations](#creating-animations)
  - [Visualization Options](#visualization-options)
  - [Examples](#visualization-examples)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Overview

This simulation models a rectangular drumhead fixed at its edges, subject to an initial displacement (Gaussian pulse at the center). The wave propagates outward and reflects from the boundaries while being attenuated by damping forces. The simulation uses:

- **Finite Difference Method**: Second-order accurate in space and time
- **MPI Parallelization**: Domain decomposition along the Y-axis for distributed computing
- **Binary Output**: Efficient file format for large datasets

## Mathematical Model

The simulation solves the 2D dampened wave equation:

```
∂²u/∂t² + γ(∂u/∂t) = c²(∂²u/∂x² + ∂²u/∂y²)
```

### Variables

| Symbol | Description | Default Value |
|--------|-------------|---------------|
| `u(x,y,t)` | Displacement of the drumhead surface | - |
| `c` | Wave propagation speed | 1.0 |
| `γ` (gamma) | Damping coefficient | 0.1 |
| `Lx`, `Ly` | Domain dimensions | 1.0 × 1.0 |

### Boundary Conditions

- **Fixed edges**: `u = 0` at all boundaries (Dirichlet conditions)
- This models a drumhead clamped at its rectangular frame

### Initial Conditions

- **Gaussian pulse** centered at `(Lx/2, Ly/2)`:
  ```
  u(x, y, 0) = exp(-((x - Lx/2)² + (y - Ly/2)²) / (2σ²))
  ```
  where `σ = 0.05` (pulse width)
- **Zero initial velocity**: `∂u/∂t(x, y, 0) = 0`

### Numerical Scheme

The finite difference discretization uses:

```
u(n+1) = [2u(n) - u(n-1)(1 - γΔt/2) + (cΔt/Δx)²∇²u(n)] / (1 + γΔt/2)
```

where `∇²` is the 5-point Laplacian stencil.

## Project Structure

```
wave_simulation/
├── src/
│   └── wave_simulation.cpp   # Main MPI simulation source code
├── obj/                       # Compiled object files (created by make)
├── Makefile                   # Build configuration
├── submit_job.slurm          # SLURM job submission script
├── visualize.py              # Python visualization script
└── README.md                 # This documentation
```

## Requirements

### For Compilation

- C++17 compatible compiler (GCC 7+, Clang 5+, Intel 2018+)
- MPI implementation (OpenMPI, MPICH, or Intel MPI)
- Make

### For Visualization

- Python 3.8+
- NumPy
- Matplotlib
- FFmpeg (for MP4 output) or Pillow (for GIF output)

### HPC Environment

- SLURM workload manager
- At least 2 compute nodes with 16+ cores each

## Building the Simulation

### On HPC Cluster

```bash
# Navigate to the simulation directory
cd wave_simulation

# Load required modules (names vary by cluster)
module load gcc/11.2.0
module load openmpi/4.1.1

# Build the simulation
make

# For debug build with symbols
make debug
```

### Clean Build

```bash
# Remove compiled files
make clean

# Remove all generated files including outputs
make cleanall
```

## Running the Simulation

### HPC Cluster (SLURM)

#### Basic Submission

```bash
# Submit with default parameters (512x512 grid, 2 nodes, 32 processes)
sbatch submit_job.slurm
```

#### Monitor Job Status

```bash
# Check queue status
squeue -u $USER

# View detailed job info
scontrol show job <JOB_ID>

# Watch job output in real-time
tail -f wave_sim_<JOB_ID>.out
```

#### Cancel a Job

```bash
scancel <JOB_ID>
```

#### Custom SLURM Parameters

Edit `submit_job.slurm` or override at submission:

```bash
# Use different partition
sbatch --partition=debug submit_job.slurm

# Request more time
sbatch --time=02:00:00 submit_job.slurm

# Use 4 nodes instead of 2
sbatch --nodes=4 submit_job.slurm
```

### Local Execution

For development and testing on a local machine or single node:

#### Single Process (Serial)

```bash
# Compile
make

# Run serially (useful for debugging)
./wave_simulation --nx 128 --ny 128 --t_end 0.5 --output ./local_output
```

#### Multiple Processes (Parallel)

```bash
# Run with 4 MPI processes
mpirun -np 4 ./wave_simulation --nx 256 --ny 256 --t_end 1.0 --output ./local_output

# Run with 8 processes
mpirun -np 8 ./wave_simulation --nx 512 --ny 512 --t_end 2.0 --output ./local_output
```

#### Quick Test Run

```bash
# Fast test with small grid and short duration
mpirun -np 2 ./wave_simulation \
    --nx 64 \
    --ny 64 \
    --t_end 0.1 \
    --save_interval 10 \
    --output ./quick_test
```

## Command Line Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--nx` | int | 512 | Number of grid points in X direction |
| `--ny` | int | 512 | Number of grid points in Y direction |
| `--gamma` | float | 0.1 | Damping coefficient (0 = no damping) |
| `--dt` | float | 0.0001 | Time step size (seconds) |
| `--t_end` | float | 2.0 | Total simulation time (seconds) |
| `--save_interval` | int | 100 | Save output every N time steps |
| `--output` | string | ./output | Output directory path |

### Example Configurations

```bash
# High resolution, long simulation
mpirun -np 32 ./wave_simulation \
    --nx 1024 --ny 1024 \
    --t_end 5.0 \
    --save_interval 200 \
    --output ./high_res

# Low damping (more oscillations)
mpirun -np 16 ./wave_simulation \
    --gamma 0.01 \
    --t_end 3.0 \
    --output ./low_damping

# High damping (quick decay)
mpirun -np 16 ./wave_simulation \
    --gamma 0.5 \
    --t_end 1.0 \
    --output ./high_damping

# Fine time resolution for accuracy
mpirun -np 16 ./wave_simulation \
    --dt 0.00005 \
    --save_interval 200 \
    --output ./fine_dt
```

## Output Format

### Binary File Structure

Each output file (`wave_NNNNNN.bin`) contains:

| Offset | Type | Size | Description |
|--------|------|------|-------------|
| 0 | int32 | 4 bytes | nx (grid points in X) |
| 4 | int32 | 4 bytes | ny (grid points in Y) |
| 8 | float64 | 8 bytes | Simulation time |
| 16 | float64[] | nx×ny×8 bytes | Displacement data (row-major) |

### Reading Output in Python

```python
import struct
import numpy as np

def read_wave_file(filepath):
    with open(filepath, 'rb') as f:
        nx = struct.unpack('i', f.read(4))[0]
        ny = struct.unpack('i', f.read(4))[0]
        time = struct.unpack('d', f.read(8))[0]
        data = np.frombuffer(f.read(), dtype=np.float64)
        data = data.reshape((ny, nx))
    return data, time, nx, ny

# Example usage
data, t, nx, ny = read_wave_file('output/wave_000100.bin')
print(f"Grid: {nx}x{ny}, Time: {t:.4f}s")
print(f"Max displacement: {data.max():.6f}")
```

### Reading Output in C++

```cpp
#include <fstream>
#include <vector>

struct WaveFrame {
    int nx, ny;
    double time;
    std::vector<double> data;
};

WaveFrame read_wave_file(const std::string& filepath) {
    WaveFrame frame;
    std::ifstream file(filepath, std::ios::binary);

    file.read(reinterpret_cast<char*>(&frame.nx), sizeof(int));
    file.read(reinterpret_cast<char*>(&frame.ny), sizeof(int));
    file.read(reinterpret_cast<char*>(&frame.time), sizeof(double));

    frame.data.resize(frame.nx * frame.ny);
    file.read(reinterpret_cast<char*>(frame.data.data()),
              frame.data.size() * sizeof(double));

    return frame;
}
```

## Visualization

### Creating Animations

#### Install Dependencies

```bash
# Using pip
pip install numpy matplotlib

# For MP4 support, install FFmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# For GIF support (alternative to FFmpeg)
pip install pillow
```

#### Basic Usage

```bash
# Create MP4 animation (recommended)
python visualize.py ./output_12345/ -o wave_animation.mp4

# Create GIF animation
python visualize.py ./output_12345/ -o wave_animation.gif
```

### Visualization Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | string | wave_animation.mp4 | Output file path |
| `--fps` | int | 30 | Frames per second |
| `--dpi` | int | 150 | Image resolution (dots per inch) |
| `--colormap` | string | viridis | Matplotlib colormap name |
| `--snapshots` | flag | - | Also create a grid of time snapshots |
| `--skip` | int | 1 | Use every Nth frame (reduces file size) |

#### Available Colormaps

- `viridis` - Perceptually uniform, good for accessibility
- `plasma` - Vibrant purple to yellow
- `seismic` - Blue-white-red diverging (good for +/- values)
- `coolwarm` - Soft diverging colormap
- `inferno` - Dark to bright yellow
- `magma` - Dark purple to light yellow

### Visualization Examples

#### High-Quality MP4 for Presentation

```bash
python visualize.py ./output_12345/ \
    -o presentation.mp4 \
    --fps 60 \
    --dpi 300 \
    --colormap viridis
```

#### Quick Preview GIF

```bash
python visualize.py ./output_12345/ \
    -o preview.gif \
    --fps 15 \
    --dpi 100 \
    --skip 5
```

#### Publication-Ready Snapshots

```bash
python visualize.py ./output_12345/ \
    -o figure.mp4 \
    --snapshots \
    --dpi 300 \
    --colormap seismic
```

This creates both `figure.mp4` and `figure_snapshots.png` with a 3×3 grid of time evolution.

#### Custom Colormap for Diverging Data

```bash
python visualize.py ./output_12345/ \
    -o wave_seismic.mp4 \
    --colormap seismic
```

## Performance Tuning

### CFL Condition

For numerical stability, the CFL number should be ≤ 1:

```
CFL = c × Δt / min(Δx, Δy) ≤ 1
```

The simulation prints the CFL number at startup. If you see instabilities (NaN values, exploding amplitudes), reduce `--dt`.

### Optimal Process Count

- Use process counts that evenly divide `ny` for best load balancing
- For 512×512 grid: 2, 4, 8, 16, 32, 64, 128, 256, or 512 processes
- Communication overhead increases with process count; test for your hardware

### Memory Estimation

Memory per process ≈ `3 × (ny/nprocs + 2) × nx × 8 bytes`

For 512×512 grid with 32 processes:
- Per process: ~3 × 18 × 512 × 8 ≈ 220 KB
- Total: ~7 MB

### I/O Optimization

- Increase `--save_interval` to reduce I/O overhead
- Use local scratch storage if available (`$SCRATCH` or `/tmp`)
- For large grids, consider parallel I/O (MPI-IO) extensions

## Troubleshooting

### Common Issues

#### "MPI_Init failed"

```bash
# Ensure MPI module is loaded
module load openmpi

# Check MPI installation
which mpirun
mpirun --version
```

#### "Segmentation fault" during run

- Check CFL condition (reduce `--dt`)
- Ensure grid size is compatible with process count
- Try running with fewer processes

#### "No output files generated"

- Check output directory exists and is writable
- Verify simulation completed (check `.err` file)
- Ensure `--save_interval` isn't larger than total steps

#### Visualization shows blank/white frames

- Check if simulation produced valid data (not NaN)
- Verify binary files are not corrupted
- Try reading a single file manually with Python

### Debug Build

```bash
make debug
mpirun -np 2 ./wave_simulation --nx 64 --ny 64 --t_end 0.01

# With GDB (single process)
./wave_simulation --nx 64 --ny 64 --t_end 0.01

# With Valgrind (memory check)
valgrind --leak-check=full ./wave_simulation --nx 64 --ny 64 --t_end 0.01
```

### Getting Help

1. Check job output files: `wave_sim_<JOB_ID>.out` and `wave_sim_<JOB_ID>.err`
2. Verify module environment: `module list`
3. Test with minimal parameters first
4. Consult your HPC system documentation for cluster-specific issues
