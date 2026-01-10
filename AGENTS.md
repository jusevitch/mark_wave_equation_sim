# Wave Simulation Project

## Project Overview

This repository contains an MPI-parallel simulation of the **dampened wave equation** modeling a rectangular drumhead. The simulation is designed for HPC clusters using SLURM job scheduling.

### Key Components

| Directory/File | Purpose |
|----------------|---------|
| `wave_simulation/` | Main simulation project |
| `wave_simulation/src/wave_simulation.cpp` | C++ MPI simulation code |
| `wave_simulation/Makefile` | Build configuration |
| `wave_simulation/submit_job.slurm` | SLURM job script |
| `wave_simulation/visualize.py` | Python visualization |
| `wave_simulation/README.md` | Detailed documentation |

## Coding Practices

### C++ (Simulation Code)

- **Standard**: C++17
- **Compiler**: Use `mpicxx` (MPI C++ wrapper)
- **Optimization**: `-O3` for release, `-g` for debug
- **Style Guidelines**:
  - Use descriptive variable names (`local_ny`, not `n`)
  - Prefer `std::vector` over raw arrays
  - Use `const` and references where appropriate
  - Document complex algorithms with comments
  - Keep functions focused and under 50 lines when possible

### MPI Practices

- Initialize/finalize MPI in `main()` only
- Use `MPI_COMM_WORLD` for global communication
- Prefer non-blocking communication (`MPI_Isend`/`MPI_Irecv`) for overlap when beneficial
- Current implementation uses blocking `MPI_Sendrecv` for ghost exchange (simpler, sufficient for this scale)
- Always check that array sizes are compatible with process counts

### Python (Visualization)

- **Version**: Python 3.8+
- **Style**: Follow PEP 8
- **Type Hints**: Use type hints for function signatures
- **Dependencies**: numpy, matplotlib (keep minimal)
- Use `argparse` for CLI interfaces
- Prefer pathlib for path operations

### Binary File Format

Output files use a simple binary format:
```
[int32: nx][int32: ny][float64: time][float64[nx*ny]: data]
```

Data is stored in **row-major order** (C-style).

## Architecture Decisions

### Domain Decomposition

- Decomposition is along the **Y-axis only** (1D decomposition)
- Each MPI rank owns `ny/nprocs` rows (plus remainder distribution)
- Ghost rows are exchanged between neighboring ranks

**Rationale**: 1D decomposition is simpler to implement and debug. For the current grid sizes (up to 1024x1024), the communication overhead is acceptable. 2D decomposition would be needed for much larger grids.

### Time Integration

- Uses explicit finite difference scheme
- Three time levels: `u_prev`, `u_curr`, `u_next`
- Array rotation via `std::swap` (efficient, no data copy)

### I/O Strategy

- Binary output for efficiency
- Rank 0 gathers and writes (simple, works for moderate sizes)
- For very large grids, consider MPI-IO parallel writes

## Extending the Simulation

### Adding New Initial Conditions

Modify `WaveSimulation::initialize()` in `wave_simulation.cpp`:

```cpp
void initialize() {
    // Example: Multiple Gaussian pulses
    std::vector<std::pair<double, double>> centers = {
        {0.3, 0.3}, {0.7, 0.7}
    };
    double sigma = 0.05;

    for (int j = 0; j < local_ny; ++j) {
        int global_j = y_start + j;
        double y = global_j * dy;

        for (int i = 0; i < params.nx; ++i) {
            double x = i * dx;
            double value = 0.0;

            for (const auto& [cx, cy] : centers) {
                double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                value += exp(-r2 / (2.0 * sigma * sigma));
            }

            int idx = (j + 1) * params.nx + i;
            u_curr[idx] = value;
            u_prev[idx] = value;
        }
    }
}
```

### Adding New Boundary Conditions

Modify the boundary check in `WaveSimulation::step()`:

```cpp
// Current: Fixed (Dirichlet) boundaries
bool is_boundary = (i == 0 || i == params.nx - 1 ||
                   global_j == 0 || global_j == params.ny - 1);

// Alternative: Periodic boundaries would require
// modifying ghost exchange and index calculations
```

### Adding New Command Line Parameters

1. Add field to `SimParams` struct
2. Add parsing in `main()`:
```cpp
else if (arg == "--new_param" && i + 1 < argc)
    params.new_param = std::stod(argv[++i]);
```

### Adding New Visualization Types

Extend `visualize.py` with new functions:

```python
def create_3d_surface(frames, times, output_path, **kwargs):
    """Create 3D surface animation."""
    from mpl_toolkits.mplot3d import Axes3D
    # Implementation here
```

## Testing

### Quick Local Test

```bash
cd wave_simulation
make
mpirun -np 2 ./wave_simulation --nx 64 --ny 64 --t_end 0.1 --output ./test
python visualize.py ./test -o test.gif
```

### Validation Checks

1. **Conservation**: Without damping (`--gamma 0`), total energy should be conserved
2. **Symmetry**: Centered initial condition should produce symmetric results
3. **Convergence**: Finer grids should converge to analytical solutions for simple cases

### Debug Build

```bash
make debug
# Runs with -g flag, enables assertions
```

## Common Modifications

### Changing Grid Resolution

Update `SimParams` defaults or use command line:
```bash
./wave_simulation --nx 1024 --ny 1024
```

### Changing Physical Parameters

- Wave speed: `--gamma` (affects propagation speed)
- Damping: `--gamma` (0 = undamped, higher = faster decay)
- Domain size: Modify `Lx`, `Ly` in `SimParams` (requires recompile)

### Changing SLURM Configuration

Edit `submit_job.slurm`:
- `--nodes`: Number of compute nodes
- `--ntasks-per-node`: MPI processes per node
- `--time`: Wall clock limit
- `--partition`: Queue/partition name (cluster-specific)

## Performance Considerations

### CFL Stability Condition

```
CFL = c * dt / min(dx, dy) <= 1
```

The simulation prints the CFL number at startup. If CFL > 1, reduce `dt`.

### Scaling

- **Strong scaling**: Fixed problem size, increase processes
- **Weak scaling**: Fixed work per process, increase both
- Current bottleneck: I/O (gather to rank 0)

### Memory

Per-process memory ≈ `3 * (ny/nprocs + 2) * nx * 8 bytes`

## Known Limitations

1. **1D decomposition only**: Limits scalability for very wide grids
2. **Serial I/O**: Rank 0 gathers all data; bottleneck for large grids
3. **Fixed boundaries only**: No periodic or absorbing boundary conditions
4. **Single precision not supported**: Uses double precision throughout

## Future Enhancements

Potential improvements for AI agents to implement:

1. **2D domain decomposition** for better scalability
2. **MPI-IO parallel output** for large grids
3. **Checkpointing/restart** capability
4. **Alternative initial conditions** (plucked string, hammer strike)
5. **Absorbing boundary conditions** (PML or similar)
6. **GPU acceleration** with CUDA or OpenACC
7. **Adaptive time stepping** based on local CFL
8. **VTK/HDF5 output** for ParaView visualization

## Dependencies

### Compilation
- C++17 compiler (GCC 7+, Clang 5+)
- MPI (OpenMPI, MPICH, or Intel MPI)
- Make

### Visualization
- Python 3.8+
- numpy
- matplotlib
- ffmpeg (for MP4) or pillow (for GIF)

### HPC Environment
- SLURM workload manager
- Environment modules (for loading compilers/MPI)
