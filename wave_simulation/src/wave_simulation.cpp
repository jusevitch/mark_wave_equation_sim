/**
 * Dampened Wave Equation Simulation for a Rectangular Drumhead
 *
 * Solves: ∂²u/∂t² + γ(∂u/∂t) = c²(∂²u/∂x² + ∂²u/∂y²)
 *
 * Using finite difference method with MPI domain decomposition
 * Boundary conditions: Fixed edges (u = 0 at boundaries)
 */

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

// Simulation parameters
struct SimParams {
    int nx = 512;           // Grid points in x
    int ny = 512;           // Grid points in y
    double Lx = 1.0;        // Domain length in x
    double Ly = 1.0;        // Domain length in y
    double c = 1.0;         // Wave speed
    double gamma = 0.1;     // Damping coefficient
    double dt = 0.0001;     // Time step
    double t_end = 2.0;     // End time
    int save_interval = 100; // Save every N steps
};

class WaveSimulation {
private:
    SimParams params;
    int rank, size;
    int local_ny;           // Local grid size in y direction
    int y_start, y_end;     // Global y indices for this rank

    double dx, dy;
    double c2_dt2_dx2, c2_dt2_dy2;
    double gamma_dt;
    double denom;

    std::vector<double> u_curr;     // Current time step
    std::vector<double> u_prev;     // Previous time step
    std::vector<double> u_next;     // Next time step

    // Ghost rows for MPI communication
    std::vector<double> ghost_top;
    std::vector<double> ghost_bottom;

public:
    WaveSimulation(const SimParams& p, int mpi_rank, int mpi_size)
        : params(p), rank(mpi_rank), size(mpi_size) {

        // Calculate grid spacing
        dx = params.Lx / (params.nx - 1);
        dy = params.Ly / (params.ny - 1);

        // Precompute coefficients for finite difference
        c2_dt2_dx2 = (params.c * params.c * params.dt * params.dt) / (dx * dx);
        c2_dt2_dy2 = (params.c * params.c * params.dt * params.dt) / (dy * dy);
        gamma_dt = params.gamma * params.dt;
        denom = 1.0 / (1.0 + gamma_dt / 2.0);

        // Domain decomposition along y-axis
        int base_rows = params.ny / size;
        int remainder = params.ny % size;

        if (rank < remainder) {
            local_ny = base_rows + 1;
            y_start = rank * local_ny;
        } else {
            local_ny = base_rows;
            y_start = rank * base_rows + remainder;
        }
        y_end = y_start + local_ny;

        // Allocate arrays (including space for ghost rows)
        int total_rows = local_ny + 2;  // +2 for ghost rows
        u_curr.resize(total_rows * params.nx, 0.0);
        u_prev.resize(total_rows * params.nx, 0.0);
        u_next.resize(total_rows * params.nx, 0.0);

        ghost_top.resize(params.nx, 0.0);
        ghost_bottom.resize(params.nx, 0.0);

        if (rank == 0) {
            std::cout << "Wave Simulation initialized:" << std::endl;
            std::cout << "  Grid: " << params.nx << " x " << params.ny << std::endl;
            std::cout << "  Domain: " << params.Lx << " x " << params.Ly << std::endl;
            std::cout << "  Wave speed: " << params.c << std::endl;
            std::cout << "  Damping: " << params.gamma << std::endl;
            std::cout << "  dt: " << params.dt << std::endl;
            std::cout << "  CFL number: " << params.c * params.dt / std::min(dx, dy) << std::endl;
            std::cout << "  Processes: " << size << std::endl;
        }
    }

    // Initialize with a Gaussian pulse in the center
    void initialize() {
        double cx = params.Lx / 2.0;
        double cy = params.Ly / 2.0;
        double sigma = 0.05;

        for (int j = 0; j < local_ny; ++j) {
            int global_j = y_start + j;
            double y = global_j * dy;

            for (int i = 0; i < params.nx; ++i) {
                double x = i * dx;

                double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                double value = exp(-r2 / (2.0 * sigma * sigma));

                // Index includes ghost row offset (+1)
                int idx = (j + 1) * params.nx + i;
                u_curr[idx] = value;
                u_prev[idx] = value;  // Start at rest
            }
        }
    }

    // Exchange ghost rows with neighboring processes
    void exchange_ghosts() {
        MPI_Status status;

        // Send bottom row to rank-1, receive top ghost from rank-1
        if (rank > 0) {
            // Send my first real row (index 1) to rank-1
            MPI_Sendrecv(&u_curr[1 * params.nx], params.nx, MPI_DOUBLE, rank - 1, 0,
                        &u_curr[0], params.nx, MPI_DOUBLE, rank - 1, 1,
                        MPI_COMM_WORLD, &status);
        }

        // Send top row to rank+1, receive bottom ghost from rank+1
        if (rank < size - 1) {
            // Send my last real row to rank+1
            MPI_Sendrecv(&u_curr[local_ny * params.nx], params.nx, MPI_DOUBLE, rank + 1, 1,
                        &u_curr[(local_ny + 1) * params.nx], params.nx, MPI_DOUBLE, rank + 1, 0,
                        MPI_COMM_WORLD, &status);
        }
    }

    // Perform one time step using the finite difference scheme
    void step() {
        exchange_ghosts();

        for (int j = 1; j <= local_ny; ++j) {
            int global_j = y_start + (j - 1);

            for (int i = 0; i < params.nx; ++i) {
                int idx = j * params.nx + i;

                // Apply fixed boundary conditions
                bool is_boundary = (i == 0 || i == params.nx - 1 ||
                                   global_j == 0 || global_j == params.ny - 1);

                if (is_boundary) {
                    u_next[idx] = 0.0;
                } else {
                    // Finite difference stencil
                    double laplacian_x = u_curr[idx - 1] - 2.0 * u_curr[idx] + u_curr[idx + 1];
                    double laplacian_y = u_curr[idx - params.nx] - 2.0 * u_curr[idx] + u_curr[idx + params.nx];

                    // Dampened wave equation update
                    u_next[idx] = denom * (
                        2.0 * u_curr[idx] - u_prev[idx] * (1.0 - gamma_dt / 2.0) +
                        c2_dt2_dx2 * laplacian_x +
                        c2_dt2_dy2 * laplacian_y
                    );
                }
            }
        }

        // Rotate arrays
        std::swap(u_prev, u_curr);
        std::swap(u_curr, u_next);
    }

    // Save current state to binary file
    void save_state(int step_num, const std::string& output_dir) {
        // Gather all data to rank 0
        std::vector<double> global_data;
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        if (rank == 0) {
            global_data.resize(params.nx * params.ny);
        }

        // Calculate receive counts and displacements
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            int r_base = params.ny / size;
            int r_remainder = params.ny % size;
            int r_local_ny = (r < r_remainder) ? r_base + 1 : r_base;
            recvcounts[r] = r_local_ny * params.nx;
            displs[r] = offset;
            offset += recvcounts[r];
        }

        // Copy local data (excluding ghost rows)
        std::vector<double> local_data(local_ny * params.nx);
        for (int j = 0; j < local_ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                local_data[j * params.nx + i] = u_curr[(j + 1) * params.nx + i];
            }
        }

        MPI_Gatherv(local_data.data(), local_ny * params.nx, MPI_DOUBLE,
                   global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::ostringstream filename;
            filename << output_dir << "/wave_" << std::setfill('0') << std::setw(6) << step_num << ".bin";

            std::ofstream file(filename.str(), std::ios::binary);
            if (file) {
                // Write header
                file.write(reinterpret_cast<const char*>(&params.nx), sizeof(int));
                file.write(reinterpret_cast<const char*>(&params.ny), sizeof(int));
                double time = step_num * params.dt;
                file.write(reinterpret_cast<const char*>(&time), sizeof(double));

                // Write data
                file.write(reinterpret_cast<const char*>(global_data.data()),
                          global_data.size() * sizeof(double));
                file.close();
            }
        }
    }

    // Run the full simulation
    void run(const std::string& output_dir) {
        int num_steps = static_cast<int>(params.t_end / params.dt);

        if (rank == 0) {
            std::cout << "Running " << num_steps << " time steps..." << std::endl;
        }

        // Save initial state
        save_state(0, output_dir);

        for (int n = 1; n <= num_steps; ++n) {
            step();

            if (n % params.save_interval == 0) {
                save_state(n, output_dir);

                if (rank == 0) {
                    double progress = 100.0 * n / num_steps;
                    std::cout << "Progress: " << std::fixed << std::setprecision(1)
                             << progress << "% (step " << n << "/" << num_steps << ")" << std::endl;
                }
            }
        }

        if (rank == 0) {
            std::cout << "Simulation complete!" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
    SimParams params;
    std::string output_dir = "./output";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--nx" && i + 1 < argc) params.nx = std::stoi(argv[++i]);
        else if (arg == "--ny" && i + 1 < argc) params.ny = std::stoi(argv[++i]);
        else if (arg == "--gamma" && i + 1 < argc) params.gamma = std::stod(argv[++i]);
        else if (arg == "--dt" && i + 1 < argc) params.dt = std::stod(argv[++i]);
        else if (arg == "--t_end" && i + 1 < argc) params.t_end = std::stod(argv[++i]);
        else if (arg == "--save_interval" && i + 1 < argc) params.save_interval = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_dir = argv[++i];
    }

    // Create output directory (rank 0 only)
    if (rank == 0) {
        std::string cmd = "mkdir -p " + output_dir;
        system(cmd.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Create and run simulation
    WaveSimulation sim(params, rank, size);
    sim.initialize();

    double start_time = MPI_Wtime();
    sim.run(output_dir);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Total runtime: " << end_time - start_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
