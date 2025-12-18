#ifndef __TIME_SCHEME_HPP__
#define __TIME_SCHEME_HPP__

#ifdef DDGMRES
#include "GMRES.hpp"
#endif

#ifdef GMRES
#include "ILU.hpp"
#endif

#ifdef SpLU
#include "LU.hpp"
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>

class TimeScheme {
public:
    TimeScheme(double timeStep, double GRID_SPACING) : dt(timeStep), dx(GRID_SPACING) {}
    virtual ~TimeScheme() = default;
    
    virtual Eigen::VectorXd advance(
        const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
        const Eigen::VectorXd& phi, 
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        int gridSize) = 0;

    virtual void setupSolver (const Eigen::SparseMatrix<double, Eigen::RowMajor>& A ) = 0;

    virtual Eigen::SparseMatrix<double> GenMatrixA(
        const Eigen::VectorXd& phi,
        const Eigen::VectorXd& Ux, 
        const Eigen::VectorXd& Uy, 
        const Eigen::VectorXd& Uz,
        double spacing,
        int gridSize) = 0;
    
protected:
    const double dt;
    const double dx;

#ifdef EIGEN_SOLVER
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver; 
#endif
#ifdef GMRES
    AsyncILU ilu = AsyncILU(20, 1e-8);
#endif
#ifdef SpLU
    std::unique_ptr<HPSparseLU> solver;
#endif

    inline int getIndex(int x, int y, int z, int GRID_SIZE) const {
        static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
        return x + y * GRID_SIZE + z * GRID_SIZE_SQ;
    }
};

class BackwardEulerScheme : public TimeScheme {
public:
    BackwardEulerScheme(double timeStep, double GRID_SPACING = 1.0) 
        : TimeScheme(timeStep, GRID_SPACING) {}
    
    Eigen::SparseMatrix<double> GenMatrixA(const Eigen::VectorXd& phi, 
        const Eigen::VectorXd& Ux, 
        const Eigen::VectorXd& Uy, 
        const Eigen::VectorXd& Uz,
        double spacing,
        int gridSize) override {
       
            const int n = phi.size();
        
            // Create the system matrix for implicit time stepping
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            tripletList.reserve(7 * n); // Each row has at most 7 non-zero entries (center + 6 neighbors)
            
            // Build the linear system (I - dt*L)phi^{n+1} = phi^n
            // Use thread-local storage to avoid critical section
            const int num_threads = omp_get_max_threads();
            std::vector<std::vector<T>> thread_triplets(num_threads);
            
            #pragma omp parallel
            {
                const int thread_id = omp_get_thread_num();
                thread_triplets[thread_id].reserve(7 * n / num_threads);
                
                #pragma omp for nowait
                for (int idx = 0; idx < n; idx++) {
                    int x = idx % gridSize;
                    int y = (idx / gridSize) % gridSize;
                    int z = idx / (gridSize * gridSize);
                    
                    // Check if this is a boundary point
                    bool isBoundary = (x == 0 || x == gridSize-1 || 
                                      y == 0 || y == gridSize-1 || 
                                      z == 0 || z == gridSize-1);
                    
                    if (isBoundary) {
                        // For boundary points, use identity equation (phi^{n+1} = phi^n)
                        thread_triplets[thread_id].emplace_back(idx, idx, 1.0);
                    } else {
                        // Diagonal term: 1 + dt*regularization
                        double diagTerm = 1.0;
    
                        // Add connections to neighboring cells based on velocity direction
                        // X direction
                        if (Ux(idx) <= 0) { // Flow from right to left
                            diagTerm -= dt * Ux(idx) / spacing; 
                            thread_triplets[thread_id].emplace_back(idx, idx+1, dt * Ux(idx) / spacing);
                        } else if (Ux(idx) > 0) { // Flow from left to right
                            diagTerm += dt * Ux(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx-1, -dt * Ux(idx) / spacing);
                        }
                        
                        // Y direction
                        if (Uy(idx) <= 0) { // Flow from top to bottom
                            diagTerm -= dt * Uy(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx+gridSize, dt * Uy(idx) / spacing);
                        } else if (Uy(idx) > 0) { // Flow from bottom to top
                            diagTerm += dt * Uy(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx-gridSize, -dt * Uy(idx) / spacing);
                        }
                        
                        // Z direction
                        if (Uz(idx) <= 0) { // Flow from front to back
                            diagTerm -= dt * Uz(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx+gridSize*gridSize, dt * Uz(idx) / spacing);
                        } else if (Uz(idx) > 0) { // Flow from back to front
                            diagTerm += dt * Uz(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx-gridSize*gridSize, -dt * Uz(idx) / spacing);
                        }
    
                        thread_triplets[thread_id].emplace_back(idx, idx, diagTerm);
                    }
                }
            }
            
            // Merge thread-local triplet lists
            size_t total_triplets = 0;
            for (const auto& thread_list : thread_triplets) {
                total_triplets += thread_list.size();
            }
            tripletList.reserve(total_triplets);
            
            for (const auto& thread_list : thread_triplets) {
                tripletList.insert(tripletList.end(), thread_list.begin(), thread_list.end());
            }
            
            // Create sparse matrix from triplets
            Eigen::SparseMatrix<double> A(n, n);
            A.setFromTriplets(tripletList.begin(), tripletList.end());
            return A;
    }
    

    Eigen::VectorXd advance(
        const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
        const Eigen::VectorXd& phi, 
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        int gridSize 
    ) override {
        
        Eigen::VectorXd b = phi;
        
#ifdef EIGEN_SOLVER
        Eigen::VectorXd phi_next = solveStandard(b);
#endif

#ifdef GMRES
        Eigen::VectorXd phi_next = preconditionedGMRES(A, b, ilu, 1000, 30, 1e-8);
#endif

#ifdef DDGMRES
        Eigen::VectorXd phi_next = parallelDomainDecompositionGMRES(A, b);
#endif

#ifdef SpLU
        Eigen::VectorXd phi_next = solver->solve(b);
#endif
        
        return phi_next;
    }
    
private:

    void setupSolver (const Eigen::SparseMatrix<double, Eigen::RowMajor>& A ) override {

#ifdef EIGEN_SOLVER
        solver.setMaxIterations(1000);
        solver.setTolerance(1e-8);
        solver.compute(A);
#endif 
#ifdef GMRES
        ilu.compute(A);
#endif

#ifdef SpLU
        solver = std::make_unique<HPSparseLU>(A.rows());
        solver->factorize(A);
#endif
    }
 
#ifdef EIGEN_SOLVER
    Eigen::VectorXd solveStandard(const Eigen::VectorXd& b) {
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed with error: " << solver.error() << std::endl;
            throw std::runtime_error("Decomposition failed");
        }
        
        // Solve the system
        Eigen::VectorXd x = solver.solve(b);
        std::cout << " Iterations " << solver.iterations() << " , error: " << solver.error() << std::endl;
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed with error: " << solver.error() << std::endl;
            std::cerr << "Iterations: " << solver.iterations() << ", estimated error: " << solver.error() << std::endl;
            throw std::runtime_error("Solver failed");
        }
        
        return x;
    }
#endif

};


class implicitCN : public TimeScheme {
public:
    implicitCN(double timeStep, double GRID_SPACING = 1.0) 
        : TimeScheme(timeStep, GRID_SPACING) {}

    Eigen::SparseMatrix<double> GenMatrixA(const Eigen::VectorXd& phi,
            const Eigen::VectorXd& Ux,
            const Eigen::VectorXd& Uy,
            const Eigen::VectorXd& Uz,
            double spacing,
            int gridSize) override {
    
        const int n = phi.size();
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(7 * n);
    
        const int num_threads = omp_get_max_threads();
        std::vector<std::vector<T>> thread_triplets(num_threads);
    
        const double epsilon = 1e-6;
    
        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
            thread_triplets[thread_id].reserve(7 * n / num_threads);
    
            #pragma omp for nowait
            for (int idx = 0; idx < n; idx++) {
                int x = idx % gridSize;
                int y = (idx / gridSize) % gridSize;
                int z = idx / (gridSize * gridSize);
    
                bool isBoundary = (x <= 1 || x >= gridSize - 3 ||
                                   y <= 1 || y >= gridSize - 3 ||
                                   z <= 1 || z >= gridSize - 3);
    
                if (isBoundary) {
                    thread_triplets[thread_id].emplace_back(idx, idx, 1.0 );
                    continue;
                }
    
                double diagTerm = 1.0;
    
                // --- X-direction ---
                int idxL = idx - 1;
                int idxR = idx + 1;
                if (x > 0 && x < gridSize - 1) {
                    double aL = 0.5 * (Ux(idxL) + Ux(idx));
                    double aR = 0.5 * (Ux(idx) + Ux(idxR));
    
                    double fluxL = dt * 0.5 * (aL + std::abs(aL) + epsilon) / (2.0 * spacing);
                    double fluxR = dt * 0.5 * (aR - std::abs(aR) - epsilon) / (2.0 * spacing);
    
                    thread_triplets[thread_id].emplace_back(idx, idxL, -fluxL);
                    thread_triplets[thread_id].emplace_back(idx, idxR, fluxR);
                    diagTerm += fluxL - fluxR;
                }
    
                // --- Y-direction ---
                int idxB = idx - gridSize;
                int idxT = idx + gridSize;
                if (y > 0 && y < gridSize - 1) {
                    double aB = 0.5 * (Uy(idxB) + Uy(idx));
                    double aT = 0.5 * (Uy(idx) + Uy(idxT));
    
                    double fluxB = dt * 0.5 * (aB + std::abs(aB) + epsilon) / (2.0 * spacing);
                    double fluxT = dt * 0.5 * (aT - std::abs(aT) - epsilon) / (2.0 * spacing);
    
                    thread_triplets[thread_id].emplace_back(idx, idxB, -fluxB);
                    thread_triplets[thread_id].emplace_back(idx, idxT, fluxT);
                    diagTerm += fluxB - fluxT;
                }
    
                // --- Z-direction ---
                int idxD = idx - gridSize * gridSize;
                int idxU = idx + gridSize * gridSize;
                if (z > 0 && z < gridSize - 1) {
                    double aD = 0.5 * (Uz(idxD) + Uz(idx));
                    double aU = 0.5 * (Uz(idx) + Uz(idxU));
    
                    double fluxD = dt * 0.5 * (aD + std::abs(aD) + epsilon) / (2.0 * spacing);
                    double fluxU = dt * 0.5 * (aU - std::abs(aU) - epsilon) / (2.0 * spacing);
    
                    thread_triplets[thread_id].emplace_back(idx, idxD, -fluxD);
                    thread_triplets[thread_id].emplace_back(idx, idxU, fluxU);
                    diagTerm += fluxD - fluxU;
                }
    
                thread_triplets[thread_id].emplace_back(idx, idx, diagTerm);
            }
        }
    
        for (const auto& thread_list : thread_triplets) {
            tripletList.insert(tripletList.end(), thread_list.begin(), thread_list.end());
        }
    
        Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        return A;
    }

    Eigen::VectorXd advance(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
                            const Eigen::VectorXd& phi_n, 
                            const Eigen::VectorXd& Ux,
                            const Eigen::VectorXd& Uy,
                            const Eigen::VectorXd& Uz,
                            int gridSize) override {
        const int n = phi_n.size();
        double spacing = dx; 
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);

        #pragma omp parallel for
        for (int idx = 0; idx < n; idx++) {
            int x = idx % gridSize;
            int y = (idx / gridSize) % gridSize;
            int z = idx / (gridSize * gridSize);

            bool isBoundary = (x <= 1 || x >= gridSize - 3 ||
                               y <= 1 || y >= gridSize - 3 ||
                               z <= 1 || z >= gridSize - 3);

            if (isBoundary) {
                b(idx) = phi_n(idx);  // Dirichlet BC
            } else {
                // Face fluxes
                double fxL = computeRoeMUSCLFlux(phi_n, Ux, idx - 1, gridSize, spacing, 0);
                double fxR = computeRoeMUSCLFlux(phi_n, Ux, idx,     gridSize, spacing, 0);
                double fyL = computeRoeMUSCLFlux(phi_n, Uy, idx - gridSize, gridSize, spacing, 1);
                double fyR = computeRoeMUSCLFlux(phi_n, Uy, idx,            gridSize, spacing, 1);
                double fzL = computeRoeMUSCLFlux(phi_n, Uz, idx - gridSize * gridSize, gridSize, spacing, 2);
                double fzR = computeRoeMUSCLFlux(phi_n, Uz, idx,                     gridSize, spacing, 2);

                b(idx) = phi_n(idx) - (dt / (2.0 * spacing)) * ((fxR - fxL) + (fyR - fyL) + (fzR - fzL));
            }
        }

#ifdef EIGEN_SOLVER
        Eigen::VectorXd phi_np1 = solveStandard(b);
#endif

#ifdef GMRES
        Eigen::VectorXd phi_np1 = preconditionedGMRES(A, b, ilu, 1000, 30, 1e-8);
#endif

#ifdef DDGMRES
        Eigen::VectorXd phi_np1 = parallelDomainDecompositionGMRES(A, b);
#endif
#ifdef SpLU
        Eigen::VectorXd phi_np1 = solver->solve(b);
#endif
        return phi_np1;
    }

private:
    double computeRoeMUSCLFlux(const Eigen::VectorXd& phi, const Eigen::VectorXd& U,
                               int idx, int gridSize, double spacing, int direction) {
        const int n = phi.size();
        int stride = (direction == 0) ? 1 : (direction == 1) ? gridSize : gridSize * gridSize;

        int idx_m1 = idx - stride;
        int idx_p1 = idx + stride;
        int idx_p2 = idx + 2 * stride;

        // Spatially aware boundary check
        int x = idx % gridSize;
        int y = (idx / gridSize) % gridSize;
        int z = idx / (gridSize * gridSize);
        if (x <= 1 || x >= gridSize - 3 ||
            y <= 1 || y >= gridSize - 3 ||
            z <= 1 || z >= gridSize - 3 )
            return 0.0;

        // MUSCL reconstruction
        double slope_L = minmod((phi[idx] - phi[idx_m1]) / spacing,
                                (phi[idx_p1] - phi[idx]) / spacing);
        double slope_R = minmod((phi[idx_p1] - phi[idx]) / spacing,
                                (phi[idx_p2] - phi[idx_p1]) / spacing);

        double phi_L = phi[idx] + 0.5 * slope_L * spacing;
        double phi_R = phi[idx_p1] - 0.5 * slope_R * spacing;

        double u_avg = 0.5 * (U[idx] + U[idx_p1]);
        double flux = 0.5 * (u_avg * (phi_L + phi_R) - std::abs(u_avg) * (phi_R - phi_L));
        return flux;
    }


    double minmod(double a, double b) {
        if (a * b <= 0) return 0;
        return std::abs(a) < std::abs(b) ? a : b;
    }


    void setupSolver (const Eigen::SparseMatrix<double, Eigen::RowMajor>& A ) override {
#ifdef EIGEN_SOLVER
        solver.setMaxIterations(1000);
        solver.setTolerance(1e-8);
        solver.compute(A);
#endif 
#ifdef GMRES
        ilu.compute(A);
#endif
#ifdef SpLU
        solver = std::make_unique<HPSparseLU>(A.rows());
        solver->factorize(A);
#endif
    }
   
#ifdef EIGEN_SOLVER
    Eigen::VectorXd solveStandard(const Eigen::VectorXd& b) {
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed" << std::endl;
            throw std::runtime_error("Decomposition failed");
        }
        
        Eigen::VectorXd x = solver.solve(b);
        std::cout << " Iterations " << solver.iterations() << " , error: " << solver.error() << std::endl;
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed. Iterations: " << solver.iterations() 
                      << ", Error: " << solver.error() << std::endl;
            throw std::runtime_error("Solver failed");
        }
        
        return x;
    }
#endif
};


class TVDRK3RoeQUICKScheme : public TimeScheme {
public:
    TVDRK3RoeQUICKScheme(double timeStep, double gridSpacing = 1.0)
        : TimeScheme(timeStep, gridSpacing) {}
    
    Eigen::SparseMatrix<double> GenMatrixA(
        const Eigen::VectorXd&, const Eigen::VectorXd&,
        const Eigen::VectorXd&, const Eigen::VectorXd&,
        double, int) override {
        return Eigen::SparseMatrix<double>(0, 0); // Not used for explicit scheme
    }

    void setupSolver (const Eigen::SparseMatrix<double, Eigen::RowMajor>& A ) override {}
    
    Eigen::VectorXd advance(
        const Eigen::SparseMatrix<double, Eigen::RowMajor>&,
        const Eigen::VectorXd& phi,
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        int gridSize) override {
        // TVD-RK3 scheme with proper coefficients
        Eigen::VectorXd L1 = computeRHS(phi, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi1 = phi + dt * L1;
        
        Eigen::VectorXd L2 = computeRHS(phi1, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi2 = 0.75 * phi + 0.25 * phi1 + 0.25 * dt * L2;
        
        Eigen::VectorXd L3 = computeRHS(phi2, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi_next = (1.0 / 3.0) * phi + (2.0 / 3.0) * phi2 + (2.0 / 3.0) * dt * L3;
        
        return phi_next;
    }

private:
    // Helper function to safely get phi value with boundary handling
    inline double getPhiSafe(const Eigen::VectorXd& phi, int x, int y, int z, int N) const {
        // Clamp coordinates to valid range
        x = std::max(0, std::min(N-1, x));
        y = std::max(0, std::min(N-1, y));
        z = std::max(0, std::min(N-1, z));
        return phi(getIndex(x, y, z, N));
    }

    // Refined QUICK interpolation function
    double quickInterpolationLW(double phi_im2, double phi_im1, double phi_i, double phi_ip1, double phi_ip2,
                             double velocity, bool is_upwind) const {
        if (is_upwind) {
            return (6.0 * phi_im1 + 3.0 * phi_i - phi_im2) / 8.0;
        } else {
            return (6.0 * phi_i + 3.0 * phi_im1 - phi_ip1) / 8.0;
        }
    }

    // Refined QUICK interpolation function
    double quickInterpolationRE(double phi_im2, double phi_im1, double phi_i, double phi_ip1, double phi_ip2, 
                             double velocity, bool is_upwind) const {
        if (is_upwind) {
            return (6.0 * phi_i + 3.0 * phi_ip1 - phi_im1) / 8.0;
        } else {
            return (6.0 * phi_ip1 + 3.0 * phi_i - phi_ip2) / 8.0;
        }
    }

    // Apply flux limiter to prevent oscillations
    double applyFluxLimiter(double phi_im1, double phi_i, double phi_ip1, double phi_ip2) const {
        const double eps = 1e-12;
        
        // Calculate consecutive differences
        double r1 = (phi_i - phi_im1) / (phi_ip1 - phi_i + eps);
        double r2 = (phi_ip1 - phi_i) / (phi_ip2 - phi_ip1 + eps);
        
        // Van Leer limiter
        auto vanLeer = [](double r) {
            return (r + std::abs(r)) / (1.0 + std::abs(r));
        };
        
        double limiter = std::min(vanLeer(r1), vanLeer(r2));
        return std::max(0.0, std::min(1.0, limiter));
    }

    double computeRoeQUICKFlux(const Eigen::VectorXd& phi,
                               const Eigen::VectorXd& velocity,
                               int idx, int N, int dir) {
        // Get 3D coordinates
        int x = idx % N;
        int y = (idx / N) % N;
        int z = idx / (N * N);
        
        // Define offsets for each direction
        int dx_off = (dir == 0) ? 1 : 0;
        int dy_off = (dir == 1) ? 1 : 0;
        int dz_off = (dir == 2) ? 1 : 0;
        
        // Get interface velocity (face-centered)
        int idx_plus = getIndex(x + dx_off, y + dy_off, z + dz_off, N);
        double u_interface = 0.5 * (velocity(idx) + velocity(idx_plus));
        
        // Get QUICK stencil points
        double phi_im2 = getPhiSafe(phi, x - 2*dx_off, y - 2*dy_off, z - 2*dz_off, N);
        double phi_im1 = getPhiSafe(phi, x - dx_off, y - dy_off, z - dz_off, N);
        double phi_i = phi(idx);
        double phi_ip1 = phi(idx_plus);
        double phi_ip2 = getPhiSafe(phi, x + 2*dx_off, y + 2*dy_off, z + 2*dz_off, N);
        
        // Determine flow direction and compute interface values
        double phi_L, phi_R;
        
        if (u_interface >= 0) {
            // Upwind flow: interpolate from left
            phi_L =quickInterpolationLW(phi_im2, phi_im1, phi_i, phi_ip1, phi_ip2, u_interface, true);
            phi_R =quickInterpolationRE(phi_im2, phi_im1, phi_i, phi_ip1, phi_ip2, u_interface, true);
            
            // Apply flux limiting
            double limiter = applyFluxLimiter(phi_im1, phi_i, phi_ip1, phi_ip2);
            phi_L = limiter * phi_L + (1.0 - limiter) * phi_i;
        } else {
            // Downwind flow: interpolate from right
            phi_L = quickInterpolationLW(phi_im2, phi_im1, phi_i, phi_ip1, phi_ip2, u_interface, false);
            phi_R = quickInterpolationRE(phi_im2, phi_im1, phi_i, phi_ip1, phi_ip2, u_interface, false);
            
            // Apply flux limiting
            double limiter = applyFluxLimiter(phi_ip2, phi_ip1, phi_i, phi_im1);
            phi_R = limiter * phi_R + (1.0 - limiter) * phi_ip1;
        }
        
        // Roe flux with proper upwinding
        // F = 0.5 * u * (phi_L + phi_R) - 0.5 * |u| * (phi_R - phi_L)
        double convective_flux = 0.5 * u_interface * (phi_L + phi_R);
        double upwind_flux = 0.5 * std::abs(u_interface) * (phi_R - phi_L);
        
        return convective_flux - upwind_flux;
    }

    Eigen::VectorXd computeRHS(const Eigen::VectorXd& phi,
                               const Eigen::VectorXd& Ux,
                               const Eigen::VectorXd& Uy,
                               const Eigen::VectorXd& Uz,
                               int N) {
        int n = phi.size();
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
        
        #pragma omp parallel for
        for (int idx = 0; idx < n; ++idx) {
            int x = idx % N;
            int y = (idx / N) % N;
            int z = idx / (N * N);
            
            // Skip boundary points where we can't compute proper derivatives
            if (x <= 2 || x >= N - 3 || y <= 2 || y >= N - 3 || z <= 2 || z >= N - 3) {
                rhs(idx) = 0.0;
                continue;
            }
            
            // Compute fluxes using refined Roe + QUICK
            double flux_x_plus = computeRoeQUICKFlux(phi, Ux, idx, N, 0);
            double flux_x_minus = computeRoeQUICKFlux(phi, Ux, 
                getIndex(x-1, y, z, N), N, 0);
            
            double flux_y_plus = computeRoeQUICKFlux(phi, Uy, idx, N, 1);
            double flux_y_minus = computeRoeQUICKFlux(phi, Uy, 
                getIndex(x, y-1, z, N), N, 1);
            
            double flux_z_plus = computeRoeQUICKFlux(phi, Uz, idx, N, 2);
            double flux_z_minus = computeRoeQUICKFlux(phi, Uz, 
                getIndex(x, y, z-1, N), N, 2);
            
            // Conservative finite difference for divergence
            double div_flux = (flux_x_plus - flux_x_minus) / dx +
                              (flux_y_plus - flux_y_minus) / dx +
                              (flux_z_plus - flux_z_minus) / dx;
            
            // Gradient magnitude for curvature term
            double dphi_x = (phi(getIndex(x+1, y, z, N)) - phi(getIndex(x-1, y, z, N))) / (2.0 * dx);
            double dphi_y = (phi(getIndex(x, y+1, z, N)) - phi(getIndex(x, y-1, z, N))) / (2.0 * dx);
            double dphi_z = (phi(getIndex(x, y, z+1, N)) - phi(getIndex(x, y, z-1, N))) / (2.0 * dx);
            double grad_mag = std::sqrt(dphi_x * dphi_x + dphi_y * dphi_y + dphi_z * dphi_z + 1e-12);
            
            // RHS: -∇·F + curvature term
            double curvature = computeMeanCurvature(idx, phi, N);
            rhs(idx) = -div_flux + 0.01 * curvature * grad_mag;
        }
        return rhs;
    }

    double computeMeanCurvature(int idx, const Eigen::VectorXd& phi, int GRID_SIZE) {
        const int x = idx % GRID_SIZE;
        const int y = (idx / GRID_SIZE) % GRID_SIZE;
        const int z = idx / (GRID_SIZE * GRID_SIZE);
        
        // Check boundary - need at least 1 point on each side
        if (x <= 0 || x >= GRID_SIZE - 1 || y <= 0 || y >= GRID_SIZE - 1 || 
            z <= 0 || z >= GRID_SIZE - 1) {
            return 0.0;
        }
        
        // First derivatives using central differences
        const double inv_2dx = 1.0 / (2.0 * dx);
        const double phi_x = (phi[getIndex(x+1, y, z, GRID_SIZE)] - 
                             phi[getIndex(x-1, y, z, GRID_SIZE)]) * inv_2dx;
        const double phi_y = (phi[getIndex(x, y+1, z, GRID_SIZE)] - 
                             phi[getIndex(x, y-1, z, GRID_SIZE)]) * inv_2dx;
        const double phi_z = (phi[getIndex(x, y, z+1, GRID_SIZE)] - 
                             phi[getIndex(x, y, z-1, GRID_SIZE)]) * inv_2dx;
        
        // Gradient magnitude with epsilon for numerical stability
        const double epsilon = 1e-12;
        const double grad_squared = phi_x*phi_x + phi_y*phi_y + phi_z*phi_z + epsilon;
        const double grad_mag = std::sqrt(grad_squared);
        
        // If gradient is negligible, return zero curvature
        if (grad_mag < 1e-10) {
            return 0.0;
        }
        
        // Second derivatives using central differences
        const double inv_dx2 = 1.0 / (dx * dx);
        const double phi_xx = (phi[getIndex(x+1, y, z, GRID_SIZE)] - 
                              2.0 * phi[idx] + 
                              phi[getIndex(x-1, y, z, GRID_SIZE)]) * inv_dx2;
        const double phi_yy = (phi[getIndex(x, y+1, z, GRID_SIZE)] - 
                              2.0 * phi[idx] + 
                              phi[getIndex(x, y-1, z, GRID_SIZE)]) * inv_dx2;
        const double phi_zz = (phi[getIndex(x, y, z+1, GRID_SIZE)] - 
                              2.0 * phi[idx] + 
                              phi[getIndex(x, y, z-1, GRID_SIZE)]) * inv_dx2;
        
        // For points near boundaries, use simplified curvature
        if (x <= 1 || x >= GRID_SIZE - 2 || y <= 1 || y >= GRID_SIZE - 2 || 
            z <= 1 || z >= GRID_SIZE - 2) {
            const double laplacian = phi_xx + phi_yy + phi_zz;
            const double grad_dot_hess_grad = phi_x*phi_x*phi_xx + phi_y*phi_y*phi_yy + phi_z*phi_z*phi_zz;
            const double numerator = laplacian * grad_squared - grad_dot_hess_grad;
            double curvature = numerator / (grad_mag * grad_squared);
            
            // Clamp to prevent instabilities
            const double max_curvature = 5.0 / dx;
            return std::max(-max_curvature, std::min(max_curvature, curvature));
        }
        
        // Full mixed derivatives for interior points
        const double inv_4dx2 = 1.0 / (4.0 * dx * dx);
        const double phi_xy = (phi[getIndex(x+1, y+1, z, GRID_SIZE)] - 
                              phi[getIndex(x+1, y-1, z, GRID_SIZE)] - 
                              phi[getIndex(x-1, y+1, z, GRID_SIZE)] + 
                              phi[getIndex(x-1, y-1, z, GRID_SIZE)]) * inv_4dx2;
        
        const double phi_xz = (phi[getIndex(x+1, y, z+1, GRID_SIZE)] - 
                              phi[getIndex(x+1, y, z-1, GRID_SIZE)] - 
                              phi[getIndex(x-1, y, z+1, GRID_SIZE)] + 
                              phi[getIndex(x-1, y, z-1, GRID_SIZE)]) * inv_4dx2;
        
        const double phi_yz = (phi[getIndex(x, y+1, z+1, GRID_SIZE)] - 
                              phi[getIndex(x, y+1, z-1, GRID_SIZE)] - 
                              phi[getIndex(x, y-1, z+1, GRID_SIZE)] + 
                              phi[getIndex(x, y-1, z-1, GRID_SIZE)]) * inv_4dx2;
        
        // Mean curvature formula
        const double numerator = phi_xx * (phi_y*phi_y + phi_z*phi_z) +
                                phi_yy * (phi_x*phi_x + phi_z*phi_z) +
                                phi_zz * (phi_x*phi_x + phi_y*phi_y) -
                                2.0 * (phi_xy * phi_x * phi_y +
                                      phi_xz * phi_x * phi_z +
                                      phi_yz * phi_y * phi_z);
        
        const double denominator = grad_mag * grad_squared;
        double curvature = numerator / denominator;
        
        // Clamp curvature to prevent numerical instabilities
        const double max_curvature = 5.0 / dx;
        curvature = std::max(-max_curvature, std::min(max_curvature, curvature));
        
        return curvature;
    }
};


#endif
