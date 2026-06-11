// Parameters.hpp -- all tunable numbers for the scattering run.
//
// Critical distinction (matches version_0, see version_0/src/main_mpi.cpp):
//
//   l_max_continuum : angular cutoff of the SCATTERING wavefunction /
//                     channel basis. "channels" = (l_max_continuum + 1)^2.
//                     Typical production value for C8F8: 100.
//
//   Lmax_sce        : angular cutoff of the STATIC POTENTIAL in its
//                     single-center expansion. Typical value for C8F8: 300.
//                     Set by the preprocessing stage.
//
//   l_exp_max       : angular cutoff in the V_sigma(r) expansion that
//                     actually couples two continuum channels. By the
//                     triangle-inequality selection rule,
//                         l_exp_max = 2 * l_max_continuum.
//                     Anything larger in V_sigma is unused -- but smaller
//                     truncates the coupling. This is the knob that sets
//                     the size of the sparse-Gaunt matrix and the
//                     effective (l, m) range of V_sigma read from the HDF5.
//
// The HDF5 preprocessing artifact stores V_sigma(r) up to Lmax_sce (so
// Nlm_sce = (Lmax_sce+1)^2 rows). We only use the first (l_exp_max+1)^2
// rows in the scattering assembly.

#pragma once

#include <cstddef>
#include <string>

namespace scatt {

struct Parameters {
    // Grid.
    double  r_min     = 0.0;
    double  dr        = 0.01;
    std::size_t N_grid = 10001;          // r_max = r_min + (N_grid - 1) * dr

    // Angular cutoffs.
    int  l_max_continuum = 100;          // scattering channel basis
    int  Lmax_sce        = 300;          // SCE of V, read from HDF5
    // Derived: l_exp_max = 2 * l_max_continuum (by triangle inequality).
    //          channels  = (l_max_continuum + 1)^2
    //          n_exp     = (l_exp_max + 1)^2

    // Energy grid for the scattering loop (for later milestones, unused today).
    double  Emin   = -0.5;
    double  Emax   =  0.067;
    double  dE     =  0.01;

    // Numerov solver.
    int  Nroots  = 1000;
    int  divide  = 8;
    int  p       = 3;

    // Runtime.
    int    NTHREADS           = 0;                 // 0 => use omp_get_max_threads()
    std::string hdf5_input_path;                   // path to preprocessing HDF5

    // Storage-mode auto-selection. If channels^2 * Nr * 8 B <=
    // memory_budget_bytes, use MEMORY mode. Otherwise spill to DISK.
    // The default is set loose (16 GB) so small runs stay in memory.
    // Adjust for the target node: LRZ big-mem nodes have 512 GB+ RAM.
    std::size_t  memory_budget_bytes = 16ULL * 1024 * 1024 * 1024;

    // Chunk size in radial points for DISK mode (matches version_0).
    int          chunk_size = 100;

    // ---- helpers ----
    inline int  channels()   const { return (l_max_continuum + 1) * (l_max_continuum + 1); }
    inline int  l_exp_max()  const { return 2 * l_max_continuum; }
    inline int  n_exp()      const { return (l_exp_max() + 1) * (l_exp_max() + 1); }
    inline int  n_lambda()   const { return (Lmax_sce + 1) * (Lmax_sce + 1); }
    inline double r(std::size_t k) const { return r_min + k * dr; }
    inline double r_max() const { return r_min + (N_grid - 1) * dr; }

    // Consistency check. Throws on nonsense.
    void validate() const;
};

}  // namespace scatt
