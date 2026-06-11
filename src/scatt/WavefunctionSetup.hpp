// WavefunctionSetup.hpp -- "prepare to call renormalize" phase.
//
// This module produces everything the coupled-channel renormalized-
// Numerov solver needs as input (but does NOT run the solver):
//
//   1. SolverParams filled with grids, cutoffs, energy, alpha = sqrt(2 pi).
//   2. chi[ir](i, lambda): orbital radial coefficients times r.
//      Reshaped from the preprocessing HDF5 /orbitals/psi_lm and
//      RESCALED by r because the PDF uses the u/r convention (chi_lambda(r)
//      = r * F_lm(r), where F_lm is our SCE coefficient).
//   3. G_coeff: sparse list of non-zero G^R_{mu, lambda, sigma} triplets
//      (PDF eq. 15). Used in the exchange source Q (eq. 18).
//   4. C_coeff: sparse list of non-zero C^R_{sigma, lambda, mu} (eq. 12).
//      For REAL harmonics C and G are the same angular integral up to
//      index relabeling, so we actually reuse one table.
//
// The scattering Potentials object (built earlier) is NOT owned or
// modified here -- it is referenced by the solver directly.
//
// Ported from the prelude of version_0/src/Wavefunctions.cpp
// (wavefunction_multichannel, lines 31-117) with the orbital load path
// switched from raw disk files to our preprocessing HDF5.

#pragma once

#include "angular/Gaunt.hpp"
#include "io/HDF5Reader.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SolverParams.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <vector>

namespace scatt {

// Angular coefficient triplet (mirrors version_0/GauntCoefficients::GauntElement).
// Field names kept identical so old callers port without renaming. Note the
// FIELD SEMANTICS vary by caller (see below).
struct AngTriplet {
    int    a;        // outer index
    int    b;        // middle index
    int    c;        // inner index
    double value;    // G^R or C^R value (they agree for real Ylm)

    // For G_coeff built by build_G_vector(n_mu, n_lambda, n_sigma):
    //     a = mu, b = lambda, c = sigma
    //
    // For C_coeff built by build_C_vector(n_sigma, n_lambda, n_mu):
    //     a = sigma, b = lambda, c = mu
    //
    // The values themselves are identical because for real harmonics
    // G^R_{mu, lambda, sigma} == C^R_{sigma, lambda, mu}
    // (the Gaunt integral is symmetric under all three permutations).
    // We build each list separately anyway because the index spaces
    // differ (mu runs 0..n_mu, sigma runs 0..n_sigma).
};

// One radial point's worth of orbitals. Matrix is (n_occ, n_lambda_cut_orb).
// Entry chi[ir](i, lambda) = r_ir * F^{(i)}_{lambda}(r_ir), i.e. orbital i's
// lambda-th SCE coefficient evaluated at r_ir, times r_ir (u/r convention).
using ChiRadial = std::vector<Eigen::MatrixXd>;

// Bundle produced by WavefunctionSetup::prepare. All members are by-value.
struct SetupBundle {
    SolverParams              params;
    ChiRadial                 chi;           // length = params.n_transition
    std::vector<AngTriplet>   G_coeff;       // non-zero G^R_{mu,lambda,sigma}
    std::vector<AngTriplet>   C_coeff;       // non-zero C^R_{sigma,lambda,mu}
};

class WavefunctionSetup {
public:
    // Defaults follow version_0/Wavefunctions.cpp:
    //   n_r_orbital         = N_grid by default (version_0 uses a coarser
    //                         5001 grid -- we can cut if desired with the
    //                         n_transition field below).
    //   n_transition        = N_grid by default.
    //   l_max_exchange      = min(l_max_continuum, 10)   -- user policy.
    //                         Passing -1 (default) lets prepare() resolve
    //                         it from l_max_continuum. Override for tests.
    //   n_occ               = provided; if 0 falls back to the HDF5's
    //                         "n_occ_alpha" (number of occupied alpha MOs).
    struct Inputs {
        int    n_occ              = 0;     // 0 = auto from HDF5
        int    l_max_exchange     = -1;    // -1 = auto (min(l_cont, 10))
        int    n_transition       = 0;     // 0 = use all Nr points
        double singular_threshold = 1e-14;
        double chi_cutoff         = 1e-15;
        bool   use_disk_checkpoints = false;
        std::string checkpoint_dir;
        int    chunk_size         = 20;
        int    max_memory_mb      = 0;
    };

    // Build the bundle. `data` is the preprocessing HDF5 dump; `params`
    // (which owns l_max_continuum etc.) is stored by reference. `energy`
    // is the scattering energy in Hartree.
    static SetupBundle prepare(const Parameters&       params,
                               const io::PreprocData&  data,
                               double                  energy,
                               const Inputs&           inputs);

    // Convenience overload using default Inputs.
    static SetupBundle prepare(const Parameters&       params,
                               const io::PreprocData&  data,
                               double                  energy) {
        return prepare(params, data, energy, Inputs{});
    }

    // Build G_coeff for a given (n_mu, n_lambda, n_sigma). Mirrors
    // version_0 GauntCoefficients::build_G_vector exactly.
    static std::vector<AngTriplet> build_G_vector(
        int n_mu, int n_lambda, int n_sigma, bool verbose = true);

    // Build C_coeff for a given (n_sigma, n_lambda, n_mu). Mirrors
    // version_0 GauntCoefficients::build_C_vector.
    static std::vector<AngTriplet> build_C_vector(
        int n_sigma, int n_lambda, int n_mu, bool verbose = true);

    // Reshape /orbitals/psi_lm from (n_sce * Nlm_sce, Nr) row-major into
    // chi[ir](i, lambda) with the r multiplication baked in. Truncates to
    // n_occ occupied orbitals (the first occupied ones in molden order)
    // and n_lambda_cut angular channels.
    static ChiRadial load_chi_from_hdf5(const io::PreprocData& data,
                                        int n_occ,
                                        int n_lambda_cut,
                                        int n_transition,
                                        double dr,
                                        bool verbose = true);
};

}  // namespace scatt
