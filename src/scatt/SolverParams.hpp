// SolverParams.hpp -- parameter bundle consumed by the (future)
// renormalized-Numerov coupled-channel solver. Mirrors
// version_0/src/RenormalizedNumerov.hpp::SolverParams field-by-field so
// the eventual port is a one-line struct rename.
//
// Equations this module prepares the inputs for come from
// general_scattering_equation_f.pdf (sajjazizi, Nov 2025):
//
//   (eq. 19)  sum_mu' [psi''_mj delta + 2(E delta - V_mm'(r)) psi_m'j]
//             = -2 alpha sum_i sum_sigma Q^i_{m,sigma}(r) f^i_{sigma,j}(r)
//   (eq. 18)  Q^i_{m,sigma}(r) = sum_lambda G_{m,lambda,sigma} chi^i_lambda(r) / r
//   (eq. 13)  f^i''_{sigma,j} - l_sigma(l_sigma+1)/r^2 f^i_{sigma,j}
//             = -(4 pi / alpha) sum_{lambda,mu} C_{sigma,lambda,mu}
//               * chi^{i*}_lambda(r) psi_{mu,j}(r) / r
// with alpha = sqrt(2 pi). All radial functions use the u/r convention:
// psi, chi, f are r times the full 3D radial amplitude.
//
// Nothing in this header actually runs the solver; it just describes the
// inputs so the prep code has a single source of truth.

#pragma once

#include <cstddef>
#include <string>

namespace scatt {

struct SolverParams {
    // ---- radial grid (for psi, eq. 19) ----------------------------------
    std::size_t n_grid           = 0;     // # radial points for psi
    double      dr               = 0.0;
    double      r_min            = 0.0;   // Numerov starts at r = dr, NOT 0,
                                          // to avoid the l(l+1)/r^2 singular
                                          // point. version_0 sets r_min = dr.
    double      asymptotic_start = 0.0;   // radius at which we switch to the
                                          // asymptotic matching region
                                          // (version_0: (n_grid - 100) * dr)

    // ---- radial grid for chi (may be coarser; see PDF eq. 9) -----------
    int         n_chi_grid       = 0;     // # radial points stored per orbital
    int         n_transition     = 0;     // # orbital radial points used in
                                          // the exchange sum (orbitals decay
                                          // exponentially so no need to
                                          // carry them to r_max of psi)

    // ---- energy -------------------------------------------------------
    double      energy           = 0.0;   // E in Hartree;  k = sqrt(2E), eq. 20

    // ---- angular cutoffs -----------------------------------------------
    //   mu/nu     : continuum channel index, 0 .. (l_cont+1)^2 - 1    eq. 7
    //   sigma     : exchange channel index, 0 .. (l_exch+1)^2 - 1     eq. 8
    //   lambda    : orbital angular index, 0 .. (l_orb+1)^2 - 1       eq. 9
    //   l_orb = l_cont + l_exch (so that G_{mu,lambda,sigma} can be non-zero
    //   via triangle inequality when mu has l_cont and sigma has l_exch).
    int         n_mu             = 0;     // (l_max_continuum + 1)^2
    int         n_sigma          = 0;     // (l_max_exchange  + 1)^2
    int         l_max_continuum  = 0;
    int         l_max_exchange   = 0;     // version_0 default 10
    int         l_max_orbitals   = 0;     // l_max_continuum + l_max_exchange
    int         n_occ            = 0;     // # orbitals in the exchange sum

    // ---- numerical ----------------------------------------------------
    int         n_threads        = 0;       // 0 => omp_get_max_threads
    double      singular_threshold = 1e-14; // pivot tolerance in the
                                            // renormalized-Numerov matrix inv
    double      chi_cutoff         = 1e-15; // drop chi below this (sparsity)

    // ---- storage (renormalized Numerov intermediates) ------------------
    bool        use_disk_checkpoints = false;
    std::string checkpoint_dir;
    int         chunk_size         = 20;    // version_0 chunk for the solver
    int         max_memory_mb      = 0;     // 0 => unlimited

    // ---- mixing constant (eq. 4, 5, 13, 16) ---------------------------
    // alpha = sqrt(2 pi). Kept as a field so downstream code never
    // re-derives it (and gets it wrong by a factor of sqrt(2)).
    double      alpha              = 0.0;   // WavefunctionSetup fills this
                                            // with std::sqrt(2 * M_PI).
};

}  // namespace scatt
