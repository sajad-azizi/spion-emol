#pragma once

#include "AnalyticSquareWell.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"

#include <complex>
#include <vector>

namespace zepe {

struct ComplexResolventEaOptions {
    // This is a diagnostic/improvement path, not a replacement for the
    // controlled "best" column unless its own LO checks converge.
    bool enabled = true;

    // Compact radial grid used by the off-real-axis inhomogeneous solve.
    // The diagnostics printed by main show whether this compact grid has
    // captured the localized action of the M_F=-5 continuum.
    double R_a0 = 5000.0;
    double dr_a0 = 0.5;

    // Lorentzian broadening in the spectral-density identity
    //   rho(E) = -Im <f|eta (E+i eta-H)^(-1) eta|h> / pi.
    // Decrease eta and increase R to check convergence.
    double eta_kHz = 10.0;

    // Same shifted-log continuum layout used by the finite-window continuum
    // diagnostic in main.cpp.
    int N_seg1 = 401;
    int N_seg2 = 401;
    double eps_kHz = 0.001;
    double E_max_GHz = 50.0;
};

struct ComplexResolventEaResult {
    bool enabled = false;
    ComplexResolventEaOptions opt;

    int radial_N = 0;
    int continuum_N = 0;

    // Scaled RF amplitudes for the M_F=-5 continuum only. The total
    // complex-resolvent ea estimate is exact_bound + c_cont_scaled.
    std::vector<std::complex<double>> c_cont_scaled;

    // LO continuum contribution computed from the same broadened spectral
    // density. This is the important check against the closed-Green result.
    std::vector<std::complex<double>> c_LO_cont_scaled;

    // Source-source spectral-density integral, useful as a completeness check:
    // it should approach the M_F=-5 continuum part of
    // <u_h|eta_m4_m5 eta_m5_m4|u_h> when eta/R/Emax are converged.
    double source_continuum_norm = 0.0;

    // Raw continuum kernels before multiplying by RF factors.
    std::vector<std::complex<double>> exact_cont_kernel;
    std::vector<double> LO_cont_kernel;
};

ComplexResolventEaResult compute_complex_resolvent_ea(
    Potentials& pot_m5,
    Parameters& p_m5,
    Potentials& pot_m4,
    Parameters& p_m4,
    const AnalyticState& halo_state,
    const std::vector<double>& Ef_grid,
    const Eigen::MatrixXd& eta_m4_m5,
    double E_halo,
    double omega,
    double tau0,
    double rf_energy2,
    const ComplexResolventEaOptions& opt);

} // namespace zepe
