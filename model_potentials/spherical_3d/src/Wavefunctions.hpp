// Wavefunctions.hpp -- bound-state and continuum-state extraction from
// the saved Numerov R-matrix sweeps.  Asymptotic A,B fit uses 3D
// Riccati-Bessel functions S_l, C_l (defined in Angular.hpp), so
// chi_l(r) ~ A·S_l(kr) + B·C_l(kr) and  K = B A^{-1}.
#pragma once

#include "Common.hpp"
#include "Equations.hpp"
#include "Parameters.hpp"

class Wavefunctions {
public:
    Wavefunctions(Equations& eqs, const Parameters& params);

    // Bound state: build (multi-channel) chi_lm(r) by sweeping
    // backward and forward from i_match, gluing at the eigenvector
    // corresponding to the smallest |eigval| of (R_back^{-1} - R_fwd).
    void calculate_eigenfunction(double E, int i_match);
    void Normalization(std::vector<Eigen::VectorXcd>& f);

    // Continuum (single-channel seed at outer boundary): one
    // VectorXcd per channel.  Used for the dipole's "ket" side too.
    void calculate_eigenfunction_continuum(double E);

    // Continuum (matrix; one independent solution per channel) -- this
    // is the analogue of polar_2d's calculate_channel_wavefunction:
    // every column j is the open-channel solution seeded by the j-th
    // unit vector at r_max.  Stored in scattering_eigenfunc.
    void calculate_channel_wavefunction(double E);

    // Asymptotic A,B fit for each (l,m) channel in scattering_eigenfunc.
    void calculate_A_B_matrices(Eigen::MatrixXcd& A, Eigen::MatrixXcd& B,
                                double E);

    // Public: bound-state and continuum.
    std::vector<Eigen::VectorXcd> eigfunc;             // [N_grid] bound chi_lm(r)
    std::vector<Eigen::MatrixXcd> scattering_eigenfunc;// [N_grid] continuum matrix

private:
    // Single-channel-fit helper; returns (a, b) such that
    //   data(r) ≈ a·S_l(k r) + b·C_l(k r) on a window past the
    //   centrifugal barrier.
    std::pair<double, double>
    getCoefficients(const std::vector<double>& data, double k, int l) const;

    Equations&        eqs_;
    const Parameters& params_;
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es_;
};
