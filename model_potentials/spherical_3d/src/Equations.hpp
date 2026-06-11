// Equations.hpp -- Johnson R-matrix Numerov propagator for the 3D
// coupled-channel radial Schrödinger equation:
//
//   chi'' + Q(r) chi = 0,    Q(r) = 2(E I − V_eff(r)),
//
// with V_eff including the centrifugal l(l+1)/(2 r²) on the diagonal.
// Same Johnson form as polar_2d:
//
//   Wmat_n = I + (h²/12)·2·(E I − V_eff_n) = I + (h²/6) Q_n
//   U_n    = 12 Wmat_n^{-1} − 10 I
//   R_n    = U_{n-1} − R_{n-1}^{-1}
//
// proper_initialization_R(E) seeds the (l=0, m=0) channel of R using
// S_0(k_loc r) = sin(k_loc r) (regular Riccati-Bessel at small r);
// other channels start with R = 0 (chi small, ~ r^{l+1}).
#pragma once

#include "Common.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"

class Equations {
public:
    Equations(const Potentials& pot, const Parameters& params);

    void proper_initialization_R(double E, Eigen::MatrixXcd& Rinv_out);
    std::pair<int, double> OutwardNodeCounting(double E);

    // Save: also fill Rinv_vector_/Winv_vector_ at every step.
    void propagateForward (double E, int i_match,
                           Eigen::MatrixXcd& Rm_out,    bool save);
    void propagateBackward(double E, int i_match,
                           Eigen::MatrixXcd& Rmp1_out,  bool save);

    // Public access for downstream stages.
    std::vector<Eigen::MatrixXcd>& Rinv_vector()      { return Rinv_vec_; }
    std::vector<Eigen::MatrixXcd>& Rinv_vector_back() { return Rinv_vec_back_; }
    std::vector<Eigen::MatrixXcd>& Winv_vector()      { return Winv_vec_; }

    int channels() const { return params_.n_channels; }
    int N_grid()   const { return params_.N_grid; }
    double dr()    const { return params_.dr; }

    // s-wave (l=0, m=0) diagonal of V_eff at radial index ir.  This is
    // the spherically-averaged potential (centrifugal is 0 for l=0), and
    // is the quantity that determines the classical turning point of an
    // s-wave bound state: V_s(r) = E.  V_eff is real-symmetric (V is
    // real); we cast the (0,0) complex entry's real part.
    double Veff_swave(int ir) const {
        return pot_.Veff(ir)(0, 0).real();
    }

private:
    const Potentials& pot_;
    const Parameters& params_;
    int p_;     // # of seed steps

    // Per-step storage (sized to N_grid).
    std::vector<Eigen::MatrixXcd> Rinv_vec_;
    std::vector<Eigen::MatrixXcd> Rinv_vec_back_;
    std::vector<Eigen::MatrixXcd> Winv_vec_;

    // Working scratch (kept as members to avoid allocation churn).
    Eigen::MatrixXcd Rinv_;   // running Rinv during propagation
    Eigen::MatrixXcd In_;
    Eigen::MatrixXcd R_;
    Eigen::MatrixXcd U_;
    Eigen::MatrixXcd Wmat_;

    // For real-V models the matrices Wmat_ and R_ are mathematically
    // real-symmetric (V real => V_R sym => V_eff sym => Wmat sym =>
    // Wmat^{-1} sym => U = 12 Wmat^{-1} - 10 I sym => R sym).
    // SelfAdjointEigenSolver<MatrixXd> with .real() cast at the call
    // site is 3-5x faster than ComplexEigenSolver<MatrixXcd> and
    // returns real eigenvalues sorted ascending.  Verified by the
    // bound-state regression suite.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_R_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_W_;
};
