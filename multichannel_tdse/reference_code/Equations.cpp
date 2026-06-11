// Equations.cpp
//
// Renormalized Numerov for the multichannel 3D s-wave radial
// Schrodinger equation. Adapted from the 2D polar code.
//
// The recurrence (B.R. Johnson 1978, J. Chem. Phys. 69, 4678):
//
//   F_{i+1} = U_i F_i − F_{i−1}
//   R_i      = F_{i+1} F_i^{−1}
//
// with
//
//   W_i   = I + (dr^2/12) · 2 (E I − V(r_i))
//   U_i   = 12 W_i^{−1} − 10 I
//
// Forward recurrence: R_i = U_i − R_{i−1}^{−1},  starting from R_0 = 0.
// R_0 = 0 corresponds to u(0) = 0 (regular 3D s-wave solution near origin).
//
// Node-counting theorem: a node occurs at step i iff W_i has a positive
// eigenvalue (classically allowed locally) AND R_i has a negative eigenvalue
// (one of the channel-basis ratios changed sign).
//
#include "Equations.hpp"

Equations::Equations(Potentials* potentials, Parameters* parameters)
    : potentials_(potentials), parameters_(parameters)
{
    N_grid_ = parameters_->N_grid;
    N_ch_   = potentials_->num_channels();
    dr_     = parameters_->dr;
    p_      = parameters_->p;
    mu_     = parameters_->mu;    // reduced mass (in m_e for nuclear problems)

    if (p_ < 1) p_ = 1;

    // Workspace allocation (per-instance, used by every Numerov sweep)
    In_   = Eigen::MatrixXcd::Identity(N_ch_, N_ch_);
    Wmat_ = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);
    U_    = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);
    R_    = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);
    Rinv_ = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);

    // Reconstruction storage. proper_initialization_R writes into
    // Rinv_vector / Winv_vector unconditionally, so these MUST be
    // allocated for any method call (not just save=true paths).
    Rinv_vector.resize(N_grid_);
    Rinv_vector_back.resize(N_grid_);
    Winv_vector.resize(N_grid_);
    for (int k = 0; k < N_grid_; ++k) {
        Rinv_vector[k]      = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);
        Rinv_vector_back[k] = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);
        Winv_vector[k]      = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);
    }
}

// ============================================================
// Initialize the ratio matrix near the origin
// ============================================================
// For 3D s-wave there is no centrifugal barrier, so the regular
// solution is u_α(r) ∝ r near r = 0, and the Renormalized Numerov
// ratio matrix can be started from zero. After p_ steps the ratio
// matrix has relaxed to the correct value for the local coupled
// potential.
//
void Equations::proper_initialization_R(double Energy, Eigen::MatrixXcd& resRinv)
{
    const double hh12 = dr_ * dr_ / 12.0;

    // Reset to zero (this is critical — the stored Rinv_ would otherwise
    // carry state from the previous call)
    Rinv_.setZero();

    for (int i = 1; i < p_; ++i) {
        Wmat_ = In_ + hh12 * 2.0 * mu_ * (Energy * In_ - potentials_->matrix_at_index(i));
        U_ = 12.0 * Wmat_.inverse() - 10.0 * In_;
        R_ = U_ - Rinv_;
        Rinv_ = R_.inverse();

        Rinv_vector[i] = Rinv_;
        Winv_vector[i] = Wmat_.inverse();
    }
    resRinv = Rinv_;
}

// ============================================================
// Outward node counting (bound-state bisection driver)
// ============================================================
std::pair<int, double> Equations::OutwardNodeCounting(double Energy)
{
    const double hh12 = dr_ * dr_ / 12.0;

    proper_initialization_R(Energy, Rinv_);

    int    node_c = 0;
    double node_pos = 0.0;

    for (int i = p_; i < N_grid_; ++i) {
        Wmat_ = In_ + hh12 * 2.0 * mu_ * (Energy * In_ - potentials_->matrix_at_index(i));
        U_ = 12.0 * Wmat_.inverse() - 10.0 * In_;
        R_ = U_ - Rinv_;
        Rinv_ = R_.inverse();

        es_.compute(R_, Eigen::EigenvaluesOnly);
        es1_.compute(Wmat_, Eigen::EigenvaluesOnly);

        for (int m = 0; m < N_ch_; ++m) {
            if (es1_.eigenvalues().real()(m) > 0.0
                && es_.eigenvalues().real()(m) < 0.0)
            {
                ++node_c;
                node_pos = i * dr_;
            }
        }
    }
    return {node_c, node_pos};
}

// ============================================================
// Forward propagation (origin -> matching point)
// ============================================================
void Equations::propagateForward(double Energy, int i_match,
                                  Eigen::MatrixXcd& resRm, bool save)
{
    const double hh12 = dr_ * dr_ / 12.0;

    proper_initialization_R(Energy, Rinv_);

    for (int i = p_; i <= i_match; ++i) {
        Wmat_ = In_ + hh12 * 2.0 * mu_ * (Energy * In_ - potentials_->matrix_at_index(i));
        U_ = 12.0 * Wmat_.inverse() - 10.0 * In_;
        R_ = U_ - Rinv_;
        Rinv_ = R_.inverse();

        if (save) {
            Rinv_vector[i] = Rinv_;
            Winv_vector[i] = Wmat_.inverse();
        }
    }
    resRm = R_;
}

// ============================================================
// Backward propagation (box edge -> matching point)
// ============================================================
// Starts from u(r = L) = 0, which gives Rinv = 0 on the outer end
// (equivalent to R = infinity). In the reversed recurrence:
//
//   R_{i-1}^{-1} = U_i - R_i
//
// we initialize R_{N-1} = infinity, i.e. Rinv_{N-1} = 0.
//
void Equations::propagateBackward(double Energy, int i_match,
                                   Eigen::MatrixXcd& resRmp1, bool save)
{
    const double hh12 = dr_ * dr_ / 12.0;

    // Reset for backward sweep. Rinv = 0 at the outer boundary corresponds
    // to the Dirichlet condition u(r = L) = 0 (R = infinity, hence Rinv = 0).
    Rinv_.setZero();

    for (int i = N_grid_ - 2; i > i_match; --i) {
        Wmat_ = In_ + hh12 * 2.0 * mu_ * (Energy * In_ - potentials_->matrix_at_index(i));
        U_ = 12.0 * Wmat_.inverse() - 10.0 * In_;
        R_ = U_ - Rinv_;
        Rinv_ = R_.inverse();

        if (save) {
            Rinv_vector[i] = Rinv_;
            Winv_vector[i] = Wmat_.inverse();
        }
        Rinv_vector_back[i] = Rinv_;
    }
    resRmp1 = R_;
}
