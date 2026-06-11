#include "Equations.hpp"
#include "Angular.hpp"

#include <stdexcept>

Equations::Equations(const Potentials& pot, const Parameters& params)
    : pot_(pot), params_(params), p_(params.p)
{
    if (p_ < 1) p_ = 1;
    const int N  = params_.N_grid;
    const int Nc = params_.n_channels;
    Rinv_vec_     .assign(N, Eigen::MatrixXcd::Zero(Nc, Nc));
    Rinv_vec_back_.assign(N, Eigen::MatrixXcd::Zero(Nc, Nc));
    Winv_vec_     .assign(N, Eigen::MatrixXcd::Zero(Nc, Nc));

    In_   = Eigen::MatrixXcd::Identity(Nc, Nc);
    R_    = Eigen::MatrixXcd::Zero(Nc, Nc);
    U_    = Eigen::MatrixXcd::Zero(Nc, Nc);
    Wmat_ = Eigen::MatrixXcd::Zero(Nc, Nc);
    Rinv_ = Eigen::MatrixXcd::Zero(Nc, Nc);
}

void Equations::proper_initialization_R(double E,
                                        Eigen::MatrixXcd& Rinv_out)
{
    const double dr   = params_.dr;
    const double hh12 = dr * dr / 12.0;
    const int    Nc   = params_.n_channels;

    // The seed only modifies the (l=0, m=0) channel of R, using
    // S_0(k_loc r) = sin(k_loc r) (regular Riccati-Bessel at small r).
    // All other (l,m) channels get R = 0 (chi ~ r^{l+1} -> negligible).
    Rinv_ = Eigen::MatrixXcd::Zero(Nc, Nc);

    // V_origin: take V at the radial origin along the polar axis.
    const double V0loc = pot_.V_origin();
    const int    s_idx = ang3d::lm_to_idx(0, 0);

    auto kloc_at = [&](double r) -> dcompx {
        // 3D s-wave local momentum at small r; no centrifugal for l=0.
        // We follow polar_2d: use V at r along (theta=0, phi=0) — for
        // potentials with cubic-well symmetry this equals V at origin.
        const double Vr = pot_.V(r, 0.0, 0.0);
        const double e_loc = E - Vr;
        if (e_loc < 0.0) return I_unit * std::sqrt(2.0 * std::abs(e_loc));
        return std::sqrt(2.0 * std::abs(e_loc));
    };
    (void)V0loc;

    // Regular small-r free s-wave: φ(r) = r · sinc(k r) = sin(kr)/k.
    // Stable in the k→0 limit (φ → r) and identical up to a constant
    // factor 1/k to the polar_2d J_0(kr) seed; the constant cancels in
    // the F2/F1 ratio that defines R_00.
    auto reg = [](dcompx k, double r) -> dcompx {
        const dcompx kr = k * r;
        if (std::abs(kr) < 1e-9) return dcompx(r, 0.0);
        return std::sin(kr) / k;
    };

    for (int i = 1; i < p_; ++i) {
        const double r   = i * dr;
        const double rp1 = (i + 1) * dr;

        Wmat_ = In_ + hh12 * 2.0 * (E * In_ - pot_.Veff(i));
        U_    = 12.0 * Wmat_.inverse() - 10.0 * In_;
        R_    = U_ - Rinv_;

        // Override the (l=0, m=0) diagonal element of R using the
        // analytic regular free-particle step: φ(r) = sin(k r)/k.
        const dcompx k1 = kloc_at(r);
        const dcompx k2 = kloc_at(rp1);

        const dcompx F1 = Wmat_(s_idx, s_idx) * reg(k1, r);

        // Build Wmat at i+1 just for the (s,s) element (same formula).
        const dcompx Wssp1 = dcompx(1.0, 0.0)
            + hh12 * 2.0 * (E - pot_.Veff(i + 1)(s_idx, s_idx));

        const dcompx F2 = Wssp1 * reg(k2, rp1);
        // Guard pathological exact zero (only happens if Wmat(s,s)==0).
        if (std::abs(F1) < 1e-300) {
            R_(s_idx, s_idx) = dcompx(1.0, 0.0);
        } else {
            R_(s_idx, s_idx) = F2 / F1;
        }

        Rinv_              = R_.inverse();
        Rinv_vec_[i]       = Rinv_;
        Winv_vec_[i]       = Wmat_.inverse();
    }

    Rinv_out = Rinv_;
}

std::pair<int, double> Equations::OutwardNodeCounting(double E)
{
    const double dr   = params_.dr;
    const double hh12 = dr * dr / 12.0;
    const int    N    = params_.N_grid;
    const int    Nc   = params_.n_channels;

    proper_initialization_R(E, Rinv_);

    int    node_c   = 0;
    double node_pos = 0.0;

    for (int i = p_; i < N; ++i) {
        Wmat_ = In_ + hh12 * 2.0 * (E * In_ - pot_.Veff(i));
        U_    = 12.0 * Wmat_.inverse() - 10.0 * In_;
        R_    = U_ - Rinv_;
        Rinv_ = R_.inverse();

        // A node in any channel: where W has a positive eigenvalue
        // (kinetic-energy-positive) and R has a negative eigenvalue.
        // (Same heuristic as polar_2d.)
        // Real-symmetric path: take .real() of the (zero-imag) matrices
        // and use SelfAdjointEigenSolver, which is 3-5x faster than
        // ComplexEigenSolver and returns real eigenvalues sorted
        // ascending.  Verified against test_spherical_well_eigenvalue
        // (analytic 1s), test_3d_harmonic_eigenvalue (analytic E=3/2)
        // and test_h2plus_johnson (Johnson 1978 N=4 reference).
        es_R_.compute(R_.real(),    Eigen::EigenvaluesOnly);
        es_W_.compute(Wmat_.real(), Eigen::EigenvaluesOnly);
        for (int m = 0; m < Nc; ++m) {
            if (es_W_.eigenvalues()(m) > 0.0 &&
                es_R_.eigenvalues()(m) < 0.0)
            {
                ++node_c;
                node_pos = i * dr;
            }
        }
    }
    return {node_c, node_pos};
}

void Equations::propagateBackward(double E, int i_match,
                                  Eigen::MatrixXcd& Rmp1_out, bool save)
{
    const double dr   = params_.dr;
    const double hh12 = dr * dr / 12.0;
    const int    N    = params_.N_grid;

    // Outer-boundary BC: chi(r_max) = 0 -> R(r_max) -> infinity, so
    // R_inv(r_max) -> 0.  Start there.
    Rinv_.setZero();

    for (int i = N - 2; i > i_match; --i) {
        Wmat_ = In_ + hh12 * 2.0 * (E * In_ - pot_.Veff(i));
        U_    = 12.0 * Wmat_.inverse() - 10.0 * In_;
        R_    = U_ - Rinv_;
        Rinv_ = R_.inverse();

        if (save) {
            Rinv_vec_[i] = Rinv_;
            Winv_vec_[i] = Wmat_.inverse();
        }
        Rinv_vec_back_[i] = Rinv_;
    }
    Rmp1_out = R_;
}

void Equations::propagateForward(double E, int i_match,
                                 Eigen::MatrixXcd& Rm_out, bool save)
{
    const double dr   = params_.dr;
    const double hh12 = dr * dr / 12.0;

    proper_initialization_R(E, Rinv_);

    for (int i = p_; i <= i_match; ++i) {
        Wmat_ = In_ + hh12 * 2.0 * (E * In_ - pot_.Veff(i));
        U_    = 12.0 * Wmat_.inverse() - 10.0 * In_;
        R_    = U_ - Rinv_;
        Rinv_ = R_.inverse();

        if (save) {
            Rinv_vec_[i] = Rinv_;
            Winv_vec_[i] = Wmat_.inverse();
        }
    }
    Rm_out = R_;
}
