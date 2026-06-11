// Potentials.cpp
//
// Implementation of the analytic multichannel square-well potential.
//
#include "Potentials.hpp"

// Factored-out common build routine. 
// If E_ref_MHz is NaN, uses the lowest channel of the block as reference.
// Otherwise uses E_ref_MHz as the absolute reference.
static void build_impl(Potentials* P,
                        Parameters* parameters, SpinAlgebra* spin,
                        double E_ref_MHz,
                        // outputs written into P's members
                        int& N_grid_, double& dr_,
                        double& V_bar_, double& dV_, double& r0_, int& N_ch_,
                        double& smooth_width_, bool& store_grid_,
                        std::vector<TwoBodyChannel>& channels_,
                        Eigen::MatrixXd& s1s2_mat_,
                        Eigen::VectorXd& thresholds_au_,
                        std::vector<Eigen::MatrixXcd>& pot_component)
{
    N_grid_ = parameters->N_grid;
    dr_     = parameters->dr;

    const double V_T = parameters->V_T;
    const double V_S = parameters->V_S;
    V_bar_ = (3.0 * V_T + V_S) / 4.0;
    dV_    = V_S - V_T;
    r0_    = parameters->r0;
    smooth_width_ = parameters->smooth_width;
    store_grid_ = parameters->store_potential_grid;

    auto all_chans = spin->channels(parameters->MF_target);
    if (all_chans.empty()) {
        throw std::runtime_error("Potentials: no channels found for M_F = "
            + std::to_string(parameters->MF_target));
    }

    int N_keep = parameters->N_ch_keep;
    if (N_keep <= 0 || N_keep > (int)all_chans.size()) N_keep = (int)all_chans.size();
    channels_.assign(all_chans.begin(), all_chans.begin() + N_keep);
    N_ch_ = N_keep;

    s1s2_mat_ = spin->s1s2_matrix(channels_);

    // Thresholds relative to the chosen reference
    double E_ref = std::isnan(E_ref_MHz) ? channels_[0].E_th_MHz : E_ref_MHz;
    thresholds_au_ = Eigen::VectorXd::Zero(N_ch_);
    for (int a = 0; a < N_ch_; ++a) {
        thresholds_au_(a) = AU::MHz_to_au(channels_[a].E_th_MHz - E_ref);
    }

    // Singlet/triplet projector decomposition:
    //   V_short = -V_S P_s - V_T P_t
    // with P_s = 1/4 I - s1.s2, P_t = 3/4 I + s1.s2. Expanding:
    //   V_short = -(V_S + 3 V_T)/4 I + (V_S - V_T) s1.s2
    //           = -Vbar I + dV S_{αβ}
    // (NOTE: the sign of the s1.s2 term is +, not -. An earlier version
    // of this code had -dV S, which gave the wrong sign for inter-channel
    // couplings. The diagonal effect is tiny since dV ≪ Vbar, but
    // off-diagonal matrix elements between channels are sign-flipped,
    // which drastically changes closed-block near-threshold states.)
    Eigen::MatrixXd V_in = -V_bar_ * Eigen::MatrixXd::Identity(N_ch_, N_ch_)
                          + dV_ * s1s2_mat_;
    Eigen::MatrixXd thresh_diag = Eigen::MatrixXd::Zero(N_ch_, N_ch_);
    for (int a = 0; a < N_ch_; ++a) thresh_diag(a,a) = thresholds_au_(a);

    if (store_grid_) {
        pot_component.resize(N_grid_);
        for (int k = 0; k < N_grid_; ++k) {
            pot_component[k] = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);
        }

        const double sw = smooth_width_;
        #pragma omp parallel for default(shared) schedule(static)
        for (int ir = 1; ir < N_grid_; ++ir) {
            double r = ir * dr_;
            Eigen::MatrixXd V_r;
            if (sw > 0.0) {
                // Smooth tanh profile: s(r) = tanh((r - r0)/sw)
                // inside weight = (1 - s)/2, outside weight = (1 + s)/2
                double s = std::tanh((r - r0_) / sw);
                double w_in  = 0.5 * (1.0 - s);
                double w_out = 0.5 * (1.0 + s);
                V_r = w_in * (V_in + thresh_diag) + w_out * thresh_diag;
            } else {
                if (r <= r0_) {
                    V_r = V_in + thresh_diag;
                } else {
                    V_r = thresh_diag;
                }
            }
            pot_component[ir] = V_r.cast<dcompx>();
        }
        pot_component[0] = Eigen::MatrixXcd::Zero(N_ch_, N_ch_);
    } else {
        pot_component.clear();
        pot_component.shrink_to_fit();
    }

    (void)P;   // unused; fields are passed by reference
}

Potentials::Potentials(Parameters* parameters, SpinAlgebra* spin)
    : parameters_(parameters), spin_(spin)
{
    build_impl(this, parameters, spin,
                std::numeric_limits<double>::quiet_NaN(),
                N_grid_, dr_, V_bar_, dV_, r0_, N_ch_, smooth_width_, store_grid_,
                channels_, s1s2_mat_, thresholds_au_, pot_component);
}

Potentials::Potentials(Parameters* parameters, SpinAlgebra* spin, double E_global_ref_MHz)
    : parameters_(parameters), spin_(spin)
{
    build_impl(this, parameters, spin, E_global_ref_MHz,
                N_grid_, dr_, V_bar_, dV_, r0_, N_ch_, smooth_width_, store_grid_,
                channels_, s1s2_mat_, thresholds_au_, pot_component);
}

Eigen::MatrixXd Potentials::real_matrix_at_index(int ir) const
{
    if (!pot_component.empty()) {
        return pot_component[ir].real();
    }

    if (ir == 0) {
        return Eigen::MatrixXd::Zero(N_ch_, N_ch_);
    }

    Eigen::MatrixXd V_in = -V_bar_ * Eigen::MatrixXd::Identity(N_ch_, N_ch_)
                          + dV_ * s1s2_mat_;
    Eigen::MatrixXd thresh_diag = Eigen::MatrixXd::Zero(N_ch_, N_ch_);
    for (int a = 0; a < N_ch_; ++a) thresh_diag(a,a) = thresholds_au_(a);

    const double r = ir * dr_;
    if (smooth_width_ > 0.0) {
        const double s = std::tanh((r - r0_) / smooth_width_);
        const double w_in  = 0.5 * (1.0 - s);
        const double w_out = 0.5 * (1.0 + s);
        return w_in * (V_in + thresh_diag) + w_out * thresh_diag;
    }
    return (r <= r0_) ? (V_in + thresh_diag) : thresh_diag;
}

Eigen::MatrixXcd Potentials::matrix_at_index(int ir) const
{
    if (!pot_component.empty()) {
        return pot_component[ir];
    }
    return real_matrix_at_index(ir).cast<dcompx>();
}

Eigen::Matrix2d Potentials::real_matrix2_at_index(int ir) const
{
    if (N_ch_ != 2) {
        throw std::runtime_error("Potentials::real_matrix2_at_index requires N_ch == 2");
    }
    if (!pot_component.empty()) {
        return pot_component[ir].real();
    }
    if (ir == 0) {
        return Eigen::Matrix2d::Zero();
    }

    Eigen::Matrix2d V_in = -V_bar_ * Eigen::Matrix2d::Identity()
                         + dV_ * s1s2_mat_;
    Eigen::Matrix2d thresh_diag = Eigen::Matrix2d::Zero();
    thresh_diag(0,0) = thresholds_au_(0);
    thresh_diag(1,1) = thresholds_au_(1);

    const double r = ir * dr_;
    if (smooth_width_ > 0.0) {
        const double s = std::tanh((r - r0_) / smooth_width_);
        const double w_in  = 0.5 * (1.0 - s);
        const double w_out = 0.5 * (1.0 + s);
        return w_in * (V_in + thresh_diag) + w_out * thresh_diag;
    }
    return (r <= r0_) ? (V_in + thresh_diag) : thresh_diag;
}
