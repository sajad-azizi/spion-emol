#include "Potentials.hpp"
#include "Angular.hpp"

#include <gsl/gsl_sf_coupling.h>

Potentials::Potentials(const Parameters& params)
    : params_(params)
{
    const int N  = params_.N_grid;
    const int Nc = params_.n_channels;
    VR_.resize(N);
    Veff_.resize(N);
    for (int ir = 0; ir < N; ++ir) {
        VR_[ir]   = Eigen::MatrixXcd::Zero(Nc, Nc);
        Veff_[ir] = Eigen::MatrixXcd::Zero(Nc, Nc);
    }

    // External-parameter mapping: L = iAA * 0.01 (matches polar_2d).
    L_box_ = params_.external_parameter * 0.01;
    if (L_box_ <= 0.0) L_box_ = 1.5;
}

void Potentials::set_potential(const std::string& kind) {
    kind_ = kind;
}

double Potentials::V(double r, double theta, double phi) const {
    const double s = std::sin(theta);
    const double c = std::cos(theta);
    const double x = r * s * std::cos(phi);
    const double y = r * s * std::sin(phi);
    const double z = r * c;

    if (kind_ == "free") {
        return 0.0;
    }
    if (kind_ == "cubic") {
        // 3D analogue of the polar_2d active potential.
        const double L = L_box_;
        return (std::fabs(x) <= 0.5 * L &&
                std::fabs(y) <= 0.5 * L &&
                std::fabs(z) <= 0.5 * L) ? -V0_ : 0.0;
    }
    if (kind_ == "spherical") {
        return (r <= L_box_) ? -V0_ : 0.0;
    }
    if (kind_ == "gaussian") {
        return -V0_ * std::exp(-(r * r) / (a_gauss_ * a_gauss_));
    }
    if (kind_ == "anis_gauss") {
        const double ax = a_gauss_, ay = b_gauss_, az = c_gauss_;
        return -V0_ * std::exp(-(x * x / (ax * ax) +
                                 y * y / (ay * ay) +
                                 z * z / (az * az)));
    }
    if (kind_ == "harmonic") {
        return 0.5 * r * r;
    }
    if (kind_ == "soft_coul") {
        return -1.0 / std::sqrt(r * r + a_soft_ * a_soft_);
    }
    if (kind_ == "h2plus") {
        const double a2 = a_h2_ * a_h2_;
        const double zA = z - 0.5 * R_h2_;
        const double zB = z + 0.5 * R_h2_;
        const double dA = std::sqrt(x * x + y * y + zA * zA + a2);
        const double dB = std::sqrt(x * x + y * y + zB * zB + a2);
        return -1.0 / dA - 1.0 / dB;
    }
    return 0.0;  // unknown -> free
}

void Potentials::build() {
    const int N    = params_.N_grid;
    const int Nc   = params_.n_channels;
    const int Lmax = params_.l_max;
    const int Nth  = params_.N_theta;
    const int Nph  = params_.N_phi;
    const double dr = params_.dr;

    // ---- Special path: H₂⁺ Johnson (hard Coulomb, analytic Legendre).
    //      V_{(k,0),(l,0)}(r) = -2·√((2k+1)(2l+1)) · Σ_n (r_<^{2n}/r_>^{2n+1})
    //                                              · (k, 2n, l; 0, 0, 0)²
    //      Sum over n ≥ 0 with 2n integer, |k-l| ≤ 2n ≤ k+l, k+l even
    //      (parity).  Other (m, m') matrix elements vanish (axial sym +
    //      m=0 sector dominant; we leave higher-m channels at 0 for the
    //      purely σ_g/σ_u problem).  R_proton = R_h2_/2 (Johnson's R).
    if (kind_ == "h2plus_johnson") {
        const double R_proton = 0.5 * R_h2_;   // Johnson's R = ±R_proton.
        for (int ir = 0; ir < N; ++ir) {
            const double r    = ir * dr;
            const double r_lo = std::min(r, R_proton);
            const double r_hi = std::max(r, R_proton);
            Eigen::MatrixXd Mr = Eigen::MatrixXd::Zero(Nc, Nc);
            for (int k = 0; k <= Lmax; ++k) {
                const int idx_k = ang3d::lm_to_idx(k, 0);
                for (int l = 0; l <= Lmax; ++l) {
                    if ((k + l) & 1) continue;       // parity forbids odd k+l
                    const int idx_l = ang3d::lm_to_idx(l, 0);
                    double sum = 0.0;
                    const int n_max = (k + l) / 2;
                    for (int n = 0; n <= n_max; ++n) {
                        const int two_n = 2 * n;
                        if (two_n < std::abs(k - l) || two_n > k + l) continue;
                        // 3j-symbol with all m=0; gsl uses doubled args.
                        const double w3j = gsl_sf_coupling_3j(
                            2 * k, 2 * two_n, 2 * l, 0, 0, 0);
                        if (w3j == 0.0) continue;
                        const double pref =
                            std::pow(r_lo, two_n) /
                            std::pow(r_hi, two_n + 1);
                        sum += pref * w3j * w3j;
                    }
                    if (sum != 0.0) {
                        sum *= -2.0 * std::sqrt(
                            (2.0 * k + 1.0) * (2.0 * l + 1.0));
                    }
                    Mr(idx_k, idx_l) = sum;
                }
            }
            VR_[ir]   = Mr.cast<dcompx>();
            Veff_[ir] = VR_[ir];
            if (r > 0.0) {
                const double inv_2r2 = 1.0 / (2.0 * r * r);
                for (int l = 0; l <= Lmax; ++l) {
                    const double cent = l * (l + 1) * inv_2r2;
                    for (int m = -l; m <= l; ++m) {
                        const int idx = ang3d::lm_to_idx(l, m);
                        Veff_[ir](idx, idx) += dcompx(cent, 0.0);
                    }
                }
            }
        }
        return;
    }


    // --- angular grid -----------------------------------------------------
    std::vector<double> ct, wct;
    ang3d::gauss_legendre(Nth, ct, wct);
    std::vector<double> theta(Nth);
    for (int i = 0; i < Nth; ++i) theta[i] = std::acos(ct[i]);
    const double dphi = 2.0 * M_PI / Nph;
    std::vector<double> phi(Nph);
    for (int j = 0; j < Nph; ++j) phi[j] = j * dphi;

    // --- precompute Y^R on the angular grid -------------------------------
    // Layout: Y[idx][i_theta * Nph + j_phi].
    const std::size_t G = static_cast<std::size_t>(Nth) * Nph;
    std::vector<std::vector<double>> Y(Nc, std::vector<double>(G, 0.0));
    for (int l = 0; l <= Lmax; ++l) {
        for (int m = -l; m <= l; ++m) {
            const int idx = ang3d::lm_to_idx(l, m);
            for (int i = 0; i < Nth; ++i) {
                for (int j = 0; j < Nph; ++j) {
                    Y[idx][i * Nph + j] = ang3d::real_Ylm(l, m, theta[i], phi[j]);
                }
            }
        }
    }

    // --- main radial loop -------------------------------------------------
    // V^R_{a,b}(r) = sum_{i,j} w_θ_i · dphi · Y_a(θ_i, φ_j) · V(r,θ_i,φ_j) · Y_b(θ_i, φ_j).
    // For real V the resulting V^R is symmetric -> compute upper triangle and mirror.
    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < N; ++ir) {
        const double r = ir * dr;

        // Sample V on the angular grid at this r.
        std::vector<double> Vsamp(G);
        for (int i = 0; i < Nth; ++i) {
            for (int j = 0; j < Nph; ++j) {
                Vsamp[i * Nph + j] = V(r, theta[i], phi[j]);
            }
        }

        Eigen::MatrixXd Mr = Eigen::MatrixXd::Zero(Nc, Nc);
        for (int a = 0; a < Nc; ++a) {
            const double* Ya = Y[a].data();
            for (int b = a; b < Nc; ++b) {
                const double* Yb = Y[b].data();
                double sum = 0.0;
                for (int i = 0; i < Nth; ++i) {
                    double row = 0.0;
                    const std::size_t base = static_cast<std::size_t>(i) * Nph;
                    for (int j = 0; j < Nph; ++j) {
                        row += Ya[base + j] * Vsamp[base + j] * Yb[base + j];
                    }
                    sum += wct[i] * row;
                }
                sum *= dphi;
                Mr(a, b) = sum;
                if (a != b) Mr(b, a) = sum;
            }
        }

        VR_[ir]   = Mr.cast<dcompx>();
        Veff_[ir] = VR_[ir];

        // Add centrifugal l(l+1)/(2 r²) on the diagonal.  At r=0 we leave
        // 0 (Numerov starts at i=1 in the seed).
        if (r > 0.0) {
            const double inv_2r2 = 1.0 / (2.0 * r * r);
            for (int l = 0; l <= Lmax; ++l) {
                const double cent = l * (l + 1) * inv_2r2;
                for (int m = -l; m <= l; ++m) {
                    const int idx = ang3d::lm_to_idx(l, m);
                    Veff_[ir](idx, idx) += dcompx(cent, 0.0);
                }
            }
        }
    }
}
