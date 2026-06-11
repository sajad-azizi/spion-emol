#include "TDSEHamiltonian.hpp"

#include <stdexcept>

namespace mc_tdse {

TDSEHamiltonian::TDSEHamiltonian(const Eigen::VectorXd&  eigvals,
                                 const Eigen::MatrixXcd& d_plus,
                                 double                  omega,
                                 double                  Omega_Rabi,
                                 PulseShape              chi)
    : E_(eigvals), dp_(d_plus),
      omega_(omega), Omega_Rabi_(Omega_Rabi), chi_(std::move(chi))
{
    const int N = static_cast<int>(E_.size());
    if (dp_.rows() != N || dp_.cols() != N) {
        throw std::runtime_error("TDSEHamiltonian: d_plus must be N x N");
    }
    if (!chi_) {
        throw std::runtime_error("TDSEHamiltonian: chi must be a callable");
    }
}

Eigen::MatrixXcd TDSEHamiltonian::at(double t) const {
    const int N = static_cast<int>(E_.size());
    const double half_OR_chi = 0.5 * Omega_Rabi_ * chi_(t);
    Eigen::MatrixXcd M(N, N);

    // M_{fi}(t) = (Ω_R/2) χ(t) [ d^(+)_{fi} e^{ i (E_f - E_i - ω) t}
    //                          + d^(-)_{fi} e^{ i (E_f - E_i + ω) t} ]
    // with d^(-)_{fi} = conj(d^(+)_{if})  (Hermitian conjugate).
    //
    // Compute per-(f,i) phase factor without ever forming M·M.
    if (half_OR_chi == 0.0) {
        M.setZero();
        return M;
    }
    for (int f = 0; f < N; ++f) {
        for (int i = 0; i < N; ++i) {
            const double dE = E_(f) - E_(i);
            const dcompx phase_minus = std::exp(I_unit * (dE - omega_) * t);
            const dcompx phase_plus  = std::exp(I_unit * (dE + omega_) * t);
            const dcompx d_plus_fi   = dp_(f, i);
            const dcompx d_minus_fi  = std::conj(dp_(i, f));     // (d^+)†
            M(f, i) = half_OR_chi * (d_plus_fi * phase_minus
                                     + d_minus_fi * phase_plus);
        }
    }
    return M;
}

}  // namespace mc_tdse
