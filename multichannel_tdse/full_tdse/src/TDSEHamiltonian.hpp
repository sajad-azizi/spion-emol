// TDSEHamiltonian.hpp -- builds the time-dependent matrix M(t) that
// drives the interaction-picture amplitude equation
//
//     i ḃ_f(t) = M_{fi}(t) b_i(t)
//
//     M_{fi}(t) = (Ω_R / 2) χ(t) [
//                    d^(+)_{fi} · e^{ i (E_f - E_i - ω) t}
//                  + d^(-)_{fi} · e^{ i (E_f - E_i + ω) t}
//                ]
//
// For Hermiticity (so the propagator is unitary) we use
//
//     d^(-) = (d^(+))†
//
// and only d^(+) needs to be supplied by the caller; d^(-) is derived.
//
// Construction is one-shot; every step's at(t) call is O(N²) phase-
// and-multiply (we never form M·M or precompute U, per the project's
// matrix-vector-only rule).
#pragma once

#include "Common.hpp"
#include "Pulse.hpp"

namespace mc_tdse {

class TDSEHamiltonian {
public:
    // eigvals     : (N) eigenenergies E_i of the pooled basis (atomic units)
    // d_plus      : (N×N) complex matrix d^(+)_{fi} = ⟨f|η^(+)|i⟩
    //               σ⁺ photon raises M_F by 1 -- d^(+)_{fi} ≠ 0 only for
    //               f in M_F+1 block, i in M_F block.
    // omega       : carrier ω (atomic units)
    // Omega_Rabi  : Rabi prefactor Ω_R = μ_B g_S B_RF (atomic units)
    // chi         : envelope χ(t)
    TDSEHamiltonian(const Eigen::VectorXd&       eigvals,
                    const Eigen::MatrixXcd&      d_plus,
                    double                       omega,
                    double                       Omega_Rabi,
                    PulseShape                   chi);

    // Assemble M(t) at instantaneous time t.  Hermitian by construction.
    Eigen::MatrixXcd at(double t) const;

    int  size()       const { return static_cast<int>(E_.size()); }
    const Eigen::VectorXd&  eigvals()  const { return E_;       }
    const Eigen::MatrixXcd& d_plus()   const { return dp_;      }
    double omega()       const { return omega_;       }
    double Omega_Rabi()  const { return Omega_Rabi_;  }
    double chi(double t) const { return chi_(t);      }

private:
    Eigen::VectorXd  E_;
    Eigen::MatrixXcd dp_;     // d^(+)_{fi}
    double           omega_;
    double           Omega_Rabi_;
    PulseShape       chi_;
};

}  // namespace mc_tdse
