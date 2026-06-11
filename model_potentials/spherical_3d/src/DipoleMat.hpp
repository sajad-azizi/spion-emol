// DipoleMat.hpp -- length-form dipole matrix elements between the bound
// initial state and the open continuum channels, for q ∈ {-1, 0, +1}
// (real-Y q-map: x→+1, y→-1, z→0).
//
//   d_q[β] = ∑_{(l,m),(l',m')} G^R(l,m; 1,q; l',m')
//          * ∫_0^∞ dr  chi^*_{cont,(l,m),β}(r) · r · chi_{bound,(l',m')}(r)
//
// then transformed to the ingoing-BC basis via
//   d_q^{in} = (A − iB)^{-T*} · d_q^{real}
// (matches polar_2d's calculate_complex_dipole_matrix_element_ingoingBC).
#pragma once

#include "Common.hpp"
#include "Parameters.hpp"
#include "Wavefunctions.hpp"

class DipoleMat {
public:
    DipoleMat(const Wavefunctions& wfs, const Parameters& params);

    // Returns d_q (size n_channels) in the ingoing-BC basis for each q.
    // Also writes real-axis (real-Y) d_q to file `dipole_real_q<q>.dat`.
    std::vector<dcompx>
    compute(int q, const Eigen::MatrixXcd& A,
                   const Eigen::MatrixXcd& B,
                   double E) const;

    // Velocity-gauge dipole.  Returns d^V_q (size n_channels) in the
    // ingoing-BC basis.  Implementation: in u/r convention (chi = u),
    //   d^V_q[β]  =  Σ_{(l_c,m_c),(l_b,m_b)} G^R(l_c m_c; 1 q; l_b m_b)
    //              · radial_velocity_integral(l_c, l_b, β)
    // with the radial-gradient operator
    //   D_l^(+)(u) = du/dr − (l+1) u/r          for l_c = l_b + 1
    //   D_l^(-)(u) = du/dr +     l u/r          for l_c = l_b − 1
    // (5-point centered FD for du/dr, with one-sided 4th-order at the
    // boundaries).  The angular pre-factor √((4π/3)·(2l_>+1)/(2l_<+1))
    // makes the result satisfy the length-velocity identity
    //   ω · d^L_q  =  −d^V_q
    // numerically (verified by test_dipole_gauge).
    std::vector<dcompx>
    compute_velocity(int q, const Eigen::MatrixXcd& A,
                            const Eigen::MatrixXcd& B,
                            double E) const;

private:
    // Pure real-axis dipole sum (no boundary fix-up beyond Simpson).
    void real_dipole_q(int q, std::vector<dcompx>& d_real) const;
    void real_velocity_q(int q, std::vector<dcompx>& d_real) const;

    // 5-point centered FD with 4th-order one-sided boundaries; df[i] = (df/dr)(r_i).
    void five_point_derivative(const std::vector<dcompx>& f,
                               std::vector<dcompx>& df) const;

    const Wavefunctions& wfs_;
    const Parameters&    params_;
};
