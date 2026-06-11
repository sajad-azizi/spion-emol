// SpinAlgebra.hpp
//
// Atomic-physics layer for the 85Rb multichannel Feshbach problem.
//
// Provides:
//   - diagonalization of the single-atom Hamiltonian H_hf + H_Z at field B
//     in the |m_I, m_S> basis (for I=5/2, S=1/2). Output: field-dependent
//     eigenstates of |f, m_f> labels.
//   - enumeration of s-wave bosonic two-body channels at fixed M_F
//   - <alpha | s1 . s2 | beta> matrix in a given M_F block
//   - <alpha | mu_x1 + mu_x2 | beta>  RF dipole matrix elements
//     (sigma_x polarization, which drives Delta m_f = +/-1 per atom)
//
// Entire computation is atomic physics: NO radial dependence.
// Outputs are N_ch x N_ch matrices that feed into Potentials (for s1.s2)
// and DipoleMat (for RF elements).
//
// Convention: field-dependent eigenstate |f, m_f> is labeled by the
// zero-field quantum numbers. At finite B, it is a superposition of two
// |m_I, m_S> states (unless it is a stretched state m_f = +/- 3 with 
// m_S = +/- 1/2).
//
#pragma once

#include "Common.hpp"
#include <map>
#include <set>

struct Atom1Body {
    // Field-dependent single-atom eigenstate, labeled by (f, mf).
    // Decomposition in the |m_I, m_S> basis:
    //   |f, mf> = Sum_k  coeffs[k] * | (mI_list[k], mS_list[k]) >
    int f;
    int mf;                        // integer, since I=5/2 and S=1/2 => f integer
    double E_MHz;                  // eigenvalue in MHz
    std::vector<double> mI_list;   // half-integer
    std::vector<double> mS_list;   // half-integer
    std::vector<double> coeffs;    // real (H is symmetric real in this basis)
};

struct TwoBodyChannel {
    // A symmetric bosonic two-body channel |f1,mf1>|f2,mf2> with (f1,mf1)<=(f2,mf2)
    int f1, mf1;
    int f2, mf2;
    double E_th_MHz;               // threshold = E1 + E2
    std::string label;             // e.g. "|2,-2>|2,-2>"
};

class SpinAlgebra {
public:
    // Build the entire atomic structure at magnetic field B (in Gauss).
    // Diagonalizes single-atom Hamiltonians for all |mf| up to I+S = 3.
    explicit SpinAlgebra(double B_gauss);

    // Get the field-dependent single-atom state by its zero-field labels.
    const Atom1Body& atom(int f, int mf) const;

    // Enumerate all s-wave symmetric two-body channels with given total M_F.
    // Sorted by threshold energy (ascending).
    std::vector<TwoBodyChannel> channels(int MF) const;

    // --- Two-body operator matrix elements ---
    // These return the channel-basis matrix of the operator between two 
    // channel lists (normally the same block for s1.s2, different blocks for RF).

    // <a | s_1 . s_2 | b> for symmetrized two-body states.
    double s1_dot_s2(const TwoBodyChannel& a, const TwoBodyChannel& b) const;

    // <a | (mu_x1 + mu_x2) / mu_B | b> for symmetrized two-body states.
    // Dimensionless; multiply by mu_B * B_RF to get an energy.
    double mu_x_two_body(const TwoBodyChannel& a, const TwoBodyChannel& b) const;

    // Build the full N_ch x N_ch matrix of s1.s2 for a single M_F block.
    Eigen::MatrixXd s1s2_matrix(const std::vector<TwoBodyChannel>& chans) const;

    // Build the N_a x N_b matrix of <a|mu_x|b> between two channel lists
    // (normally a in M_F block, b in M_F±1 block).
    Eigen::MatrixXd rf_matrix(const std::vector<TwoBodyChannel>& a_list,
                              const std::vector<TwoBodyChannel>& b_list) const;

    double B_gauss() const { return B_; }

private:
    double B_;
    // Storage: map from (f, mf) -> Atom1Body
    std::map<std::pair<int,int>, Atom1Body> atoms_;

    // Helpers (all operate on basic quantum numbers, no allocations)
    static double IS_element(double mI1, double mS1, double mI2, double mS2);
    static double s1s2_element(double mS1a, double mS2a, double mS1b, double mS2b);
    static double Sx_single(double mSf, double mIf, double mSi, double mIi);
    static double Ix_single(double mSf, double mIf, double mSi, double mIi);

    // Single-atom mu_x matrix element between two field-dependent states.
    // mu_x / mu_B  =  g_J * <Sx> + g_I * <Ix>
    double mu_x_atom(const Atom1Body& fstate, const Atom1Body& istate) const;

    // Check if two single-atom states are the same (zero overlap otherwise, 
    // since they are orthonormal eigenstates).
    static bool same_atom(int f1, int mf1, int f2, int mf2) {
        return (f1 == f2) && (mf1 == mf2);
    }
};
