// SpinAlgebra.cpp
//
// Implementation of the atomic physics layer for 85Rb multichannel Feshbach.
//
#include "SpinAlgebra.hpp"

// ============================================================
// Helpers
// ============================================================

// <mI1, mS1 | I . S | mI2, mS2> for arbitrary I, S
//
//   I . S = I_z S_z + (I+ S- + I- S+) / 2
//
// <mI1|I+|mI2> = sqrt(I(I+1) - mI2(mI2+1)) delta_{mI1, mI2+1}
// <mS1|S+|mS2> = sqrt(S(S+1) - mS2(mS2+1)) delta_{mS1, mS2+1}
// etc.
double SpinAlgebra::IS_element(double mI1, double mS1, double mI2, double mS2)
{
    using AU::I_nuc;
    using AU::S_elec;
    double val = 0.0;

    // I_z S_z
    if (std::abs(mI1 - mI2) < 1e-10 && std::abs(mS1 - mS2) < 1e-10) {
        val += mI2 * mS2;
    }
    // I+ S-  (raises mI, lowers mS)
    if (std::abs(mI1 - (mI2 + 1.0)) < 1e-10 && std::abs(mS1 - (mS2 - 1.0)) < 1e-10) {
        val += 0.5 * std::sqrt(I_nuc*(I_nuc+1.0) - mI2*(mI2+1.0))
                    * std::sqrt(S_elec*(S_elec+1.0) - mS2*(mS2-1.0));
    }
    // I- S+  (lowers mI, raises mS)
    if (std::abs(mI1 - (mI2 - 1.0)) < 1e-10 && std::abs(mS1 - (mS2 + 1.0)) < 1e-10) {
        val += 0.5 * std::sqrt(I_nuc*(I_nuc+1.0) - mI2*(mI2-1.0))
                    * std::sqrt(S_elec*(S_elec+1.0) - mS2*(mS2+1.0));
    }
    return val;
}

// <mS1a, mS2a | s1.s2 | mS1b, mS2b> for two spin-1/2 particles
// s1.s2 = s1z s2z + (s1+ s2- + s1- s2+)/2
// For s = 1/2: sqrt(3/4 - mS(mS+1)) gives 1 for the only allowed flip
// (mS = -1/2 -> +1/2) and 0 otherwise.
double SpinAlgebra::s1s2_element(double mS1a, double mS2a, double mS1b, double mS2b)
{
    double val = 0.0;
    // s1z s2z
    if (std::abs(mS1a - mS1b) < 1e-10 && std::abs(mS2a - mS2b) < 1e-10) {
        val += mS1a * mS2a;
    }
    // s1+ s2-
    if (std::abs(mS1a - (mS1b + 1.0)) < 1e-10 && std::abs(mS2a - (mS2b - 1.0)) < 1e-10) {
        val += 0.5 * std::sqrt(0.75 - mS1b*(mS1b+1.0))
                    * std::sqrt(0.75 - mS2b*(mS2b-1.0));
    }
    // s1- s2+
    if (std::abs(mS1a - (mS1b - 1.0)) < 1e-10 && std::abs(mS2a - (mS2b + 1.0)) < 1e-10) {
        val += 0.5 * std::sqrt(0.75 - mS1b*(mS1b-1.0))
                    * std::sqrt(0.75 - mS2b*(mS2b+1.0));
    }
    return val;
}

// Single-atom S_x matrix element in the |mI, mS> basis
//
// S_x = (S+ + S-) / 2
// <mI', mS'|S_x|mI, mS> = (1/2) [<mS'|S+|mS> + <mS'|S-|mS>] delta_{mI', mI}
// For S=1/2: the nonzero element is <+/-1/2 | S_x | -/+ 1/2> = 1/2.
double SpinAlgebra::Sx_single(double mSf, double mIf, double mSi, double mIi)
{
    if (std::abs(mIf - mIi) > 1e-10) return 0.0;
    // For S = 1/2, the only nonzero matrix element of S_x between two
    // m_S states is the flip |+1/2> <-> |-1/2>, with value 1/2.
    if (std::abs(mSf - mSi - 1.0) < 1e-10) return 0.5;   // mSf = mSi + 1
    if (std::abs(mSf - mSi + 1.0) < 1e-10) return 0.5;   // mSf = mSi - 1
    return 0.0;
}

// Single-atom I_x matrix element in the |mI, mS> basis
//
// I_x = (I+ + I-) / 2
// <mI'|I+|mI> = sqrt(I(I+1) - mI(mI+1)) delta_{mI', mI+1}
double SpinAlgebra::Ix_single(double mSf, double mIf, double mSi, double mIi)
{
    using AU::I_nuc;
    if (std::abs(mSf - mSi) > 1e-10) return 0.0;
    if (std::abs(mIf - (mIi + 1.0)) < 1e-10) {
        return 0.5 * std::sqrt(I_nuc*(I_nuc+1.0) - mIi*(mIi+1.0));
    }
    if (std::abs(mIf - (mIi - 1.0)) < 1e-10) {
        return 0.5 * std::sqrt(I_nuc*(I_nuc+1.0) - mIi*(mIi-1.0));
    }
    return 0.0;
}

// ============================================================
// Constructor: build atomic eigenstates at field B
// ============================================================
SpinAlgebra::SpinAlgebra(double B_gauss) : B_(B_gauss)
{
    using AU::I_nuc;
    using AU::S_elec;
    using AU::delta_hf_MHz;
    using AU::mu_B_MHz_per_G;
    using AU::g_I_Rb85;
    using AU::g_J_Rb85;

    // Hyperfine constant: A_hf * I . S gives the hyperfine energy
    // with Delta_hf = A_hf * (I + 1/2) * (2S+1)/2... no, the standard form is:
    //
    // For J = 1/2, H_hf = A_hf I . S with A_hf = Delta_hf / (I + 1/2)
    //
    const double A_hf = delta_hf_MHz / (I_nuc + S_elec);

    // Valid mI values for I = 5/2: {-5/2, -3/2, -1/2, 1/2, 3/2, 5/2}
    const std::vector<double> mI_list = {-2.5, -1.5, -0.5, 0.5, 1.5, 2.5};

    // Total f_max = I + S = 3, mf_values from -3 to +3
    for (int mf = -3; mf <= 3; ++mf) {
        // Basis: |mI, mS> with mI + mS = mf (max 2 states)
        std::vector<std::pair<double,double>> basis;
        for (double mS : {+0.5, -0.5}) {
            double mI = mf - mS;
            // Check mI is in the valid list
            bool ok = false;
            for (double mv : mI_list) if (std::abs(mv - mI) < 1e-10) { ok = true; break; }
            if (ok) basis.emplace_back(mI, mS);
        }

        const int n = static_cast<int>(basis.size());
        if (n == 0) continue;

        // Build Hamiltonian: H = A_hf I.S + (g_J mu_B S_z + g_I mu_B I_z) B
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double mIi = basis[i].first, mSi = basis[i].second;
                double mIj = basis[j].first, mSj = basis[j].second;
                H(i,j) = A_hf * IS_element(mIi, mSi, mIj, mSj);
                if (i == j) {
                    H(i,j) += (g_J_Rb85 * mu_B_MHz_per_G * mSi
                              + g_I_Rb85 * mu_B_MHz_per_G * mIi) * B_gauss;
                }
            }
        }

        // Diagonalize
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        const Eigen::VectorXd& eigvals = es.eigenvalues();   // ascending
        const Eigen::MatrixXd& eigvecs = es.eigenvectors();

        for (int k = 0; k < n; ++k) {
            // Label: lower eigenvalue -> f = 2, upper -> f = 3.
            // Stretched states (n == 1, mf = +/- 3) have only f = 3.
            int f_label = (n == 1) ? 3 : (k == 0 ? 2 : 3);

            Atom1Body state;
            state.f = f_label;
            state.mf = mf;
            state.E_MHz = eigvals(k);
            for (int kk = 0; kk < n; ++kk) {
                double amp = eigvecs(kk, k);
                if (std::abs(amp) > 1e-15) {
                    state.mI_list.push_back(basis[kk].first);
                    state.mS_list.push_back(basis[kk].second);
                    state.coeffs.push_back(amp);
                }
            }
            atoms_[{f_label, mf}] = state;
        }
    }
}

// ============================================================
// Access atomic state
// ============================================================
const Atom1Body& SpinAlgebra::atom(int f, int mf) const
{
    auto it = atoms_.find({f, mf});
    if (it == atoms_.end()) {
        throw std::runtime_error("SpinAlgebra::atom: (f="
            + std::to_string(f) + ", mf=" + std::to_string(mf) + ") not found");
    }
    return it->second;
}

// ============================================================
// Enumerate s-wave bosonic two-body channels at given M_F
// ============================================================
std::vector<TwoBodyChannel> SpinAlgebra::channels(int MF) const
{
    std::vector<TwoBodyChannel> out;

    // Iterate over all ordered pairs of atomic states
    for (const auto& [k1, a1] : atoms_) {
        for (const auto& [k2, a2] : atoms_) {
            int f1 = k1.first, mf1 = k1.second;
            int f2 = k2.first, mf2 = k2.second;
            if (mf1 + mf2 != MF) continue;

            // Bosonic symmetry: for symmetric two-body state, the canonical 
            // ordering is (f1, mf1) <= (f2, mf2) lexicographically
            if (std::make_pair(f1, mf1) > std::make_pair(f2, mf2)) continue;

            TwoBodyChannel ch;
            ch.f1 = f1; ch.mf1 = mf1;
            ch.f2 = f2; ch.mf2 = mf2;
            ch.E_th_MHz = a1.E_MHz + a2.E_MHz;
            ch.label = "|" + std::to_string(f1) + "," + std::to_string(mf1) + ">"
                     + "|" + std::to_string(f2) + "," + std::to_string(mf2) + ">";
            out.push_back(ch);
        }
    }

    // Sort by threshold
    std::sort(out.begin(), out.end(),
        [](const TwoBodyChannel& a, const TwoBodyChannel& b) {
            return a.E_th_MHz < b.E_th_MHz;
        });
    return out;
}

// ============================================================
// <alpha|s1.s2|beta> for symmetrized two-body states
// ============================================================
//
// For identical-boson symmetry, if the atom pair consists of two 
// DIFFERENT atomic states (f1,mf1) != (f2,mf2), we use
//   |alpha> = (|a1>|a2> + |a2>|a1>) / sqrt(2)
// If the atom pair is identical ((f1,mf1) == (f2,mf2)), then
//   |alpha> = |a>|a>
// (with N = 1 so that the state is already normalized).
//
// To handle both cases uniformly in the s1.s2 matrix element sum
// Sum_{perm_a, perm_b} of the raw 4 terms, we use:
//   N = 1/2  for identical pairs   (<alpha|alpha> = 1 needs 4 terms / 4)
//   N = 1/sqrt(2) for non-identical (<alpha|alpha> = 1 with 2 cross terms)
// and then multiply all 4 raw terms by N_a * N_b.
//
// This is the convention validated in Step 1 (eigenvalues of s1.s2 matrix
// are {-3/4, +1/4, +1/4, +1/4, +1/4} for the M_F = -4 block).
//
double SpinAlgebra::s1_dot_s2(const TwoBodyChannel& a, const TwoBodyChannel& b) const
{
    const Atom1Body& a1 = atom(a.f1, a.mf1);
    const Atom1Body& a2 = atom(a.f2, a.mf2);
    const Atom1Body& b1 = atom(b.f1, b.mf1);
    const Atom1Body& b2 = atom(b.f2, b.mf2);

    const double Na = (a.f1 == a.f2 && a.mf1 == a.mf2) ? 0.5 : 1.0/std::sqrt(2.0);
    const double Nb = (b.f1 == b.f2 && b.mf1 == b.mf2) ? 0.5 : 1.0/std::sqrt(2.0);

    auto raw = [](const Atom1Body& sa1, const Atom1Body& sa2,
                  const Atom1Body& sb1, const Atom1Body& sb2) -> double
    {
        double v = 0.0;
        for (size_t i1a = 0; i1a < sa1.coeffs.size(); ++i1a) {
            for (size_t i2a = 0; i2a < sa2.coeffs.size(); ++i2a) {
                for (size_t i1b = 0; i1b < sb1.coeffs.size(); ++i1b) {
                    for (size_t i2b = 0; i2b < sb2.coeffs.size(); ++i2b) {
                        // Nuclear spins must match (spectators)
                        if (std::abs(sa1.mI_list[i1a] - sb1.mI_list[i1b]) > 1e-10) continue;
                        if (std::abs(sa2.mI_list[i2a] - sb2.mI_list[i2b]) > 1e-10) continue;
                        double s12 = s1s2_element(
                            sa1.mS_list[i1a], sa2.mS_list[i2a],
                            sb1.mS_list[i1b], sb2.mS_list[i2b]);
                        v += sa1.coeffs[i1a] * sa2.coeffs[i2a]
                           * sb1.coeffs[i1b] * sb2.coeffs[i2b] * s12;
                    }
                }
            }
        }
        return v;
    };

    double dd = raw(a1, a2, b1, b2);
    double de = raw(a1, a2, b2, b1);
    double ed = raw(a2, a1, b1, b2);
    double ee = raw(a2, a1, b2, b1);
    return Na * Nb * (dd + de + ed + ee);
}

// ============================================================
// Single-atom mu_x matrix element
// ============================================================
// <f', mf' | mu_x / mu_B | f, mf>  =  g_J <Sx> + g_I <Ix>
// where <...> are taken between field-dependent eigenstates expressed 
// in the |m_I, m_S> basis.
//
double SpinAlgebra::mu_x_atom(const Atom1Body& fs, const Atom1Body& is) const
{
    using AU::g_J_Rb85;
    using AU::g_I_Rb85;
    double vS = 0.0, vI = 0.0;
    for (size_t ii = 0; ii < is.coeffs.size(); ++ii) {
        for (size_t jj = 0; jj < fs.coeffs.size(); ++jj) {
            vS += fs.coeffs[jj] * is.coeffs[ii]
                * Sx_single(fs.mS_list[jj], fs.mI_list[jj],
                            is.mS_list[ii], is.mI_list[ii]);
            vI += fs.coeffs[jj] * is.coeffs[ii]
                * Ix_single(fs.mS_list[jj], fs.mI_list[jj],
                            is.mS_list[ii], is.mI_list[ii]);
        }
    }
    return g_J_Rb85 * vS + g_I_Rb85 * vI;
}

// ============================================================
// Two-body <alpha | mu_x1 + mu_x2 | beta>
// ============================================================
// Same symmetrization as s1.s2 above: 4 raw terms with N_a * N_b prefactor.
//
// Raw element:
//   <a1 a2 | mu_x1 + mu_x2 | b1 b2>
//     = <a1|mu_x|b1> <a2|b2>  +  <a1|b1> <a2|mu_x|b2>
//
double SpinAlgebra::mu_x_two_body(const TwoBodyChannel& a, const TwoBodyChannel& b) const
{
    const Atom1Body& a1 = atom(a.f1, a.mf1);
    const Atom1Body& a2 = atom(a.f2, a.mf2);
    const Atom1Body& b1 = atom(b.f1, b.mf1);
    const Atom1Body& b2 = atom(b.f2, b.mf2);

    const double Na = (a.f1 == a.f2 && a.mf1 == a.mf2) ? 0.5 : 1.0/std::sqrt(2.0);
    const double Nb = (b.f1 == b.f2 && b.mf1 == b.mf2) ? 0.5 : 1.0/std::sqrt(2.0);

    auto ovlp = [](int f1, int mf1, int f2, int mf2) -> double {
        return (f1 == f2 && mf1 == mf2) ? 1.0 : 0.0;
    };

    auto raw = [this, &ovlp](const Atom1Body& sa1, const Atom1Body& sa2,
                              const Atom1Body& sb1, const Atom1Body& sb2) -> double
    {
        return mu_x_atom(sa1, sb1) * ovlp(sa2.f, sa2.mf, sb2.f, sb2.mf)
             + ovlp(sa1.f, sa1.mf, sb1.f, sb1.mf) * mu_x_atom(sa2, sb2);
    };

    double dd = raw(a1, a2, b1, b2);
    double de = raw(a1, a2, b2, b1);
    double ed = raw(a2, a1, b1, b2);
    double ee = raw(a2, a1, b2, b1);
    return Na * Nb * (dd + de + ed + ee);
}

// ============================================================
// Matrix builders
// ============================================================
Eigen::MatrixXd SpinAlgebra::s1s2_matrix(const std::vector<TwoBodyChannel>& chans) const
{
    const int N = static_cast<int>(chans.size());
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            double v = s1_dot_s2(chans[i], chans[j]);
            M(i,j) = v;
            M(j,i) = v;   // Hermitian
        }
    }
    return M;
}

Eigen::MatrixXd SpinAlgebra::rf_matrix(const std::vector<TwoBodyChannel>& a_list,
                                       const std::vector<TwoBodyChannel>& b_list) const
{
    const int Na = static_cast<int>(a_list.size());
    const int Nb = static_cast<int>(b_list.size());
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(Na, Nb);
    for (int i = 0; i < Na; ++i) {
        for (int j = 0; j < Nb; ++j) {
            M(i,j) = mu_x_two_body(a_list[i], b_list[j]);
        }
    }
    return M;
}
