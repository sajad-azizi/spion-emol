#include "Rb85Spin.hpp"

// The reference Common.hpp brings in `dcompx`, `I_unit`, `AU::*` at
// GLOBAL scope.  We include reference headers HERE (one translation
// unit), and use only standard / Eigen types in the bridge interface.
#include "SpinAlgebra.hpp"

#include <stdexcept>

namespace mc_tdse {

// ----------------------------------------------------------------
// Pimpl: holds the reference SpinAlgebra
// ----------------------------------------------------------------
struct Rb85Spin::Impl {
    ::SpinAlgebra spin;
    explicit Impl(double B_gauss) : spin(B_gauss) {}
};

// ----------------------------------------------------------------
// Single-atom Ŝ_+ matrix element in the |m_I, m_S⟩ basis.
// For S=1/2:  ⟨m_I',m_S'=+½ | Ŝ_+ | m_I,m_S=−½⟩ = δ_{m_I',m_I} · 1
//             0 otherwise
// ----------------------------------------------------------------
static double Sx_or_Splus_element(double mIf, double mSf,
                                   double mIi, double mSi)
{
    // Ŝ_+ = Ŝ_x + i Ŝ_y;  for S=1/2 it raises m_S=−½ → +½ with amplitude 1.
    constexpr double tol = 1e-9;
    if (std::abs(mIf - mIi) > tol)         return 0.0;     // I unchanged
    if (std::abs(mSi - (-0.5)) > tol)      return 0.0;     // initial m_S = −½
    if (std::abs(mSf - (+0.5)) > tol)      return 0.0;     // final   m_S = +½
    return 1.0;
}

// Single-atom σ⁺ between two field-dressed eigenstates.
static double atom_sigma_plus_impl(const ::Atom1Body& fstate,
                                    const ::Atom1Body& istate)
{
    // ⟨fstate|S_+|istate⟩ = Σ_{k,l} c_f[k]* c_i[l] · ⟨mI[k],mS[k]|S_+|mI[l],mS[l]⟩
    // Coefficients are real for this Hamiltonian, so no conjugation issue.
    const int K = static_cast<int>(fstate.coeffs.size());
    const int L = static_cast<int>(istate.coeffs.size());
    double s = 0.0;
    for (int k = 0; k < K; ++k) {
        const double cf = fstate.coeffs[k];
        if (cf == 0.0) continue;
        for (int l = 0; l < L; ++l) {
            const double ci = istate.coeffs[l];
            if (ci == 0.0) continue;
            const double e = Sx_or_Splus_element(
                fstate.mI_list[k], fstate.mS_list[k],
                istate.mI_list[l], istate.mS_list[l]);
            s += cf * ci * e;
        }
    }
    return s;
}

// ----------------------------------------------------------------
// Two-body symmetrized ⟨A|Ŝ_{1,+}+Ŝ_{2,+}|B⟩.
// Same logic as SpinAlgebra::mu_x_two_body but for the raising
// operator instead of σ_x.
//
//   |A⟩ = Pos:  for atoms in distinct states (a1≠a2), use the
//                symmetrized two-particle ket |a1 a2⟩_S =
//                (|a1⟩|a2⟩ + |a2⟩|a1⟩)/√2.  Identical states (a1=a2)
//                are unsymmetrized: |a1⟩|a1⟩.
//
//   ⟨A|S_{1,+}+S_{2,+}|B⟩ acts on each atom in turn.  Crossed terms
//   come from the boson symmetrization.  See SpinAlgebra::mu_x_two_body
//   for the algebra; here we just substitute σ_+ for μ_x.
// ----------------------------------------------------------------
static double sigma_plus_two_body(const ::SpinAlgebra&    spin,
                                   const ::TwoBodyChannel& A,
                                   const ::TwoBodyChannel& B)
{
    const ::Atom1Body& a1 = spin.atom(A.f1, A.mf1);
    const ::Atom1Body& a2 = spin.atom(A.f2, A.mf2);
    const ::Atom1Body& b1 = spin.atom(B.f1, B.mf1);
    const ::Atom1Body& b2 = spin.atom(B.f2, B.mf2);

    auto same_atom = [](int fa, int mfa, int fb, int mfb) {
        return (fa == fb) && (mfa == mfb);
    };
    const bool A_identical = same_atom(A.f1, A.mf1, A.f2, A.mf2);
    const bool B_identical = same_atom(B.f1, B.mf1, B.f2, B.mf2);

    // Helper: 1-atom sandwich ⟨a|S_+|b⟩.
    auto sp = [&](const ::Atom1Body& a, const ::Atom1Body& b) {
        return atom_sigma_plus_impl(a, b);
    };
    // Helper: orthonormal overlap of two atomic eigenstates.
    auto ovl = [&](const ::Atom1Body& a, const ::Atom1Body& b) -> double {
        // Same eigenstate ⇒ 1, otherwise 0 (orthonormal basis).
        return (&a == &b) ? 1.0 : 0.0;
    };
    // Pointer-equality is robust here since SpinAlgebra returns a const
    // ref to a stored map entry; (f, mf) equal ⇒ same address.

    // Sum over all four (S_{1,+} or S_{2,+}) × (atom 1 or atom 2 in B).
    // Following SpinAlgebra::mu_x_two_body, with appropriate symmetry
    // factors:
    double m = 0.0;

    if (!A_identical && !B_identical) {
        // Both pairs distinct: |a1 a2>_S, |b1 b2>_S.  Symmetric.
        // ⟨A|S_{1,+}+S_{2,+}|B⟩ = sum_{i=1,2} sum_{j=1,2}
        //     (1/2)[sp(a_i, b_j) ovl(a_{3-i}, b_{3-j})
        //         + sp(a_i, b_{3-j}) ovl(a_{3-i}, b_j)]
        m += 0.5 * (sp(a1, b1) * ovl(a2, b2) + sp(a1, b2) * ovl(a2, b1));
        m += 0.5 * (sp(a2, b1) * ovl(a1, b2) + sp(a2, b2) * ovl(a1, b1));
        m += 0.5 * (ovl(a1, b1) * sp(a2, b2) + ovl(a1, b2) * sp(a2, b1));
        m += 0.5 * (ovl(a2, b1) * sp(a1, b2) + ovl(a2, b2) * sp(a1, b1));
        // Combine duplicate terms; the 4 symmetric paths above give
        // exactly the same result as the bra-ket symmetrization (each
        // crossed term counted twice but normalized by 1/√2 · 1/√2 = 1/2).
        m *= 0.5;
    }
    else if (A_identical && B_identical) {
        // Both pairs identical: |a1 a1>, |b1 b1> (no √2 factors).
        // ⟨A|S_{1,+}+S_{2,+}|B⟩ = 2 · sp(a1, b1) · ovl(a1, b1)
        m = 2.0 * sp(a1, b1) * ovl(a1, b1);
    }
    else if (!A_identical && B_identical) {
        // Distinct on the bra, identical on the ket.
        // |B⟩ = |b1 b1⟩, |A⟩ = (|a1 a2⟩ + |a2 a1⟩)/√2
        // ⟨A|S_+1+S_+2|B⟩ = √2 · [ sp(a1,b1) ovl(a2,b1) + sp(a2,b1) ovl(a1,b1) ]
        m = std::sqrt(2.0) *
            ( sp(a1, b1) * ovl(a2, b1) + sp(a2, b1) * ovl(a1, b1) );
    }
    else { // A_identical && !B_identical
        m = std::sqrt(2.0) *
            ( ovl(a1, b1) * sp(a1, b2) + ovl(a1, b2) * sp(a1, b1) );
    }
    return m;
}

// ----------------------------------------------------------------
// Bridge implementation
// ----------------------------------------------------------------

Rb85Spin::Rb85Spin(double B_gauss) : impl_(new Impl(B_gauss)) {}
Rb85Spin::~Rb85Spin() { delete impl_; }

double Rb85Spin::B_gauss() const { return impl_->spin.B_gauss(); }

// The reference SpinAlgebra::channels() returns thresholds in MHz,
// referenced to the SAME-block lowest threshold.  The recipe wants
// thresholds referenced to the M_F = -4 entrance threshold (zero).
// We compute that absolute zero once and shift each block.
namespace {
double mf4_entrance_threshold_MHz(::SpinAlgebra& spin) {
    auto chs = spin.channels(-4);
    if (chs.empty()) {
        throw std::runtime_error("M_F=-4 has no channels");
    }
    return chs.front().E_th_MHz;
}
}  // namespace

std::vector<ChannelInfo> Rb85Spin::channels(int MF) const {
    auto raw = impl_->spin.channels(MF);
    // Shift so M_F=-4 entrance threshold is zero.
    const double zero_MHz = mf4_entrance_threshold_MHz(impl_->spin);
    std::vector<ChannelInfo> out;
    out.reserve(raw.size());
    for (const auto& c : raw) {
        ChannelInfo ci;
        ci.f1       = c.f1;  ci.mf1 = c.mf1;
        ci.f2       = c.f2;  ci.mf2 = c.mf2;
        ci.E_th_MHz = c.E_th_MHz - zero_MHz;
        ci.label    = c.label;
        out.push_back(ci);
    }
    return out;
}

double Rb85Spin::atom_sigma_plus(int f_a, int mf_a,
                                  int f_b, int mf_b) const
{
    const ::Atom1Body& a = impl_->spin.atom(f_a, mf_a);
    const ::Atom1Body& b = impl_->spin.atom(f_b, mf_b);
    return atom_sigma_plus_impl(a, b);
}

Eigen::MatrixXcd Rb85Spin::sigma_plus_block(int MF_low) const
{
    auto low_chs  = impl_->spin.channels(MF_low);
    auto high_chs = impl_->spin.channels(MF_low + 1);
    const int N_low  = static_cast<int>(low_chs.size());
    const int N_high = static_cast<int>(high_chs.size());
    Eigen::MatrixXcd M(N_high, N_low);
    for (int f = 0; f < N_high; ++f) {
        for (int i = 0; i < N_low; ++i) {
            const double v = sigma_plus_two_body(
                impl_->spin, high_chs[f], low_chs[i]);
            M(f, i) = std::complex<double>(v, 0.0);
        }
    }
    return M;
}

}  // namespace mc_tdse
