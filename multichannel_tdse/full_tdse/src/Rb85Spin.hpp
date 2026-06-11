// Rb85Spin.hpp -- bridge to the reference atomic-physics layer.
//
// The reference SpinAlgebra (in multichannel_tdse/reference_code/) gives
// us field-dressed |f, m_f⟩ atomic eigenstates and ⟨α|Ŝ_1·Ŝ_2|β⟩ at a
// chosen B field.  It does NOT supply the σ⁺ raising vertex
// η^(+) = ⟨α | Ŝ_{1,+} + Ŝ_{2,+} | β⟩ that the TDSE recipe uses, so
// we add it here.
//
// This header is the ONLY interface to the reference code seen by the
// rest of `full_tdse/` -- the reference's `Common.hpp` (which defines
// `dcompx`, `I_unit`, etc. at global scope) is included only in
// Rb85Spin.cpp's translation unit, never leaking outward.
//
// Recipe quantities returned by this layer:
//
//   * `channels(MF)`  -> ordered list of two-body s-wave channels at
//      total M_F, sorted by threshold ascending.  Threshold E_th is in
//      MHz, referenced to the M_F=-4 entrance threshold (recipe's zero).
//
//   * `sigma_plus_block(MF_low, MF_high)`  -> N_high × N_low complex
//      matrix η^(+)_{f, i} = ⟨f ∈ M_F+1 | Ŝ_{1,+} + Ŝ_{2,+} | i ∈ M_F⟩.
//      Hermiticity check via η^(-) = (η^(+))†.
#pragma once

#include <Eigen/Dense>

#include <complex>
#include <string>
#include <vector>

namespace mc_tdse {

// One two-body s-wave channel at fixed total M_F.
struct ChannelInfo {
    int    f1, mf1;            // single-atom labels of one of the two atoms
    int    f2, mf2;            // single-atom labels of the other
    double E_th_MHz;            // threshold energy (MHz, referenced to M_F=-4 entrance)
    std::string label;          // pretty label, e.g. "|2,-2>|2,-2>"
};

// Lightweight Rb85 atomic-physics handle.  All matrix outputs are in
// the channel basis (the order returned by `channels(MF)`).
class Rb85Spin {
public:
    // B in Gauss.  Default 155.04 G (recipe operating point).
    explicit Rb85Spin(double B_gauss = 155.04);
    ~Rb85Spin();

    // No copy / move (the underlying SpinAlgebra owns std::map's; keep
    // the bridge unique).
    Rb85Spin(const Rb85Spin&) = delete;
    Rb85Spin& operator=(const Rb85Spin&) = delete;

    // List of two-body s-wave channels with total M_F = MF, sorted by
    // threshold (ascending).  E_th in MHz, with the M_F = -4 entrance
    // threshold defined as zero.
    std::vector<ChannelInfo> channels(int MF) const;

    // Build the σ⁺ vertex matrix η^(+)_{f, i} = ⟨f|Ŝ_{1,+}+Ŝ_{2,+}|i⟩
    // where i ∈ channels(MF_low) and f ∈ channels(MF_low + 1).
    // Result has shape (N_high, N_low) where N_high = #channels in
    // M_F=MF_low+1 and N_low = #channels in M_F=MF_low.
    Eigen::MatrixXcd sigma_plus_block(int MF_low) const;

    // The Ŝ_{1,-} + Ŝ_{2,-} vertex; in our convention this is exactly
    // (sigma_plus_block(MF_low))^†, but exposed here for convenience.
    Eigen::MatrixXcd sigma_minus_block(int MF_low) const {
        return sigma_plus_block(MF_low).adjoint();
    }

    // Single-atom σ_+ matrix element ⟨α|Ŝ_+|β⟩ where α and β are
    // field-dressed |f,m_f⟩ states identified by their zero-field
    // labels.  Used internally and exposed for unit tests.
    double atom_sigma_plus(int f_a, int mf_a, int f_b, int mf_b) const;

    // Field strength.
    double B_gauss() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace mc_tdse
