// test_backprop_asym_buffer.cpp
//
// Validates the "asymptotic-buffer" trick in BackPropagator (borrowed from
// version_0). The trick splits ψ storage into:
//
//   * a truncated main store for ir ∈ [n_keep_lo, n_keep_hi]
//     (intended: orbital-support window, for the dipole matrix element)
//   * a small in-memory tail psi_asym_ for ir ∈ [N - n_asym + 1, N]
//     (intended: asymptotic window, for AsymptoticAmplitudes)
//
// If the split covers the orbital-support region (n_keep_hi ≥ orbital
// support) AND the asymptotic fit window (n_asym ≥ fit window size), the
// two downstream results — D_reduced and (A, B) — MUST agree bit-for-bit
// with the old "full range" path.

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/DipoleMatrixElement.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace scatt;

static int g_fail = 0;
static void check(bool cond, const std::string& what) {
    if (!cond) { std::cerr << "FAIL  " << what << "\n"; ++g_fail; }
    else       { std::cout << "ok    " << what << "\n"; }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <preproc.h5>\n";
        return 2;
    }

    io::HDF5Reader reader(argv[1]);
    io::PreprocData data = reader.load_all_except_V_H();

    Parameters params;
    params.l_max_continuum = 6;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    data.V_H               = reader.load_V_H(params.n_exp());
    params.validate();

    // Scattering energy: 0.5 Ha (k=1). Plenty of asymptotic oscillations
    // for the fit, and orbital support extends only ~3–5 bohr for H2O.
    const double E_kin = 0.5;
    auto bundle = WavefunctionSetup::prepare(params, data, E_kin);
    ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu, bundle.params.n_sigma,
                        bundle.params.n_occ, data.rmin, data.dr);

    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", false);

    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic;
    sic.verbose = false; sic.try_load_checkpoint = false; sic.save_checkpoint = false;
    SI.build(sic);

    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);
    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc;
    frc.verbose = false; frc.try_load_checkpoint = false; frc.save_checkpoint = false;
    FRP.run(frc);

    KMatrixExtractor KME(bundle.params, FRP);
    auto res_K = KME.extract();
    auto bc = KMatrixExtractor::make_psi_boundary(bundle.params, res_K.K_matrix);

    const int N = static_cast<int>(bundle.params.n_grid) - 1;

    // -------- Reference run: full-range ψ (n_asym = 0, old path) --------
    BackPropagator BP_ref(bundle.params, pot, FRP, WI);
    BackPropagator::Config c_ref;
    c_ref.n_keep_lo = 0;
    c_ref.n_keep_hi = N;
    c_ref.n_asym    = 0;            // old path -- no split
    c_ref.compute_f = false;
    c_ref.psi_storage = StorageMode::MEMORY;
    c_ref.try_load_checkpoint = false;
    c_ref.save_checkpoint     = false;
    c_ref.verbose = false;
    BP_ref.run(bc, c_ref);

    AsymptoticAmplitudes AA_ref(bundle.params, BP_ref);
    AsymptoticAmplitudes::Config aac; aac.verbose = false;
    auto AB_ref = AA_ref.extract(aac);

    // -------- Split-range run: truncated main + 300-pt asym buffer --------
    const int n_main_hi = N / 4;    // deliberately small; well below the fit window
    const int n_asym    = 300;
    BackPropagator BP_split(bundle.params, pot, FRP, WI);
    BackPropagator::Config c_split = c_ref;
    c_split.n_keep_hi = n_main_hi;
    c_split.n_asym    = n_asym;
    BP_split.run(bc, c_split);

    AsymptoticAmplitudes AA_split(bundle.params, BP_split);
    auto AB_split = AA_split.extract(aac);

    // -------- Asymptotic (A, B) must be BIT-EQUAL --------
    // Both paths fit ψ at indices in the tail (≥ N − 200); both see the
    // same numbers because the back-prop recursion is deterministic.
    {
        const double dA = (AB_ref.A - AB_split.A).cwiseAbs().maxCoeff();
        const double dB = (AB_ref.B - AB_split.B).cwiseAbs().maxCoeff();
        check(dA == 0.0, "A matrix bit-equal between full and split paths");
        check(dB == 0.0, "B matrix bit-equal between full and split paths");
    }

    // -------- Dipole (reading from main store only) must be BIT-EQUAL --------
    // We need i_homo and chi_init_homo from the bundle — same setup as main.cpp.
    const int Nlm_init = static_cast<int>(bundle.chi[0].cols());
    const int Nr       = static_cast<int>(params.N_grid);
    const int n_occ    = bundle.params.n_occ;
    const int i_homo   = n_occ - 1;

    Eigen::MatrixXd chi_init_homo(Nr, Nlm_init);
    for (int ir = 0; ir < Nr; ++ir)
        for (int lam = 0; lam < Nlm_init; ++lam)
            chi_init_homo(ir, lam) = bundle.chi[ir](i_homo, lam);

    std::vector<OccupiedOrbital> occ(n_occ);
    for (int j = 0; j < n_occ; ++j) {
        occ[j].phi = Eigen::MatrixXd::Zero(Nr, Nlm_init);
        for (int ir = 0; ir < Nr; ++ir)
            for (int lam = 0; lam < Nlm_init; ++lam)
                occ[j].phi(ir, lam) = bundle.chi[ir](j, lam);
        occ[j].spin_factor = 2.0;
    }

    DipoleMatrixElement::Config dmc;
    dmc.n_overlap_hi = n_main_hi + 1;  // cap integration at the main store edge
    dmc.orthogonalize = true;
    dmc.verbose       = false;

    auto D_ref_L_z = DipoleMatrixElement(bundle.params, BP_ref, chi_init_homo, occ)
                         .compute(AB_ref.A, AB_ref.B,
                                  DipoleGauge::Length, Polarization::Z, dmc);
    auto D_split_L_z = DipoleMatrixElement(bundle.params, BP_split, chi_init_homo, occ)
                           .compute(AB_split.A, AB_split.B,
                                    DipoleGauge::Length, Polarization::Z, dmc);

    const double dD_raw = (D_ref_L_z.D_reduced_raw - D_split_L_z.D_reduced_raw)
                             .cwiseAbs().maxCoeff();
    const double dD_ort = (D_ref_L_z.D_reduced     - D_split_L_z.D_reduced)
                             .cwiseAbs().maxCoeff();
    const double scale = std::max(D_ref_L_z.D_reduced.cwiseAbs().maxCoeff(), 1e-30);
    std::ostringstream os;
    os << "|D_raw_full - D_raw_split| = " << dD_raw
       << ", |D_ort_full - D_ort_split| = " << dD_ort
       << ", scale ~ " << scale;
    check(dD_raw == 0.0, "D_raw bit-equal full vs split");
    check(dD_ort == 0.0, "D_ortho bit-equal full vs split");
    if (dD_raw != 0.0 || dD_ort != 0.0) std::cerr << "  " << os.str() << "\n";

    // -------- Guard: get_psi on gap index should throw --------
    try {
        // Mid-range index between n_main_hi and asym_offset — must be in the gap.
        const int mid = (n_main_hi + (N - n_asym + 1)) / 2;
        (void)BP_split.get_psi(static_cast<std::size_t>(mid));
        check(false, "expected gap get_psi to throw");
    } catch (const std::runtime_error&) {
        check(true, "gap get_psi throws as expected");
    }

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " backprop_asym_buffer  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
