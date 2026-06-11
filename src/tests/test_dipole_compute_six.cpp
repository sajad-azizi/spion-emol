// test_dipole_compute_six.cpp -- bit-identity gate for the batched dipole API.
//
// Runs the full H2O pipeline once (pot, sinv, rinv, BP, A/B fit), then:
//
//   path A:  six sequential `DipoleMatrixElement::compute(A, B, g, p, cfg)`
//            calls with cfg.cached_b_overlap reused across calls 2..6.
//            This is the legacy production code path.
//
//   path B:  one `DipoleMatrixElement::compute_six(A, B, cfg)` call.
//
// Asserts every (g, p)'s `D_reduced`, `D_reduced_raw`, `d_raw`,
// `b_overlap`, `d_correction`, and `partial_sigma` are BYTE-IDENTICAL to
// the legacy path (memcmp on the underlying double arrays).
//
// If this test ever loosens to a tolerance check, that is a regression --
// the user's contract is strict bit-identity.

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/DipoleIO.hpp"           // slice_index
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

#include <array>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace scatt;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

// Byte-identity check on two dense Eigen blocks of the same shape.
template <class M>
static bool bytes_equal(const M& a, const M& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    if (a.size() == 0) return true;
    return std::memcmp(a.data(), b.data(),
                       a.size() * sizeof(typename M::Scalar)) == 0;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }

    io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    Parameters params;
    params.r_min          = data.rmin;
    params.dr             = data.dr;
    params.N_grid         = data.Nr;
    params.Lmax_sce       = data.Lmax_sce;
    params.l_max_continuum = 4;
    params.validate();

    auto bundle = WavefunctionSetup::prepare(params, data, 0.5);
    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", false, false, false);
    ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu, bundle.params.n_sigma,
                        bundle.params.n_occ, data.rmin, data.dr);

    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic;
    sic.verbose = false; sic.try_load_checkpoint = false;
    sic.save_checkpoint = false;
    sic.checkpoint_dir  = "./checkpoints/dip6_sinv";
    std::filesystem::remove_all(sic.checkpoint_dir);
    SI.build(sic);

    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);
    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc;
    frc.verbose = false; frc.try_load_checkpoint = false;
    frc.save_checkpoint = false;
    frc.checkpoint_dir  = "./checkpoints/dip6_rinv";
    std::filesystem::remove_all(frc.checkpoint_dir);
    FRP.run(frc);

    KMatrixExtractor KME(bundle.params, FRP);
    auto res_K = KME.extract();
    auto bc    = KMatrixExtractor::make_psi_boundary(bundle.params, res_K.K_matrix);

    BackPropagator BP(bundle.params, pot, FRP, WI);
    BackPropagator::Config bpc;
    bpc.n_keep_lo = 0;
    bpc.n_keep_hi = static_cast<int>(bundle.params.n_grid) - 1;
    bpc.compute_f = false;
    bpc.psi_storage = StorageMode::MEMORY;
    bpc.try_load_checkpoint = false;
    bpc.save_checkpoint     = false;
    bpc.verbose             = false;
    bpc.checkpoint_dir      = "./checkpoints/dip6_psi";
    std::filesystem::remove_all(bpc.checkpoint_dir);
    BP.run(bc, bpc);

    AsymptoticAmplitudes AA(bundle.params, BP);
    AsymptoticAmplitudes::Config aac; aac.verbose = false;
    auto res_AB = AA.extract(aac);

    // Build chi_init_homo and occupied bystanders (same shape as production).
    const int N_psi    = bundle.params.n_mu;
    const int Nr       = static_cast<int>(bundle.params.n_grid);
    const int n_occ    = bundle.params.n_occ;
    const int Nlm_init = (data.Lmax_sce + 1) * (data.Lmax_sce + 1);
    const int i_homo   = n_occ - 1;
    const int n_lambda = static_cast<int>(bundle.chi[0].cols());
    Eigen::MatrixXd chi_init_homo = Eigen::MatrixXd::Zero(Nr, Nlm_init);
    for (int ir = 0; ir < Nr; ++ir) {
        for (int lam = 0; lam < n_lambda && lam < Nlm_init; ++lam) {
            chi_init_homo(ir, lam) = bundle.chi[ir](i_homo, lam);
        }
    }
    std::vector<OccupiedOrbital> occ;
    for (int j = 0; j < n_occ - 1; ++j) {
        OccupiedOrbital o;
        o.phi = Eigen::MatrixXd::Zero(Nr, n_lambda);
        for (int ir = 0; ir < Nr; ++ir) {
            for (int lam = 0; lam < n_lambda; ++lam) {
                o.phi(ir, lam) = bundle.chi[ir](j, lam);
            }
        }
        o.spin_factor = 2.0;
        occ.push_back(std::move(o));
    }

    DipoleMatrixElement DME(bundle.params, BP, chi_init_homo, occ);

    DipoleMatrixElement::Config cfg;
    cfg.orthogonalize = true;
    cfg.verbose       = false;
    cfg.n_overlap_hi  = -1;

    // ----- Path A: legacy 6 x compute() with cached b_overlap on calls 2..6.
    const DipoleGauge  gs[2] = { DipoleGauge::Length, DipoleGauge::Velocity };
    const Polarization ps[3] = { Polarization::X, Polarization::Y, Polarization::Z };
    std::array<DipoleResult, 6> A_results;
    Eigen::MatrixXd b_cache;
    for (auto g : gs) for (auto p : ps) {
        DipoleMatrixElement::Config dmc = cfg;
        dmc.cached_b_overlap = (b_cache.size() > 0) ? &b_cache : nullptr;
        auto r = DME.compute(res_AB.A, res_AB.B, g, p, dmc);
        if (b_cache.size() == 0) b_cache = r.b_overlap;
        A_results[DipoleWriter::slice_index(g, p)] = std::move(r);
    }

    // ----- Path B: one compute_six() call.  No cached b_overlap.
    std::array<DipoleResult, 6> B_results =
        DME.compute_six(res_AB.A, res_AB.B, cfg);

    // ----- Strict byte-identity compare across all 6 slots and every field.
    std::cout << "=== compute() x6  vs  compute_six()  (strict byte-identity) ===\n";
    for (int slot = 0; slot < 6; ++slot) {
        const DipoleResult& a = A_results[slot];
        const DipoleResult& b = B_results[slot];
        const std::string tag =
            "slot=" + std::to_string(slot) + " (" +
            (a.gauge == DipoleGauge::Length ? "L" : "V") + "," +
            std::string(name_of(a.pol)) + ")";

        check(a.gauge == b.gauge && a.pol == b.pol, tag + " gauge/pol tag");
        check(bytes_equal(a.D_reduced,     b.D_reduced),     tag + " D_reduced bytes");
        check(bytes_equal(a.D_reduced_raw, b.D_reduced_raw), tag + " D_reduced_raw bytes");
        check(bytes_equal(a.d_raw,         b.d_raw),         tag + " d_raw bytes");
        check(bytes_equal(a.b_overlap,     b.b_overlap),     tag + " b_overlap bytes");
        check(bytes_equal(a.d_correction,  b.d_correction),  tag + " d_correction bytes");
        check(a.partial_sigma == b.partial_sigma,            tag + " partial_sigma scalar");
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << "  test_dipole_compute_six  (" << fails << " failures)\n";
    return fails == 0 ? 0 : 1;
}
