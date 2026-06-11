// test_cc_dipole_bruteforce.cpp
//
// PURPOSE
//   Independent ground-truth check on compute_cc_dipole's per-ir math.
//
//   We compute cc_raw two ways for the SAME (κ, κ) pair:
//
//     (A) PRODUCTION  -- cc_dipole::compute_cc_dipole (whatever the
//                        current internal path is: dense Ang × dense ψ,
//                        sparse Ang × dense ψ, GPU, etc.)
//     (B) BRUTE FORCE -- explicit triple-nested loop, no GEMM:
//
//                          tmp[μ, α]    = Σ_ν  Ang[μ, ν] · ψ_ν[ν, α]
//                          M_ir[β, α]   = Σ_μ  ψ_κ[μ, β] · tmp[μ, α]
//                          cc_raw[β, α]+= w[ir] · r · M_ir[β, α]
//
//                        with hand-coded summation order, NOT calling
//                        any BLAS routine.  This is the unambiguous
//                        single source of truth.
//
//   We assert |A − B| ≤ 1e-12 absolute OR |A − B| / |B| ≤ 1e-10
//   relative on every cc_raw element, for every polarization.  This
//   matches the tolerance used by test_cc_dipole_symmetry and is what
//   the scientific results have been validated at.
//
//   THIS TEST IS THE BIT-EQUIVALENCE GATE.  Any future internal
//   reorganization of the cc_dipole hot loop (e.g. switching to a
//   sparse-Ang × dense-ψ first product, GEMM tile changes, etc.) MUST
//   still pass this test or it's wrong.

#include "CCDipole.hpp"

#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/Banner.hpp"
#include "scatt/DipoleMatrixElement.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/SolverParams.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace scatt;

static int n_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++n_fail;
}

// Local (lm) ↔ flat-index mapping — must match cc_dipole's idx_to_lm.
static inline void idx_to_lm_(int idx, int& l, int& m) {
    l = static_cast<int>(std::sqrt(static_cast<double>(idx)));
    if ((l + 1) * (l + 1) <= idx) ++l;
    m = idx - l * l - l;
}

// Brute-force composite-Simpson weights identical to cc_dipole's
// simpson_weights_ (Simpson 1/3 + last-4 Simpson 3/8 if even).
static std::vector<double> simpson_weights_local_(int n_pts, double h) {
    std::vector<double> w(static_cast<std::size_t>(n_pts), 0.0);
    if (n_pts % 2 == 1) {
        w[0]         = h / 3.0;
        w[n_pts - 1] = h / 3.0;
        for (int i = 1; i < n_pts - 1; ++i)
            w[i] = (i & 1 ? 4.0 : 2.0) * h / 3.0;
    } else {
        const int n13 = n_pts - 3;
        if (n13 >= 2) {
            w[0]         += h / 3.0;
            w[n13 - 1]   += h / 3.0;
            for (int i = 1; i < n13 - 1; ++i)
                w[i] += (i & 1 ? 4.0 : 2.0) * h / 3.0;
        }
        const double s38 = 3.0 * h / 8.0;
        const int j = n_pts - 4;
        w[j]     += s38;
        w[j + 1] += 3.0 * s38;
        w[j + 2] += 3.0 * s38;
        w[j + 3] += s38;
    }
    return w;
}

// Brute-force reference: triple nested loops over (β, α, ir) and (μ, ν).
// No BLAS.  Bit-fixed summation order: for each output element
// cc_raw[β, α] we sum w[ir]·r·(Σ_μ ψ_κ[μ, β] · Σ_ν Ang[μ, ν] · ψ_ν[ν, α])
// with the explicit ν-then-μ-then-ir loop nesting below.
static Eigen::MatrixXd
compute_cc_bruteforce_(const SolverParams& sp,
                       BackPropagator&     bp_kappa,
                       BackPropagator&     bp_nu,
                       Polarization        pol)
{
    const int N_psi = sp.n_mu;
    const int Nr    = static_cast<int>(sp.n_grid);
    const int q     = q_of(pol);

    const int n_lo = std::max({0, bp_kappa.n_keep_lo(), bp_nu.n_keep_lo()});
    const int n_hi = std::min({Nr,
                               bp_kappa.n_keep_hi() + 1,
                               bp_nu.n_keep_hi()    + 1});
    const int n_pts = n_hi - n_lo;
    const auto w    = simpson_weights_local_(n_pts, sp.dr);

    // Build dense Ang the same way build_ang_table_ does.
    std::vector<int> l_idx(N_psi), m_idx(N_psi);
    for (int mu = 0; mu < N_psi; ++mu) idx_to_lm_(mu, l_idx[mu], m_idx[mu]);
    Eigen::MatrixXd Ang = Eigen::MatrixXd::Zero(N_psi, N_psi);
    for (int mu = 0; mu < N_psi; ++mu) {
        for (int nu = 0; nu < N_psi; ++nu) {
            if (std::abs(l_idx[mu] - l_idx[nu]) != 1) continue;
            const double a = DipoleMatrixElement::angular_dipole(
                l_idx[mu], m_idx[mu], q, l_idx[nu], m_idx[nu]);
            if (a != 0.0) Ang(mu, nu) = a;
        }
    }

    Eigen::MatrixXd cc_raw = Eigen::MatrixXd::Zero(N_psi, N_psi);
    Eigen::MatrixXd tmp(N_psi, N_psi);
    Eigen::MatrixXd M_ir(N_psi, N_psi);

    for (int ir = n_lo; ir < n_hi; ++ir) {
        const Eigen::MatrixXd& psi_k = bp_kappa.get_psi(static_cast<std::size_t>(ir));
        const Eigen::MatrixXd& psi_n = bp_nu   .get_psi(static_cast<std::size_t>(ir));
        const double r   = sp.r_min + ir * sp.dr;
        const double w_r = w[ir - n_lo] * r;

        // tmp[μ, α] = Σ_ν Ang[μ, ν] · psi_n[ν, α]   --  hand summed.
        for (int alpha = 0; alpha < N_psi; ++alpha) {
            for (int mu = 0; mu < N_psi; ++mu) {
                double s = 0.0;
                for (int nu = 0; nu < N_psi; ++nu) {
                    s += Ang(mu, nu) * psi_n(nu, alpha);
                }
                tmp(mu, alpha) = s;
            }
        }
        // M_ir[β, α] = Σ_μ psi_k[μ, β] · tmp[μ, α]   --  hand summed.
        for (int alpha = 0; alpha < N_psi; ++alpha) {
            for (int beta = 0; beta < N_psi; ++beta) {
                double s = 0.0;
                for (int mu = 0; mu < N_psi; ++mu) {
                    s += psi_k(mu, beta) * tmp(mu, alpha);
                }
                M_ir(beta, alpha) = s;
            }
        }
        // cc_raw += w_r · M_ir.
        for (int alpha = 0; alpha < N_psi; ++alpha)
            for (int beta = 0; beta < N_psi; ++beta)
                cc_raw(beta, alpha) += w_r * M_ir(beta, alpha);
    }
    return cc_raw;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <h2o_orbitals.h5>\n", argv[0]);
        return 2;
    }
    print_la_banner();

    // ---- Fixture setup (identical to test_cc_dipole_symmetry.cpp) ----
    io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    Parameters params;
    params.r_min = data.rmin;  params.dr = data.dr;  params.N_grid = data.Nr;
    params.Lmax_sce = data.Lmax_sce;
    params.l_max_continuum = 4;     // N_psi = 25, fast enough for triple loop
    params.validate();

    auto bundle = WavefunctionSetup::prepare(params, data, 0.5);
    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", false, false, false);
    ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu, bundle.params.n_sigma,
                        bundle.params.n_occ, data.rmin, data.dr);
    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic; sic.verbose = false;
    sic.try_load_checkpoint = false; sic.save_checkpoint = false;
    sic.checkpoint_dir = "./checkpoints/cc_test_bf_sinv";
    std::filesystem::remove_all(sic.checkpoint_dir); SI.build(sic);
    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);
    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc;
    frc.verbose = false;  frc.try_load_checkpoint = false;
    frc.save_checkpoint = false;
    frc.checkpoint_dir = "./checkpoints/cc_test_bf_rinv";
    std::filesystem::remove_all(frc.checkpoint_dir);
    FRP.run(frc);

    KMatrixExtractor KME(bundle.params, FRP);
    auto res_K = KME.extract();
    auto bc = KMatrixExtractor::make_psi_boundary(bundle.params, res_K.K_matrix);

    BackPropagator BP(bundle.params, pot, FRP, WI);
    BackPropagator::Config bpc;
    bpc.n_keep_lo = 0;
    bpc.n_keep_hi = static_cast<int>(bundle.params.n_grid) - 1;
    bpc.compute_f = false;
    bpc.psi_storage = StorageMode::MEMORY;
    bpc.try_load_checkpoint = false;  bpc.save_checkpoint = false;
    bpc.verbose = false;
    bpc.checkpoint_dir = "./checkpoints/cc_test_bf_psi";
    std::filesystem::remove_all(bpc.checkpoint_dir);
    BP.run(bc, bpc);

    SolverParams sp;
    sp.n_grid          = bundle.params.n_grid;
    sp.dr              = bundle.params.dr;
    sp.r_min           = bundle.params.r_min;
    sp.energy          = bundle.params.energy;
    sp.n_mu            = bundle.params.n_mu;
    sp.l_max_continuum = bundle.params.l_max_continuum;

    std::cout << "\n--- cc_dipole brute-force ground-truth check ---\n";
    std::cout << "  N_psi=" << sp.n_mu
              << "  Nr=" << sp.n_grid
              << "  dr=" << sp.dr << "\n";

    // For each polarization, compute production and brute-force and
    // compare element-by-element.
    for (auto pol : {Polarization::X, Polarization::Y, Polarization::Z}) {
        auto prod = cc_dipole::compute_cc_dipole(
            sp, BP, BP, DipoleGauge::Length, pol);
        auto ref  = compute_cc_bruteforce_(sp, BP, BP, pol);

        double max_abs_diff = 0.0;
        double max_abs_ref  = 0.0;
        int    n_bad_abs    = 0;
        int    n_bad_rel    = 0;
        const double tol_abs = 1e-12;
        const double tol_rel = 1e-10;
        for (int i = 0; i < ref.rows(); ++i) {
            for (int j = 0; j < ref.cols(); ++j) {
                const double d = std::abs(prod.cc_raw(i, j) - ref(i, j));
                const double r = std::abs(ref(i, j));
                max_abs_diff = std::max(max_abs_diff, d);
                max_abs_ref  = std::max(max_abs_ref,  r);
                if (d > tol_abs && (r <= 0.0 || d / r > tol_rel)) {
                    if (n_bad_abs + n_bad_rel < 5) {
                        std::printf("    bad [%d,%d]  prod=%.17e  ref=%.17e  "
                                    "diff=%.3e\n",
                                    i, j, prod.cc_raw(i, j), ref(i, j), d);
                    }
                    if (d > tol_abs)             ++n_bad_abs;
                    if (r > 0.0 && d / r > tol_rel) ++n_bad_rel;
                }
            }
        }
        const double rel = (max_abs_ref > 0) ? max_abs_diff / max_abs_ref : 0.0;
        std::printf("  pol=%s   max|prod-ref|=%.3e  max|ref|=%.3e  rel=%.3e  "
                    "n_bad(abs|rel)=%d/%d\n",
                    name_of(pol), max_abs_diff, max_abs_ref, rel,
                    n_bad_abs, n_bad_rel);
        check(max_abs_diff < tol_abs || rel < tol_rel,
              std::string("brute-force equivalence  pol=") + name_of(pol));
    }

    std::printf("\n  Total failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
