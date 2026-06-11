// test_exchange_coupling.cpp -- validate Q_ψf(r) on H2O + micro-benchmark.
//
// Correctness checks:
//   (1) Fast (sparse-gemm) vs Reference (triple loop): bit-close at many ir.
//   (2) μ=0 analytic identity: G(0,λ,σ) = δ_{λσ}/√(4π) implies
//       Q_ψf[0, (i,σ)] = +√2 · χ^i_σ(r) / r. Pins magnitude AND sign.
//   (3) Symmetry under μ ↔ σ: Q_ψf[μ, (i,σ)] == Q_ψf[σ, (i,μ)] when
//       n_mu == n_sigma.
//   (4) Finiteness + shape.
//   (5) Far-field decay: Q → 0 as orbitals decay.
//
// Benchmark:
//   - Fast path: average time per Q_ψf(ir) call over all ir in [1, n_trans),
//     averaged over REPEATS runs (warmed-up).
//   - Reference path: same, on a small subset of ir (it's O(|G| · n_occ) per
//     call and we don't want the test to take minutes).
//   - Reports: total wall time, per-call μs, throughput (ir/s), and the
//     speed-up factor fast/ref. The numbers are logged for regression; the
//     test only FAILS on correctness, not speed.
//
// Run: test_exchange_coupling <h2o_hdf5>

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using scatt::ExchangeCoupling;
using scatt::Parameters;
using scatt::SetupBundle;
using scatt::WavefunctionSetup;
using scatt::compute_Q_psi_f_reference;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int fails = 0;

static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

// Simple wall-clock timer.
struct Timer {
    using clk = std::chrono::steady_clock;
    clk::time_point t0;
    void   start() { t0 = clk::now(); }
    double elapsed_s() const {
        return std::chrono::duration<double>(clk::now() - t0).count();
    }
};

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2; }

    scatt::print_la_banner();

    HDF5Reader reader(argv[1]);
    PreprocData data = reader.load_all();

    Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = 4;                // l_exch=4, l_orb=8, n_lambda=81
    params.validate();

    SetupBundle b = WavefunctionSetup::prepare(params, data, /*energy=*/0.5);

    const int N_psi   = b.params.n_mu;
    const int n_sigma = b.params.n_sigma;
    const int n_occ   = b.params.n_occ;
    const int N_f     = n_occ * n_sigma;
    const double r_min = data.rmin;
    const double dr    = data.dr;
    const int n_ir    = static_cast<int>(b.chi.size());

    ExchangeCoupling EC(b.G_coeff, N_psi, n_sigma, n_occ, r_min, dr);

    std::cout << "\n--- Setup ---\n"
              << "  N_psi=" << N_psi << "  n_sigma=" << n_sigma
              << "  n_occ=" << n_occ << "  N_f=" << N_f
              << "  n_lambda_max=" << EC.n_lambda_max()
              << "  |G_coeff|=" << b.G_coeff.size()
              << "  sparse bytes=" << EC.sparse_bytes()
              << "  n_ir=" << n_ir << "\n";

    // ------------------------------------------------------------------
    // (1) Fast == reference at several ir
    // ------------------------------------------------------------------
    std::cout << "\n--- (1) fast (sparse-gemm) vs reference (triple loop) ---\n";
    const std::vector<int> test_irs = {1, 5, 50, 200, 500, 1500, n_ir - 1};
    double worst_rel = 0.0;
    for (int ir : test_irs) {
        if (ir >= n_ir) continue;
        const double r = EC.radius(ir);
        Eigen::MatrixXd Q_fast = EC.compute(ir, b.chi[ir]);
        Eigen::MatrixXd Q_ref  = compute_Q_psi_f_reference(
            b.G_coeff, b.chi[ir], N_psi, n_sigma, n_occ, r);
        const double max_err = (Q_fast - Q_ref).cwiseAbs().maxCoeff();
        const double scale   = std::max(Q_fast.cwiseAbs().maxCoeff(), 1e-30);
        const double rel     = max_err / scale;
        worst_rel = std::max(worst_rel, rel);
        std::cout << "     ir=" << std::setw(5) << ir
                  << "  r=" << std::setw(6) << std::fixed << std::setprecision(3) << r
                  << "  |Q|_max=" << std::scientific << std::setprecision(3) << scale
                  << "  |fast-ref|_max=" << max_err
                  << "  rel=" << rel << "\n";
        check(rel < 1e-12, "fast ≡ ref at ir=" + std::to_string(ir));
    }
    std::cout << "     worst relative error across all probed ir: " << worst_rel << "\n";

    // ------------------------------------------------------------------
    // (2) μ=0 identity pins sign and magnitude
    // ------------------------------------------------------------------
    std::cout << "\n--- (2) μ=0 identity: Q[0, (i,σ)] = +sqrt(2) * χ^i_σ / r ---\n";
    {
        const std::vector<int> probe_irs = {10, 100, 500, 1500};
        double worst_err = 0.0;
        for (int ir : probe_irs) {
            if (ir >= n_ir) continue;
            const double r = EC.radius(ir);
            Eigen::MatrixXd Q = EC.compute(ir, b.chi[ir]);
            for (int i = 0; i < n_occ; ++i) {
                for (int sigma = 0; sigma < n_sigma; ++sigma) {
                    const int f_idx = i * n_sigma + sigma;
                    const double pred = std::sqrt(2.0) * b.chi[ir](i, sigma) / r;
                    worst_err = std::max(worst_err, std::abs(Q(0, f_idx) - pred));
                }
            }
        }
        check(worst_err < 1e-12,
              "μ=0 identity holds exactly (worst_err=" + std::to_string(worst_err) + ")");
    }

    // ------------------------------------------------------------------
    // (3) Permutation symmetry Q[μ,(i,σ)] == Q[σ,(i,μ)]  (needs n_mu==n_sigma)
    // ------------------------------------------------------------------
    std::cout << "\n--- (3) Q_ψf[μ, (i,σ)] == Q_ψf[σ, (i,μ)] ---\n";
    if (N_psi == n_sigma) {
        const int ir = 200;
        Eigen::MatrixXd Q = EC.compute(ir, b.chi[ir]);
        double max_asym = 0.0;
        for (int mu = 0; mu < N_psi; ++mu) {
            for (int sigma = 0; sigma < n_sigma; ++sigma) {
                for (int i = 0; i < n_occ; ++i) {
                    const double a    = Q(mu,    i * n_sigma + sigma);
                    const double bval = Q(sigma, i * n_sigma + mu);
                    max_asym = std::max(max_asym, std::abs(a - bval));
                }
            }
        }
        check(max_asym < 1e-12,
              "Q symmetric under μ↔σ (max_asym=" + std::to_string(max_asym) + ")");
    } else {
        std::cout << "     skipped (N_psi != n_sigma)\n";
    }

    // ------------------------------------------------------------------
    // (4) Finiteness + shape
    // ------------------------------------------------------------------
    std::cout << "\n--- (4) finiteness and shape ---\n";
    {
        for (int ir : {0, 1, 50, n_ir - 1}) {
            Eigen::MatrixXd Q = EC.compute(ir, b.chi[ir]);
            check(Q.rows() == N_psi && Q.cols() == N_f,
                  "shape (N_psi, N_f) at ir=" + std::to_string(ir));
            check(Q.allFinite(), "all entries finite at ir=" + std::to_string(ir));
        }
    }

    // ------------------------------------------------------------------
    // (5) Far-field decay
    // ------------------------------------------------------------------
    std::cout << "\n--- (5) far-field decay ---\n";
    {
        const int ir_near = 100;
        const int ir_far  = n_ir - 1;
        const double near = EC.compute(ir_near, b.chi[ir_near]).cwiseAbs().maxCoeff();
        const double farv = EC.compute(ir_far,  b.chi[ir_far ]).cwiseAbs().maxCoeff();
        std::cout << "     |Q|_max at ir=" << ir_near << " (r=" << EC.radius(ir_near)
                  << "): " << std::scientific << near << "\n";
        std::cout << "     |Q|_max at ir=" << ir_far  << " (r=" << EC.radius(ir_far)
                  << "): " << std::scientific << farv  << "\n";
        check(near > 1e-6, "Q is nonzero near r=1 au");
        check(farv < 1e-6, "Q is decayed at r_max");
    }

    // ------------------------------------------------------------------
    // Benchmark A: alloc vs preallocated Q_ψf at the base l_cont
    // ------------------------------------------------------------------
    std::cout << "\n--- Benchmark A: compute vs compute_into (l_cont="
              << params.l_max_continuum << ") ---\n";
    {
        volatile double sink = 0.0;
        const int REPEATS = 3;
        Timer t;

        // Warm-up.
        for (int ir = 0; ir < n_ir; ++ir)
            sink += EC.compute(ir, b.chi[ir]).squaredNorm();

        // Allocating path.
        t.start();
        for (int rep = 0; rep < REPEATS; ++rep) {
            for (int ir = 0; ir < n_ir; ++ir) {
                sink += EC.compute(ir, b.chi[ir]).squaredNorm();
            }
        }
        const double alloc_total = t.elapsed_s();
        const double alloc_per   = alloc_total / (n_ir * REPEATS) * 1e6;

        // Preallocated path.
        auto ws = EC.make_workspace();
        Eigen::MatrixXd Q = EC.make_output();
        t.start();
        for (int rep = 0; rep < REPEATS; ++rep) {
            for (int ir = 0; ir < n_ir; ++ir) {
                EC.compute_into(ir, b.chi[ir], ws, Q);
                sink += Q.squaredNorm();
            }
        }
        const double prealloc_total = t.elapsed_s();
        const double prealloc_per   = prealloc_total / (n_ir * REPEATS) * 1e6;

        // Reference (strided).
        const int ref_stride = std::max(1, n_ir / 200);
        int ref_calls = 0;
        t.start();
        for (int rep = 0; rep < REPEATS; ++rep) {
            for (int ir = 0; ir < n_ir; ir += ref_stride) {
                sink += compute_Q_psi_f_reference(
                    b.G_coeff, b.chi[ir], N_psi, n_sigma, n_occ, EC.radius(ir))
                    .squaredNorm();
                ++ref_calls;
            }
        }
        const double ref_total = t.elapsed_s();
        const double ref_per   = ref_total / ref_calls * 1e6;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  compute     : " << alloc_per     << " μs/call  ("
                  << std::setprecision(0) << (1e6 / alloc_per)
                  << " ir/s)\n";
        std::cout << std::setprecision(2);
        std::cout << "  compute_into: " << prealloc_per  << " μs/call  ("
                  << std::setprecision(0) << (1e6 / prealloc_per)
                  << " ir/s)  [" << std::setprecision(2)
                  << (alloc_per / prealloc_per) << "× vs alloc]\n";
        std::cout << std::setprecision(2);
        std::cout << "  reference   : " << ref_per       << " μs/call  ("
                  << std::setprecision(0) << (1e6 / ref_per)
                  << " ir/s)  [stride=" << ref_stride << "]\n";
        std::cout << "  (sink checksum: "
                  << static_cast<int>(std::fmod(std::abs(sink), 10.0)) << ")\n";
    }

    // ------------------------------------------------------------------
    // Benchmark B: fast vs reference crossover as l_cont grows.
    //   Rebuilds the bundle at each l_cont so the |G_coeff| and the chi
    //   column-cut grow together. H2O HDF5 has Lmax_sce=32 -> l_orb up to 32
    //   -> l_cont up to 16 (since l_orb = 2 l_cont under min(l_cont,10) rule,
    //   but exch caps at 10, so here l_orb = l_cont + min(l_cont,10)).
    // ------------------------------------------------------------------
    std::cout << "\n--- Benchmark B: fast vs reference crossover ---\n";
    {
        std::cout << "  l_cont | n_mu | n_sigma | n_lambda | |G|   | "
                     "fast μs | ref μs | speed-up\n";
        std::cout << "  -------|------|---------|----------|-------|"
                     "---------|--------|---------\n";

        for (int l_cont : {2, 4, 6, 8, 10}) {
            // Need l_orb = l_cont + min(l_cont,10) <= Lmax_sce=32.
            const int l_exch = std::min(l_cont, 10);
            if (l_cont + l_exch > data.Lmax_sce) continue;

            Parameters p = params;
            p.l_max_continuum = l_cont;
            try { p.validate(); } catch (...) { continue; }

            SetupBundle bb = WavefunctionSetup::prepare(p, data, /*energy=*/0.5);
            ExchangeCoupling EC2(
                bb.G_coeff, bb.params.n_mu, bb.params.n_sigma, bb.params.n_occ,
                data.rmin, data.dr);

            // Choose a moderate ir-count so even slow reference fits the budget.
            const int N_IR = std::min<int>(200, static_cast<int>(bb.chi.size()));

            auto ws = EC2.make_workspace();
            Eigen::MatrixXd Q = EC2.make_output();
            volatile double sink = 0.0;

            // Warm-up.
            for (int ir = 0; ir < N_IR; ++ir) {
                EC2.compute_into(ir, bb.chi[ir], ws, Q);
                sink += Q.squaredNorm();
            }

            Timer t;
            t.start();
            const int REPS = 3;
            for (int r = 0; r < REPS; ++r)
                for (int ir = 0; ir < N_IR; ++ir) {
                    EC2.compute_into(ir, bb.chi[ir], ws, Q);
                    sink += Q.squaredNorm();
                }
            const double fast_per = t.elapsed_s() / (N_IR * REPS) * 1e6;

            t.start();
            for (int r = 0; r < REPS; ++r)
                for (int ir = 0; ir < N_IR; ++ir) {
                    sink += compute_Q_psi_f_reference(
                        bb.G_coeff, bb.chi[ir],
                        bb.params.n_mu, bb.params.n_sigma, bb.params.n_occ,
                        EC2.radius(ir)).squaredNorm();
                }
            const double ref_per = t.elapsed_s() / (N_IR * REPS) * 1e6;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "    " << std::setw(4) << l_cont
                      << " | " << std::setw(4) << bb.params.n_mu
                      << " | " << std::setw(7) << bb.params.n_sigma
                      << " | " << std::setw(8)
                      << (bb.params.l_max_orbitals + 1) * (bb.params.l_max_orbitals + 1)
                      << " | " << std::setw(5) << bb.G_coeff.size()
                      << " | " << std::setw(7) << fast_per
                      << " | " << std::setw(6) << ref_per
                      << " | " << std::setprecision(1) << std::setw(6)
                      << (ref_per / fast_per) << "×"
                      << std::setprecision(2) << "\n";

            // Suppress dead-code elimination.
            (void) sink;
        }
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failed)\n";
    return fails == 0 ? 0 : 1;
}
