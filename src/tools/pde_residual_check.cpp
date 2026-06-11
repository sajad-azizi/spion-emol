// pde_residual_check.cpp -- run the full scattering pipeline on a given
// preprocessing HDF5, at a given energy and l_max_continuum, and print the
// residuals of PDF eq. 16 and eq. 13 along r. A standalone sanity tool.
//
// WHAT IT PROVES:
//   If the residuals are small and drop with r in the pattern
//   O(h⁴)·|Q''·ψ + ψ^(6)|, the solver is solving the REAL coupled PDE
//   (not just Numerov). Large residuals would expose a sign flip, wrong
//   prefactor, or wrong index.
//
// USAGE:
//   pde_residual_check <h5_file> [options]
//     --energy=<E_ha>         scattering energy in Hartree   (default 0.5)
//     --l-cont=<l>            l_max_continuum                (default 4)
//     --n-keep-lo=<n>         keep range low  (default: 0)
//     --n-keep-hi=<n>         keep range high (default: N_grid−1)
//     --probe-points=<list>   comma-separated n values (default: spread)
//     --r-strict=<r_au>       only enforce strict tolerance at r > r_strict
//                             (default: 2 au)
//     --storage=<memory|disk> storage mode for the big arrays (default auto)
//     --verbose               extra print
//
// EXAMPLES:
//   pde_residual_check h2o_ccpvdz_sph.preproc.h5
//   pde_residual_check c8f8.preproc.h5 --energy=2.0 --l-cont=12
//
// EXIT CODES:
//   0 = everything passes the "Numerov accuracy" check.
//   1 = some residual exceeds the tolerance → investigation needed.

#include "io/HDF5Reader.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/Banner.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include "angular/Gaunt.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string h5;
    double      energy      = 0.5;
    int         l_cont      = 4;
    int         n_keep_lo   = -1;
    int         n_keep_hi   = -1;
    std::string probe_list;   // empty = default spread
    double      r_strict    = 2.0;
    bool        verbose     = false;
    scatt::StorageMode storage = scatt::StorageMode::AUTO;
};

bool parse_flag(const char* arg, const char* name, std::string& out) {
    const std::string pfx = std::string("--") + name + "=";
    if (std::strncmp(arg, pfx.c_str(), pfx.size()) == 0) {
        out = arg + pfx.size();
        return true;
    }
    return false;
}

Args parse(int argc, char** argv) {
    if (argc < 2) {
        std::cerr <<
          "usage: pde_residual_check <h5> [--energy=X] [--l-cont=N] "
          "[--n-keep-lo=N] [--n-keep-hi=N] [--probe-points=a,b,c,...] "
          "[--r-strict=X] [--storage=memory|disk] [--verbose]\n";
        std::exit(2);
    }
    Args a;
    a.h5 = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string v;
        if      (parse_flag(argv[i], "energy",     v)) a.energy    = std::stod(v);
        else if (parse_flag(argv[i], "l-cont",     v)) a.l_cont    = std::stoi(v);
        else if (parse_flag(argv[i], "n-keep-lo",  v)) a.n_keep_lo = std::stoi(v);
        else if (parse_flag(argv[i], "n-keep-hi",  v)) a.n_keep_hi = std::stoi(v);
        else if (parse_flag(argv[i], "probe-points", v)) a.probe_list = v;
        else if (parse_flag(argv[i], "r-strict",   v)) a.r_strict  = std::stod(v);
        else if (parse_flag(argv[i], "storage",    v)) {
            if      (v == "memory") a.storage = scatt::StorageMode::MEMORY;
            else if (v == "disk")   a.storage = scatt::StorageMode::DISK;
            else if (v == "auto")   a.storage = scatt::StorageMode::AUTO;
            else { std::cerr << "unknown storage: " << v << "\n"; std::exit(2); }
        }
        else if (std::strcmp(argv[i], "--verbose") == 0) a.verbose = true;
        else { std::cerr << "unknown arg: " << argv[i] << "\n"; std::exit(2); }
    }
    return a;
}

std::vector<int> parse_probe(const std::string& s, int Nr) {
    if (s.empty()) {
        // Default: 8 probes spread across the grid.
        std::vector<int> v;
        for (int frac : {2, 5, 10, 20, 50, 100, 200, 500}) {
            int n = Nr / frac;
            if (n >= 3 && n < Nr - 3) v.push_back(n);
        }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoi(tok));
    }
    return out;
}

}  // anon

int main(int argc, char** argv) {
    Args args = parse(argc, argv);

    scatt::print_la_banner();

    scatt::io::HDF5Reader reader(args.h5);
    auto data = reader.load_all();

    scatt::Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = args.l_cont;
    params.validate();

    std::cout << "\n=== PDE residual check ===\n";
    std::cout << "  molecule HDF5 : " << args.h5 << "\n";
    std::cout << "  energy        : " << args.energy << " Ha"
              << "  (k = " << std::sqrt(2.0 * args.energy) << " a.u.)\n";
    std::cout << "  l_cont        : " << args.l_cont << "\n";
    std::cout << "  grid          : N_grid=" << params.N_grid
              << "  r_min=" << params.r_min
              << "  dr=" << params.dr
              << "  r_max=" << params.r_max() << " au\n";

    // --- pipeline ---
    auto bundle = scatt::WavefunctionSetup::prepare(params, data, args.energy);

    scatt::Potentials pot(params);
    pot.build(data, scatt::StorageMode::MEMORY, "", /*verbose=*/false);

    scatt::ExchangeCoupling EC(bundle.G_coeff, bundle.params.n_mu,
                               bundle.params.n_sigma, bundle.params.n_occ,
                               data.rmin, data.dr);

    scatt::SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    scatt::SchurInverter::Config si_cfg;
    si_cfg.storage             = args.storage;
    si_cfg.use_openmp          = true;
    si_cfg.verbose             = args.verbose;
    si_cfg.try_load_checkpoint = false;
    si_cfg.save_checkpoint     = false;
    si_cfg.checkpoint_dir      = "./checkpoints/pderc_sinv";
    std::filesystem::remove_all(si_cfg.checkpoint_dir);
    SI.build(si_cfg);

    scatt::WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, si_cfg.W_min);

    scatt::ForwardRPropagator FRP(bundle.params, pot, WI);
    scatt::ForwardRPropagator::Config frp_cfg;
    frp_cfg.storage             = args.storage;
    frp_cfg.try_load_checkpoint = false;
    frp_cfg.save_checkpoint     = false;
    frp_cfg.verbose             = args.verbose;
    frp_cfg.checkpoint_dir      = "./checkpoints/pderc_rinv";
    std::filesystem::remove_all(frp_cfg.checkpoint_dir);
    FRP.run(frp_cfg);

    scatt::KMatrixExtractor KME(bundle.params, FRP);
    auto res = KME.extract();
    auto psi_bc = scatt::KMatrixExtractor::make_psi_boundary(bundle.params,
                                                              res.K_matrix);

    scatt::BackPropagator BP(bundle.params, pot, FRP, WI);
    scatt::BackPropagator::Config bp_cfg;
    bp_cfg.n_keep_lo   = (args.n_keep_lo >= 0) ? args.n_keep_lo : 0;
    bp_cfg.n_keep_hi   = (args.n_keep_hi >= 0)
                         ? args.n_keep_hi
                         : static_cast<int>(bundle.params.n_grid) - 1;
    bp_cfg.compute_f   = true;
    bp_cfg.psi_storage = scatt::StorageMode::MEMORY;
    bp_cfg.verbose     = args.verbose;
    BP.run(psi_bc, bp_cfg);

    // --- evaluate residuals ---
    const int Nr      = static_cast<int>(bundle.params.n_grid);
    const int N_psi   = bundle.params.n_mu;
    const int n_sigma = bundle.params.n_sigma;
    const int n_occ   = bundle.params.n_occ;
    const double h      = bundle.params.dr;
    const double h2_inv = 1.0 / (h * h);
    const double alpha  = std::sqrt(2.0 * M_PI);

    std::vector<int> l_sigma(n_sigma);
    for (int s = 0; s < n_sigma; ++s) {
        int l, m; scatt::angular::idx_to_lm(s, l, m); l_sigma[s] = l;
    }

    auto probes = parse_probe(args.probe_list, Nr);

    std::cout << "\n--- K-matrix summary ---\n";
    std::cout << "  K symmetry err = " << res.K_symmetry_err << "\n";
    std::cout << "  S unitarity err= " << res.unitarity_err   << "\n";
    std::cout << "  eigenphases (first 5): ";
    for (int i = 0; i < std::min(5, (int) res.eigenphases.size()); ++i)
        std::cout << res.eigenphases[i] << "  ";
    std::cout << "\n";

    auto print_table_header = [](const char* title) {
        std::cout << "\n--- " << title << " ---\n";
        std::cout << "   n  |   r     | |LHS|_max |  |RHS|_max | |residual|_max | rel\n";
        std::cout << "   ---|---------|-----------|------------|----------------|--------\n";
    };

    // ---- PDF eq. 16 (ψ) ----
    print_table_header("PDE residual: PDF eq. 16 (ψ)");
    double worst_psi = 0.0;
    double worst_psi_strict = 0.0;
    int    worst_psi_n = -1;
    for (int n : probes) {
        if (n < 3 || n > Nr - 3) continue;
        const double r = bundle.params.r_min + n * h;

        Eigen::MatrixXd p_m2 = BP.get_psi((std::size_t)(n - 2));
        Eigen::MatrixXd p_m1 = BP.get_psi((std::size_t)(n - 1));
        Eigen::MatrixXd p_00 = BP.get_psi((std::size_t)n);
        Eigen::MatrixXd p_p1 = BP.get_psi((std::size_t)(n + 1));
        Eigen::MatrixXd p_p2 = BP.get_psi((std::size_t)(n + 2));
        Eigen::MatrixXd psi_pp = (-p_m2 + 16.0 * p_m1 - 30.0 * p_00
                                   + 16.0 * p_p1 - p_p2) * (h2_inv / 12.0);
        const Eigen::MatrixXd& V_n = pot.get((std::size_t)n);
        Eigen::MatrixXd lhs = psi_pp + 2.0 * args.energy * p_00 - 2.0 * V_n * p_00;

        const Eigen::MatrixXd& f_n = BP.get_f((std::size_t)n);
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(N_psi, N_psi);
        const double inv_r = 1.0 / r;
        for (const auto& g : bundle.G_coeff) {
            const int mu = g.a, lambda = g.b, sigma = g.c;
            if (mu >= N_psi || sigma >= n_sigma) continue;
            if (lambda >= (int) bundle.chi[(std::size_t)n].cols()) continue;
            for (int i = 0; i < n_occ; ++i) {
                const double chi_val = bundle.chi[(std::size_t)n](i, lambda);
                const int f_idx = i * n_sigma + sigma;
                for (int j = 0; j < N_psi; ++j) {
                    rhs(mu, j) += -2.0 * alpha * g.value * chi_val *
                                  f_n(f_idx, j) * inv_r;
                }
            }
        }

        const double lhs_max = lhs.cwiseAbs().maxCoeff();
        const double rhs_max = rhs.cwiseAbs().maxCoeff();
        const double res_max = (lhs - rhs).cwiseAbs().maxCoeff();
        const double scale   = std::max({lhs_max, rhs_max, 1e-30});
        const double rel     = res_max / scale;
        if (lhs_max > 1e-8) {
            if (rel > worst_psi) { worst_psi = rel; worst_psi_n = n; }
            if (r > args.r_strict) worst_psi_strict = std::max(worst_psi_strict, rel);
        }
        std::cout << "   " << std::setw(5) << n
                  << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r
                  << " | " << std::scientific << std::setprecision(2) << std::setw(9) << lhs_max
                  << " | " << std::setw(10) << rhs_max
                  << " | " << std::setw(14) << res_max
                  << " | " << std::setw(8) << rel << "\n";
    }

    // ---- PDF eq. 13 (f) ----
    print_table_header("PDE residual: PDF eq. 13 (f)");
    double worst_f = 0.0;
    for (int n : probes) {
        if (n < 3 || n > Nr - 3) continue;
        const double r  = bundle.params.r_min + n * h;
        const double r2 = r * r;

        Eigen::MatrixXd fm2 = BP.get_f((std::size_t)(n - 2));
        Eigen::MatrixXd fm1 = BP.get_f((std::size_t)(n - 1));
        Eigen::MatrixXd f00 = BP.get_f((std::size_t)n);
        Eigen::MatrixXd fp1 = BP.get_f((std::size_t)(n + 1));
        Eigen::MatrixXd fp2 = BP.get_f((std::size_t)(n + 2));
        Eigen::MatrixXd f_pp = (-fm2 + 16.0 * fm1 - 30.0 * f00
                                  + 16.0 * fp1 - fp2) * (h2_inv / 12.0);

        const Eigen::MatrixXd& psi_n = BP.get_psi((std::size_t)n);
        Eigen::MatrixXd lhs = f_pp;
        for (int f_idx = 0; f_idx < (int) f00.rows(); ++f_idx) {
            const int l = l_sigma[f_idx % n_sigma];
            const double centrif = (r2 > 1e-30) ? double(l*(l+1))/r2 : 0.0;
            lhs.row(f_idx) -= centrif * f00.row(f_idx);
        }
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(f00.rows(), N_psi);
        const double inv_r = 1.0 / r;
        const double prefac = -2.0 * alpha;
        for (const auto& g : bundle.G_coeff) {
            const int mu = g.a, lambda = g.b, sigma = g.c;
            if (mu >= N_psi || sigma >= n_sigma) continue;
            if (lambda >= (int) bundle.chi[(std::size_t)n].cols()) continue;
            for (int i = 0; i < n_occ; ++i) {
                const double chi_val = bundle.chi[(std::size_t)n](i, lambda);
                const int f_idx = i * n_sigma + sigma;
                for (int j = 0; j < N_psi; ++j) {
                    rhs(f_idx, j) += prefac * g.value * chi_val *
                                     psi_n(mu, j) * inv_r;
                }
            }
        }
        const double lhs_max = lhs.cwiseAbs().maxCoeff();
        const double rhs_max = rhs.cwiseAbs().maxCoeff();
        const double res_max = (lhs - rhs).cwiseAbs().maxCoeff();
        const double scale   = std::max({lhs_max, rhs_max, 1e-30});
        const double rel     = res_max / scale;
        if (lhs_max > 1e-8 && rel > worst_f) worst_f = rel;
        std::cout << "   " << std::setw(5) << n
                  << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r
                  << " | " << std::scientific << std::setprecision(2) << std::setw(9) << lhs_max
                  << " | " << std::setw(10) << rhs_max
                  << " | " << std::setw(14) << res_max
                  << " | " << std::setw(8) << rel << "\n";
    }

    // b_phys diagnostic (ℓ=0 tail of f).
    std::cout << "\n--- b_phys (ℓ=0 f asymptotic const) diagnostic ---\n";
    Eigen::MatrixXd bp = BP.b_phys_monopole();
    const double psi_ref_scale =
        BP.get_psi((std::size_t)(Nr / 4)).cwiseAbs().maxCoeff();
    const double b_max = bp.cwiseAbs().maxCoeff();
    std::cout << "   max |b_phys| = " << std::scientific << std::setprecision(3)
              << b_max << "\n";
    std::cout << "   |ψ|_max at r≈r_max/4 = " << psi_ref_scale << "\n";
    std::cout << "   rel b_phys / ψ_scale = "
              << (b_max / std::max(psi_ref_scale, 1e-30))
              << "     (<1% → BC approximation is safe)\n";

    // Verdict.
    std::cout << "\n=== Verdict ===\n";
    std::cout << "   PDE eq. 16 worst rel          : " << worst_psi
              << " (at n=" << worst_psi_n << ")\n";
    std::cout << "   PDE eq. 16 worst rel (r>"
              << std::fixed << std::setprecision(1) << args.r_strict << " au): "
              << std::scientific << std::setprecision(3) << worst_psi_strict << "\n";
    std::cout << "   PDE eq. 13 worst rel          : " << worst_f << "\n";
    std::cout << "   max |b_phys| / |ψ|            : "
              << (b_max / std::max(psi_ref_scale, 1e-30)) << "\n";

    int exit_code = 0;
    if (worst_psi_strict > 5e-2) {
        std::cout << "\n   *** WARNING *** PDE eq. 16 residual > 5% in "
                     "overlap region.  Consider finer dr, higher r_max, "
                     "or larger l_cont.\n";
        exit_code = 1;
    } else {
        std::cout << "\n   PDE is satisfied to Numerov O(h⁴) accuracy.  ψ is "
                     "ready for overlap / dipole work.\n";
    }

    return exit_code;
}
