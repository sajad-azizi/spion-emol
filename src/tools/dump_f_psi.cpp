// dump_f_psi.cpp -- dump selected radial channels of f(r) and ψ(r) to CSV
// so we can plot what the exchange and scattering solutions look like,
// with a focus on ℓ=0 f channels (the PDF eq.-24 limit f → const that the
// outer BC audit was about).
//
// CSV layout:
//     r,  f_{i,σ,j}_...,  psi_{μ,j}_...
// First line is a header; subsequent lines are the radial samples.
//
// Usage:
//     dump_f_psi <h2o_hdf5>  [output.csv]

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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <h2o_hdf5> [out.csv]\n";
        return 2;
    }
    const std::string out_csv = (argc >= 3) ? argv[2] : "f_psi_slices.csv";

    scatt::print_la_banner();

    scatt::io::HDF5Reader reader(argv[1]);
    auto data = reader.load_all();

    scatt::Parameters params;
    params.r_min           = data.rmin;
    params.dr              = data.dr;
    params.N_grid          = data.Nr;
    params.Lmax_sce        = data.Lmax_sce;
    params.l_max_continuum = 4;
    params.validate();

    auto bundle = scatt::WavefunctionSetup::prepare(params, data, 0.5);

    scatt::Potentials pot(params);
    pot.build(data, scatt::StorageMode::MEMORY, "", /*verbose=*/false);

    scatt::ExchangeCoupling EC(bundle.G_coeff,
                               bundle.params.n_mu, bundle.params.n_sigma,
                               bundle.params.n_occ, data.rmin, data.dr);

    scatt::SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    scatt::SchurInverter::Config si_cfg;
    si_cfg.storage             = scatt::StorageMode::MEMORY;
    si_cfg.verbose             = false;
    si_cfg.try_load_checkpoint = false;
    si_cfg.save_checkpoint     = false;
    si_cfg.checkpoint_dir      = "./checkpoints/dump_sinv";
    std::filesystem::remove_all(si_cfg.checkpoint_dir);
    SI.build(si_cfg);

    scatt::WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, si_cfg.W_min);

    scatt::ForwardRPropagator FRP(bundle.params, pot, WI);
    scatt::ForwardRPropagator::Config frp_cfg;
    frp_cfg.storage             = scatt::StorageMode::MEMORY;
    frp_cfg.try_load_checkpoint = false;
    frp_cfg.save_checkpoint     = false;
    frp_cfg.verbose             = false;
    frp_cfg.checkpoint_dir      = "./checkpoints/dump_rinv";
    std::filesystem::remove_all(frp_cfg.checkpoint_dir);
    FRP.run(frp_cfg);

    scatt::KMatrixExtractor KME(bundle.params, FRP);
    auto res = KME.extract();
    auto psi_bc = scatt::KMatrixExtractor::make_psi_boundary(bundle.params,
                                                              res.K_matrix);

    scatt::BackPropagator BP(bundle.params, pot, FRP, WI);
    scatt::BackPropagator::Config bp_cfg;
    bp_cfg.n_keep_lo   = 0;
    bp_cfg.n_keep_hi   = static_cast<int>(bundle.params.n_grid) - 1;
    bp_cfg.compute_f   = true;
    bp_cfg.psi_storage = scatt::StorageMode::MEMORY;
    bp_cfg.verbose     = false;
    BP.run(psi_bc, bp_cfg);

    // --- build index maps ---
    const int N_psi   = bundle.params.n_mu;
    const int n_sigma = bundle.params.n_sigma;
    const int n_occ   = bundle.params.n_occ;
    const int Nr      = static_cast<int>(bundle.params.n_grid);

    // f_idx = i * n_sigma + σ.  We pick a small selection:
    //   - all n_occ orbitals, σ = 0  (ℓ = 0 "monopole" — key question)
    //   - orbital i = 0, σ = 0, 1, 4, 9, 16  (ℓ = 0,1,2,3,4 for comparison)
    // j (column) = scattering-solution index; always use j = 0 for simplicity.
    // ψ: μ = 0 (s-wave open channel) and μ = 1 (p-wave), j = 0.

    struct FSpec { int i, sigma, j; };
    std::vector<FSpec> f_specs;
    // (a) all occupied orbitals at ℓ=0
    for (int i = 0; i < n_occ; ++i) f_specs.push_back({i, 0, 0});
    // (b) orbital 0 at ℓ = 0..4  (σ indices for m = 0 at each ℓ)
    //     ℓ=0 m=0  -> σ=0,  ℓ=1 m=0 -> σ=2, ℓ=2 m=0 -> σ=6,
    //     ℓ=3 m=0 -> σ=12,  ℓ=4 m=0 -> σ=20
    //     (we already have σ=0 from (a), reuse and add 2,6,12,20)
    for (int sig : {2, 6, 12, 20}) {
        if (sig < n_sigma) f_specs.push_back({0, sig, 0});
    }

    struct PsiSpec { int mu, j; };
    std::vector<PsiSpec> psi_specs;
    psi_specs.push_back({0, 0});   // ℓ_μ=0, j=0
    psi_specs.push_back({2, 0});   // ℓ_μ=1 m=0, j=0  (p_z)
    psi_specs.push_back({6, 0});   // ℓ_μ=2 m=0, j=0  (d_z²)

    // --- write header ---
    std::ofstream out(out_csv);
    if (!out) { std::cerr << "cannot open " << out_csv << " for writing\n"; return 2; }
    out << "r";
    for (auto& s : f_specs) {
        int l, m; scatt::angular::idx_to_lm(s.sigma, l, m);
        out << ",f_orb" << s.i << "_l" << l << "_m" << m << "_j" << s.j;
    }
    for (auto& p : psi_specs) {
        int l, m; scatt::angular::idx_to_lm(p.mu, l, m);
        out << ",psi_mu_l" << l << "_m" << m << "_j" << p.j;
    }
    out << "\n";

    // --- dump rows ---
    out << std::scientific;
    out.precision(8);
    for (int ir = 0; ir < Nr; ++ir) {
        const double r = bundle.params.r_min + ir * bundle.params.dr;
        out << r;
        const auto& f_n   = BP.get_f((std::size_t)ir);
        const auto& psi_n = BP.get_psi((std::size_t)ir);
        for (auto& s : f_specs) {
            const int f_idx = s.i * n_sigma + s.sigma;
            out << "," << f_n(f_idx, s.j);
        }
        for (auto& p : psi_specs) {
            out << "," << psi_n(p.mu, p.j);
        }
        out << "\n";
    }

    std::cout << "[dump_f_psi] wrote " << Nr << " rows  x  "
              << (1 + f_specs.size() + psi_specs.size()) << " cols  ->  "
              << out_csv << "\n";
    std::cout << "   f channels: " << f_specs.size() << "\n";
    std::cout << "   ψ channels: " << psi_specs.size() << "\n";
    return 0;
}
