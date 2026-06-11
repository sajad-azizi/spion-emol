// test_sce_h2.cpp — Milestone 4 physics test.
//
// Project the H2 HOMO and total electron density (parsed from the molden
// file) onto real spherical harmonics around the bond midpoint, then
// verify:
//
//   (P1) || psi_HOMO ||^2 = sum_{lm} int r^2 |psi_lm(r)|^2 dr ~= 1
//   (P2) N_e = sqrt(4 pi) * int r^2 rho_{00}(r) dr ~= 2   (closed-shell H2)
//   (P3) Parity: H2 1 sigma_g has ungerade-zero content; projected on the
//        bond midpoint, all f^{psi}_{lm} with odd l should vanish.
//
// Convergence study: run two grids and report differences.
//
// Usage:
//   test_sce_h2 <molden_path>

#include "basis/MoldenBasis.hpp"
#include "molden/Molden.hpp"
#include "sce/RadialGrid.hpp"
#include "sce/SCE.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using preproc::angular::AngGrid;
using preproc::angular::lm_index;
using preproc::basis::build_basis;
using preproc::basis::evaluate_density;
using preproc::basis::evaluate_mo;
using preproc::molden::MoldenParser;
using preproc::sce::RadialGrid;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <molden_path>\n";
        return 2;
    }
    MoldenParser P(/*verbose=*/false);
    auto mol = P.parse(argv[1]);
    auto basis = build_basis(mol, /*verbose=*/false);

    // HOMO: highest-energy alpha MO with occ > 0.
    const preproc::molden::MO* homo = nullptr;
    for (const auto& m : mol.mos_alpha) {
        if (m.spin != 'A' || m.occ == 0.0) continue;
        if (!homo || m.energy > homo->energy) homo = &m;
    }
    if (!homo) { std::cerr << "no HOMO\n"; return 1; }

    // Origin: geometric center of the atoms (good default; H2 symmetry center).
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    for (const auto& a : mol.atoms) origin += a.xyz;
    origin /= static_cast<double>(mol.atoms.size());
    std::cerr << std::setprecision(6)
              << "[h2-sce] origin (Bohr) = " << origin.transpose() << "\n";

    // F3D lambdas for orbital and density.
    preproc::sce::F3D F_psi = [&](const Eigen::Vector3d& r) {
        return evaluate_mo(basis, *homo, r);
    };
    preproc::sce::F3D F_rho = [&](const Eigen::Vector3d& r) {
        return evaluate_density(basis, mol, r);
    };

    // Run twice: coarse + fine, to estimate convergence.
    struct GridSpec { int Lmax; double dr; double Rmax; };
    GridSpec specs[] = {
        {10, 0.05, 10.0},
        {16, 0.025, 12.0},
    };

    std::cerr << std::setprecision(12);
    int fails = 0;
    for (int s = 0; s < 2; ++s) {
        const auto& sp = specs[s];
        const int N = static_cast<int>(std::round(sp.Rmax / sp.dr)) + 1;
        auto rg = RadialGrid::build(0.0, sp.dr, N);
        auto ag = AngGrid::build(sp.Lmax);

        std::cerr << "\n=== grid (Lmax=" << sp.Lmax << " dr=" << sp.dr
                  << " Rmax=" << sp.Rmax << " Nr=" << N << ") ===\n";
        auto Fpsi = project(rg, ag, origin, F_psi, false);
        auto Frho = project(rg, ag, origin, F_rho, false);

        const double nrm_psi = preproc::sce::norm_squared(Fpsi, rg);
        const double N_e     = preproc::sce::integrate_total(Frho, rg);
        std::cerr << "  ||psi_HOMO||^2 = " << nrm_psi << "  (expected 1)\n";
        std::cerr << "  N_e            = " << N_e     << "  (expected 2)\n";

        // (P1) orbital normalization
        if (std::abs(nrm_psi - 1.0) > 1e-4) {
            std::cerr << "  [FAIL] orbital norm off by " << (nrm_psi - 1.0) << "\n";
            ++fails;
        } else std::cerr << "  [ok] orbital norm within 1e-4\n";
        // (P2) electron number
        if (std::abs(N_e - 2.0) > 1e-4) {
            std::cerr << "  [FAIL] N_e off by " << (N_e - 2.0) << "\n";
            ++fails;
        } else std::cerr << "  [ok] N_e within 1e-4\n";

        // (P3) parity of sigma_g: odd-l channels should have vanishing
        //      ||psi_{lm}||_r = sqrt(int r^2 |psi_lm|^2 dr)
        double max_odd_l_norm = 0.0;
        double sum_even_l_norm = 0.0;
        for (int l = 0; l <= sp.Lmax; ++l) {
            for (int m = -l; m <= l; ++m) {
                std::vector<double> g(rg.N);
                const int ch = lm_index(l, m);
                for (int k = 0; k < rg.N; ++k) g[k] = rg.r[k] * rg.r[k] * Fpsi(ch, k) * Fpsi(ch, k);
                const double nrm = std::sqrt(rg.integrate(g));
                if (l & 1) max_odd_l_norm = std::max(max_odd_l_norm, nrm);
                else       sum_even_l_norm += nrm * nrm;
            }
        }
        std::cerr << "  max ||psi_{l,m}||_r for odd l  = " << max_odd_l_norm
                  << "  (expected ~0 by sigma_g parity)\n";
        std::cerr << "  sum_{even l} ||psi_{l,m}||^2_r = " << sum_even_l_norm << "\n";
        if (max_odd_l_norm > 1e-10) {
            std::cerr << "  [FAIL] odd-l channels non-zero\n"; ++fails;
        } else std::cerr << "  [ok] odd-l channels vanish\n";

        // Dominant channels for psi_HOMO
        std::cerr << "  dominant psi channels (||psi_{l,m}||_r):\n";
        for (int l = 0; l <= sp.Lmax; ++l) {
            double nrm2 = 0.0;
            for (int m = -l; m <= l; ++m) {
                const int ch = lm_index(l, m);
                std::vector<double> g(rg.N);
                for (int k = 0; k < rg.N; ++k)
                    g[k] = rg.r[k] * rg.r[k] * Fpsi(ch, k) * Fpsi(ch, k);
                nrm2 += rg.integrate(g);
            }
            std::cerr << "    l=" << std::setw(2) << l
                      << "  ||psi_l||^2 = " << nrm2 << "\n";
        }
    }

    std::cerr << "\n==> " << (fails == 0 ? "PASS" : "FAIL: " + std::to_string(fails)) << "\n";
    return fails == 0 ? 0 : 1;
}
