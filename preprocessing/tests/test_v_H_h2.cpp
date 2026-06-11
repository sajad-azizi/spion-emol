// test_v_H_h2.cpp — Milestone 6a physics test.
//
// Pipeline:
//   (1) parse H2 molden, build AO basis
//   (2) SCE-project the total closed-shell density rho
//   (3) build V_H via radial Poisson (Hartree.hpp)
//   (4) compute  J = (1/2) * <rho | V_H>  and compare to Psi4's E_coulomb_J
//
// For H2 / cc-pVDZ the reference is  E_coulomb_J = 1.317964920156 Ha.
//
// We also sanity-check:
//   - V_H(r -> infinity) * r  ~=  N_e = 2           (the monopole tail)
//   - V_H^R_{l,m} vanishes for odd l (H2 sigma_g density has gerade parity)
//
// Sweep (Lmax, dr) to show convergence.
//
// Usage: test_v_H_h2 <molden> <reference_json>

#include "basis/MoldenBasis.hpp"
#include "molden/Molden.hpp"
#include "potential/Hartree.hpp"
#include "potential/Vnuclear.hpp"         // uses inner_product_radial
#include "sce/RadialGrid.hpp"
#include "sce/SCE.hpp"
#include "tiny_json.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using preproc::angular::AngGrid;
using preproc::angular::lm_index;
using preproc::basis::build_basis;
using preproc::basis::evaluate_density;
using preproc::molden::MoldenParser;
using preproc::potential::build_V_H;
using preproc::potential::inner_product_radial;
using preproc::sce::RadialGrid;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <molden> <reference_json>\n";
        return 2;
    }
    MoldenParser P(false);
    auto mol = P.parse(argv[1]);
    auto basis = build_basis(mol, false);
    auto ref = tinyjson::parse_file(argv[2]);
    const double E_J_ref = ref["energies"].as_obj().at("E_coulomb_J").as_num();
    const int    N_e_ref = static_cast<int>(ref["nelectron"].as_num());
    std::cerr << std::setprecision(12);
    std::cerr << "[h2-VH] Psi4 reference  E_J = " << E_J_ref << " Ha\n";

    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    for (const auto& a : mol.atoms) origin += a.xyz;
    origin /= static_cast<double>(mol.atoms.size());

    preproc::sce::F3D F_rho = [&](const Eigen::Vector3d& r) {
        return evaluate_density(basis, mol, r);
    };

    struct Spec { int L; double dr; double Rmax; };
    Spec specs[] = {
        {16, 0.025,  12.0},
        {24, 0.0125, 12.0},
        {32, 0.005,  12.0},
        {40, 0.0025, 15.0},
    };

    int fails = 0;
    double last_err = 1e9;
    for (const auto& sp : specs) {
        const int Nr = static_cast<int>(std::round(sp.Rmax / sp.dr)) + 1;
        auto rg = RadialGrid::build(0.0, sp.dr, Nr);
        auto ag = AngGrid::build_basic(sp.L);

        auto t0 = std::chrono::steady_clock::now();
        auto Frho = project(rg, ag, origin, F_rho, false);
        auto t1 = std::chrono::steady_clock::now();
        auto V_H = build_V_H(Frho, rg, sp.L);
        auto t2 = std::chrono::steady_clock::now();

        const double E_two_rho_V = inner_product_radial(Frho, V_H, rg);
        const double J = 0.5 * E_two_rho_V;
        const double err = std::abs(J - E_J_ref);
        last_err = err;

        // Monopole tail check: for r >> atomic size, V_{H,00}(r) * r / sqrt(4pi) ~ N_e
        const int ch00 = lm_index(0, 0);
        const int k_far = rg.N - 1;
        const double tail = V_H(ch00, k_far) * rg.r[k_far] / std::sqrt(4.0 * M_PI);

        // Parity check: odd-l channels should vanish
        double max_odd_l = 0.0;
        for (int l = 1; l <= sp.L; l += 2) {
            for (int m = -l; m <= l; ++m) {
                const int ch = lm_index(l, m);
                for (int k = 0; k < rg.N; ++k) {
                    max_odd_l = std::max(max_odd_l, std::abs(V_H(ch, k)));
                }
            }
        }

        const double dt_sce = std::chrono::duration<double>(t1 - t0).count();
        const double dt_VH  = std::chrono::duration<double>(t2 - t1).count();

        std::cerr << "  Lmax=" << std::setw(3) << sp.L
                  << "  dr="   << std::setw(7) << sp.dr
                  << "  Nr="   << std::setw(5) << Nr
                  << "   J = " << std::setw(14) << J
                  << "  |err| = " << std::setw(9) << err
                  << "  tail(N_e) = " << std::setw(10) << tail
                  << "  max|V_odd_l| = " << std::setw(10) << max_odd_l
                  << "  t_sce=" << std::setw(6) << dt_sce
                  << " t_VH="  << std::setw(6) << dt_VH << "\n";
    }

    // PASS criterion: finest grid achieves |err| < 1e-4 Ha.
    // Chemical accuracy is ~1 mHa; we're tightening to ~0.1 mHa.
    if (last_err > 1e-4) {
        std::cerr << "  [FAIL] J did not converge to 1e-4 Ha (err=" << last_err << ")\n";
        ++fails;
    }

    std::cerr << "\n==> " << (fails == 0 ? "PASS" : "FAIL") << "\n";
    return fails ? 1 : 0;
}
