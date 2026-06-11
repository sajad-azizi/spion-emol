// test_v_en_h2.cpp — Milestone 5 physics test.
//
// Build V_en(r) on the SCE grid (multipole expansion) and verify
//
//     <rho | V_en>  =  int rho(r) V_en(r) d^3r   ==   Psi4 E_Vne
//
// For H2 / cc-pVDZ the reference value from gen_reference.py is
//     E_Vne ~= -3.59996839 Ha
// (see reference JSON key  energies.E_nuclear_electron).
//
// We also sweep L_max to show convergence of the multipole expansion.
//
// Usage:
//   test_v_en_h2 <molden_path> <reference_json_path>

#include "basis/MoldenBasis.hpp"
#include "molden/Molden.hpp"
#include "potential/Vnuclear.hpp"
#include "sce/RadialGrid.hpp"
#include "sce/SCE.hpp"
#include "tiny_json.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using preproc::angular::AngGrid;
using preproc::basis::build_basis;
using preproc::basis::evaluate_density;
using preproc::molden::MoldenParser;
using preproc::potential::build_V_en;
using preproc::potential::inner_product_radial;
using preproc::sce::RadialGrid;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <molden> <reference_json>\n"; return 2;
    }
    MoldenParser P(false);
    auto mol = P.parse(argv[1]);
    auto basis = build_basis(mol, false);
    auto ref = tinyjson::parse_file(argv[2]);
    const double E_Vne_ref = ref["energies"].as_obj().at("E_nuclear_electron").as_num();
    std::cerr << std::setprecision(12);
    std::cerr << "[h2-Ven] Psi4 reference  E_Vne = " << E_Vne_ref << " Ha\n";

    // SCE origin = geometric center of the two H atoms.
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    for (const auto& a : mol.atoms) origin += a.xyz;
    origin /= static_cast<double>(mol.atoms.size());

    preproc::sce::F3D F_rho = [&](const Eigen::Vector3d& r) {
        return evaluate_density(basis, mol, r);
    };

    struct Row { int L; double dr; double E; double dt_sce; double dt_ven; };
    std::vector<Row> rows;

    // Sweep both angular (Lmax) and radial (dr) cutoffs.
    struct Spec { int L; double dr; double Rmax; };
    Spec specs[] = {
        {16, 0.025,  12.0},
        {24, 0.025,  12.0},
        {24, 0.0125, 12.0},
        {32, 0.005,  12.0},
        {40, 0.0025, 12.0},
    };
    for (const auto& sp : specs) {
        const int L = sp.L;
        const int Nr = static_cast<int>(std::round(sp.Rmax / sp.dr)) + 1;
        auto rg = RadialGrid::build(0.0, sp.dr, Nr);
        auto ag = AngGrid::build(L);
        auto t0 = std::chrono::steady_clock::now();
        auto Frho = project(rg, ag, origin, F_rho, false);
        auto t1 = std::chrono::steady_clock::now();
        auto Ven  = build_V_en(mol.atoms, origin, rg, L, /*verbose=*/false);
        auto t2 = std::chrono::steady_clock::now();

        const double E = inner_product_radial(Frho, Ven, rg);
        const double dt_sce = std::chrono::duration<double>(t1 - t0).count();
        const double dt_ven = std::chrono::duration<double>(t2 - t1).count();
        rows.push_back({L, sp.dr, E, dt_sce, dt_ven});

        std::cerr << "  Lmax=" << std::setw(3) << L
                  << "  dr="   << std::setw(7) << sp.dr
                  << "  Nr="   << std::setw(5) << Nr
                  << "   E_Vne = " << std::setw(14) << E
                  << "   |err| = " << std::setw(9) << std::abs(E - E_Vne_ref)
                  << "   t_sce=" << std::setw(6) << dt_sce << " s"
                  << "   t_Ven=" << std::setw(6) << dt_ven << " s\n";
    }

    // PASS criterion: largest L achieves |err| < 5e-3 Ha for H2.
    // (H2 is a hard test for multipoles about the midpoint because the
    //  nuclei are only 0.7 Bohr off-center; even so, E_Vne should converge.)
    const double final_err = std::abs(rows.back().E - E_Vne_ref);
    const int fails = (final_err < 5e-3) ? 0 : 1;
    std::cerr << "\n==> final |err| = " << final_err
              << "   " << (fails ? "FAIL" : "PASS") << "\n";
    return fails;
}
