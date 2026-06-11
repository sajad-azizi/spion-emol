// test_v_H_pointwise.cpp — Milestone 6b independent cross-check.
//
// Independent V_H path: Psi4 computes V_H(P) at a set of 3D test points
// using libint analytic <mu|1/|r-P||nu> integrals contracted with the
// density matrix. These numbers are stored in the reference JSON under
// pointwise_reference.v_H.
//
// This test builds V_H on the SCE grid (via radial Poisson, path 6a),
// synthesizes V_H(P) at the same test points by cubic radial interpolation
// of V_H^R_{lm}(r) and Y^R summation, and compares to the Psi4 values.
//
// PASS: max |V_H_SCE(P) - V_H_Psi4(P)| <= 5e-5 Ha  (chosen to tolerate the
// Lmax truncation plus interpolation + radial-SCE integration errors we
// already measured in test_v_H_h2).
//
// Usage: test_v_H_pointwise <molden> <reference_json>

#include "basis/MoldenBasis.hpp"
#include "molden/Molden.hpp"
#include "potential/Hartree.hpp"
#include "sce/RadialGrid.hpp"
#include "sce/SCE.hpp"
#include "sce/Synthesize.hpp"
#include "tiny_json.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using preproc::angular::AngGrid;
using preproc::basis::build_basis;
using preproc::basis::evaluate_density;
using preproc::molden::MoldenParser;
using preproc::potential::build_V_H;
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

    const auto& pw = ref["pointwise_reference"].as_obj();
    const auto& vh_block = pw.at("v_H").as_obj();
    const int n = static_cast<int>(vh_block.at("n_points").as_num());
    const auto& pts_flat = vh_block.at("points_xyz_bohr_rowmajor").as_arr();
    const auto& vh_ref  = vh_block.at("V_H_hartree").as_arr();

    // Origin (molecular center)
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    for (const auto& a : mol.atoms) origin += a.xyz;
    origin /= static_cast<double>(mol.atoms.size());

    preproc::sce::F3D F_rho = [&](const Eigen::Vector3d& r) {
        return evaluate_density(basis, mol, r);
    };

    // Use a grid tight enough to reach <= 1e-7 Ha on J; that gives us
    // headroom for the pointwise synthesis.
    const int    Lmax = 40;
    const double dr   = 0.0025;
    const double Rmax = 15.0;
    const int    Nr   = static_cast<int>(std::round(Rmax / dr)) + 1;
    auto rg = RadialGrid::build(0.0, dr, Nr);
    auto ag = AngGrid::build_basic(Lmax);

    auto Frho = project(rg, ag, origin, F_rho, false);
    auto V_H  = build_V_H(Frho, rg, Lmax);

    std::cerr << std::setprecision(12);
    std::cerr << "[vh-pointwise] Lmax=" << Lmax
              << " dr=" << dr << " Rmax=" << Rmax << " (Nr=" << Nr << ")\n";
    std::cerr << "            n_test_points=" << n << "\n";

    double max_abs = 0.0, max_rel = 0.0;
    int argmax_abs = -1;
    double worst_P[3] = {0,0,0};
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d Pabs;
        Pabs << pts_flat[3*i].as_num(), pts_flat[3*i + 1].as_num(), pts_flat[3*i + 2].as_num();
        const double ours  = preproc::sce::synthesize_at(V_H, rg, origin, Lmax, Pabs);
        const double theirs = vh_ref[i].as_num();
        const double d = std::abs(ours - theirs);
        const double rel = d / std::max(std::abs(theirs), 1e-12);
        if (d > max_abs) {
            max_abs = d; argmax_abs = i;
            worst_P[0] = Pabs[0]; worst_P[1] = Pabs[1]; worst_P[2] = Pabs[2];
        }
        max_rel = std::max(max_rel, rel);
    }
    std::cerr << "  max |V_H_SCE - V_H_Psi4|  = " << max_abs
              << "  at point #" << argmax_abs
              << " P=(" << worst_P[0] << "," << worst_P[1] << "," << worst_P[2] << ")\n";
    std::cerr << "  max relative error        = " << max_rel << "\n";

    const double TOL = 5e-5;
    int fails = (max_abs > TOL) ? 1 : 0;
    std::cerr << "\n==> " << (fails ? "FAIL" : "PASS")
              << "  (TOL = " << TOL << " Ha)\n";
    return fails;
}
