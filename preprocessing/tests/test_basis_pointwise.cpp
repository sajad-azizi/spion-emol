// test_basis_pointwise.cpp — Milestone 2a cross-check.
//
// Reads the molden + companion JSON, then for every "pointwise_reference"
// entry compares:
//     rho_cpp(P)       vs  JSON rho(P)
//     psi_HOMO_cpp(P)  vs  JSON psi_HOMO(P)
//
// The JSON values come from an independent Python evaluator that parses the
// same molden file and uses the same AO ordering (see gen_reference.py).
// So differences must be ~machine-epsilon — tolerance 1e-12 absolute.
//
// Usage:
//   test_basis_pointwise <molden_path> <reference_json_path>

#include "basis/MoldenBasis.hpp"
#include "molden/Molden.hpp"
#include "tiny_json.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using preproc::basis::build_basis;
using preproc::basis::evaluate_density;
using preproc::basis::evaluate_mo;
using preproc::molden::MoldenParser;
using preproc::molden::Molecule;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <molden_path> <reference_json_path>\n";
        return 2;
    }
    const std::string molden_path = argv[1];
    const std::string json_path   = argv[2];

    std::cerr << "==> parse molden: " << molden_path << "\n";
    MoldenParser P(/*verbose=*/false);
    Molecule mol = P.parse(molden_path);

    std::cerr << "==> build basis (s,p only)\n";
    auto basis = build_basis(mol, /*verbose=*/true);

    std::cerr << "==> parse JSON ref: " << json_path << "\n";
    tinyjson::Value ref = tinyjson::parse_file(json_path);

    // Locate pointwise_reference; may be absent for molecules containing d/f/g.
    const auto& top = ref.as_obj();
    auto it = top.find("pointwise_reference");
    if (it == top.end()) {
        std::cerr << "  [skip] pointwise_reference missing in JSON (likely d/f/g basis — defer to Milestone 2b)\n";
        return 0;
    }
    const auto& pw = it->second.as_obj();
    const int n_pts  = static_cast<int>(pw.at("n_points").as_num());
    const auto& pts_flat = pw.at("points_xyz_bohr_rowmajor").as_arr();
    const auto& rho_ref  = pw.at("rho").as_arr();
    const auto& psi_ref  = pw.at("psi_homo").as_arr();
    const double homo_e_ref = pw.at("homo_energy_hartree").as_num();

    if ((int)rho_ref.size() != n_pts || (int)psi_ref.size() != n_pts)
        throw std::runtime_error("pointwise arrays size mismatch");

    // Locate HOMO in our parsed MOs (highest-energy alpha MO with occ>0).
    const preproc::molden::MO* homo = nullptr;
    for (const auto& m : mol.mos_alpha) {
        if (m.spin != 'A' || m.occ == 0.0) continue;
        if (!homo || m.energy > homo->energy) homo = &m;
    }
    if (!homo) { std::cerr << "  [FAIL] no occupied alpha MO\n"; return 1; }
    std::cerr << "  HOMO energy (ours) = " << std::setprecision(17) << homo->energy
              << "  (ref) = " << homo_e_ref
              << "  |diff| = " << std::abs(homo->energy - homo_e_ref) << "\n";
    if (std::abs(homo->energy - homo_e_ref) > 1e-10) {
        std::cerr << "  [FAIL] HOMO energy mismatch\n"; return 1;
    }

    // Evaluate at every point, track max |diff|.
    double max_rho_diff = 0.0, max_psi_diff = 0.0;
    int    argmax_rho = -1, argmax_psi = -1;
    Eigen::Vector3d R;
    for (int p = 0; p < n_pts; ++p) {
        R[0] = pts_flat[3 * p + 0].as_num();
        R[1] = pts_flat[3 * p + 1].as_num();
        R[2] = pts_flat[3 * p + 2].as_num();

        const double rho_cpp = evaluate_density(basis, mol, R);
        const double psi_cpp = evaluate_mo(basis, *homo, R);

        const double d_rho = std::abs(rho_cpp - rho_ref[p].as_num());
        const double d_psi = std::abs(psi_cpp - psi_ref[p].as_num());
        if (d_rho > max_rho_diff) { max_rho_diff = d_rho; argmax_rho = p; }
        if (d_psi > max_psi_diff) { max_psi_diff = d_psi; argmax_psi = p; }
    }

    std::cerr << std::scientific << std::setprecision(3);
    std::cerr << "  n_points tested    = " << n_pts << "\n";
    std::cerr << "  max |rho_cpp - rho_py|       = " << max_rho_diff
              << "   at point " << argmax_rho << "\n";
    std::cerr << "  max |psi_homo_cpp - psi_py|  = " << max_psi_diff
              << "   at point " << argmax_psi << "\n";

    const double TOL = 1e-12;
    int fails = 0;
    if (max_rho_diff > TOL) {
        std::cerr << "  [FAIL] rho diff exceeds " << TOL << "\n"; ++fails;
    }
    if (max_psi_diff > TOL) {
        std::cerr << "  [FAIL] psi_homo diff exceeds " << TOL << "\n"; ++fails;
    }
    std::cerr << (fails == 0 ? "\n==> PASS\n" : "\n==> FAIL\n");
    return fails == 0 ? 0 : 1;
}
