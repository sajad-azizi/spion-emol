// test_hdf5_split_roundtrip.cpp -- HDF5Reader auto-detects split layout.
//
// preprocess_molden writes two files: <stem>.orbitals.h5 (psi_lm,
// initial_state) and <stem>.potentials.h5 (V_en, V_H, V_x, V_total, rho,
// polarizability, meta).  HDF5Reader's single-arg constructor must
// accept any of:
//
//   * <stem>.orbitals.h5    -> open both, route reads
//   * <stem>.potentials.h5  -> open both, route reads
//   * legacy <stem>.preproc.h5  -> open one combined file
//
// and produce a PreprocData identical (numerically) regardless of which
// path was passed.  This test verifies that:
//
//   1. Open via .orbitals.h5
//   2. Open via .potentials.h5
//   3. (If the legacy combined file is also present) open via .preproc.h5
//   compare relevant fields across all three -- they must all be byte-
//   identical for grid, geometry, atoms, and bit-equal for the Eigen
//   matrices.

#include "io/HDF5Reader.hpp"

#include <Eigen/Dense>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

static bool eigen_eq(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                     double tol, double& worst) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        worst = 1e300;
        return false;
    }
    worst = (A - B).cwiseAbs().maxCoeff();
    return worst <= tol;
}

static void compare_pairs(const PreprocData& a, const PreprocData& b,
                          const std::string& label_a, const std::string& label_b)
{
    std::cout << "--- comparing  " << label_a << "  vs  " << label_b << "  ---\n";
    check(a.rmin == b.rmin,                "grid.rmin matches");
    check(a.dr   == b.dr,                  "grid.dr matches");
    check(a.Nr   == b.Nr,                  "grid.Nr matches");
    check(a.Lmax_sce == b.Lmax_sce,        "angular.Lmax matches");
    check(a.atoms.size() == b.atoms.size(), "geometry.atoms count matches");
    if (a.atoms.size() == b.atoms.size()) {
        bool all_eq = true;
        for (size_t i = 0; i < a.atoms.size(); ++i) {
            const auto& A = a.atoms[i];
            const auto& B = b.atoms[i];
            if (A.Z != B.Z || A.x != B.x || A.y != B.y || A.z != B.z) {
                all_eq = false; break;
            }
        }
        check(all_eq, "geometry.atoms[i] match");
    }
    check(a.n_alpha     == b.n_alpha,     "n_alpha matches");
    check(a.n_occ_alpha == b.n_occ_alpha, "n_occ_alpha matches");
    check(a.n_sce       == b.n_sce,       "n_sce matches");

    double worst;
    check(eigen_eq(a.V_H,    b.V_H,    0.0, worst),
          "V_H bit-identical (worst=" + std::to_string(worst) + ")");
    check(eigen_eq(a.psi_lm, b.psi_lm, 0.0, worst),
          "psi_lm bit-identical (worst=" + std::to_string(worst) + ")");
    if (a.has_initial_state && b.has_initial_state) {
        check(eigen_eq(a.init_state_psi_lm, b.init_state_psi_lm, 0.0, worst),
              "initial_state.psi_lm bit-identical (worst=" + std::to_string(worst) + ")");
    } else {
        check(a.has_initial_state == b.has_initial_state, "has_initial_state matches");
    }
    check(a.has_polarizability == b.has_polarizability, "has_polarizability matches");
    if (a.has_polarizability && b.has_polarizability) {
        check(eigen_eq(a.alpha_tensor, b.alpha_tensor, 0.0, worst),
              "alpha_tensor bit-identical");
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << "  <h2o_split_path: e.g. .../h2o_ccpvdz_sph.orbitals.h5>"
                  << "  [legacy_combined.h5]\n";
        return 2;
    }
    const std::string split_path = argv[1];

    // Derive the sibling potentials path so we can also test opening THAT
    // path and getting identical data.
    std::string sibling_path;
    {
        const std::string ORB = ".orbitals.h5";
        const std::string POT = ".potentials.h5";
        if (split_path.size() >= ORB.size() &&
            split_path.compare(split_path.size() - ORB.size(), ORB.size(), ORB) == 0) {
            sibling_path = split_path.substr(0, split_path.size() - ORB.size()) + POT;
        } else if (split_path.size() >= POT.size() &&
                   split_path.compare(split_path.size() - POT.size(), POT.size(), POT) == 0) {
            sibling_path = split_path.substr(0, split_path.size() - POT.size()) + ORB;
        } else {
            std::cerr << "split_path must end in .orbitals.h5 or .potentials.h5\n";
            return 2;
        }
    }
    if (!std::filesystem::exists(sibling_path)) {
        std::cerr << "sibling not found: " << sibling_path
                  << "\n  (regenerate with preprocess_molden using the same stem)\n";
        return 2;
    }

    std::cout << "=== HDF5Reader split-layout roundtrip ===\n";
    std::cout << "  via orbitals path  : " << split_path   << "\n";
    std::cout << "  via potentials path: " << sibling_path << "\n";

    PreprocData via_orb, via_pot;
    {
        HDF5Reader r(split_path);
        via_orb = r.load_all();
    }
    {
        HDF5Reader r(sibling_path);
        via_pot = r.load_all();
    }
    compare_pairs(via_orb, via_pot, "via orbitals.h5", "via potentials.h5");

    // Optional: also compare against legacy combined file if given.
    if (argc >= 3) {
        const std::string legacy_path = argv[2];
        std::cout << "\n  legacy combined    : " << legacy_path << "\n";
        if (std::filesystem::exists(legacy_path)) {
            HDF5Reader r(legacy_path);
            PreprocData via_legacy = r.load_all();
            compare_pairs(via_orb, via_legacy,
                          "split (orbitals path)", "legacy combined");
        } else {
            std::cout << "  [skip] legacy file not present, no extra comparison\n";
        }
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failures)\n";
    return fails == 0 ? 0 : 1;
}
