// test_orbital_truncation.cpp -- verify the --orb-lmax / --orb-rmax(-auto)
// truncation in preprocess_molden gives the same chi(j, lambda)(r) for
// (lambda < Nlm_orb_store) and (ir < Nr_orb_store) as the untruncated
// fixture, AND that asking the loader for content past the stored cuts
// fails with a clear error.
//
// The point of the truncation: drop orbital coefficients we never index
// in scattering (lambda > l_cont + l_exch, or r > r_cut where chi is
// numerical noise) so the orbitals HDF5 shrinks from O(GB) to O(MB)
// for typical C8F8 / Lmax_sce=300 production runs.
//
// What this test compares:
//   * truncated fixture: <stem>.orbitals.h5 generated with --orb-lmax 8
//     (and full Nr) -- shape (n_sce * 81, Nr)
//   * full fixture:      <stem>.orbitals.h5 generated WITHOUT --orb-lmax,
//     i.e. with the molden's Lmax=32 -- shape (n_sce * 1089, Nr)
//
// For lambda < 81 they MUST match bit-exactly: the truncation only
// crops; it doesn't recompute.

#include "io/HDF5Reader.hpp"
#include "scatt/WavefunctionSetup.hpp"
#include "scatt/Parameters.hpp"

#include <Eigen/Dense>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using scatt::io::HDF5Reader;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0]
                  << " <full_fixture.orbitals.h5> <truncated_fixture.orbitals.h5>\n";
        return 2;
    }
    const std::string full_path  = argv[1];
    const std::string trunc_path = argv[2];

    std::cout << "=== orbital truncation round-trip ===\n";
    std::cout << "  full      : " << full_path  << "\n";
    std::cout << "  truncated : " << trunc_path << "\n";

    // The truncated fixture is OPTIONAL: the new design (full angular
    // on disk, hyperslab cut at scattering load time) doesn't require
    // it.  If the fixture isn't there, just print a SKIP and exit 0 so
    // ctest stays green on the common path.  To actually run this test,
    // generate the fixture by hand:
    //   preprocess_molden REFERENCE.molden <stem> \
    //       --lmax 32 --dr 0.005 --rmax 15 --orbitals occupied \
    //       --exchange none --origin origin_of_file --orb-lmax 16
    if (!std::filesystem::exists(full_path) ||
        !std::filesystem::exists(trunc_path)) {
        std::cout << "  [SKIP] fixture not present:\n"
                  << "         full      exists: "
                  << std::filesystem::exists(full_path)  << "\n"
                  << "         truncated exists: "
                  << std::filesystem::exists(trunc_path) << "\n"
                  << "         (this fixture is opt-in; the new design"
                     " no longer needs it)\n";
        return 0;
    }

    HDF5Reader r_full(full_path);
    auto full = r_full.load_all();
    HDF5Reader r_trunc(trunc_path);
    auto trunc = r_trunc.load_all();

    // Sanity prints.
    std::cout << "  full     : Lmax_orb_store=" << full.Lmax_orb_store
              << "  Nlm=" << full.Nlm_orb_store
              << "  Nr_store=" << full.Nr_orb_store
              << "  psi_lm.shape=(" << full.psi_lm.rows() << ", "
              << full.psi_lm.cols() << ")\n";
    std::cout << "  truncated: Lmax_orb_store=" << trunc.Lmax_orb_store
              << "  Nlm=" << trunc.Nlm_orb_store
              << "  Nr_store=" << trunc.Nr_orb_store
              << "  psi_lm.shape=(" << trunc.psi_lm.rows() << ", "
              << trunc.psi_lm.cols() << ")\n";

    check(full.Lmax_orb_store > trunc.Lmax_orb_store,
          "truncated has smaller Lmax_orb_store");
    check(full.Nlm_orb_store > trunc.Nlm_orb_store,
          "truncated has smaller Nlm_orb_store");
    check(full.n_sce == trunc.n_sce,
          "n_sce identical");

    // Within the truncated lambda + r extent, every orbital coefficient
    // must be bit-equal between the full and the truncated fixtures.
    const int n_orb       = trunc.n_sce;
    const int Nlm_t       = trunc.Nlm_orb_store;
    const int Nlm_f       = full.Nlm_orb_store;
    const int Nr_compare  = std::min(trunc.Nr_orb_store, full.Nr_orb_store);

    double worst = 0.0;
    int worst_j = -1, worst_lam = -1, worst_ir = -1;
    for (int j = 0; j < n_orb; ++j) {
        const Eigen::Index row0_t = Eigen::Index(j) * Nlm_t;
        const Eigen::Index row0_f = Eigen::Index(j) * Nlm_f;
        for (int lam = 0; lam < Nlm_t; ++lam) {
            for (int ir = 0; ir < Nr_compare; ++ir) {
                const double a = trunc.psi_lm(row0_t + lam, ir);
                const double b =  full.psi_lm(row0_f + lam, ir);
                const double diff = std::abs(a - b);
                if (diff > worst) {
                    worst = diff; worst_j = j; worst_lam = lam; worst_ir = ir;
                }
            }
        }
    }
    std::cout << "  worst |trunc - full| over (lambda < " << Nlm_t
              << ", ir < " << Nr_compare << ") = " << worst
              << "  at (j=" << worst_j << ", lam=" << worst_lam
              << ", ir=" << worst_ir << ")\n";
    check(worst == 0.0, "psi_lm bit-identical inside the truncated tile");

    // load_chi_from_hdf5 should refuse to load past Nlm_orb_store.
    {
        bool threw = false;
        try {
            scatt::WavefunctionSetup::load_chi_from_hdf5(
                trunc, /*n_occ=*/n_orb,
                /*n_lambda_cut=*/Nlm_t + 1,    // one past stored
                /*n_transition=*/std::min(50, trunc.Nr_orb_store),
                /*dr=*/trunc.dr,
                /*verbose=*/false);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "loader rejects n_lambda_cut > Nlm_orb_store");
    }
    // ... but accept it at the boundary.
    {
        bool ok = true;
        try {
            auto chi = scatt::WavefunctionSetup::load_chi_from_hdf5(
                trunc, /*n_occ=*/n_orb,
                /*n_lambda_cut=*/Nlm_t,        // exactly the stored width
                /*n_transition=*/std::min(50, trunc.Nr_orb_store),
                /*dr=*/trunc.dr,
                /*verbose=*/false);
            (void)chi;
        } catch (...) { ok = false; }
        check(ok, "loader accepts n_lambda_cut == Nlm_orb_store");
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failures)\n";
    return fails == 0 ? 0 : 1;
}
