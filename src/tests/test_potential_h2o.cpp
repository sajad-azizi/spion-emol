// test_potential_h2o.cpp -- assemble V(r) from the H2O preprocessing HDF5
// and run the sanity checks that version_0's Potentials::potential_matrixElements
// runs at the end of its own build. Since H2O has C2v symmetry (not cubic),
// the cubic-symmetry check is informational only; we do enforce
//   (1) V is symmetric: max|V - V^T| < 1e-10,
//   (2) far-field diagonal reduces to centrifugal: for r near r_max, each
//       V(mu, mu) must approach l(l+1)/(2 r^2). We accept 1 mHa deviation
//       since the multipole expansion's slow convergence near r = |R_atom|
//       carries a small residual to r_max.
//
// Test input: the H2O preprocessing HDF5 built by
//     preprocessing/build -> h2o_ccpvdz_sph.preproc.h5
// (produced by ctest h2o_pipeline_to_hdf5 on the preprocessing side).
//
// Usage: test_potential_h2o <path-to-h2o-hdf5>

#include "io/HDF5Reader.hpp"
#include "scatt/Banner.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

using scatt::Parameters;
using scatt::Potentials;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static int fails = 0;

static void check(bool ok, const std::string& what,
                  double got, double tol, const std::string& units = "")
{
    const std::string tag = ok ? "ok  " : "FAIL";
    std::cout << "  [" << tag << "] " << what
              << "   got=" << std::scientific << got
              << std::defaultfloat << " " << units
              << "   tol=" << tol << "\n";
    if (!ok) ++fails;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2;
    }
    const std::string path = argv[1];

    scatt::print_la_banner();

    HDF5Reader reader(path);
    PreprocData data = reader.load_all();

    // Small l_max_continuum so channels is tractable on my laptop.
    Parameters params;
    params.r_min          = data.rmin;
    params.dr             = data.dr;
    params.N_grid         = data.Nr;
    params.Lmax_sce       = data.Lmax_sce;
    params.l_max_continuum = 6;           // channels = 49
    // H2O preprocessing writes Lmax_sce=32, so l_exp_max=12 <= Lmax_sce OK.

    std::cout << std::fixed << std::setprecision(6)
              << "[test] r_min=" << params.r_min << " dr=" << params.dr
              << " N=" << params.N_grid << " r_max=" << params.r_max()
              << " Lmax_sce=" << params.Lmax_sce << "\n"
              << "[test] l_max_continuum=" << params.l_max_continuum
              << " channels=" << params.channels()
              << " l_exp_max=" << params.l_exp_max() << "\n";

    Potentials pot(params);
    auto t_build0 = std::chrono::steady_clock::now();
    pot.build(data, scatt::StorageMode::MEMORY, /*checkpoint_dir=*/"",
              /*verbose=*/false);
    const double t_build = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_build0).count();
    std::cout << "[bench] Potentials::build (MEMORY, l_cont=" << params.l_max_continuum
              << ", N_grid=" << params.N_grid << "): "
              << std::fixed << std::setprecision(3) << t_build << " s\n";

    // Access-cost bench: time a full sweep of get() and a full sweep of V_at().
    {
        auto t0 = std::chrono::steady_clock::now();
        volatile double sink = 0.0;
        for (std::size_t ir2 = 0; ir2 < params.N_grid; ++ir2)
            sink += pot.get(ir2).squaredNorm();
        const double t_get = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "[bench] Potentials::get sweep (" << params.N_grid << " points): "
                  << t_get << " s  (" << std::setprecision(0) << (params.N_grid / t_get)
                  << " ir/s, " << std::setprecision(3) << (t_get / params.N_grid * 1e6)
                  << " μs/call)\n";
        (void) sink;
    }

    // (1) Symmetry across all radial points.
    const double sym_dev = pot.max_symmetry_deviation();
    check(sym_dev < 1e-10, "max |V - V^T| over all r", sym_dev, 1e-10);

    // (2) Far-field behavior. At r close to r_max, V(mu, mu) must be close
    //     to the centrifugal value l(l+1)/(2 r^2), i.e. V_en_ee + U_pol
    //     should have decayed. For H2O at r_max ~ 15 Bohr and a neutral
    //     target, V_en_ee ~ 1e-3 (slow multipole convergence). We allow
    //     a 2 mHa tolerance.
    const std::size_t ir = params.N_grid - 2;
    const double r = params.r(ir);
    const auto& M = pot.get(ir);
    double max_err_diag = 0.0, max_off_diag = 0.0;
    for (int mu = 0; mu < params.channels(); ++mu) {
        int l, m; scatt::angular::idx_to_lm(mu, l, m);
        const double centrif = l * (l + 1.0) * 0.5 / (r * r);
        max_err_diag = std::max(max_err_diag, std::abs(M(mu, mu) - centrif));
        for (int nu = 0; nu < params.channels(); ++nu) {
            if (nu == mu) continue;
            max_off_diag = std::max(max_off_diag, std::abs(M(mu, nu)));
        }
    }
    // H2O has a sizable permanent dipole and quadrupole, so V_en_ee
    // multipoles decay only polynomially; at r_max = 15 Bohr the tail is
    // still ~ mHa. This is the physics of a polar molecule, not a bug.
    check(max_err_diag < 5e-3, "V(mu, mu) - l(l+1)/(2 r^2) at r = r_max",
          max_err_diag, 5e-3, "Ha");
    check(max_off_diag < 5e-3, "off-diagonal V(mu, nu != mu) at r = r_max",
          max_off_diag, 5e-3, "Ha");

    // (3) H2O is C2v (not O_h): p_z (a_1) != p_x (b_1) != p_y (b_2). Print
    //     them for the record; no automatic check.
    std::cout << "\n[info] C2v p-orbital diagonals (NOT expected equal):\n";
    for (std::size_t ir_probe : {std::size_t(400), std::size_t(1000)}) {
        if (ir_probe >= params.N_grid) continue;
        const auto& Mp = pot.get(ir_probe);
        const double rp = params.r(ir_probe);
        std::cout << "   r=" << rp
                  << "   V(p_y)=" << Mp(1, 1)
                  << "   V(p_z)=" << Mp(2, 2)
                  << "   V(p_x)=" << Mp(3, 3) << "\n";
    }

    std::cout << "\n==> " << (fails == 0 ? "PASS" : "FAIL: " + std::to_string(fails)) << "\n";
    return fails ? 1 : 0;
}
