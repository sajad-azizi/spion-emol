// main.cpp -- end-to-end driver for the 3D coupled-channel model code,
// matching the parent project's ergonomics:
//
//   $WORK/
//       bound_<molhash>.h5                      # energy-independent cache
//       dipole_<molhash>_<scan_id>/             # one directory per scan
//           manifest.h5                         # scan-level metadata
//           ikNNNN.h5                           # per-energy dipole payload
//
// Steps per run:
//   1. Build coupled-channel V_eff(r) for the chosen potential.
//   2. Try to LOAD the bound state from $WORK/bound_<molhash>.h5.
//      On cache miss: bisection -> calculate_eigenfunction -> Normalize
//      -> save to $WORK/bound_<molhash>.h5.
//   3. Write manifest.h5 once if absent.
//   4. For each ik in [ik_min, ik_max]:
//      - If ikNNNN.h5 exists already, skip.
//      - Compute continuum at E_kin = 0.5 * (ik*dk)^2, fit A,B.
//      - Compute length + velocity dipole for q ∈ {-1, 0, +1}.
//      - Save ikNNNN.h5 (atomic write).
//
// The output is byte-compatible with static_exchangeHF/postprocessing/
// gather_dipoles.py and cross_section_delay.py.

#include "Common.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"
#include "Wavefunctions.hpp"
#include "DipoleMat.hpp"
#include "Angular.hpp"
#include "RunDirs.hpp"
#include "BoundStateIO.hpp"
#include "DipoleIO.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cstdio>     // setvbuf, _IOLBF, _IONBF, printf
#include <thread>

namespace fs = std::filesystem;

namespace {

std::string iso_date_utc_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

void usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [flags]\n"
        "\n"
        "Potential selection:\n"
        "  -k <kind>        cubic | spherical | gaussian | anis_gauss |\n"
        "                   harmonic | soft_coul | h2plus | h2plus_johnson |\n"
        "                   free                              [default cubic]\n"
        "  -V0 <a.u.>       well depth (positive)            [default from src]\n"
        "  -L  <100*Lbox>   box half-side / radius * 100     [default from src]\n"
        "\n"
        "Grid + angular cutoff:\n"
        "  -lmax  <N>       angular cutoff                   [default 4]\n"
        "  -N     <Ngrid>   number of radial points          [default 3001]\n"
        "  -dr    <H>       radial step (a.u.)               [default 0.01]\n"
        "\n"
        "Energy scan (k_n = ik*dk; E_kin = k_n^2 / 2):\n"
        "  -ik-min <N>      first ik (default 1)\n"
        "  -ik-max <M>      last ik, INCLUSIVE (default 1)\n"
        "  -dk     <H>      momentum step (default 0.01)\n"
        "\n"
        "Output:\n"
        "  -work    DIR     persistent dir   (default $WORK or ./work)\n"
        "  -scratch DIR     scratch dir      (default $SCRATCH or ./scratch)\n"
        "  -scan-id STR     scan tag (default auto: ik<min>-<max>_dk<dk>)\n"
        "  -name    STR     molecule name to record in manifest\n"
        "\n"
        "  -h, --help       this help\n";
}

}  // namespace



void apply_line_buffering(){
    // Force line-buffering on the C FILE* streams.
    // MUST be called before any I/O on the stream.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);
 
    // Keep C++ streams (cout/cerr) in sync with the C FILE* layer.
    std::ios::sync_with_stdio(true);
 
    // Tie cerr to cout so cerr flushes cout first (prevents interleaving).
    std::cerr.tie(&std::cout);
}

int main(int argc, char** argv)
{
    apply_line_buffering();
    // ---- Defaults --------------------------------------------------
    Parameters params;
    params.N_grid     = 10001;
    params.dr         = 0.01;
    params.l_max      = 15;
    params.n_channels = ang3d::n_channels(params.l_max);
    params.Emin       = -3.0;
    params.Emax       = -0.001;
    params.N_theta    = 64;
    params.N_phi      = 128;
    params.p          = 9;
    params.external_parameter = 294;     // L_box = 1.5 a.u. by default
    params.out_decimation = 5;
#ifdef _OPENMP
    params.n_threads = omp_get_max_threads();
#else
    params.n_threads = 1;
#endif

    std::string kind        = "cubic";
    std::string molecule    = "spherical_3d_model";
    std::string cli_work, cli_scratch, scan_id_override;
    int    ik_min = 1, ik_max = 1;
    double dk     = 0.01;
    bool   user_set_V0 = false;
    double user_V0     = 0.75;

    // ---- CLI parsing ----------------------------------------------
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value after " << a << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if      (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "-k")        kind = next();
        else if (a == "-V0")       { user_V0 = std::stod(next()); user_set_V0 = true; }
        else if (a == "-L")        params.external_parameter = std::stoi(next());
        else if (a == "-lmax")     {
            params.l_max = std::stoi(next());
            params.n_channels = ang3d::n_channels(params.l_max);
        }
        else if (a == "-N")        params.N_grid = std::stoi(next());
        else if (a == "-dr")       params.dr     = std::stod(next());
        else if (a == "-ik-min")   ik_min = std::stoi(next());
        else if (a == "-ik-max")   ik_max = std::stoi(next());
        else if (a == "-dk")       dk     = std::stod(next());
        else if (a == "-work")     cli_work    = next();
        else if (a == "-scratch")  cli_scratch = next();
        else if (a == "-scan-id")  scan_id_override = next();
        else if (a == "-name")     molecule    = next();
        else {
            std::cerr << "unknown flag: " << a << "\n";
            usage(argv[0]); return 2;
        }
    }
    if (ik_max < ik_min) {
        std::cerr << "ik_max < ik_min\n"; return 2;
    }

    // ---- Resolve directories + molhash ----------------------------
    auto dirs = sph3d::resolve_dirs(cli_work, cli_scratch);

    sph3d::RunConfig rc;
    rc.kind     = kind;
    rc.V0       = user_set_V0 ? user_V0 : 0.75;
    rc.L_box    = params.external_parameter * 0.01;
    rc.N_grid   = params.N_grid;
    rc.dr       = params.dr;
    rc.l_max    = params.l_max;
    rc.N_theta  = params.N_theta;
    rc.N_phi    = params.N_phi;
    const std::string mhash = sph3d::molhash(rc);

    std::ostringstream scan_oss;
    if (!scan_id_override.empty()) {
        scan_oss << scan_id_override;
    } else {
        scan_oss << "ik" << ik_min << "-" << ik_max
                 << "_dk" << std::fixed << std::setprecision(4) << dk;
    }
    const std::string scan_id  = scan_oss.str();
    const std::string bound_h5 = dirs.work + "/bound_" + mhash + ".h5";
    const std::string scan_dir = dirs.work + "/dipole_" + mhash + "_" + scan_id;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "[3d] kind=" << kind
              << "  V0=" << rc.V0
              << "  L_box=" << rc.L_box
              << "  N_grid=" << params.N_grid
              << "  dr=" << params.dr
              << "  l_max=" << params.l_max
              << "  n_channels=" << params.n_channels
              << "\n";
    std::cout << "[3d] ik=[" << ik_min << ".." << ik_max
              << "]  dk=" << dk << "\n";
    std::cout << "[3d] WORK    = " << dirs.work    << "\n"
              << "[3d] SCRATCH = " << dirs.scratch << "\n"
              << "[3d] molhash = " << mhash << "\n"
              << "[3d] bound   = " << bound_h5 << "\n"
              << "[3d] scan    = " << scan_dir << "\n";

    // ---- Potential -------------------------------------------------
    Potentials pot(params);
    if (user_set_V0) pot.set_V0(user_V0);
    pot.set_L(rc.L_box);
    pot.set_potential(kind);
    pot.build();
    std::cout << "[3d] potential matrix built\n";

    Equations eqs(pot, params);

    // ---- Bound state: load or compute -----------------------------
    sph3d::BoundState bs;
    bool loaded = sph3d::load_bound_state(bound_h5, mhash,
                                           params.N_grid,
                                           params.n_channels, bs);
    Wavefunctions wfs(eqs, params);
    int    i_match = 0;
    double E_HOMO  = 0.0;

    if (loaded) {
        std::cout << "[3d] bound state: LOADED from " << bound_h5
                  << "  E = " << bs.gsEnergy
                  << "  i_match = " << bs.i_match << "\n";
        wfs.eigfunc = bs.eigfunc;
        i_match     = bs.i_match;
        E_HOMO      = bs.gsEnergy;
    } else {
        std::cout << "[3d] bound state: COMPUTING (cache miss)\n";
        Eigenvalues eig(eqs, params);
        eig.groundstate_finder();
        wfs.calculate_eigenfunction(eig.gsEnergy, eig.i_match);
        wfs.Normalization(wfs.eigfunc);

        bs.gsEnergy   = eig.gsEnergy;
        bs.i_match    = eig.i_match;
        bs.N_grid     = params.N_grid;
        bs.n_channels = params.n_channels;
        bs.dr         = params.dr;
        bs.molhash    = mhash;
        bs.eigfunc    = wfs.eigfunc;
        sph3d::save_bound_state(bound_h5, bs);
        std::cout << "[3d] bound state: SAVED to " << bound_h5
                  << "  E = " << bs.gsEnergy
                  << "  i_match = " << bs.i_match << "\n";
        i_match = bs.i_match;
        E_HOMO  = bs.gsEnergy;
    }
    (void)i_match;   // not used downstream but kept for clarity

    // ---- Write manifest (idempotent: rewrites every run) ----------
    sph3d::ScanMeta meta;
    meta.r_min            = 0.0;
    meta.dr               = params.dr;
    meta.N_grid           = params.N_grid;
    meta.l_max_continuum  = params.l_max;
    meta.E_HOMO           = E_HOMO;
    meta.dk               = dk;
    meta.ik_min           = ik_min;
    meta.ik_max           = ik_max;
    meta.n_occ            = 1;
    meta.occ_energies     = { E_HOMO };
    meta.occ_spin_factors = { 1.0 };  // single-particle bound state
    meta.molecule_name    = molecule;
    meta.molhash          = mhash;
    meta.iso_date_utc     = iso_date_utc_now();
    sph3d::write_manifest(scan_dir, meta);
    std::cout << "[3d] manifest written: "
              << (scan_dir + "/manifest.h5") << "\n";

    // ---- Per-ik loop ----------------------------------------------
    DipoleMat dip(wfs, params);
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(params.n_channels, params.n_channels);
    Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(params.n_channels, params.n_channels);
    Eigen::MatrixXcd In = Eigen::MatrixXcd::Identity(params.n_channels, params.n_channels);

    int n_done = 0, n_skipped = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int ik = ik_min; ik <= ik_max; ++ik) {
        if (sph3d::ik_exists(scan_dir, ik)) {
            ++n_skipped;
            std::cout << "[3d] " << sph3d::ik_tag(ik)
                      << ": EXISTS, skipping\n";
            continue;
        }
        const double k_au = ik * dk;
        const double E_kin = 0.5 * k_au * k_au;
        const double omega = E_kin - E_HOMO;

        std::cout << "[3d] " << sph3d::ik_tag(ik)
                  << "  k=" << k_au
                  << "  E_kin=" << E_kin
                  << "  omega=" << omega << "\n";

        // Continuum + asymptotic A,B at this energy.
        wfs.calculate_channel_wavefunction(E_kin);
        wfs.calculate_A_B_matrices(A, B, E_kin);

        // S/K diagnostic (kept consistent with the older single-shot main).
        Eigen::MatrixXcd K = B * A.inverse();
        Eigen::MatrixXcd S = (In + I_unit * K) * (In - I_unit * K).inverse();
        const double unit_err =
            (S.adjoint() * S - In).cwiseAbs().maxCoeff();
        std::cout << "      |S^* S - I|_max = " << unit_err << "\n";

        // Compute dipole for each q and gauge.
        sph3d::DipolePerIK p;
        p.ik    = ik;
        p.k     = k_au;
        p.E     = E_kin;
        p.omega = omega;
        for (int qi = 0; qi < 3; ++qi) {
            // pol=0:x (q=+1), 1:y (q=-1), 2:z (q=0)
            const int q = (qi == 0) ? +1 : (qi == 1 ? -1 : 0);
            auto dL = dip.compute         (q, A, B, E_kin);
            auto dV = dip.compute_velocity(q, A, B, E_kin);
            const int slot_L = 0 * 3 + qi;
            const int slot_V = 1 * 3 + qi;
            p.D_raw  [slot_L] = dL;
            p.D_ortho[slot_L] = dL;     // no occ-orthogonalization
            p.D_raw  [slot_V] = dV;
            p.D_ortho[slot_V] = dV;
        }
        sph3d::write_ik(scan_dir, p);
        ++n_done;
    }

    const double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "[3d] done.  written=" << n_done
              << "  skipped=" << n_skipped
              << "  wall=" << dt << " s\n";
    return 0;
}
