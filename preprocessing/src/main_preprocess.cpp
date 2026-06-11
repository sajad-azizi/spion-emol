// main_preprocess.cpp — orchestrator that turns a Psi4 molden file into
// the SCE-based preprocessing artifact (HDF5).
//
// Pipeline:
//   1) Parse molden           (molden/Molden.hpp)
//   2) Build AO basis         (basis/MoldenBasis.hpp)          [s + p for now]
//   3) Pick SCE origin        (default = atomic geometric mean)
//   4) Build radial + angular grids
//   5) SCE-project occupied MOs and the total density rho
//   6) Build V_en (multipole)
//   7) Build V_H  (radial Poisson)
//   8) Build local exchange V_x (KS-LDA) via SCE projection of f(rho(r_vec))
//   9) Sanity checks: N_e, orbital norms, <rho|V_en>, 0.5*<rho|V_H>
//  10) Write everything to HDF5
//
// Usage:
//   preprocess_molden <molden_in> <hdf5_out> [--lmax L] [--dr H] [--rmax R]
//
// Accuracy defaults (chosen from the Milestone 6 convergence study that
// hit nanoHartree agreement with Psi4):
//   --lmax 24  --dr 0.01  --rmax 15

#include "angular/Grid.hpp"
#include "angular/Ylm.hpp"
#include "basis/MoldenBasis.hpp"
#include "Banner.hpp"
#include "io/HDF5Writer.hpp"
#include "molden/Molden.hpp"
#include "potential/Hartree.hpp"
#include "potential/LocalExchange.hpp"
#include "potential/Vnuclear.hpp"
#include "sce/RadialGrid.hpp"
#include "sce/SCE.hpp"
#include "tiny_json.hpp"      // used to read polarizability from the reference JSON

#include <Eigen/Dense>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

using preproc::angular::AngGrid;
using preproc::angular::lm_index;

static void usage(const char* prog) {
    std::cerr <<
      "Usage: " << prog << " <molden_in> [hdf5_out] [flags...]\n"
      "       " << prog << " --help\n"
      "\n"
      "Positional:\n"
      "  molden_in              Path to the Psi4/Molden input file (required).\n"
      "  hdf5_out               Output HDF5 path (optional). If omitted:\n"
      "                           - If $WORK is set: $WORK/<stem>.preproc.h5\n"
      "                           - Else          : ./<stem>.preproc.h5\n"
      "                         where <stem> is the molden_in basename without extension.\n"
      "                         If given, this path is used verbatim (absolute or relative).\n"
      "\n"
      "Grid + angular:\n"
      "  --lmax L               SCE angular cutoff                  (default 24)\n"
      "  --dr H                 radial step, au                     (default 0.01)\n"
      "  --rmax R               outer radius, au                    (default 15)\n"
      "\n"
      "Physics:\n"
      "  --exchange MODE        slater | lda | none                 (default lda)\n"
      "  --origin CHOICE        auto | com | atoms | origin_of_file (default atoms)\n"
      "  --orbitals SET         all | occupied                      (default occupied)\n"
      "                           'occupied' : SCE only MOs with occ>0 (static-exchange K default)\n"
      "                           'all'      : also SCE virtuals (post-HF / close-coupling)\n"
      "\n"
      "Input-convention validation gate (catches QC-format conversion bugs):\n"
      "  --no-validate-input          disable the SCE self-check (default: ENABLED)\n"
      "  --validate-norm-lo X         min allowed occupied-orbital ||psi||^2 (default 0.50)\n"
      "  --validate-norm-hi X         max allowed occupied-orbital ||psi||^2 (default 1.10)\n"
      "  --validate-ne-tol  X         max |N_e(SCE rho)/N_e(occ) - 1|        (default 0.10)\n"
      "                           The gate aborts (exit 3) if a converted QC input has the\n"
      "                           wrong spherical-harmonic order / normalisation / sign --\n"
      "                           a silent-wrong-cross-section bug otherwise.  Use\n"
      "                           qc_to_molden.py to convert Gaussian/PySCF/MOLPRO/GAMESS/\n"
      "                           ORCA/Q-Chem outputs into the canonical molden this reads.\n"
      "\n"
      "Optional extras:\n"
      "  --polarizability <9 floats>             3x3 alpha tensor, row-major, au\n"
      "  --polarizability-from-json <path>       load alpha tensor from a psi4 reference JSON\n"
      "  --initial-state-molden <path>           anion molden (e.g. from a solvent SCF).\n"
      "                                          Its SOMO is SCE-projected onto the same grid\n"
      "                                          and stored under /initial_state/, consumed\n"
      "                                          downstream by the dipole matrix element.\n"
      "\n"
      "Orbital storage truncation (saves orders of magnitude of disk + RAM):\n"
      "  --orb-lmax L                  store orbitals only up to angular l=L  (default: full Lmax)\n"
      "                                Set this to ~ Lmax_continuum + Lmax_exchange  (e.g. 40)\n"
      "                                so that downstream scattering still has every (l,m)\n"
      "                                channel it ever indexes.  V_H / density use the FULL\n"
      "                                Lmax internally and are unaffected.\n"
      "  --orb-rmax R                  store orbitals only up to r=R bohr      (default: full rmax)\n"
      "  --orb-rmax-auto THRESH        auto-detect r_cut: smallest r where every orbital is\n"
      "                                below |chi(r)|=THRESH (e.g. 1e-7) for all r' >= r_cut.\n"
      "                                Mutually exclusive with --orb-rmax.\n"
      "\n"
      "  -h, --help             print this help and exit\n";
}

int main(int argc, char** argv) {
    // Handle --help anywhere on the command line, before any other parsing.
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    }
    if (argc < 2) { usage(argv[0]); return 2; }
    const std::string molden_path = argv[1];

    // Output path: either positional argv[2] (if it isn't a flag), or derived
    // from molden_path + $WORK. "--" starts a flag, everything else is the
    // output path.
    std::string out_path;
    int first_flag_idx = 2;
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
        out_path = argv[2];
        first_flag_idx = 3;
    }
    if (out_path.empty()) {
        const std::string stem = std::filesystem::path(molden_path).stem().string();
        const char* work_env = std::getenv("WORK");
        const bool have_work = (work_env && *work_env);
        std::string base_dir = have_work ? std::string(work_env) : std::string(".");
        if (have_work) std::filesystem::create_directories(base_dir);
        out_path = base_dir + "/" + stem;
        std::cerr << "[preprocess] no output path given; defaulting stem to " << out_path
                  << "  (WORK=" << (have_work ? work_env : "unset") << ")\n";
    }

    // The output is split into TWO HDF5 files: orbitals (the big, expensive
    // SCE-projected MO data) and potentials (V_en, V_H, V_x, V_total, rho,
    // metadata).  Derive both paths from the user's `out_path` by stripping
    // any of {.h5, .preproc.h5, .orbitals.h5, .potentials.h5} and appending
    // `.orbitals.h5` / `.potentials.h5`.  This means the same `out_path`
    // value can be handed to either preprocess_molden or repair_v_h /
    // HDF5Reader without surprises.
    auto strip_suffix = [](std::string s, const std::string& suf) {
        if (s.size() >= suf.size() &&
            s.compare(s.size() - suf.size(), suf.size(), suf) == 0)
            s.resize(s.size() - suf.size());
        return s;
    };
    std::string out_stem = out_path;
    for (const auto& suf : { std::string(".orbitals.h5"),
                             std::string(".potentials.h5"),
                             std::string(".preproc.h5"),
                             std::string(".h5") }) {
        const std::string before = out_stem;
        out_stem = strip_suffix(out_stem, suf);
        if (out_stem != before) break;
    }
    const std::string orb_out = out_stem + ".orbitals.h5";
    const std::string pot_out = out_stem + ".potentials.h5";

    int    Lmax = 24;
    double dr   = 0.01;
    double rmax = 15.0;
    std::string exchange = "lda";
    std::string origin_choice = "atoms";
    std::string orbital_set = "occupied";    // "all" | "occupied" -- static-exchange K needs only occupied
    std::vector<double> polarizability;      // size 9 if provided
    std::string polarizability_json;         // optional path to reference JSON
    std::string initial_state_molden;        // optional path to anion molden
    // Orbital-storage truncation knobs:
    //   * orb_lmax_store < 0  => store full Lmax (back-compat)
    //   * orb_rmax_store < 0  => store full radial grid (back-compat)
    //   * orb_rmax_auto_thresh > 0 => auto-detect cutoff radius
    int    orb_lmax_store        = -1;
    double orb_rmax_store        = -1.0;
    double orb_rmax_auto_thresh  = -1.0;
    // SCE input-validation gate.  After the SCE projection we check that
    // the density is consistent with the input occupations -- this is the
    // safety net that catches a quantum-chemistry-format CONVENTION error
    // (wrong spherical-harmonic order / normalisation / sign) from a
    // converted input BEFORE it silently corrupts the scattering cross
    // section.  Thresholds are intentionally LOOSE so they pass legitimate
    // low-Lmax SCE truncation but fail a convention disaster:
    //   * each occupied orbital norm ||psi||^2 in [norm_lo, norm_hi]
    //   * |N_e(SCE rho) / N_e(occupations) - 1| <= ne_tol
    // A separate ADVISORY (not a failure) fires when norms are below the
    // ePolyScat 0.99 production criterion -- "increase --lmax".
    bool   validate_input        = true;
    double val_norm_lo           = 0.50;   // below this => convention error
    double val_norm_hi           = 1.10;   // above this => double-normalisation
    double val_ne_tol            = 0.10;   // 10% electron-count tolerance
    for (int i = first_flag_idx; i < argc; ++i) {
        std::string a = argv[i];
        auto arg = [&](int o) -> std::string { return (i + o < argc) ? argv[i + o] : ""; };
        if      (a == "--lmax")    { Lmax = std::stoi(arg(1)); ++i; }
        else if (a == "--dr")      { dr   = std::stod(arg(1)); ++i; }
        else if (a == "--rmax")    { rmax = std::stod(arg(1)); ++i; }
        else if (a == "--exchange"){ exchange = arg(1); ++i; }
        else if (a == "--origin")  { origin_choice = arg(1); ++i; }
        else if (a == "--orbitals"){ orbital_set = arg(1); ++i; }
        else if (a == "--orb-lmax")     { orb_lmax_store       = std::stoi(arg(1)); ++i; }
        else if (a == "--orb-rmax")     { orb_rmax_store       = std::stod(arg(1)); ++i; }
        else if (a == "--orb-rmax-auto"){ orb_rmax_auto_thresh = std::stod(arg(1)); ++i; }
        else if (a == "--no-validate-input") { validate_input = false; }
        else if (a == "--validate-norm-lo")  { val_norm_lo = std::stod(arg(1)); ++i; }
        else if (a == "--validate-norm-hi")  { val_norm_hi = std::stod(arg(1)); ++i; }
        else if (a == "--validate-ne-tol")   { val_ne_tol  = std::stod(arg(1)); ++i; }
        else if (a == "--polarizability") {
            polarizability.clear();
            for (int k = 0; k < 9; ++k) polarizability.push_back(std::stod(arg(1 + k)));
            i += 9;
        }
        else if (a == "--polarizability-from-json") {
            polarizability_json = arg(1); ++i;
        }
        else if (a == "--initial-state-molden") {
            initial_state_molden = arg(1); ++i;
        }
        else { std::cerr << "unknown option: " << a << "\n"; usage(argv[0]); return 2; }
    }
    if (orb_rmax_store > 0.0 && orb_rmax_auto_thresh > 0.0) {
        std::cerr << "[preprocess] --orb-rmax and --orb-rmax-auto are mutually exclusive\n";
        return 2;
    }
    const int Nr = static_cast<int>(std::round(rmax / dr)) + 1;

    // If polarizability was requested via JSON, load it here.
    if (!polarizability_json.empty() && polarizability.empty()) {
        tinyjson::Value ref = tinyjson::parse_file(polarizability_json);
        const auto& obj = ref.as_obj();
        auto it = obj.find("polarizability");
        if (it == obj.end())
            throw std::runtime_error("polarizability-from-json: missing 'polarizability' key");
        const auto& arr = it->second.as_obj().at("alpha_tensor").as_arr();
        if (arr.size() != 9)
            throw std::runtime_error("polarizability tensor must have 9 elements");
        polarizability.reserve(9);
        for (const auto& v : arr) polarizability.push_back(v.as_num());
        std::cerr << "[preprocess] loaded polarizability tensor from "
                  << polarizability_json << "\n";
    }

    std::cerr << std::setprecision(12);
    preproc::print_banner();
    std::cerr << "[preprocess] molden = " << molden_path << "\n";
    std::cerr << "[preprocess] out (orbitals)   = " << orb_out << "\n";
    std::cerr << "[preprocess] out (potentials) = " << pot_out << "\n";
    std::cerr << "[preprocess] Lmax=" << Lmax << " dr=" << dr
              << " rmax=" << rmax << " Nr=" << Nr
              << " exchange=" << exchange << " origin=" << origin_choice
              << " orbitals=" << orbital_set << "\n";

    // -------- 1) parse molden + build AO basis --------
    preproc::molden::MoldenParser P(false);
    auto mol   = P.parse(molden_path);
    auto basis = preproc::basis::build_basis(mol, false);
    std::cerr << "[preprocess] atoms=" << mol.atoms.size()
              << "  nbf=" << mol.nbf
              << "  alpha_MOs=" << mol.mos_alpha.size()
              << "  beta_MOs="  << mol.mos_beta.size() << "\n";

    // Counts. We SCE ALL alpha MOs (occupied + virtual) because the
    // exchange operator constructed by the downstream scattering code
    // needs bound orbitals beyond just the truly-occupied subset (see
    // version_0/Wavefunctions.cpp where n_occ is a user-chosen cutoff,
    // typically larger than the number of doubly-occupied MOs).
    const int n_alpha = static_cast<int>(mol.mos_alpha.size());
    int n_occ_alpha = 0;
    double N_e_expected = 0.0;
    for (const auto& m : mol.mos_alpha) if (m.occ > 0.0) ++n_occ_alpha;
    for (const auto& m : mol.mos_alpha) N_e_expected += m.occ;
    for (const auto& m : mol.mos_beta ) N_e_expected += m.occ;
    std::cerr << "[preprocess] n_alpha(all)=" << n_alpha
              << "  n_occ_alpha=" << n_occ_alpha
              << "  N_e(from occupations)=" << N_e_expected << "\n";

    // -------- 2) SCE origin --------
    // We ALWAYS place the SCE origin at (0,0,0) in the output and translate
    // all atoms so the chosen center coincides with the origin. This matches
    // version_0 and simplifies all downstream bookkeeping (no origin offset
    // ever needs to be carried around).
    Eigen::Vector3d chosen_center = Eigen::Vector3d::Zero();
    if (origin_choice == "atoms" || origin_choice == "auto") {
        for (const auto& a : mol.atoms) chosen_center += a.xyz;
        chosen_center /= static_cast<double>(mol.atoms.size());
    } else if (origin_choice == "com") {
        double Mtot = 0.0;
        for (const auto& a : mol.atoms) {
            const double w = static_cast<double>(a.Z);   // Z-weighted proxy
            chosen_center += w * a.xyz; Mtot += w;
        }
        chosen_center /= Mtot;
    } else if (origin_choice == "origin_of_file") {
        chosen_center = Eigen::Vector3d::Zero();
    }
    std::cerr << "[preprocess] chosen center (Bohr, in input coords) = "
              << chosen_center.transpose() << "\n";
    // Translate all atoms so the chosen center is at (0,0,0).
    for (auto& a : mol.atoms) a.xyz -= chosen_center;
    // AO basis needs rebuilding because its centers are cached.
    basis = preproc::basis::build_basis(mol, false);
    const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    std::cerr << "[preprocess] SCE origin (after translation) = "
              << origin.transpose() << "\n";

    // -------- 3) build grids + SCE density --------
    auto rg = preproc::sce::RadialGrid::build(0.0, dr, Nr);
    auto ag = AngGrid::build_basic(Lmax);
    const int Nlm = preproc::angular::n_channels(Lmax);

    preproc::sce::F3D F_rho = [&](const Eigen::Vector3d& r) {
        return preproc::basis::evaluate_density(basis, mol, r);
    };

    auto t0 = std::chrono::steady_clock::now();
    std::cerr << "[preprocess] SCE: projecting total density ρ(r,Ω) onto "
                 "Y_{lm} up to Lmax=" << Lmax
              << "  (Nlm=" << Nlm << ", Nr=" << Nr << ") ...\n";
    Eigen::MatrixXd Frho = preproc::sce::project(rg, ag, origin, F_rho, false);
    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "[preprocess] SCE: ρ done in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    // Alpha orbitals to SCE. Selection governed by --orbitals:
    //   all      -> every alpha MO (n_alpha)  -- needed for downstream exchange
    //   occupied -> only MOs with occ > 0     -- faster, smaller HDF5
    std::vector<int> orb_indices;
    if (orbital_set == "all") {
        orb_indices.reserve(n_alpha);
        for (int j = 0; j < n_alpha; ++j) orb_indices.push_back(j);
    } else if (orbital_set == "occupied") {
        for (int j = 0; j < n_alpha; ++j) {
            if (mol.mos_alpha[j].occ > 0.0) orb_indices.push_back(j);
        }
    } else {
        throw std::runtime_error("--orbitals must be 'all' or 'occupied'");
    }
    const int n_sce_orb = static_cast<int>(orb_indices.size());
    std::cerr << "[preprocess] SCE will be applied to " << n_sce_orb
              << " alpha MOs (selection=" << orbital_set << ")\n";

    // Decide orbital storage extents.  Default: full Lmax * Nr (back-compat
    // with files generated by older preprocess_molden).  CLI flags
    // --orb-lmax / --orb-rmax / --orb-rmax-auto narrow them down to save
    // disk + RAM (an order of magnitude or more for C8F8 at Lmax_sce=300).
    int Lmax_orb_store = (orb_lmax_store >= 0) ? orb_lmax_store : Lmax;
    if (Lmax_orb_store > Lmax) {
        std::cerr << "[preprocess] WARN: --orb-lmax (" << orb_lmax_store
                  << ") > --lmax (" << Lmax << "); clamping to Lmax\n";
        Lmax_orb_store = Lmax;
    }
    const int Nlm_orb_store = preproc::angular::n_channels(Lmax_orb_store);

    int Nr_orb_store = Nr;
    if (orb_rmax_store > 0.0) {
        Nr_orb_store = std::min(
            Nr,
            static_cast<int>(std::round(orb_rmax_store / dr)) + 1);
    }
    // orb_rmax_auto_thresh handled BELOW after we've projected each
    // orbital -- we need the data to scan for the cutoff radius.

    Eigen::MatrixXd Fpsi(
        static_cast<Eigen::Index>(n_sce_orb) * Nlm_orb_store, Nr_orb_store);
    Fpsi.setZero();
    std::vector<double> orb_energies(n_sce_orb), orb_occ(n_sce_orb);
    std::vector<int>    orb_molden_index(n_sce_orb);
    // Per-orbital untruncated SCE norm ||psi||^2, captured for the
    // input-validation gate below.
    std::vector<double> orb_norm_sq(n_sce_orb, 0.0);

    if (Lmax_orb_store < Lmax || Nr_orb_store < Nr || orb_rmax_auto_thresh > 0.0) {
        std::cerr << "[preprocess] orbital storage: Lmax_store="
                  << Lmax_orb_store << " (vs Lmax=" << Lmax << ")"
                  << "  Nr_store=" << Nr_orb_store
                  << " (vs Nr=" << Nr << ")"
                  << "  -- truncating saves "
                  << std::fixed << std::setprecision(1)
                  << ((double(Nlm) * Nr * 8 / 1e9) /
                      (double(Nlm_orb_store) * Nr_orb_store * 8 / 1e9))
                  << "x per orbital\n";
        if (orb_rmax_auto_thresh > 0.0)
            std::cerr << "[preprocess]   (Nr_store will be tightened further "
                         "by --orb-rmax-auto " << orb_rmax_auto_thresh << ")\n";
    }
    {
        std::cerr << "[preprocess] SCE: projecting " << n_sce_orb
                  << " alpha orbitals ...\n";
        auto t_orb_start = std::chrono::steady_clock::now();
        for (int jj = 0; jj < n_sce_orb; ++jj) {
            const int j = orb_indices[jj];
            const auto& mo = mol.mos_alpha[j];
            orb_energies[jj]     = mo.energy;
            orb_occ[jj]          = mo.occ;
            orb_molden_index[jj] = j;
            auto t_orb0 = std::chrono::steady_clock::now();
            preproc::sce::F3D F_psi = [&](const Eigen::Vector3d& r) {
                return preproc::basis::evaluate_mo(basis, mo, r);
            };
            Eigen::MatrixXd Fp = preproc::sce::project(rg, ag, origin, F_psi, false);
            // Compute norm BEFORE truncation -- this is the ground-truth
            // SCE projection accuracy gate.
            const double norm_sq = preproc::sce::norm_squared(Fp, rg);
            orb_norm_sq[jj] = norm_sq;
            // Slice into the (smaller) storage tile.
            // Rows: the first Nlm_orb_store of the full Nlm.
            // Cols: the first Nr_orb_store of the full Nr.
            Fpsi.block(jj * Nlm_orb_store, 0, Nlm_orb_store, Nr_orb_store) =
                Fp.topLeftCorner(Nlm_orb_store, Nr_orb_store);
            auto t_orb1 = std::chrono::steady_clock::now();
            const double dt_orb = std::chrono::duration<double>(t_orb1 - t_orb0).count();
            // Elapsed so far + ETA from running average.
            const double dt_all = std::chrono::duration<double>(t_orb1 - t_orb_start).count();
            const double eta    = dt_all * (n_sce_orb - (jj + 1)) / double(jj + 1);
            std::cerr << "  [preprocess] orbital " << (jj + 1) << "/" << n_sce_orb
                      << "  (molden idx=" << j
                      << ", occ=" << mo.occ
                      << ", ε=" << mo.energy << " Ha"
                      << ", ||ψ||²=" << norm_sq << ")"
                      << "  in " << dt_orb << " s"
                      << "  (elapsed=" << dt_all << " s, ETA=" << eta << " s)\n";
        }
    }
    auto t2 = std::chrono::steady_clock::now();

    // Optional auto-detect of orbital cutoff radius.  Scan from the
    // outermost ir backwards: find the smallest ir s.t. for all r > ir,
    // |chi(j,lambda)(r)| = |r * psi_lm(...)| < threshold for every
    // orbital + every (l,m) channel we kept.  A 5-bohr buffer is added so
    // the scattering boundary fit isn't biased by an abrupt zeroing.
    if (orb_rmax_auto_thresh > 0.0) {
        const double thresh = orb_rmax_auto_thresh;
        int last_active_ir = 0;
        for (int ir = 0; ir < Nr_orb_store; ++ir) {
            const double r = rg.r[ir];
            for (int j = 0; j < n_sce_orb; ++j) {
                const Eigen::Index row0 = Eigen::Index(j) * Nlm_orb_store;
                for (int lam = 0; lam < Nlm_orb_store; ++lam) {
                    if (std::abs(r * Fpsi(row0 + lam, ir)) > thresh) {
                        last_active_ir = ir;
                    }
                }
            }
        }
        const int buffer_pts = static_cast<int>(std::round(5.0 / dr)) + 1;
        const int new_Nr = std::min(
            Nr_orb_store,
            std::max(last_active_ir + buffer_pts + 1, 100));
        if (new_Nr < Nr_orb_store) {
            std::cerr << "[preprocess] --orb-rmax-auto " << thresh
                      << ": tightening orbital Nr_store from "
                      << Nr_orb_store << " (r=" << rg.r[Nr_orb_store - 1]
                      << ") to " << new_Nr << " (r=" << rg.r[new_Nr - 1]
                      << ")  -- saves "
                      << std::fixed << std::setprecision(1)
                      << (100.0 * (Nr_orb_store - new_Nr) / Nr_orb_store)
                      << "% of orbital storage\n";
            // In-place tile shrink: copy each orbital's first new_Nr
            // columns down into a contiguous buffer, then resize.
            Eigen::MatrixXd Fpsi_tight(
                Eigen::Index(n_sce_orb) * Nlm_orb_store, new_Nr);
            for (int jj = 0; jj < n_sce_orb; ++jj) {
                Fpsi_tight.middleRows(
                    Eigen::Index(jj) * Nlm_orb_store, Nlm_orb_store) =
                    Fpsi.block(Eigen::Index(jj) * Nlm_orb_store,
                               0, Nlm_orb_store, new_Nr);
            }
            Fpsi   = std::move(Fpsi_tight);
            Nr_orb_store = new_Nr;
        }
    }

    // -------- 4) potentials --------
    std::cerr << "[preprocess] building V_en (nuclear attraction, SCE-projected) ...\n";
    Eigen::MatrixXd V_en = preproc::potential::build_V_en(mol.atoms, origin, rg, Lmax);
    auto t3 = std::chrono::steady_clock::now();
    std::cerr << "[preprocess] V_en done in "
              << std::chrono::duration<double>(t3 - t2).count() << " s\n";

    std::cerr << "[preprocess] building V_H (Hartree, radial ODEs over Nlm channels) ...\n";
    Eigen::MatrixXd V_H  = preproc::potential::build_V_H(Frho, rg, Lmax);
    auto t4 = std::chrono::steady_clock::now();
    std::cerr << "[preprocess] V_H done in "
              << std::chrono::duration<double>(t4 - t3).count() << " s\n";

    Eigen::MatrixXd V_x;
    if (exchange == "slater") {
        std::cerr << "[preprocess] building V_x (local exchange, Slater 1.0) ...\n";
        auto F_vx = preproc::potential::make_V_x_local(F_rho, preproc::potential::LDAAlpha::Slater_1_0);
        V_x = preproc::sce::project(rg, ag, origin, F_vx, false);
    } else if (exchange == "lda") {
        std::cerr << "[preprocess] building V_x (local exchange, Kohn-Sham 2/3) ...\n";
        auto F_vx = preproc::potential::make_V_x_local(F_rho, preproc::potential::LDAAlpha::KohnSham_2_3);
        V_x = preproc::sce::project(rg, ag, origin, F_vx, false);
    } else {
        std::cerr << "[preprocess] V_x skipped (exchange=none; scattering code handles "
                     "exchange via ExchangeCoupling)\n";
        V_x = Eigen::MatrixXd::Zero(Nlm, Nr);
    }
    auto t5 = std::chrono::steady_clock::now();

    Eigen::MatrixXd V_total = V_en + V_H + V_x;

    // -------- 5) sanity checks --------
    const double N_e_computed = preproc::sce::integrate_total(Frho, rg);
    const double E_Vne        = preproc::potential::inner_product_radial(Frho, V_en, rg);
    const double J            = 0.5 * preproc::potential::inner_product_radial(Frho, V_H, rg);
    // Sum ||psi||^2 over the SCEd orbitals; each unit-normalized MO
    // contributes 1, so the total should equal n_sce_orb.
    //
    // NOTE: this norm is taken over the TRUNCATED tile actually written
    // to disk -- (Lmax_orb_store+1)^2 channels and Nr_orb_store radial
    // points.  When the user passed --orb-lmax / --orb-rmax(-auto), the
    // sum can be slightly less than n_sce_orb because we deliberately
    // dropped high-l / large-r content that's below precision.  The
    // per-orbital ||ψ||^2 printed earlier in the SCE loop is the
    // un-truncated norm and is the right gate for "is the SCE
    // projection good enough".
    double orb_norm_sum = 0.0;
    auto rg_orb = preproc::sce::RadialGrid::build(0.0, dr, Nr_orb_store);
    for (int jj = 0; jj < n_sce_orb; ++jj) {
        Eigen::MatrixXd block = Fpsi.block(
            Eigen::Index(jj) * Nlm_orb_store, 0,
            Nlm_orb_store, Nr_orb_store);
        orb_norm_sum += preproc::sce::norm_squared(block, rg_orb);
    }
    std::cerr << "[sanity] N_e (from SCE rho)   = " << N_e_computed
              << "  vs expected " << N_e_expected << "\n";
    std::cerr << "[sanity] sum ||psi_i||^2      = " << orb_norm_sum
              << "  vs n_sce_orb=" << n_sce_orb << "\n";
    std::cerr << "[sanity] <rho|V_en>           = " << E_Vne << "\n";
    std::cerr << "[sanity] J = 0.5<rho|V_H>     = " << J << "\n";

    // -------- 5b) input-validation gate --------
    // A density-consistency self-check.  CATCHES:
    //   * normalisation-convention bugs (a shell's contraction coeffs
    //     off by a factor -> orbital norm scaled away from 1)
    //   * shell-level AO mis-assignment (a whole shell's coeffs landing
    //     on the wrong shell -> norms wrong, N_e wrong)
    //   * missing / extra / mis-occupied orbitals
    //   * gross parse failures (N_e grossly wrong)
    // DOES NOT CATCH (mathematically invisible to a norm check):
    //   * WITHIN-shell component reordering (e.g. PySCF's -l..+l vs
    //     molden's 0,+1,-1 order) -- the (2l+1) components of one shell
    //     are mutually orthonormal, so ||psi||^2 = sum |c_mu|^2 is
    //     INVARIANT under any permutation of them.
    //   * Condon-Shortley sign flips of a component -- c_mu^2 unchanged.
    // Those two classes are prevented BY CONSTRUCTION by converting QC
    // inputs through iodata (qc_to_molden.py), which gets the ordering
    // and sign right for every supported format; the molden round-trip
    // is validated bit-tight by test_qc_to_molden_roundtrip.sh.  This
    // gate is the SECOND net for the error classes a norm CAN see.
    // Thresholds are loose enough to pass legitimate low-Lmax SCE
    // truncation (a core 1s at Lmax=16 has ||psi||^2 ~ 0.96).
    // Disable with --no-validate-input.
    if (validate_input) {
        std::cerr << "[validate] input-convention gate (norm in ["
                  << val_norm_lo << ", " << val_norm_hi << "], N_e within "
                  << (val_ne_tol * 100.0) << "%):\n";
        bool   failed = false;
        int    n_advisory = 0;
        double worst_norm_lo = 2.0, worst_norm_hi = 0.0;
        for (int jj = 0; jj < n_sce_orb; ++jj) {
            const double ns = orb_norm_sq[jj];
            worst_norm_lo = std::min(worst_norm_lo, ns);
            worst_norm_hi = std::max(worst_norm_hi, ns);
            if (ns < val_norm_lo || ns > val_norm_hi) {
                std::cerr << "[validate]   FAIL  orbital " << (jj + 1)
                          << " (molden idx=" << orb_molden_index[jj]
                          << ", occ=" << orb_occ[jj]
                          << ", eps=" << orb_energies[jj] << " Ha)"
                          << "  ||psi||^2=" << ns
                          << "  -- OUT OF BOUNDS [" << val_norm_lo
                          << ", " << val_norm_hi << "]\n";
                failed = true;
            } else if (ns < 0.99) {
                ++n_advisory;
            }
        }
        // Electron-count check (loose; gross-error detector).
        const double ne_rel = (N_e_expected > 0.0)
            ? std::abs(N_e_computed / N_e_expected - 1.0) : 0.0;
        if (ne_rel > val_ne_tol) {
            std::cerr << "[validate]   FAIL  N_e(SCE rho)=" << N_e_computed
                      << " vs occupations=" << N_e_expected
                      << "  (rel error " << (ne_rel * 100.0) << "% > tol "
                      << (val_ne_tol * 100.0) << "%)\n";
            failed = true;
        }
        if (failed) {
            std::cerr <<
                "[validate] ========================================================\n"
                "[validate] INPUT VALIDATION FAILED.  The SCE-projected density is\n"
                "[validate] inconsistent with the input orbital occupations.  This\n"
                "[validate] is the signature of a quantum-chemistry-format CONVENTION\n"
                "[validate] mismatch (wrong spherical-harmonic ordering, missing /\n"
                "[validate] double normalisation, or Condon-Shortley sign) -- almost\n"
                "[validate] certainly from the QC->molden conversion step.\n"
                "[validate]\n"
                "[validate] DO NOT trust scattering results from this preprocessing.\n"
                "[validate]\n"
                "[validate] What to do:\n"
                "[validate]   1. Re-run qc_to_molden.py with --verbose and check the\n"
                "[validate]      atom count / nbf / nelec match your QC calculation.\n"
                "[validate]   2. If the orbital norm is roughly DOUBLE or HALF the\n"
                "[validate]      expected, it is a normalisation-convention bug.\n"
                "[validate]   3. If only SOME orbitals fail, a whole shell's coeffs\n"
                "[validate]      were likely assigned to the wrong shell (shell-level\n"
                "[validate]      offset bug).  NOTE: within-shell component reordering\n"
                "[validate]      and sign flips are norm-invariant and are NOT caught\n"
                "[validate]      here -- prevent those by converting via qc_to_molden.py\n"
                "[validate]      (iodata) rather than a hand-written parser.\n"
                "[validate]   4. As a stopgap you can bypass with --no-validate-input,\n"
                "[validate]      but ONLY after you understand why the norms are off.\n"
                "[validate] ========================================================\n";
            std::cerr.flush();
            return 3;
        }
        std::cerr << "[validate]   PASS  (worst orbital norm in ["
                  << worst_norm_lo << ", " << worst_norm_hi << "]; N_e rel error "
                  << (ne_rel * 100.0) << "%)\n";
        if (n_advisory > 0) {
            std::cerr << "[validate]   ADVISORY: " << n_advisory
                      << " orbital(s) have ||psi||^2 < 0.99 (ePolyScat's "
                         "production criterion).  This is normal at low --lmax "
                         "(core cusps); for publication-quality results increase "
                         "--lmax until all occupied norms are >= 0.99.\n";
        }
    }

    // -------- 6) write HDF5 --------
    // Two output files: one for orbitals (the big, expensive psi_lm /
    // initial_state data), one for potentials + meta (small, recomputable
    // via repair_v_h).  Both files are independently self-describing: each
    // gets a copy of /grid, /angular, /geometry and the format string.
    preproc::io::H5File f_orb(orb_out);
    preproc::io::H5File f_pot(pot_out);

    // Helper: write the shared (small) descriptor groups into both files.
    auto write_shared_descriptors = [&](preproc::io::H5File& f) {
        f.write_string_attr("format", "static_exchangeHF_preprocessing/v2");
        f.write_string_attr("molden_source", molden_path);
        f.write_string_attr("exchange_kind", exchange);

        f.make_group("/grid");
        f.write_scalar_double("/grid/rmin", rg.rmin);
        f.write_scalar_double("/grid/dr",   rg.dr);
        f.write_scalar_int   ("/grid/N",    rg.N);
        f.write_1d_double    ("/grid/r",    rg.r.data(), rg.N);

        f.make_group("/angular");
        f.write_scalar_int("/angular/Lmax",   Lmax);
        f.write_scalar_int("/angular/nTheta", ag.nTheta);
        f.write_scalar_int("/angular/nPhi",   ag.nPhi);
    };
    write_shared_descriptors(f_orb);
    write_shared_descriptors(f_pot);

    // Channel-lm index (per-channel l,m pair).  Goes into both files.
    std::vector<int> lm_idx(2 * Nlm);
    for (int l = 0; l <= Lmax; ++l) for (int m = -l; m <= l; ++m) {
        int ch = lm_index(l, m);
        lm_idx[2 * ch + 0] = l;
        lm_idx[2 * ch + 1] = m;
    }
    f_orb.write_2d_int("/angular/channel_lm", lm_idx.data(), Nlm, 2);
    f_pot.write_2d_int("/angular/channel_lm", lm_idx.data(), Nlm, 2);

    // Geometry block: also goes into both files.
    const int Na = static_cast<int>(mol.atoms.size());
    std::vector<double> atoms_xyz(3 * Na);
    std::vector<int>    atoms_Z(Na);
    for (int i = 0; i < Na; ++i) {
        atoms_xyz[3 * i + 0] = mol.atoms[i].xyz[0];
        atoms_xyz[3 * i + 1] = mol.atoms[i].xyz[1];
        atoms_xyz[3 * i + 2] = mol.atoms[i].xyz[2];
        atoms_Z[i]           = mol.atoms[i].Z;
    }
    auto write_geometry = [&](preproc::io::H5File& f) {
        f.make_group("/geometry");
        f.write_2d_double("/geometry/xyz_bohr", atoms_xyz.data(), Na, 3, /*gzip*/ 0);
        f.write_1d_int   ("/geometry/Z",         atoms_Z.data(),   Na);
        f.write_1d_double("/geometry/origin_bohr", origin.data(), 3);
        f.write_1d_double("/geometry/translation_applied_bohr", chosen_center.data(), 3);
        f.write_string_attr("origin_choice", origin_choice);
    };
    write_geometry(f_orb);
    write_geometry(f_pot);

    // Eigen is column-major. Convert potential matrices to row-major (Nlm rows, Nr cols)
    // for stable cross-language access.
    auto write_matrix = [](preproc::io::H5File& f,
                           const std::string& path, const Eigen::MatrixXd& M) {
        // Row-major storage: shape (rows, cols).
        std::vector<double> rowmajor(static_cast<size_t>(M.rows()) * M.cols());
        for (Eigen::Index i = 0; i < M.rows(); ++i)
            for (Eigen::Index j = 0; j < M.cols(); ++j)
                rowmajor[static_cast<size_t>(i) * M.cols() + j] = M(i, j);
        f.write_2d_double(path, rowmajor.data(),
                          static_cast<hsize_t>(M.rows()),
                          static_cast<hsize_t>(M.cols()),
                          /*gzip*/ 4);
    };

    // ---- Potentials file: V_*, rho, meta ----
    f_pot.make_group("/potential");
    write_matrix(f_pot, "/potential/V_en",              V_en);
    write_matrix(f_pot, "/potential/V_H",               V_H);
    write_matrix(f_pot, "/potential/V_local_exchange",  V_x);
    write_matrix(f_pot, "/potential/V_total_local",     V_total);

    f_pot.make_group("/rho");
    write_matrix(f_pot, "/rho/rho_lm", Frho);

    f_pot.make_group("/meta");
    f_pot.write_scalar_double("/meta/N_e_computed",      N_e_computed);
    f_pot.write_scalar_double("/meta/N_e_expected",      N_e_expected);
    f_pot.write_scalar_double("/meta/E_nuclear_electron", E_Vne);
    f_pot.write_scalar_double("/meta/E_coulomb_J",        J);
    f_pot.write_scalar_double("/meta/sum_orb_norm_sq",    orb_norm_sum);

    // ---- Orbitals file: psi_lm + per-orbital metadata ----
    f_orb.make_group("/orbitals");
    f_orb.write_scalar_int("/orbitals/n_alpha",     n_alpha);
    f_orb.write_scalar_int("/orbitals/n_occ_alpha", n_occ_alpha);
    f_orb.write_scalar_int("/orbitals/n_sce",       n_sce_orb);
    // psi_lm shape is (n_sce_orb * Nlm_orb_store, Nr_orb_store) -- both
    // dimensions can be SMALLER than (Nlm_sce, Nr) when --orb-lmax /
    // --orb-rmax(-auto) were used.  Record both so the downstream loader
    // (HDF5Reader / WavefunctionSetup) can validate that any requested
    // n_lambda_cut and n_transition stay within the stored data.  When
    // these scalars are absent (loading a file from older
    // preprocess_molden), the loader assumes full Nlm_sce x Nr.
    f_orb.write_scalar_int("/orbitals/Lmax_orb_store",   Lmax_orb_store);
    f_orb.write_scalar_int("/orbitals/Nlm_orb_store",    Nlm_orb_store);
    f_orb.write_scalar_int("/orbitals/Nr_orb_store",     Nr_orb_store);
    f_orb.write_scalar_double("/orbitals/r_max_orb_store", rg.r[Nr_orb_store - 1]);
    f_orb.write_string_attr("orbital_selection", orbital_set);
    f_orb.write_1d_double("/orbitals/energies_hartree", orb_energies.data(), n_sce_orb);
    f_orb.write_1d_double("/orbitals/occupations",      orb_occ.data(),      n_sce_orb);
    f_orb.write_1d_int   ("/orbitals/molden_index",     orb_molden_index.data(), n_sce_orb);
    write_matrix(f_orb, "/orbitals/psi_lm", Fpsi);

    // ------------------------------------------------------------------
    // Optional: initial state for downstream dipole matrix elements.
    // The SOMO (highest-energy alpha MO with occ > 0) of a separate anion
    // molden file -- typically computed in THF solvent -- is projected
    // on the SAME (radial, angular) grid as the neutral static potential.
    // The anion geometry is assumed to coincide with the neutral's (vertical
    // attachment); we translate by the same `chosen_center` vector so the
    // resulting SCE-coordinate atoms match exactly.
    // ------------------------------------------------------------------
    if (!initial_state_molden.empty()) {
        std::cerr << "[initial-state] reading " << initial_state_molden << "\n";
        preproc::molden::MoldenParser P_ini(false);
        auto mol_ini = P_ini.parse(initial_state_molden);
        std::cerr << "[initial-state] atoms=" << mol_ini.atoms.size()
                  << "  alpha_MOs=" << mol_ini.mos_alpha.size()
                  << "  beta_MOs="  << mol_ini.mos_beta.size()
                  << "  nbf=" << mol_ini.nbf << "\n";
        // Translate the anion's atoms by the SAME vector we used for neutral.
        for (auto& a : mol_ini.atoms) a.xyz -= chosen_center;
        // Verify: the translated atom positions should match the neutral to
        // machine epsilon. Mismatch here is a geometry inconsistency (different
        // conformer, different reorientation etc.) and will silently corrupt
        // downstream dipole matrix elements if ignored.
        if (mol_ini.atoms.size() == mol.atoms.size()) {
            double max_disp = 0.0;
            for (size_t i = 0; i < mol.atoms.size(); ++i) {
                const double d = (mol_ini.atoms[i].xyz - mol.atoms[i].xyz).norm();
                max_disp = std::max(max_disp, d);
            }
            std::cerr << "[initial-state] max atom-displacement vs neutral = "
                      << max_disp << " Bohr\n";
            if (max_disp > 1e-6) {
                std::cerr << "[initial-state][WARN] anion geometry differs from neutral by "
                          << max_disp << " Bohr; proceeding but dipole matrix elements\n"
                          << "                      may not correspond to vertical attachment.\n";
            }
        } else {
            throw std::runtime_error("initial-state: anion has different number of atoms than neutral");
        }
        auto basis_ini = preproc::basis::build_basis(mol_ini, false);

        // Locate the anion SOMO: the highest-energy alpha MO with occupation > 0.
        // For RHF (neutral), this is the ordinary HOMO; for UHF (doublet anion),
        // it is the alpha orbital carrying the extra electron.
        const preproc::molden::MO* homo_ini = nullptr;
        int homo_ini_idx = -1;
        for (int j = 0; j < static_cast<int>(mol_ini.mos_alpha.size()); ++j) {
            const auto& m = mol_ini.mos_alpha[j];
            if (m.occ <= 0.0) continue;
            if (!homo_ini || m.energy > homo_ini->energy) {
                homo_ini = &m; homo_ini_idx = j;
            }
        }
        if (!homo_ini) throw std::runtime_error("initial-state: no occupied alpha MO found");
        std::cerr << "[initial-state] anion SOMO: alpha MO index " << homo_ini_idx
                  << "  energy = " << homo_ini->energy << " Ha"
                  << "  occ = "    << homo_ini->occ << "\n";

        // SCE the anion SOMO on the same grid.
        preproc::sce::F3D F_ini = [&](const Eigen::Vector3d& r) {
            return preproc::basis::evaluate_mo(basis_ini, *homo_ini, r);
        };
        auto tis0 = std::chrono::steady_clock::now();
        Eigen::MatrixXd Fini_full = preproc::sce::project(rg, ag, origin, F_ini, false);
        // Truncate to the same (Lmax_orb_store, Nr_orb_store) as the
        // bound-state orbitals so the downstream loader treats them
        // identically.  The norm is computed BEFORE truncation, on the
        // full Lmax projection, so the printed value is the true SCE
        // accuracy (not a function of the storage cut).
        Eigen::MatrixXd Fini = Fini_full.topLeftCorner(Nlm_orb_store, Nr_orb_store);
        auto tis1 = std::chrono::steady_clock::now();
        const double ini_norm = preproc::sce::norm_squared(Fini_full, rg);
        std::cerr << "[initial-state] ||psi_ini||^2 (on SCE grid) = " << ini_norm
                  << "  (should be 1)  t_sce=" << std::chrono::duration<double>(tis1 - tis0).count()
                  << " s\n";

        // Store in the orbitals file (initial-state ψ is logically an
        // orbital and lives next to the other ψ_lm data).
        f_orb.make_group("/initial_state");
        write_matrix(f_orb, "/initial_state/psi_lm", Fini);
        f_orb.write_scalar_double("/initial_state/energy_hartree",  homo_ini->energy);
        f_orb.write_scalar_double("/initial_state/occupation",      homo_ini->occ);
        f_orb.write_scalar_int   ("/initial_state/molden_index",    homo_ini_idx);
        f_orb.write_scalar_int   ("/initial_state/n_alpha_anion",   static_cast<long long>(mol_ini.mos_alpha.size()));
        f_orb.write_scalar_int   ("/initial_state/n_beta_anion",    static_cast<long long>(mol_ini.mos_beta.size()));
        f_orb.write_scalar_double("/initial_state/norm_on_sce_grid", ini_norm);
        f_orb.write_string_attr("initial_state_source_molden", initial_state_molden);
    }

    // Polarizability tensor (if provided), stored in a.u. as a 3x3 matrix.
    // Goes in the potentials file -- it parameterises V_pol in scattering.
    if (polarizability.size() == 9) {
        f_pot.make_group("/polarizability");
        f_pot.write_2d_double("/polarizability/alpha_tensor",
                              polarizability.data(), 3, 3, /*gzip*/ 0);
        const double iso = (polarizability[0] + polarizability[4] + polarizability[8]) / 3.0;
        f_pot.write_scalar_double("/polarizability/alpha_iso_au", iso);
        std::cerr << "[preprocess] wrote polarizability tensor, alpha_iso = "
                  << iso << " a.u.\n";
    }

    const double dt_sce_rho  = std::chrono::duration<double>(t1 - t0).count();
    const double dt_sce_psi  = std::chrono::duration<double>(t2 - t1).count();
    const double dt_Ven      = std::chrono::duration<double>(t3 - t2).count();
    const double dt_VH       = std::chrono::duration<double>(t4 - t3).count();
    const double dt_Vx       = std::chrono::duration<double>(t5 - t4).count();
    std::cerr << "[timing] SCE rho = " << dt_sce_rho << " s   "
              << "SCE orbitals = " << dt_sce_psi << " s   "
              << "V_en = " << dt_Ven << " s   "
              << "V_H = "  << dt_VH  << " s   "
              << "V_x = "  << dt_Vx  << " s\n";

    std::cerr << "[preprocess] wrote " << orb_out << "\n";
    std::cerr << "[preprocess] wrote " << pot_out << "\n";
    return 0;
}
