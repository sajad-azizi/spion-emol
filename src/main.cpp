// main.cpp -- full scattering pipeline driver.
//
// CLI:
//   scattering <preproc.h5> [ik_min] [ik_max] [dk] [flags...]
//   scattering --help
//
// Defaults: ik_min=1, ik_max=1 (INCLUSIVE, so a single energy), dk=0.01.
// The internal EnergyGrid is half-open; ik_max on the CLI is inclusive and
// we add 1 before handing it in, so "ik_min=ik_max" -> exactly one point.
//
// Directory layout (agreed 2026-04-23):
//
//   $WORK    (persistent, survives jobs)
//       pot_<molhash>/                 # energy-independent, reused across ik
//       dipole_<molhash>_<scan_id>/    # scan output (DipoleWriter)
//
//   $SCRATCH (short-lived, cleaned between jobs)
//       sinv_<molhash>_ik<nnnn>/       # deleted after this ik is written
//       rinv_<molhash>_ik<nnnn>/       # deleted after this ik is written
//       psi_<molhash>_ik<nnnn>/        # kept for possible c-c dipole post-proc
//
// The storage mode for {pot, sinv, rinv, psi} is chosen at runtime by
// StoragePlanner based on detected RAM (see scatt/StoragePlanner.hpp).

#include "angular/Gaunt.hpp"
#include "io/HDF5Reader.hpp"
#include "scatt/AsymptoticAmplitudes.hpp"
#include "scatt/BackPropagator.hpp"
#include "scatt/Banner.hpp"
#include "scatt/Bench.hpp"
#include "scatt/DipoleIO.hpp"
#include "scatt/DipoleMatrixElement.hpp"
#include "scatt/EnergyGrid.hpp"
#include "scatt/ExchangeCoupling.hpp"
#include "scatt/ForwardRPropagator.hpp"
#include "scatt/KMatrixExtractor.hpp"
#include "scatt/MoleculeHash.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"
#include "scatt/SchurInverter.hpp"
#include "scatt/StoragePlanner.hpp"
#include "scatt/SystemMemory.hpp"
#include "scatt/WInverseOperator.hpp"
#include "scatt/WavefunctionSetup.hpp"

#include <Eigen/Core>

#if defined(SCATT_HAS_OPENMP)
#  include <omp.h>
#endif

#if defined(SCATT_HAS_MKL)
#  include <mkl.h>
#  include <mkl_service.h>
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>     // setvbuf, _IOLBF, _IONBF, printf
#include <thread>

#include <gsl/gsl_errno.h>

namespace fs = std::filesystem;
using namespace scatt;


 
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

// ---------------------------------------------------------------------------
// Forward declarations (definitions below main).
// ---------------------------------------------------------------------------
struct CLI {
    std::string hdf5_input;
    int    ik_min = 1;
    int    ik_max_inclusive = 1;
    double dk     = 0.01;
    int    l_max_continuum = 10;
    std::optional<double>  dr_override;
    std::optional<double>  r_max_override;
    std::string  work_dir, scratch_dir;
    std::size_t  memory_budget = 0;       // 0 = auto
    std::string  scan_id_override;
    int          omp_threads = 0;         // 0 = leave OMP default
    std::string  bench_out;               // empty = auto under dipole_dir
    bool         use_gpu = false;         // SYCL / oneMKL offload of FRP+BP steps
    // Allow PotentialStorage to RE-CHUNK on-disk checkpoints at load time
    // when their on-disk chunk_size > the runtime budget chunk_size.
    // Default ON: cross-node-size moves (different RAM → different planner
    // output) become a ~minutes disk-I/O re-layout instead of a multi-hour
    // compute rebuild.  Opt out with --no-checkpoint-rechunk if your
    // scratch quota can't absorb the transient 2x checkpoint size.  See
    // PotentialStorage::set_chunk_rechunk_allowed for the safety contract.
    bool         allow_checkpoint_rechunk = true;
    bool         check_residual = false;  // run a per-energy PDE residual probe
    int          residual_probes = 8;     // number of (evenly-spaced) ir samples
    // Opt-in: chunk-blocked OpenMP parallelism for the DISK-mode
    // SchurInverter.  Default OFF preserves the historic serial-DISK
    // behaviour exactly.  See SchurInverter::Config::parallel_disk_chunks
    // and the bit-identical tests test_sinv_parallel_disk_path.cpp +
    // test_sinv_serial_vs_parallel.cpp.  Memory: adds ~chunk_size ·
    // matrix_bytes per stage on top of the planner reservation; at
    // l_cont=100 you may need to cap OMP_NUM_THREADS to ~80.
    bool         parallel_disk_sinv = false; //keep it always *false*

    // Opt-in: store ONLY the lower triangle of pot/sinv/rinv on disk.
    // Halves disk space and disk I/O on the three symmetric stages
    // (psi remains full -- it's not symmetric).  ZERO accuracy loss:
    // pot/sinv/rinv are bit-symmetric by construction, so the upper
    // triangle is byte-equal to the transpose of the lower; reading
    // back reconstructs the full matrix exactly.  Validated by
    // test_storage_symmetric and test_storage_symmetric_end_to_end.
    // Backward-compatible at load time: pre-existing v1 (full-matrix)
    // checkpoints still read correctly under the new code regardless
    // of this flag.  Default false preserves the legacy on-disk format
    // byte-for-byte.
    bool         symmetric_storage = true;

    // Opt-in: parallel chunk writer for pot/sinv/rinv DISK output.
    // Multi-threaded pwrite-at-distinct-offsets with atomic temp +
    // rename + fsync.  ZERO accuracy loss: bytes on disk are byte-
    // identical to the serial path (validated by
    // test_storage_parallel_write).  Default false preserves the
    // legacy serial single-threaded write byte-for-byte.
    bool         parallel_chunk_write = true;

    // Opt-in: closed-channel substitution in AsymptoticAmplitudes.
    // For ℓ values where the asymptotic-fit 2×2 normal matrix is
    // catastrophically singular (k·r_max ≲ ℓ -- the channel is
    // exponentially evanescent across the entire fit window) the
    // default behaviour is to ABORT with a clear diagnostic so the
    // user can either extend r_max or reduce l_cont.
    //
    // Setting this flag substitutes A_{μν}=δ_{μν}, B_{μν}=0 for the
    // affected channels (their K-matrix block is then 0).  Result is
    // equivalent to running with a smaller l_cont within the closed
    // ℓ values; safe IFF the photoionization observable is converged
    // in l_cont within the open-ℓ subset.  The user must justify this
    // independently (e.g. by comparing to a converged l_cont=L_safe
    // run where L_safe < k·r_max).  Default FALSE.
    bool         allow_closed_channels = false;

    // Recovery / re-postprocessing mode.  When set, SKIP the per-ik
    // Sinv build, FRP run, K extraction, and BP backward propagation.
    // Load psi (and the asym buffer) directly from the existing
    // $SCRATCH/psi_<hash>_ik<NNNN>/ checkpoint, run AsymptoticAmplitudes
    // to re-fit (A, B) from the loaded psi, and run DipoleMatrixElement
    // to recompute the dipoles with the CURRENT initial state
    // (/initial_state/ if present, target HOMO otherwise).
    //
    // Intended use: any old runs that wrote psi to disk but whose
    // dipole values were computed under the pre-fix bug (always-target-
    // HOMO initial state).  At L=100 PVC this drops the per-ik
    // recompute cost from ~42 h to ~minutes (just dipole + A/B fit).
    //
    // Hard requirements:
    //   * $SCRATCH/psi_<hash>_ik<NNNN>/ must contain finalized chunk
    //     files + __SUCCESS__ marker (BP completed cleanly).
    //   * psi storage manifest must match (Nr, dr, l_cont, N_psi, etc.).
    //   * If using the asym buffer (n_asym > 0 in BP::Config -- always
    //     true in production), $SCRATCH/.../asym/__SUCCESS__ must also
    //     exist.  AsymptoticAmplitudes reads from this window.
    // On a missing/mismatched checkpoint the code throws a clear error
    // -- it does NOT silently fall through to a rebuild.
    bool         dipole_only = false;
};

static void usage(const char* prog);
static CLI  parse_cli(int argc, char** argv);
static std::string env_or(const char* key, const std::string& fallback);
static std::string iso_date_utc_now();

// One-energy body. Writes dipole payload into `writer`.
static void run_one_energy(int                       ik,
                           double                    E_kin,
                           const io::PreprocData&    data,
                           Parameters&               params,
                           Potentials&               pot,
                           const StoragePlan&        splan,
                           const std::string&        sinv_dir,
                           const std::string&        rinv_dir,
                           const std::string&        psi_dir,
                           const Eigen::MatrixXd&    chi_init_homo,
                           const std::vector<OccupiedOrbital>& occ,
                           DipoleWriter&             writer,
                           BenchReport&              bench,
                           bool                      use_gpu,
                           bool                      check_residual,
                           int                       residual_probes,
                           bool                      parallel_disk_sinv,
                           bool                      symmetric_storage,
                           bool                      parallel_chunk_write,
                           StorageMode               psi_asym_mode,
                           int                       psi_asym_chunk_size,
                           bool                      dipole_only,
                           bool                      allow_closed_channels,
                           bool                      allow_checkpoint_rechunk);

// ===========================================================================
// int main
// ===========================================================================
int main(int argc, char** argv) try {
    apply_line_buffering();

    // Enable Level Zero Sysman BEFORE any Level Zero / SYCL init occurs.
    // This is what lets BenchReport read PVC energy counters via
    // zesPowerGetEnergyCounter().  Without it, zesDriverGet returns 0
    // drivers and the GPU energy column in the bench printout shows
    // "L0 n/a".  Setting it after the first SYCL device acquisition has
    // no effect -- so this MUST be the first thing in main.  No-op when
    // SCATT_HAS_SYCL is off (the CPU-only build doesn't query Sysman).
    ::setenv("ZES_ENABLE_SYSMAN", "1", /*overwrite=*/0);

    // GSL by default calls abort() on errors (including GSL_EUNDRFLW).
    // At l_cont >= ~80, j_l(k r) and gamma normalisations underflow during
    // FRP analytic-init / KMatrix asymptotic fits; the *_e variants in
    // the call sites already tolerate GSL_EUNDRFLW, but only if the
    // global handler doesn't abort first. Disable it once at startup.
    gsl_set_error_handler_off();

    CLI cli = parse_cli(argc, argv);

    // Apply --omp-threads before anything uses OpenMP.
#if defined(SCATT_HAS_OPENMP)
    if (cli.omp_threads > 0) omp_set_num_threads(cli.omp_threads);
#endif

    // MKL threading policy:
    //   * mkl_set_num_threads(N): cap MKL's internal thread pool.
    //   * mkl_set_dynamic(1): when MKL is called from INSIDE an OpenMP
    //     parallel region, it auto-detects nesting and drops to 1 thread
    //     per caller -- preserving our outer-loop parallelism in
    //     SchurInverter and Potentials without oversubscribing.
    //   * When MKL is called from a SERIAL section (FRP::run, BP::run,
    //     DipoleMatrixElement), it uses all N threads for the inner LU /
    //     triangular solve.  This is the hot path -- exactly where we
    //     want multi-threading.
#if defined(SCATT_HAS_MKL)
    const int _mkl_n = (cli.omp_threads > 0) ? cli.omp_threads
#if defined(SCATT_HAS_OPENMP)
                                             : omp_get_max_threads();
#else
                                             : 1;
#endif
    mkl_set_num_threads(_mkl_n);
    mkl_set_dynamic(1);
#endif

    // ---- Construct BenchReport FIRST (before any L0/SYCL touchpoint) ----
    // BenchReport's ctor calls gpu_energy_uj_now() which routes through
    // zesInit().  On modern PVC drivers Sysman requires zesInit() BEFORE
    // any zeInit() / SYCL device probe; the ZES_ENABLE_SYSMAN env-var
    // legacy path no longer works on Intel oneAPI ≥ 2024.x compute-runtime
    // builds.  print_la_banner() below initialises L0 via SYCL device
    // introspection, so we must reach BenchReport's ctor first.  This
    // ordering is the difference between "gpu_energy(L0)=unavailable"
    // and a working J/stage column in the bench printout.
    BenchReport bench;

    // ---- Backend banner ----
    // Single source of truth for "which LA/GPU/HDF5 are we actually running
    // with". Printed before any heavy work so every log captures it.
    scatt::print_la_banner();
    if (cli.use_gpu) {
        std::cout << "[banner] GPU offload REQUESTED (--use-gpu) -- FRP + BP "
                     "per-step work will run on the above device.\n";
    }

    // ---- Threading diagnosis ----
    // Explicit Eigen thread set + report what's actually active at run time.
    // If EIGEN_HAS_OPENMP is NOT defined, Eigen dense LA (GEMM, LU, LLT) is
    // SERIAL regardless of what we tell it. That's the case on macOS with
    // a header-only Eigen install and no MKL. On LRZ with MKL, Eigen's BLAS
    // calls are routed through MKL, which parallelizes automatically via
    // MKL_NUM_THREADS / OMP_NUM_THREADS.
    {
#if defined(SCATT_HAS_OPENMP)
        const int omp_max = omp_get_max_threads();
        if (cli.omp_threads > 0) Eigen::setNbThreads(cli.omp_threads);
        else                     Eigen::setNbThreads(omp_max);
#else
        const int omp_max = 1;
#endif
        std::cout << "[threading]"
                  << "  omp_get_max_threads = " << omp_max
                  << "  Eigen::nbThreads = "    << Eigen::nbThreads()
#ifdef EIGEN_HAS_OPENMP
                  << "  EIGEN_HAS_OPENMP = yes"
#else
                  << "  EIGEN_HAS_OPENMP = NO (dense BLAS will be serial)"
#endif
#ifdef SCATT_HAS_MKL
                  << "  MKL = yes"
#else
                  << "  MKL = no"
#endif
                  << "\n";
    }

    const auto t_total_start = std::chrono::steady_clock::now();


    //cout and cerr shoud be in secintific notation with 10 digits after the decimal point, to make it easier to read logs with large/small numbers and to compare with other tools.
    std::cout << std::scientific << std::setprecision(10);
    std::cerr << std::scientific << std::setprecision(10);

    // -------- Parameters + HDF5 --------
    Parameters params;
    params.l_max_continuum = cli.l_max_continuum;
    params.hdf5_input_path = cli.hdf5_input;

    // Staged loading.  Memory peak goes from ~810 GB (whole-file load) to
    // ~35 GB on a C8F8 / Lmax_sce=300 case by:
    //   1. load_header()           -- small metadata only (no V_H, no psi_lm)
    //   2. load_V_H(n_exp)         -- subset for Potentials::build
    //   3. (Potentials::build)     -- runs on data.V_H
    //   4. data.V_H.resize(0,0)    -- frees V_H now that pot matrix is built
    //   5. read_psi_lm_truncated() -- ONLY the (n_occ * n_lambda_cut, Nr_store)
    //                                 rows actually used by chi/exchange,
    //                                 via HDF5 hyperslab.  At LOAD time we
    //                                 cut angular -- so one preprocessing
    //                                 file works for any l_cont without
    //                                 re-running the 7-hour SCE projection.
    //   6. (chi build)             -- runs on data.psi_lm
    //   7. data.psi_lm.resize(0,0) -- frees orbitals once chi is in bundle
    //
    // The HDF5Reader stays alive across all stages so we can do targeted
    // reads at any point.
    std::cout << "[scattering] reading " << cli.hdf5_input << "\n";
    io::HDF5Reader reader(cli.hdf5_input);
    io::PreprocData data = reader.load_header();    // (1) small metadata

    params.r_min    = data.rmin;
    params.dr       = cli.dr_override.value_or(data.dr);
    params.Lmax_sce = data.Lmax_sce;
    params.N_grid   = data.Nr;
    if (cli.r_max_override) {
        params.N_grid = static_cast<std::size_t>(
            std::round(*cli.r_max_override / params.dr)) + 1;
    }
    data.V_H = reader.load_V_H(/*V_H_max_rows=*/params.n_exp());   // (2)
    params.validate();

    std::cout << "[scattering] grid  rmin=" << params.r_min
              << "  dr=" << params.dr
              << "  Nr=" << params.N_grid
              << "  r_max=" << params.r_max()
              << "  l_cont=" << params.l_max_continuum
              << "  channels=" << params.channels() << "\n";

    // -------- Energy grid --------
    EnergyGrid kgrid;
    kgrid.dk     = cli.dk;
    kgrid.ik_min = cli.ik_min;
    kgrid.ik_max = cli.ik_max_inclusive + 1;   // inclusive CLI -> half-open internal
    kgrid.validate();
    std::cout << "[scattering] energy grid  dk=" << kgrid.dk
              << "  ik=[" << kgrid.ik_min << ".." << (kgrid.ik_max - 1)
              << "] (" << kgrid.size() << " points)\n";

    // -------- Suggested-parameters note --------
    // Semiclassical estimates that bound the trustworthy energy window for
    // a fixed (l_cont, r_max) basis:
    //   * channel-truncation upper bound :  l_cont >= 1.2 * k * R_mol
    //   * asymptotic-fit lower bound     :  fit window >= 0.3 * (2π/k)
    //
    // R_mol is the chi-cutoff radius (where occupied orbitals decay below
    // chi_cutoff): R_mol ≈ Nr_orb_store · dr.  If preprocessing didn't
    // record it, fall back to the box "natural" 30 a_0.
    //
    // The fit window in AsymptoticAmplitudes scales with the grid (last
    // few percent of points), so we use Δr_fit ≈ 0.02 · r_max as a
    // first-order estimate.  Treat the note as a suggestion, not a hard
    // requirement — actual fit residuals are reported per-energy by the
    // AsymptoticAmplitudes diagnostics below.
    {
        const double k_lo  = kgrid.k(kgrid.ik_min);
        const double k_hi  = kgrid.k(kgrid.ik_max - 1);
        const double R_mol = (data.Nr_orb_store > 0
                              ? static_cast<double>(data.Nr_orb_store)
                              : 30.0 / params.dr) * params.dr;   // a_0
        const double r_max_now   = params.r_max();
        const int    l_cont_now  = params.l_max_continuum;

        // Recommended l_cont from the highest-k point in the scan.
        const int l_cont_rec = std::max(10,
            static_cast<int>(std::ceil(1.2 * k_hi * R_mol)));
        // Recommended r_max from the lowest-k point: target alpha=0.5
        // wavelengths in the asymptotic-fit window (~2% of r_max).
        // Empirical calibration: alpha = 0.39 at the C8F8 anchor (k=1.2,
        // r_max=100) gives rel fit_residual ≈ 1e-4 -- already good.
        constexpr double ALPHA_REC = 0.5;
        const double r_max_rec = (k_lo > 0.0)
            ? std::max(100.0, ALPHA_REC * 100.0 * M_PI / k_lo)
            : r_max_now;

        // Diagnostics for the user's CURRENT (l_cont, r_max).
        const double k_max_supported_by_lcont = static_cast<double>(l_cont_now)
                                              / (1.2 * R_mol);
        const double E_max_supported_by_lcont = 0.5 * k_max_supported_by_lcont
                                              * k_max_supported_by_lcont;
        const double k_min_supported_by_rmax  = (r_max_now > 0.0)
            ? (2.0 * M_PI) / (0.5 * 0.02 * r_max_now)   // 0.5 λ in fit window
            : 0.0;
        const double E_min_supported_by_rmax  = 0.5 * k_min_supported_by_rmax
                                              * k_min_supported_by_rmax;

        std::cout << "\n[scattering] === SUGGESTED-PARAMETERS NOTE ===\n";
        std::cout << "  Scan   k = " << k_lo << " .. " << k_hi
                  << "   (E = " << (0.5 * k_lo * k_lo)
                  << " .. " << (0.5 * k_hi * k_hi) << " Ha)\n";
        std::cout << "  R_mol  = " << R_mol << " a_0   (chi cutoff radius)\n";
        std::cout << "  current:  l_cont=" << l_cont_now
                  << "   r_max=" << r_max_now << " a_0\n";
        std::cout << "  Trust window for current (l_cont, r_max):\n"
                  << "    E_min ≳ " << E_min_supported_by_rmax << " Ha"
                  << "  (k ≳ " << k_min_supported_by_rmax << ")  "
                  << "[set by r_max]\n"
                  << "    E_max ≲ " << E_max_supported_by_lcont << " Ha"
                  << "  (k ≲ " << k_max_supported_by_lcont << ")  "
                  << "[set by l_cont]\n";
        std::cout << "  Suggested for the requested scan:\n"
                  << "    l_cont ≳ " << l_cont_rec
                  << "   r_max ≳ " << r_max_rec << " a_0\n"
                  << "    (given r_max=" << r_max_now
                  << " a_0, suggested l_cont = " << l_cont_rec << ")\n"
                  << "    (given l_cont=" << l_cont_now
                  << ", suggested r_max = " << r_max_rec << " a_0)\n";
        // Verdict: are we safe with the user's chosen (r_max, l_cont)?
        // ok_r accepts alpha >= 0.3 wavelengths in fit window (= the
        // empirical L=80 anchor's calibration); ok_l accepts the
        // semiclassical 1.2 * k * R_mol coverage.
        constexpr double ALPHA_MIN = 0.3;
        const double r_max_safe_min = (k_lo > 0.0)
            ? (ALPHA_MIN * 100.0 * M_PI / k_lo) : 0.0;
        const bool ok_l = (l_cont_now >= l_cont_rec);
        const bool ok_r = (r_max_now  >= r_max_safe_min);
        std::cout << "  >>> for given parameters (r_max=" << r_max_now
                  << " a_0, l_cont=" << l_cont_now << ") we are ";
        if (ok_l && ok_r) {
            std::cout << "SAFE for the requested scan.\n";
        } else {
            std::cout << "in DANGER:";
            if (!ok_l) std::cout << " l_cont under-converges high-k;";
            if (!ok_r) std::cout << " r_max too small for low-k asymptotic fit;";
            std::cout << " accuracy may be degraded.\n";
        }
        if (l_cont_now < l_cont_rec) {
            std::cout << "  WARNING: l_cont=" << l_cont_now
                      << " may UNDER-CONVERGE the highest-k point in this scan."
                      << "  Bump --l-cont (or use the cached recipe)\n"
                      << "           and verify by checking that Σ|D|^2 changes"
                      << " by < 1% when l_cont is increased.\n";
        }
        if (r_max_now < 0.5 * r_max_rec) {
            std::cout << "  WARNING: r_max=" << r_max_now
                      << " a_0 is SMALL for the lowest-k point in this scan."
                      << "  The asymptotic\n"
                      << "           A,B fit may degrade -- watch"
                      << " ‖A − I‖_∞ and fit_residual_max in the per-ik"
                      << " diagnostics.\n";
        }
        std::cout << "[scattering] === end note ===\n\n";
    }

    // -------- Memory plan --------
    // N_total is the joined Johnson-block dimension actually used by
    // ForwardRPropagator: N_total = N_psi + N_f = N_ch + n_occ * n_sigma.
    // Here n_sigma = (l_max_exchange + 1)^2 with the production rule
    // l_max_exchange = min(l_max_continuum, 10).  The earlier formula
    // (1 + n_occ) * N_ch is only correct in the special case
    // l_exch == l_cont (i.e. l_cont <= 10); for l_cont > 10 (e.g.
    // C8F8 at l_cont >= 20) it OVER-states rinv by ~(N_ch/n_sigma)^2,
    // which can reach 400x at l_cont=60.  That makes the planner pick
    // a chunk_size that's far too small for disk I/O and prints an
    // alarming (but bogus) multi-PB rinv footprint.
    const int n_ch          = params.channels();
    const int n_beta        = n_ch;
    const int n_occ_planner = data.n_occ_alpha;
    const int l_exch_rule   = std::min(params.l_max_continuum, 10);
    const int n_sigma       = (l_exch_rule + 1) * (l_exch_rule + 1);
    const int n_total       = n_ch + n_occ_planner * n_sigma;

    // Estimate the RAM consumed by everything OUTSIDE the four chunked
    // storages (pot, sinv, rinv, psi).  This is what's resident during BP
    // and dominates the planner's invisible overhead at L=100.
    //
    //   l_orb_max       = l_cont + l_exch        (rule l_exch = min(l_cont,10))
    //   n_lambda_cut    = (l_orb_max + 1)^2
    //   n_transition    ~ Nr_orb_store from preprocess (chi_cutoff truncation)
    //
    //   data.psi_lm        = n_orb * n_lambda_cut * n_transition * 8
    //   bundle.chi (per E) = n_transition * n_orb * n_lambda_cut * 8
    //   occ[j].phi (n_occ) = n_orb * Nr * n_lambda_cut * 8     <- biggest at L=100
    //   chi_init_homo      = Nr * n_lambda_cut * 8
    //   BP scratch (W^-1)  = n_total^2 * 8
    //   G/C sparse coeffs  ~ 1 GB (rough, scales with non-zero count)
    //
    // We over-estimate slightly (assume both ref_bundle and per-energy
    // bundle.chi are simultaneously live during the energy-loop block) so
    // the planner stays on the conservative side.
    const int l_orb_max_pl    = params.l_max_continuum + l_exch_rule;
    const int n_lambda_cut_pl = (l_orb_max_pl + 1) * (l_orb_max_pl + 1);
    const int n_orb_pl        = data.n_sce > 0 ? data.n_sce : 1;
    const int Nr_pl           = static_cast<int>(params.N_grid);
    const int n_trans_pl      = (data.Nr_orb_store > 0) ? data.Nr_orb_store : Nr_pl;
    const std::size_t bytes_psi_lm    = std::size_t(n_orb_pl) * n_lambda_cut_pl
                                      * n_trans_pl * 8ull;
    const std::size_t bytes_chi_bun   = std::size_t(n_trans_pl) * n_orb_pl
                                      * n_lambda_cut_pl * 8ull;
    const std::size_t bytes_occ_phi   = std::size_t(n_orb_pl) * Nr_pl
                                      * n_lambda_cut_pl * 8ull;
    const std::size_t bytes_chi_init  = std::size_t(Nr_pl) * n_lambda_cut_pl * 8ull;
    const std::size_t bytes_w_scratch = std::size_t(n_total) * n_total * 8ull;
    const std::size_t bytes_sparse_co = 1ull << 30;   // ~1 GB rough upper bound
    const std::size_t non_storage_bytes =
        bytes_psi_lm + bytes_chi_bun + bytes_occ_phi
      + bytes_chi_init + bytes_w_scratch + bytes_sparse_co;

    // -------- Directories (resolved before plan_storage so we can peek
    //          an existing pot checkpoint's on-disk chunk_size) ----------
    const std::string work    = !cli.work_dir.empty()    ? cli.work_dir
                              : env_or("WORK",    fs::current_path().string() + "/work");
    const std::string scratch = !cli.scratch_dir.empty() ? cli.scratch_dir
                              : env_or("SCRATCH", fs::current_path().string() + "/scratch");
    fs::create_directories(work);
    fs::create_directories(scratch);

    const std::string mhash = molecule_hash(data, params.l_max_continuum,
                                            params.dr, params.N_grid);
    const std::string scan_id = !cli.scan_id_override.empty()
        ? cli.scan_id_override
        : ("ik" + std::to_string(kgrid.ik_min) + "-"
                + std::to_string(kgrid.ik_max - 1)
                + "_dk" + std::to_string(kgrid.dk));
    const std::string pot_dir    = work + "/pot_" + mhash;
    const std::string dipole_dir = work + "/dipole_" + mhash + "_" + scan_id;

    // Cross-node-size protection: if a pot checkpoint already exists (built
    // on a possibly larger node), the on-disk chunk_size may be bigger than
    // anything this node's RAM can afford.  Peek it here and pin the
    // planner to the on-disk value so the budget for sinv/rinv/psi adapts.
    const int pinned_pot_cs = scatt::peek_checkpoint_chunk_size(pot_dir);
    if (pinned_pot_cs > 0) {
        std::cout << "[scattering] pot checkpoint at " << pot_dir
                  << " has on-disk chunk_size=" << pinned_pot_cs
                  << "  -> planner will pin pot to this value\n";
    }

    const StoragePlan splan = plan_storage(
        params.N_grid, n_ch, n_beta, params.N_grid,
        detect_total_ram_bytes(), cli.memory_budget,
        /*reserve_fraction=*/0.10,
        /*N_total=*/n_total,
        /*non_storage_bytes=*/non_storage_bytes,
        /*fixed_runtime_bytes=*/scatt::kDefaultRuntimeOverheadBytes,
        /*pinned_chunk_pot=*/pinned_pot_cs,
        /*prefetch_request_mask=*/scatt::kPrefetchRequestSinv
                                  | scatt::kPrefetchRequestRinv);
    std::cout << splan.report();

    // -------- Asymptotic-buffer (psi_asym_) sizing --------------------------
    // The BackPropagator keeps the last n_asym matrices (N_psi x N_psi) in
    // RAM for the AsymptoticAmplitudes fit.  At L=100 / N_psi~10201 /
    // n_asym=300 this is ~250 GB -- DOMINANT at high L, and previously
    // unaccounted by plan_storage().  We do NOT add it to non_storage_bytes
    // (that would shrink the auto chunk_size for pot/sinv/rinv/psi and
    // invalidate any on-disk checkpoints with the legacy chunk size).
    // Instead, we route psi_asym to a separate DISK-chunked store when the
    // planner's leftover RAM is too tight to keep it resident.
    //
    // n_asym_main MUST stay in sync with bpc.n_asym in run_one_energy().
    constexpr int      n_asym_main   = 300;
    const std::size_t  bytes_per_psi = std::size_t(n_ch) * n_ch * 8ull;
    const std::size_t  bytes_psi_asym = std::size_t(n_asym_main) * bytes_per_psi;

    // Free-RAM estimate AFTER the planner placed its four stores.  This is
    // the planner's reserve + any MEMORY-mode slack.  Same effective_raw
    // logic as plan_storage() so we don't disagree about the budget.
    std::size_t effective_raw = 0;
    {
        const std::size_t sys = detect_total_ram_bytes();
        const std::size_t cap = cli.memory_budget;
        if (sys > 0 && cap > 0)      effective_raw = std::min(sys, cap);
        else if (sys > 0)            effective_raw = sys;
        else                         effective_raw = cap;
        if (effective_raw == 0)      effective_raw = 8ull << 30;
    }
    const std::size_t splan_peak_resident =
          splan.pot.resident_bytes  + splan.sinv.resident_bytes
        + splan.rinv.resident_bytes + splan.psi.resident_bytes;
    const std::size_t fixed_costs =
          scatt::kDefaultRuntimeOverheadBytes + non_storage_bytes
        + splan_peak_resident;
    const std::size_t slack_after_planner =
        (effective_raw > fixed_costs) ? (effective_raw - fixed_costs) : 0;

    // Decision: prefer MEMORY if psi_asym fits with a 5 GB safety margin;
    // otherwise DISK, with chunk_size sized so the resident chunk eats at
    // most 50% of the remaining slack (and clamped to PotentialStorage's
    // [4, 200] policy plus n_asym).
    constexpr std::size_t kAsymMemoryMargin = 5ull << 30;  // 5 GB safety
    StorageMode psi_asym_mode = StorageMode::MEMORY;
    int         psi_asym_chunk_size_main = 20;
    if (bytes_psi_asym + kAsymMemoryMargin > slack_after_planner) {
        psi_asym_mode = StorageMode::DISK;
        const std::size_t half_slack = slack_after_planner / 2;
        std::size_t chunk = (bytes_per_psi > 0)
                          ? (half_slack / bytes_per_psi) : 4;
        if (chunk < 4)              chunk = 4;
        if (chunk > 200)            chunk = 200;
        if (static_cast<int>(chunk) > n_asym_main) chunk = n_asym_main;
        psi_asym_chunk_size_main = static_cast<int>(chunk);
    }

    auto fmt_GB = [](std::size_t b) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(3)
          << (b / double(1ull << 30)) << " GB";
        return o.str();
    };
    std::cout << "[storage-plan] psi_asym :"
              << "  n_asym=" << n_asym_main
              << "  total=" << fmt_GB(bytes_psi_asym)
              << "  slack_after_planner=" << fmt_GB(slack_after_planner)
              << "  -> "
              << (psi_asym_mode == StorageMode::MEMORY ? "MEMORY" : "DISK")
              << "  chunk_size=" << psi_asym_chunk_size_main
              << "  resident="
              << fmt_GB(psi_asym_mode == StorageMode::MEMORY
                          ? bytes_psi_asym
                          : std::size_t(psi_asym_chunk_size_main) * bytes_per_psi)
              << "\n";

    std::cout << "[scattering] molhash=" << mhash << "\n"
              << "             work   =" << work    << "\n"
              << "             scratch=" << scratch << "\n"
              << "             pot    =" << pot_dir << "\n"
              << "             dipole =" << dipole_dir << "\n";

    // -------- Pot (persistent, reused across all ik) --------
    Potentials pot(params);
    { ProfileScope _s(bench, "Potentials::build");
      pot.build(data, splan.pot.mode, pot_dir, /*verbose=*/true,
                /*try_load_checkpoint=*/true,
                /*save_checkpoint=*/true,
                /*symmetric_storage=*/cli.symmetric_storage,
                /*parallel_chunk_write=*/cli.parallel_chunk_write); }

    // (4) Free V_H now that the Potential matrix elements are baked in.
    // V_H is the largest single object loaded so far (subset is
    // n_exp x Nr ≈ 3.2 GB for C8F8 at l_cont=100; full would be 14x that).
    {
        const std::size_t bytes = static_cast<std::size_t>(data.V_H.rows())
                                  * data.V_H.cols() * sizeof(double);
        data.V_H.resize(0, 0);
        std::cout << "[scattering] freed data.V_H ("
                  << (bytes >> 20) << " MB) after Potentials::build\n";
    }

    // (5) Load the truncated orbital tile.  ANGULAR truncation happens
    // here at LOAD TIME via HDF5 hyperslab -- so one preprocessing file
    // serves any subsequent l_continuum without re-running the 7-hour
    // SCE projection.  The radial extent is whatever preprocess_molden
    // recorded as Nr_orb_store (typically tightened by --orb-rmax-auto).
    {
        ProfileScope _s(bench, "psi_lm::load_truncated");
        // The on-disk row stride is data.Nlm_orb_store, which equals
        // Nlm_sce when preprocess didn't truncate angular (the new
        // default), or smaller if --orb-lmax was passed.  Either way we
        // pass the actual stored stride to read_psi_lm_truncated.
        const int Nlm_stored = data.Nlm_orb_store;
        // Maximum l on the orbital side: l_cont + l_exch.  l_exch
        // defaults to min(l_cont, 10) inside WavefunctionSetup; pick the
        // same here.
        const int l_exch_max = std::min(params.l_max_continuum, 10);
        const int l_orb_max  = params.l_max_continuum + l_exch_max;
        const int n_lambda_cut = (l_orb_max + 1) * (l_orb_max + 1);
        const int n_transition_load = data.Nr_orb_store > 0
                                      ? data.Nr_orb_store
                                      : static_cast<int>(data.Nr);
        if (n_lambda_cut > Nlm_stored) {
            throw std::runtime_error(
                "psi_lm load: requested n_lambda_cut=" + std::to_string(n_lambda_cut) +
                " (l_cont=" + std::to_string(params.l_max_continuum) +
                ", l_exch=" + std::to_string(l_exch_max) + ")"
                " exceeds the on-disk Nlm_orb_store=" + std::to_string(Nlm_stored) +
                ".  Rerun preprocess_molden without --orb-lmax (or with"
                " --orb-lmax >= " + std::to_string(l_orb_max) + ").");
        }
        std::cout << "[scattering] loading psi_lm: hyperslab"
                     " n_orb=" << data.n_sce
                  << "  n_lambda_cut=" << n_lambda_cut
                  << " (on-disk Nlm=" << Nlm_stored << ")"
                  << "  n_transition=" << n_transition_load << "\n";
        data.psi_lm = reader.read_psi_lm_truncated(
            data.n_sce, Nlm_stored, n_lambda_cut, n_transition_load);
        // Update the loaded layout so WavefunctionSetup uses the right
        // row stride.
        data.Nlm_orb_store  = n_lambda_cut;
        data.Lmax_orb_store = l_orb_max;
        data.Nr_orb_store   = n_transition_load;
        const std::size_t bytes = static_cast<std::size_t>(data.psi_lm.rows())
                                  * data.psi_lm.cols() * sizeof(double);
        std::cout << "[scattering] data.psi_lm shape=("
                  << data.psi_lm.rows() << ", " << data.psi_lm.cols()
                  << ")  bytes=" << (bytes >> 20) << " MB\n";
    }
    // Sub-stage breakdown. In MEMORY mode the ir loop is parallel, so
    // summed ns can exceed top-level wall by roughly n_threads.
    {
        const auto& ps = pot.stats();
        const std::size_t rss = current_peak_rss_bytes();
        bench.add("Pot::gaunt_build",  ps.t_gaunt_build_ns,  rss);
        bench.add("Pot::v_sigma",      ps.t_v_sigma_ns,      rss);
        bench.add("Pot::gaunt_matvec", ps.t_gaunt_matvec_ns, rss);
        bench.add("Pot::v_pol",        ps.t_v_pol_ns,        rss);
        bench.add("Pot::store",        ps.t_store_ns,        rss);
    }

    // -------- Initial state + occupied orbitals (energy-independent) --------
    // The TARGET orbitals (data.orb_*  /  ref_bundle.chi) come from the
    // closed-shell neutral molden -- they are the bystander occupied set
    // used for Lagrange orthogonalisation in the dipole matrix element.
    //
    // The INITIAL STATE for the dipole element is selected as follows:
    //   * If the preproc HDF5 has an /initial_state/ group (the user
    //     passed preprocess_molden --initial-state-molden <anion.molden>),
    //     use that SOMO.  This is the anion-photodetachment process
    //     H2O^-(THF) + h*omega -> H2O + e^-  (the C8F8 production setup).
    //     The recorded ionisation potential becomes the anion SOMO
    //     energy, i.e. the vertical detachment energy.
    //   * Otherwise fall back to the neutral target's HOMO -- the
    //     legacy "neutral photoionisation" interpretation.
    // Previously the code ALWAYS used the target HOMO regardless of
    // whether /initial_state/ existed -- silently overriding the user's
    // intent.  This bug under-reported the anion process and gave the
    // wrong omega axis (used neutral IP instead of anion VDE).
    Eigen::MatrixXd chi_init_homo;
    std::vector<OccupiedOrbital> occ;
    double E_HOMO = 0.0;
    int n_occ = 0;
    {   ProfileScope _s(bench, "ref_bundle::prepare");
        auto ref_bundle = WavefunctionSetup::prepare(params, data, /*energy=*/0.01);
        const int Nlm_init = static_cast<int>(ref_bundle.chi[0].cols());
        const int Nr       = static_cast<int>(params.N_grid);
        n_occ              = ref_bundle.params.n_occ;
        const int i_homo   = n_occ - 1;

        // Bystander occupied orbitals (always the target's, regardless of
        // initial-state source).
        const int n_chi = static_cast<int>(ref_bundle.chi.size());
        occ.resize(n_occ);
        for (int j = 0; j < n_occ; ++j) {
            occ[j].phi = Eigen::MatrixXd::Zero(Nr, Nlm_init);
            for (int ir = 0; ir < n_chi; ++ir)
                for (int lam = 0; lam < Nlm_init; ++lam)
                    occ[j].phi(ir, lam) = ref_bundle.chi[ir](j, lam);
            occ[j].spin_factor = 2.0;   // closed-shell default
        }

        chi_init_homo = Eigen::MatrixXd::Zero(Nr, Nlm_init);
        if (data.has_initial_state) {
            // Load the anion SOMO (or other user-supplied initial state)
            // via a truncated hyperslab read.  Disk layout is
            //   /initial_state/psi_lm  shape (Nlm_sce, Nr_orb_store)
            // We pull the first Nlm_init rows (matches our l_orb_max cut)
            // and the first n_chi columns (matches WavefunctionSetup's
            // chi-support cutoff), then transpose into (Nr, Nlm_init)
            // layout to match chi_init_homo.
            const int Nlm_sce_full   = data.psi_lm.rows() > 0
                ? static_cast<int>(data.psi_lm.rows() / std::max(1, data.n_sce))
                : (data.Lmax_sce + 1) * (data.Lmax_sce + 1);
            const int Nr_orb_store   = (data.Nr_orb_store > 0)
                ? data.Nr_orb_store : Nr;
            const int Nlm_take       = std::min(Nlm_init, Nlm_sce_full);
            const int Nr_take        = std::min(n_chi, Nr_orb_store);
            Eigen::MatrixXd raw = reader.read_initial_state_psi_lm_truncated(
                Nlm_sce_full, Nlm_take, Nr_take);
            // raw is (Nlm_take, Nr_take).  Transpose into chi_init_homo.
            for (int ir = 0; ir < Nr_take; ++ir)
                for (int lam = 0; lam < Nlm_take; ++lam)
                    chi_init_homo(ir, lam) = raw(lam, ir);
            E_HOMO = data.init_state_energy;
            std::cout << "[scattering] initial state = anion SOMO from "
                         "/initial_state/  (E_init=" << E_HOMO << " Ha, "
                         "VDE Koopmans=" << (-E_HOMO * 27.2113862459) << " eV)\n";
        } else {
            // Legacy: use neutral HOMO as initial state.
            for (int ir = 0; ir < n_chi; ++ir)
                for (int lam = 0; lam < Nlm_init; ++lam)
                    chi_init_homo(ir, lam) = ref_bundle.chi[ir](i_homo, lam);
            E_HOMO = data.orb_energies.at(i_homo);
            std::cout << "[scattering] initial state = neutral HOMO  "
                         "(E_HOMO=" << E_HOMO << " Ha, IP=" << (-E_HOMO*27.21)
                      << " eV)\n";
        }
    }

    // -------- DipoleWriter (persistent) --------
    DipoleScanMeta meta;
    meta.r_min           = params.r_min;
    meta.dr              = params.dr;
    meta.N_grid          = params.N_grid;
    meta.l_max_continuum = params.l_max_continuum;
    meta.kgrid           = kgrid;
    meta.E_HOMO          = E_HOMO;
    meta.occ_energies.assign(n_occ, 0.0);
    meta.occ_spin_factors.assign(n_occ, 2.0);
    for (int j = 0; j < n_occ; ++j)
        meta.occ_energies[j] = data.orb_energies.at(j);
    for (const auto& a : data.atoms)
        meta.atoms.push_back({ a.Z, a.x, a.y, a.z });
    meta.molecule_name  = fs::path(cli.hdf5_input).stem().string();
    meta.git_hash       = "";
    meta.iso_date_utc   = iso_date_utc_now();
    meta.psi_dir_prefix = scratch + "/psi_" + mhash + "_";

    DipoleWriter writer(dipole_dir, meta);

    // -------- Energy loop --------
    std::cout << "[scan] " << kgrid.size() << " energy point(s)\n";
    for (int ik = kgrid.ik_min; ik < kgrid.ik_max; ++ik) {
        if (writer.has_energy(ik)) {
            std::cout << "  ik=" << ik << "  (skip, already done)\n";
            continue;
        }
        const double E_kin = kgrid.E(ik);
        const std::string ik_tag  = kgrid.tag(ik);
        const std::string sinv_d  = scratch + "/sinv_" + mhash + "_" + ik_tag;
        const std::string rinv_d  = scratch + "/rinv_" + mhash + "_" + ik_tag;
        const std::string psi_d   = scratch + "/psi_"  + mhash + "_" + ik_tag;

        run_one_energy(ik, E_kin, data, params, pot, splan,
                       sinv_d, rinv_d, psi_d,
                       chi_init_homo, occ, writer, bench, cli.use_gpu,
                       cli.check_residual, cli.residual_probes,
                       cli.parallel_disk_sinv,
                       cli.symmetric_storage,
                       cli.parallel_chunk_write,
                       psi_asym_mode,
                       psi_asym_chunk_size_main,
                       cli.dipole_only,
                       cli.allow_closed_channels,
                       cli.allow_checkpoint_rechunk);

        // Cleanup: sinv and rinv are not needed once psi is saved and the
        // dipole has been written. Psi stays in $SCRATCH for possible
        // continuum-continuum dipole post-processing.
        std::error_code ec;
        fs::remove_all(sinv_d, ec);
        fs::remove_all(rinv_d, ec);
    }

    writer.finalize();

    // ----- bench summary -----
    const auto t_total_end = std::chrono::steady_clock::now();
    bench.set_total_wall_ns(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_total_end - t_total_start).count()));
    bench.print(std::cout);

    const std::string bench_path = !cli.bench_out.empty()
        ? cli.bench_out
        : (dipole_dir + "/bench.dat");
    bench.save_tsv(bench_path);
    std::cout << "[scattering] bench dump -> " << bench_path << "\n";

    std::cout << "[scattering] done.\n";
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}

// ===========================================================================
// Helper definitions
// ===========================================================================
static void usage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " <preproc.h5> [ik_min] [ik_max] [dk] [flags]\n"
        "       " << prog << " --help\n"
        "\n"
        "Positional (optional; if one is given, give all three):\n"
        "  ik_min         starting ik                       (default 1)\n"
        "  ik_max         ending ik, INCLUSIVE              (default 1  -> one point)\n"
        "  dk             momentum step, au                 (default 0.01)\n"
        "\n"
        "Energy grid: k(ik) = ik * dk,  E(ik) = k^2 / 2.\n"
        "Uniform-in-k spacing gives finer E-spacing near threshold.\n"
        "\n"
        "Flags:\n"
        "  --lmax-cont L          continuum angular cutoff  (default 10)\n"
        "  --dr H                 radial step override      (default from HDF5)\n"
        "  --r-max R              outer radius\n"
        "  --work DIR             persistent dir            (default $WORK or ./work)\n"
        "  --scratch DIR          scratch dir               (default $SCRATCH or ./scratch)\n"
        "  --memory-budget BYTES  cap memory budget         (default auto-detect)\n"
        "  --scan-id STR          override scan_id          (default ik<min>-<max>_dk<dk>)\n"
        "  --omp-threads N        pin OpenMP to N threads   (default: OMP_NUM_THREADS or max)\n"
        "  --bench-out PATH       write bench TSV to PATH   (default: <dipole_dir>/bench.dat)\n"
        "  --use-gpu              offload FRP + BP per-step work to SYCL GPU\n"
        "  --no-checkpoint-rechunk\n"
        "                         disable the auto re-chunking of on-disk\n"
        "                         checkpoints whose chunk_size > runtime budget.\n"
        "                         Default ON: a cross-node move that would otherwise\n"
        "                         force a multi-hour rebuild becomes a ~minutes\n"
        "                         disk-I/O re-layout (byte-identical to a fresh\n"
        "                         build).  Pass this flag to fall back to the older\n"
        "                         reject-and-rebuild behaviour (e.g. on disk-tight\n"
        "                         scratch where the transient 2x checkpoint size\n"
        "                         won't fit).\n"
        "  --dipole-only          RECOVERY MODE: SKIP Sinv / FRP / K-extraction /\n"
        "                         BP backprop.  Load psi from existing\n"
        "                         $SCRATCH/psi_<hash>_ik<NNNN>/ checkpoint and\n"
        "                         JUST recompute (A, B) + dipoles with the current\n"
        "                         initial state.  Use this to salvage runs whose\n"
        "                         psi/__SUCCESS__ markers are intact but whose\n"
        "                         dipoles need re-evaluation -- e.g. after the\n"
        "                         init-state fix.  Per-ik cost drops from ~hours\n"
        "                         to ~minutes.  Throws clearly if psi checkpoint\n"
        "                         missing or manifest mismatches.\n"
        "  --check-residual       after BP, evaluate the PDE residual at probe ir's\n"
        "                         and print a per-probe table.  Forces compute_f=true\n"
        "                         (BP cost roughly doubles).\n"
        "  --residual-probes N    number of probe ir's spread across the kept window\n"
        "                         (default 8).  Only used with --check-residual.\n"
        "                          (requires build with -DSCATT_WITH_SYCL=ON)\n"
        "  --parallel-disk-sinv   Sinv builds in DISK mode use chunk-blocked OpenMP\n"
        "                         (~10-50x speedup vs serial DISK at l_cont >= 80).\n"
        "                         Default OFF.  Bit-identical to the serial DISK output\n"
        "                         (verified by test_sinv_parallel_disk_path).\n"
        "                         At l_cont >= 100 cap OMP_NUM_THREADS to ~80\n"
        "                         to fit the planner memory budget. We recomamned to keep it off for larger l_count, not efficient!\n"
        "  --symmetric-storage    Store ONLY the lower triangle of pot/sinv/rinv on disk.\n"
        "                         Halves disk space and disk I/O on those three stages\n"
        "                         (psi remains full -- it's not symmetric).\n"
        "                         ZERO accuracy loss: pot/sinv/rinv are bit-symmetric\n"
        "                         by construction; the upper triangle is byte-equal to\n"
        "                         the transpose of the lower.  Validated by\n"
        "                         test_storage_symmetric_end_to_end.\n"
        "                         Backward-compatible: pre-existing v1 (full-matrix)\n"
        "                         checkpoints still load correctly under the new code.\n"
        "                         Default ON.\n"
        "  --parallel-chunk-write Multi-threaded write_chunk: pwrite at distinct offsets\n"
        "                         (POSIX-thread-safe), atomic temp + rename + fsync for\n"
        "                         crash safety.  Caps at 4 worker threads per chunk.\n"
        "                         ZERO accuracy loss: bytes on disk are byte-identical\n"
        "                         to the legacy serial writer (validated by\n"
        "                         test_storage_parallel_write).  Affects pot, sinv,\n"
        "                         rinv (not psi).  Default OFF.\n"
        "  --allow-closed-channels  Opt-in: in AsymptoticAmplitudes, ℓ values whose\n"
        "                         2×2 normal matrix is catastrophically singular\n"
        "                         (k·r_max ≲ ℓ -- channel is under the centrifugal\n"
        "                         barrier across the entire fit window) are MARKED\n"
        "                         CLOSED with A_{μν}=δ_{μν}, B_{μν}=0 instead of\n"
        "                         aborting.  Equivalent to running with a smaller\n"
        "                         l_cont in the under-barrier subset.  Safe IFF the\n"
        "                         observable is converged in l_cont within the open-ℓ\n"
        "                         channels (verify with a converged smaller-l_cont\n"
        "                         reference run).  Default OFF -- accuracy preserved.\n"
        "  -h, --help             print this help and exit\n"
        "\n"
        "Outputs:\n"
        "  $WORK/pot_<hash>/                         # persistent potentials checkpoint\n"
        "  $WORK/dipole_<hash>_<scan_id>/            # per-ik HDF5 + manifest\n"
        "      ikNNNN.h5     -- A, B, 6 dipole slices (raw+ortho, L+V, x/y/z)\n"
        "      manifest.h5   -- channels, occ, grid, conventions\n"
        "      __SUCCESS__   -- marker on scan completion\n"
        "\n"
        "  $SCRATCH/psi_<hash>_ikNNNN/               # kept: wavefunctions for c-c dipole\n"
        "  (sinv/rinv scratch dirs are deleted after each ik is saved)\n";
}

static CLI parse_cli(int argc, char** argv) {
    // Handle --help before requiring a positional argument.
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
    }
    if (argc < 2) { usage(argv[0]); std::exit(2); }

    CLI c;
    c.hdf5_input = argv[1];

    // Positional block (optional, all-or-nothing).
    int i = 2;
    std::vector<std::string> pos;
    while (i < argc && std::string(argv[i]).rfind("--", 0) != 0) {
        pos.push_back(argv[i]); ++i;
    }
    if (pos.size() == 3) {
        c.ik_min           = std::stoi(pos[0]);
        c.ik_max_inclusive = std::stoi(pos[1]);
        c.dk               = std::stod(pos[2]);
    } else if (!pos.empty()) {
        std::cerr << "Positional args: expected 3 (ik_min ik_max dk) or none; got "
                  << pos.size() << "\n";
        usage(argv[0]); std::exit(2);
    }

    for (; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << a << "\n"; std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--lmax-cont")     c.l_max_continuum = std::stoi(next());
        else if (a == "--dr")            c.dr_override     = std::stod(next());
        else if (a == "--r-max")         c.r_max_override  = std::stod(next());
        else if (a == "--work")          c.work_dir        = next();
        else if (a == "--scratch")       c.scratch_dir     = next();
        else if (a == "--memory-budget") c.memory_budget   = std::stoull(next());
        else if (a == "--scan-id")       c.scan_id_override= next();
        else if (a == "--omp-threads")   c.omp_threads     = std::stoi(next());
        else if (a == "--bench-out")     c.bench_out       = next();
        else if (a == "--use-gpu")       c.use_gpu         = true;
        else if (a == "--no-checkpoint-rechunk")  c.allow_checkpoint_rechunk = false;
        else if (a == "--check-residual"){ c.check_residual  = true; }
        else if (a == "--dipole-only")   { c.dipole_only     = true; }
        else if (a == "--parallel-disk-sinv"){ c.parallel_disk_sinv = true; }
        else if (a == "--symmetric-storage")  { c.symmetric_storage   = true; }
        else if (a == "--parallel-chunk-write"){ c.parallel_chunk_write = true; }
        else if (a == "--allow-closed-channels"){ c.allow_closed_channels = true; }
        else if (a == "--residual-probes"){ c.residual_probes = std::stoi(next()); }
        else {
            std::cerr << "unknown option: " << a << "\n";
            usage(argv[0]); std::exit(2);
        }
    }

    if (c.ik_min < 0 || c.ik_max_inclusive < c.ik_min)
        throw std::invalid_argument("ik_min/ik_max out of order or negative");
    if (!(c.dk > 0.0))
        throw std::invalid_argument("dk must be > 0");
    return c;
}

static std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

static std::string iso_date_utc_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

static void run_one_energy(int                       ik,
                           double                    E_kin,
                           const io::PreprocData&    data,
                           Parameters&               params,
                           Potentials&               pot,
                           const StoragePlan&        splan,
                           const std::string&        sinv_dir,
                           const std::string&        rinv_dir,
                           const std::string&        psi_dir,
                           const Eigen::MatrixXd&    chi_init_homo,
                           const std::vector<OccupiedOrbital>& occ,
                           DipoleWriter&             writer,
                           BenchReport&              bench,
                           bool                      use_gpu,
                           bool                      check_residual,
                           int                       residual_probes,
                           bool                      parallel_disk_sinv,
                           bool                      symmetric_storage,
                           bool                      parallel_chunk_write,
                           StorageMode               psi_asym_mode,
                           int                       psi_asym_chunk_size,
                           bool                      dipole_only,
                           bool                      allow_closed_channels,
                           bool                      allow_checkpoint_rechunk)
{
    auto t0 = std::chrono::steady_clock::now();

    // Per-stage progress pings -- give the operator a mental model of
    // where a long run is at without waiting for the final banner.
    auto stage_log = [&](const char* what) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "  [ik=" << ik << "  k="
                  << std::fixed << std::setprecision(4) << std::sqrt(2.0 * E_kin)
                  << "]  " << what
                  << std::defaultfloat << "  (t+" << std::setprecision(1)
                  << elapsed << "s)\n" << std::flush;
    };

    std::cout << std::scientific << std::setprecision(10);
    std::cerr << std::scientific << std::setprecision(10);

    stage_log("setup: prepare bundle (chi, G_coeff, n_transition)");
    auto bundle = [&]{
        ProfileScope _s(bench, "WavefunctionSetup::prepare");
        return WavefunctionSetup::prepare(params, data, E_kin);
    }();

    auto EC_ptr = [&]{
        ProfileScope _s(bench, "ExchangeCoupling::ctor");
        return std::make_unique<ExchangeCoupling>(
            bundle.G_coeff, bundle.params.n_mu, bundle.params.n_sigma,
            bundle.params.n_occ, data.rmin, data.dr);
    }();
    ExchangeCoupling& EC = *EC_ptr;

    // RECOVERY MODE: --dipole-only.  We are about to re-evaluate the
    // dipole on an existing psi checkpoint produced by a previous run.
    // The Sinv / FRP / K-extraction / BP backward-propagation stages do
    // NOT depend on the initial state, so the on-disk psi is still valid
    // -- we just skip them entirely and let BP::run try_load the
    // checkpoint.
    //
    // The four objects (Sinv, FRP, KMatrixExtractor, BackPropagator) are
    // still CONSTRUCTED so that BackPropagator has the FRP/WI references
    // it expects.  BP.run() with try_load=true and a finalized
    // checkpoint short-circuits without touching those references.
    if (dipole_only) {
        stage_log("--dipole-only: skipping Sinv / FRP / K-extract / BP "
                  "backprop; will try-load psi from disk");
    }

    // Sinv.
    if (!dipole_only) {
        stage_log("Sinv: building Schur inverse on radial grid (parallel over ir)");
    }
    SchurInverter SI(bundle.params, pot, &EC, &bundle.chi);
    SchurInverter::Config sic;
    sic.verbose             = true;
    sic.try_load_checkpoint = true;
    sic.save_checkpoint     = (splan.sinv.mode != StorageMode::MEMORY);
    sic.storage             = splan.sinv.mode;
    sic.chunk_size          = splan.sinv.chunk_size;
    sic.checkpoint_dir      = sinv_dir;
    sic.symmetric_storage   = symmetric_storage;
    sic.parallel_chunk_write = parallel_chunk_write;
    // Forward the planner's prefetch decision.  Async chunk prefetch on
    // sinv is only enabled when the planner confirmed there is budget
    // for the EXTRA chunk-sized buffer; otherwise BP's start_prefetch
    // hops are no-ops (memory-safe).  See StoragePlanner Pass 3.
    sic.enable_prefetch     = splan.sinv.enable_prefetch;
    sic.allow_chunk_rechunk = allow_checkpoint_rechunk;
    // GPU dispatch for per-n Schur GEMM + LU inverse on the SYCL device.
    // Same --use-gpu flag as FRP and BP.  Measured at L=100 on Intel
    // PVC Max 1550: Sinv build ~11 h vs ~28 h CPU baseline = ~2.5x
    // speedup.  First-iteration cost is high (oneMKL JIT compile,
    // ~minutes); steady-state per-step is ~3-4 s.  Byte-equivalence
    // proven against CPU LAPACKE (test_gpu_sinv: max rel diff 6.7e-16
    // on H2O fixture).
    sic.use_gpu             = use_gpu;
    // Opt-in chunk-blocked parallel for DISK-mode Sinv.  Only meaningful
    // when storage is DISK -- MEMORY mode already parallelises directly.
    sic.parallel_disk_chunks = (parallel_disk_sinv
                                && splan.sinv.mode == StorageMode::DISK);
    if (sic.parallel_disk_chunks) {
        std::cout << "[scattering] --parallel-disk-sinv is ON "
                     "(chunk-blocked OpenMP, bit-identical to serial DISK)\n";
    }
    if (!dipole_only) {
        {   ProfileScope _s(bench, "SchurInverter::build", /*uses_gpu=*/sic.use_gpu);
            SI.build(sic); }
        // Sub-stage breakdown. NOTE: when the ir-loop is run in parallel,
        // these timers are summed ACROSS threads -- so they can exceed the
        // top-level SchurInverter::build wall time. Divide by n_threads for
        // per-thread estimate.
        const auto& s = SI.stats();
        const std::size_t rss = current_peak_rss_bytes();
        bench.add("Sinv::A_build", s.t_A_build_ns, rss);
        bench.add("Sinv::B_build", s.t_B_build_ns, rss);
        bench.add("Sinv::schur",   s.t_schur_ns,   rss);
        bench.add("Sinv::invert",  s.t_invert_ns,  rss);
        bench.add("Sinv::store",   s.t_store_ns,   rss);
    }

    WInverseOperator WI(bundle.params, SI, &EC, &bundle.chi, sic.W_min);

    // Rinv.
    if (!dipole_only) {
        stage_log(use_gpu
            ? "FRP: forward R-recursion  (GPU: getrf+getri on device)"
            : "FRP: forward R-recursion  (CPU: MKL / Eigen LU per step)");
    }
    ForwardRPropagator FRP(bundle.params, pot, WI);
    ForwardRPropagator::Config frc;
    frc.verbose             = true;
    frc.try_load_checkpoint = true;
    frc.save_checkpoint     = (splan.rinv.mode != StorageMode::MEMORY);
    frc.storage             = splan.rinv.mode;
    frc.chunk_size          = splan.rinv.chunk_size;
    frc.checkpoint_dir      = rinv_dir;
    frc.symmetric_storage   = symmetric_storage;
    frc.parallel_chunk_write = parallel_chunk_write;
    frc.use_gpu             = use_gpu;
    // Forward the planner's prefetch decision (see sinv block above for
    // rationale).  At C8F8 / l_cont=100 this stays false because the
    // planner has no budget for the +95 GB prefetch buffer.
    frc.enable_prefetch     = splan.rinv.enable_prefetch;
    frc.allow_chunk_rechunk = allow_checkpoint_rechunk;
    if (!dipole_only) {
        {   ProfileScope _s(bench, "ForwardRPropagator::run", /*uses_gpu=*/frc.use_gpu);
            FRP.run(frc); }
        // Sub-stage breakdown (internal timers).
        const auto& s = FRP.stats();
        const std::size_t rss = current_peak_rss_bytes();
        bench.add("FRP::u_assemble",   s.t_u_assemble_ns, rss);
        bench.add("FRP::linsolve_pLU", s.t_linsolve_ns,   rss);
        bench.add("FRP::store",        s.t_store_ns,      rss);
    }

    // K extraction: only meaningful when FRP was built.  In --dipole-only
    // mode we pass a placeholder psi_boundary to BP::run -- it's unused
    // because BP's try_load short-circuits before compute_Z_at_outer_*
    // gets called.
    Eigen::MatrixXd bc;
    if (!dipole_only) {
        stage_log("K: extracting K-matrix from Rinv at outer matching point");
        KMatrixExtractor KME(bundle.params, FRP);
        auto res_K = [&]{
            ProfileScope _s(bench, "KMatrixExtractor::extract");
            return KME.extract();
        }();
        bc = KMatrixExtractor::make_psi_boundary(bundle.params, res_K.K_matrix);
    } else {
        // Placeholder; will not be read because BP::run try-loads psi.
        bc = Eigen::MatrixXd::Zero(bundle.params.n_mu, bundle.params.n_mu);
    }

    // Psi back-prop.
    // ψ-storage trick (version_0 inspired):
    //   * Main store: [0, n_transition - 1]. WavefunctionSetup::prepare
    //     auto-selects n_transition from chi_cutoff (last orbital-support
    //     ir + 5-bohr buffer), so here we just use bundle.params.n_transition.
    //     Nothing past orbital support is used by the dipole integrand, so
    //     truncation is lossless.
    //   * Asymptotic buffer psi_asym_ (in memory): last 300 matrices near r_max
    //     for the A/B fit (AsymptoticAmplitudes reads from the outer window).
    const int N_grid_int = static_cast<int>(bundle.params.n_grid);
    const int n_main_hi  = std::min(
        N_grid_int - 1,
        std::max(1, bundle.params.n_transition - 1));

    // --dipole-only precheck: fail FAST and CLEARLY if the psi
    // checkpoint we need to load is missing or incomplete, BEFORE
    // BackPropagator::run tries try_load and (on partial failure) falls
    // through to the rebuild path -- which would crash later with
    // "PotentialStorage::get before init" from the unbuilt FRP, leaving
    // the operator hunting for the root cause.
    if (dipole_only) {
        namespace fs2 = std::filesystem;
        const fs2::path psi_root         = psi_dir;
        const fs2::path psi_succ         = psi_root / "__SUCCESS__";
        const fs2::path asym_chunked_dir = psi_root / "asym";
        const fs2::path asym_chunked_ok  = asym_chunked_dir / "__SUCCESS__";
        const fs2::path asym_single_file = psi_root / "psi_asym.bin";
        const bool psi_ok           = fs2::exists(psi_succ);
        const bool asym_chunked_ok_ = fs2::exists(asym_chunked_ok);
        const bool asym_single_ok_  = fs2::exists(asym_single_file);
        // We always require psi/__SUCCESS__.  The asym buffer is required
        // when n_asym>0 (production: yes; toy tests with n_asym=0 don't).
        // The buffer can be stored in EITHER format -- the writing node's
        // memory budget decided which.  Accept either:
        //   single-file : <dir>/psi_asym.bin
        //   chunked dir : <dir>/asym/__SUCCESS__
        // BackPropagator::load_asym auto-detects which is on disk.
        const bool need_asym = (300 > 0);   // matches bpc.n_asym below
        const bool asym_ok   = (asym_single_ok_ || asym_chunked_ok_);
        if (!psi_ok || (need_asym && !asym_ok)) {
            std::ostringstream oss;
            oss << "--dipole-only requested but psi checkpoint at "
                << psi_dir << " is missing or incomplete:\n"
                << "  __SUCCESS__               : "
                << (psi_ok ? "found" : "MISSING") << "\n"
                << "  asym buffer (either fmt)  : "
                << (asym_ok ? "found"
                            : (need_asym ? "MISSING (n_asym>0 requires it)"
                                         : "(not required, n_asym=0)")) << "\n"
                << "    psi_asym.bin (single)   : "
                << (asym_single_ok_  ? "found" : "absent") << "\n"
                << "    asym/__SUCCESS__ (chunk): "
                << (asym_chunked_ok_ ? "found" : "absent") << "\n"
                << "  If BP backprop did not finalise cleanly for this ik,\n"
                << "  --dipole-only cannot recover it -- the only path is a\n"
                << "  full re-run (omit --dipole-only) for THIS ik specifically.";
            throw std::runtime_error(oss.str());
        }
        std::cout << "[scattering] --dipole-only precheck: psi/__SUCCESS__ + "
                  << (asym_single_ok_ ? "psi_asym.bin (single-file)"
                                       : "asym/__SUCCESS__ (chunked)")
                  << " present.\n";
    }

    stage_log(use_gpu
        ? "BP: back-propagating psi  (GPU: two GEMMs on device per step)"
        : "BP: back-propagating psi  (CPU: GEMM + W^-1 per step)");
    BackPropagator BP(bundle.params, pot, FRP, WI);
    BackPropagator::Config bpc;
    bpc.verbose             = true;
    // The PDE residual check needs the f-block (the exchange-source term
    // RHS = -2α/r·Σ G·χ·f), so force compute_f when --check-residual is
    // requested.  This roughly doubles the BP runtime but is what makes
    // the residual diagnostic actually informative.
    bpc.compute_f           = check_residual;
    bpc.n_keep_lo           = 0;
    bpc.n_keep_hi           = n_main_hi;
    bpc.n_asym              = 300;         // tail for asymptotic fit; main.cpp's
                                           // n_asym_main MUST stay in sync.
    bpc.psi_storage         = splan.psi.mode;
    bpc.chunk_size          = splan.psi.chunk_size;
    // psi_asym storage decision is made up-front in main() and threaded in
    // here -- it must NOT be re-derived locally (consistency with the
    // storage-plan log).
    bpc.psi_asym_storage    = psi_asym_mode;
    bpc.psi_asym_chunk_size = psi_asym_chunk_size;
    // Enable resume-from-checkpoint: if a previous run wrote finalized
    // psi + psi_asym chunks (SUCCESS markers + matching manifests), BP
    // skips its ~3 h rebuild.  Falls through gracefully when no checkpoint
    // exists or the manifest mismatches.  Combined with the cleanup of
    // sinv/rinv after each ik (below), an externally-killed run can be
    // restarted and pick up at "A/B fit + dipole" instead of "full BP".
    bpc.try_load_checkpoint = true;
    bpc.save_checkpoint     = true;       // always keep psi for post-processing
    bpc.checkpoint_dir      = psi_dir;
    bpc.allow_chunk_rechunk = allow_checkpoint_rechunk;
    bpc.use_gpu             = use_gpu;
    bpc.parallel_chunk_write = parallel_chunk_write;

    // Memory optimisation (--check-residual OFF only): pot is touched by BP
    // exactly ONCE -- pot_.get(N_grid-1) at the outer boundary -- and not at
    // all by K extraction, AsymptoticAmplitudes, or DipoleMatrixElement.  The
    // residual probe (--check-residual) DOES read pot, so we keep the chunk
    // cache resident in that mode.  When the probe is off, deep-copy V_N once
    // and free pot's chunk read cache (~155 GB at L=100) for the duration of
    // BP+downstream.  Bit-identical: BP's compute_Z_at_outer_boundary_ uses
    // V_N's bytes verbatim, regardless of where it came from.
    Eigen::MatrixXd V_N_cached;          // empty unless we cache below
    if (!check_residual) {
        V_N_cached = pot.get(static_cast<std::size_t>(N_grid_int - 1));  // deep copy
        pot.release_read_buffer();       // no-op for MEMORY pot; frees chunk cache for DISK pot
        bpc.cached_V_outer = &V_N_cached;
        std::cout << "[scattering] pot read cache released; BP will use a "
                     "pre-cached V_N copy ("
                  << ((static_cast<std::size_t>(V_N_cached.rows())
                       * V_N_cached.cols() * sizeof(double)) >> 20)
                  << " MB).  --check-residual not requested.\n";
    }

    {   ProfileScope _s(bench, "BackPropagator::run", /*uses_gpu=*/bpc.use_gpu);
        BP.run(bc, bpc); }
    if (dipole_only && !BP.psi_in_memory()) {
        // BP returned successfully but the storage is DISK-backed:
        // BP::run try_load succeeded.  Good.
    }
    if (dipole_only) {
        // Sanity gate: if try_load FAILED, BP::run would have fallen
        // through to the rebuild path which calls FRP::get and crashes
        // because FRP wasn't run.  If we reach here without a crash,
        // the load succeeded.  Verify by checking get_psi at a couple of
        // grid points (cheap; surfaces a clear error before the dipole
        // call starts grinding).
        try {
            (void)BP.get_psi(static_cast<std::size_t>(BP.n_keep_hi()));
            (void)BP.get_psi(static_cast<std::size_t>(BP.n_keep_lo()));
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("--dipole-only: psi checkpoint at ") + psi_dir +
                " did not load cleanly. Cause: " + e.what() +
                "  (Was the checkpoint finalized? __SUCCESS__ marker present? "
                "Does the on-disk manifest match the current run's params?)");
        }
        std::cout << "[scattering] --dipole-only: psi loaded from "
                  << psi_dir << "  (BP backprop skipped)\n";
    }
    // Sub-stage breakdown (internal timers).
    {
        const auto& s = BP.stats();
        const std::size_t rss = current_peak_rss_bytes();
        bench.add("BP::rinv_fetch", s.t_rinv_fetch_ns, rss);
        bench.add("BP::gemm_z",     s.t_gemm_z_ns,     rss);
        bench.add("BP::wi_apply",   s.t_wi_apply_ns,   rss);
        bench.add("BP::store",      s.t_store_ns,      rss);
    }

    // ----- Optional PDE residual probe -------------------------------------
    // Evaluate the local PDE for ψ at a handful of `n` points spread across
    // the kept window:
    //
    //   LHS_n = ψ''(n) + 2 E ψ(n) - 2 V_n ψ(n)
    //   RHS_n = -2α/r_n · Σ_{μ λ σ} G_{μ λ σ} · χ_λ(n) · f_σ(n)
    //
    // and reports max|LHS - RHS| / max(|LHS|, |RHS|).  This is the per-
    // energy version of `pde_residual_check`'s "PDF eq. 16" block, but
    // working on the in-memory bundle/BP/pot we already have -- no
    // re-running of FRP+BP.  Cheap (a few ms per probe).  Activated by
    // --check-residual; --residual-probes N selects how many ir's to sample.
    if (check_residual && residual_probes > 0) {
        ProfileScope _s(bench, "check_residual");
        const int Nr_full = static_cast<int>(bundle.params.n_grid);
        const int N_psi   = bundle.params.n_mu;
        const int n_sigma = bundle.params.n_sigma;
        const int n_occ   = bundle.params.n_occ;
        const double h     = bundle.params.dr;
        const double h2_inv = 1.0 / (h * h);
        const double alpha  = std::sqrt(2.0 * M_PI);

        // Pick `residual_probes` interior ir's spread across BP's kept
        // ranges.  BP only retains ψ on the main keep range
        // [n_keep_lo, n_keep_hi] and the asymptotic buffer
        // [asym_lo, asym_hi]; sampling outside either window throws
        // (`get_psi: n=... outside kept range...`).  We need n±2 in
        // bounds for the 5-point ψ'' stencil, so each window contributes
        // [lo+2, hi-2].  We allocate ~80% of probes to the main range
        // (physically interesting -- exchange source lives there) and
        // the rest to the asymptotic buffer.
        std::vector<int> probes;
        probes.reserve(residual_probes);
        const int main_lo = std::max(2, BP.n_keep_lo() + 2);
        const int main_hi = std::min(Nr_full - 3, BP.n_keep_hi() - 2);
        const int asym_lo = std::max(2, BP.outer_window_lo() + 2);
        const int asym_hi = std::min(Nr_full - 3, BP.outer_window_hi() - 2);
        const bool main_ok = (main_hi > main_lo);
        const bool asym_ok = (asym_hi > asym_lo) &&
                             (asym_lo > BP.n_keep_hi());  // distinct window
        int n_main = main_ok ? std::max(1, (residual_probes * 4) / 5) : 0;
        int n_asym = asym_ok ? (residual_probes - n_main) : 0;
        if (!main_ok && asym_ok) n_asym = residual_probes;
        if (main_ok && !asym_ok) n_main = residual_probes;
        auto fill = [&](int lo, int hi, int N) {
            for (int k = 0; k < N; ++k) {
                const double t = (N == 1) ? 0.5
                                          : double(k) / double(N - 1);
                probes.push_back(lo + static_cast<int>(t * double(hi - lo)));
            }
        };
        if (n_main > 0) fill(main_lo, main_hi, n_main);
        if (n_asym > 0) fill(asym_lo, asym_hi, n_asym);

        // l_sigma[s] is the orbital-side l for index s in [0, n_sigma).
        std::vector<int> l_sigma(n_sigma);
        for (int s = 0; s < n_sigma; ++s) {
            int l, m; angular::idx_to_lm(s, l, m); l_sigma[s] = l;
            (void)l;  // l_sigma unused by current residual but kept for parity
        }

        std::cout << "\n--- [ik=" << ik << "  k="
                  << std::sqrt(2.0 * E_kin)
                  << "]  PDE residual probe (after BP) ---\n";
        std::cout << "    n  |    r       |  |LHS|_max   |  |RHS|_max   |"
                     "  |residual|  |  rel\n";

        double worst_rel = 0.0;
        int    worst_n   = -1;
        for (int n : probes) {
            if (n < 2 || n > Nr_full - 3) continue;
            const double r = bundle.params.r_min + double(n) * h;

            // ψ'' via 5-point centered stencil.
            const Eigen::MatrixXd p_m2 = BP.get_psi(std::size_t(n - 2));
            const Eigen::MatrixXd p_m1 = BP.get_psi(std::size_t(n - 1));
            const Eigen::MatrixXd p_00 = BP.get_psi(std::size_t(n));
            const Eigen::MatrixXd p_p1 = BP.get_psi(std::size_t(n + 1));
            const Eigen::MatrixXd p_p2 = BP.get_psi(std::size_t(n + 2));
            const Eigen::MatrixXd psi_pp =
                (-p_m2 + 16.0 * p_m1 - 30.0 * p_00
                 + 16.0 * p_p1 - p_p2) * (h2_inv / 12.0);

            const Eigen::MatrixXd& V_n = pot.get(std::size_t(n));
            Eigen::MatrixXd lhs = psi_pp + 2.0 * E_kin * p_00 - 2.0 * V_n * p_00;

            // RHS = -2α/r · Σ G_{μ λ σ} · χ_λ(n) · f_σ(n)
            Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(N_psi, N_psi);
            const Eigen::MatrixXd& f_n   = BP.get_f(std::size_t(n));
            const auto&            chi_n = bundle.chi[std::size_t(n)];
            const double inv_r = (r > 0.0) ? 1.0 / r : 0.0;
            for (const auto& g : bundle.G_coeff) {
                const int mu = g.a, lambda = g.b, sigma = g.c;
                if (mu >= N_psi || sigma >= n_sigma) continue;
                if (lambda >= chi_n.cols())          continue;
                for (int i = 0; i < n_occ; ++i) {
                    const double chi_v = chi_n(i, lambda);
                    const int    f_idx = i * n_sigma + sigma;
                    for (int j = 0; j < N_psi; ++j) {
                        rhs(mu, j) += -2.0 * alpha * g.value * chi_v *
                                      f_n(f_idx, j) * inv_r;
                    }
                }
            }

            const double lhs_max = lhs.cwiseAbs().maxCoeff();
            const double rhs_max = rhs.cwiseAbs().maxCoeff();
            const double res_max = (lhs - rhs).cwiseAbs().maxCoeff();
            const double scale   = std::max({lhs_max, rhs_max, 1e-30});
            const double rel     = res_max / scale;
            if (lhs_max > 1e-8 && rel > worst_rel) {
                worst_rel = rel; worst_n = n;
            }
            std::cout << "    " << std::setw(5) << n
                      << " | "  << std::setw(10) << r
                      << " | "  << std::setw(11) << lhs_max
                      << " | "  << std::setw(11) << rhs_max
                      << " | "  << std::setw(11) << res_max
                      << " | "  << std::setw(8)  << rel << "\n";
        }
        std::cout << "    worst rel = " << worst_rel
                  << "  at n = " << worst_n
                  << "  (lhs threshold > 1e-8)\n";
    }

    // A, B.
    stage_log("AB: fitting A, B from psi tail in asymptotic buffer");
    AsymptoticAmplitudes AA(bundle.params, BP);
    AsymptoticAmplitudes::Config aac;
    aac.verbose = true;
    aac.allow_closed_channels = allow_closed_channels;
    auto res_AB = [&]{
        ProfileScope _s(bench, "AsymptoticAmplitudes::extract");
        return AA.extract(aac);
    }();

    // Dipole (six slices).
    stage_log(use_gpu
        ? "Dipole: computing 6 reduced matrix elements (L+V gauge x x/y/z pol)  (GPU: 1 DGEMM/ir)"
        : "Dipole: computing 6 reduced matrix elements (L+V gauge x x/y/z pol)");
    DipoleMatrixElement DME(bundle.params, BP, chi_init_homo, occ);
    DipoleMatrixElement::Config dmc; dmc.verbose = true;
    // GPU offload of the per-ir GEMV loop -- single --use-gpu flag controls
    // FRP / BP / Sinv / DME together (matches the rest of the pipeline so
    // changing the device toggles every offload point at once).  Accuracy
    // floor ε_mach × N (~1e-13 relative on DGEMM output); validated by
    // test_gpu_dme on the H2O fixture.
    dmc.use_gpu              = use_gpu;

    DipoleEnergyPayload payload;
    payload.ik               = ik;
    payload.A                = res_AB.A;
    payload.B                = res_AB.B;
    payload.fit_residual_rel = res_AB.fit_residual_rel;
    payload.K_symmetry_err   = res_AB.K_symmetry_err;

    // Batched 6-call: one ir-pass, ψ_n read once across all six gauge×pol
    // slots, b_overlap rebuilt once.  Byte-identical to the legacy 6x
    // compute() loop with cached b_overlap (gated by
    // test_dipole_compute_six on the H2O fixture).  At L=100 / DISK ψ
    // production this drops the dipole wall from ~7-8 h to ~1 h.
    DipoleMatrixElement::Stats dip_sum{};
    std::array<DipoleResult, 6> six_results;
    {   ProfileScope _s(bench, "DipoleMatrixElement::compute_six",
                        /*uses_gpu=*/dmc.use_gpu);
        six_results = DME.compute_six(res_AB.A, res_AB.B, dmc);
        dip_sum = DME.stats();
    }
    for (int i = 0; i < 6; ++i) {
        const DipoleResult& r = six_results[i];
        DipoleSlice& s = payload.slices[i];
        s.gauge          = r.gauge;
        s.pol            = r.pol;
        s.D_reduced      = r.D_reduced;
        s.D_reduced_raw  = r.D_reduced_raw;
        s.d_raw          = r.d_raw;
        s.d_correction   = r.d_correction;
        s.partial_sigma  = r.partial_sigma;
    }
    // The b_overlap is identical across the 6 slots; pick the first
    // populated one for the payload.
    for (const auto& r : six_results) {
        if (r.b_overlap.size() > 0) { payload.b_overlap = r.b_overlap; break; }
    }
    // Sub-stage breakdown of the 6 dipole compute calls (summed).
    {
        const std::size_t rss = current_peak_rss_bytes();
        bench.add("Dip::angular_table", dip_sum.t_angular_table_ns, rss);
        bench.add("Dip::xi_build",      dip_sum.t_xi_build_ns,      rss);
        bench.add("Dip::psi_fetch",     dip_sum.t_psi_fetch_ns,     rss);
        bench.add("Dip::integrand",     dip_sum.t_integrand_ns,     rss);
        bench.add("Dip::simpson",       dip_sum.t_simpson_ns,       rss);
        bench.add("Dip::M_lu",          dip_sum.t_M_lu_ns,          rss);
        bench.add("Dip::M_apply",       dip_sum.t_M_apply_ns,       rss);
        bench.add("Dip::ortho",         dip_sum.t_ortho_ns,         rss);
    }

    {   ProfileScope _s(bench, "DipoleWriter::write_energy");
        writer.write_energy(payload); }

    const double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "  ik=" << ik
              << "  k=" << std::fixed << std::setprecision(4) << std::sqrt(2.0*E_kin)
              << "  E=" << E_kin
              << "  fit_res=" << std::scientific << std::setprecision(10) << res_AB.fit_residual_rel
              << "  K_sym=" << res_AB.K_symmetry_err
              << std::defaultfloat << "  (" << std::setprecision(1) << dt << " s)\n";
    std::cout << std::scientific << std::setprecision(10);
    std::cerr << std::scientific << std::setprecision(10);
}
