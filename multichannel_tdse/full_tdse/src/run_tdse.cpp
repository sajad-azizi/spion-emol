// run_tdse.cpp -- production driver for the recipe full-TDSE Route A.
//
// CLI knobs (all optional, defaults match the recipe's reference point):
//
//   --L 50000              radial box length in a_0
//   --dr 0.5               radial grid spacing in a_0
//   --N_grid 100000        derived from L/dr if both omitted
//   --E_cut_open_kHz 300   cutoff above each open block's threshold
//   --E_cut_m5_kHz 1000    cutoff above M_F=-5 threshold (recipe ramp: 5e6, 1e7, 2e7, 5e7 kHz)
//   --omega_kHz 80.896     RF photon (= 8 E_b at recipe params)
//   --Omega_R_kHz 179      Rabi frequency (recipe operating point)
//   --tau_us 30            Gaussian pulse 1/e half-width
//   --T_us 90              propagation half-window (so [−T, +T])
//   --dt_us 0.01           time step
//   --delta_E_kHz 0.5      Gaussian smoothing width for dP/dE readout
//   --out_prefix run       output file prefix
//
// Outputs:
//   {prefix}_summary.txt           parameters + block totals + integrated peaks
//   {prefix}_block_M{MF}_dPdE.csv  E_kHz, dPdE_perHa  per block
//
// To run on LRZ:
//   module load intel mkl
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
//   ./build/run_tdse --L 100000 --dr 0.5 --E_cut_m5_kHz 5000000 ...
#include "BlockEigenstates.hpp"
#include "PooledBasis.hpp"
#include "PooledTDSE.hpp"
#include "SpectralReadout.hpp"
#include "Rb85Spin.hpp"
#include "BasisCache.hpp"
#include "PerturbationTheory.hpp"
#include "Common.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Knobs {
    double L_a0           = 50000.0;
    double dr_a0          = 0.5;
    int    N_grid         = 0;          // 0 => derive from L/dr
    double E_cut_open_kHz = 300.0;
    double E_cut_m5_kHz   = 1000.0;
    double omega_kHz      = 80.896;
    double Omega_R_kHz    = 179.0;
    double tau_us         = 30.0;
    double T_us           = 90.0;
    double dt_us          = 0.01;
    double delta_E_kHz    = 0.5;
    std::string out_prefix = "run";
    std::string basis_cache_dir = "";   // empty = caching off
    int    omp_threads     = 0;          // 0 = auto (OMP_NUM_THREADS / nproc)
    // Comma-separated list of M_F blocks to include in the basis; default
    // is the full 4-block ladder.  Use "-4,-3,-2" for the closed-channel
    // ablation test (drop the M_F=-5 virtual block).  Order does not
    // matter: we sort ascending so M_F=-4 (halo) lands at a known index.
    std::string blocks_csv = "-5,-4,-3,-2";
};

double get_double(int& i, int argc, char** argv, const char* flag) {
    if (i + 1 >= argc) { std::fprintf(stderr, "%s requires a value\n", flag); std::exit(1); }
    return std::atof(argv[++i]);
}
int get_int(int& i, int argc, char** argv, const char* flag) {
    if (i + 1 >= argc) { std::fprintf(stderr, "%s requires a value\n", flag); std::exit(1); }
    return std::atoi(argv[++i]);
}
std::string get_str(int& i, int argc, char** argv, const char* flag) {
    if (i + 1 >= argc) { std::fprintf(stderr, "%s requires a value\n", flag); std::exit(1); }
    return std::string(argv[++i]);
}

Knobs parse(int argc, char** argv) {
    Knobs k;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if      (!std::strcmp(a, "--L"))             k.L_a0 = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--dr"))            k.dr_a0 = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--N_grid"))        k.N_grid = get_int(i, argc, argv, a);
        else if (!std::strcmp(a, "--E_cut_open_kHz"))k.E_cut_open_kHz = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--E_cut_m5_kHz")) k.E_cut_m5_kHz   = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--omega_kHz"))    k.omega_kHz = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--Omega_R_kHz"))  k.Omega_R_kHz = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--tau_us"))       k.tau_us = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--T_us"))         k.T_us = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--dt_us"))        k.dt_us = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--delta_E_kHz"))  k.delta_E_kHz = get_double(i, argc, argv, a);
        else if (!std::strcmp(a, "--out_prefix"))   k.out_prefix = get_str(i, argc, argv, a);
        else if (!std::strcmp(a, "--basis_cache_dir")) k.basis_cache_dir = get_str(i, argc, argv, a);
        else if (!std::strcmp(a, "--omp_threads"))  k.omp_threads = get_int(i, argc, argv, a);
        else if (!std::strcmp(a, "--blocks"))       k.blocks_csv = get_str(i, argc, argv, a);
        else { std::fprintf(stderr, "unknown flag: %s\n", a); std::exit(1); }
    }
    if (k.N_grid <= 0) k.N_grid = static_cast<int>(k.L_a0 / k.dr_a0);
    return k;
}

void write_dPdE(const std::string& path,
                const std::vector<double>& E_kHz,
                const std::vector<double>& dPdE_perHa) {
    std::ofstream f(path);
    f << "E_kHz,dP_dE_per_kHz\n";
    for (size_t i = 0; i < E_kHz.size(); ++i)
        f << E_kHz[i] << "," << dPdE_perHa[i] / AU::Hartree_in_kHz << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace mc_tdse;
    using clock_t = std::chrono::steady_clock;

    // CRITICAL: when stdout is redirected to a file (e.g. by SLURM or
    // `> log.txt`), it becomes FULLY BUFFERED.  A long-running build
    // can then produce no visible output for HOURS even though the run
    // is making progress.  Force line-buffering so each \n flushes.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);

    Knobs k = parse(argc, argv);

#ifdef _OPENMP
    if (k.omp_threads > 0) omp_set_num_threads(k.omp_threads);
    const int n_threads = omp_get_max_threads();
    std::printf("[run_tdse] OpenMP threads = %d\n", n_threads);
#else
    std::printf("[run_tdse] OpenMP not enabled (single-thread propagation)\n");
#endif

    std::printf("[run_tdse] L=%.1f a_0  dr=%.3f  N_grid=%d\n",
                k.L_a0, k.dr_a0, k.N_grid);
    std::printf("           E_cut_open=%.1f kHz  E_cut_m5=%.1f kHz\n",
                k.E_cut_open_kHz, k.E_cut_m5_kHz);
    std::printf("           ω=%.4f kHz  Ω_R=2π·%.1f kHz  τ=%.1f μs\n",
                k.omega_kHz, k.Omega_R_kHz, k.tau_us);
    std::printf("           T=±%.1f μs  dt=%.4f μs  δE=%.4f kHz\n",
                k.T_us, k.dt_us, k.delta_E_kHz);

    // ---- upfront sanity: estimate level count + memory per block ----
    // Free-particle-in-box level count:  n_max = sqrt(2μ E_cut) · L / π
    // Per-state storage = N_grid · N_ch · 8 B.  Print BEFORE any
    // expensive work so the user can abort if the budget is wrong.
    {
        const double mu_au = AU::Rb85::mu_au;        // ~77392 m_e
        auto n_levels = [&](double E_cut_kHz_above_thresh) {
            const double E_au = AU::kHz_to_au(E_cut_kHz_above_thresh);
            if (E_au <= 0.0) return 0;
            return static_cast<int>(std::sqrt(2.0 * mu_au * E_au) * k.L_a0 / M_PI);
        };
        const int n_M5 = n_levels(k.E_cut_m5_kHz);
        const int n_open = n_levels(k.E_cut_open_kHz);
        const double per_state_MB = (double)k.N_grid * 2.0 * 8.0 / (1024.0 * 1024.0);
        const double total_GB =
            (n_M5 + 3 * n_open + 1)       // M_F=-5 + 3 open blocks + halo
            * per_state_MB / 1024.0;
        std::printf("[run_tdse] level-count estimate (free-particle-in-box):\n");
        std::printf("           M_F=-5  n_levels ≈ %d   (above threshold by %g kHz)\n",
                    n_M5, k.E_cut_m5_kHz);
        std::printf("           open    n_levels ≈ %d  per block\n", n_open);
        std::printf("           per-state grid storage = %.2f MB\n", per_state_MB);
        std::printf("           total state storage    = %.1f GB\n", total_GB);
        if (total_GB > 256.0) {
            std::printf("\n[run_tdse] WARNING: estimated %.0f GB exceeds typical LRZ "
                        "node RAM (256 GB).\n", total_GB);
            std::printf("[run_tdse] Consider stepping down: smaller --L, smaller "
                        "--E_cut_m5_kHz, or both.\n");
            std::printf("[run_tdse] Recipe convergence ramp: start at "
                        "--L 100000 --E_cut_m5_kHz 1000000 (1 MHz), then 100 MHz, then 1 GHz.\n\n");
        }
    }

    Rb85Spin spin(155.04);

    BlockBuildOptions base;
    base.N_grid = k.N_grid;
    base.dr_a0  = k.dr_a0;

    // Parse --blocks "-5,-4,-3,-2" → sorted ascending vector of M_F values.
    std::vector<int> MFs;
    {
        std::string s = k.blocks_csv;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, comma - pos);
            if (!tok.empty()) MFs.push_back(std::atoi(tok.c_str()));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        std::sort(MFs.begin(), MFs.end());
        MFs.erase(std::unique(MFs.begin(), MFs.end()), MFs.end());
        if (std::find(MFs.begin(), MFs.end(), -4) == MFs.end()) {
            std::fprintf(stderr,
                "ERROR: --blocks must include -4 (halo block).\n");
            return 1;
        }
        std::printf("[run_tdse] basis blocks (from --blocks):");
        for (int m : MFs) std::printf(" %+d", m);
        std::printf("\n");
    }
    std::vector<BlockBuildOptions> per_block;
    per_block.reserve(MFs.size());
    for (int m : MFs) {
        BlockBuildOptions o = base;
        if (m == -5) {
            // Virtual block: large E_cut covering the GHz-scale Zeeman shift.
            o.E_max_kHz_above_threshold = k.E_cut_m5_kHz;
        } else {
            o.E_max_kHz_above_threshold = k.E_cut_open_kHz;
        }
        if (m == -4) o.use_analytic_halo = true;
        per_block.push_back(o);
    }

    // ---- Try the basis cache first (saves the ~hours-long build phase
    // when only pulse parameters change between runs).
    auto t0 = clock_t::now();
    PooledBasis pb;
    bool cache_hit = false;
    std::string cache_key, canon;
    if (!k.basis_cache_dir.empty()) {
        canon     = basis_cache_canonical_string(MFs, per_block, 155.04);
        cache_key = basis_cache_key(MFs, per_block, 155.04);
        std::printf("[run_tdse] basis cache key=%s  dir=%s\n",
                    cache_key.c_str(), k.basis_cache_dir.c_str());
        cache_hit = try_load_pooled_basis(&pb, k.basis_cache_dir,
                                          cache_key, canon);
        if (cache_hit) {
            std::printf("[run_tdse] basis CACHE HIT — skipping build\n");
        } else {
            std::printf("[run_tdse] basis CACHE MISS — building from scratch\n");
        }
    }
    if (!cache_hit) {
        pb = build_pooled_basis(MFs, per_block, spin);
        if (!k.basis_cache_dir.empty()) {
            try {
                save_pooled_basis(pb, k.basis_cache_dir, cache_key, canon);
                std::printf("[run_tdse] basis CACHED to %s/%s.bin\n",
                            k.basis_cache_dir.c_str(), cache_key.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[run_tdse] basis cache save failed: %s\n",
                             e.what());
            }
        }
    }
    auto dt_build_s = std::chrono::duration<double>(clock_t::now() - t0).count();
    std::printf("[run_tdse] basis built in %.1f s.  N_total=%d.  block sizes:",
                dt_build_s, pb.N_total);
    for (size_t kk = 0; kk < pb.blocks.size(); ++kk) {
        std::printf("  (M_F=%+d : %d)",
                    pb.block_MFs[kk], pb.blocks[kk].n_states());
    }
    std::printf("\n");

    // Halo lives in the M_F=-4 block.  Find that block's flat-index offset
    // (which is no longer always block_offset[1] when --blocks shrinks).
    Eigen::VectorXcd c0 = Eigen::VectorXcd::Zero(pb.N_total);
    int halo_block = -1;
    for (size_t kk = 0; kk < pb.block_MFs.size(); ++kk) {
        if (pb.block_MFs[kk] == -4) { halo_block = static_cast<int>(kk); break; }
    }
    if (halo_block < 0) {
        std::fprintf(stderr, "ERROR: M_F=-4 block missing from basis.\n");
        return 1;
    }
    const int halo_idx = pb.block_offset[halo_block];
    c0(halo_idx) = 1.0;

    PooledTDSEConfig cfg;
    cfg.chi        = make_gaussian(AU::us_to_au(k.tau_us), 0.0);
    cfg.omega_au   = AU::kHz_to_au(k.omega_kHz);
    cfg.Omega_R_au = 2.0 * M_PI * AU::kHz_to_au(k.Omega_R_kHz);
    cfg.t_start    = AU::us_to_au(-k.T_us);
    cfg.t_end      = AU::us_to_au(+k.T_us);
    cfg.dt         = AU::us_to_au(k.dt_us);

    auto t1 = clock_t::now();
    PooledTDSEStats stats;
    Eigen::VectorXcd c_T = propagate_pooled(pb, c0, cfg, &stats);
    auto dt_prop_s = std::chrono::duration<double>(clock_t::now() - t1).count();
    std::printf("[run_tdse] TDSE propagated in %.1f s   K_avg=%d  max_taylor_res=%.2e\n",
                dt_prop_s, stats.K_avg, stats.max_err);

    // ---- Closed-form PT (1st + 2nd order) for the same Gaussian pulse ----
    // Gives a per-state predicted population that we write alongside the
    // TDSE result.  Comparison TDSE-vs-PT is the cleanest PT-validity check.
    auto t_pt0 = clock_t::now();
    pt::PTConfig ptc;
    ptc.initial_block      = halo_block;                 // M_F=-4 (halo block)
    ptc.initial_state      = 0;                          // halo eigenstate
    ptc.tau_au             = AU::us_to_au(k.tau_us);
    ptc.t_center_au        = 0.0;                        // matches make_gaussian above
    ptc.compute_2nd_order  = true;
    auto pt_res = pt::compute_pt(pb, cfg, ptc);
    auto dt_pt_s = std::chrono::duration<double>(clock_t::now() - t_pt0).count();
    std::printf("[run_tdse] PT (1st+2nd order) computed in %.1f s\n", dt_pt_s);

    const double norm_T = c_T.norm();
    const double err_unitary = std::fabs(norm_T - 1.0);
    auto P = block_populations(pb, c_T);
    double P_zepe_total = 0.0;
    // ZEPE continuum = M_F=-4 states EXCLUDING the halo (eigenstate 0).
    {
        const int hb_lo = pb.block_offset[halo_block] + 1;          // skip halo
        const int hb_hi = pb.block_offset[halo_block]
                          + pb.blocks[halo_block].n_states();
        for (int a = hb_lo; a < hb_hi; ++a) P_zepe_total += std::norm(c_T(a));
    }
    const double P_halo = std::norm(c_T(halo_idx));

    // ---- dP/dE per block ----
    SpectralOptions so;
    so.delta_E_au = AU::kHz_to_au(k.delta_E_kHz);
    for (size_t kk = 0; kk < pb.blocks.size(); ++kk) {
        // Make a per-block sample grid spanning [E_min - 5 δE, E_max + 5 δE].
        const auto& blk = pb.blocks[kk];
        if (blk.n_states() == 0) continue;
        double E_lo = blk.E_au.front();
        double E_hi = blk.E_au.back();
        E_lo -= 5.0 * so.delta_E_au;
        E_hi += 5.0 * so.delta_E_au;
        const int N_E = 4000;
        so.E_grid_au.assign(N_E, 0.0);
        std::vector<double> E_kHz(N_E, 0.0);
        for (int i = 0; i < N_E; ++i) {
            so.E_grid_au[i] = E_lo + (E_hi - E_lo) * i / (N_E - 1);
            E_kHz[i] = AU::au_to_kHz(so.E_grid_au[i]);
        }
        auto y = dPdE_block(pb, c_T, static_cast<int>(kk), so);
        char path[256];
        std::snprintf(path, sizeof path, "%s_block_M%+d_dPdE.csv",
                      k.out_prefix.c_str(), pb.block_MFs[kk]);
        write_dPdE(path, E_kHz, y);
        std::printf("[run_tdse] wrote %s   N_E=%d states=%d\n", path, N_E, blk.n_states());

        // ---- per-state populations (raw |b_α(T)|², for re-smoothing) ----
        // Saves the discrete spectrum in BOTH lab-frame and block-relative
        // energy.  Postprocessing can re-smooth with any δE without
        // re-running the TDSE.
        char states_path[256];
        std::snprintf(states_path, sizeof states_path,
                      "%s_block_M%+d_states.csv",
                      k.out_prefix.c_str(), pb.block_MFs[kk]);
        std::ofstream sf(states_path);
        sf << "n,E_kHz_lab,E_kHz_block,prob,prob_pt\n";
        const double E_th_kHz_lab = AU::au_to_kHz(pb.E_th_au[kk]);
        const int o = pb.block_offset[kk];
        for (int n = 0; n < blk.n_states(); ++n) {
            const double E_lab    = AU::au_to_kHz(pb.E_au[o + n]);
            const double E_block  = E_lab - E_th_kHz_lab;
            const double prob     = std::norm(c_T(o + n));
            const double prob_pt  = pt_res.prob_pt(o + n);
            sf << n << "," << E_lab << "," << E_block << ","
               << prob << "," << prob_pt << "\n";
        }
        std::printf("[run_tdse] wrote %s\n", states_path);
    }

    // ---- Summary ----
    char summ_path[256];
    std::snprintf(summ_path, sizeof summ_path, "%s_summary.txt", k.out_prefix.c_str());
    std::ofstream s(summ_path);
    s << "# parameters\n"
      << "L_a0=" << k.L_a0 << "\n"
      << "dr_a0=" << k.dr_a0 << "\n"
      << "N_grid=" << k.N_grid << "\n"
      << "E_cut_open_kHz=" << k.E_cut_open_kHz << "\n"
      << "E_cut_m5_kHz=" << k.E_cut_m5_kHz << "\n"
      << "omega_kHz=" << k.omega_kHz << "\n"
      << "Omega_R_kHz=" << k.Omega_R_kHz << "\n"
      << "tau_us=" << k.tau_us << "\n"
      << "T_us=" << k.T_us << "\n"
      << "dt_us=" << k.dt_us << "\n"
      << "delta_E_kHz=" << k.delta_E_kHz << "\n"
      << "# diagnostics\n"
      << "build_time_s=" << dt_build_s << "\n"
      << "prop_time_s=" << dt_prop_s << "\n"
      << "K_avg=" << stats.K_avg << "\n"
      << "max_taylor_res=" << stats.max_err << "\n"
      << "norm_final=" << norm_T << "\n"
      << "err_unitary=" << err_unitary << "\n"
      << "# block sizes + thresholds (kHz, recipe origin = M_F=-4 entrance)\n";
    for (size_t kk = 0; kk < pb.blocks.size(); ++kk) {
        s << "n_states_M" << pb.block_MFs[kk] << "=" << pb.blocks[kk].n_states() << "\n";
        s << "E_th_M" << pb.block_MFs[kk] << "_kHz="
          << AU::au_to_kHz(pb.E_th_au[kk]) << "\n";
    }
    s << "# block-totals (Σ_α∈block |b_α(T)|²)\n";
    for (size_t kk = 0; kk < pb.blocks.size(); ++kk)
        s << "P_M" << pb.block_MFs[kk] << "=" << P[kk] << "\n";
    s << "# halo / ZEPE / 1γ / 2γ / virtual block totals\n"
      << "P_halo=" << P_halo << "\n"
      << "P_M-4_continuum=" << P_zepe_total << "\n"
      << "P_M-3_total=" << P[2] << "\n"
      << "P_M-2_total=" << P[3] << "\n"
      << "P_M-5_total=" << P[0] << "\n";

    std::printf("[run_tdse] wrote %s\n", summ_path);
    std::printf("[run_tdse] DONE.  ‖c‖−1=%.2e   P_halo=%.4e   P_(-3)=%.4e   "
                "P_(-2)=%.4e   P_(-5)=%.4e\n",
                err_unitary, P_halo, P[2], P[3], P[0]);
    return 0;
}
