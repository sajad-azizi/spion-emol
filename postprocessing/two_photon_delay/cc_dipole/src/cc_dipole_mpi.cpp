// cc_dipole_mpi.cpp -- MPI + OpenMP distributed driver for the
// continuum-continuum dipole matrix elements <ψ_κ|r·F|ψ_ν> of the C8F8
// (or similarly-sized) photodetachment two-photon RABBITT pipeline.
//
// What it does
// ------------
// Same math as cc_dipole_driver --reuse-kappa-chunks:
//     cc_raw[β, α] = ∫ dr · r · (ψ_κ(r)^T · A^q · ψ_ν(r))[β, α]
// with the sparse A^q angular Gaunt table from cc_dipole::make_accum_state.
// Bit-equivalent to the existing single-node driver (validated by
// test_cc_dipole_bruteforce, test_cc_dipole_symmetry, test_cc_dipole_two_energy).
//
// The DIFFERENCE is the distribution layer:
//   * MPI rank r handles κ's at strided positions r, r + size, r + 2·size, …
//   * Each rank reads ψ_κ once + ψ_ν chunks in DISK mode (Lustre stream).
//   * Each rank writes a per-κ HDF5 file: <out-dir>/cc_dipole_kKKKK.h5.
//   * A small Python post-processor merges them into a single
//     cc_dipole.h5 compatible with phase_a_assembler.
//
// Threading: each MPI rank is single-threaded at the MPI level
// (MPI_THREAD_FUNNELED).  MKL is locked at OMP_NUM_THREADS via
// mkl_set_num_threads + mkl_set_dynamic(0) so the dense N×N×N GEMM
// inside the inner ir loop uses ALL cores allocated to the rank.  This
// is the most important knob -- the original single-node run was
// observed ~8× slower than the I/O+compute floor, which is exactly the
// signature of MKL silently running at 1/8 the thread count.
//
// SLURM usage  (40 nodes × 112 threads, 1 MPI rank per node):
//     #SBATCH --nodes=40
//     #SBATCH --ntasks-per-node=1
//     #SBATCH --cpus-per-task=112
//     export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
//     srun cc_dipole_mpi \
//         --psi_dir_prefix $SCRATCH/psi_${HASH}_ik \
//         --ik_kappa $KAPPA_LIST  --ik_nu $NU_LIST \
//         --pol all  --chunk-size 100 \
//         --out-dir $WORK/cc_dipole_mpi
//
// After srun finishes, merge per-κ files:
//     python3 merge_cc_dipole_h5.py \
//         --in-dir $WORK/cc_dipole_mpi \
//         --output $WORK/cc_dipole.h5
//
// All ranks parse argv independently (no MPI broadcast of args -- argv
// is the same across ranks under any standard MPI launcher).
#include "CCDipole.hpp"

#include "scatt/PotentialStorage.hpp"
#include "scatt/SolverParams.hpp"
#include "scatt/DipoleMatrixElement.hpp"

#include <hdf5.h>
#include <mpi.h>

#ifdef _OPENMP
#  include <omp.h>
#endif
#ifdef EIGEN_USE_MKL_ALL
#  include <mkl_service.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct PsiManifest {
    double energy   = 0.0;
    int    Nr       = 0;
    double dr       = 0.0;
    int    N_psi    = 0;
    int    N_f      = 0;
    int    l_cont   = 0;
    int    l_exch   = 0;
    int    n_occ    = 0;
    int    n_trans  = 0;
    int    n_keep_lo = 0;
    int    n_keep_hi = -1;
    std::string raw;
};

PsiManifest parse_manifest(const fs::path& dir) {
    const fs::path mf_path = dir / "manifest.txt";
    std::ifstream in(mf_path);
    if (!in.good())
        throw std::runtime_error("cc_dipole_mpi: cannot open " + mf_path.string());
    std::stringstream buf;  buf << in.rdbuf();
    PsiManifest m;
    m.raw = buf.str();
    auto pull = [&m](const char* key, auto setter) {
        std::regex re(std::string(key) + R"(=([\-+0-9.eE]+))");
        std::smatch sm;
        if (std::regex_search(m.raw, sm, re)) {
            std::string v = sm[1];
            std::stringstream ss(v);
            decltype(setter(0)) val{};
            ss >> val;
            setter(val);
        } else {
            throw std::runtime_error(
                "cc_dipole_mpi: manifest missing key " + std::string(key));
        }
    };
    pull("energy",  [&](double v) -> double { m.energy = v;  return v; });
    pull("Nr",      [&](int v) -> int    { m.Nr = v;       return v; });
    pull("dr",      [&](double v) -> double { m.dr = v;      return v; });
    pull("Npsi",    [&](int v) -> int    { m.N_psi = v;    return v; });
    pull("Nf",      [&](int v) -> int    { m.N_f = v;      return v; });
    pull("l_cont",  [&](int v) -> int    { m.l_cont = v;   return v; });
    pull("l_exch", [&](int v) -> int     { m.l_exch = v;   return v; });
    pull("n_occ",   [&](int v) -> int    { m.n_occ = v;    return v; });
    pull("n_trans", [&](int v) -> int    { m.n_trans = v;  return v; });
    {
        std::regex re(R"(keep=([0-9]+)\.\.([0-9]+))");
        std::smatch sm;
        if (std::regex_search(m.raw, sm, re)) {
            m.n_keep_lo = std::stoi(sm[1]);
            m.n_keep_hi = std::stoi(sm[2]);
        } else {
            throw std::runtime_error(
                "cc_dipole_mpi: manifest missing keep=lo..hi");
        }
    }
    return m;
}

std::vector<int> parse_ik_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) out.push_back(std::stoi(token));
    }
    return out;
}

std::string ik_to_str(int ik) {
    char buf[16]; std::snprintf(buf, sizeof(buf), "%04d", ik);
    return std::string(buf);
}

void write_h5_attr_double(hid_t parent, const char* name, double v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(parent, name, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a); H5Sclose(sp);
}
void write_h5_attr_int(hid_t parent, const char* name, int v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(parent, name, H5T_NATIVE_INT, sp,
                          H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT, &v);
    H5Aclose(a); H5Sclose(sp);
}
void write_h5_matrix_2d(hid_t parent, const char* name,
                         const Eigen::MatrixXd& M) {
    hsize_t dims[2] = { static_cast<hsize_t>(M.rows()),
                        static_cast<hsize_t>(M.cols()) };
    hid_t sp = H5Screate_simple(2, dims, nullptr);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        M_rm = M;
    hid_t ds = H5Dcreate2(parent, name, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, M_rm.data());
    H5Dclose(ds); H5Sclose(sp);
}

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: mpirun -n N %s --psi_dir_prefix <prefix> --ik_kappa <list>\n"
        "    [--ik_nu <list|all>] [--pol x|y|z|all]\n"
        "    [--chunk-size N] (default 100)\n"
        "    [--out-dir <dir>] (default cc_dipole_mpi)\n"
        "\n"
        "Each rank handles κ's at strided positions and writes a per-κ HDF5\n"
        "file <out-dir>/cc_dipole_kKKKK.h5.  Use merge_cc_dipole_h5.py to\n"
        "combine into a single cc_dipole.h5.\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    // -------------------- arg parse (all ranks) ----------------------------
    std::string psi_dir_prefix;
    std::string ik_kappa_str, ik_nu_str = "all";
    std::string out_dir = "cc_dipole_mpi";
    std::string pol_str = "all";
    int chunk_size = 100;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " requires arg");
            return argv[++i];
        };
        if      (a == "--psi_dir_prefix") psi_dir_prefix = next();
        else if (a == "--ik_kappa")       ik_kappa_str   = next();
        else if (a == "--ik_nu")          ik_nu_str      = next();
        else if (a == "--out-dir")        out_dir        = next();
        else if (a == "--pol")            pol_str        = next();
        else if (a == "--chunk-size")     chunk_size     = std::stoi(next());
        else if (a == "-h" || a == "--help") {
            if (rank == 0) usage(argv[0]);
            MPI_Finalize(); return 0;
        }
        else {
            if (rank == 0) std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
            if (rank == 0) usage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    if (psi_dir_prefix.empty() || ik_kappa_str.empty()) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize(); return 1;
    }

    auto ik_kappa = parse_ik_list(ik_kappa_str);
    auto ik_nu    = (ik_nu_str == "all") ? ik_kappa : parse_ik_list(ik_nu_str);
    std::vector<scatt::Polarization> pols;
    if      (pol_str == "all") pols = {scatt::Polarization::X,
                                       scatt::Polarization::Y,
                                       scatt::Polarization::Z};
    else if (pol_str == "x")   pols = {scatt::Polarization::X};
    else if (pol_str == "y")   pols = {scatt::Polarization::Y};
    else if (pol_str == "z")   pols = {scatt::Polarization::Z};
    else {
        if (rank == 0) std::fprintf(stderr, "bad pol: %s\n", pol_str.c_str());
        MPI_Finalize(); return 1;
    }

    // -------------------- threading lockdown ------------------------------
    // The most important config knob -- ensures the dense N×N×N GEMM
    // inside accumulate_cc_range uses ALL cores the rank was given.
#ifdef _OPENMP
    const int omp_threads = omp_get_max_threads();
#else
    const int omp_threads = 1;
#endif
#ifdef EIGEN_USE_MKL_ALL
    mkl_set_num_threads(omp_threads);
    mkl_set_dynamic(0);
#endif

    if (rank == 0) {
        std::printf("[cc_dipole_mpi] %d MPI rank(s) × %d thread(s)/rank\n",
                    size, omp_threads);
        std::printf("[cc_dipole_mpi] MPI thread support: requested FUNNELED, got %d\n",
                    provided);
        std::printf("[cc_dipole_mpi] |ik_kappa|=%zu  |ik_nu|=%zu  pols=%zu\n",
                    ik_kappa.size(), ik_nu.size(), pols.size());
    }

    // -------------------- manifests (all ranks; cheap) ---------------------
    std::vector<int> all_ik = ik_kappa;
    for (int n : ik_nu) {
        if (std::find(all_ik.begin(), all_ik.end(), n) == all_ik.end())
            all_ik.push_back(n);
    }
    std::vector<PsiManifest> manifests(all_ik.size());
    PsiManifest ref;
    bool ref_set = false;
    try {
        for (size_t j = 0; j < all_ik.size(); ++j) {
            const fs::path dir = psi_dir_prefix + ik_to_str(all_ik[j]);
            manifests[j] = parse_manifest(dir);
            if (!ref_set) { ref = manifests[j]; ref_set = true; }
            else if (manifests[j].Nr        != ref.Nr
                  || manifests[j].N_psi     != ref.N_psi
                  || manifests[j].n_keep_lo != ref.n_keep_lo
                  || manifests[j].n_keep_hi != ref.n_keep_hi) {
                if (rank == 0) std::fprintf(stderr,
                    "ERROR: manifest mismatch at ik=%d\n", all_ik[j]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    auto manifest_of = [&](int ik) -> const PsiManifest& {
        auto it = std::find(all_ik.begin(), all_ik.end(), ik);
        return manifests[std::distance(all_ik.begin(), it)];
    };
    if (rank == 0) {
        std::printf("[cc_dipole_mpi] ref manifest: Nr=%d dr=%g N_psi=%d "
                    "l_cont=%d keep=%d..%d\n",
                    ref.Nr, ref.dr, ref.N_psi, ref.l_cont,
                    ref.n_keep_lo, ref.n_keep_hi);
    }

    scatt::SolverParams sp;
    sp.n_grid          = ref.Nr;
    sp.dr              = ref.dr;
    sp.r_min           = 0.0;
    sp.n_mu            = ref.N_psi;
    sp.l_max_continuum = ref.l_cont;

    // -------------------- κ distribution (strided) ------------------------
    std::vector<int> my_kappas;
    for (size_t i = static_cast<size_t>(rank); i < ik_kappa.size();
         i += static_cast<size_t>(size)) {
        my_kappas.push_back(ik_kappa[i]);
    }
    std::printf("[rank %d] assigned %zu κ: ", rank, my_kappas.size());
    for (int k : my_kappas) std::printf("%d ", k);
    std::printf("\n");

    // -------------------- output dir (rank 0 creates) ---------------------
    if (rank == 0) fs::create_directories(out_dir);
    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------- helper: open a ψ checkpoint ---------------------
    const std::size_t Nr_stored = static_cast<std::size_t>(
        ref.n_keep_hi - ref.n_keep_lo + 1);
    auto load_storage = [&](int ik, scatt::PotentialStorage& store) {
        const fs::path dir = psi_dir_prefix + ik_to_str(ik);
        const int on_disk_cs = scatt::peek_checkpoint_chunk_size(dir.string());
        const int effective_cs =
            (on_disk_cs > 0) ? std::max(on_disk_cs, chunk_size) : chunk_size;
        if (!store.initialize_from_checkpoint(Nr_stored, ref.N_psi,
                                              dir.string(), effective_cs)) {
            throw std::runtime_error(
                "rank " + std::to_string(rank) +
                ": failed to load ψ from " + dir.string());
        }
    };

    // -------------------- main κ loop --------------------------------------
    using clk = std::chrono::steady_clock;
    for (size_t a = 0; a < my_kappas.size(); ++a) {
        const int kappa = my_kappas[a];
        const auto t_k0 = clk::now();
        std::printf("[rank %d] κ=%d (%zu/%zu, E=%g)  starting\n",
                    rank, kappa, a + 1, my_kappas.size(),
                    manifest_of(kappa).energy);

        // ψ_κ open (DISK, chunked).
        scatt::PotentialStorage psi_kappa;
        try { load_storage(kappa, psi_kappa); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "[rank %d] %s\n", rank, e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const int cs      = psi_kappa.chunk_size();
        const int nchunks = psi_kappa.num_chunks();
        if (cs <= 0 || nchunks <= 0) {
            std::fprintf(stderr,
                "[rank %d] ψ_κ ik=%d not in DISK mode (cs=%d nch=%d)\n",
                rank, kappa, cs, nchunks);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Per-pol shared CCAccumState (sparse Ang + Simpson weights).  ν-
        // independent because we assert identical n_keep across all ψ.
        std::vector<cc_dipole::CCAccumState> states;
        states.reserve(pols.size());
        for (auto pol : pols) {
            states.push_back(cc_dipole::make_accum_state(
                sp, ref.n_keep_lo, ref.n_keep_hi,
                    ref.n_keep_lo, ref.n_keep_hi,
                scatt::DipoleGauge::Length, pol, /*n_overlap_hi=*/-1));
        }

        // Per-(ν, pol) running accumulators (Zero, grown into chunk by chunk).
        const int N_psi   = ref.N_psi;
        const int num_pol = static_cast<int>(pols.size());
        const int num_nu  = static_cast<int>(ik_nu.size());
        std::vector<std::vector<Eigen::MatrixXd>> cc_raws(num_nu);
        for (auto& v : cc_raws) {
            v.reserve(num_pol);
            for (int p = 0; p < num_pol; ++p)
                v.emplace_back(Eigen::MatrixXd::Zero(N_psi, N_psi));
        }

        // Chunk-blocked iteration with async I/O overlap.
        //
        // κ side: psi_kappa.start_prefetch(c+1) is issued right after
        // chunk c is in RAM, so the disk read of c+1 overlaps with the
        // |ν| GEMMs of chunk c.  The next outer iteration's
        // psi_kappa.get(c+1) is then a zero-copy buffer swap.
        //
        // ν side: double-buffered PotentialStorage via unique_ptr.  At
        // iteration j we GEMM against psi_nu_curr (chunk c resident)
        // while psi_nu_next prefetches its chunk c in background.  End
        // of iteration: rotate curr <- next.
        //
        // Bit-identical to the synchronous code path (validated by
        // test_storage_prefetch_bit_equivalence): the only thing that
        // changes is WHEN the chunk data lands in RAM.
        for (int c = 0; c < nchunks; ++c) {
            const int ir_lo_c = c * cs;
            const int ir_hi_c = std::min((c + 1) * cs,
                                         static_cast<int>(ref.Nr));
            // Materialise κ chunk c (sync first time; later a zero-copy
            // prefetch-hit swap).
            const auto t_kload_0 = clk::now();
            (void) psi_kappa.get(static_cast<std::size_t>(ir_lo_c));
            const auto t_kload_1 = clk::now();
            // Background κ chunk c+1 read.
            if (c + 1 < nchunks) psi_kappa.start_prefetch(c + 1);

            // ν double-buffer: pre-load j=0 synchronously, then for each
            // j start prefetching j+1 in background while computing j.
            auto psi_nu_curr = std::make_unique<scatt::PotentialStorage>();
            try { load_storage(ik_nu[0], *psi_nu_curr); }
            catch (const std::exception& e) {
                std::fprintf(stderr, "[rank %d] %s\n", rank, e.what());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            (void) psi_nu_curr->get(static_cast<std::size_t>(ir_lo_c));

            const auto t_inner_0 = clk::now();
            double t_io_ns = 0.0, t_cpu_ns = 0.0;
            for (int j = 0; j < num_nu; ++j) {
                // Stage ν[j+1] in background.
                std::unique_ptr<scatt::PotentialStorage> psi_nu_next;
                if (j + 1 < num_nu) {
                    psi_nu_next =
                        std::make_unique<scatt::PotentialStorage>();
                    try { load_storage(ik_nu[j + 1], *psi_nu_next); }
                    catch (const std::exception& e) {
                        std::fprintf(stderr, "[rank %d] %s\n", rank, e.what());
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    psi_nu_next->start_prefetch(c);
                }

                const auto t_cpu_0 = clk::now();
                for (int p = 0; p < num_pol; ++p) {
                    cc_dipole::accumulate_cc_range(
                        states[p], psi_kappa, *psi_nu_curr,
                        ir_lo_c, ir_hi_c, cc_raws[j][p]);
                }
                const auto t_cpu_1 = clk::now();
                t_cpu_ns += std::chrono::duration<double, std::nano>(
                                 t_cpu_1 - t_cpu_0).count();

                // Rotate to next ν.  The get() on the promoted storage
                // consumes the prefetched chunk via zero-copy swap.
                psi_nu_curr = std::move(psi_nu_next);
                if (psi_nu_curr) {
                    const auto t_io_0 = clk::now();
                    (void) psi_nu_curr->get(static_cast<std::size_t>(ir_lo_c));
                    const auto t_io_1 = clk::now();
                    t_io_ns += std::chrono::duration<double, std::nano>(
                                    t_io_1 - t_io_0).count();
                }
            }
            const auto t_inner_1 = clk::now();
            const double dt_kload = std::chrono::duration<double>(
                                         t_kload_1 - t_kload_0).count();
            const double dt_total = std::chrono::duration<double>(
                                         t_inner_1 - t_kload_0).count();
            std::printf(
                "[rank %d] κ=%d chunk %d/%d ir=[%d,%d) done in %.1fs "
                "(κ-load %.1fs, ν-IO %.1fs, ν-CPU %.1fs)\n",
                rank, kappa, c + 1, nchunks, ir_lo_c, ir_hi_c,
                dt_total, dt_kload, t_io_ns * 1e-9, t_cpu_ns * 1e-9);
        }

        // Write per-κ HDF5.
        std::string fn = out_dir + "/cc_dipole_k" + ik_to_str(kappa) + ".h5";
        hid_t fout = H5Fcreate(fn.c_str(), H5F_ACC_TRUNC,
                                H5P_DEFAULT, H5P_DEFAULT);
        if (fout < 0) {
            std::fprintf(stderr, "[rank %d] cannot create %s\n",
                         rank, fn.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        write_h5_attr_int   (fout, "Nr",        ref.Nr);
        write_h5_attr_double(fout, "dr",        ref.dr);
        write_h5_attr_int   (fout, "N_psi",     ref.N_psi);
        write_h5_attr_int   (fout, "l_cont",    ref.l_cont);
        write_h5_attr_int   (fout, "n_keep_lo", ref.n_keep_lo);
        write_h5_attr_int   (fout, "n_keep_hi", ref.n_keep_hi);
        for (int j = 0; j < num_nu; ++j) {
            const int nu = ik_nu[j];
            char gname[64];
            std::snprintf(gname, sizeof(gname),
                          "pair_k%04d_n%04d", kappa, nu);
            hid_t gp = H5Gcreate2(fout, gname, H5P_DEFAULT,
                                   H5P_DEFAULT, H5P_DEFAULT);
            write_h5_attr_int   (gp, "ik_kappa", kappa);
            write_h5_attr_int   (gp, "ik_nu",    nu);
            write_h5_attr_double(gp, "E_kappa",  manifest_of(kappa).energy);
            write_h5_attr_double(gp, "E_nu",     manifest_of(nu).energy);
            for (int p = 0; p < num_pol; ++p) {
                std::string dname =
                    std::string("cc_raw_len_") + scatt::name_of(pols[p]);
                write_h5_matrix_2d(gp, dname.c_str(), cc_raws[j][p]);
            }
            H5Gclose(gp);
        }
        H5Fclose(fout);

        const auto t_k1 = clk::now();
        const double dt_k = std::chrono::duration<double>(t_k1 - t_k0).count();
        std::printf("[rank %d] κ=%d  wrote %s   κ-wall %.1fs\n",
                    rank, kappa, fn.c_str(), dt_k);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("[cc_dipole_mpi] all ranks finished\n");
    MPI_Finalize();
    return 0;
}
