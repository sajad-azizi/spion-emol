// cc_dipole_driver.cpp -- production CLI: load saved ψ checkpoints
// from a scattering scan and compute the back-prop-basis c-c dipole
// matrix elements for a list of (ik_κ, ik_ν) pairs.
//
// Output: a single HDF5 file with one group per (gauge, polarization,
// pair) holding the (N_psi × N_psi) cc_raw matrix and the energies.
//
// Usage:
//   cc_dipole_driver
//       --psi_dir_prefix <prefix>      e.g.  /scratch/psi_<molhash>_ik
//       --ik_kappa <list>              comma-separated (e.g. 50,60,70)
//       --ik_nu    <list>              comma-separated, OR "all" to use
//                                      the same list as ik_kappa
//       --gathered <gathered_dir>      to read the manifest (Nr, dr, Npsi,
//                                      l_cont, n_keep_lo/hi, energies)
//       --gauge length                 currently only length supported
//       --pol all                      x|y|z|all (default: all)
//       --out <path>                   output HDF5 file
//
// The "manifest" of each ψ checkpoint is BackPropagator's:
//   "kind=psi energy=... Nr=... dr=... Npsi=... Nf=... l_cont=...
//    l_exch=... n_occ=... n_trans=... keep=lo..hi"
// We read it from each per-ik psi-dir to get (Nr, dr, Npsi, n_keep_lo,
// n_keep_hi, energy).
#include "CCDipole.hpp"

#include "scatt/PotentialStorage.hpp"
#include "scatt/SolverParams.hpp"
#include "scatt/DipoleMatrixElement.hpp"      // DipoleGauge, Polarization

#include <hdf5.h>

#include <algorithm>
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
#if defined(__unix__) || defined(__APPLE__)
#  include <unistd.h>   // sysconf, _SC_AVPHYS_PAGES, _SC_PAGESIZE
#endif

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
        throw std::runtime_error("cc_dipole_driver: cannot open " + mf_path.string());
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
                "cc_dipole_driver: manifest missing key " + std::string(key));
        }
    };
    pull("energy",  [&](double v) -> double { m.energy = v;  return v; });
    pull("Nr",      [&](int v) -> int    { m.Nr = v;       return v; });
    pull("dr",      [&](double v) -> double { m.dr = v;      return v; });
    pull("Npsi",    [&](int v) -> int    { m.N_psi = v;    return v; });
    pull("Nf",      [&](int v) -> int    { m.N_f = v;      return v; });
    pull("l_cont",  [&](int v) -> int    { m.l_cont = v;   return v; });
    pull("l_exch",  [&](int v) -> int    { m.l_exch = v;   return v; });
    pull("n_occ",   [&](int v) -> int    { m.n_occ = v;    return v; });
    pull("n_trans", [&](int v) -> int    { m.n_trans = v;  return v; });

    // "keep=lo..hi" needs a different regex.
    {
        std::regex re(R"(keep=([0-9]+)\.\.([0-9]+))");
        std::smatch sm;
        if (std::regex_search(m.raw, sm, re)) {
            m.n_keep_lo = std::stoi(sm[1]);
            m.n_keep_hi = std::stoi(sm[2]);
        } else {
            throw std::runtime_error(
                "cc_dipole_driver: manifest missing keep=lo..hi");
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

void write_h5_double(hid_t parent, const char* name, double v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a  = H5Acreate2(parent, name, H5T_NATIVE_DOUBLE, sp,
                           H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a);  H5Sclose(sp);
}
void write_h5_int(hid_t parent, const char* name, int v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a  = H5Acreate2(parent, name, H5T_NATIVE_INT, sp,
                           H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT, &v);
    H5Aclose(a);  H5Sclose(sp);
}
void write_h5_matrix_2d(hid_t parent, const char* name,
                        const Eigen::MatrixXd& M) {
    hsize_t dims[2] = { static_cast<hsize_t>(M.rows()),
                        static_cast<hsize_t>(M.cols()) };
    hid_t sp = H5Screate_simple(2, dims, nullptr);
    // Eigen MatrixXd is column-major by default; HDF5 expects row-major.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        M_rm = M;
    hid_t ds = H5Dcreate2(parent, name, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, M_rm.data());
    H5Dclose(ds);  H5Sclose(sp);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string psi_dir_prefix;
    std::string ik_kappa_str, ik_nu_str = "all";
    std::string out_path = "cc_dipole.h5";
    std::string pol_str  = "all";
    std::string mem_mode = "auto";           // auto | memory | disk
    std::int64_t memory_budget_bytes = -1;   // -1 = auto-detect via sysconf
    int          chunk_size = 100;           // DISK-mode chunk size
    bool         reuse_kappa_chunks = false; // chunk-blocked loop reorder

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(a + " requires an argument");
            return argv[++i];
        };
        if      (a == "--psi_dir_prefix")     psi_dir_prefix = next();
        else if (a == "--ik_kappa")           ik_kappa_str   = next();
        else if (a == "--ik_nu")              ik_nu_str      = next();
        else if (a == "--out")                out_path       = next();
        else if (a == "--pol")                pol_str        = next();
        else if (a == "--memory-mode")        mem_mode       = next();
        else if (a == "--memory-budget")      memory_budget_bytes = std::stoll(next());
        else if (a == "--chunk-size")         chunk_size     = std::stoi(next());
        else if (a == "--reuse-kappa-chunks") reuse_kappa_chunks = true;
        else {
            std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
            return 1;
        }
    }
    if (psi_dir_prefix.empty() || ik_kappa_str.empty()) {
        std::fprintf(stderr,
            "usage: %s --psi_dir_prefix <prefix> --ik_kappa <list> "
            "[--ik_nu <list|all>] [--pol x|y|z|all] [--out <h5>]\n"
            "       [--memory-mode auto|memory|disk]  (default auto)\n"
            "       [--memory-budget BYTES]           (override avail-RAM detection)\n"
            "       [--chunk-size N]                  (DISK chunk_size, default 100)\n"
            "       [--reuse-kappa-chunks]            (chunk-blocked loop: read each\n"
            "                                          κ chunk once per κ instead of\n"
            "                                          once per (κ, ν).  Requires DISK\n"
            "                                          mode and identical n_keep window\n"
            "                                          across all ψ checkpoints; gives\n"
            "                                          ~2x wall-time speedup on the\n"
            "                                          I/O-bound C8F8 l_cont=100 run.)\n",
            argv[0]);
        return 1;
    }
    if (mem_mode != "auto" && mem_mode != "memory" && mem_mode != "disk") {
        std::fprintf(stderr,
            "--memory-mode must be auto|memory|disk, got '%s'\n", mem_mode.c_str());
        return 1;
    }
    auto ik_kappa = parse_ik_list(ik_kappa_str);
    auto ik_nu    = (ik_nu_str == "all") ? ik_kappa : parse_ik_list(ik_nu_str);

    std::vector<scatt::Polarization> pols;
    if (pol_str == "all") pols = {scatt::Polarization::X, scatt::Polarization::Y,
                                  scatt::Polarization::Z};
    else if (pol_str == "x") pols = {scatt::Polarization::X};
    else if (pol_str == "y") pols = {scatt::Polarization::Y};
    else if (pol_str == "z") pols = {scatt::Polarization::Z};
    else { std::fprintf(stderr, "bad pol: %s\n", pol_str.c_str()); return 1; }

    // ---- Read manifests for all unique ik's, build the SolverParams skeleton ----
    auto ik_str = [](int ik) {
        char buf[16]; std::snprintf(buf, sizeof(buf), "%04d", ik); return std::string(buf);
    };
    std::vector<int> all_ik = ik_kappa;
    for (int n : ik_nu) if (std::find(all_ik.begin(), all_ik.end(), n) == all_ik.end())
        all_ik.push_back(n);

    std::printf("[cc_dipole_driver] loading manifests for %zu ik's...\n", all_ik.size());
    PsiManifest ref;
    bool ref_set = false;
    std::vector<PsiManifest> manifests(all_ik.size());
    for (size_t j = 0; j < all_ik.size(); ++j) {
        const fs::path dir = psi_dir_prefix + ik_str(all_ik[j]);
        manifests[j] = parse_manifest(dir);
        if (!ref_set) { ref = manifests[j]; ref_set = true; }
        else {
            // Cross-check that all psi shapes are compatible.
            if (manifests[j].Nr != ref.Nr || manifests[j].N_psi != ref.N_psi ||
                manifests[j].n_keep_lo != ref.n_keep_lo ||
                manifests[j].n_keep_hi != ref.n_keep_hi) {
                std::fprintf(stderr,
                    "[cc_dipole_driver] WARNING: ik=%d manifest does not match "
                    "ref (Nr=%d/%d, N_psi=%d/%d, keep=%d..%d / %d..%d)\n",
                    all_ik[j], manifests[j].Nr, ref.Nr,
                    manifests[j].N_psi, ref.N_psi,
                    manifests[j].n_keep_lo, manifests[j].n_keep_hi,
                    ref.n_keep_lo, ref.n_keep_hi);
            }
        }
    }
    std::printf("[cc_dipole_driver] reference manifest: Nr=%d dr=%g N_psi=%d "
                "l_cont=%d keep=%d..%d\n",
                ref.Nr, ref.dr, ref.N_psi, ref.l_cont, ref.n_keep_lo, ref.n_keep_hi);

    scatt::SolverParams sp;
    sp.n_grid          = ref.Nr;
    sp.dr              = ref.dr;
    sp.r_min           = 0.0;       // matches scattering convention (r_min in
                                    // Parameters is read from preproc; here we
                                    // only need it for the radial weight,
                                    // and r = r_min + ir*dr.  Scattering uses
                                    // r_min from data.rmin which is 0 in
                                    // typical preproc outputs.
    sp.n_mu            = ref.N_psi;
    sp.l_max_continuum = ref.l_cont;

    // ---- Memory-mode resolution -------------------------------------------
    // MEMORY mode loads the full ψ (Nr_stored × N_psi² × 8 B) into RAM per
    // checkpoint; the outer loop holds psi_κ while the inner loop loads each
    // psi_ν, so peak resident is 2 × bytes_per_psi.  For C8F8 production
    // (l_cont=80, Nr_stored≈2000-3000) one psi is ~700 GB, two psis ≈ 1.4 TB
    // → OOM on a 1 TB node.  DISK mode keeps only the chunk_size matrices
    // of the chunk buffer in RAM (≪ 1 GB) and streams the rest from
    // pot_chunk_*.bin.
    const std::size_t Nr_stored = static_cast<std::size_t>(
        ref.n_keep_hi - ref.n_keep_lo + 1);
    const std::size_t bytes_per_psi =
        Nr_stored * static_cast<std::size_t>(ref.N_psi) *
        static_cast<std::size_t>(ref.N_psi) * sizeof(double);

    // Detect available RAM (Linux/macOS).  -1 -> couldn't detect → assume DISK.
    auto detect_avail_bytes = []() -> std::int64_t {
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
        long pages = sysconf(_SC_AVPHYS_PAGES);
        long psz   = sysconf(_SC_PAGESIZE);
        if (pages <= 0 || psz <= 0) return -1;
        return static_cast<std::int64_t>(pages) * static_cast<std::int64_t>(psz);
#else
        return -1;
#endif
    };

    if (mem_mode == "auto") {
        std::int64_t avail = (memory_budget_bytes > 0)
            ? memory_budget_bytes
            : detect_avail_bytes();
        // Reserve a safety margin (8 GB) for the integration buffers, HDF5
        // writes, OS, MKL/Eigen scratch, etc.  Require both κ and ν psis to
        // fit (peak resident = 2 × bytes_per_psi) before choosing MEMORY.
        constexpr std::size_t kSafety = 8ull << 30;   // 8 GB
        const std::size_t needed = 2 * bytes_per_psi + kSafety;
        if (avail < 0) {
            std::fprintf(stderr,
                "[cc_dipole_driver] --memory-mode auto: could not detect "
                "available RAM; defaulting to DISK\n");
            mem_mode = "disk";
        } else if (static_cast<std::size_t>(avail) < needed) {
            std::printf("[cc_dipole_driver] auto -> DISK : avail %.2f GB < "
                        "needed %.2f GB (2 ψ + 8 GB safety; bytes_per_psi=%.2f GB)\n",
                        avail / double(1ull<<30),
                        needed / double(1ull<<30),
                        bytes_per_psi / double(1ull<<30));
            mem_mode = "disk";
        } else {
            std::printf("[cc_dipole_driver] auto -> MEMORY: avail %.2f GB ≥ "
                        "needed %.2f GB  (bytes_per_psi=%.2f GB)\n",
                        avail / double(1ull<<30),
                        needed / double(1ull<<30),
                        bytes_per_psi / double(1ull<<30));
            mem_mode = "memory";
        }
    } else {
        std::printf("[cc_dipole_driver] forced --memory-mode=%s "
                    "(bytes_per_psi=%.2f GB)\n",
                    mem_mode.c_str(), bytes_per_psi / double(1ull<<30));
    }

    // ---- Build per-ik PotentialStorage objects (read-only, on disk) ----
    auto load_storage = [&](int ik, scatt::PotentialStorage& store) {
        const fs::path dir = psi_dir_prefix + ik_str(ik);
        // For molecules with compact orbitals (e.g. H2) the scattering pipeline
        // may write only ψ over the kept radial window [n_keep_lo, n_keep_hi]
        // rather than the full Nr.  Use the actual count of stored matrices.
        //
        // PotentialStorage::initialize_from_checkpoint compares the on-disk
        // chunk_size against the runtime-budgeted chunk_size and REJECTS the
        // checkpoint if on-disk > runtime (would over-allocate read_buffer).
        // The C8F8 production scattering chose its own chunk_size from main.cpp's
        // memory planner, which can be larger than our 100-default and DOES
        // make initialize_from_checkpoint return false.  Peek the on-disk
        // chunk_size first and pass it through verbatim so the check passes.
        const int on_disk_cs = scatt::peek_checkpoint_chunk_size(dir.string());
        const int effective_cs =
            (on_disk_cs > 0) ? std::max(on_disk_cs, chunk_size) : chunk_size;
        if (mem_mode == "memory") {
            if (!store.try_load_into_memory(Nr_stored, ref.N_psi, dir.string(),
                                            effective_cs)) {
                throw std::runtime_error(
                    "cc_dipole_driver: --memory-mode=memory but try_load_into_memory "
                    "failed for " + dir.string() +
                    "  (manifest mismatch or chunks missing?  Try --memory-mode=disk)");
            }
        } else {
            // DISK mode: stream chunks from pot_chunk_*.bin on demand.
            if (!store.initialize_from_checkpoint(Nr_stored, ref.N_psi,
                                                  dir.string(), effective_cs)) {
                throw std::runtime_error(
                    "cc_dipole_driver: failed to load psi (DISK) from " + dir.string() +
                    "  (on-disk chunk_size=" + std::to_string(on_disk_cs) +
                    ", effective=" + std::to_string(effective_cs) + ")");
            }
        }
    };

    // ---- Open output HDF5 ----
    hid_t fout = H5Fcreate(out_path.c_str(), H5F_ACC_TRUNC,
                            H5P_DEFAULT, H5P_DEFAULT);
    if (fout < 0) {
        std::fprintf(stderr, "cannot create %s\n", out_path.c_str());
        return 1;
    }
    write_h5_int(fout,    "Nr",         ref.Nr);
    write_h5_double(fout, "dr",         ref.dr);
    write_h5_int(fout,    "N_psi",      ref.N_psi);
    write_h5_int(fout,    "l_cont",     ref.l_cont);
    write_h5_int(fout,    "n_keep_lo",  ref.n_keep_lo);
    write_h5_int(fout,    "n_keep_hi",  ref.n_keep_hi);

    // Small helper: find manifest by ik value.
    auto manifest_of = [&](int ik) -> const PsiManifest& {
        auto it = std::find(all_ik.begin(), all_ik.end(), ik);
        return manifests[std::distance(all_ik.begin(), it)];
    };

    // Small helper: write one (κ, ν, all pols) HDF5 group.
    auto write_pair_group =
        [&](int kappa, int nu,
            const std::vector<Eigen::MatrixXd>& cc_raws_by_pol)
    {
        char gname[64];
        std::snprintf(gname, sizeof(gname), "pair_k%04d_n%04d", kappa, nu);
        hid_t gp = H5Gcreate2(fout, gname, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT);
        const auto& m_kappa = manifest_of(kappa);
        const auto& m_nu    = manifest_of(nu);
        write_h5_int(gp,    "ik_kappa", kappa);
        write_h5_int(gp,    "ik_nu",    nu);
        write_h5_double(gp, "E_kappa",  m_kappa.energy);
        write_h5_double(gp, "E_nu",     m_nu.energy);
        for (size_t p = 0; p < pols.size(); ++p) {
            std::string dname =
                std::string("cc_raw_len_") + scatt::name_of(pols[p]);
            write_h5_matrix_2d(gp, dname.c_str(), cc_raws_by_pol[p]);
        }
        H5Gclose(gp);
    };

    // ---- Iterate over (ik_κ, ik_ν) pairs ----
    if (!reuse_kappa_chunks) {
        // Legacy path: one compute_cc_dipole per (κ, ν).  Re-reads ALL κ
        // chunks for every ν.  Unchanged behaviour, kept for correctness
        // reference and for small jobs / MEMORY-mode runs.
        for (size_t a = 0; a < ik_kappa.size(); ++a) {
            const int kappa = ik_kappa[a];
            scatt::PotentialStorage psi_kappa;
            load_storage(kappa, psi_kappa);
            std::printf("[cc_dipole_driver] κ ik=%d  E=%g loaded (mode=%s)\n",
                        kappa, manifest_of(kappa).energy,
                        psi_kappa.mode() == scatt::PotentialStorage::Mode::MEMORY
                            ? "MEM" : "DISK");

            for (int nu : ik_nu) {
                scatt::PotentialStorage psi_nu;
                load_storage(nu, psi_nu);

                std::vector<Eigen::MatrixXd> cc_raws_by_pol;
                cc_raws_by_pol.reserve(pols.size());
                for (auto pol : pols) {
                    auto res = cc_dipole::compute_cc_dipole(
                        sp, psi_kappa, psi_nu, ref.n_keep_lo, ref.n_keep_hi,
                        scatt::DipoleGauge::Length, pol);
                    cc_raws_by_pol.emplace_back(std::move(res.cc_raw));
                }
                write_pair_group(kappa, nu, cc_raws_by_pol);
                std::printf("  pair (κ=%d, ν=%d)  done\n", kappa, nu);
            }
        }
    } else {
        // ---- Chunk-blocked path: read each κ chunk once per κ ----------
        //
        // Loop reorganisation: outer κ → for each radial chunk c → for
        // each ν → for each pol.  After all chunks are processed for a
        // given κ, write all 91 ν × 3 pol cc_raws to HDF5.  Net I/O:
        // (num_chunks) κ-chunk reads + (num_chunks × |ν|) ν-chunk reads
        // per κ, vs (num_chunks × |ν|) of EACH in the legacy path -- a
        // 2× total reduction.
        //
        // Memory: 3 per-pol CCAccumStates (Ang each ~N_psi²·8 = 833 MB
        // at l_cont=100) + |ν|·3 cc_raw accumulators (each 833 MB) +
        // two 83 GB chunk read-buffers (psi_κ and the currently-live
        // psi_ν).  For C8F8 |ν|=91 → 227 GB cc_raws + 2.5 GB Ang + 166
        // GB read-buffers ≈ 400 GB.  Fits the 985 GB node.
        if (mem_mode != "disk") {
            std::fprintf(stderr,
                "[cc_dipole_driver] --reuse-kappa-chunks requires "
                "--memory-mode=disk (got '%s')\n", mem_mode.c_str());
            H5Fclose(fout);
            return 1;
        }
        // Cross-check that all ψ have identical keep windows so the
        // per-pol state (which encodes n_lo, n_hi) is valid for every
        // (κ, ν) pair.
        for (const auto& m : manifests) {
            if (m.n_keep_lo != ref.n_keep_lo || m.n_keep_hi != ref.n_keep_hi) {
                std::fprintf(stderr,
                    "[cc_dipole_driver] --reuse-kappa-chunks requires "
                    "identical n_keep across all ψ checkpoints (got "
                    "%d..%d, expected %d..%d)\n",
                    m.n_keep_lo, m.n_keep_hi, ref.n_keep_lo, ref.n_keep_hi);
                H5Fclose(fout);
                return 1;
            }
        }

        // Per-pol shared state.  Ang depends only on q(pol); the
        // integration window [n_lo, n_hi] depends only on the keep
        // window (already verified identical for all ψ).
        std::vector<cc_dipole::CCAccumState> states;
        states.reserve(pols.size());
        for (auto pol : pols) {
            states.push_back(cc_dipole::make_accum_state(
                sp, ref.n_keep_lo, ref.n_keep_hi,
                    ref.n_keep_lo, ref.n_keep_hi,
                scatt::DipoleGauge::Length, pol, /*n_overlap_hi=*/-1));
        }
        std::printf("[cc_dipole_driver] --reuse-kappa-chunks: "
                    "integration window [n_lo, n_hi)=[%d, %d), "
                    "n_pts=%d\n",
                    states[0].n_lo, states[0].n_hi,
                    states[0].n_hi - states[0].n_lo);

        const int N_psi  = ref.N_psi;
        const int num_pol = static_cast<int>(pols.size());
        const int num_nu  = static_cast<int>(ik_nu.size());

        for (size_t a = 0; a < ik_kappa.size(); ++a) {
            const int kappa = ik_kappa[a];
            scatt::PotentialStorage psi_kappa;
            load_storage(kappa, psi_kappa);
            const int cs       = psi_kappa.chunk_size();
            const int nchunks  = psi_kappa.num_chunks();
            if (cs <= 0 || nchunks <= 0) {
                std::fprintf(stderr,
                    "[cc_dipole_driver] --reuse-kappa-chunks: ψ_κ not in "
                    "DISK mode (chunk_size=%d, num_chunks=%d)\n",
                    cs, nchunks);
                H5Fclose(fout);
                return 1;
            }
            std::printf("[cc_dipole_driver] κ ik=%d  E=%g loaded  "
                        "chunk_size=%d  num_chunks=%d\n",
                        kappa, manifest_of(kappa).energy, cs, nchunks);

            // Per-(ν, pol) running accumulators.  Initialised to zero;
            // accumulate_cc_range adds into them.
            std::vector<std::vector<Eigen::MatrixXd>> cc_raws(num_nu);
            for (auto& v : cc_raws) {
                v.reserve(num_pol);
                for (int p = 0; p < num_pol; ++p)
                    v.emplace_back(Eigen::MatrixXd::Zero(N_psi, N_psi));
            }

            // Chunk-blocked: read κ chunk c ONCE per c, process all ν
            // against it, then advance to chunk c+1.
            //
            // ASYNC I/O OVERLAP (accuracy-preserving, byte-identical to the
            // synchronous code -- guarded by test_storage_prefetch_bit_equivalence):
            //
            //   * κ side: while we GEMM through all |ν| ν's of chunk c, the
            //     next κ chunk c+1 is being pread()'d in background.  The
            //     next outer iteration's psi_kappa.get(c+1) hits the
            //     prefetched buffer and swaps zero-copy.  Saves ~one
            //     ν-equivalent of κ-read I/O per chunk.
            //
            //   * ν side: DOUBLE-BUFFERED PotentialStorage.  At iteration j
            //     we GEMM against psi_nu_curr (chunk c already resident)
            //     while psi_nu_next's chunk c read happens in background.
            //     End of iteration: rotate curr <- next; the get() that
            //     materialises the rotated buffer is a free zero-copy
            //     swap.  Hides ~91 ν chunk reads / chunk behind compute.
            //
            // Peak memory adds ~2 chunks worth of read_buffer (one each
            // for κ-prefetch and the ν-next storage): at L=100 / cs≈100,
            // that's ~2·83 GB = 166 GB extra peak vs the no-prefetch
            // version.  Still fits a 1 TB node with the cc_raws + κ + ν
            // + scratch budget.
            for (int c = 0; c < nchunks; ++c) {
                const int ir_lo_c = c * cs;
                const int ir_hi_c = std::min((c + 1) * cs,
                                             static_cast<int>(ref.Nr));
                // Materialise κ chunk c (synchronous on first iter; a
                // zero-copy prefetch-hit swap on subsequent iters).
                (void) psi_kappa.get(static_cast<std::size_t>(ir_lo_c));
                // Kick off the κ chunk c+1 read in background.  No-op if
                // we're already on the last chunk.  Bit-identical to the
                // synchronous path -- the bytes brought in are the same
                // pread()'d bytes, just landed earlier.
                if (c + 1 < nchunks)
                    psi_kappa.start_prefetch(c + 1);

                // ν double-buffer: psi_nu_curr (this j) + psi_nu_next
                // (prefetching for j+1).  Use unique_ptr because
                // PotentialStorage has std::atomic members and so is
                // neither copyable nor movable -- but unique_ptr swap is
                // O(1) and doesn't touch the storage itself.
                auto psi_nu_curr =
                    std::make_unique<scatt::PotentialStorage>();
                load_storage(ik_nu[0], *psi_nu_curr);
                (void) psi_nu_curr->get(static_cast<std::size_t>(ir_lo_c));

                for (int j = 0; j < num_nu; ++j) {
                    // Stage ν[j+1] in the background: open its checkpoint
                    // and prefetch chunk c while ν[j] is GEMM'd below.
                    std::unique_ptr<scatt::PotentialStorage> psi_nu_next;
                    if (j + 1 < num_nu) {
                        psi_nu_next =
                            std::make_unique<scatt::PotentialStorage>();
                        load_storage(ik_nu[j + 1], *psi_nu_next);
                        psi_nu_next->start_prefetch(c);
                    }

                    // GEMM ν[j] against κ chunk c (both resident in RAM).
                    for (int p = 0; p < num_pol; ++p) {
                        cc_dipole::accumulate_cc_range(
                            states[p], psi_kappa, *psi_nu_curr,
                            ir_lo_c, ir_hi_c, cc_raws[j][p]);
                    }

                    // Rotate: drop the old curr (frees its read_buffer_),
                    // promote next.  The first get() on the promoted
                    // storage consumes the prefetched chunk c via zero-
                    // copy swap.
                    psi_nu_curr = std::move(psi_nu_next);
                    if (psi_nu_curr)
                        (void) psi_nu_curr->get(static_cast<std::size_t>(ir_lo_c));
                }
                std::printf("  κ=%d  chunk %d/%d  ir=[%d,%d)  done\n",
                            kappa, c + 1, nchunks, ir_lo_c, ir_hi_c);
            }

            // All chunks processed for this κ → flush all (ν, pol) to HDF5.
            for (int j = 0; j < num_nu; ++j) {
                write_pair_group(kappa, ik_nu[j], cc_raws[j]);
            }
            std::printf("  κ=%d  wrote %d ν × %d pol pair groups\n",
                        kappa, num_nu, num_pol);
        }
    }

    H5Fclose(fout);
    std::printf("[cc_dipole_driver] wrote %s\n", out_path.c_str());
    return 0;
}
