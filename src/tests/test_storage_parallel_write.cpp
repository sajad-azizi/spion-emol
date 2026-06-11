// test_storage_parallel_write.cpp
//
// Gate test for the OPT-IN parallel chunk writer in PotentialStorage.
//
// Two write paths must produce byte-identical CHUNK FILES on disk:
//   * SERIAL  -- the legacy path (parallel_chunk_write = false).
//   * PARALLEL -- the new opt-in path (parallel_chunk_write = true,
//                 multi-threaded pwrite at distinct offsets, atomic
//                 temp + rename + fsync).
//
// We test BOTH symmetric_storage modes:
//   * symmetric_storage = false  (full N×N on disk per matrix)
//   * symmetric_storage = true   (packed lower triangle per matrix)
//
// For each (sym, parallel) combo we:
//   1. Write the same set of bit-symmetric reference matrices.
//   2. Read back via initialize_from_checkpoint.
//   3. Assert read-back is byte-equal to the reference.
//
// Plus, MOST IMPORTANTLY: we compare the on-disk CHUNK FILES between
// the SERIAL and PARALLEL writers byte-for-byte.  If the parallel
// writer ever differs from the serial output by a single bit, the
// test fails -- there is no tolerance.

#include "scatt/PotentialStorage.hpp"

#include <Eigen/Dense>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace scatt;
namespace fs = std::filesystem;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL  " << what << "\n"; ++g_fail; }
    else     { std::cout << "ok    " << what << "\n"; }
}

static Eigen::MatrixXd make_bit_symmetric(int N, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    Eigen::MatrixXd M(N, N);
    for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {
            const double v = U(rng);
            M(i, j) = v;
            if (i != j) M(j, i) = v;
        }
    }
    return M;
}

static std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(n));
    f.read(reinterpret_cast<char*>(bytes.data()), n);
    return bytes;
}

static bool byte_equal_files(const std::string& a, const std::string& b) {
    const auto A = read_file_bytes(a);
    const auto B = read_file_bytes(b);
    return A == B;
}

struct WriteResult {
    std::string dir;
    int         num_chunk_files = 0;
};

static WriteResult write_with(
    const std::vector<Eigen::MatrixXd>& ref,
    int N, int Nr, int chunk_size,
    bool symmetric, bool parallel,
    const std::string& dir,
    const std::string& manifest)
{
    fs::remove_all(dir);
    PotentialStorage S;
    S.set_manifest(manifest);
    S.initialize_for_write(Nr, N, PotentialStorage::Mode::DISK,
                           dir, chunk_size,
                           /*symmetric_storage=*/symmetric,
                           /*parallel_chunk_write=*/parallel);
    for (int ir = 0; ir < Nr; ++ir) S.store(ir, ref[ir]);
    S.finalize_write();

    WriteResult r{dir, 0};
    for (const auto& e : fs::directory_iterator(dir)) {
        const auto fn = e.path().filename().string();
        if (fn.find("pot_chunk_") != std::string::npos) ++r.num_chunk_files;
    }
    return r;
}

static void compare_disk_byte_equal(
    const std::string& dir_serial,
    const std::string& dir_parallel,
    int n_chunks,
    const std::string& label)
{
    int n_diff = 0;
    for (int c = 0; c < n_chunks; ++c) {
        const std::string fn = "/pot_chunk_" + std::to_string(c) + ".bin";
        const bool eq = byte_equal_files(dir_serial + fn, dir_parallel + fn);
        if (!eq) ++n_diff;
    }
    check(n_diff == 0,
          "(" + label + ")  chunk files byte-equal between serial and parallel  "
          "(" + std::to_string(n_chunks) + " files, n_diff = "
          + std::to_string(n_diff) + ")");
}

static void roundtrip_byte_equal_to_ref(
    const std::vector<Eigen::MatrixXd>& ref,
    int N, int Nr, int chunk_size,
    const std::string& dir,
    const std::string& manifest,
    const std::string& label)
{
    PotentialStorage S;
    S.set_manifest(manifest);
    const bool ok = S.initialize_from_checkpoint(Nr, N, dir, chunk_size);
    check(ok, "(" + label + ")  load checkpoint");
    int n_diff = 0;
    for (int ir = 0; ir < Nr; ++ir) {
        const Eigen::MatrixXd& M = S.get(static_cast<std::size_t>(ir));
        if ((M - ref[ir]).cwiseAbs().maxCoeff() != 0.0) ++n_diff;
    }
    check(n_diff == 0,
          "(" + label + ")  every read-back matrix byte-equals ref  "
          "(n_diff = " + std::to_string(n_diff) + ")");
}

int main() {
    constexpr int N = 89;          // odd-ish, exercises packing
    constexpr int Nr = 11;         // multi-chunk
    constexpr int chunk_size = 4;  // forces 3 chunks (4+4+3)

    std::vector<Eigen::MatrixXd> ref(Nr);
    for (int ir = 0; ir < Nr; ++ir) {
        ref[ir] = make_bit_symmetric(N, 0xC0DEFEED + ir * 31);
    }

    const std::string root  = "./checkpoints/test_parallel_write";
    fs::remove_all(root);
    fs::create_directories(root);

    const std::string manifest = "test_parallel_write N=" + std::to_string(N) +
                                 " Nr=" + std::to_string(Nr);

    // ---------------- FULL  storage: serial vs parallel ---------------
    const auto dir_full_serial   = root + "/full_serial";
    const auto dir_full_parallel = root + "/full_parallel";
    auto rfs = write_with(ref, N, Nr, chunk_size,
                          /*sym=*/false, /*par=*/false,
                          dir_full_serial, manifest);
    auto rfp = write_with(ref, N, Nr, chunk_size,
                          /*sym=*/false, /*par=*/true,
                          dir_full_parallel, manifest);
    check(rfs.num_chunk_files == rfp.num_chunk_files,
          "(FULL)     same chunk count: "
          + std::to_string(rfs.num_chunk_files) + " vs "
          + std::to_string(rfp.num_chunk_files));
    compare_disk_byte_equal(dir_full_serial, dir_full_parallel,
                            rfs.num_chunk_files, "FULL");
    roundtrip_byte_equal_to_ref(ref, N, Nr, chunk_size,
                                dir_full_parallel, manifest,
                                "FULL parallel");

    // ---------------- SYMMETRIC storage: serial vs parallel -----------
    const auto dir_sym_serial   = root + "/sym_serial";
    const auto dir_sym_parallel = root + "/sym_parallel";
    auto rss = write_with(ref, N, Nr, chunk_size,
                          /*sym=*/true, /*par=*/false,
                          dir_sym_serial, manifest);
    auto rsp = write_with(ref, N, Nr, chunk_size,
                          /*sym=*/true, /*par=*/true,
                          dir_sym_parallel, manifest);
    check(rss.num_chunk_files == rsp.num_chunk_files,
          "(SYM)      same chunk count: "
          + std::to_string(rss.num_chunk_files) + " vs "
          + std::to_string(rsp.num_chunk_files));
    compare_disk_byte_equal(dir_sym_serial, dir_sym_parallel,
                            rss.num_chunk_files, "SYM");
    roundtrip_byte_equal_to_ref(ref, N, Nr, chunk_size,
                                dir_sym_parallel, manifest,
                                "SYM parallel");

    // ---------------- Cross-check: SYM-parallel vs FULL-parallel
    // ----------------- (different formats, same end matrices)
    {
        PotentialStorage Sf;  Sf.set_manifest(manifest);
        Sf.initialize_from_checkpoint(Nr, N, dir_full_parallel, chunk_size);
        PotentialStorage Ss;  Ss.set_manifest(manifest);
        Ss.initialize_from_checkpoint(Nr, N, dir_sym_parallel, chunk_size);
        int n_diff = 0;
        for (int ir = 0; ir < Nr; ++ir) {
            const Eigen::MatrixXd& Mf = Sf.get(static_cast<std::size_t>(ir));
            const Eigen::MatrixXd& Ms = Ss.get(static_cast<std::size_t>(ir));
            if ((Mf - Ms).cwiseAbs().maxCoeff() != 0.0) ++n_diff;
        }
        check(n_diff == 0,
              "(CROSS)    SYM-parallel and FULL-parallel produce byte-equal "
              "matrices in memory after read  (n_diff = "
              + std::to_string(n_diff) + ")");
    }

    // ---------------- File-size sanity ----------------
    {
        const auto sz_full = fs::file_size(dir_full_parallel + "/pot_chunk_0.bin");
        const auto sz_sym  = fs::file_size(dir_sym_parallel  + "/pot_chunk_0.bin");
        check(sz_sym < sz_full,
              "(SIZE)     symmetric < full chunk size: "
              + std::to_string(sz_sym) + " B  <  "
              + std::to_string(sz_full) + " B");
    }

    fs::remove_all(root);

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " storage_parallel_write  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
