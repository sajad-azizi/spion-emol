// test_storage_symmetric.cpp
//
// Gate test for PotentialStorage's symmetric_storage = true on-disk
// format.  THIS IS THE PROOF THAT THE OPTIMISATION IS BIT-EQUIVALENT.
//
// Three independent byte-equality checks:
//
//   (A) ROUNDTRIP: write a known bit-symmetric matrix M with
//       symmetric_storage=true; read back via initialize_from_checkpoint;
//       assert |M_read - M_original|_inf == 0 at every grid point.
//
//   (B) UPPER vs LOWER bit-equality after roundtrip: M_read(i,j) ==
//       M_read(j,i) for all i,j.
//
//   (C) SYMMETRIC vs FULL CROSS-CHECK: write the SAME matrices once
//       with symmetric_storage=true and once with =false; read each
//       back; both must be byte-equal to the originals AND to each
//       other.
//
// Plus a v1-format compatibility check:
//   (D) BACKWARD COMPATIBILITY: write a checkpoint with the LEGACY
//       format (symmetric_storage=false), then load via the new code.
//       Must succeed and produce byte-equal results.  This proves the
//       magic-prefix detection works and pre-existing checkpoints stay
//       readable.
//
// All assertions use literal `== 0` (NOT `< tol`).  A nonzero diff is
// a real bug.

#include "scatt/PotentialStorage.hpp"

#include <Eigen/Dense>

#include <cstdlib>
#include <cstdio>
#include <filesystem>
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

// Generate a bit-symmetric matrix of a given size with reproducible
// random doubles in [-1, 1].  Sets M(i,j) = M(j,i) by writing both
// from the same scalar -- guarantees byte-equality of the upper and
// lower triangles.
static Eigen::MatrixXd make_bit_symmetric(int N, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    Eigen::MatrixXd M(N, N);
    for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {
            const double v = U(rng);
            M(i, j) = v;
            if (i != j) M(j, i) = v;     // SAME `v` for both halves
        }
    }
    return M;
}

static void cleanup_dir(const std::string& d) {
    std::error_code ec;
    fs::remove_all(d, ec);
}

int main() {
    constexpr int N = 53;          // small enough that the full+packed both fit in a few KB
    constexpr int Nr = 7;          // a handful of grid points
    constexpr int chunk_size = 3;  // forces multi-chunk writes (Nr=7, chunk=3 -> 3 chunks)

    // -------- Build the reference matrices (in-memory) --------
    std::vector<Eigen::MatrixXd> ref(Nr);
    for (int ir = 0; ir < Nr; ++ir) {
        ref[ir] = make_bit_symmetric(N, 42 + ir);
    }
    // Sanity: the generator produced bit-symmetric matrices.
    for (int ir = 0; ir < Nr; ++ir) {
        const double a = (ref[ir] - ref[ir].transpose()).cwiseAbs().maxCoeff();
        check(a == 0.0, "reference matrix ir=" + std::to_string(ir)
                        + " is bit-symmetric (sanity)");
    }

    const std::string dir_sym  = "./checkpoints/test_storage_symmetric_SYM";
    const std::string dir_full = "./checkpoints/test_storage_symmetric_FULL";
    cleanup_dir(dir_sym);
    cleanup_dir(dir_full);

    const std::string manifest = "test_storage_symmetric N=" + std::to_string(N) +
                                 " Nr=" + std::to_string(Nr);

    // ---------------- (A,B) write + read with symmetric_storage=true ----
    {
        PotentialStorage S;
        S.set_manifest(manifest);
        S.initialize_for_write(Nr, N, PotentialStorage::Mode::DISK,
                               dir_sym, chunk_size, /*symmetric_storage=*/true);
        for (int ir = 0; ir < Nr; ++ir) S.store(ir, ref[ir]);
        S.finalize_write();
        check(S.is_symmetric_storage() == true,
              "symmetric flag stored after finalize_write");
    }

    {
        PotentialStorage S;
        S.set_manifest(manifest);
        const bool ok = S.initialize_from_checkpoint(Nr, N, dir_sym, chunk_size);
        check(ok, "load symmetric checkpoint");
        check(S.is_symmetric_storage() == true,
              "symmetric flag detected from metadata.bin v2 magic");

        int n_diff = 0;
        int n_asym = 0;
        for (int ir = 0; ir < Nr; ++ir) {
            const Eigen::MatrixXd& Mr = S.get(static_cast<std::size_t>(ir));
            const double d = (Mr - ref[ir]).cwiseAbs().maxCoeff();
            if (d != 0.0) ++n_diff;
            const double a = (Mr - Mr.transpose()).cwiseAbs().maxCoeff();
            if (a != 0.0) ++n_asym;
        }
        check(n_diff == 0,
              "(A) every Sinv_read byte-equals ref  (n_diff = "
              + std::to_string(n_diff) + ")");
        check(n_asym == 0,
              "(B) every Sinv_read is bit-symmetric after reflection  (n_asym = "
              + std::to_string(n_asym) + ")");
    }

    // ---------------- (C) cross-check sym vs full ----------------------
    {
        PotentialStorage S;
        S.set_manifest(manifest);
        S.initialize_for_write(Nr, N, PotentialStorage::Mode::DISK,
                               dir_full, chunk_size,
                               /*symmetric_storage=*/false);
        for (int ir = 0; ir < Nr; ++ir) S.store(ir, ref[ir]);
        S.finalize_write();
        check(S.is_symmetric_storage() == false,
              "full storage produces full on-disk format");
    }

    {
        PotentialStorage S_sym;
        S_sym.set_manifest(manifest);
        S_sym.initialize_from_checkpoint(Nr, N, dir_sym, chunk_size);

        PotentialStorage S_full;
        S_full.set_manifest(manifest);
        S_full.initialize_from_checkpoint(Nr, N, dir_full, chunk_size);

        int n_diff = 0;
        for (int ir = 0; ir < Nr; ++ir) {
            const Eigen::MatrixXd Ms = S_sym .get(static_cast<std::size_t>(ir));
            const Eigen::MatrixXd Mf = S_full.get(static_cast<std::size_t>(ir));
            if ((Ms - Mf).cwiseAbs().maxCoeff() != 0.0) ++n_diff;
        }
        check(n_diff == 0,
              "(C) symmetric-storage and full-storage paths produce "
              "byte-equal matrices  (n_diff = " + std::to_string(n_diff) + ")");
    }

    // ---------------- (D) on-disk file size sanity --------------------
    // Each chunk file contains:  int header (4 B) + count * bytes_per_matrix_disk_.
    // For symmetric: bytes_per_matrix_disk_ = N*(N+1)/2 * 8.
    // For full:      bytes_per_matrix_disk_ = N*N      * 8.
    {
        const std::size_t expect_sym  =
            sizeof(int) + chunk_size * (static_cast<std::size_t>(N) * (N + 1) / 2) * sizeof(double);
        const std::size_t expect_full =
            sizeof(int) + chunk_size * static_cast<std::size_t>(N) * N * sizeof(double);
        const auto sym_size  = fs::file_size(dir_sym  + "/pot_chunk_0.bin");
        const auto full_size = fs::file_size(dir_full + "/pot_chunk_0.bin");
        check(sym_size == expect_sym,
              "(D-sym)  chunk file size = " + std::to_string(sym_size)
              + "  expected " + std::to_string(expect_sym));
        check(full_size == expect_full,
              "(D-full) chunk file size = " + std::to_string(full_size)
              + "  expected " + std::to_string(expect_full));
        check(sym_size < full_size,
              "(D-cmp)  symmetric storage uses less disk: "
              + std::to_string(sym_size) + " B  <  "
              + std::to_string(full_size) + " B");
    }

    // ---------------- (E) backward compat: legacy v1 metadata loads ----
    {
        // Manually write a v1-format metadata.bin alongside the FULL
        // checkpoint chunk files.  Then load via the new code: it must
        // detect "no magic" and fall through to the legacy parser.
        const std::string dir_v1 = "./checkpoints/test_storage_symmetric_V1";
        cleanup_dir(dir_v1);
        fs::create_directories(dir_v1);
        // Copy chunk files from dir_full.
        for (const auto& e : fs::directory_iterator(dir_full)) {
            const auto fn = e.path().filename().string();
            if (fn.find("pot_chunk_") != std::string::npos) {
                fs::copy_file(e.path(), dir_v1 + "/" + fn);
            }
            if (fn == "manifest.txt" || fn == "__SUCCESS__") {
                fs::copy_file(e.path(), dir_v1 + "/" + fn);
            }
        }
        // Hand-write v1 metadata: [size_t Nr][int channels][int chunk_size][int num_chunks_written].
        const std::string mfn = dir_v1 + "/metadata.bin";
        std::FILE* f = std::fopen(mfn.c_str(), "wb");
        const std::size_t Nr_w = Nr;
        const int cw[3] = { N, chunk_size,
                            static_cast<int>((Nr + chunk_size - 1) / chunk_size) };
        std::fwrite(&Nr_w, sizeof(std::size_t), 1, f);
        std::fwrite(&cw[0], sizeof(int), 1, f);
        std::fwrite(&cw[1], sizeof(int), 1, f);
        std::fwrite(&cw[2], sizeof(int), 1, f);
        std::fclose(f);

        PotentialStorage S;
        S.set_manifest(manifest);
        const bool ok = S.initialize_from_checkpoint(Nr, N, dir_v1, chunk_size);
        check(ok, "(E) load legacy v1 (no magic prefix) checkpoint");
        check(S.is_symmetric_storage() == false,
              "(E) v1 checkpoint correctly detected as full-format (not symmetric)");

        int n_diff = 0;
        for (int ir = 0; ir < Nr; ++ir) {
            const Eigen::MatrixXd& Mr = S.get(static_cast<std::size_t>(ir));
            if ((Mr - ref[ir]).cwiseAbs().maxCoeff() != 0.0) ++n_diff;
        }
        check(n_diff == 0,
              "(E) v1 legacy checkpoint reads byte-equal to ref  (n_diff = "
              + std::to_string(n_diff) + ")");

        cleanup_dir(dir_v1);
    }

    // ---------------- (F) chunk_size mismatch safety on sym ----------
    // If the runtime requests a smaller chunk_size than what's on disk
    // AND the caller has opted OUT of the re-chunk policy (the
    // pre-2026-05-23 default), initialize_from_checkpoint rejects the
    // load -- the cross-node-size guard.  Verify it still works in v2
    // path with allow_chunk_rechunk=false.  (The default-ON re-chunk
    // path is gated separately by test_storage_chunk_rechunk; here we
    // exercise the opt-out path.)
    {
        PotentialStorage S;
        S.set_manifest(manifest);
        S.set_chunk_rechunk_allowed(false);    // opt out
        // dir_sym was written with chunk_size=3.  Ask runtime for chunk_size=2
        // -> on-disk 3 > runtime budget 2 -> reject.
        const bool ok = S.initialize_from_checkpoint(Nr, N, dir_sym, /*chunk*/2);
        check(ok == false,
              "(F) symmetric checkpoint with chunk_size=3 is rejected "
              "when runtime budget chunk_size=2 AND re-chunk is disabled "
              "(legacy cross-node-size guard, opt-out path)");
    }

    cleanup_dir(dir_sym);
    cleanup_dir(dir_full);

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " storage_symmetric  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
