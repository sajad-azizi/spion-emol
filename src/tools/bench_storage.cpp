// bench_storage.cpp -- measure write/read throughput of PotentialStorage
// at various matrix sizes. Drives a realistic picture of how much of a
// 2-day HPC run goes to disk I/O versus compute.
//
// Usage:
//   bench_storage [--dir=PATH] [--sizes=150,500,1000] [--n=3001]
//                 [--chunk=100] [--fsync]
//
// Prints, for each matrix size:
//   matrix bytes, total file-tree bytes, write time, read time,
//   write MB/s, read MB/s, matrix-fill time (Eigen).

#include "scatt/PotentialStorage.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<int> parse_sizes(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
    return v;
}

int main(int argc, char** argv) {
    std::string dir = "./bench_storage_tmp";
    std::vector<int> sizes = {150, 300, 500, 1000};
    std::size_t Nr = 3001;
    int chunk = 100;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--dir=", 6) == 0) dir = argv[i] + 6;
        else if (std::strncmp(argv[i], "--sizes=", 8) == 0) sizes = parse_sizes(argv[i] + 8);
        else if (std::strncmp(argv[i], "--n=", 4) == 0) Nr = static_cast<std::size_t>(std::stoul(argv[i] + 4));
        else if (std::strncmp(argv[i], "--chunk=", 8) == 0) chunk = std::stoi(argv[i] + 8);
        else {
            std::cerr << "unknown arg: " << argv[i] << "\n";
            std::cerr << "usage: bench_storage [--dir=PATH] [--sizes=150,300,500] "
                         "[--n=3001] [--chunk=100]\n";
            return 2;
        }
    }

    fs::remove_all(dir);
    fs::create_directories(dir);

    std::cout << "=== PotentialStorage I/O benchmark ===\n";
    std::cout << "  output: " << dir << "\n";
    std::cout << "  Nr    : " << Nr << "\n";
    std::cout << "  chunk : " << chunk << "\n";
    std::cout << "  sizes : ";
    for (auto s : sizes) std::cout << s << " ";
    std::cout << "\n\n";

    std::cout << "  ch  |  mat MB |  tot GB |   fill s |   write s | w MB/s |    read s |  r MB/s | match?\n";
    std::cout << "  ----|---------|---------|----------|-----------|--------|-----------|---------|-------\n";

    for (int ch : sizes) {
        const std::size_t mat_bytes = static_cast<std::size_t>(ch) * ch * sizeof(double);
        const std::size_t tot_bytes = mat_bytes * Nr;
        const std::size_t bytes_per_chunk =
            static_cast<std::size_t>(chunk) * mat_bytes;

        // Safety: don't run something silly.
        if (tot_bytes > 50ULL * 1024ULL * 1024ULL * 1024ULL) {
            std::cout << "  " << ch << "   (skipped: tot_bytes "
                      << tot_bytes / (1024ULL*1024ULL*1024ULL) << " GB > 50 GB limit)\n";
            continue;
        }

        // ---- 1. fill matrix in RAM (not part of disk I/O) ----
        Eigen::MatrixXd M(ch, ch);
        auto t0 = std::chrono::steady_clock::now();
        // Non-trivial pattern so compressibility/zero-detection doesn't skew.
        for (int i = 0; i < ch; ++i)
            for (int j = 0; j < ch; ++j)
                M(i, j) = 1e-6 * (i * 31.0 + j * 7.0 + (i == j ? 1000.0 : 0.0));
        const double fill_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        // ---- 2. write (via PotentialStorage DISK mode) ----
        const std::string ckdir = dir + "/ch_" + std::to_string(ch);
        fs::remove_all(ckdir);
        fs::create_directories(ckdir);
        scatt::PotentialStorage ps;
        ps.initialize_for_write(Nr, ch, scatt::PotentialStorage::Mode::DISK,
                                ckdir, chunk);

        t0 = std::chrono::steady_clock::now();
        for (std::size_t ir = 0; ir < Nr; ++ir) {
            ps.store(ir, M);
        }
        ps.finalize_write();
        const double write_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double write_MBs =
            (tot_bytes / (1024.0 * 1024.0)) / std::max(write_s, 1e-9);

        // ---- 3. read (sequential) ----
        t0 = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (std::size_t ir = 0; ir < Nr; ++ir) {
            const auto& R = ps.get(ir);
            checksum += R(0, 0) + R(ch - 1, ch - 1);
        }
        const double read_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double read_MBs =
            (tot_bytes / (1024.0 * 1024.0)) / std::max(read_s, 1e-9);

        // Validate: checksum matches closed form.
        const double expected =
            static_cast<double>(Nr) * (M(0, 0) + M(ch - 1, ch - 1));
        const bool match = std::abs(checksum - expected) <
                           1e-6 * std::abs(expected);

        std::printf("  %-4d|%8.1f |%8.2f |%9.3f |%10.3f |%7.1f |%10.3f |%8.1f | %s\n",
                    ch,
                    mat_bytes / 1024.0 / 1024.0,
                    tot_bytes / 1024.0 / 1024.0 / 1024.0,
                    fill_s, write_s, write_MBs, read_s, read_MBs,
                    match ? "ok" : "MISMATCH");
        std::cout.flush();

        // Clean up.
        fs::remove_all(ckdir);
        (void) bytes_per_chunk;
    }

    fs::remove_all(dir);
    return 0;
}
