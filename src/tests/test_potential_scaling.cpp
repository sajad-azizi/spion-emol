// test_potential_scaling.cpp -- informational stress test, NOT a pass/fail.
//
// For each (l_max_continuum) in a sweep, build the sparse Gaunt matrix
// and time one V(r) assembly. This lets us read off:
//   - time to build the Gaunt table (scales as O(L^6 * gaunt_per_triplet))
//   - memory of the sparse Gaunt matrix
//   - time for one ON_DEMAND V(r) assembly (sparse matvec + dense unpack)
// so that we can project the cost at L=100/L=300 for the eventual Numerov
// sweeps.
//
// This test does NOT run the full V(r) assembly for all r -- that would
// take hours and isn't needed for the capacity audit. It just measures
// one radial point.
//
// Usage:  test_potential_scaling <small_h5>  [L1 L2 L3 ...]
// Example: test_potential_scaling h2o.h5 10 15 20 25 30

#include "io/HDF5Reader.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using scatt::Parameters;
using scatt::Potentials;
using scatt::StorageMode;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static std::string human_bytes(std::size_t b) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int u = 0; double v = static_cast<double>(b);
    while (v > 1024.0 && u < 4) { v /= 1024.0; ++u; }
    char buf[64]; std::snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <hdf5> [L_cont ...]\n"; return 2;
    }
    const std::string path = argv[1];

    std::vector<int> L_list;
    for (int i = 2; i < argc; ++i) L_list.push_back(std::stoi(argv[i]));
    if (L_list.empty()) L_list = {6, 10, 15, 20};

    HDF5Reader reader(path);
    PreprocData data = reader.load_all_except_V_H();

    std::cout << std::left
              << std::setw(6)  << "Lcont"
              << std::setw(10) << "channels"
              << std::setw(10) << "n_exp"
              << std::setw(12) << "V_H_bytes"
              << std::setw(14) << "Gaunt_nnz"
              << std::setw(14) << "Gaunt_bytes"
              << std::setw(14) << "t_build_gaunt"
              << std::setw(14) << "t_one_V(r)"
              << "\n";
    std::cout << std::string(96, '-') << "\n";

    for (int L : L_list) {
        Parameters params;
        params.r_min    = data.rmin;
        params.dr       = data.dr;
        params.N_grid   = data.Nr;
        params.Lmax_sce = data.Lmax_sce;
        params.l_max_continuum = L;

        // Skip if HDF5 doesn't have enough SCE angular content.
        if (params.Lmax_sce < params.l_exp_max()) {
            std::cout << L << "    (skipped: HDF5 Lmax_sce="
                      << params.Lmax_sce << " < 2*L=" << params.l_exp_max() << ")\n";
            continue;
        }

        // Load just the n_exp rows of V_H we need.
        data.V_H = reader.load_V_H(/*V_H_max_rows=*/params.n_exp());
        const std::size_t V_H_bytes = static_cast<std::size_t>(data.V_H.rows())
                                    * data.V_H.cols() * sizeof(double);

        Potentials pot(params);
        auto t0 = std::chrono::steady_clock::now();
        pot.build(data, StorageMode::ON_DEMAND, /*checkpoint_dir=*/"",
                  /*verbose=*/false);
        auto t1 = std::chrono::steady_clock::now();

        // One V(r) assembly.
        auto t2 = std::chrono::steady_clock::now();
        Eigen::MatrixXd V = pot.V_at(params.N_grid / 2);
        auto t3 = std::chrono::steady_clock::now();
        (void)V;

        const int          channels = params.channels();
        const std::size_t  nnz      = pot.gaunt_nonzeros();
        const std::size_t  gbytes   = pot.gaunt_memory_bytes();
        const double       dt_build = std::chrono::duration<double>(t1 - t0).count();
        const double       dt_one_V = std::chrono::duration<double>(t3 - t2).count();

        std::cout << std::left
                  << std::setw(6)  << L
                  << std::setw(10) << channels
                  << std::setw(10) << params.n_exp()
                  << std::setw(12) << human_bytes(V_H_bytes)
                  << std::setw(14) << nnz
                  << std::setw(14) << human_bytes(gbytes)
                  << std::setw(14) << (std::to_string(dt_build) + " s")
                  << std::setw(14) << (std::to_string(dt_one_V) + " s")
                  << "\n";
    }

    std::cout << "\nProjections to large Lmax at fixed preprocessing grid:\n"
              << "  Gaunt memory ~ L^5-L^6 depending on sparsity; n_pairs * n_exp ~ L^6.\n"
              << "  one V(r) build: sparse matvec O(nnz) + dense unpack O(channels^2).\n";
    return 0;
}
