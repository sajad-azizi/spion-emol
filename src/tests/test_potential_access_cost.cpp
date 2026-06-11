// test_potential_access_cost.cpp -- prove that in MEMORY mode we do NOT
// recompute V(r) per call. Nothing fancy: time N_rep calls of get() in
// MEMORY mode vs V_at() in ON_DEMAND mode and show the ratio.
//
// Concretely, get() in MEMORY should be sub-microsecond (cache hit on
// the pre-allocated slot), while V_at() in ON_DEMAND takes the full
// Gaunt-matvec + dense-unpack time (measurable ms even at small L).
//
// This is a pass/fail test on a RATIO check, not an absolute time --
// CI hosts vary widely. We just require MEMORY to be at least 100x
// faster than ON_DEMAND. For real C8F8 / L=100 the ratio is 10^6+.

#include "io/HDF5Reader.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>

using scatt::Parameters;
using scatt::Potentials;
using scatt::StorageMode;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

static double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <h2o_hdf5>\n"; return 2;
    }
    const std::string path = argv[1];
    HDF5Reader reader(path);
    PreprocData data = reader.load_all_except_V_H();

    Parameters params;
    params.r_min    = data.rmin;
    params.dr       = data.dr;
    params.N_grid   = data.Nr;
    params.Lmax_sce = data.Lmax_sce;
    params.l_max_continuum = 10;     // channels = 121
    data.V_H = reader.load_V_H(/*V_H_max_rows=*/params.n_exp());

    const int N_rep = 2000;
    const std::size_t ir_hit = params.N_grid / 2;

    // --- MEMORY: build once, then many get() calls. ---
    Potentials pot_mem(params);
    pot_mem.build(data, StorageMode::MEMORY, /*ckpt=*/"", /*verbose=*/false);

    // Warm up + sanity read.
    {
        const auto& V = pot_mem.get(ir_hit);
        if (V.rows() != params.channels()) return 1;
    }

    double t0 = now_sec();
    double checksum_mem = 0.0;
    for (int k = 0; k < N_rep; ++k) {
        const auto& V = pot_mem.get(ir_hit);
        checksum_mem += V(0, 0);    // prevent the compiler from eliding the call
    }
    const double dt_mem = now_sec() - t0;

    // --- ON_DEMAND: each V_at rebuilds the whole matrix. ---
    Potentials pot_od(params);
    pot_od.build(data, StorageMode::ON_DEMAND, /*ckpt=*/"", /*verbose=*/false);

    t0 = now_sec();
    double checksum_od = 0.0;
    for (int k = 0; k < N_rep; ++k) {
        Eigen::MatrixXd V = pot_od.V_at(ir_hit);
        checksum_od += V(0, 0);
    }
    const double dt_od = now_sec() - t0;

    std::cout << "N_rep         = " << N_rep << "\n";
    std::cout << "MEMORY  get():  " << (dt_mem * 1e6 / N_rep) << " us/call   (total " << dt_mem << " s)\n";
    std::cout << "ON_DEMAND V_at(): " << (dt_od * 1e6 / N_rep) << " us/call   (total " << dt_od << " s)\n";
    std::cout << "checksum_mem = " << checksum_mem << ",  checksum_od = " << checksum_od << "\n";
    const double ratio = dt_od / std::max(dt_mem, 1e-12);
    std::cout << "speedup (MEMORY vs ON_DEMAND) = " << ratio << "x\n";

    // Values must match (MEMORY and ON_DEMAND have to build the same V).
    const double diff = std::abs(checksum_mem / N_rep - checksum_od / N_rep);
    std::cout << "V(ir_hit)[0,0] MEMORY vs ON_DEMAND diff = " << diff << "\n";
    if (diff > 1e-10) {
        std::cerr << "FAIL: V mismatch between modes\n";
        return 1;
    }

    if (ratio < 100.0) {
        std::cerr << "FAIL: MEMORY is not at least 100x faster than ON_DEMAND "
                     "-- get() is probably copying instead of caching.\n";
        return 1;
    }
    std::cout << "PASS: MEMORY cache hit is " << ratio << "x faster than ON_DEMAND rebuild.\n";
    return 0;
}
