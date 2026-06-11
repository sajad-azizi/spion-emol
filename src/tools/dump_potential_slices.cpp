// dump_potential_slices.cpp -- write selected V(r) matrix-element slices
// to a CSV file that a Python plotter can consume.
//
// Emits the following columns (all in atomic units):
//
//   r                     radial coordinate (Bohr)
//   V_00_00               V_{(0,0),(0,0)}(r)  = monopole of V(r,Omega)
//                         (angular average, since Y^R_{0,0} = 1/sqrt(4pi))
//   centrif_L             ell*(ell+1)/(2 r^2)   (for various ell)
//   V_1m_1m               V_{(1,-1),(1,-1)}(r)  p_y diagonal
//   V_10_10               V_{(1, 0),(1, 0)}(r)  p_z diagonal
//   V_1p_1p               V_{(1,+1),(1,+1)}(r)  p_x diagonal
//   V_10_10_noCF          V_{(1,0),(1,0)} - centrif_1 (non-centrifugal part)
//   V_20_20_noCF          V_{(2,0),(2,0)} - centrif_2
//   V_00_10               V_{(0,0),(1,0)}(r)    s-p_z off-diagonal
//                         (monopole-to-p_z coupling)
//   V_10_20               V_{(1,0),(2,0)}(r)    p_z-d_{z^2} off-diagonal
//   V_1m_1p               V_{(1,-1),(1,+1)}(r)  p_y-p_x off-diagonal
//                         (nonzero iff the potential has no C4 about z)
//
// Usage:
//   dump_potential_slices <hdf5> <csv_out> [--lmax L] [--ir-stride N]
//
// L defaults to 6 (channels=49), N defaults to 1 (every point).

#include "io/HDF5Reader.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/Potentials.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using scatt::Parameters;
using scatt::Potentials;
using scatt::StorageMode;
using scatt::angular::lm_to_idx;
using scatt::io::HDF5Reader;
using scatt::io::PreprocData;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <hdf5> <csv_out> [--lmax L] [--ir-stride N]\n";
        return 2;
    }
    const std::string hdf5_path = argv[1];
    const std::string csv_path  = argv[2];

    int L        = 6;
    int ir_stride = 1;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](int o) { return (i + o < argc) ? argv[i + o] : ""; };
        if      (a == "--lmax")      { L         = std::stoi(next(1)); ++i; }
        else if (a == "--ir-stride") { ir_stride = std::stoi(next(1)); ++i; }
        else { std::cerr << "unknown arg: " << a << "\n"; return 2; }
    }
    if (L < 2) { std::cerr << "need L >= 2 to reach d channels\n"; return 2; }

    HDF5Reader reader(hdf5_path);
    PreprocData data = reader.load_all_except_V_H();

    Parameters params;
    params.r_min          = data.rmin;
    params.dr             = data.dr;
    params.N_grid         = data.Nr;
    params.Lmax_sce       = data.Lmax_sce;
    params.l_max_continuum = L;
    data.V_H = reader.load_V_H(params.n_exp());

    Potentials pot(params);
    pot.build(data, StorageMode::MEMORY, "", /*verbose=*/false);

    const int idx_00 = lm_to_idx(0,  0);
    const int idx_1m = lm_to_idx(1, -1);   // p_y
    const int idx_10 = lm_to_idx(1,  0);   // p_z
    const int idx_1p = lm_to_idx(1, +1);   // p_x
    const int idx_20 = lm_to_idx(2,  0);   // d_{z^2}

    std::ofstream csv(csv_path);
    if (!csv) { std::cerr << "cannot open " << csv_path << "\n"; return 1; }
    csv << std::setprecision(12);
    csv << "# Nr=" << params.N_grid << "  channels=" << params.channels()
        << "  L=" << L << "  ir_stride=" << ir_stride << "\n";
    csv << "r,V_00_00,centrif_0,centrif_1,centrif_2,"
           "V_1m_1m,V_10_10,V_1p_1p,"
           "V_10_10_noCF,V_20_20_noCF,"
           "V_00_10,V_10_20,V_1m_1p\n";

    for (std::size_t ir = 0; ir < params.N_grid; ir += static_cast<std::size_t>(ir_stride)) {
        const double r = params.r(ir);
        // get() returns ref to MEMORY-cache slot -- no copy.
        const Eigen::MatrixXd& V = pot.get(ir);

        const double inv_r2    = (r > 1e-14) ? 1.0 / (r * r) : 0.0;
        const double centrif_0 = 0.0;
        const double centrif_1 = 1.0 * 2.0 * 0.5 * inv_r2;   // l(l+1)/(2 r^2), l=1
        const double centrif_2 = 2.0 * 3.0 * 0.5 * inv_r2;   // l=2

        csv << r << ","
            << V(idx_00, idx_00) << ","
            << centrif_0 << "," << centrif_1 << "," << centrif_2 << ","
            << V(idx_1m, idx_1m) << ","
            << V(idx_10, idx_10) << ","
            << V(idx_1p, idx_1p) << ","
            << (V(idx_10, idx_10) - centrif_1) << ","
            << (V(idx_20, idx_20) - centrif_2) << ","
            << V(idx_00, idx_10) << ","
            << V(idx_10, idx_20) << ","
            << V(idx_1m, idx_1p) << "\n";
    }
    csv.close();

    std::cout << "[dump_potential_slices] wrote " << csv_path << "\n";
    return 0;
}
