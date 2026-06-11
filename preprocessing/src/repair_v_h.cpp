// repair_v_h.cpp -- in-place fix for the V_H NaN bug in already-built
// preproc HDF5 files.
//
// What it does:
//   * Opens an existing preproc.h5 in read/write mode.
//   * Reads /rho/rho_lm (the SCE-projected density: the expensive part of
//     the original 6-hour run is the orbital projection AND the density
//     projection -- both are already stored).
//   * Reads /potential/V_en (correct, never had a bug).
//   * Reads /potential/V_local_exchange (correct; zero when --exchange
//     none was passed).
//   * Reads /grid/dr, /grid/N, /grid/r, /angular/Lmax.
//   * Recomputes V_H = build_V_H(rho_lm, ...) -- now NaN-free thanks to
//     the l_safe cap + isfinite scrub in Hartree.hpp.
//   * Recomputes V_total = V_en + V_H + V_x.
//   * Recomputes J = 0.5 * <rho|V_H>.
//   * Deletes the broken /potential/V_H and /potential/V_total_local
//     datasets and rewrites them with the corrected values.
//   * Updates /meta/J_classical_hartree if it exists.
//
// Safety: refuses to overwrite if the recomputed V_H still contains
// non-finite values, or if /rho/rho_lm cannot be found (file might be
// from an older preproc version).
//
// Usage:
//     repair_v_h /path/to/c8f8.preproc.h5

#include "potential/Hartree.hpp"
#include "potential/Vnuclear.hpp"        // pulls in inner_product_radial helper
#include "sce/RadialGrid.hpp"

#include <hdf5.h>

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void _check(hid_t h, const char* what) {
    if (h < 0) throw std::runtime_error(std::string("HDF5: ") + what + " failed");
}

// Read a scalar int from a dataset path.
static long long read_scalar_int(hid_t f, const char* path) {
    hid_t ds = H5Dopen2(f, path, H5P_DEFAULT);
    _check(ds, path);
    long long v = 0;
    H5Dread(ds, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(ds);
    return v;
}
static double read_scalar_double(hid_t f, const char* path) {
    hid_t ds = H5Dopen2(f, path, H5P_DEFAULT);
    _check(ds, path);
    double v = 0;
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(ds);
    return v;
}
// Read a 1D double dataset.
static std::vector<double> read_1d_double(hid_t f, const char* path) {
    hid_t ds = H5Dopen2(f, path, H5P_DEFAULT);
    _check(ds, path);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    std::vector<double> v(dims[0]);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(sp); H5Dclose(ds);
    return v;
}
// Read a 2D row-major double dataset and return as Eigen column-major
// matrix.  Same convention as the production HDF5Reader.
static Eigen::MatrixXd read_matrix(hid_t f, const char* path) {
    hid_t ds = H5Dopen2(f, path, H5P_DEFAULT);
    _check(ds, path);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    const hsize_t rows = dims[0], cols = dims[1];
    std::vector<double> buf(rows * cols);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    Eigen::MatrixXd M(rows, cols);
    for (hsize_t i = 0; i < rows; ++i)
        for (hsize_t j = 0; j < cols; ++j)
            M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))
                = buf[i * cols + j];
    H5Sclose(sp); H5Dclose(ds);
    return M;
}
// Delete a dataset / link if it exists.
static void delete_if_exists(hid_t f, const char* path) {
    if (H5Lexists(f, path, H5P_DEFAULT) > 0) {
        H5Ldelete(f, path, H5P_DEFAULT);
    }
}
// Write Eigen column-major matrix as 2D row-major HDF5 dataset.  Matches
// main_preprocess.cpp's `write_matrix` lambda.
static void write_matrix(hid_t f, const char* path, const Eigen::MatrixXd& M) {
    const hsize_t rows = static_cast<hsize_t>(M.rows());
    const hsize_t cols = static_cast<hsize_t>(M.cols());
    std::vector<double> rowmajor(rows * cols);
    for (hsize_t i = 0; i < rows; ++i)
        for (hsize_t j = 0; j < cols; ++j)
            rowmajor[i * cols + j] =
                M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    hsize_t dims[2] = {rows, cols};
    hid_t sp = H5Screate_simple(2, dims, nullptr);
    hid_t ds = H5Dcreate2(f, path, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    _check(ds, path);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             rowmajor.data());
    H5Dclose(ds); H5Sclose(sp);
}
static void write_scalar_double(hid_t f, const char* path, double v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t ds = H5Dcreate2(f, path, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    _check(ds, path);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(ds); H5Sclose(sp);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <potentials.h5 | preproc.h5>\n";
        std::cerr << "\n"
                     "  In-place repair of /potential/V_H, /potential/V_total_local,\n"
                     "  and /meta/J_classical_hartree from the SCE density already\n"
                     "  stored at /rho/rho_lm.  Skips re-running the (expensive)\n"
                     "  orbital SCE projection.\n"
                     "\n"
                     "  Pass either of:\n"
                     "    * the new split potentials file  (e.g. c8f8.potentials.h5)\n"
                     "    * the legacy combined preproc file (e.g. c8f8.preproc.h5)\n"
                     "  -- both contain /rho/rho_lm and /potential/V_*.\n";
        return 2;
    }
    const std::string path = argv[1];
    std::cerr << "[repair_v_h] opening: " << path << "\n";

    // Suppress HDF5's own error printing (we throw nice messages on failure).
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    // Disable POSIX file locking when opening for R/W.  LRZ's WORK / SCRATCH
    // are Lustre/HSM-backed and frequently reject the lock that HDF5 tries
    // to take by default ("H5Fopen failed" with no useful detail).  This
    // is the same situation that the env var HDF5_USE_FILE_LOCKING=FALSE
    // fixes; we prefer to bake it into the tool so the user doesn't have
    // to remember it.  Available in HDF5 1.10.7+; LRZ's hdf5/1.10.11-intel25
    // module includes it.
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    _check(fapl, "H5Pcreate FILE_ACCESS");
#if (H5_VERS_MAJOR > 1) \
 || (H5_VERS_MAJOR == 1 && H5_VERS_MINOR > 10) \
 || (H5_VERS_MAJOR == 1 && H5_VERS_MINOR == 10 && H5_VERS_RELEASE >= 7)
    // (use_file_locking=false, ignore_disabled_file_locks=true)
    H5Pset_file_locking(fapl, false, true);
#endif

    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDWR, fapl);
    H5Pclose(fapl);
    if (f < 0) {
        std::cerr << "[repair_v_h] H5Fopen R/W failed.\n"
                     "  Most common causes on LRZ:\n"
                     "    1. File is currently open by another process\n"
                     "       (close any HDF5 viewers, h5dump sessions, ...).\n"
                     "    2. POSIX locking is disabled on the filesystem.\n"
                     "       Try:  export HDF5_USE_FILE_LOCKING=FALSE\n"
                     "       and re-run.\n"
                     "    3. File is on a read-only mount.\n"
                  << "  Path: " << path << "\n";
        return 1;
    }

    try {
        // ---- 1. Pull in everything needed ----
        const long long Nr_ll = read_scalar_int(f, "/grid/N");
        const int Nr = static_cast<int>(Nr_ll);
        const double rmin = read_scalar_double(f, "/grid/rmin");
        const double dr   = read_scalar_double(f, "/grid/dr");
        const int Lmax    = static_cast<int>(read_scalar_int(f, "/angular/Lmax"));

        std::cerr << "[repair_v_h] grid: rmin=" << rmin << " dr=" << dr
                  << " Nr=" << Nr << "  Lmax=" << Lmax << "\n";

        if (H5Lexists(f, "/rho/rho_lm", H5P_DEFAULT) <= 0) {
            throw std::runtime_error(
                "/rho/rho_lm not found in HDF5 -- this preproc.h5 is from "
                "an older version that didn't store the SCE density. "
                "You will need to re-run preprocessing from scratch.");
        }

        std::cerr << "[repair_v_h] reading /rho/rho_lm ...\n";
        Eigen::MatrixXd rho = read_matrix(f, "/rho/rho_lm");
        if (rho.cols() != Nr) {
            throw std::runtime_error("rho_lm column count != Nr");
        }
        std::cerr << "[repair_v_h] rho_lm shape = (" << rho.rows()
                  << ", " << rho.cols() << ")\n";

        std::cerr << "[repair_v_h] reading /potential/V_en ...\n";
        Eigen::MatrixXd V_en = read_matrix(f, "/potential/V_en");

        Eigen::MatrixXd V_x;
        if (H5Lexists(f, "/potential/V_local_exchange", H5P_DEFAULT) > 0) {
            std::cerr << "[repair_v_h] reading /potential/V_local_exchange ...\n";
            V_x = read_matrix(f, "/potential/V_local_exchange");
        } else {
            std::cerr << "[repair_v_h] /potential/V_local_exchange missing; assuming zero\n";
            V_x = Eigen::MatrixXd::Zero(rho.rows(), Nr);
        }

        // ---- 2. Recompute V_H using the fixed Hartree.hpp ----
        auto rg = preproc::sce::RadialGrid::build(rmin, dr, Nr);
        auto t0 = std::chrono::steady_clock::now();
        Eigen::MatrixXd V_H = preproc::potential::build_V_H(rho, rg, Lmax);
        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "[repair_v_h] build_V_H done in "
                  << std::chrono::duration<double>(t1 - t0).count() << " s\n";

        // ---- 3. Validate: every entry must be finite ----
        std::size_t n_nonfinite = 0;
        for (int i = 0; i < V_H.rows(); ++i)
            for (int j = 0; j < V_H.cols(); ++j)
                if (!std::isfinite(V_H(i, j))) ++n_nonfinite;
        if (n_nonfinite > 0) {
            std::ostringstream os;
            os << "[repair_v_h] ABORT: recomputed V_H still has "
               << n_nonfinite << " non-finite values.\n"
               << "             Lower Lmax_sce or check the rho_lm content "
               << "in the input HDF5.";
            throw std::runtime_error(os.str());
        }

        // ---- 4. Diagnostics: classical-Hartree integral J ----
        const double J = 0.5 * preproc::potential::inner_product_radial(rho, V_H, rg);
        if (!std::isfinite(J)) {
            throw std::runtime_error("[repair_v_h] ABORT: 0.5*<rho|V_H> is non-finite");
        }
        std::cerr << "[repair_v_h] new J = 0.5 * <rho|V_H> = " << J << "  (Hartree)\n";

        // ---- 5. Rebuild V_total ----
        Eigen::MatrixXd V_total = V_en + V_H + V_x;

        // ---- 6. Overwrite the file in place ----
        std::cerr << "[repair_v_h] overwriting /potential/V_H ...\n";
        delete_if_exists(f, "/potential/V_H");
        write_matrix(f, "/potential/V_H", V_H);

        std::cerr << "[repair_v_h] overwriting /potential/V_total_local ...\n";
        delete_if_exists(f, "/potential/V_total_local");
        write_matrix(f, "/potential/V_total_local", V_total);

        if (H5Lexists(f, "/meta", H5P_DEFAULT) > 0) {
            std::cerr << "[repair_v_h] updating /meta/J_classical_hartree ...\n";
            delete_if_exists(f, "/meta/J_classical_hartree");
            write_scalar_double(f, "/meta/J_classical_hartree", J);
        }

        // ---- 7. Force HDF5 to flush dirty buffers to disk before close.
        // Without this, on some filesystems (Lustre / network mounts) a
        // crash between H5Fclose and physical fsync could leave the new
        // datasets in cache with the file image still showing the old ones.
        // H5F_SCOPE_GLOBAL flushes the file AND any associated mounts.
        if (H5Fflush(f, H5F_SCOPE_GLOBAL) < 0) {
            throw std::runtime_error("H5Fflush failed; on-disk image may not match");
        }

        // ---- 8. Defensive read-back verification.  Re-open the dataset
        // we just wrote and check the first few values are finite.  Any
        // failure here means the write didn't take effect (would be
        // a silent corruption otherwise).
        {
            hid_t ds = H5Dopen2(f, "/potential/V_H", H5P_DEFAULT);
            _check(ds, "H5Dopen post-write /potential/V_H");
            hid_t sp = H5Dget_space(ds);
            hsize_t dims[2] = {0, 0};
            H5Sget_simple_extent_dims(sp, dims, nullptr);
            std::vector<double> first_row(dims[1]);
            // Read just the first row.
            hsize_t start[2] = {0, 0}, count[2] = {1, dims[1]};
            H5Sselect_hyperslab(sp, H5S_SELECT_SET, start, nullptr, count, nullptr);
            hid_t mem_sp = H5Screate_simple(2, count, nullptr);
            H5Dread(ds, H5T_NATIVE_DOUBLE, mem_sp, sp, H5P_DEFAULT, first_row.data());
            H5Sclose(mem_sp);
            H5Sclose(sp);
            H5Dclose(ds);
            int n_bad = 0;
            for (double v : first_row) if (!std::isfinite(v)) ++n_bad;
            if (n_bad > 0) {
                throw std::runtime_error(
                    "[repair_v_h] readback after write FOUND " +
                    std::to_string(n_bad) + " non-finite values in row 0; "
                    "write did NOT take effect on disk.");
            }
            std::cerr << "[repair_v_h] readback OK: V_H row 0 finite "
                         "(" << first_row.size() << " values, sample "
                      << first_row.front() << " ... " << first_row.back() << ")\n";
        }

        H5Fclose(f);
        std::cerr << "[repair_v_h] done. File is now self-consistent.\n";
        return 0;
    } catch (const std::exception& e) {
        H5Fclose(f);
        std::cerr << "[repair_v_h] FAILED: " << e.what() << "\n";
        return 1;
    }
}
