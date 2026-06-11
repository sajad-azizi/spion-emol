// HDF5Reader.hpp -- read the preprocessing artifact produced by
// preprocessing/src/main_preprocess.cpp. Mirror of the writer's layout,
// so the file schema lives in exactly two places and one can audit both.
//
// What we load for the scattering stage:
//   /grid/r, /grid/dr, /grid/rmin, /grid/N                   (radial grid)
//   /angular/Lmax                                            (SCE Lmax)
//   /geometry/Z, /geometry/xyz_bohr                          (atoms, translated so origin is 0)
//   /potential/V_H                                           (Nlm, Nr) V_ee_lm(r)
//   /orbitals/psi_lm, /orbitals/energies_hartree, /orbitals/occupations
//   /orbitals/n_alpha, /orbitals/n_occ_alpha, /orbitals/n_sce, /orbitals/molden_index
//   /initial_state/psi_lm, /initial_state/energy_hartree, /initial_state/occupation
//   /polarizability/alpha_tensor (if present)
//
// This is a thin RAII wrapper over the HDF5 C API, matching the style of
// preprocessing/src/io/HDF5Writer.hpp so code reviews only have one I/O
// idiom to understand.

#pragma once

#include <hdf5.h>

#include <Eigen/Dense>

#include "scatt/SystemMemory.hpp"

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scatt::io {

inline void _check(hid_t h, const char* what) {
    if (h < 0) throw std::runtime_error(std::string("HDF5: ") + what + " failed");
}

// All the data a scattering stage consumes.
struct PreprocData {
    // Radial grid (uniform).
    double          rmin = 0.0;
    double          dr   = 0.0;
    std::size_t     Nr   = 0;

    // SCE angular cutoff.
    int             Lmax_sce = 0;            // n_lambda = (Lmax_sce + 1)^2

    // Geometry (positions are TRANSLATED so origin is (0,0,0)).
    struct Atom { int Z; double x, y, z; };
    std::vector<Atom>  atoms;

    // V_ee(r) spherically expanded: shape (Nlm_sce, Nr), row-major.
    // Stored as column-major Eigen for cheap column access (V at one r).
    Eigen::MatrixXd    V_H;                  // (Nlm_sce, Nr)

    // Occupied + virtual orbitals (selection written at preprocessing time).
    int                n_alpha      = 0;     // all alpha MOs in molden
    int                n_occ_alpha  = 0;     // occupied alpha MOs
    int                n_sce        = 0;     // how many were SCE'd
    // psi_lm is stored at the truncated extent (n_sce * Nlm_orb_store,
    // Nr_orb_store) -- both can be smaller than (n_sce * Nlm_sce, Nr)
    // when preprocess_molden was run with --orb-lmax / --orb-rmax.
    // Older HDF5 files (no Lmax_orb_store recorded) default to full
    // size, so the layout is always self-describing.
    int                Lmax_orb_store = 0;   // == Lmax_sce when not truncated
    int                Nlm_orb_store  = 0;   // (Lmax_orb_store + 1)^2
    int                Nr_orb_store   = 0;   // <= Nr when not truncated
    Eigen::MatrixXd    psi_lm;               // (n_sce * Nlm_orb_store, Nr_orb_store)
    std::vector<double> orb_energies;
    std::vector<double> orb_occupations;
    std::vector<int>    orb_molden_index;

    // Optional anion SOMO (initial state for dipole matrix elements).
    bool               has_initial_state = false;
    Eigen::MatrixXd    init_state_psi_lm;   // (Nlm_sce, Nr)
    double             init_state_energy = 0.0;
    double             init_state_occ    = 0.0;

    // Optional polarizability tensor (au).
    bool               has_polarizability = false;
    Eigen::Matrix3d    alpha_tensor;
    double             alpha_iso = 0.0;
};

class HDF5Reader {
public:
    // Single-arg constructor.  Auto-detects layout:
    //   * If `path` ends in ".orbitals.h5" and the sibling
    //     "<stem>.potentials.h5" exists, opens BOTH files (split layout
    //     produced by preprocess_molden).
    //   * If `path` ends in ".potentials.h5" same idea, sibling is the
    //     orbitals file.
    //   * Else: treats `path` as a legacy combined preproc.h5 (everything
    //     in one file) and opens it once.
    // Reads of /orbitals/* and /initial_state/* are routed to the orbitals
    // file; /potential/*, /rho/*, /polarizability/*, /meta/* go to the
    // potentials file; /grid/*, /angular/*, /geometry/* are present in
    // both files and read from the orbitals file (arbitrary; identical).
    explicit HDF5Reader(const std::string& path) {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

        auto ends_with = [](const std::string& s, const std::string& suf) {
            return s.size() >= suf.size() &&
                   s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
        };
        const std::string ORB = ".orbitals.h5";
        const std::string POT = ".potentials.h5";

        std::string orb_path, pot_path;
        if (ends_with(path, ORB)) {
            const std::string stem = path.substr(0, path.size() - ORB.size());
            orb_path = path;
            pot_path = stem + POT;
        } else if (ends_with(path, POT)) {
            const std::string stem = path.substr(0, path.size() - POT.size());
            pot_path = path;
            orb_path = stem + ORB;
        }

        const bool split_layout =
            !orb_path.empty() &&
            !pot_path.empty() &&
            (H5Fis_hdf5(orb_path.c_str()) > 0) &&
            (H5Fis_hdf5(pot_path.c_str()) > 0);

        if (split_layout) {
            file_orb_ = H5Fopen(orb_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            _check(file_orb_, ("H5Fopen " + orb_path).c_str());
            file_pot_ = H5Fopen(pot_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            _check(file_pot_, ("H5Fopen " + pot_path).c_str());
            owned_pot_ = true;
            file_      = file_orb_;        // alias for back-compat reads
        } else {
            // Legacy combined-file mode.
            file_      = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            _check(file_, ("H5Fopen " + path).c_str());
            file_orb_  = file_;
            file_pot_  = file_;
            owned_pot_ = false;
        }
    }
    // For paths starting with one of these prefixes, the dataset lives in
    // the potentials file (in split layout).  Everything else goes to the
    // orbitals file (which also contains the shared /grid, /angular,
    // /geometry).
    static bool path_is_potential_side(const std::string& p) {
        return p.rfind("/potential/",      0) == 0
            || p.rfind("/rho/",            0) == 0
            || p.rfind("/polarizability/", 0) == 0
            || p.rfind("/meta/",           0) == 0;
    }
    hid_t file_for(const std::string& p) const {
        return path_is_potential_side(p) ? file_pot_ : file_orb_;
    }
    ~HDF5Reader() {
        // In split layout we own two handles; in legacy mode file_orb_,
        // file_pot_ and file_ all alias the same handle and we close once.
        if (owned_pot_ && file_pot_ >= 0 && file_pot_ != file_orb_)
            H5Fclose(file_pot_);
        if (file_orb_ >= 0)
            H5Fclose(file_orb_);
    }
    HDF5Reader(const HDF5Reader&) = delete;
    HDF5Reader& operator=(const HDF5Reader&) = delete;

    double read_scalar_double(const std::string& path) const {
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        double v;
        H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
        H5Dclose(ds);
        return v;
    }
    long long read_scalar_int(const std::string& path) const {
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        long long v;
        H5Dread(ds, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
        H5Dclose(ds);
        return v;
    }
    std::vector<double> read_1d_double(const std::string& path) const {
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        hid_t sp = H5Dget_space(ds);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        std::vector<double> v(dims[0]);
        H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
        H5Sclose(sp); H5Dclose(ds);
        return v;
    }
    std::vector<int> read_1d_int(const std::string& path) const {
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        hid_t sp = H5Dget_space(ds);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        std::vector<int> v(dims[0]);
        H5Dread(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
        H5Sclose(sp); H5Dclose(ds);
        return v;
    }
    // 2D double dataset stored ROW-MAJOR by the writer; returned as
    // Eigen column-major matrix for efficient column access (each column
    // is the potential at one radial point).
    Eigen::MatrixXd read_2d_double_as_colmajor(const std::string& path) const {
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        hid_t sp = H5Dget_space(ds);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        const hsize_t rows = dims[0], cols = dims[1];

        // Read into a contiguous row-major buffer, then copy to a column-
        // major Eigen matrix.
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

    // Read only the first `n_rows_wanted` rows of a 2D row-major double
    // dataset. Returns an Eigen column-major matrix (rows = n_rows_wanted,
    // cols = full dataset columns). Critical at large Lmax: the HDF5
    // potential file stores V_H with Nlm_sce = (Lmax_sce+1)^2 rows but the
    // scattering contraction only needs n_exp = (2*l_max_continuum+1)^2
    // of them. Reading the rest wastes memory and I/O.
    Eigen::MatrixXd read_2d_double_rows_subset_as_colmajor(
        const std::string& path, hsize_t n_rows_wanted) const
    {
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        hid_t sp_file = H5Dget_space(ds);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(sp_file, dims, nullptr);
        const hsize_t total_rows = dims[0], cols = dims[1];
        const hsize_t n_rows = std::min(n_rows_wanted, total_rows);

        hsize_t off[2]   = {0, 0};
        hsize_t count[2] = {n_rows, cols};
        H5Sselect_hyperslab(sp_file, H5S_SELECT_SET, off, nullptr, count, nullptr);
        hid_t sp_mem = H5Screate_simple(2, count, nullptr);

        std::vector<double> buf(n_rows * cols);
        H5Dread(ds, H5T_NATIVE_DOUBLE, sp_mem, sp_file, H5P_DEFAULT, buf.data());

        Eigen::MatrixXd M(n_rows, cols);
        for (hsize_t i = 0; i < n_rows; ++i)
            for (hsize_t j = 0; j < cols; ++j)
                M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))
                    = buf[i * cols + j];

        H5Sclose(sp_mem); H5Sclose(sp_file); H5Dclose(ds);
        return M;
    }

    // Targeted partial read of /orbitals/psi_lm.
    //
    // psi_lm on disk is shape (n_orb * Nlm_sce_full, Nr_orb_store), packed
    // ROW-major with row index = orbital_j * Nlm_sce_full + lm.  Scattering
    // only ever uses the first n_lambda_cut <= Nlm_sce_full lambda channels
    // and the first n_transition <= Nr_orb_store radial points.  Reading
    // the full matrix then discarding most of it eats RAM that isn't there
    // (135 GB for C8F8 / Lmax=300 / Nr=3000).
    //
    // This method uses a strided HDF5 hyperslab to read ONLY the rows we
    // need: count=n_orb blocks, each block of n_lambda_cut rows (wide
    // n_transition cols), with stride=Nlm_sce_full.  Result is a
    // (n_orb * n_lambda_cut, n_transition) matrix laid out so that
    // psi_lm(j*n_lambda_cut + lm, ir) gives orbital j's lm-channel at ir.
    //
    // Memory footprint: n_orb * n_lambda_cut * n_transition * 8 bytes for
    // the read buffer, plus the same for the returned Eigen colmajor.
    // Peak 2x that during the row-major -> colmajor copy.
    // Truncated read of /initial_state/psi_lm.  This dataset is a single
    // orbital (no n_orb dimension), shape (Nlm_sce, Nr_orb_store) on disk.
    // We optionally take the first n_lambda_cut <= Nlm_sce rows and the
    // first n_transition <= Nr_orb_store columns.  Useful for anion
    // photodetachment runs where the initial state is the anion SOMO
    // loaded via preprocess_molden --initial-state-molden ...
    // Returns a colmajor MatrixXd of shape (n_lambda_cut, n_transition).
    Eigen::MatrixXd read_initial_state_psi_lm_truncated(
        int Nlm_sce_full,
        int n_lambda_cut,
        int n_transition) const
    {
        if (Nlm_sce_full <= 0 || n_lambda_cut <= 0 || n_transition <= 0)
            throw std::runtime_error(
                "read_initial_state_psi_lm_truncated: bad sizes");
        if (n_lambda_cut > Nlm_sce_full)
            throw std::runtime_error(
                "read_initial_state_psi_lm_truncated: n_lambda_cut > Nlm_sce_full");
        const std::string path = "/initial_state/psi_lm";
        if (!link_exists(path))
            throw std::runtime_error(
                "read_initial_state_psi_lm_truncated: /initial_state/psi_lm "
                "not in HDF5 (rerun preprocess_molden with "
                "--initial-state-molden ...)");
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        hid_t sp_file = H5Dget_space(ds);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(sp_file, dims, nullptr);
        if (dims[0] < hsize_t(n_lambda_cut))
            throw std::runtime_error(
                "read_initial_state_psi_lm_truncated: dataset has fewer "
                "rows than n_lambda_cut");
        if (dims[1] < hsize_t(n_transition))
            throw std::runtime_error(
                "read_initial_state_psi_lm_truncated: dataset has fewer "
                "cols than n_transition");
        const hsize_t off[2]    = {0, 0};
        const hsize_t count[2]  = {hsize_t(n_lambda_cut), hsize_t(n_transition)};
        H5Sselect_hyperslab(sp_file, H5S_SELECT_SET, off, nullptr, count, nullptr);
        hid_t sp_mem = H5Screate_simple(2, count, nullptr);
        std::vector<double> buf(count[0] * count[1]);
        H5Dread(ds, H5T_NATIVE_DOUBLE, sp_mem, sp_file, H5P_DEFAULT, buf.data());
        Eigen::MatrixXd M(count[0], count[1]);
        for (hsize_t i = 0; i < count[0]; ++i)
            for (hsize_t j = 0; j < count[1]; ++j)
                M(Eigen::Index(i), Eigen::Index(j)) = buf[i * count[1] + j];
        H5Sclose(sp_mem); H5Sclose(sp_file); H5Dclose(ds);
        return M;
    }

    Eigen::MatrixXd read_psi_lm_truncated(
        int n_orb,
        int Nlm_sce_full,
        int n_lambda_cut,
        int n_transition) const
    {
        if (n_orb <= 0 || Nlm_sce_full <= 0 ||
            n_lambda_cut <= 0 || n_transition <= 0)
            throw std::runtime_error("read_psi_lm_truncated: bad sizes");
        if (n_lambda_cut > Nlm_sce_full)
            throw std::runtime_error(
                "read_psi_lm_truncated: n_lambda_cut > Nlm_sce_full");

        const std::string path = "/orbitals/psi_lm";
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        hid_t sp_file = H5Dget_space(ds);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(sp_file, dims, nullptr);
        const hsize_t total_rows = dims[0];
        const hsize_t total_cols = dims[1];
        if (total_rows < hsize_t(n_orb) * hsize_t(Nlm_sce_full))
            throw std::runtime_error(
                "read_psi_lm_truncated: dataset has fewer rows than "
                "n_orb * Nlm_sce_full");
        if (total_cols < hsize_t(n_transition))
            throw std::runtime_error(
                "read_psi_lm_truncated: dataset has fewer cols than n_transition");

        // Hyperslab: count = n_orb blocks; block = (n_lambda_cut x n_transition);
        // stride = (Nlm_sce_full, 1); start = (0,0).
        const hsize_t off[2]    = {0, 0};
        const hsize_t stride[2] = {hsize_t(Nlm_sce_full), 1};
        const hsize_t count[2]  = {hsize_t(n_orb), 1};
        const hsize_t block[2]  = {hsize_t(n_lambda_cut),
                                   hsize_t(n_transition)};
        H5Sselect_hyperslab(sp_file, H5S_SELECT_SET, off, stride, count, block);

        // Memory dataspace: contiguous (n_orb * n_lambda_cut, n_transition).
        const hsize_t mem_dims[2] = {hsize_t(n_orb) * hsize_t(n_lambda_cut),
                                     hsize_t(n_transition)};
        hid_t sp_mem = H5Screate_simple(2, mem_dims, nullptr);

        std::vector<double> buf(mem_dims[0] * mem_dims[1]);
        H5Dread(ds, H5T_NATIVE_DOUBLE, sp_mem, sp_file, H5P_DEFAULT, buf.data());

        Eigen::MatrixXd M(mem_dims[0], mem_dims[1]);
        for (hsize_t i = 0; i < mem_dims[0]; ++i)
            for (hsize_t j = 0; j < mem_dims[1]; ++j)
                M(Eigen::Index(i), Eigen::Index(j)) = buf[i * mem_dims[1] + j];

        H5Sclose(sp_mem); H5Sclose(sp_file); H5Dclose(ds);
        return M;
    }

    // Shape of a 2D dataset without loading any data.
    std::pair<hsize_t, hsize_t> dataset_shape_2d(const std::string& path) const {
        hid_t ds = H5Dopen2(file_for(path), path.c_str(), H5P_DEFAULT);
        _check(ds, ("H5Dopen " + path).c_str());
        hid_t sp = H5Dget_space(ds);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        H5Sclose(sp); H5Dclose(ds);
        return {dims[0], dims[1]};
    }
    bool link_exists(const std::string& path) const {
        return H5Lexists(file_for(path), path.c_str(), H5P_DEFAULT) > 0;
    }

    // One-shot: load EVERYTHING except the potentially-huge V_H matrix.
    // V_H must be loaded via load_V_H() with an explicit row budget, so
    // the caller (Potentials) can restrict to just n_exp rows.
    PreprocData load_all_except_V_H() {
        return load_all_impl(/*load_V_H=*/false, /*V_H_max_rows=*/0,
                             /*load_psi_lm=*/true);
    }
    // "Header" load: everything EXCEPT V_H and the (large) /orbitals/psi_lm
    // tensor.  Returns small metadata only (grid, angular, geometry, atom
    // data, orbital metadata, polarizability, alpha_iso) so the main()
    // can stage further loads:
    //   1. load_V_H(...)  -> build Potentials  -> data.V_H.resize(0,0)
    //   2. read_psi_lm_truncated(...) -> assign into data.psi_lm
    //      -> build chi -> data.psi_lm.resize(0,0)
    // Avoids holding the 100-GB-scale full orbital tensor + V_H in RAM
    // simultaneously.
    PreprocData load_header() {
        return load_all_impl(/*load_V_H=*/false, /*V_H_max_rows=*/0,
                             /*load_psi_lm=*/false);
    }

    // One-shot: load everything the scattering stage needs.
    PreprocData load_all() {
        return load_all_impl(/*load_V_H=*/true, /*V_H_max_rows=*/0,
                             /*load_psi_lm=*/true);
    }

    // Load only V_H, with an explicit row budget. Call AFTER
    // load_all_except_V_H(). Pass V_H_max_rows = 0 to load every row.
    Eigen::MatrixXd load_V_H(std::size_t V_H_max_rows = 0) {
        if (V_H_max_rows == 0) {
            return read_2d_double_as_colmajor("/potential/V_H");
        }
        return read_2d_double_rows_subset_as_colmajor("/potential/V_H", V_H_max_rows);
    }

private:
    hid_t file_      = -1;     // legacy alias = file_orb_
    hid_t file_orb_  = -1;     // orbitals + initial_state + shared meta
    hid_t file_pot_  = -1;     // potentials + rho + polarizability + meta
    bool  owned_pot_ = false;  // true iff file_pot_ != file_orb_ and we own it

    PreprocData load_all_impl(bool load_V_H_, std::size_t V_H_max_rows,
                              bool load_psi_lm = true) {
        PreprocData d;
        d.rmin     = read_scalar_double("/grid/rmin");
        d.dr       = read_scalar_double("/grid/dr");
        d.Nr       = static_cast<std::size_t>(read_scalar_int("/grid/N"));
        d.Lmax_sce = static_cast<int>(read_scalar_int("/angular/Lmax"));

        // Atoms.
        auto Z   = read_1d_int("/geometry/Z");
        auto xyz_row = read_2d_double_as_colmajor("/geometry/xyz_bohr");  // (Na, 3)
        const int Na = static_cast<int>(Z.size());
        d.atoms.resize(Na);
        for (int i = 0; i < Na; ++i) {
            d.atoms[i] = PreprocData::Atom{
                Z[i],
                xyz_row(i, 0), xyz_row(i, 1), xyz_row(i, 2)
            };
        }

        // V_ee(r) in SCE -- optionally restricted to the first V_H_max_rows rows.
        if (load_V_H_) {
            d.V_H = (V_H_max_rows == 0)
                    ? read_2d_double_as_colmajor("/potential/V_H")
                    : read_2d_double_rows_subset_as_colmajor("/potential/V_H", V_H_max_rows);
        }

        // Orbitals.
        d.n_alpha     = static_cast<int>(read_scalar_int("/orbitals/n_alpha"));
        d.n_occ_alpha = static_cast<int>(read_scalar_int("/orbitals/n_occ_alpha"));
        d.n_sce       = link_exists("/orbitals/n_sce")
                        ? static_cast<int>(read_scalar_int("/orbitals/n_sce"))
                        : d.n_alpha;
        // Truncated-storage extents.  Default to the full SCE / radial
        // grid when not present (legacy preproc files).
        d.Lmax_orb_store = link_exists("/orbitals/Lmax_orb_store")
                           ? static_cast<int>(read_scalar_int("/orbitals/Lmax_orb_store"))
                           : d.Lmax_sce;
        d.Nlm_orb_store  = link_exists("/orbitals/Nlm_orb_store")
                           ? static_cast<int>(read_scalar_int("/orbitals/Nlm_orb_store"))
                           : (d.Lmax_sce + 1) * (d.Lmax_sce + 1);
        d.Nr_orb_store   = link_exists("/orbitals/Nr_orb_store")
                           ? static_cast<int>(read_scalar_int("/orbitals/Nr_orb_store"))
                           : static_cast<int>(d.Nr);

        // Defensive RAM check before the big read.  On C8F8 at Lmax_sce=300
        // /orbitals/psi_lm is (60 * 90601, 10001) doubles == 435 GB, which
        // OOM-kills the process if the node has less RAM.  Detecting this
        // before the read produces a clear actionable error instead of a
        // bare `Killed` signal from the kernel.
        //
        // We also need ~the same again for the temporary read buffer (we
        // unpack row-major HDF5 into a column-major Eigen matrix), so the
        // factor 2 budget applies.
        if (load_psi_lm) {
            const auto sh = dataset_shape_2d("/orbitals/psi_lm");
            const std::size_t rows0 = static_cast<std::size_t>(sh.first);
            const std::size_t cols0 = static_cast<std::size_t>(sh.second);
            const std::size_t bytes_one_copy = rows0 * cols0 * sizeof(double);
            const std::size_t bytes_needed   = 2 * bytes_one_copy;  // buf + Eigen
            const std::size_t total_ram      = scatt::detect_total_ram_bytes();
            const std::size_t budget         = (total_ram > 0)
                ? static_cast<std::size_t>(total_ram * 0.85)
                : 0;
            const auto fmt_gb = [](std::size_t b) {
                std::ostringstream os;
                os.precision(2); os << std::fixed << double(b) / (1024.0 * 1024.0 * 1024.0);
                return os.str();
            };
            if (total_ram > 0 && bytes_needed > budget) {
                std::ostringstream os;
                os << "HDF5Reader: /orbitals/psi_lm is too large to load.\n"
                   << "  shape         : (" << rows0 << ", " << cols0 << ")\n"
                   << "  one copy      : " << fmt_gb(bytes_one_copy) << " GB\n"
                   << "  read peak     : " << fmt_gb(bytes_needed)   << " GB"
                                              " (row-major buf + Eigen colmajor)\n"
                   << "  system RAM    : " << fmt_gb(total_ram)      << " GB total,"
                                              " ~" << fmt_gb(budget) << " GB usable budget\n"
                   << "Likely fixes:\n"
                   << "  * regenerate preprocessing with smaller --lmax (Lmax_sce):\n"
                   << "    real molecular density has ~no l>50 content; Lmax_sce=80-150\n"
                   << "    is plenty for a Lmax_cont up to ~50.\n"
                   << "  * use a coarser --dr (e.g. 0.02 instead of 0.005)\n"
                   << "  * shrink --rmax if your orbital support is well inside it.";
                throw std::runtime_error(os.str());
            }
            // Even if we fit the budget, make the size visible -- the user
            // will then know which knob to turn down for the next run.
            std::cerr << "[HDF5Reader] /orbitals/psi_lm: shape=(" << rows0
                      << ", " << cols0 << ")  one-copy=" << fmt_gb(bytes_one_copy)
                      << " GB  read-peak=" << fmt_gb(bytes_needed) << " GB"
                      << "  (system RAM=" << fmt_gb(total_ram) << " GB)\n";
            d.psi_lm = read_2d_double_as_colmajor("/orbitals/psi_lm");
        } else {
            // load_header() path: leave d.psi_lm empty.  Caller will do
            // a targeted read via reader.read_psi_lm_truncated(...).
        }
        d.orb_energies     = read_1d_double("/orbitals/energies_hartree");
        d.orb_occupations  = read_1d_double("/orbitals/occupations");
        if (link_exists("/orbitals/molden_index"))
            d.orb_molden_index = read_1d_int("/orbitals/molden_index");

        // Initial state (optional).  Same gating as psi_lm: the SOMO is
        // a (Nlm_sce, Nr_orb_store) tensor that's wasteful to keep in
        // RAM unless the caller actually needs it.
        if (link_exists("/initial_state/psi_lm")) {
            d.has_initial_state   = true;
            if (load_psi_lm) {
                d.init_state_psi_lm = read_2d_double_as_colmajor("/initial_state/psi_lm");
            }
            d.init_state_energy   = read_scalar_double("/initial_state/energy_hartree");
            d.init_state_occ      = read_scalar_double("/initial_state/occupation");
        }

        // Polarizability (optional).
        if (link_exists("/polarizability/alpha_tensor")) {
            d.has_polarizability = true;
            auto M = read_2d_double_as_colmajor("/polarizability/alpha_tensor");
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    d.alpha_tensor(i, j) = M(i, j);
            d.alpha_iso = read_scalar_double("/polarizability/alpha_iso_au");
        }
        return d;
    }
};

}  // namespace scatt::io
