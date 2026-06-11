// HDF5Writer.hpp — thin RAII wrapper around the HDF5 C API for the
// preprocessing output file. Covers exactly what we need:
//
//   - create / open file
//   - create groups (/grid, /angular, /geometry, /potential, /orbitals, /meta)
//   - write scalar and 1D/2D/3D double arrays (with chunking + gzip)
//   - write strings as attributes
//   - write integer arrays
//
// Not a general-purpose HDF5 library; deliberately ~200 LOC so it is easy
// to audit. If we ever need fancier features we can switch to HighFive or
// netCDF-4 without touching the data model.

#pragma once

#include <hdf5.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace preproc::io {

inline void _check(hid_t h, const char* what) {
    if (h < 0) throw std::runtime_error(std::string("HDF5: ") + what + " failed");
}

class H5File {
public:
    H5File() = default;

    // Overwrites any existing file at `path`.
    explicit H5File(const std::string& path) {
        file_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        _check(file_, "H5Fcreate");
    }
    ~H5File() { if (file_ >= 0) H5Fclose(file_); }
    H5File(const H5File&)            = delete;
    H5File& operator=(const H5File&) = delete;
    H5File(H5File&& o) noexcept : file_(o.file_) { o.file_ = -1; }
    H5File& operator=(H5File&& o) noexcept {
        if (this != &o) { if (file_ >= 0) H5Fclose(file_); file_ = o.file_; o.file_ = -1; }
        return *this;
    }

    // Create a group "/a/b/c" creating intermediate groups if needed.
    void make_group(const std::string& path) {
        // Split on '/', create each level if missing.
        std::string cur;
        size_t i = 0;
        while (i < path.size()) {
            if (path[i] == '/') { ++i; continue; }
            size_t j = path.find('/', i);
            std::string seg = path.substr(i, j - i);
            cur += "/" + seg;
            if (!H5Lexists(file_, cur.c_str(), H5P_DEFAULT)) {
                hid_t g = H5Gcreate2(file_, cur.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                _check(g, ("H5Gcreate2 " + cur).c_str());
                H5Gclose(g);
            }
            i = (j == std::string::npos) ? path.size() : j;
        }
    }

    void write_scalar_double(const std::string& path, double v) {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t ds = H5Dcreate2(file_, path.c_str(), H5T_NATIVE_DOUBLE,
                              sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        _check(ds, ("H5Dcreate scalar " + path).c_str());
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
        H5Dclose(ds);  H5Sclose(sp);
    }
    void write_scalar_int(const std::string& path, long long v) {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t ds = H5Dcreate2(file_, path.c_str(), H5T_NATIVE_LLONG,
                              sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        _check(ds, ("H5Dcreate scalar int " + path).c_str());
        H5Dwrite(ds, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
        H5Dclose(ds);  H5Sclose(sp);
    }

    void write_1d_double(const std::string& path, const double* data, hsize_t n) {
        hsize_t dims[1] = {n};
        hid_t sp = H5Screate_simple(1, dims, nullptr);
        hid_t ds = H5Dcreate2(file_, path.c_str(), H5T_NATIVE_DOUBLE,
                              sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        _check(ds, ("H5Dcreate 1d " + path).c_str());
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);  H5Sclose(sp);
    }
    void write_1d_int(const std::string& path, const int* data, hsize_t n) {
        hsize_t dims[1] = {n};
        hid_t sp = H5Screate_simple(1, dims, nullptr);
        hid_t ds = H5Dcreate2(file_, path.c_str(), H5T_NATIVE_INT,
                              sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        _check(ds, ("H5Dcreate int1d " + path).c_str());
        H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);  H5Sclose(sp);
    }

    // Write a 2D C-order (row-major) array. `n_rows` is the outer dim.
    // If gzip_level > 0 we enable chunking + deflate.
    void write_2d_double(const std::string& path,
                         const double* data, hsize_t n_rows, hsize_t n_cols,
                         int gzip_level = 4) {
        hsize_t dims[2] = {n_rows, n_cols};
        hid_t sp  = H5Screate_simple(2, dims, nullptr);
        hid_t dcp = H5Pcreate(H5P_DATASET_CREATE);
        if (gzip_level > 0 && n_rows > 0 && n_cols > 0) {
            // Chunk per row (good for column-by-column or row-by-row access).
            hsize_t chunk[2] = {1, n_cols};
            H5Pset_chunk(dcp, 2, chunk);
            H5Pset_deflate(dcp, static_cast<unsigned>(gzip_level));
        }
        hid_t ds = H5Dcreate2(file_, path.c_str(), H5T_NATIVE_DOUBLE,
                              sp, H5P_DEFAULT, dcp, H5P_DEFAULT);
        _check(ds, ("H5Dcreate 2d " + path).c_str());
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);  H5Pclose(dcp);  H5Sclose(sp);
    }
    void write_2d_int(const std::string& path,
                      const int* data, hsize_t n_rows, hsize_t n_cols) {
        hsize_t dims[2] = {n_rows, n_cols};
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(file_, path.c_str(), H5T_NATIVE_INT,
                              sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        _check(ds, ("H5Dcreate int2d " + path).c_str());
        H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);  H5Sclose(sp);
    }

    // Write a string attribute attached to the root group "/".
    void write_string_attr(const std::string& name, const std::string& value) {
        hid_t root = H5Gopen2(file_, "/", H5P_DEFAULT);
        hid_t atype = H5Tcopy(H5T_C_S1);
        H5Tset_size(atype, value.size() + 1);
        H5Tset_strpad(atype, H5T_STR_NULLTERM);
        hid_t aspace = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(root, name.c_str(), atype, aspace,
                                H5P_DEFAULT, H5P_DEFAULT);
        _check(attr, ("H5Acreate " + name).c_str());
        H5Awrite(attr, atype, value.c_str());
        H5Aclose(attr);  H5Sclose(aspace);  H5Tclose(atype);  H5Gclose(root);
    }

private:
    hid_t file_ = -1;
};

}  // namespace preproc::io
