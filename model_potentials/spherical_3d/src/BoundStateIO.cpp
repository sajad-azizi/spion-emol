#include "BoundStateIO.hpp"

#include <hdf5.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace sph3d {

namespace {

void check_(hid_t id, const char* what) {
    if (id < 0) throw std::runtime_error(std::string("HDF5: ") + what);
}

void write_attr_double(hid_t loc, const char* name, double v) {
    hid_t s = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a); H5Sclose(s);
}
void write_attr_int(hid_t loc, const char* name, int v) {
    hid_t s = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_INT, s,
                         H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT, &v);
    H5Aclose(a); H5Sclose(s);
}
void write_attr_str(hid_t loc, const char* name, const std::string& v) {
    hid_t st = H5Tcopy(H5T_C_S1);
    H5Tset_size(st, v.size() + 1);
    H5Tset_strpad(st, H5T_STR_NULLTERM);
    hid_t s = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, st, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, st, v.c_str());
    H5Aclose(a); H5Sclose(s); H5Tclose(st);
}

double read_attr_double(hid_t loc, const char* name) {
    hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
    double v = 0.0;
    H5Aread(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a);
    return v;
}
int read_attr_int(hid_t loc, const char* name) {
    hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
    int v = 0;
    H5Aread(a, H5T_NATIVE_INT, &v);
    H5Aclose(a);
    return v;
}
std::string read_attr_str(hid_t loc, const char* name) {
    hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
    hid_t t = H5Aget_type(a);
    const std::size_t n = H5Tget_size(t);
    std::string buf(n, '\0');
    H5Aread(a, t, buf.data());
    if (!buf.empty() && buf.back() == '\0') buf.pop_back();
    H5Tclose(t); H5Aclose(a);
    return buf;
}

void write_dataset_2d(hid_t loc, const char* name,
                      const std::vector<double>& flat,
                      hsize_t rows, hsize_t cols)
{
    hsize_t dims[2] = { rows, cols };
    hid_t s = H5Screate_simple(2, dims, nullptr);
    hid_t d = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
             H5P_DEFAULT, flat.data());
    H5Dclose(d); H5Sclose(s);
}

void read_dataset_2d(hid_t loc, const char* name,
                     std::vector<double>& flat,
                     hsize_t rows, hsize_t cols)
{
    hid_t d = H5Dopen2(loc, name, H5P_DEFAULT);
    flat.resize(rows * cols);
    H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, flat.data());
    H5Dclose(d);
}

}  // namespace

void save_bound_state(const std::string& path, const BoundState& bs) {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(path).parent_path());
    const std::string tmp = path + ".tmp";

    hid_t f = H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    check_(f, "H5Fcreate bound");
    hid_t g = H5Gcreate2(f, "/bound", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_(g, "H5Gcreate /bound");

    write_attr_double(g, "energy",     bs.gsEnergy);
    write_attr_int   (g, "i_match",    bs.i_match);
    write_attr_int   (g, "N_grid",     bs.N_grid);
    write_attr_int   (g, "n_channels", bs.n_channels);
    write_attr_double(g, "dr",         bs.dr);
    write_attr_str   (g, "molhash",    bs.molhash);

    const hsize_t Nr = bs.N_grid;
    const hsize_t Nc = bs.n_channels;
    std::vector<double> chi_re(Nr * Nc, 0.0);
    std::vector<double> chi_im(Nr * Nc, 0.0);
    for (hsize_t k = 0; k < Nr; ++k) {
        for (hsize_t c = 0; c < Nc; ++c) {
            chi_re[k * Nc + c] = bs.eigfunc[k](c).real();
            chi_im[k * Nc + c] = bs.eigfunc[k](c).imag();
        }
    }
    write_dataset_2d(g, "chi_re", chi_re, Nr, Nc);
    write_dataset_2d(g, "chi_im", chi_im, Nr, Nc);

    H5Gclose(g);
    H5Fclose(f);

    // Atomic rename
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        // Fallback: copy-and-remove on cross-device or other ec
        fs::copy_file(tmp, path, fs::copy_options::overwrite_existing, ec);
        fs::remove(tmp);
        if (ec) throw std::runtime_error("BoundStateIO: rename failed: " + ec.message());
    }
}

bool load_bound_state(const std::string& path,
                      const std::string& expected_molhash,
                      int expected_N_grid,
                      int expected_n_channels,
                      BoundState& out)
{
    namespace fs = std::filesystem;
    if (!fs::exists(path)) return false;

    // Suppress GSL/HDF5 default error printing for the open attempt;
    // we treat any failure as cache-miss and recompute.
    H5E_auto2_t old_func; void* old_data;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_data);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    H5Eset_auto2(H5E_DEFAULT, old_func, old_data);
    if (f < 0) return false;

    hid_t g = H5Gopen2(f, "/bound", H5P_DEFAULT);
    if (g < 0) { H5Fclose(f); return false; }

    BoundState bs;
    bs.molhash    = read_attr_str   (g, "molhash");
    bs.N_grid     = read_attr_int   (g, "N_grid");
    bs.n_channels = read_attr_int   (g, "n_channels");
    bs.dr         = read_attr_double(g, "dr");
    bs.gsEnergy   = read_attr_double(g, "energy");
    bs.i_match    = read_attr_int   (g, "i_match");

    if (bs.molhash    != expected_molhash    ||
        bs.N_grid     != expected_N_grid     ||
        bs.n_channels != expected_n_channels)
    {
        H5Gclose(g); H5Fclose(f);
        return false;   // cache miss (but file exists)
    }

    const hsize_t Nr = bs.N_grid;
    const hsize_t Nc = bs.n_channels;
    std::vector<double> chi_re, chi_im;
    read_dataset_2d(g, "chi_re", chi_re, Nr, Nc);
    read_dataset_2d(g, "chi_im", chi_im, Nr, Nc);
    bs.eigfunc.assign(Nr, Eigen::VectorXcd::Zero(Nc));
    for (hsize_t k = 0; k < Nr; ++k) {
        for (hsize_t c = 0; c < Nc; ++c) {
            bs.eigfunc[k](c) = dcompx(chi_re[k * Nc + c],
                                       chi_im[k * Nc + c]);
        }
    }
    H5Gclose(g);
    H5Fclose(f);
    out = std::move(bs);
    return true;
}

}  // namespace sph3d
