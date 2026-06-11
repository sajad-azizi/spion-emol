#include "DipoleIO.hpp"

#include <hdf5.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace sph3d {

namespace {

void check_(hid_t id, const char* what) {
    if (id < 0) throw std::runtime_error(std::string("HDF5: ") + what);
}

// --- attribute writers ----------------------------------------------
void wattr_d(hid_t loc, const char* name, double v) {
    hid_t s = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a); H5Sclose(s);
}
void wattr_i(hid_t loc, const char* name, int v) {
    hid_t s = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_INT, s,
                         H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT, &v);
    H5Aclose(a); H5Sclose(s);
}
void wattr_s(hid_t loc, const char* name, const std::string& v) {
    hid_t st = H5Tcopy(H5T_C_S1);
    H5Tset_size(st, v.size() + 1);
    H5Tset_strpad(st, H5T_STR_NULLTERM);
    hid_t s = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, st, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, st, v.c_str());
    H5Aclose(a); H5Sclose(s); H5Tclose(st);
}

// --- 1-D dataset writers --------------------------------------------
void wd1d(hid_t loc, const char* name, const std::vector<double>& v) {
    const hsize_t dims[1] = { v.size() };
    hid_t s = H5Screate_simple(1, dims, nullptr);
    hid_t d = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
             H5P_DEFAULT, v.data());
    H5Dclose(d); H5Sclose(s);
}

void atomic_finalize(const std::string& tmp_path,
                     const std::string& final_path)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::rename(tmp_path, final_path, ec);
    if (ec) {
        fs::copy_file(tmp_path, final_path,
                      fs::copy_options::overwrite_existing, ec);
        fs::remove(tmp_path);
        if (ec) {
            throw std::runtime_error("DipoleIO: rename "
                + tmp_path + " -> " + final_path + ": " + ec.message());
        }
    }
}

}  // namespace

std::string ik_tag(int ik) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "ik%04d", ik);
    return buf;
}

void write_manifest(const std::string& scan_dir, const ScanMeta& m) {
    namespace fs = std::filesystem;
    fs::create_directories(scan_dir);

    const std::string final_path = scan_dir + "/manifest.h5";
    const std::string tmp_path   = final_path + ".tmp";

    hid_t f = H5Fcreate(tmp_path.c_str(), H5F_ACC_TRUNC,
                        H5P_DEFAULT, H5P_DEFAULT);
    check_(f, "H5Fcreate manifest");

    // /grid
    {
        hid_t g = H5Gcreate2(f, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check_(g, "H5Gcreate /grid");
        wattr_d(g, "r_min",           m.r_min);
        wattr_d(g, "dr",              m.dr);
        wattr_i(g, "N_grid",          m.N_grid);
        wattr_i(g, "l_max_continuum", m.l_max_continuum);
        wattr_d(g, "E_HOMO",          m.E_HOMO);
        H5Gclose(g);
    }
    // /kgrid
    {
        hid_t g = H5Gcreate2(f, "/kgrid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check_(g, "H5Gcreate /kgrid");
        wattr_d(g, "dk",     m.dk);
        wattr_i(g, "ik_min", m.ik_min);
        wattr_i(g, "ik_max", m.ik_max);
        H5Gclose(g);
    }
    // /occ
    {
        hid_t g = H5Gcreate2(f, "/occ", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check_(g, "H5Gcreate /occ");
        wd1d(g, "energies",     m.occ_energies);
        wd1d(g, "spin_factors", m.occ_spin_factors);
        H5Gclose(g);
    }
    // /run
    {
        hid_t g = H5Gcreate2(f, "/run", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check_(g, "H5Gcreate /run");
        wattr_s(g, "molecule_name", m.molecule_name);
        wattr_s(g, "molhash",       m.molhash);
        wattr_s(g, "iso_date_utc",  m.iso_date_utc);
        wattr_s(g, "real_Ylm_q_map", "x=+1, y=-1, z=0");
        wattr_s(g, "psi_norm",       "incoming-wave Psi- = (A - iB)^(-T*)");
        wattr_s(g, "u_convention",   "chi = r * F_lm(r)");
        H5Gclose(g);
    }

    H5Fclose(f);
    atomic_finalize(tmp_path, final_path);
}

void write_ik(const std::string& scan_dir, const DipolePerIK& p) {
    namespace fs = std::filesystem;
    fs::create_directories(scan_dir);
    const std::string final_path = scan_dir + "/" + ik_tag(p.ik) + ".h5";
    const std::string tmp_path   = final_path + ".tmp";

    hid_t f = H5Fcreate(tmp_path.c_str(), H5F_ACC_TRUNC,
                        H5P_DEFAULT, H5P_DEFAULT);
    check_(f, "H5Fcreate per-ik");

    {
        hid_t g = H5Gcreate2(f, "/meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        check_(g, "H5Gcreate /meta");
        wattr_i(g, "ik",    p.ik);
        wattr_d(g, "k",     p.k);
        wattr_d(g, "E",     p.E);
        wattr_d(g, "omega", p.omega);
        H5Gclose(g);
    }

    static const char* gauge_names[2] = { "length", "velocity" };
    static const char* pol_names[3]   = { "x", "y", "z" };
    for (int gi = 0; gi < 2; ++gi) {
        for (int pi = 0; pi < 3; ++pi) {
            const int slot = gi * 3 + pi;
            const auto& Dr = p.D_raw[slot];
            const auto& Do = p.D_ortho[slot];
            const std::string path =
                std::string("/dipole/") + gauge_names[gi] + "/" + pol_names[pi];
            // Group hierarchy: /dipole, /dipole/length, /dipole/length/x
            // Use H5Gcreate with intermediate-create LCPL flag.
            hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
            H5Pset_create_intermediate_group(lcpl, 1);
            hid_t g = H5Gcreate2(f, path.c_str(), lcpl, H5P_DEFAULT, H5P_DEFAULT);
            H5Pclose(lcpl);
            check_(g, ("H5Gcreate " + path).c_str());

            std::vector<double> re_buf(Dr.size()), im_buf(Dr.size());
            for (std::size_t b = 0; b < Dr.size(); ++b) {
                re_buf[b] = Dr[b].real();
                im_buf[b] = Dr[b].imag();
            }
            wd1d(g, "D_raw_re", re_buf);
            wd1d(g, "D_raw_im", im_buf);
            for (std::size_t b = 0; b < Do.size(); ++b) {
                re_buf[b] = Do[b].real();
                im_buf[b] = Do[b].imag();
            }
            wd1d(g, "D_ortho_re", re_buf);
            wd1d(g, "D_ortho_im", im_buf);
            H5Gclose(g);
        }
    }

    H5Fclose(f);
    atomic_finalize(tmp_path, final_path);
}

bool ik_exists(const std::string& scan_dir, int ik) {
    namespace fs = std::filesystem;
    return fs::exists(scan_dir + "/" + ik_tag(ik) + ".h5");
}

}  // namespace sph3d
