// DipoleIO.cpp -- see DipoleIO.hpp for on-disk layout and design rationale.

#include "scatt/DipoleIO.hpp"

#include "io/HDF5Reader.hpp"   // for scatt::io::_check

#include <hdf5.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>            // getpid -- per-process tmp filename
#include <vector>
#include <string>

namespace scatt {

namespace fs = std::filesystem;
using scatt::io::_check;

// ---------------------------------------------------------------------------
// Small HDF5 write helpers. Kept inline here; if we grow a second writer we
// can promote them to a shared header.
// ---------------------------------------------------------------------------
namespace {

void write_attr_double(hid_t loc, const char* name, double v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t at = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT);
    _check(at, name);
    H5Awrite(at, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(at); H5Sclose(sp);
}
void write_attr_int(hid_t loc, const char* name, long long v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t at = H5Acreate2(loc, name, H5T_NATIVE_LLONG, sp, H5P_DEFAULT, H5P_DEFAULT);
    _check(at, name);
    H5Awrite(at, H5T_NATIVE_LLONG, &v);
    H5Aclose(at); H5Sclose(sp);
}
void write_attr_string(hid_t loc, const char* name, const std::string& s) {
    hid_t tp = H5Tcopy(H5T_C_S1);
    H5Tset_size(tp, s.size() == 0 ? 1 : s.size());
    H5Tset_strpad(tp, H5T_STR_NULLPAD);
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t at = H5Acreate2(loc, name, tp, sp, H5P_DEFAULT, H5P_DEFAULT);
    _check(at, name);
    H5Awrite(at, tp, s.c_str());
    H5Aclose(at); H5Sclose(sp); H5Tclose(tp);
}

void write_1d_double(hid_t loc, const char* path, const double* data, hsize_t n) {
    hsize_t dims[1] = {n};
    hid_t sp = H5Screate_simple(1, dims, nullptr);
    hid_t ds = H5Dcreate2(loc, path, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    _check(ds, path);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds); H5Sclose(sp);
}
void write_1d_int(hid_t loc, const char* path, const int* data, hsize_t n) {
    hsize_t dims[1] = {n};
    hid_t sp = H5Screate_simple(1, dims, nullptr);
    hid_t ds = H5Dcreate2(loc, path, H5T_NATIVE_INT, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    _check(ds, path);
    H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds); H5Sclose(sp);
}

// Write an Eigen real matrix as (rows, cols) row-major so h5py sees it with
// the natural orientation. Eigen default storage is column-major so we copy.
void write_2d_double(hid_t loc, const char* path, const Eigen::MatrixXd& M) {
    const hsize_t r = static_cast<hsize_t>(M.rows()), c = static_cast<hsize_t>(M.cols());
    std::vector<double> buf(r * c);
    for (hsize_t i = 0; i < r; ++i)
        for (hsize_t j = 0; j < c; ++j)
            buf[i * c + j] = M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    hsize_t dims[2] = {r, c};
    hid_t sp = H5Screate_simple(2, dims, nullptr);
    hid_t ds = H5Dcreate2(loc, path, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    _check(ds, path);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds); H5Sclose(sp);
}

// Split a complex vector into _re / _im datasets under `group`.
void write_complex_vec(hid_t group, const char* base_re, const char* base_im,
                       const Eigen::VectorXcd& v) {
    const hsize_t n = static_cast<hsize_t>(v.size());
    std::vector<double> re(n), im(n);
    for (hsize_t i = 0; i < n; ++i) { re[i] = v[i].real(); im[i] = v[i].imag(); }
    write_1d_double(group, base_re, re.data(), n);
    write_1d_double(group, base_im, im.data(), n);
}

const char* gauge_name(DipoleGauge g) {
    return (g == DipoleGauge::Length) ? "length" : "velocity";
}

}  // namespace

// ---------------------------------------------------------------------------
// DipoleWriter
// ---------------------------------------------------------------------------
DipoleWriter::DipoleWriter(const std::string& scan_dir, const DipoleScanMeta& meta)
    : scan_dir_(scan_dir), meta_(meta)
{
    meta_.kgrid.validate();
    ensure_dir_();

    const std::string mpath = scan_dir_ + "/manifest.h5";
    // Re-write manifest only if absent. If present we trust the existing one
    // (resuming a scan) -- mismatched meta would be a programmer error and
    // we don't silently overwrite.
    if (!fs::exists(mpath)) {
        write_manifest_();
    }
}

void DipoleWriter::ensure_dir_() const {
    std::error_code ec;
    fs::create_directories(scan_dir_, ec);
    if (ec) throw std::runtime_error("DipoleWriter: cannot create " + scan_dir_
                                     + ": " + ec.message());
}

std::string DipoleWriter::path_for_(int ik) const {
    return scan_dir_ + "/" + meta_.kgrid.tag(ik) + ".h5";
}

bool DipoleWriter::has_energy(int ik) const {
    return fs::exists(path_for_(ik));
}

void DipoleWriter::write_manifest_() const {
    const std::string mpath = scan_dir_ + "/manifest.h5";
    // Per-PROCESS tmp filename.  When multiple MPI ranks of run_mpi each
    // launch their own scattering binary against the SAME scan_dir (the
    // typical job.mpi.slurm pattern with ik partitioned across ranks),
    // they all hit this code path nearly simultaneously.  Using a fixed
    // "manifest.h5.tmp" caused two failure modes:
    //   * "H5Fcreate manifest failed" -- two ranks racing on
    //     H5Fcreate(tmp, TRUNC) where one had it open exclusively.
    //   * "cannot rename: No such file or directory" -- rank A renamed
    //     tmp->mpath, then rank B's rename of the (now-missing) tmp
    //     failed.
    // With a PID-tagged tmp each rank has its own file; renames are
    // independent, and the "last writer wins" semantics on mpath is
    // safe because every rank writes the same deterministic content
    // for a given scan_dir.
    const std::string tmp = mpath + ".tmp.pid"
                          + std::to_string(static_cast<long long>(::getpid()));

    hid_t f = H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    _check(f, "H5Fcreate manifest");

    // Grid.
    hid_t gg = H5Gcreate2(f, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_attr_double(gg, "r_min",            meta_.r_min);
    write_attr_double(gg, "dr",               meta_.dr);
    write_attr_int   (gg, "N_grid",           static_cast<long long>(meta_.N_grid));
    write_attr_int   (gg, "l_max_continuum",  meta_.l_max_continuum);
    write_attr_double(gg, "E_HOMO",           meta_.E_HOMO);
    H5Gclose(gg);

    // k-grid.
    hid_t gk = H5Gcreate2(f, "/kgrid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_attr_double(gk, "dk",      meta_.kgrid.dk);
    write_attr_int   (gk, "ik_min",  meta_.kgrid.ik_min);
    write_attr_int   (gk, "ik_max",  meta_.kgrid.ik_max);
    H5Gclose(gk);

    // Occupied orbitals.
    if (!meta_.occ_energies.empty()) {
        hid_t go = H5Gcreate2(f, "/occ", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        write_1d_double(go, "energies",     meta_.occ_energies.data(),
                        meta_.occ_energies.size());
        write_1d_double(go, "spin_factors", meta_.occ_spin_factors.data(),
                        meta_.occ_spin_factors.size());
        H5Gclose(go);
    }

    // Atoms.
    if (!meta_.atoms.empty()) {
        hid_t ga = H5Gcreate2(f, "/atoms", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        const hsize_t Na = meta_.atoms.size();
        std::vector<int>    Z(Na);
        std::vector<double> xyz(Na * 3);
        for (hsize_t i = 0; i < Na; ++i) {
            Z[i] = meta_.atoms[i].Z;
            xyz[3*i + 0] = meta_.atoms[i].x;
            xyz[3*i + 1] = meta_.atoms[i].y;
            xyz[3*i + 2] = meta_.atoms[i].z;
        }
        write_1d_int(ga, "Z", Z.data(), Na);
        hsize_t dims[2] = {Na, 3};
        hid_t sp = H5Screate_simple(2, dims, nullptr);
        hid_t ds = H5Dcreate2(ga, "xyz_bohr", H5T_NATIVE_DOUBLE, sp,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, xyz.data());
        H5Dclose(ds); H5Sclose(sp);
        H5Gclose(ga);
    }

    // Conventions (free-text).
    hid_t gc = H5Gcreate2(f, "/conventions", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_attr_string(gc, "real_Ylm_q_map", "x=+1, y=-1, z=0");
    write_attr_string(gc, "psi_norm",       "incoming-wave Psi- = (A - iB)^(-dagger)");
    write_attr_string(gc, "u_convention",   "chi_lm(r) = r * F_lm(r)");
    write_attr_string(gc, "gauge_cross_section",
                      "sigma^L = (4 pi^2 / c) omega Sum_mu |D^L|^2;  "
                      "sigma^V = (4 pi^2 / c) (1/omega) Sum_mu |D^V|^2");
    H5Gclose(gc);

    // Run info.
    hid_t gr = H5Gcreate2(f, "/run", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_attr_string(gr, "molecule_name",  meta_.molecule_name);
    write_attr_string(gr, "git_hash",       meta_.git_hash);
    write_attr_string(gr, "iso_date_utc",   meta_.iso_date_utc);
    write_attr_string(gr, "psi_dir_prefix", meta_.psi_dir_prefix);
    H5Gclose(gr);

    H5Fclose(f);
    // Idempotent rename.  POSIX rename(tmp, mpath) is atomic per-FS, so
    // mpath ends up as either the previous content or the new content.
    // If two ranks rename in rapid succession, both succeed; "last
    // writer wins".  If our tmp is missing (e.g. somebody nuked it),
    // accept that as long as mpath already exists.
    std::error_code ec;
    fs::rename(tmp, mpath, ec);
    if (ec && !fs::exists(mpath)) {
        // Best-effort cleanup of our orphan tmp; ignore failure.
        std::error_code rm_ec;
        fs::remove(tmp, rm_ec);
        throw std::runtime_error(
            "DipoleWriter: cannot finalise manifest -- "
            "rename " + tmp + " -> " + mpath + " failed (" + ec.message() +
            ") and " + mpath + " was not written by anyone else.");
    }
    if (ec) {
        // Rename failed but mpath exists -- another rank already finished
        // first.  Clean up our orphan tmp; success.
        std::error_code rm_ec;
        fs::remove(tmp, rm_ec);
    }
}

void DipoleWriter::write_energy(const DipoleEnergyPayload& p) {
    const int n_ch  = static_cast<int>(p.A.rows());
    if (p.A.cols() != n_ch) throw std::runtime_error("DipoleWriter: A not square");
    if (p.B.rows() != n_ch || p.B.cols() != n_ch)
        throw std::runtime_error("DipoleWriter: B shape mismatch");
    if (p.b_overlap.rows() != n_ch)
        throw std::runtime_error("DipoleWriter: b_overlap row count != n_ch");

    for (const auto& s : p.slices) {
        if (s.D_reduced.size()     != n_ch ||
            s.D_reduced_raw.size() != n_ch ||
            s.d_raw.size()         != n_ch)
            throw std::runtime_error("DipoleWriter: per-slice size mismatch");
    }

    const std::string fpath = path_for_(p.ik);
    // PID-tagged tmp.  In a typical MPI scan each rank writes its OWN
    // ik (disjoint), so a single-rank race here is rare.  But two
    // ranks accidentally assigned overlapping ik (or a retry of a
    // killed run) used to collide on the bare "<fpath>.tmp" filename.
    const std::string tmp = fpath + ".tmp.pid"
                          + std::to_string(static_cast<long long>(::getpid()));

    hid_t f = H5Fcreate(tmp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    _check(f, "H5Fcreate per-ik");

    // /meta
    hid_t gm = H5Gcreate2(f, "/meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_attr_int   (gm, "ik",                 p.ik);
    write_attr_double(gm, "k",                  meta_.kgrid.k(p.ik));
    write_attr_double(gm, "E",                  meta_.kgrid.E(p.ik));
    write_attr_double(gm, "omega",              meta_.kgrid.E(p.ik) - meta_.E_HOMO);
    write_attr_double(gm, "fit_residual_rel",   p.fit_residual_rel);
    write_attr_double(gm, "K_symmetry_err",     p.K_symmetry_err);
    H5Gclose(gm);

    // Amplitudes and overlap (once per energy).
    write_2d_double(f, "/A", p.A);
    write_2d_double(f, "/B", p.B);
    write_2d_double(f, "/b_overlap", p.b_overlap);

    // /dipole/{length,velocity}/{x,y,z}/{D_ortho_re, D_ortho_im,
    //                                    D_raw_re,   D_raw_im,
    //                                    d_raw, d_correction}
    hid_t gd = H5Gcreate2(f, "/dipole", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(gd);
    for (const auto& s : p.slices) {
        const std::string gpath = std::string("/dipole/") + gauge_name(s.gauge);
        if (H5Lexists(f, gpath.c_str(), H5P_DEFAULT) <= 0) {
            hid_t gg = H5Gcreate2(f, gpath.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(gg);
        }
        const std::string ppath = gpath + "/" + name_of(s.pol);
        hid_t gp = H5Gcreate2(f, ppath.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        write_complex_vec(gp, "D_ortho_re", "D_ortho_im", s.D_reduced);
        write_complex_vec(gp, "D_raw_re",   "D_raw_im",   s.D_reduced_raw);
        write_1d_double  (gp, "d_raw",       s.d_raw.data(),       s.d_raw.size());
        write_1d_double  (gp, "d_correction",
                              s.d_correction.data(), s.d_correction.size());
        write_attr_double(gp, "partial_sigma", s.partial_sigma);
        H5Gclose(gp);
    }

    H5Fclose(f);
    // Idempotent rename: tolerate "another rank already wrote it" but
    // surface real failures.  See write_manifest_ for the rationale.
    std::error_code ec;
    fs::rename(tmp, fpath, ec);
    if (ec && !fs::exists(fpath)) {
        std::error_code rm_ec;
        fs::remove(tmp, rm_ec);
        throw std::runtime_error(
            "DipoleWriter: cannot finalise per-ik file -- "
            "rename " + tmp + " -> " + fpath + " failed (" + ec.message() +
            ") and " + fpath + " was not written by anyone else.");
    }
    if (ec) {
        std::error_code rm_ec;
        fs::remove(tmp, rm_ec);
    }
}

void DipoleWriter::finalize() {
    std::ofstream(scan_dir_ + "/__SUCCESS__") << "";
}

// ---------------------------------------------------------------------------
// DipoleReader
// ---------------------------------------------------------------------------
namespace {

double read_attr_double(hid_t loc, const char* name) {
    hid_t at = H5Aopen(loc, name, H5P_DEFAULT);
    _check(at, name);
    double v;
    H5Aread(at, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(at);
    return v;
}
long long read_attr_int(hid_t loc, const char* name) {
    hid_t at = H5Aopen(loc, name, H5P_DEFAULT);
    _check(at, name);
    long long v;
    H5Aread(at, H5T_NATIVE_LLONG, &v);
    H5Aclose(at);
    return v;
}
std::string read_attr_string(hid_t loc, const char* name) {
    hid_t at = H5Aopen(loc, name, H5P_DEFAULT);
    _check(at, name);
    hid_t tp = H5Aget_type(at);
    size_t sz = H5Tget_size(tp);
    std::string s(sz, '\0');
    H5Aread(at, tp, s.data());
    // Strip any trailing \0 padding.
    while (!s.empty() && s.back() == '\0') s.pop_back();
    H5Tclose(tp); H5Aclose(at);
    return s;
}

std::vector<double> read_1d_double(hid_t loc, const char* path) {
    hid_t ds = H5Dopen2(loc, path, H5P_DEFAULT);
    _check(ds, path);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    std::vector<double> v(dims[0]);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(sp); H5Dclose(ds);
    return v;
}
std::vector<int> read_1d_int(hid_t loc, const char* path) {
    hid_t ds = H5Dopen2(loc, path, H5P_DEFAULT);
    _check(ds, path);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    std::vector<int> v(dims[0]);
    H5Dread(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(sp); H5Dclose(ds);
    return v;
}
Eigen::MatrixXd read_2d_double(hid_t loc, const char* path) {
    hid_t ds = H5Dopen2(loc, path, H5P_DEFAULT);
    _check(ds, path);
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    std::vector<double> buf(dims[0] * dims[1]);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    Eigen::MatrixXd M(dims[0], dims[1]);
    for (hsize_t i = 0; i < dims[0]; ++i)
        for (hsize_t j = 0; j < dims[1]; ++j)
            M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))
                = buf[i * dims[1] + j];
    H5Sclose(sp); H5Dclose(ds);
    return M;
}

Eigen::VectorXcd read_complex_vec(hid_t loc, const std::string& base_re,
                                  const std::string& base_im) {
    auto re = read_1d_double(loc, base_re.c_str());
    auto im = read_1d_double(loc, base_im.c_str());
    if (re.size() != im.size())
        throw std::runtime_error("DipoleReader: complex re/im size mismatch");
    Eigen::VectorXcd v(re.size());
    for (size_t i = 0; i < re.size(); ++i) v[i] = {re[i], im[i]};
    return v;
}

}  // namespace

DipoleReader::DipoleReader(const std::string& scan_dir) : scan_dir_(scan_dir) {
    read_manifest_();
}

void DipoleReader::read_manifest_() {
    const std::string mpath = scan_dir_ + "/manifest.h5";
    if (!fs::exists(mpath))
        throw std::runtime_error("DipoleReader: missing " + mpath);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t f = H5Fopen(mpath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    _check(f, "H5Fopen manifest");

    hid_t gg = H5Gopen2(f, "/grid", H5P_DEFAULT);
    meta_.r_min           = read_attr_double(gg, "r_min");
    meta_.dr              = read_attr_double(gg, "dr");
    meta_.N_grid          = static_cast<std::size_t>(read_attr_int(gg, "N_grid"));
    meta_.l_max_continuum = static_cast<int>(read_attr_int(gg, "l_max_continuum"));
    meta_.E_HOMO          = read_attr_double(gg, "E_HOMO");
    H5Gclose(gg);

    hid_t gk = H5Gopen2(f, "/kgrid", H5P_DEFAULT);
    meta_.kgrid.dk     = read_attr_double(gk, "dk");
    meta_.kgrid.ik_min = static_cast<int>(read_attr_int(gk, "ik_min"));
    meta_.kgrid.ik_max = static_cast<int>(read_attr_int(gk, "ik_max"));
    H5Gclose(gk);

    if (H5Lexists(f, "/occ", H5P_DEFAULT) > 0) {
        meta_.occ_energies     = read_1d_double(f, "/occ/energies");
        meta_.occ_spin_factors = read_1d_double(f, "/occ/spin_factors");
    }
    if (H5Lexists(f, "/atoms", H5P_DEFAULT) > 0) {
        auto Z   = read_1d_int   (f, "/atoms/Z");
        auto xyz = read_2d_double(f, "/atoms/xyz_bohr");
        meta_.atoms.resize(Z.size());
        for (size_t i = 0; i < Z.size(); ++i) {
            meta_.atoms[i] = { Z[i], xyz(i,0), xyz(i,1), xyz(i,2) };
        }
    }
    if (H5Lexists(f, "/run", H5P_DEFAULT) > 0) {
        hid_t gr = H5Gopen2(f, "/run", H5P_DEFAULT);
        meta_.molecule_name  = read_attr_string(gr, "molecule_name");
        meta_.git_hash       = read_attr_string(gr, "git_hash");
        meta_.iso_date_utc   = read_attr_string(gr, "iso_date_utc");
        meta_.psi_dir_prefix = read_attr_string(gr, "psi_dir_prefix");
        H5Gclose(gr);
    }
    H5Fclose(f);
}

std::vector<int> DipoleReader::available_ik() const {
    std::vector<int> out;
    for (const auto& e : fs::directory_iterator(scan_dir_)) {
        const std::string n = e.path().filename().string();
        if (n.rfind("ik", 0) == 0 &&
            n.size() > 5 && n.substr(n.size()-3) == ".h5")
        {
            try { out.push_back(std::stoi(n.substr(2, n.size()-5))); }
            catch (...) { /* not a valid ik file */ }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

DipoleEnergyReadback DipoleReader::read_energy(int ik) const {
    const std::string fpath = scan_dir_ + "/" + meta_.kgrid.tag(ik) + ".h5";
    if (!fs::exists(fpath))
        throw std::runtime_error("DipoleReader: missing " + fpath);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t f = H5Fopen(fpath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    _check(f, "H5Fopen per-ik");

    DipoleEnergyReadback r;
    r.ik = ik;

    hid_t gm = H5Gopen2(f, "/meta", H5P_DEFAULT);
    r.fit_residual_rel = read_attr_double(gm, "fit_residual_rel");
    r.K_symmetry_err   = read_attr_double(gm, "K_symmetry_err");
    r.k                = read_attr_double(gm, "k");
    r.E                = read_attr_double(gm, "E");
    H5Gclose(gm);

    r.A         = read_2d_double(f, "/A");
    r.B         = read_2d_double(f, "/B");
    r.b_overlap = read_2d_double(f, "/b_overlap");

    const DipoleGauge  gauges[2] = { DipoleGauge::Length, DipoleGauge::Velocity };
    const Polarization pols[3]   = { Polarization::X, Polarization::Y, Polarization::Z };
    for (auto g : gauges) {
        for (auto p : pols) {
            const int i = DipoleWriter::slice_index(g, p);
            const std::string ppath = std::string("/dipole/") + gauge_name(g) + "/" + name_of(p);
            hid_t gp = H5Gopen2(f, ppath.c_str(), H5P_DEFAULT);
            DipoleSlice s;
            s.gauge = g;
            s.pol   = p;
            s.D_reduced     = read_complex_vec(gp, "D_ortho_re", "D_ortho_im");
            s.D_reduced_raw = read_complex_vec(gp, "D_raw_re",   "D_raw_im");
            auto dr = read_1d_double(gp, "d_raw");
            auto dc = read_1d_double(gp, "d_correction");
            s.d_raw        = Eigen::Map<const Eigen::VectorXd>(dr.data(), dr.size());
            s.d_correction = Eigen::Map<const Eigen::VectorXd>(dc.data(), dc.size());
            s.partial_sigma = read_attr_double(gp, "partial_sigma");
            H5Gclose(gp);
            r.slices[i] = std::move(s);
        }
    }
    H5Fclose(f);
    return r;
}

}  // namespace scatt
