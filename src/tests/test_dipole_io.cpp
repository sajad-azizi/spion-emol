// test_dipole_io.cpp -- DipoleWriter/DipoleReader round-trip test.
//
// Covers:
//   * EnergyGrid arithmetic (ik -> k -> E, tag formatting)
//   * manifest.h5 write + read-back (grid, kgrid, occ, atoms, run strings)
//   * per-ik ik<nnnn>.h5 write + read-back with all 6 slices
//   * bit-equality of all numeric fields (real matrices, complex vectors)
//   * has_energy() / available_ik() / __SUCCESS__ marker
//   * atomic write behaviour (no stray .tmp files left behind)
//   * overwrite/resume: re-opening a scan dir does not rewrite manifest

#include "scatt/DipoleIO.hpp"
#include "scatt/EnergyGrid.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using scatt::DipoleEnergyPayload;
using scatt::DipoleGauge;
using scatt::DipoleReader;
using scatt::DipoleScanMeta;
using scatt::DipoleSlice;
using scatt::DipoleWriter;
using scatt::EnergyGrid;
using scatt::Polarization;

static int g_fail = 0;
static void check(bool cond, const std::string& what) {
    if (!cond) { std::cerr << "FAIL  " << what << "\n"; ++g_fail; }
    else       { std::cout << "ok    " << what << "\n"; }
}

static DipoleSlice make_slice(DipoleGauge g, Polarization p,
                              int n_ch, int n_occ, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    DipoleSlice s;
    s.gauge = g; s.pol = p;
    s.D_reduced.resize(n_ch);
    s.D_reduced_raw.resize(n_ch);
    s.d_raw.resize(n_ch);
    s.d_correction.resize(n_occ);
    for (int i = 0; i < n_ch;  ++i) {
        s.D_reduced[i]     = {U(rng), U(rng)};
        s.D_reduced_raw[i] = {U(rng), U(rng)};
        s.d_raw[i]         = U(rng);
    }
    for (int i = 0; i < n_occ; ++i) s.d_correction[i] = U(rng);
    s.partial_sigma = std::abs(U(rng));
    return s;
}

int main() {
    // ---------- EnergyGrid ----------
    EnergyGrid kg; kg.dk = 0.02; kg.ik_min = 1; kg.ik_max = 6;
    kg.validate();
    check(std::abs(kg.k(3) - 0.06) < 1e-15, "EnergyGrid::k(3) = 3*dk");
    check(std::abs(kg.E(3) - 0.5*0.06*0.06) < 1e-15, "EnergyGrid::E(3) = k^2/2");
    check(kg.tag(7) == "ik0007", "EnergyGrid::tag pads to 4 digits");
    check(kg.tag(1234) == "ik1234", "EnergyGrid::tag on 4-digit ik");

    // ---------- scratch dir ----------
    const std::string scratch = fs::temp_directory_path().string()
                              + "/static_exchangeHF_dipole_io_test";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    const std::string scan_dir = scratch + "/dipole_testscan";

    // ---------- meta ----------
    DipoleScanMeta meta;
    meta.r_min  = 0.0;
    meta.dr     = 0.005;
    meta.N_grid = 4001;
    meta.l_max_continuum = 8;
    meta.E_HOMO = -0.493;
    meta.kgrid  = kg;
    meta.occ_energies     = {-19.25, -1.34, -0.72, -0.58, -0.493};
    meta.occ_spin_factors = {2.0, 2.0, 2.0, 2.0, 1.0};
    meta.atoms = { {8, 0.0, 0.0, 0.0},
                   {1, 0.757, 0.0, 0.587},
                   {1,-0.757, 0.0, 0.587} };
    meta.molecule_name  = "H2O_test";
    meta.git_hash       = "deadbeef";
    meta.iso_date_utc   = "2026-04-23T00:00:00Z";
    meta.psi_dir_prefix = "/scratch/di35ker/psi_H2O_test";

    // ---------- payload ----------
    std::mt19937 rng(0x1234u);
    const int n_ch  = (meta.l_max_continuum + 1) * (meta.l_max_continuum + 1);
    const int n_occ = static_cast<int>(meta.occ_energies.size());

    DipoleEnergyPayload p;
    p.ik = 3;
    p.A.resize(n_ch, n_ch);
    p.B.resize(n_ch, n_ch);
    p.b_overlap.resize(n_ch, n_occ);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int i = 0; i < n_ch; ++i)
        for (int j = 0; j < n_ch; ++j) { p.A(i,j)=U(rng); p.B(i,j)=U(rng); }
    for (int i = 0; i < n_ch; ++i)
        for (int j = 0; j < n_occ; ++j) p.b_overlap(i,j) = U(rng);

    const DipoleGauge  gs[2] = { DipoleGauge::Length, DipoleGauge::Velocity };
    const Polarization ps[3] = { Polarization::X, Polarization::Y, Polarization::Z };
    for (auto g : gs) for (auto pp : ps) {
        p.slices[DipoleWriter::slice_index(g, pp)] = make_slice(g, pp, n_ch, n_occ, rng);
    }
    p.fit_residual_rel = 1.2e-7;
    p.K_symmetry_err   = 3.4e-9;

    // ---------- write ----------
    {
        DipoleWriter w(scan_dir, meta);
        check(!w.has_energy(3), "has_energy(3) false before write");
        w.write_energy(p);
        check(w.has_energy(3), "has_energy(3) true after write");
        w.finalize();
    }
    check(fs::exists(scan_dir + "/manifest.h5"),  "manifest.h5 written");
    check(fs::exists(scan_dir + "/ik0003.h5"),    "ik0003.h5 written");
    check(fs::exists(scan_dir + "/__SUCCESS__"),  "__SUCCESS__ marker written");

    // No stale .tmp files.
    for (const auto& e : fs::directory_iterator(scan_dir)) {
        const std::string n = e.path().filename().string();
        check(n.size() < 4 || n.substr(n.size()-4) != ".tmp",
              "no .tmp leftover: " + n);
    }

    // ---------- resume: re-open should NOT rewrite manifest ----------
    const auto mtime1 = fs::last_write_time(scan_dir + "/manifest.h5");
    {
        DipoleWriter w2(scan_dir, meta);  // should just pick up
        check(w2.has_energy(3), "resume: has_energy(3) still true");
    }
    const auto mtime2 = fs::last_write_time(scan_dir + "/manifest.h5");
    check(mtime1 == mtime2, "manifest.h5 not rewritten on resume");

    // ---------- read back ----------
    DipoleReader r(scan_dir);
    const auto& rm = r.meta();
    check(rm.r_min == meta.r_min, "meta.r_min round-trip");
    check(rm.dr    == meta.dr,    "meta.dr round-trip");
    check(rm.N_grid == meta.N_grid, "meta.N_grid round-trip");
    check(rm.l_max_continuum == meta.l_max_continuum, "meta.l_max_continuum round-trip");
    check(rm.E_HOMO == meta.E_HOMO, "meta.E_HOMO round-trip");
    check(rm.kgrid.dk == meta.kgrid.dk, "meta.kgrid.dk round-trip");
    check(rm.kgrid.ik_min == meta.kgrid.ik_min, "meta.kgrid.ik_min round-trip");
    check(rm.kgrid.ik_max == meta.kgrid.ik_max, "meta.kgrid.ik_max round-trip");
    check(rm.occ_energies == meta.occ_energies, "occ_energies round-trip");
    check(rm.occ_spin_factors == meta.occ_spin_factors, "occ_spin_factors round-trip");
    check(rm.atoms.size() == meta.atoms.size(), "atoms count round-trip");
    if (rm.atoms.size() == meta.atoms.size()) {
        bool atoms_ok = true;
        for (size_t i = 0; i < rm.atoms.size(); ++i) {
            if (rm.atoms[i].Z != meta.atoms[i].Z ||
                rm.atoms[i].x != meta.atoms[i].x ||
                rm.atoms[i].y != meta.atoms[i].y ||
                rm.atoms[i].z != meta.atoms[i].z) atoms_ok = false;
        }
        check(atoms_ok, "atoms coords/Z round-trip");
    }
    check(rm.molecule_name  == meta.molecule_name,  "molecule_name round-trip");
    check(rm.git_hash       == meta.git_hash,       "git_hash round-trip");
    check(rm.iso_date_utc   == meta.iso_date_utc,   "iso_date_utc round-trip");
    check(rm.psi_dir_prefix == meta.psi_dir_prefix, "psi_dir_prefix round-trip");

    check(r.available_ik() == std::vector<int>{3}, "available_ik = {3}");

    auto rp = r.read_energy(3);
    check(rp.ik == 3, "read ik");
    check(std::abs(rp.k - meta.kgrid.k(3)) < 1e-15, "read k matches grid");
    check(std::abs(rp.E - meta.kgrid.E(3)) < 1e-15, "read E matches grid");
    check(rp.fit_residual_rel == p.fit_residual_rel, "fit_residual_rel round-trip");
    check(rp.K_symmetry_err   == p.K_symmetry_err,   "K_symmetry_err round-trip");
    check(rp.A.isApprox(p.A, 0.0), "A bit-equal");
    check(rp.B.isApprox(p.B, 0.0), "B bit-equal");
    check(rp.b_overlap.isApprox(p.b_overlap, 0.0), "b_overlap bit-equal");

    bool slices_ok = true;
    for (auto g : gs) for (auto pp : ps) {
        const int i = DipoleWriter::slice_index(g, pp);
        const auto& a = p.slices[i];
        const auto& b = rp.slices[i];
        if (b.gauge != a.gauge || b.pol != a.pol) slices_ok = false;
        if (!(b.D_reduced     - a.D_reduced    ).isZero(0.0)) slices_ok = false;
        if (!(b.D_reduced_raw - a.D_reduced_raw).isZero(0.0)) slices_ok = false;
        if (!(b.d_raw         - a.d_raw        ).isZero(0.0)) slices_ok = false;
        if (!(b.d_correction  - a.d_correction ).isZero(0.0)) slices_ok = false;
        if (b.partial_sigma  != a.partial_sigma) slices_ok = false;
    }
    check(slices_ok, "all 6 slices round-trip bit-equal");

    // cleanup
    fs::remove_all(scratch, ec);

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " dipole_io  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
