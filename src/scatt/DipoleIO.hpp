// DipoleIO.hpp -- persistent HDF5 output for a scattering energy scan.
//
// Layout (on disk):
//
//   $PERSISTENT/<scan_dir>/
//       manifest.h5          # scan-level metadata (once)
//       ik0001.h5            # per-energy payload (dipole, A, B, overlaps, ...)
//       ik0002.h5
//       ...
//       __SUCCESS__          # zero-byte marker written on finalize()
//
// Design choices (from design discussion 2026-04-23):
//
//   * Persistent, not checkpoint. Survives across jobs. Separate dir per scan.
//   * ik-indexed: k(ik) = ik·dk, E(ik) = 0.5·k². File name = zero-padded ik.
//   * A and B only (not K, S). K = B·A^(-1), S = (A+iB)(A-iB)^(-1) in post.
//   * Store both raw and ortho dipoles for both gauges × 3 polarizations.
//   * Store everything post-processing needs for continuum–continuum dipole:
//       channel list, occupied list, grid, A, B, ψ-dir pointer (in manifest).
//   * Resumable: if ik<nnnn>.h5 exists, skip. Atomic writes via .tmp + rename.
//
// All HDF5 calls use the scatt::io::_check helper for error reporting, and
// the same C-API style as HDF5Reader / the preprocessing writer. No external
// JSON dependency.

#pragma once

#include "scatt/DipoleMatrixElement.hpp"
#include "scatt/EnergyGrid.hpp"

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <string>
#include <vector>

namespace scatt {

// Everything fixed across a scan -- channel basis, occupied orbitals,
// radial grid, solver setup. Written to manifest.h5 once on open().
struct DipoleScanMeta {
    // Radial grid (uniform).
    double        r_min  = 0.0;
    double        dr     = 0.0;
    std::size_t   N_grid = 0;

    // Angular cutoff of the continuum basis.
    int           l_max_continuum = 0;

    // Energy grid.
    EnergyGrid    kgrid;

    // Initial-state energy (Koopmans hν = E_kin - E_HOMO).
    double        E_HOMO  = 0.0;

    // Occupied-orbital info (per α): energy, spin occupation N_α.
    // Useful for post-processing and for rebuilding orthogonalization.
    std::vector<double>  occ_energies;     // size n_occ
    std::vector<double>  occ_spin_factors; // size n_occ

    // Molecule (atom list, translated to origin=(0,0,0) as per convention).
    struct Atom { int Z; double x, y, z; };
    std::vector<Atom>    atoms;

    // Pointer to the ψ directory (scratch) for possible continuum–continuum
    // dipole computation in post-processing. Optional; empty = not recorded.
    std::string          psi_dir_prefix;

    // Free-form run identification.
    std::string          molecule_name;
    std::string          git_hash;
    std::string          iso_date_utc;    // "2026-04-23T10:00:00Z"

    // Conventions (written as attributes for future-proof readers):
    //   "real_Ylm_q_map" : "x=+1, y=-1, z=0"
    //   "psi_norm"       : "incoming-wave Psi- = (A - iB)^(-†)"
    //   "u_convention"   : "chi = r * F_lm(r)"
};

// Per-(gauge, polarization) payload. Six of these per energy point.
struct DipoleSlice {
    DipoleGauge       gauge;
    Polarization      pol;

    Eigen::VectorXcd  D_reduced;       // (n_ch) with orthogonalization
    Eigen::VectorXcd  D_reduced_raw;   // (n_ch) without orthogonalization
    Eigen::VectorXd   d_raw;           // (n_ch) real ⟨ψ_β|O|Φ_i⟩
    Eigen::VectorXd   d_correction;    // (n_occ) real ⟨φ_α|O|Φ_i⟩

    // Lightweight diagnostics (written as attributes).
    double            partial_sigma     = 0.0;
};

// Per-energy data, gathered in solve() and handed in a single call.
struct DipoleEnergyPayload {
    int              ik;
    // A, B as returned by AsymptoticAmplitudes. Shape (n_ch, n_ch), real.
    Eigen::MatrixXd  A;
    Eigen::MatrixXd  B;

    // Overlap b_{βα} = ⟨φ_α|ψ_β⟩. Gauge/polarization-independent.
    Eigen::MatrixXd  b_overlap;        // (n_ch, n_occ), real

    // All six (gauge × pol) slices.
    std::array<DipoleSlice, 6> slices;

    // Asymptotic-fit quality at this energy (from KMatrixExtractor/
    // AsymptoticAmplitudes). Optional -- default NaN is fine if unknown.
    double           fit_residual_rel  = 0.0;
    double           K_symmetry_err    = 0.0;
};

// Writer owning the scan directory. One instance per scan.
class DipoleWriter {
public:
    DipoleWriter(const std::string& scan_dir, const DipoleScanMeta& meta);

    // Has ik<ik>.h5 already been written? Used by solve() to skip.
    bool has_energy(int ik) const;

    // Atomic write (.tmp -> rename). Throws on I/O failure.
    void write_energy(const DipoleEnergyPayload& payload);

    // Write __SUCCESS__ marker. Call once at the end of the scan.
    void finalize();

    const std::string& scan_dir() const { return scan_dir_; }

    // Helper: assemble six-slice array in canonical order.
    // Order: {L,X}, {L,Y}, {L,Z}, {V,X}, {V,Y}, {V,Z}.
    static int slice_index(DipoleGauge g, Polarization p) {
        const int ig = (g == DipoleGauge::Length) ? 0 : 1;
        int ip = 0;
        switch (p) { case Polarization::X: ip = 0; break;
                     case Polarization::Y: ip = 1; break;
                     case Polarization::Z: ip = 2; break; }
        return ig * 3 + ip;
    }

private:
    std::string      scan_dir_;
    DipoleScanMeta   meta_;

    std::string path_for_(int ik) const;
    void        ensure_dir_() const;
    void        write_manifest_() const;
};

// Round-trip reader, primarily for tests. Post-processing is Python-first
// but C++ consumers (continuum–continuum dipole) will need to read back.
struct DipoleEnergyReadback {
    int              ik;
    double           k, E;
    Eigen::MatrixXd  A, B, b_overlap;
    std::array<DipoleSlice, 6> slices;
    double           fit_residual_rel = 0.0;
    double           K_symmetry_err   = 0.0;
};

class DipoleReader {
public:
    explicit DipoleReader(const std::string& scan_dir);

    const DipoleScanMeta& meta() const { return meta_; }

    // List of ik values that have a written ik<nnnn>.h5 (sorted ascending).
    std::vector<int> available_ik() const;

    DipoleEnergyReadback read_energy(int ik) const;

private:
    std::string     scan_dir_;
    DipoleScanMeta  meta_;

    void read_manifest_();
};

}  // namespace scatt
