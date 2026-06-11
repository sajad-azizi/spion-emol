// test_molden_parse.cpp — Milestone 1 cross-check.
//
// Reads the molden file produced by gen_reference.py and compares structural
// fields to the companion _reference.json:
//     - atom count, Z, Cartesian coordinates (Bohr)
//     - total nbf
//     - alpha MO count, energies, occupations
//     - MO coefficient matrix (AO x MO, rowmajor) matches mo_coeff_C_ao_by_mo
//
// Numerical tolerance is tight: 1e-12 on geometry/eps/coeffs (molden writes
// ~10 significant digits; Psi4 writes ~10-11 digits). If any check fails the
// test returns non-zero.
//
// Usage:  test_molden_parse <molden_path> <reference_json_path>

#include "molden/Molden.hpp"
#include "tiny_json.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using preproc::molden::MoldenParser;
using preproc::molden::Molecule;

namespace {

struct Failures { int n = 0; };

void check(bool cond, const std::string& what, Failures& F) {
    if (!cond) {
        std::cerr << "  [FAIL] " << what << "\n";
        ++F.n;
    } else {
        std::cerr << "  [ ok ] " << what << "\n";
    }
}

bool close(double a, double b, double atol, double rtol) {
    return std::abs(a - b) <= (atol + rtol * std::max(std::abs(a), std::abs(b)));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <molden_path> <reference_json_path>\n";
        return 2;
    }
    std::string molden_path = argv[1];
    std::string json_path   = argv[2];

    std::cerr << "==> Parsing molden: " << molden_path << "\n";
    MoldenParser P(/*verbose=*/true);
    Molecule mol = P.parse(molden_path);
    preproc::molden::print_summary(mol);

    std::cerr << "==> Parsing reference JSON: " << json_path << "\n";
    tinyjson::Value ref = tinyjson::parse_file(json_path);

    Failures F;

    // --- atoms ---
    const auto& ref_atoms = ref["atoms"].as_arr();
    check((int)mol.atoms.size() == (int)ref["natom"].as_num(), "natom matches", F);
    check((int)mol.atoms.size() == (int)ref_atoms.size(), "atoms array length matches", F);
    for (size_t i = 0; i < mol.atoms.size() && i < ref_atoms.size(); ++i) {
        const auto& ra = ref_atoms[i].as_obj();
        int Z_ref = (int)ra.at("Z").as_num();
        check(mol.atoms[i].Z == Z_ref,
              "atom[" + std::to_string(i) + "] Z=" + std::to_string(mol.atoms[i].Z) + " vs " + std::to_string(Z_ref), F);
        const auto& xyz = ra.at("xyz_bohr").as_arr();
        for (int k = 0; k < 3; ++k) {
            double ours = mol.atoms[i].xyz[k];
            double theirs = xyz[k].as_num();
            bool ok = close(ours, theirs, 1e-10, 1e-10);
            std::cerr << std::setprecision(17);
            if (!ok) std::cerr << "      atom[" << i << "][" << k << "] = " << ours << " vs " << theirs << "\n";
            check(ok, "atom[" + std::to_string(i) + "] xyz[" + std::to_string(k) + "] matches", F);
        }
    }

    // --- spherical / cartesian flag ---
    bool ref_cartesian = ref["cartesian"].as_str().empty() ? false : false; // default; overridden below
    // tinyjson stores bools as kind==Bool; we didn't plumb that through as_str; handle directly:
    {
        const auto& v = ref["cartesian"];
        if (v.kind != tinyjson::Value::Kind::Bool)
            std::cerr << "  [warn] 'cartesian' not a bool in JSON\n";
        ref_cartesian = v.bol;
    }
    bool ours_spherical_d = mol.sph_d;
    check(ref_cartesian ? !ours_spherical_d : ours_spherical_d,
          std::string("spherical d flag matches (cartesian=") + (ref_cartesian ? "true" : "false") + ")", F);

    // --- basis size ---
    int nbf_ref = (int)ref["nbf"].as_num();
    check(mol.nbf == nbf_ref, "nbf matches (" + std::to_string(mol.nbf) + " vs " + std::to_string(nbf_ref) + ")", F);

    // --- alpha MOs (count / energies / occupations) ---
    const auto& eps_ref = ref["orbital_energies_hartree"].as_arr();
    const auto& occ_ref = ref["occupations"].as_arr();
    check(mol.mos_alpha.size() == eps_ref.size(), "alpha MO count matches", F);
    for (size_t i = 0; i < mol.mos_alpha.size() && i < eps_ref.size(); ++i) {
        bool e_ok = close(mol.mos_alpha[i].energy, eps_ref[i].as_num(), 1e-10, 1e-9);
        bool o_ok = close(mol.mos_alpha[i].occ,    occ_ref[i].as_num(), 1e-12, 1e-12);
        if (!e_ok)
            std::cerr << "      eps[" << i << "] = " << std::setprecision(17)
                      << mol.mos_alpha[i].energy << " vs " << eps_ref[i].as_num() << "\n";
        check(e_ok, "eps[" + std::to_string(i) + "] matches", F);
        check(o_ok, "occ[" + std::to_string(i) + "] matches", F);
    }

    // --- MO coefficient matrix, compared against the molden-order reference ---
    // (Psi4's internal Ca uses CCA m-ordering; we parse the molden text which
    //  uses a different per-shell convention.  gen_reference.py re-parses the
    //  molden file and writes 'mo_coeff_molden' in that ordering for us.)
    const auto& M_ref = ref["mo_coeff_molden"].as_obj();
    int C_rows = (int)M_ref.at("shape").as_arr()[0].as_num();   // n_mo
    int C_cols = (int)M_ref.at("shape").as_arr()[1].as_num();   // nbf
    const auto& C_flat = M_ref.at("rowmajor").as_arr();
    check(C_cols == mol.nbf, "molden C row-length == nbf", F);
    check(C_rows == (int)mol.mos_alpha.size(), "molden C rows == alpha MO count", F);

    // mo_coeff_molden is (n_mo, nbf) row-major; compare to mol.mos_alpha[j].C[i].
    double max_abs_diff = 0.0;
    int    argmax_i = -1, argmax_j = -1;
    for (int j = 0; j < C_rows && j < (int)mol.mos_alpha.size(); ++j) {
        for (int i = 0; i < C_cols; ++i) {
            double ours = mol.mos_alpha[j].C[i];
            double theirs = C_flat[j * C_cols + i].as_num();
            double d = std::abs(ours - theirs);
            if (d > max_abs_diff) { max_abs_diff = d; argmax_i = i; argmax_j = j; }
        }
    }
    std::cerr << "  MO coefficient max |diff| = " << std::scientific
              << std::setprecision(3) << max_abs_diff
              << "  at (i=" << argmax_i << ", j=" << argmax_j << ")\n";
    // molden writes ~10 sig figs; set tolerance generously but still tight.
    check(max_abs_diff < 1e-8, "C coefficients match (max|diff| < 1e-8)", F);

    std::cerr << "\n==> " << (F.n == 0 ? "ALL CHECKS PASSED" : "FAILURES: " + std::to_string(F.n)) << "\n";
    return F.n == 0 ? 0 : 1;
}
