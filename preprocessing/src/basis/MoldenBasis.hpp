// MoldenBasis.hpp — build a unit-normalized contracted AO basis from a
// parsed Molden Molecule.
//
// Data model: each AO is a spherical-harmonic Gaussian,
//
//     chi^sph_{l,m}(r) = |r - A|^l * Y^R_{l,m}(theta, phi)
//                        * [ sum_a c_tilde_a * exp(-alpha_a |r - A|^2) ]
//
// where theta, phi are the spherical angles of (r - A). The coefficients
// c_tilde_a absorb (i) the per-primitive spherical normalization
// N_sph(alpha_a, l), and (ii) a scale that renormalizes the contracted AO
// to unit overlap.
//
// This works uniformly for s, p, d, f, g (and higher if ever needed) with
// NO polynomial tables -- we use angular::real_Ylm directly.
//
// Supported shell types:
//   - PURE (spherical, molden flagged [5D]/[5D7F]/[9G]): any l up to the
//     Lmax that real_Ylm handles. p is always pure per molden spec.
//   - CARTESIAN (l <= 1): trivial (same as pure for s/p).
//   - CARTESIAN for l >= 2: NOT YET SUPPORTED; we throw. Psi4's default
//     output for cc-pV?Z uses spherical so this is rarely hit. Adding it
//     requires the Cart->Sph transform matrices (Milestone 2c).

#pragma once

#include "angular/Ylm.hpp"
#include "basis/Primitive.hpp"
#include "basis/SphericalGaussian.hpp"
#include "molden/Molden.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace preproc::basis {

// One atomic orbital chi_mu in MOLDEN order.
// We evaluate:
//     ang(dx,dy,dz) = r^l * Y^R_{l,m}(theta, phi)      where r = |d|
//     rad(r)        = sum_a c_tilde_a * exp(-alpha_a r^2)
//     chi(R_abs)    = ang * rad
struct AtomicOrbital {
    int shell_index;
    int atom_index;
    int l;
    int m;                           // in [-l, l], molden-order semantics
    Eigen::Vector3d center;          // in Bohr, absolute
    std::vector<double> exponents;   // alpha_a
    std::vector<double> c_tilde;     // includes sqrt-normalization factor

    // Evaluate chi at an absolute position R_abs.
    double eval(const Eigen::Vector3d& R_abs) const {
        const double dx = R_abs[0] - center[0];
        const double dy = R_abs[1] - center[1];
        const double dz = R_abs[2] - center[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double r  = std::sqrt(r2);

        double Yang;
        if (r < 1e-300) {
            // At the nucleus: r^l * Y^R is 0 for l >= 1 (regular solid
            // harmonic is a polynomial vanishing at the origin). For l=0
            // it is simply Y^R_{0,0} = 1/sqrt(4 pi).
            if (l == 0) Yang = 1.0 / std::sqrt(4.0 * M_PI);
            else        return 0.0;
        } else {
            const double theta = std::acos(std::clamp(dz / r, -1.0, 1.0));
            const double phi   = std::atan2(dy, dx);
            const double r_l   = (l == 0) ? 1.0 : std::pow(r, l);
            Yang = r_l * angular::real_Ylm(l, m, theta, phi);
        }

        double rad = 0.0;
        for (size_t a = 0; a < exponents.size(); ++a) {
            rad += c_tilde[a] * std::exp(-exponents[a] * r2);
        }
        return Yang * rad;
    }
};

struct MoldenBasis {
    std::vector<AtomicOrbital> aos;   // length == mol.nbf

    // Evaluate all AOs at once at a single point. Useful for MO/density
    // evaluation via dot products.
    Eigen::VectorXd evaluate_all(const Eigen::Vector3d& r) const {
        Eigen::VectorXd v(aos.size());
        for (size_t m = 0; m < aos.size(); ++m) v[m] = aos[m].eval(r);
        return v;
    }
};

// Convert molden shell-index-within-shell (0..2l) to actual m value.
// Molden order:  s:   m = 0
//                p:   m = +1, -1, 0     (but expressed as px, py, pz)
//                d:   m = 0, +1, -1, +2, -2
//                f:   m = 0, +1, -1, +2, -2, +3, -3
//                g:   m = 0, +1, -1, +2, -2, +3, -3, +4, -4
//
// Note: p is special. In molden, the [5D7F] flag does not apply to p (p is
// always 3 components). The molden ordering for p is (p_x, p_y, p_z),
// which, under the standard (no-CS) real-Ylm convention, is (m=+1, m=-1,
// m=0). That mapping is baked in below.
inline int molden_index_to_m(int l, int idx_in_shell) {
    if (l == 0) return 0;
    if (l == 1) {
        // p: order (p_x, p_y, p_z) == (m=+1, m=-1, m=0)
        switch (idx_in_shell) {
            case 0: return +1;
            case 1: return -1;
            case 2: return  0;
            default: throw std::runtime_error("p shell idx out of range");
        }
    }
    // l >= 2 (spherical): m = 0, +1, -1, +2, -2, ..., +l, -l
    if (idx_in_shell == 0) return 0;
    // For idx_in_shell = 2k-1 -> m = +k,  idx_in_shell = 2k -> m = -k.
    const int half = (idx_in_shell + 1) / 2;
    return (idx_in_shell & 1) ? +half : -half;
}

// Build the normalized AO basis from a parsed Molecule.
inline MoldenBasis build_basis(const molden::Molecule& mol, bool verbose = true) {
    MoldenBasis basis;
    basis.aos.reserve(mol.nbf);

    for (size_t s = 0; s < mol.shells.size(); ++s) {
        const auto& sh = mol.shells[s];
        const auto& at = mol.atoms[sh.atom_index];
        const int L = sh.l;

        // Cartesian d/f/g not yet implemented (Milestone 2c).
        if (!sh.pure && L >= 2) {
            std::ostringstream oss;
            oss << "build_basis: Cartesian l=" << L << " shell not supported yet. "
                << "Molden file should have [5D7F] / [9G] flags to select spherical.";
            throw std::runtime_error(oss.str());
        }

        // Contraction renormalization. Same formula works for all l (see
        // SphericalGaussian.hpp for the derivation):
        //   < chi_normalized | chi_normalized > = 1  after dividing by
        //   sqrt( sum_{a,b} c_a c_b * primitive_overlap_same_shell(alpha_a, alpha_b, L) ).
        const int n_prim = static_cast<int>(sh.exponents.size());
        double self_overlap = 0.0;
        for (int a = 0; a < n_prim; ++a)
            for (int b = 0; b < n_prim; ++b)
                self_overlap += sh.coeffs[a] * sh.coeffs[b]
                              * primitive_overlap_same_shell(sh.exponents[a], sh.exponents[b], L);
        if (self_overlap <= 0.0)
            throw std::runtime_error("build_basis: non-positive self-overlap");
        const double scale = 1.0 / std::sqrt(self_overlap);

        // For each of the (2L+1) spherical components, emit one AtomicOrbital
        // in molden ordering.
        const int n_ao_in_shell = (L == 0) ? 1 : (2 * L + 1);
        for (int idx = 0; idx < n_ao_in_shell; ++idx) {
            const int m = molden_index_to_m(L, idx);
            AtomicOrbital ao;
            ao.shell_index = static_cast<int>(s);
            ao.atom_index  = sh.atom_index;
            ao.l = L;
            ao.m = m;
            ao.center = at.xyz;
            ao.exponents = sh.exponents;
            ao.c_tilde.resize(n_prim);
            for (int a = 0; a < n_prim; ++a) {
                ao.c_tilde[a] = scale * sh.coeffs[a]
                              * spherical_primitive_norm(sh.exponents[a], L);
            }
            basis.aos.push_back(std::move(ao));
        }

        if (verbose) {
            std::cerr << "[basis] shell " << s << " l=" << L
                      << " n_prim=" << n_prim
                      << " self_overlap(pre-norm)=" << self_overlap
                      << " -> scale=" << scale
                      << "  (" << n_ao_in_shell << " AOs)\n";
        }
    }

    if (static_cast<int>(basis.aos.size()) != mol.nbf) {
        std::ostringstream oss;
        oss << "build_basis: built " << basis.aos.size() << " AOs but mol.nbf=" << mol.nbf;
        throw std::runtime_error(oss.str());
    }
    return basis;
}

// Evaluate a single molecular orbital psi_i(r) = sum_mu C_mu * chi_mu(r).
inline double evaluate_mo(const MoldenBasis& basis, const molden::MO& mo,
                          const Eigen::Vector3d& r) {
    Eigen::VectorXd phi = basis.evaluate_all(r);
    Eigen::Map<const Eigen::VectorXd> C(mo.C.data(),
                                        static_cast<Eigen::Index>(mo.C.size()));
    return phi.dot(C);
}

// Closed-shell total density rho(r) = sum_i n_i |psi_i(r)|^2.
inline double evaluate_density(const MoldenBasis& basis,
                               const molden::Molecule& mol,
                               const Eigen::Vector3d& r) {
    Eigen::VectorXd phi = basis.evaluate_all(r);
    double rho = 0.0;
    for (const auto& mo : mol.mos_alpha) {
        if (mo.occ == 0.0) continue;
        Eigen::Map<const Eigen::VectorXd> C(mo.C.data(),
                                            static_cast<Eigen::Index>(mo.C.size()));
        const double psi = phi.dot(C);
        rho += mo.occ * psi * psi;
    }
    for (const auto& mo : mol.mos_beta) {
        if (mo.occ == 0.0) continue;
        Eigen::Map<const Eigen::VectorXd> C(mo.C.data(),
                                            static_cast<Eigen::Index>(mo.C.size()));
        const double psi = phi.dot(C);
        rho += mo.occ * psi * psi;
    }
    return rho;
}

}  // namespace preproc::basis
