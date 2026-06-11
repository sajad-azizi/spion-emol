# Two-photon photoionization delays (exact 2nd-order perturbation theory)

Implementation of attosecond two-photon delays in **molecular** photoionization
following Baykusheva & Wörner, J. Chem. Phys. **146**, 124306 (2017),
referred to here as "BW17".

This pipeline consumes the **saved scattering wavefunctions** ψ produced by
the main `scattering` C++ driver, computes all matrix elements
⟨ψₙ | μ | ψₙ′⟩ between every pair of states (bound–bound, bound–continuum,
continuum–continuum), and from those builds the exact 2nd-order amplitude

> _M(k⃗; εᵢ + Ω; R̂_γ)_  (BW17 Eq. 7),

the molecular-frame two-photon delay τ₂ₕᵥ (BW17 Eq. 21), the universal
continuum–continuum delay τ_cc (BW17 Eq. 24), the molecular contribution
τ_mol = τ₂ₕᵥ − τ_cc, and the orientation/angle averages (Eqs. 26–29).

**Accuracy is the priority.** No high-Ω truncation; both the bound-state
sum and the continuum integral in BW17 Eq. 12 are evaluated.

---

## Convention reminder (atomic units throughout)

Final state with incoming-wave BC (BW17 Eq. 2):

    ψ_{f,k̂}^(-)(r⃗) = √(2/π) Σ_lm i^l ψ_{f,lm}^(-)(r⃗) Y*_lm(k̂)

Single-center expansion (BW17 Eq. 8):

    ψ_{f,lm}^(-)(r⃗) = √(2/π) i^L e^{-i η_L(k)} Y*_LM(k̂) Y_LM(r̂) R_kL(r)
    R_kL(r→∞) ~ (1/k r) sin(kr − Lπ/2 + η_L(k))    (Coulomb-shifted)

Real spherical harmonics with q-map x → +1, y → −1, z → 0
(matches the convention used elsewhere in this codebase; see
`postprocessing/cross_section_delay.py`).

u/r convention: χ = r · F.  All radial functions stored as χ(r) = r · F(r).

---

## What we compute, in order

### Phase A — matrix elements (C++)

Saves to `${WORK}/two_photon_me_<scan_id>.h5`:

1. **bound–bound** ⟨φ_α | r̂·ε̂ | φ_β⟩ — between occupied orbitals.
   Cheap; uses preprocessing R_α(r) on the radial grid.
2. **bound–continuum (XUV step)** ⟨ψ_ν^(-) | r̂·ε̂ | φ_i⟩ — for every ν in
   the energy scan.  This is the existing `d_raw` from the main scattering
   pipeline; we just gather it.
3. **continuum–continuum (IR step)** ⟨ψ_κ^(-) | r̂·ε̂ | ψ_ν⟩ — between
   every pair (ε_κ, ε_ν) of saved continuum states, in length AND
   velocity gauges, for x/y/z polarization.  **NEW** computation,
   reads two psi files per pair.

### Phase B — SEHF bound spectrum (C++)

The 2nd-order PT sum runs over a **complete** set of intermediate
states ν.  In addition to the continuum integral (item 3 above), it
includes a discrete sum over **bound** intermediate states of the
**static-exchange-HF** Hamiltonian at the same partial-wave channel —
i.e. eigenstates of the operator that produced the continuum we
already have, just at energies ε_ν < 0.

These are NOT the same as the neutral-system occupied orbitals; they
are eigenstates of (ion + electron in static-exchange field) with E < 0.
A small Numerov-bisection finder is added that:
   * walks the same SEHF Hamiltonian operator the scattering code uses;
   * locates roots of det(boundary condition) at each negative energy;
   * dumps R_νλ(r) on the radial grid for use in Phase C.

### Phase C — assemble the 2-photon amplitude (Python)

From the matrix-element file, for each scan energy ε_κ:

1. Form the **radial transition T_{Lλl}(k; ε_κ)** (BW17 Eq. 12) by:
   * **discrete sum** over bound intermediate states (occ + SEHF bound);
   * **principal-value + iπ-residue** continuum integral, using the
     gathered c-c radial dipoles on the energy grid.  The on-shell pole
     at ε_ν = ε_i + Ω is treated by Sokhotski–Plemelj:

         lim_{ε→0+} 1/(x + iε) = P/x − iπ δ(x)

     The principal-value piece is integrated by the standard pole-
     subtraction quadrature (Simpson on (E − E_pole) f(E) plus the
     analytic correction).
2. Combine T with the angular Gaunt's and Wigner-D's per BW17 Eq. 11
   to get **M(k⃗; ε_i + Ω; R̂_γ)**.
3. Compute **τ(2q, k̂, R̂_γ) = (1/2ω) arg[M^(2q-1)* M^(2q+1)]**
   (BW17 Eq. 21).
4. Compute **τ_cc(2q)** (BW17 Eq. 24) from the universal A_{κk}.
5. **τ_mol = τ − τ_cc** (BW17 Eq. 25).
6. Orientation / angular averages (Eqs. 26–29).

### Phase D — convergence + plots

Convergence checks (test-driven — every step verified before moving on):
* c-c dipole symmetry under (κ ↔ ν) swap (Hermitian operator)
* gauge consistency: τ from length and velocity should agree at
  converged Hamiltonian; deviation = quality diagnostic
* bound-state contribution magnitude vs continuum integral (so we
  know when the bound piece matters)
* convergence of the principal-value integral with refinement of dε_ν
* free-particle / hydrogenic-Coulomb limits, where closed-form
  τ_cc and τ_1hν are known (Lindroth, Dahlstrom)
* atomic-limit consistency: when angular-averaged for an effectively
  spherical target, τ ≈ τ_1hν + τ_cc (BW17 Eq. 30)

---

## Folder layout

```
two_photon_delay/
├── README.md                  this file
├── docs/                      math derivations, conventions
├── cc_dipole/                 C++: continuum-continuum dipole calculator
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── cc_dipole.cpp      main: reads two psi dirs, writes CC HDF5
│   │   └── CCAssembler.cpp    radial integral + angular Gaunt
│   ├── include/
│   │   └── CCAssembler.hpp
│   └── tests/                 ctest suite
├── sehf_bound/                C++: bound-state finder for SEHF Hamiltonian
│   ├── CMakeLists.txt
│   └── src/
├── python/                    end-to-end pipeline
│   ├── compute_T.py           BW17 Eq. 12 (energy integral)
│   ├── compute_M.py           BW17 Eq. 11 (angular assembly)
│   ├── tau_cc.py              BW17 Eq. 24
│   ├── tau_2hv.py             BW17 Eq. 21
│   └── tests/                 pytest suite
└── tests/                     end-to-end integration tests
```

---

## Status

Phase A: in progress (cc_dipole)
Phase B: pending
Phase C: pending
Phase D: pending

Each phase ends with a documented test pass; the next phase starts only
after the previous has been verified.
