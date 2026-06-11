# diagnostic scripts

Standalone python diagnostics, used to investigate the `σ_y → 1e-17` bug
on the C₈F₈ photoionization run (April 2026).  Each script is a single
file with no project deps beyond `numpy` (and `scipy` for one of them).

## diagnose_somo.py

Reads a Psi4-format molden file, locates the alpha SOMO (highest-energy
alpha occupied MO), and decomposes its coefficients across basis-function
(l, m).  Reports the per-Cartesian (m=+1, -1, 0) shares so you can see
whether Psi4's UKS picked a single Cartesian-aligned t₁ᵤ component
(would explain σ_y = 0 if y-share ≈ 0) or a fully symmetric mixture.

```
python3 diagnose_somo.py <anion.molden>
```

For the production C₈F₈ B3LYP/cc-pVDZ run the output is
```
m = +1  (x-like)  : 10.99%
m = -1  (y-like)  : 10.99%       <-- y is fully present
m =  0  (z-like)  : 10.76%
```
which **rules out** the "no-y SOMO" hypothesis and points the σ_y bug
downstream of preprocessing.

## audit_real_gaunt_table.py

Numerically integrates G^R(l_mu m_mu; 1 q; l_nu m_nu) on a Gauss-
Legendre × trapezoid grid and tells you which couplings the C++
scattering code's `m_mu == m_nu + q` gate accepts vs which legitimate
real-Y couplings it drops.

```
python3 audit_real_gaunt_table.py
```

Output for q = -1 (y polarization), pre-fix:
```
KEPT  5 couplings, sum|G| = 0.97
DROPPED 21 couplings, sum|G| = 3.72
```
Largest dropped: `G^R(s; y; p_y) = +0.282` -- the canonical y-dipole
matrix element.

Post-fix `src/scatt/DipoleMatrixElement.cpp:205` no longer applies the
complex-Y gate; the C++ regression `test_real_gaunt_dipole` covers all
20 representative triples (l ≤ 2) and runs in <1 ms.
