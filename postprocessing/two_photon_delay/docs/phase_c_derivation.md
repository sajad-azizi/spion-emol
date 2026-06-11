# Phase C derivation — m-resolved two-photon delay in our real-Y_lm basis

**SCOPE.**  Derive the two-photon photoionisation amplitude `M(k⃗; ε_i + Ω; R̂_γ)`
in a form that maps directly onto our Phase A HDF5 quantities,
m-resolved (no single-center approximation), with explicit signs.

**FORMALISM.**  We follow:
* Baykusheva & Wörner JCP 2017 ("BW17") for the **angular** structure
  (Eqs. 2, 4, 11, 21, 24-29) — the m-resolved generalisation is derived
  in §D4-D6 below;
* **Azizi, Saalmann, Rost (arXiv:2407.16270, 2024) Eq. 7** for the
  **radial / energy** structure of the 2nd-order amplitude.  This
  replaces the Sokhotski-Plemelj principal-value treatment with a
  finite-pulse closed-form involving the Dawson function — see §D7
  below.  This is rigorously the 2nd-order PT amplitude for a Gaussian
  pulse of FWHM T, and recovers BW17's monochromatic / Sokhotski-Plemelj
  result as T → ∞.

Before any code is written this document must be reviewed.  If anything
is wrong here it propagates silently into τ.

---

## Notation summary

| Symbol | Meaning | Where it lives in our data |
|---|---|---|
| `Ψ_i` | initial bound state (anion SOMO) | preproc `/initial_state/psi_lm` |
| `ε_i` | initial state energy | `E_HOMO` in Phase A HDF5 |
| `Ω`   | XUV photon energy (atomic units) | scan parameter |
| `ω`   | IR photon energy | scan parameter |
| `ε_κ` | final photoelectron kinetic energy = ε_i + 2ω + Ω (on shell at sideband 2q) | derived per pair |
| `ε_ν` | intermediate-state energy | swept in continuum integral |
| `Ψ⁻_(κ, μ_κ)` | final continuum in-state at energy ε_κ, ASYMPTOTIC channel μ_κ | `/per_ik/` + rotation |
| `μ` (channel index) | μ = l·l + l + m (real-Y_lm packing) | `/channels/l_mu`, `m_mu` |
| `Y^R_(l,m)` | REAL spherical harmonics (Condon-Shortley) | our convention everywhere |
| `q ∈ {x, y, z}` | Cartesian polarisation direction in MOLECULAR frame | `/per_ik/...x|y|z` |
| `m_p ∈ {-1, 0, +1}` | spherical-tensor polarisation in MOLECULAR frame | derived from q via Eq. (D5) below |
| `R̂_γ = (α, β, γ)` | Euler angles MOL → LAB | runtime parameter |
| `k̂ = (θ_k, φ_k)` | LAB-frame photoelectron emission direction | runtime parameter |

The dipole operator in the molecular frame is `ε̂_q · r = r_q`.  Real
Cartesian basis q ∈ {x, y, z}; spherical-tensor basis m_p ∈ {-1, 0, +1}.

---

## (D1) Real-Y_lm ↔ complex-Y_lm transformation

Following our C++ convention (see `src/angular/Gaunt.hpp:90-109`):

    Y^C(l, m_C) = Σ_{m_R} U(l, m_R, m_C) · Y^R(l, m_R)

with the block-diagonal (per-l) unitary matrix

    U(l, 0, 0)         =  1
    U(l, +k, -k)       =  1/√2
    U(l, +k, +k)       =  (-1)^k / √2
    U(l, -k, -k)       =  +i/√2
    U(l, -k, +k)       = -i·(-1)^k / √2
    all other          =  0

Unitarity: `U U† = I` per l-block.  Inverse:

    Y^R(l, m_R) = Σ_{m_C} U*(l, m_R, m_C) · Y^C(l, m_C)

A state `Ψ = Σ_(l,m_R) c^R_(l,m_R) Y^R_(l,m_R) = Σ_(l,m_C) c^C_(l,m_C) Y^C_(l,m_C)`
has coefficient transform `c^C = U† · c^R` (rows = m_R, cols = m_C).

A matrix element transforms as

    M^C[μ_C, μ_C'] = (U† M^R U)[μ_C, μ_C']

with the full N_psi × N_psi U built block-diagonal-by-l from the
per-l blocks above.

---

## (D2) Cartesian → spherical polarisation

The Cartesian molecular-frame dipole `ε̂_q · r` with q ∈ {x, y, z}
relates to the spherical-tensor `T^{(1)}_{m_p}` with m_p ∈ {-1, 0, +1}
via (Condon-Shortley):

    T^{(1)}_(-1)  =  ( +x − i·y) / √2
    T^{(1)}_( 0)  =    z
    T^{(1)}_(+1)  =  (−x − i·y) / √2

Inverting:

    x  =  ( T^{(1)}_(-1) − T^{(1)}_(+1) ) / √2          ... (D2a)
    y  =  i · ( T^{(1)}_(-1) + T^{(1)}_(+1) ) / √2      ... (D2b)
    z  =    T^{(1)}_(0)                                  ... (D2c)

So a Cartesian matrix-element vector `M^q ∈ {M^x, M^y, M^z}` converts
to spherical-tensor components `M^{m_p}` via

    M^{m_p = -1}  =  ( M^x − i·M^y) / √2     ... (D2d)
    M^{m_p =  0}  =     M^z                  ... (D2e)
    M^{m_p = +1}  =  (−M^x − i·M^y) / √2     ... (D2f)

(These are linear relations.  M^q is real-valued × the angular Gaunt's;
the resulting M^{m_p} is complex.)

---

## (D3) Asymptotic in-state matrix elements in our basis

From Phase A we have, for each (κ, ν, gauge, q) and (in-state μ_κ, μ_ν):

    M_in^{q}_{κν}[μ_κ, μ_ν]
        = U_κ†  ·  cc_raw^{q}[β, α]  ·  U_ν                                ... (D3a)

where `U_κ = (A_κ − iB_κ)⁻¹`.  `μ_κ` and `μ_ν` index our real-Y_lm
channels (μ = l² + l + m).  M_in is **complex**.

By the Hermiticity of the dipole operator and the reality of ψ_β:

    M_in^{q}_{κν}[μ_κ, μ_ν]  =  M_in^{q}_{νκ}[μ_ν, μ_κ]*                  ... (D3b)

(Verified numerically: rel error 3 × 10⁻¹⁵ on H₂O fixture.)

Similarly, for the one-photon (b-c) step we already have from the main
scattering pipeline:

    D^{q}_{ν, μ_ν}  =  <Ψ⁻_(ν, μ_ν) | r·ε̂_q | Ψ_i>                       ... (D3c)

in the same in-state basis (stored as `D_ortho_<gauge>_<pol>_{re,im}`
per ik; orthogonalisation correction already applied).

---

## (D4) BW17 Eq. 7 in our basis (m-resolved, no single-center)

BW17 Eq. 7:

    M(k⃗; ε_i + Ω; R̂_γ)
        = (1/i) lim_{ε→0+} ∫ dε_ν
            <Ψ⁻_{f, k̂} | ξ̂^IR_{m_p^IR} | ν> <ν | ξ̂^XUV_{m_p^XUV} | Ψ_i>
            / (ε_i + Ω − ε_ν + iε)

BW17 Eq. 2 (lab-frame partial-wave expansion of Ψ⁻):

    Ψ⁻_{f, k̂}(r⃗) = √(2/π) Σ_{L M_C} i^L  ψ⁻_(f, LM_C)(r⃗)  Y^C*_(LM_C)(k̂)
                                                                       ... (D4a)

The label (L, M_C) here is the asymptotic OUTGOING channel of the
in-state Ψ⁻ in **complex** Y_lm.  We have these states in our **real**
Y_lm basis: ψ⁻_{f, μ_κ}(r⃗) with μ_κ = L² + L + M_R.  The relation:

    ψ⁻_(f, LM_C)(r⃗) = Σ_{M_R} U*(L, M_R, M_C) · ψ⁻_(f, (L, M_R))(r⃗)
                                                                       ... (D4b)

so the matrix element decomposes as:

    <Ψ⁻_{f, k̂} | ξ̂^q_mol | ν>
        = √(2/π) Σ_{L M_C} (−i)^L  Y^C_(LM_C)(k̂)
                            · Σ_{M_R} U(L, M_R, M_C) <ψ⁻_{f, (L, M_R)} | ξ̂^q_mol | ν>
                                                                       ... (D4c)

We did NOT yet decompose ν.  The intermediate state ν is parameterised
by (ε_ν, μ_ν=(λ, μ_R^ν)).  Plugging in,

    <ν=(ε_ν, μ_ν) | ξ̂^XUV_{m_p^XUV} | Ψ_i> = D^{q^XUV → m_p^XUV}_{ν, μ_ν}(ε_ν)
                                                                       ... (D4d)

where `D^{q → m_p}` is the b-c matrix element in **spherical-tensor**
polarisation (D2d-f).

The c-c part `<ψ⁻_{f, (L, M_R)} | ξ̂^IR | ν=(ε_ν, μ_ν)>` is **exactly** the
m-resolved M_in^{m_p^IR}_{κν}[μ_κ=(L, M_R), μ_ν] (after Cartesian →
spherical polarisation per D2).

Now plug in:

    M(k⃗; ε_i + Ω; R̂_γ)
        =  (1/i) √(2/π)
           Σ_{L M_C M_R}  (−i)^L  Y^C_(LM_C)(k̂)  U(L, M_R, M_C)
           Σ_{μ_ν}
           lim_{ε→0+} ∫ dε_ν
              M_in^{m_p^IR}_{κν}[(L, M_R), μ_ν]
              ·  D^{m_p^XUV}_{ν, μ_ν}(ε_ν)
              / (ε_i + Ω − ε_ν + iε)                                    ... (D4e)

**This is the m-resolved replacement for BW17 Eq. 11.**  Compare to
BW17 Eq. 11: their `T_{Lλl}(k; ε_κ)` is replaced by

    T̃_{(L, M_R), μ_ν}(κ)
        ≡  lim_{ε→0+} ∫ dε_ν
              M_in^{m_p^IR}_{κν}[(L, M_R), μ_ν]
              ·  D^{m_p^XUV}_{ν, μ_ν}(ε_ν)
              / (ε_i + Ω − ε_ν + iε)                                    ... (D4f)

which is **m-resolved**: it has two channel indices, not three l-only.

---

## (D5) Lab → molecular frame for the polarisation

The lab-frame polarisation direction transforms to molecular-frame via
the Wigner D matrix:

    ξ̂^LAB_{m_p}  =  Σ_{m_p^mol} D^(1)_{m_p^mol, m_p}(R̂_γ) · ξ̂^MOL_{m_p^mol}
                                                                       ... (D5)

So for each matrix element above we substitute the molecular-frame
spherical-tensor component (which we already have via D2d-f) and the
Wigner-D factor.

For LINEARLY polarised light along the lab Z-axis (the experimentally
relevant case), m_p^lab = 0 and the rotation just picks out one column:
`m_p^lab=0` → `Σ_{m_p^mol} D^(1)_{m_p^mol, 0}(R̂_γ) · ξ̂^MOL_{m_p^mol}`.

For circularly polarised XUV / IR (also handled by BW17), m_p^lab = ±1.

---

## (D6) Final form for M(k⃗; ε_i + Ω; R̂_γ), substituted

Plug (D5) into (D4e):

    M(k⃗; ε_i + Ω; R̂_γ)
        = (1/i) √(2/π)
          Σ_{L M_C M_R}  (−i)^L  Y^C_(LM_C)(k̂)  U(L, M_R, M_C)
          Σ_{μ_ν}
          Σ_{m_p^IR_mol, m_p^XUV_mol}
              D^(1)_{m_p^IR_mol, m_p^IR_lab}(R̂_γ)
            · D^(1)_{m_p^XUV_mol, m_p^XUV_lab}(R̂_γ)
            · T̃_{(L, M_R), μ_ν}^{m_p^IR_mol, m_p^XUV_mol}(κ)                ... (D6)

with

    T̃_{(L, M_R), μ_ν}^{m_p^IR_mol, m_p^XUV_mol}(κ)
        =  lim_{ε→0+} ∫ dε_ν
              M_in^{m_p^IR_mol}_{κν}[(L, M_R), μ_ν]
              ·  D^{m_p^XUV_mol}_{ν, μ_ν}(ε_ν)
              / (ε_i + Ω − ε_ν + iε)                                       ... (D6a)

This is the FULL m-resolved amplitude.  Compared to BW17 Eq. 11 we have:
- explicit m_R sum (no single-center collapse);
- complex-Y_lm Y^C_(LM_C)(k̂) and Wigner D's (BW17 standard);
- the U(L, M_R, M_C) matrices are the real ↔ complex bridge.

---

## (D7) Finite-pulse (Gaussian) closed form via the Dawson function

We **replace** the monochromatic / Sokhotski-Plemelj formulation with the
finite-pulse closed form of 2nd-order perturbation theory derived in

> S. Azizi, U. Saalmann, J. M. Rost, *Zero-energy photoelectric effect*,
> arXiv:2407.16270 (2024), Eq. 7.

Setting:

* The combined XUV + IR field is taken as a single Gaussian pulse of FWHM
  duration `T_FWHM = T · √(2 ln 2)` and carrier frequency ω (the IR
  frequency for RABBITT; the XUV photon enters via the detuning Δ_k^η).
* In RABBITT-like geometry the XUV pulse is much shorter than the IR
  pulse, so the convolution is dominated by the IR pulse spectrum and
  the single-pulse formula with T = T_IR captures the physics.  For
  ultrashort IR (T_IR · ω ≲ 1) the formula needs the two-pulse
  generalisation; we will warn the user if T · ω falls below 5.
* The on-shell pole at `E_k = E_intermediate` is now SMOOTH — handled
  analytically by the Dawson function evaluated at a finite detuning
  scaled by T.

The amplitude for sideband energy E (relative to the bound binding
energy E_EA = −ε_i > 0, i.e. E_final_kinetic = E):

    a(E) = (π/8) T² A_0²
           Σ_k  d_{Ek}^{(IR)}  d_{k,EA}^{(XUV)}
           Σ_{η = ±}  K_T^η(E, E_k, E_EA, ω)                    ... (D7a)

with the per-state kernel

    K_T^η(E, E_k, E_EA, ω)
        =  exp{ −[(Δ_k^η + E_EA)² + (E + E_EA)²] T² / 4 }
           −  (2i / √π) ·
              exp{ −(E + E_EA)² T² / 8 } ·
              F( [2 Δ_k^η + E_EA − E] · T / √8 )                ... (D7b)

with

    Δ_k^η = E_k − η · ω                                          ... (D7c)
    F(x)  =  e^{−x²}  ∫_0^x e^{u²}  du     (Dawson function)    ... (D7d)

and the matrix elements (BW17 → our notation):

    d_{Ek}^{(IR)}      ≡  ⟨ E | r·ε̂_IR | k ⟩                    ... (D7e)
                       =  M_in^{m_p^IR}_{κν}[μ_κ, μ_ν]    in our HDF5
                            (c-c element, in-state basis, see D3-D4)

    d_{k,EA}^{(XUV)}   ≡  ⟨ k | r·ε̂_XUV | EA ⟩                  ... (D7f)
                       =  D^{m_p^XUV}_{ν, μ_ν}            in our HDF5
                            (b-c element, in-state basis,
                             D_raw or D_ortho — choose ortho for
                             accuracy, see Phase A layout)

The Σ_k sum spans all intermediate states.  For our **anion case** there
are no discrete bound intermediate states (the only bound state of the
M⁰+e Hamiltonian is the SOMO itself, which is the initial state and is
excluded — see README §Phase B).  Hence Σ_k reduces to the discretised
continuum at our scan points (ε_ν, μ_ν):

    Σ_k  →  Σ_{ν ∈ scan} · Σ_{μ_ν}  · w_ν                       ... (D7g)

where `w_ν = dε_ν / (k_ν)` is the energy-measure weight on the uniform-
in-k grid (E_ν = k_ν²/2 → dε_ν = k_ν · dk).  For the m-resolved
generalisation (Path B), `μ_ν` runs over all real-Y_lm channels and the
sum is FULL.

### Numerical advantages

* **No principal-value, no pole subtraction.** The Dawson function is
  smooth and finite everywhere.  GSL provides `gsl_sf_dawson`; scipy
  provides `scipy.special.dawsn` (same definition).
* **Discrete sum** — straightforward parallel reduction.
* **T appears as a tunable physical parameter**: τ-vs-T sweep validates
  the long-pulse limit (T → ∞ recovers BW17 / Sokhotski-Plemelj).

### Mapping to (D6) m-resolved structure

The amplitude in (D6) factorised as
`M(k⃗; ε_i + Ω; R̂_γ) = Σ angular_terms × T̃_{(L, M_R), μ_ν}^{m_p}(κ)`
with T̃ defined by the Sokhotski-Plemelj integral (D6a).  Under (D7) we
simply replace

    T̃_{(L, M_R), μ_ν}^{m_p^IR, m_p^XUV}(κ)
        =  Σ_{ν ∈ scan} M_in^{m_p^IR}_{κν}[(L, M_R), μ_ν]
                          · D^{m_p^XUV}_{ν, μ_ν}
                          · Σ_η K_T^η(E_κ, ε_ν, E_EA, ω) · w_ν
                                                                ... (D7h)

— same indices, same sum structure, just a different kernel `K` per
state.  This means the m-resolved code from C.3 onwards is UNCHANGED
in shape; only the kernel function changes.

### What's NEW vs the BW17 plan

| BW17 plan (old)                    | Azizi-Saalmann-Rost plan (new) |
|------------------------------------|--------------------------------|
| Principal-value + iπ residue       | Closed-form per intermediate state |
| Sokhotski-Plemelj pole subtraction | Dawson function evaluated at finite x |
| Long-pulse / monochromatic only    | Finite-pulse T + reduces to monochromatic as T → ∞ |
| T not a parameter                  | T_FWHM is an input — physically meaningful |
| Quadrature errors near pole        | None — Dawson is analytic |
| Compare against Dahlström Coulomb  | Compare against TDSE (the validation in Azizi 2024) |

### Validation strategy for (D7)

Once `compute_T.py` (D7h) is coded, two checks before we trust τ:

1. **Long-pulse limit (T → ∞).** As T grows, the Dawson piece becomes
   the principal value of `1/(2Δ_k^η + E_EA − E)` and the Gaussian
   becomes a delta function in detuning.  τ should converge to the
   monochromatic BW17 result.  Plot τ_2hν vs T.  Plateau = converged.
2. **Single-channel atomic limit (H⁻ → s).** Use a synthetic spherical
   target and check the τ_2hν matches the published Azizi 2024 Fig. 3
   curves to plotting accuracy.  Cross-check with their TDSE numbers in
   Table I (total ionisation probability).

---

## (D8) τ_2hν, τ_cc, τ_mol via (D6)

Once M(k⃗; ε_i + Ω; R̂_γ) is in hand at the two sideband energies (2q−1)ω
and (2q+1)ω (final kinetic energy at the sideband):

    τ_2hν(2q, k̂, R̂_γ) = (1/2ω) · arg [ M^(2q−1)*  ·  M^(2q+1) ]            ... (D8)

For the universal continuum-continuum delay τ_cc(2q), BW17 Eq. 24
applies UNCHANGED in our basis (it's a one-channel atomic-Coulomb
asymptotic quantity; for our anion case with no Coulomb tail it
reduces to the centrifugal-only correction, which is small).

The molecular delay τ_mol = τ_2hν − τ_cc (BW17 Eq. 25).

---

## (D9) Orientation/angle averages

BW17 Eqs. 26-29.  These are integrals over R̂_γ (Euler angles) and k̂
(emission direction) of M*·M-weighted quantities.  In our framework the
Wigner-D rotation acts on M(k̂; R̂_γ) as written in (D6), and the integrals
are done numerically on a (Euler) × (k̂) grid.

---

## Concrete checks before any τ output is trusted

1. **Gauge consistency**: τ_2hν from length and velocity must agree at
   converged Hamiltonian.  Today we only have length-gauge c-c; the gauge
   check needs c-c velocity (Phase A TODO).
2. **Atomic limit**: feed a synthetic spherical target (or shift to a
   one-channel approximation) and check τ_cc matches the closed-form
   Dahlström-Pazourek formula (BW17 Eq. 14).
3. **Free-particle limit**: turn off the molecular potential (or use a
   small radius); τ_2hν should reduce to τ_cc + (Wigner from the b-c step).
4. **Numerical convergence**: refine ε_ν grid spacing in (D7) and check
   T̃ values stabilise.
5. **Cross-pair conjugation** of M_in: already verified, machine eps.
6. **Hermiticity at κ=ν**: already verified, machine eps.

---

## Implementation plan from this derivation

1. `complex_Y_transform.py`  — build the block-diagonal U matrix; apply to
   any (N_psi, N_psi) array.  Validates against `U_real_to_complex` in
   C++ Gaunt code for at least one (l, mC, mR).
2. `m_p_polarization.py`  — combine Cartesian M_in^x, M_in^y, M_in^z into
   m_p ∈ {-1, 0, +1} per (D2).
3. `dawson_kernel.py`  — evaluate K_T^η(E, E_k, E_EA, ω) per (D7b)
   using `scipy.special.dawsn` (or `gsl_sf_dawson` from a C extension if
   we ever need it for performance).  Test against scipy at a few points
   AND the asymptotic F(x) → 1/(2x) for x→∞.  This is the ONE function
   whose correctness all of τ depends on.
4. `compute_T.py`  — evaluate T̃_{(L, M_R), μ_ν}^{m_p^IR, m_p^XUV}(κ) per (D7h)
   — single discrete sum over ν, no quadrature subtraction.
5. `compute_M.py`  — assemble M(k⃗; ε_i + Ω; R̂_γ) per (D6).
6. `tau_2hv.py`, `tau_cc.py`, `tau_mol.py` — per Eqs. 21, 24, 25.
7. `tau_avg.py`  — angle/orientation averages per Eqs. 26-29.
8. **Validation (Phase D)**: τ-vs-T sweep (long-pulse limit), and the
   atomic single-channel benchmark against Azizi 2024 Fig. 3 / Table I.

Each step has a documented test before the next runs.

---

## Open questions for the user before I proceed

1. **Wigner-D convention.**  BW17 doesn't pin down the active-vs-passive
   convention.  Active rotation matrix `D^j_{m', m}(α, β, γ)` rotates a
   ket; passive rotates the axes.  Plan: use the **active** convention
   from Varshalovich §4.3 (most common in atomic physics), `D^j_{m', m}
   = e^{-iα m'} d^j_{m', m}(β) e^{-iγ m}`.  Need to confirm.

2. **Real-Y_lm phase.**  Our gaunt_real values are computed via the
   `U_real_to_complex` transformation from complex Y.  Confirmed signs
   match Condon-Shortley.  This drives signs of M^q via D2d-f.

3. **Initial state Ψ_i partial-wave decomposition.**  In our framework
   Ψ_i = anion SOMO is multi-channel.  The b-c element D^{q}_{ν, μ_ν}
   from Phase A already includes the SOMO angular structure.  We do NOT
   need to re-decompose; it's already absorbed.

4. **(D9) orientation average grid.**  Standard practice is 30 × 30 × 30
   Lebedev or Euler grid for (α, β, γ).  Want to discuss the trade
   between accuracy and runtime later.
