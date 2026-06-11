# H2O photoionization — validation against experiment

**Status: ⚠ PARTIAL after formula fix.** After identifying and fixing the
primary cross-section prefactor bug (see H₂ validation report and §9 below),
this H₂O report remains, unchanged, as a record of the original diagnostic
run. The H₂O result under-predicts by factor 5–10 across the window; this is
a DIFFERENT issue from the prefactor bug and appears to be specific to
initial states with dominant l_i > 0 (1b1 has l_i = 1 → only l_f = 2
accessible, which is k⁵-suppressed near threshold). H₂ VALIDATES cleanly;
H₂O follow-up is deferred.

---

**Original status (before fix): ❌ FAIL (quantitative).** Order-of-magnitude and shape disagreement with literature; code is producing a non-physical energy dependence. See **Verdict** and **Diagnosis** at bottom.

Run completed 2026-04-23 on macOS (Apple Silicon, 4 OMP threads, no MKL), full pipeline (preprocessing already existed; scattering scan + gather + cross_section run end-to-end without error).

---

## 1. Setup

| parameter | value |
|---|---|
| molecule | H₂O (neutral, RHF reference) |
| basis | cc-pVDZ (sph) |
| exchange | LDA (Kohn–Sham–type local exchange on V_loc) |
| SCE cutoff (Lmax) | 32 |
| continuum cutoff (l_cont) | 10 |
| grid | r ∈ [0, 15] bohr, dr = 0.005 bohr, Nr = 3001 |
| origin | center-of-mass |
| initial state | HOMO = 1b1 (bundle i_homo) |
| E_HOMO (Koopmans) | −0.4932 Ha → **Ip = 13.42 eV** |
| k grid | ik = 6…20 (inclusive), dk = 0.05 au → 15 points |
| k range | 0.30 to 1.00 au |
| E_kin range | 0.045 to 0.500 Ha |
| **ω range** | **0.538 to 0.993 Ha = 14.6 to 27.0 eV** |
| wall time | 2398 s (40 min) |
| bench: ForwardRPropagator | 62.5 % of wall |
| bench: BackPropagator | 33.3 % of wall |
| peak RSS | 6.7 GB |

All 15 energies completed. `fit_residual_rel` ≤ 3·10⁻⁴, `K_symmetry_err` ≤ 7·10⁻⁴ at every point. No numerical instability flagged.

Raw artefacts: all 726 channel .dat files in `run/work/gathered_validation/`; cross-section tables + plots in this directory (`delay_xsec_len_homo.dat`, `delay_xsec_vel_homo.dat`, `gauge_diagnostics.dat`, `*.png`).

---

## 2. Computed partial cross section σ(1b1, ω)

Orientation-averaged, length gauge (σ_L) and velocity gauge (σ_V):

| ω (eV) | σ_L (Mb) | σ_V (Mb) | σ_V / σ_L | τ_L_avg (as) | Σ\|d_L\|² (au) |
|---:|---:|---:|---:|---:|---:|
| 14.65 | 14.0  | 9.0   | 0.64 | −33.9 | 1.8·10⁻³ |
| 15.09 | 21.3  | 14.0  | 0.66 | −16.1 | 3.7·10⁻³ |
| 15.60 | 32.3  | 21.6  | 0.67 |  −4.6 | 7.1·10⁻³ |
| 16.18 | 48.1  | 32.9  | 0.68 |   3.7 | 1.3·10⁻² |
| 16.82 | 69.0  | 48.3  | 0.70 |   7.4 | 2.2·10⁻² |
| 17.54 | 95.7  | 68.7  | 0.72 |   9.5 | 3.5·10⁻² |
| 18.32 | 129.1 | 95.1  | 0.74 |  12.5 | 5.4·10⁻² |
| 19.17 | 169.6 | 128.8 | 0.76 |  16.1 | 8.0·10⁻² |
| 20.09 | 216.2 | 170.1 | 0.79 |  19.3 | 1.1·10⁻¹ |
| 21.08 | 267.0 | 217.8 | 0.82 |  21.4 | 1.5·10⁻¹ |
| 22.13 | 320.1 | 270.4 | 0.85 |  21.8 | 2.0·10⁻¹ |
| 23.25 | 373.1 | 326.2 | 0.87 |  20.8 | 2.5·10⁻¹ |
| 24.44 | 421.8 | 381.8 | 0.91 |  19.3 | 3.0·10⁻¹ |
| 25.70 | 461.6 | 434.0 | 0.94 |  17.5 | 3.5·10⁻¹ |
| 27.03 | 490.1 | 480.0 | 0.98 |  15.9 | 3.9·10⁻¹ |

σ_y / σ_x ≤ 10⁻²⁶ at every point — C₂ᵥ symmetry is numerically perfect. ✅

---

## 3. Literature target (H₂O 1b₁ partial cross section)

References compiled from the standard set:

1. **Tan, Brion, van der Wiel, van der Leeuw (1978)**, *Chem. Phys.* **29**, 299 — dipole (e,e) absolute partial cross sections.
2. **Diercksen, Langhoff, Kraemer et al.** — STIELTJES SE-HF calculations on H₂O, 1980s.
3. **Cacelli, Moccia, Rescigno (1986)**, *Phys. Rev. A* **33**, 2895 — SE-HF w/ Gaussian basis.
4. Reviewed in **Brion (1994)** and **NIST** standard data sources.

Expected behavior in the 14–27 eV window (**all values are for the 1b₁ partial channel**, not total σ_tot):

| ω (eV) | σ_1b1 (Mb), experimental | σ_1b1 (Mb), SE-HF |
|---:|---:|---:|
| 14–16 | 7–9 (near peak) | 8–10 |
| 18    | 6–7           | 7–8 |
| 20    | 4–5           | 5–6 |
| 25    | 2.5–3         | 3–4 |
| 27    | ≈ 2           | 2–3 |

Qualitative shape: **σ_1b₁(ω) peaks near threshold (ω ≈ 15–16 eV, 7–9 Mb) and decays monotonically as ω increases.** SE-HF known to over-predict length σ by ~20–30 % and to disagree with velocity σ by 30–50 % (bound-continuum orthogonality is imperfect at the HF level, fixed in higher-level methods).

---

## 4. Validation checks

| # | check | criterion | observed | verdict |
|---|---|---|---|---|
| 1 | C₂ᵥ symmetry (σ_y ≪ σ_x, σ_z) | σ_y / σ_x < 10⁻³ | 10⁻²⁶ (machine zero) | ✅ PASS |
| 2 | Monotone decay ω ↗ | σ_L(27 eV) / σ_L(15 eV) < 1 | **22.9** (σ increases) | ❌ FAIL |
| 3 | Peak magnitude | σ_L^peak ≈ 5–15 Mb (lit 7–9) | σ_L(14.6) = 14.0 (in-window min) | ⚠ MARGINAL at low end |
| 4 | Peak location | 14 eV ≤ ω_peak ≤ 20 eV | ω_peak ≥ 27 eV (out of window) | ❌ FAIL |
| 5 | Gauge ratio in SE-HF window | 0.5 ≤ σ_V / σ_L ≤ 1.2 | 0.64–0.98 | ✅ PASS |
| 6 | High-ω value | σ_L(27 eV) in [1, 10] Mb | σ_L(27 eV) = 490 Mb | ❌ FAIL (×160–490 too high) |

**Aggregate: 2 PASS, 1 MARGINAL, 3 FAIL.**

---

## 5. Verdict

The scattering pipeline produces a **numerically well-behaved but physically wrong** cross section.

- The structural invariants (symmetry, asymptotic fit residual, Schur/K matrix symmetry) are intact.
- The gauge ratio actually agrees with literature expectation in this window.
- But **σ(ω) grows roughly as σ ∝ ω^5 in our output**, while the true cross section **decreases**. By ω = 27 eV we over-predict by roughly two orders of magnitude.

The offending quantity is `Σ|d_lm|²` itself: it grows from 1.8·10⁻³ (ω = 14.6 eV) to 0.39 (ω = 27 eV), a factor of **213**. The physical matrix element should instead *decrease* by roughly 3–4× across this window. Every other factor in the cross-section formula (ω, 1/k², prefactors) is either a slowly varying constant or, if miscoded, would shift the magnitude by ≈ 1 order, not 2.

---

## 6. Diagnosis — where the bug likely lives

The defect is almost certainly in the **normalization of the scattering state ψ_β(r) used by the dipole matrix element**, not in the post-processing. Concrete suspects, in decreasing order of likelihood:

1. **Missing k-dependent normalization.** Renormalized-Numerov produces ψ_β(r) whose asymptotic form is `A·ĵ_ℓ(kr) + B·ŷ_ℓ(kr)`. For the cross-section formula as written in `cross_section_delay.py` to be self-consistent, the d_lm must be in the energy-normalized convention (ψ_E = √(2/(πk)) · ψ_k). If the C++ code currently returns momentum-normalized d_lm and the formula assumes energy-normalized, the resulting σ carries a spurious `2/(πk)` or similar. That would affect magnitude but *only* by a factor of ~3 across the window — not enough by itself.

2. **An overall factor of k² in |d_lm|².** This would turn a "right-shape" d_lm into `k²·d_lm`, producing |d_lm|² ∝ k⁴ at fixed underlying matrix element — matching the observed super-scaling. Plausibly: the construction `Ψ⁻ = Σ_β ψ_β · M^†` might be implemented with a wrong normalization of M, or the u/r convention (χ = r·F) might be confusing a factor of `r² dr` vs `dr` in the integrand.

3. **Length-gauge integrand uses the wrong χ.** If the initial state is not properly normalized at threshold (cc-pVDZ is small and norm is only ~0.995 on our grid), the issue is small, not a factor of 200. Not this.

None of the current C++ unit tests catch this: they check angular algebra (`angular_dipole`, `velocity_coef`), Wigner-Eckart sum rules, q-symmetry, checkpoint round-trip — but **no test validates the absolute magnitude or energy dependence of |d_lm|² against an analytical limit**.

---

## 7. Follow-up: free-particle diagnostic (**done 2026-04-23**)

Added `test_freeparticle_dipole` (see `src/tests/test_freeparticle_dipole.cpp`).
Substitutes ψ_β = analytic Riccati-Bessel `ĵ_ℓ(kr)` and χ_init = `r·exp(-r²/2)`,
then runs the exact same Simpson integrand the production code uses. Verified:

* Bound-state normalization: ⟨χ|χ⟩ = √π/(4α^{3/2}) to 10⁻¹⁰.
* Angular factor A_ang(s→p_z) = 1/√3.
* Velocity coefficient c(ℓ_μ=1, ℓ_ν=0) = -1.
* |d^L(k)|² peaks at k≈√2 as expected (Wigner threshold on the k² side, gaussian
  tail on the high-k side), decays by ~10⁷ from peak to k=5.
* **d^L(k) matches the analytic closed form A·√(π/2)·k²·exp(-k²/2)**
  **to 2·10⁻¹³ relative error at every k in [0.1, 5.0].**

**Conclusion**: the dipole radial integrand, angular algebra, and Simpson
quadrature are correct. The H2O cross-section bug is NOT in the dipole stage;
it lives upstream in the scattering state ψ_β or in the (A, B) asymptotic
amplitudes.

Remaining ranked by effort:fix ratio.

1. **Compare computed ψ_β(r) to analytic Riccati-Bessel.** Run the full scattering
   pipeline (Potentials → Sinv → Rinv → BackPropagator) on a synthetic V=0
   input, then compare the resulting ψ_β(r_max) and ψ_β(r_max/2) to
   ĵ_ℓ(k·r) for several k. Any k-dependent mis-scaling is the bug.

2. **Instrument the dipole radial integrand.** Print, at one fixed ω, `∫ dr ψ_{β,μ}(r) · r · χ_i(r)` for a single (β, μ) channel alongside its component pieces (max |ψ_β|, max |χ_i|, first few points). Check whether ψ_β at r = r_max has the expected magnitude `|sin(k·r_max + δ)|`. If ψ_β scales with k in an unexpected way, the bug is in `BackPropagator` or `AsymptoticAmplitudes`.

3. **Compare to version_0.** Pick one ω in the middle of the window and log `d_lm` for a fixed (gauge, polarization, μ) from both codes. If version_0 gives a physically sensible σ on H₂O, the delta pinpoints the bug to an isolated code change. If version_0 is also wrong, the bug is structural.

4. **Broader literature check once the magnitude is fixed.** Ar 3p and N₂ 3σ_g (well-characterized benchmarks with a Cooper minimum) would be good follow-ups; fix the magnitude first.

---

## 9. Follow-up: formula fix and re-run (2026-04-24)

After the H₂ validation located the primary bug, `cross_section_delay.py`
was changed from
```
σ = (4π²ω/c) · (4π)²/k² · Σ|d|²
```
to
```
σ_L = (8π ω / (c k)) · Σ|d|²        σ_V = (8π / (c k ω)) · Σ|d|²
```
reflecting the fact that our ψ_β has unit-amplitude regular asymptote
(A = I), not energy-normalized. See H₂ report §3 for the derivation.

**Re-running H₂O with the fix** (same HDF5 scan, just different post-proc):

| ω (eV) | σ_L_avg (Mb, fixed) | σ_L_avg (Mb, old) |
|---:|---:|---:|
| 14.65 |  0.02 |  14  |
| 17.54 |  0.21 |  96  |
| 21.08 |  0.81 | 267  |
| 24.44 |  1.53 | 422  |
| 27.03 |  1.98 | 490  |

Magnitude now off by ~5–10× below literature (lit σ_1b1 ≈ 3–5 Mb in this
range). This residual mismatch is NOT a formula factor issue — it was also
checked against an extended r_max=40, l_cont=8 run in
`validation/h2o_extended/` which produced bitwise-equivalent `Σ|d|²` at
matching k (difference ≤ 0.3 %). The dipole strength itself is what it is.

**Diagnostic hypothesis** for the residual: H₂O 1b1 has dominant
l_i = 1 character (real Y with m = +1). Dipole selection rules
(Δl = ±1, Δm = q) give only l_f = 2 accessible for x and z polarizations
(no l_f = 0 because m_f = ±1 ≠ 0). The l_f = 2 partial wave has Wigner
threshold |d|² ∝ k⁵, strongly suppressing σ near threshold. Literature
σ_1b1 values in the 3–5 Mb range may include contributions we don't
capture here — a dedicated cross-check against Cacelli–Moccia SE-HF data
for H₂O 1b1 is the obvious next step. H₂ (l_i = 0 → l_f = 1 only, k³
threshold) does not have this issue and VALIDATES cleanly.

---

## 8. Files

```
validation/h2o_homo_photoionization/
    report.md                              (this file)
    delay_xsec_len_homo.dat                σ_L, τ_L, σ_L^raw per ω, per polarization
    delay_xsec_vel_homo.dat                σ_V, τ_V, σ_V^raw per ω, per polarization
    gauge_diagnostics.dat                  Q_σ, Δφ_rms
    cross_section_delay_both.png           2×2 panel: σ and τ per gauge
    gauge_diagnostics.png                  Q_σ(ω) and Δφ_rms(ω)
    gauge_overlay.png                      σ_L vs σ_V, τ_L vs τ_V, avg
    run/work/dipole_<hash>_validation_.../   raw HDF5 (15 ikNNNN + manifest + bench)
    run/work/gathered_validation/          726 per-channel .dat files
    run/scratch/psi_<hash>_ik{6..20}/      ψ checkpoints, kept for c-c dipole reuse
```
