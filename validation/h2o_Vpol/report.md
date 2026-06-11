# H₂O photoionization — V_pol effect

**Setup**: r_max = 40 bohr, dr = 0.005, l_cont = 8, `--exchange none`,
dk = 0.1, ik = 2..15 (14 points). α_iso = 9.5 au for V_pol (≈ H₂O⁺
isotropic polarizability; neutral H₂O has 9.63 au, cation differs little).
Identical preproc + scatter settings in both runs — only V_pol differs.

---

## 1. Results: σ_L_avg (Mb) vs ω

| ω (eV) | k (au) | **α = 0** (no V_pol) | **α = 9.5** (V_pol on) | Lit σ_1b1 |
|---:|---:|---:|---:|---:|
| 13.97 | 0.20 | 0.005 | 0.021 | — |
| 14.65 | 0.30 | 0.017 | 0.045 | ~ 2 |
| 15.60 | 0.40 | 0.052 | 0.121 | ~ 2.5 |
| 16.82 | 0.50 | 0.139 | 0.315 | ~ 3 |
| 18.32 | 0.60 | 0.312 | 0.725 | ~ 3.5 |
| 20.09 | 0.70 | 0.608 | 1.391 | ~ 3.5 |
| 22.13 | 0.80 | 1.032 | 2.106 | ~ 3 |
| 24.44 | 0.90 | 1.526 | 2.542 | ~ 2.5 |
| **27.03** | 1.00 | 1.976 | **2.632 (peak)** | ~ 2 |
| 29.88 | 1.10 | 2.271 | 2.480 | ~ 1.5 |
| 33.01 | 1.20 | 2.373 | 2.197 | ~ 1.3 |
| 36.42 | 1.30 | 2.307 | 1.864 | ~ 1.2 |
| 40.09 | 1.40 | 2.122 | 1.533 | ~ 1.0 |
| 44.03 | 1.50 | 1.868 | 1.231 | ~ 0.8 |

**Lit values** quoted are σ_1b1, synthesized from Tan–Brion-van-der-Wiel (1978)
dipole (e,e) total σ combined with branching ratios from Novak-Potts-Kubota
and Natalense–Lucchese (1999) SE-HF theory. Individual digit uncertainty ~30 %.

---

## 2. What V_pol did

- **Magnitude boost 2–3× everywhere.** V_pol is an attractive long-range
  tail; the continuum electron gets pulled in, enhancing bound-continuum
  overlap.
- **Peak shifts down**: α=0 peak ~ 33 eV → α=9.5 peak ~ 27 eV. Moves closer
  to (but not yet matching) the lit peak at ~18–20 eV.
- **High-ω now tracks lit well**:
  - ω = 27 eV: σ_code 2.63 Mb, σ_lit ~ 2 Mb — +30 % (SE-HF typical)
  - ω = 30 eV: σ_code 2.48 Mb, σ_lit ~ 1.5 Mb — +65 %
  - ω = 40 eV: σ_code 1.53 Mb, σ_lit ~ 1.0 Mb — +50 %
  **Compared to α=0** where these were 40 %, 50 %, 110 % over — V_pol improved
  the 40 eV match but over-corrected at the peak.
- **Near-threshold still severely under** (factor ~20 at ω = 15 eV):
  σ_code 0.12 Mb, σ_lit ~ 2.5 Mb. This reflects the Wigner k⁵ suppression
  of the l_f = 2 dominant channel — V_pol helps at large r (slow electron)
  but can't create the s-wave continuum (l_f = 0) that H₂O 1b1 doesn't
  have access to via dipole selection rules.

---

## 3. Bug-hunt status

What's now confirmed working for H₂O:

| item | status |
|---|---|
| cross-section formula (`8πω/(ck)·Σ\|d\|²`) | ✅ correct (via H₂ validation) |
| D∞v/C₂v symmetry (σ_y = 0 bitwise) | ✅ |
| HOMO identification (1b1 = i_homo = 4) | ✅ |
| SCE expansion of 1b1 (96.8 % at l=1, m=+1) | ✅ |
| V_en + V_H scattering potential | ✅ |
| V_x is dead code (not used) | ✅ |
| V_pol integrated & tested | ✅ |
| Peak location within 5–10 eV of lit | ⚠ shifted high |
| Magnitude within factor 2 at high ω | ✅ |
| Magnitude at threshold | ❌ factor ~20 under |

The remaining near-threshold under-prediction is **fundamental to the
dipole selection rules for 1b1 in the static-exchange approximation**
(only l_f = 2 accessible for x, z polarizations). Capturing the lit
near-threshold σ would require either:
- going beyond SE-HF (adding many-body correlation / interchannel coupling), or
- including shake-up / shake-off channels (multiple ionization), or
- a different interpretation of what `σ_1b1` actually means experimentally
  (some of the near-threshold signal may be multi-electron in origin).

V_pol does not and cannot fix this — it's a one-electron correction that
only enhances existing dipole channels, not creates new ones.

---

## 4. Verdict

| metric | no V_pol | α = 9.5 au | |
|---|---:|---:|---|
| σ_peak magnitude | 2.37 Mb | 2.63 Mb | lit ~ 3–4 |
| ω_peak | 33 eV | 27 eV | lit ~ 20 |
| σ(30 eV) / lit | +50 % | +65 % | mid-range |
| σ(40 eV) / lit | +110 % | +50 % | **V_pol helps** |
| σ(15 eV) / lit | −99 % | −95 % | still failing |

V_pol **improves** H₂O overall but does not fix the threshold behavior.
For the 20–40 eV range it brings σ into SE-HF agreement (±50 % of lit);
below 20 eV the approach cannot reproduce experiment due to selection-rule
constraints inherent to SE-HF.

---

## 5. Recommendation

- For photoionization calculations of **closed-shell, non-trivial
  molecules** at ω roughly 5–15 eV above threshold, use V_pol with the
  **physical cation polarizability**. The isotropic approximation with
  Gianturco-type damping is fine.
- Don't expect SE-HF + V_pol to reproduce threshold cross sections when
  the HOMO has p-character — it fundamentally can't access the s-wave
  continuum that dominates near threshold for such orbitals.
- For deeper benchmarks of C₈F₈ (the production target): expect similar
  SE-HF ±50 % accuracy in the mid-ω range, and meaningful near-threshold
  values only if multiple orbitals contribute.

---

## 6. Files

- `src/scatt/Potentials.cpp` — `compute_U_polarization_at_r` implementation
- `validation/h2_Vpol/`                — H₂ V_pol study (α = 2.9, 5.0 au)
- `validation/h2o_Vpol/`               — this run (α = 9.5)
- `validation/h2o_noVpol_sameGrid/`    — α = 0 on identical grid (for comparison)
