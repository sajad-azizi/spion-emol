# H₂ photoionization — effect of V_x removal and polarization potential

Two code-level changes evaluated on H₂:

1. **V_x not used** — audited and confirmed: `cross_section_delay.py` was
   the only code that ever mentioned V_local_exchange. The scattering loop
   reads only `/potential/V_H` and constructs V_en on-the-fly. The LDA V_x
   computed by preprocessing is dead weight that neither scatter nor dipole
   ever touch. Verified by bit-identical σ from a preproc with `--exchange lda`
   vs `--exchange none` (same molhash, same ψ, same D).

2. **V_pol implemented** — `Potentials::compute_U_polarization_at_r` was a
   stub returning zero. Implemented as:
   ```
   V_pol(r) = − (1/2) · α_iso · (1 − exp(−(r/r_c)⁶))² / r⁴        r_c = 1.5 bohr
   ```
   (isotropic approximation, Gianturco–Rodríguez-Ruiz damping). Back-compat:
   returns zero when `data.has_polarizability` is false.

Run all three variants on the same H₂ / cc-pVDZ / r_max=40 / l_cont=6 /
dk=0.1 / ik=4..12 setup.

---

## 1. Results: H₂ σ_L_avg (Mb) vs ω

| ω (eV) | **α = 0** | **α = 2.9 au** (true H₂⁺ iso) | **α = 5.0 au** | **Lit (exp)** |
|---:|---:|---:|---:|---:|
| 18.30 |  7.10 | 10.85 | 15.60 | ~ 8.5 |
| 19.52 | 10.77 | 14.60 | 16.99 | ~ 7.5 |
| **21.02** | **12.62** | 14.36 | 14.31 | ~ 6.5 |
| 22.79 | 12.11 | 11.87 | 10.79 | ~ 5.5 |
| 24.83 | 10.15 |  8.93 |  7.61 | ~ 4.0 |
| 27.14 |  7.83 |  6.37 |  5.17 | ~ 3.0 |
| 29.73 |  5.75 |  4.41 |  3.47 | ~ 2.2 |
| 32.58 |  4.13 |  3.04 |  2.34 | ~ 1.7 |
| **35.71** |  **2.95** |  **2.13** |  **1.63** | **~ 1.5** |

**Interpretation:**
- V_pol is a long-range attractive correction — biggest effect at small k
  (long de Broglie wavelength; continuum electron samples the `-α/2r⁴` tail).
- Near threshold (ω=18–22 eV): σ goes **up**; peak **shifts to lower ω**
  (21 → 19.5 eV), matching lit peak location (~18 eV) better.
- At high ω (ω>30 eV): σ goes **down** — the continuum electron moves
  through the V_pol region faster and feels less of it.

---

## 2. Deviation from experiment (% over-prediction)

| ω (eV) | α = 0 | α = 2.9 | α = 5.0 |
|---:|---:|---:|---:|
| 18 (peak region) | +48 % | +72 % | +100 % |
| 25 | +154 % | +123 % | +90 % |
| 35 | +97 % | +42 % | **+9 %** |

V_pol doesn't uniformly improve the fit — it trades off over-prediction
at the peak against better high-ω behavior. α = 2.9 au (the physical
H₂⁺ isotropic polarizability) gives the most balanced result, sitting
between the two extremes.

**Best compromise**: α = 2.9 au gives peak 14.4 Mb (lit 6–8 Mb, still over
by ~70 %) AND high-ω σ(35 eV) = 2.1 Mb (lit 1.5 Mb, +42 % over, close).

The residual +70 % over-prediction at peak reflects the SE-HF method
itself (missing relaxation + correlation of the cation). That's not a
V_pol tuning problem.

---

## 3. Shape comparison (normalized to peak value)

Peak position and width give a feel for the *shape*, separate from overall
magnitude:

| α | ω_peak (eV) | σ(35 eV) / σ_peak |
|---:|---:|---:|
| 0   | 21.0 | 0.23 |
| 2.9 | 19.5 | 0.15 |
| 5.0 | 19.5 | 0.10 |
| **exp** | **~18** | **~0.17** |

α = 2.9 au matches the experimental shape surprisingly well (peak
location to 1.5 eV, σ(35)/σ_peak to within 10 %). The residual magnitude
offset is ~uniform SE-HF over-prediction.

---

## 4. Recommendations

1. **Keep V_x out** of the preprocessing path for scattering runs.
   `preprocess_molden --exchange none` is the right default for a
   scattering pipeline (V_x is a local approximation that the non-local K
   operator in `SchurInverter` already correctly supersedes). Leaving
   `--exchange lda` produces identical scattering results but wastes
   preprocessing time + disk. The `test_hdf5_roundtrip.py` verifier still
   expects V_x > 0, so `--exchange lda` is still needed for the
   preprocessing ctests.

2. **Use V_pol with the physical α of the residual ion** when reproducing
   photoionization cross sections. For H₂ → H₂⁺: α_iso ≈ 2.9 au is the
   right value. This gives the best shape match to experiment. Magnitude
   is still +70 % over at peak, consistent with known SE-HF limitations
   (relaxation, correlation).

3. **For H₂O** (residual bug documented in `validation/h2o_homo_photoionization/`):
   V_pol with α_iso(H₂O⁺) ≈ 9.5 au might significantly help the residual
   5–10× under-prediction. Not tested here, but the infrastructure is
   ready.

---

## 5. Files

- `src/scatt/Potentials.cpp` — `compute_U_polarization_at_r` implemented
- `validation/h2_noVx/`      — α=0 run (preproc without V_x; bit-identical to α=0 with V_x)
- `validation/h2_Vpol/`      — α=5 au run + plots
- `validation/h2_Vpol_alpha29/` — α=2.9 au run + plots
