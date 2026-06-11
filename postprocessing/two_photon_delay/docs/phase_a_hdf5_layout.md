# Phase A consolidated HDF5 — layout reference

This file is produced by
[`python/phase_a_assembler.py`](../python/phase_a_assembler.py) and is the
ONLY input Phase C should need.  Conventions are atomic units throughout,
real spherical harmonics with q-map `x → +1, y → −1, z → 0`, and the
u/r convention `χ(r) = r · F(r)`.

## How to produce

```bash
# Step 1 — c-c radial dipole HDF5 (C++ driver).
build/cc_dipole_driver \
    --psi_dir_prefix $SCRATCH/psi_<molhash>_ik \
    --ik_kappa 50,60,70,80 \
    --ik_nu    50,60,70,80 \
    --pol all \
    --out      $WORK/cc_dipole.h5

# Step 2 — consolidate into the Phase A file.
python3 postprocessing/two_photon_delay/python/phase_a_assembler.py \
    --cc-h5    $WORK/cc_dipole.h5 \
    --scan-dir $WORK/dipole_<molhash>_<scan_id> \
    --out      $WORK/two_photon_me_<scan_id>.h5
```

## Layout

```
/                                  (HDF5 root, attrs only)
├── attr E_HOMO         (float)    ε_i = initial-state (anion SOMO) energy [au]
├── attr N_grid         (int)      number of radial points
├── attr dr, r_min      (float)    radial grid step + start [au]
├── attr l_cont         (int)      continuum angular cutoff L_max
├── attr N_psi          (int)      (l_cont + 1)² = # channels per energy
├── attr n_occ          (int)      # closed-shell target occupied orbitals
├── attr dk             (float)    momentum-grid step [au]
├── attr ik_min, ik_max (int)      kgrid range; k(ik) = ik · dk
├── attr n_ik           (int)      number of per-ik groups stored
├── attr molecule_name, git_hash, iso_date_utc, scan_id
├── attr cc_h5_*                   cross-reference info from cc_dipole.h5
│
├── /channels/
│   ├── l_mu  (N_psi,)             l of channel μ = l² + l + m  (real-Y_lm)
│   └── m_mu  (N_psi,)             m of channel μ
│
├── /per_ik/ik<NNNN>/              ONE group per scan energy
│   ├── attr ik, k, E_kin, omega   k = ik·dk, E_kin = k²/2, ω = E_kin − E_HOMO
│   ├── attr fit_residual_rel      AsymptoticAmplitudes fit residual
│   ├── attr K_symmetry_err        K-matrix antisymmetry (should be small)
│   ├── A           (N_psi, N_psi)
│   ├── B           (N_psi, N_psi)  ψ_β(r→∞) ~ A·S(kr) + B·C(kr)
│   ├── b_overlap   (N_psi, n_occ) ⟨φ_α | ψ_β⟩ on the kept window
│   │
│   │   For each (gauge ∈ {len, vel}) × (pol ∈ {x, y, z}):
│   ├── d_raw_<gauge>_<pol>         (N_psi,)    REAL bound-continuum dipole
│   │                                            in BACK-PROP basis
│   │                                            (= d^q_β in DipoleMatrixElement)
│   ├── D_raw_<gauge>_<pol>_re      (N_psi,)    in-state-basis matrix element,
│   ├── D_raw_<gauge>_<pol>_im      (N_psi,)      M_μ = ((A−iB)⁻¹)†·d_raw,
│   │                                            NO orthogonalisation correction
│   ├── D_ortho_<gauge>_<pol>_re    (N_psi,)    with Lagrange-multiplier
│   ├── D_ortho_<gauge>_<pol>_im    (N_psi,)      orthogonalisation against
│   │                                            the n_occ bystander orbitals
│   └── d_correction_<gauge>_<pol>  (n_occ,)    bound part of the ortho corr.
│
└── /pairs/pair_k<KKKK>_n<NNNN>/  ONE group per c-c (κ, ν) pair
    ├── attr ik_kappa, ik_nu       
    ├── attr E_kappa, E_nu          E_κ = ε_i + Ω = final, E_ν = intermediate
    │
    │   For each pol ∈ {x, y, z}:
    └── cc_raw_len_<pol>  (N_psi, N_psi)  c-c radial dipole, BACK-PROP basis,
                                          LENGTH gauge.  cc_raw[β, α] =
                                          ⟨ψ_β^(κ) | r·ε̂_q | ψ_α^(ν)⟩ with the
                                          angular factor A^q_{μν} applied.
                                          REAL by construction.
```

## Phase C uses

For each scan energy ε_κ:

1. **One-photon (b-c) phase**: get D_raw / D_ortho per (gauge, pol) at ik = κ from
   `/per_ik/ik<NNNN>/D_ortho_len_<pol>_{re,im}`.

2. **Continuum-continuum (c-c) radial dipole** for the IR step: get
   cc_raw_len_<pol> at every (κ, ν) pair from `/pairs/...`, then rotate
   from back-prop basis to in-state basis using
   `M_in = ((A_κ − iB_κ)⁻¹)†  ·  cc_raw  ·  (A_ν − iB_ν)⁻¹`
   where (A, B) come from `/per_ik/`.

3. **T_{L,λ,l}(k; ε_κ)** (BW17 Eq. 12, continuum part only -- bound part
   is zero for anion photodetachment, see README §"Phase B"):

       T_{L,λ,l}(k; ε_κ) = lim_{ε→0+} ∫dε_ν
                            ⟨R_{kL}|r|R_{ε_νλ}⟩ · ⟨R_{ε_νλ}|r|R_{nl}⟩
                            / (ε_i + Ω − ε_ν + iε)

   where the radial integrals are obtained from the in-state-rotated cc_raw
   (numerator left factor) and D_raw (numerator right factor), respectively.

4. **M(k⃗; ε_i + Ω; R̂_γ)** (BW17 Eq. 11): assembly of T with Gaunt
   integrals and Wigner-D rotations.

5. **τ_2hν** (Eq. 21), **τ_cc** (Eq. 24), **τ_mol** (Eq. 25),
   orientation/angle averages (Eqs. 26-29).

## Conventions to keep clear

* `E_kappa` = ε_κ = ε_i + Ω = energy of the *final* continuum state for the
  XUV-only one-photon step.  Equivalent to the BW17 "ε_κ".  In our data
  this is E_kin at the ik in question; the relation `ε_κ = E_kin` holds
  because the **kinetic** energy of the final photoelectron is `k²/2` and
  the saved scattering scan parameterises kinetic energy.
* `E_nu` = ε_ν = ε_i + Ω = intermediate-state energy.  For the
  *on-shell* (κ = ν) pair this equals ε_κ.
* `omega` in `/per_ik/`: photon energy of the one-photon step,
  ω = E_kin − E_HOMO.  For two-photon delays you also have the IR ω,
  which lives at the Phase C call site, not in the Phase A HDF5.

## Limitations / TODO

* **Velocity gauge for c-c is not yet computed**.  `d_raw_vel_*` and
  `D_*_vel_*` come from the existing scattering pipeline (one-photon b-c
  velocity gauge), so the b-c side has both gauges.  The c-c side has
  length only -- adding velocity is needed for the gauge-consistency
  check (Phase D).
* Each (κ, ν) cc_raw matrix is `N_psi² × 8 B ≈ 850 GB` at the production
  scan size (L = 100, N_psi = 10201) -- the HDF5 will become huge.  At
  H₂O test scale (N_psi = 121) one matrix is ~ 117 kB, trivial.  For
  the LRZ run we will need to write the file with HDF5 chunked +
  compressed datasets and document expected disk usage.
