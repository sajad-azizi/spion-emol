# Two-photon RABBITT delay for C₈F₈⁻ — LRZ recipe

## Default pulse parameters (community-standard RABBITT)

| Quantity | Default | Source |
|---|---|---|
| IR carrier ω_IR | **1.55 eV** (800 nm) | Ti:Sapphire fundamental; standard in RABBITT |
| IR FWHM T_L | **30 fs** | ~11 IR cycles; the convention across Klünder PRL 2011, Isinger Science 2017, Mauritsson PRL 2010, Sabbar PRA 2015, Cattaneo Nature 2018, Vos Science 2018, Heuser PRA 2016 |
| XUV single-burst FWHM T_X | **0.30 fs** | 300 as APT burst, characteristic of H15-H23 from 800 nm HHG (Isinger 2017: 150 as; Mauritsson/Sabbar: 300 as) |
| τ delay | scanned in experiment; we evaluate at τ=0 | The molecular phase φ is τ-independent at long T_L; experiment fits cos[2ωτ + φ] over a τ scan |
| Lab polarization | linear z, both XUV + IR (m_p=0) | All cited RABBITT experiments use linearly-polarized XUV and IR collinear |



End-to-end pipeline from a Psi4 anion molden to the final
`cross_section_delay_both.png` with the 3rd row showing
τ_2ℏω(E_κ) and `|⟨M_<* M_>⟩|`.

The recipe assumes:

- Repo cloned to `$SCRATCH_HOME/static_exchangeHF` (or wherever your
  build/ lives — the recipe uses `$REPO`).
- Psi4 conda env exists for the anion SCF (used by step **1**).
- Scattering/cc_dipole binaries are built (`./build/scattering`,
  `./build/cc_dipole_driver`, `preprocessing/build/preprocess_molden`).
- `$WORK` (persistent) and `$SCRATCH` (fast) are set.

If anything below is unclear, the H₂⁻ run in [h2_test/](../h2_test/) is
a fully-worked smaller example that uses exactly these commands.

---

## 0. environment

```bash
export REPO=$SCRATCH_HOME/static_exchangeHF        # adjust
cd $REPO

# point WORK/SCRATCH at the paths you want artefacts to land in:
export WORK=/hppfs/work/pr28fa/$USER/static_exHF/c8f8_data
export SCRATCH=/hppfs/scratch/08/$USER/static_exHF/c8f8_data
mkdir -p $WORK $SCRATCH

# psi4 conda env for step 1 only (loadable on LRZ login nodes):
# module load anaconda                # if your site needs this
# conda activate psi4
```

---

## 1. anion SCF (Psi4) — produces the molden

Run Psi4 once for the C₈F₈⁻ anion in THF via IEFPCM, B3LYP/aug-cc-pVTZ.
The recipe and a working H₂⁻ analogue are in
[`preprocessing/reference_data/h2_anion_thf.psi4.py`](../../preprocessing/reference_data/h2_anion_thf.psi4.py).
For C₈F₈⁻ the equivalent driver lives at
[`version_0/psi4_Qchem/solvant/c8f8m_solvent.py`](../../version_0/psi4_Qchem/solvant/c8f8m_solvent.py).

```bash
# from a node with Psi4 installed:
mkdir -p $WORK/c8f8_b3lyp && cd $WORK/c8f8_b3lyp
psi4 $REPO/version_0/psi4_Qchem/solvant/c8f8m_solvent.py
# produces: c8f8_anion_thf.molden
```

**SLURM (LRZ Phase 2):** the SCF is cheap (~hour single-node).  An
example single-node `srun` is fine; no MPI needed.

---

## 2. preprocess (SCE projection + V_en/V_H) — produces orbitals/potentials HDF5

```bash
cd $REPO
$REPO/preprocessing/build/preprocess_molden \
    $REPO/preprocessing/reference_data/c8f8_ccpvdz_sph.molden \
    $WORK/c8f8_anion.orbitals.h5 \
    --initial-state-molden $WORK/c8f8_b3lyp/c8f8_anion_thf.molden \
    --lmax 80   --dr 0.01   --rmax 100 \
    --exchange none --orbitals occupied
# produces:
#   $WORK/c8f8_anion.orbitals.h5
#   $WORK/c8f8_anion.potentials.h5
```

Recommended settings for C₈F₈ taken from CLAUDE.md §"Sizing rules":

- `--lmax 80` for production (l_cont up to 100 if needed; CLAUDE.md
  warns that lmax=300 is overkill and breaks V_H at high l).
- `--dr 0.01 --rmax 100` matches the existing C₈F₈ scan dimensions.

**Cost:** ~5-30 min on a single node (no MPI).

---

## 3. scattering scan — produces ψ checkpoints + per-ik dipoles

This is the big compute.  For a RABBITT-relevant photoelectron energy
range of 3-8 eV, use `ik=30..80, dk=0.01` (k = 0.30..0.80 au;
ε = 0.045..0.32 au; 1.2-8.7 eV).

```bash
# SLURM job script (mpi.slurm) — multi-node MPI + GPU offload:
sbatch $REPO/job.mpi.slurm \
    --partition=gpu --time=12:00:00 --nodes=2 \
    -- \
    $REPO/build/scattering $WORK/c8f8_anion.orbitals.h5 \
       30 80 0.01 \
       --lmax-cont 80 \
       --work $WORK --scratch $SCRATCH \
       --scan-id rabbitt_30to80 \
       --use-gpu  --memory-budget 200000000000
```

Refer to `run_c8f8_lrz.sh` in the repo root for the LRZ-tuned launch.

**Outputs:**
- `$WORK/dipole_<hash>_rabbitt_30to80/` with `ik0030.h5`..`ik0080.h5`,
  `manifest.h5`, `__SUCCESS__`
- `$SCRATCH/psi_<hash>_ik0030/`..`psi_<hash>_ik0080/` (kept; needed for
  step 4)

**Cost:** ~hours per ik at l_cont=80 (CLAUDE.md §"GPU build" tells you
how much speed-up `--use-gpu` gives).  Memory-budget the I/O accordingly.

---

## 4. cc_dipole_driver — radial c-c dipole matrices for chosen (κ, ν) pairs

For a RABBITT sweep at sidebands κ ∈ {40, 50, 60, 70, 80} with ν
covering the on-shell windows for both paths (ν_± = ε_κ ∓ ω):

```bash
HASH=$(ls $SCRATCH | grep '^psi_' | head -1 | sed 's/psi_//;s/_ik.*//')
PREFIX="$SCRATCH/psi_${HASH}_ik"

# all ν values present in the scan:
NU_LIST=$(seq -s , 30 80)

$REPO/build/cc_dipole_driver \
    --psi_dir_prefix "$PREFIX" \
    --ik_kappa 40,50,60,70,80 \
    --ik_nu   "$NU_LIST" \
    --pol all \
    --out $WORK/cc_dipole_c8f8.h5
```

**Cost:** for C₈F₈⁻ l_cont=80, each pair is ~50 MB and ~30 s with the
current `try_load_into_memory` path; 5×51 = 255 pairs ≈ 2-3 hours.
Single-node, OpenMP-only (no MPI).

**Outputs:**
- `$WORK/cc_dipole_c8f8.h5`

---

## 5. phase_a_assembler — consolidate into a single HDF5

```bash
SCAN_DIR=$(ls -d $WORK/dipole_${HASH}_rabbitt_30to80)
python3 $REPO/postprocessing/two_photon_delay/python/phase_a_assembler.py \
    --cc-h5    $WORK/cc_dipole_c8f8.h5 \
    --scan-dir $SCAN_DIR \
    --out      $WORK/two_photon_me_c8f8.h5
```

**Cost:** minutes.  Output ~GB-scale.

---

## 6. two_photon_delay.py — compute τ_2ℏω(E_κ) and write .dat / .png

```bash
# Pick the gathered_dipole_* dir cross_section_delay.py uses
# (built by gather_dipoles.py from the per-ik HDF5s):
GATHER_DIR=$WORK/gathered_dipole_${HASH}_rabbitt_30to80

python3 $REPO/postprocessing/two_photon_delay.py \
    --phase-a    $WORK/two_photon_me_c8f8.h5 \
    --output-dir $GATHER_DIR \
    --sidebands  40,50,60,70,80 \
    --omega-IR-eV 1.55 \
    --T-X-fs 0.30 \
    --T-L-fs 30.0 \
    --tau-delay-fs 0.0 \
    --angle-grid 8,8,8,8,8 \
    --trim-sigma 8.0
```

**Cost:** minutes.  Output:
- `$GATHER_DIR/two_photon_delay.dat`
- `$GATHER_DIR/two_photon_delay.png`

The `.dat` has 9 columns:
`ik_κ  E_κ_au  E_κ_eV  τ_au  τ_as  |⟨M_<* M_>⟩|  arg⟨…⟩  |⟨M_<⟩|  |⟨M_>⟩|`.

**ANGLE GRID GUIDE.**  For production-quality results use at least
`6,6,6,6,6` (3 mins) or `8,8,8,8,8` (~10 min).  Lower grids (`4,4,4,4,4`)
are useful for fast turnaround but may differ from the converged value
by 10-30 as for unoriented molecules — see the validation in
[`docs/`](docs/).

---

## 7. cross_section_delay.py — final plot with 3rd row

```bash
python3 $REPO/postprocessing/cross_section_delay.py \
    $GATHER_DIR \
    --xaxis E \
    --idx-start 2 \
    --with-two-photon
# (--two-photon-dat ... can override the auto-located .dat path)
```

This regenerates `cross_section_delay_both.png` with **three rows**:
1. σ(E) length & velocity
2. τ_W(E) length & velocity
3. τ_2ℏω(E_κ) and |⟨M_<* M_>⟩|

**Cost:** seconds.

---

## TL;DR copy-paste recipe

After step 0 environment setup:

```bash
# 1. Psi4 anion SCF (offline on Psi4 node)
psi4 c8f8m_solvent.py

# 2. preprocess
preprocessing/build/preprocess_molden  c8f8_ccpvdz_sph.molden \
    $WORK/c8f8_anion.orbitals.h5 \
    --initial-state-molden c8f8_anion_thf.molden \
    --lmax 80 --dr 0.01 --rmax 100 --exchange none

# 3. scatter (SLURM)
sbatch job.mpi.slurm  ./build/scattering $WORK/c8f8_anion.orbitals.h5 \
    30 80 0.01 --lmax-cont 80 \
    --work $WORK --scratch $SCRATCH \
    --scan-id rabbitt_30to80 --use-gpu

# 4. cc_dipole (single node)
./build/cc_dipole_driver \
    --psi_dir_prefix $SCRATCH/psi_<hash>_ik \
    --ik_kappa 40,50,60,70,80 --ik_nu $(seq -s , 30 80) --pol all \
    --out $WORK/cc_dipole_c8f8.h5

# 5. consolidate
python3 postprocessing/two_photon_delay/python/phase_a_assembler.py \
    --cc-h5 $WORK/cc_dipole_c8f8.h5 \
    --scan-dir $WORK/dipole_<hash>_rabbitt_30to80 \
    --out $WORK/two_photon_me_c8f8.h5

# 6. compute τ_2ℏω
python3 postprocessing/two_photon_delay.py \
    --phase-a $WORK/two_photon_me_c8f8.h5 \
    --output-dir $WORK/gathered_dipole_<hash>_rabbitt_30to80 \
    --sidebands 40,50,60,70,80 \
    --omega-IR-eV 1.55 --T-X-fs 0.30 --T-L-fs 30.0 \
    --angle-grid 8,8,8,8,8

# 7. final plot
python3 postprocessing/cross_section_delay.py \
    $WORK/gathered_dipole_<hash>_rabbitt_30to80 \
    --xaxis E --with-two-photon
```

---

## Validation knobs

If the τ_2ℏω numbers look surprising, run these convergence checks
*before* attributing them to molecular physics:

- **T_L sweep**: rerun step 6 with `--T-L-fs 1,3,5,10` (separately) and
  check the result plateaus (it should at long T_L; if not, the
  symmetric-trim window is too tight for the local M·D curvature).
- **Angle-grid**: rerun with `8,8,8,8,8` and `10,10,10,10,10`; values
  should agree to <5 as.
- **Identity-cc sanity**: replace `cc_raw` with the identity in your
  Phase A HDF5 (see `test_validation_*.py` for a worked example) and
  rerun step 6.  The result should be ≲ 50 as for any energy (the
  Lindroth-Dahlström expectation for closed-shell-residual anions).
  If you see hundreds of as with identity-cc, the angular/rotation
  machinery has a bug — STOP and re-run the T1-T5 validation suite.
- **Off-resonance value**: pick the κ furthest from any structure and
  report τ_2ℏω there as the "background" two-photon delay.  Compare
  to Lindroth-Dahlström "~tens of as" expectation.
