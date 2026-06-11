#!/usr/bin/env python3
"""
Psi4 script: H2- (H2 + e-) in THF via IEFPCM, B3LYP/aug-cc-pVDZ.

Same recipe as version_0/psi4_Qchem/solvant/c8f8m_solvent.py but stripped
down to H2.  Produces a Molden file that can be fed directly to
preprocessing/build/preprocess_molden to generate h2_anion.orbitals.h5
and h2_anion.potentials.h5 (the inputs the scattering pipeline expects).

Run on a machine with Psi4 installed:
    psi4 h2_anion_thf.psi4.py

Outputs:
    h2_anion_thf.molden     (alpha + beta MOs, occupied + virtual)

Anion-in-solvent bound state.  Without PCM (vacuum) H2- is metastable
(autodetaching shape resonance), so we use IEFPCM(THF) to stabilise the
SOMO into a true bound state, exactly as for H2O-.
"""
import psi4
import numpy as np

# --- resources ---
psi4.set_memory("8 GB")
psi4.set_num_threads(4)

# --- molecule: H2^- (anion), charge -1, multiplicity 2 ---
# Bond length = 1.398 bohr (matches preprocessing/reference_data/h2_ccpvdz_sph.molden)
mol = psi4.geometry("""
-1 2
units bohr
symmetry c1
nocom
noreorient
H   0.0   0.0   0.0
H   0.0   0.0   1.3983973328
""")

psi4.set_options({
    "basis": "aug-cc-pVDZ",
    "scf_type": "df",
    "reference": "uks",
    "e_convergence": 1e-9,
    "d_convergence": 1e-9,
    "maxiter": 300,
    "puream": False,
    "pcm": True,
    "pcm_scf_type": "total",
})

pcm_input = """
Medium {
    SolverType = IEFPCM
    Solvent    = Tetrahydrofurane
}
Cavity {
    Type = GePol
    RadiiSet = UFF
    Scaling = True
    Mode = Implicit
}
"""
psi4.pcm_helper(pcm_input)

E, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
print(f"B3LYP/aug-cc-pVDZ PCM(THF)  E(H2-) = {E:.12f} Ha")

psi4.molden(wfn, "h2_anion_thf.molden")
print("Wrote h2_anion_thf.molden")

# Diagnostic: print SOMO energy
eps_a = np.asarray(wfn.epsilon_a())
nocc_a = wfn.nalpha()
print(f"  alpha occupied energies (au): {eps_a[:nocc_a].tolist()}")
print(f"  SOMO (highest occupied alpha): {eps_a[nocc_a-1]:.6f} au "
      f"= {eps_a[nocc_a-1]*27.2114:.3f} eV")
