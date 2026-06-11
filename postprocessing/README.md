# Post-processing

Two scripts that consume the output of the scattering driver and produce
cross sections and Wigner time delays.

```
scattering (C++)                   gather_dipoles.py                cross_section_delay.py
  $WORK/dipole_<hash>_<scan>/   ->   $WORK/gathered_<scan>/     ->   delay_xsec_*.dat
      manifest.h5                      dipole_len_homo_{x,y,z}_{mu}.dat   + 3 plots
      ik0001.h5 ... ikNNNN.h5          dipole_vel_homo_{x,y,z}_{mu}.dat
      __SUCCESS__
```

## gather_dipoles.py

Reads the HDF5 scan dir, writes one `.dat` per (gauge × polarization × channel).
Output format (kept unchanged so existing analysis flows still work):

```
# Columns: k_elec[au]  E_kin[au]  omega[au]  Re(D)  Im(D)  |D|^2  arg(D)
```

By default uses the orthogonalized dipole `D_ortho`. Pass `--raw` to use
`D_raw` (for diagnostics only — raw has the cross-section-corrupting overlap
with the bound orbital). Defaults write into `$WORK/gathered_<scan_basename>/`.

## cross_section_delay.py

Reads the gathered `.dat` files and writes:

- `delay_xsec_len_homo.dat` / `delay_xsec_vel_homo.dat` — σ(E), τ(E), σ_raw per polarization
- `gauge_diagnostics.dat` — Q_σ and Δφ_rms (both should tend to their ideal values
  for an exact Hamiltonian; deviation measures SEHF inexactness)
- `cross_section_delay_both.png` — 2×2 panel, σ and τ per gauge
- `gauge_diagnostics.png` — Q_σ(E) and Δφ_rms(E)
- `gauge_overlay.png` — gauge comparison for orientation-averaged σ and τ

## Conventions (enforced)

- Atomic units throughout.
- `omega = E_kin + Ip = E_kin - E_HOMO` (Koopmans), stored in each per-ik file.
- Dipole is energy-normalized in the C++ output already; no extra √(2/(πk)) factor here.
- Real Y_{lm}, q-map: x→+1, y→−1, z→0.
- Orientation-averaged τ is σ-weighted (physically correct for unpolarized light).

## Usage

```bash
# End-to-end (assuming $WORK is set)
$ scattering <preproc.h5> 1 200 0.01            # produces $WORK/dipole_<hash>_ik1-200_.../
$ python postprocessing/gather_dipoles.py dipole_<hash>_ik1-200_...
$ python postprocessing/cross_section_delay.py gathered_dipole_<hash>_ik1-200_...
```

Both scripts auto-resolve relative paths against `$WORK` if not found in `$PWD`.
