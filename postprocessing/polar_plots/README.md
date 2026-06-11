# Molecular-frame polar plots

Two-step tool that produces a single 2 × 3 figure for a given gauge:

|  | σ(E, θ) | τ(E, θ) | σ·τ(E, θ) |
|---|---|---|---|
| **Row A — fixed (α, β)** (defaults α=β=0)        | ✓ | ✓ | ✓ |
| **Row B — orientation-averaged** (Eq. S38)       | ✓ | ✓ | ✓ |

All panels are polar plots:  **radius = energy** (E_kin, ω, or k — selectable),
**polar angle = θ** (electron emission polar angle in the molecular frame),
**φ = 0** (the in-plane half-disc; mirrored to a full disc by left/right
symmetry — disable with `--no-mirror` for fully asymmetric targets).

## Formulas (Azizi et al. supplementary, used verbatim)

### Fixed Euler angles (α, β)

The dipole operator in the molecular frame (Eq. S26-S27):

```
µ̂  =  cos α sin β · x̂  +  sin α sin β · ŷ  +  cos β · ẑ.
```

Since the dipole channels are linear in the field components, the
effective channel amplitudes at fixed (α, β) are

```
d^{eff}_{ℓm}(E)  =  cos α sin β · d^x_{ℓm}(E)
                 +  sin α sin β · d^y_{ℓm}(E)
                 +        cos β · d^z_{ℓm}(E)
```

and the angular dipole amplitude is

```
D^{(-)}(θ; α, β; E)  =  (4π/k) · Σ_{ℓ m}  i^{-ℓ}  d^{eff}_{ℓm}(E)
                                                 · Y^R_{ℓm}(θ, φ=0)         [Eq. S23]
```

Cross section and time delay (Eqs. S22, S25):

```
σ(θ; α, β; E)  =  (4 π² ω / c) · |D^{(-)}|²
τ(θ; α, β; E)  =  Im[D^{(-)*} ∂_E D^{(-)}] / |D^{(-)}|²
```

### Orientation (polarisation) average (Eq. S38)

Define D_q(θ; E) = (4π/k) Σ_{ℓm} i^{-ℓ} d^q_{ℓm}(E) Y^R_{ℓm}(θ, 0).  Then

```
σ_pol(θ; E)  =  (4 π² ω / c) · (1/3) · Σ_q |D_q(θ; E)|²
τ_pol(θ; E)  =  Σ_q  Im[ D_q*  ∂_E D_q ]  /  Σ_q  |D_q|²
```

### Note on φ = 0

At φ = 0 the real spherical harmonics with m < 0 vanish identically
(`Y^R_{ℓ,m<0}(θ, 0) ∝ sin(|m|·0) = 0`).  Only m ≥ 0 channels contribute.

## Numerics

Real-Y_{ℓm}(θ, 0) is built from a numerically-stable
normalised-Legendre 3-term recurrence (Press et al., NR §6.7) — no
factorials computed explicitly.  Verified stable to ℓ = 1000.  This is
the same recurrence used in `../cross_section_delay.py` (the `scipy.special.sph_harm`
path overflows the factorial normaliser at high (ℓ, |m|) on older
scipy and gives NaN at ℓ ≳ 70).

## Usage

```bash
# Step 1 — compute σ, τ, σ·τ data on (E, θ) grid
python3 postprocessing/polar_plots/compute_polar_data.py \
    --input-dir  <gathered_dipole_dir>                 \
    --output-dir <run_dir>/polar                       \
    --gauge len                                        \
    --alpha-deg 0   --beta-deg 0                       \
    --n-theta 181

# (writes two .dat files:
#    polar_fixed_alpha0_beta0_len.dat
#    polar_pol_avg_len.dat                                                )

# Step 2a — render the full 2×3 figure (σ | τ | σ·τ, fixed | avg)
python3 postprocessing/polar_plots/plot_polar.py \
    --fixed  <run_dir>/polar/polar_fixed_alpha0_beta0_len.dat \
    --avg    <run_dir>/polar/polar_pol_avg_len.dat            \
    --output <run_dir>/polar/polar_panels.png                 \
    --radius-mode Ekin_eV

# Step 2b — render the standalone σ·τ-only side-by-side figure
python3 postprocessing/polar_plots/plot_sigma_tau_only.py \
    --fixed  <run_dir>/polar/polar_fixed_alpha0_beta0_len.dat \
    --avg    <run_dir>/polar/polar_pol_avg_len.dat            \
    --output <run_dir>/polar/sigma_tau_only.png               \
    --e-max  300
```

Useful flags for `plot_polar.py`:

| flag | default | meaning |
|---|---|---|
| `--radius-mode` | `Ekin_eV` | `Ekin_eV` / `omega_eV` / `k_au` |
| `--no-mirror`   | off       | don't mirror θ ∈ [0,π] to [0,2π] |
| `--sigma-cmap`  | viridis   | colormap for σ panels |
| `--tau-cmap`    | bwr       | colormap for τ panels |
| `--sigtau-cmap` | seismic   | colormap for σ·τ panels |
| `--sigma-vmin / --sigma-vmax` | auto | σ colourbar range (Mb) |
| `--tau-vlim`    | auto      | symmetric τ colourbar limit (as) |
| `--sigtau-vlim` | auto      | symmetric σ·τ colourbar outer extent (Mb·as) — 99th percentile of \|σ·τ\| |
| `--sigtau-norm` | `asinh`   | `asinh` (smooth signed-log, no white band) or `symlog` (linear-near-zero + log-outside) |
| `--sigtau-lo-pct` | `1.0`   | percentile of \|σ·τ\| used as the near-zero transition threshold. Bottom 1% maps to the near-zero band; remaining 99% gets log-color resolution. |
| `--sigtau-linthresh` | auto | explicit near-zero half-width (Mb·as); overrides `--sigtau-lo-pct` |
| `--sigtau-linscale` | `0.5` | (`symlog` only) fraction of the colormap budget reserved for the linear near-zero band |

### Why the σ·τ panel uses `AsinhNorm` by default

σ·τ for a typical run has ~10 decades of dynamic range (e.g. the
spherical-model test: |σ·τ| 1st-percentile = 1.6×10⁻⁶ Mb·as,
99th-percentile = 1.4×10⁴ Mb·as). With `SymLogNorm` the linear-near-zero
band swallows everything below `linthresh`, and the natural choices of
`linthresh` (median, or a fraction of `vlim`) put most of the disc into
the near-white linear region. `AsinhNorm` (matplotlib ≥ 3.6) has no
linear band: values map smoothly through zero via `arcsinh(x/linthresh)`,
so structure at every magnitude is resolved at the same time. The
`--sigtau-lo-pct` default of 1.0 picks `linthresh` = 1st percentile of
|σ·τ| — only the bottom 1% of the data sits in the near-zero region.

If you need to suppress a known noise floor by mapping it explicitly to
white, use `--sigtau-norm symlog` and pick `--sigtau-lo-pct` (or
`--sigtau-linthresh`) at the noise level.

The standalone σ·τ figure (`plot_sigma_tau_only.py`) uses the same
controls, just without the `--sigtau-` prefix: `--norm`, `--lo-pct`,
`--linthresh`, `--linscale`.

## Validation

The implementation has been cross-checked against the existing reference
`generate_data_fig2_correct.py` (which produces the same σ, τ for a
single Cartesian polarization):

| Test case | Reference | σ rel err | τ abs err |
|---|---|---|---|
| α = β = 0  | pure z-pol | **1.1 × 10⁻¹⁴** | **1.0 × 10⁻¹² as** |
| α = 0, β = 90° | pure x-pol | **4.6 × 10⁻¹⁴** | **1.0 × 10⁻¹⁰ as** |

Both reduce to machine precision against the existing reference — the
Cartesian-combination math is correct.

## Output file format

Both `.dat` files share the same 7-column layout:

```
1 k          [a.u.]
2 E_kin      [a.u.]
3 omega      [a.u.]
4 theta      [rad]
5 sigma      [bohr²]
6 tau        [as]
7 sigma·tau  [bohr²·as]
```

The fixed-(α, β) file's header records the Cartesian projection
coefficients `(c_x, c_y, c_z) = (cos α sin β, sin α sin β, cos β)` so the
Euler angles are recoverable from the saved data alone.
