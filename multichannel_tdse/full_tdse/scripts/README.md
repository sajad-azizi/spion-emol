# LRZ runscripts for the full-TDSE driver

The recipe operating point is heavy enough that production runs go to
LRZ.  These scripts wrap the `run_tdse` binary for SLURM submission
and a recipe-style convergence sweep.

## Files

| file | purpose |
|---|---|
| `run_tdse_lrz.sh` | SLURM batch script for one `run_tdse` invocation |
| `sweep_lrz.sh`    | submits a series of jobs varying L, dr, E_cut^(-5), Δt |
| `check_convergence.py` | postprocess the sweep outputs, report 1% convergence |

## One-off run

```bash
sbatch scripts/run_tdse_lrz.sh \
    --L 500000 --dr 0.5 \
    --E_cut_open_kHz 300 \
    --E_cut_m5_kHz 20000000 \
    --tau_us 30 --T_us 90 --dt_us 0.01 \
    --delta_E_kHz 1.0 \
    --out_prefix $WORK/tdse/run_ref
```

This produces, in `$WORK/tdse/`:
- `run_ref_summary.txt` -- knob values, build/prop wallclock, block totals
- `run_ref_block_M{-5,-4,-3,-2}_dPdE.csv` -- dP/dE per block (E_kHz, dP/dE per kHz)

## Convergence sweep

```bash
OUT_BASE=$WORK/mc_tdse_conv bash scripts/sweep_lrz.sh
# wait for jobs to finish (squeue -u $USER)
python3 scripts/check_convergence.py $WORK/mc_tdse_conv
```

Submits four sweeps following the recipe convergence checklist:
1. **E_cut^(-5) ramp** (recipe item 6): {5, 10, 20, 50} GHz above the
   M_F=-5 threshold.  This is the heavy one; recipe says start at 20 GHz
   and continue to 50 GHz if observables are still moving.
2. **Box length L** (recipe item 7): {2.5e5, 5e5, 1e6} a₀.
3. **Open-block cutoff** (recipe item 5): E_cut^open in {200, 300, 400} kHz.
4. **Time step Δt** (recipe item 4): {0.02, 0.01, 0.005} μs.

`check_convergence.py` reports relative changes between consecutive
runs of each sweep; recipe stop-criterion is **1% on P_ZEPE^(-4),
P_aa^(-2), E_peak^(-4), P_ea/P_ae**.

## Knobs reference

All passed as CLI flags to `run_tdse`; see `src/run_tdse.cpp` for defaults.

| flag | recipe value | purpose |
|---|---|---|
| `--L`              | 5e5  a₀     | radial box length |
| `--dr`             | 0.5  a₀     | radial grid spacing |
| `--E_cut_open_kHz` | 300         | E_cut above each open-block threshold |
| `--E_cut_m5_kHz`   | 20×10⁶      | E_cut above M_F=-5 threshold (kHz) |
| `--omega_kHz`      | 80.896      | RF detuning (= 8·E_b at recipe params) |
| `--Omega_R_kHz`    | 179         | Rabi frequency (recipe operating point) |
| `--tau_us`         | 30          | Gaussian pulse 1/e half-width |
| `--T_us`           | 90          | propagation half-window |
| `--dt_us`          | 0.01        | time step |
| `--delta_E_kHz`    | 1.0         | dP/dE Gaussian smoothing width |
| `--out_prefix`     | (required)  | absolute path prefix for output files |

## Local test (small basis, fast)

For a sub-minute sanity run on a workstation:

```bash
./build/run_tdse \
    --L 50000 --dr 0.5 \
    --E_cut_open_kHz 100 --E_cut_m5_kHz 200 \
    --tau_us 30 --T_us 60 --dt_us 0.02 \
    --delta_E_kHz 1.0 \
    --out_prefix /tmp/test_run
```

Block totals and dP/dE files land in `/tmp/test_run_*`.

## Step-down ramp (avoid the M_F=-5 trap)

The recipe-quoted reference cell `--L 5e5 --E_cut_m5_kHz 2e7` produces
~109k M_F=-5 states at ~16 MB each → **~1.6 TB** of state storage. That
does NOT fit on a single LRZ node (largest is ~1 TB).  `run_tdse`
prints an upfront level-count + storage estimate and a WARNING when the
budget exceeds 256 GB; abort and step down rather than wait for the
build to finish (or OOM).

Recipe items 6-7 (convergence ramp) translate to this concrete schedule
for a SINGLE node — start small, double until observables converge:

| step | --L     | --E_cut_m5_kHz | M_F=-5 state count | state storage | when |
|------|---------|----------------|---------------------|----------------|------|
| 1    | 100000  | 1,000,000      | ~770               | ~12 GB         | smoke; first end-to-end |
| 2    | 200000  | 1,000,000      | ~1,540             | ~25 GB         | check P_-5 < 5% |
| 3    | 200000  | 5,000,000      | ~3,440             | ~55 GB         | first cutoff bump |
| 4    | 500000  | 5,000,000      | ~8,610             | ~135 GB        | recipe L; 5 GHz cap |
| 5    | 500000  | 20,000,000     | ~17,200            | ~270 GB        | needs HiMem node (`mem_huge` 1.5 TB) |

Stop ramping when `P_ZEPE^(-4)`, `P_aa^(-2)`, `E_peak^(-4)`, and
`P_ea/P_ae` change by < 1% between consecutive steps (recipe item 6).

**Open-block cutoff E_cut_open_kHz** scales much more cheaply than
E_cut_m5: ~430 levels per open block at L=5e5 / E_cut=300 kHz, so the
open blocks are not the bottleneck.  Hold them at recipe values
throughout the ramp.
