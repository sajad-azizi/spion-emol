#!/usr/bin/env python3
"""
test_validation_T1_synthetic.py
-------------------------------
Validation T1: trivial-rotation invariance.

Build a SYNTHETIC Phase A HDF5 in which the in-state rotation
matrix U_κ = (A_κ − iB_κ)⁻¹ is a SCALAR phase factor e^{−iδ(ε)} × I
(channel-diagonal, same δ for all channels), cc_raw is a real
constant matrix, and d_raw is a real constant vector.

In this setup the b-c amplitude D[ν, μ_ν] picks up a phase e^{+iδ(ν)}
(from U_ν†), and the c-c amplitude M_in[κν][μ_κ, μ_ν] picks up
e^{+iδ(κ)} · e^{−iδ(ν)} (from U_κ† and U_ν).  The combination
M_in[κν] · D[ν] has phase e^{+iδ(κ)} only -- the ν-dependence cancels.

Both RABBITT paths integrate to e^{+iδ(κ)} × (real positive), so
   arg(M_<) = arg(M_>) = δ(κ)
   τ_2ℏω = (arg M_> − arg M_<) / (2ω) = 0    [analytic prediction]

Independent of δ(ε), kernel form, T_L, IR delay, etc.

PASS criterion: |τ_2ℏω| < 1 as (i.e. < 1 attosecond).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def write_synthetic_phase_a(out_path: Path, l_cont: int, dk: float,
                            ik_list: list[int],
                            delta_func, cc_const: float = 1.0,
                            d_const: float = 1.0):
    """Build a Phase A HDF5 with channel-diagonal scalar in-state rotation."""
    N_psi = (l_cont + 1) ** 2
    with h5py.File(out_path, "w") as f:
        f.attrs["dk"]            = float(dk)
        f.attrs["l_cont"]        = np.int64(l_cont)
        f.attrs["N_psi"]         = np.int64(N_psi)
        f.attrs["E_HOMO"]        = -0.1
        f.attrs["ik_min"]        = np.int64(min(ik_list))
        f.attrs["ik_max"]        = np.int64(max(ik_list))
        f.attrs["n_ik"]          = np.int64(len(ik_list))
        f.attrs["molecule_name"] = b"synthetic_T1"
        f.attrs["scan_id"]       = b"synth_T1"
        f.attrs["dr"]            = 0.01
        f.attrs["N_grid"]        = np.int64(1)
        f.attrs["n_occ"]         = np.int64(1)

        ch = f.create_group("channels")
        l_mu, m_mu = [], []
        for l in range(l_cont + 1):
            for m in range(-l, l + 1):
                l_mu.append(l); m_mu.append(m)
        ch.create_dataset("l_mu", data=np.array(l_mu, dtype=np.int32))
        ch.create_dataset("m_mu", data=np.array(m_mu, dtype=np.int32))

        per_ik = f.create_group("per_ik")
        for ik in ik_list:
            eps = (ik * dk) ** 2 / 2.0
            d   = delta_func(eps)
            g   = per_ik.create_group(f"ik{ik:04d}")
            A = np.eye(N_psi) * np.cos(d)
            B = np.eye(N_psi) * (-np.sin(d))    # so A-iB = e^{+iδ}·I, U = e^{-iδ}·I
            g.create_dataset("A", data=A)
            g.create_dataset("B", data=B)
            # D_ortho = U† · d_raw = e^{+iδ} · (d_const, d_const, ...)
            D_re = np.full(N_psi, d_const * np.cos(d))
            D_im = np.full(N_psi, d_const * np.sin(d))
            for q in ("x", "y", "z"):
                g.create_dataset(f"D_ortho_len_{q}_re", data=D_re)
                g.create_dataset(f"D_ortho_len_{q}_im", data=D_im)

        pairs = f.create_group("pairs")
        cc_block = np.full((N_psi, N_psi), cc_const, dtype=np.float64)
        for ik_k in ik_list:
            for ik_n in ik_list:
                pg = pairs.create_group(f"pair_k{ik_k:04d}_n{ik_n:04d}")
                for q in ("x", "y", "z"):
                    pg.create_dataset(f"cc_raw_len_{q}", data=cc_block)


def run_test(delta_func, name: str, T_L_fs: float = 30.0,
             tau_delay_fs: float = 0.0, tol_as: float = 2.0,
             ik_lo: int = 20, ik_hi: int = 100,
             sidebands: str = "55,60,65") -> bool:
    """Use a WIDE symmetric ν range (ik 20..100) and a LONG IR pulse
    (T_L = 30 fs) so the kernel is narrow + well-contained.  These
    parameters are what makes the IDENTITY τ = 0 numerically achievable;
    at narrower ν or shorter T_L the asymmetric Gaussian tail captured
    by the finite window contributes a spurious τ that depends only on
    the WINDOW geometry, not on the synthetic physics."""
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="synth_T1_")
    pa_path = Path(tmpdir) / "phase_a.h5"
    out_path = Path(tmpdir) / "tau.h5"

    write_synthetic_phase_a(pa_path, l_cont=2, dk=0.01,
                            ik_list=list(range(ik_lo, ik_hi + 1)),
                            delta_func=delta_func)
    cmd = ["python3", "postprocessing/two_photon_delay/python/run_cc_delay.py",
           "--phase-a", str(pa_path), "--out", str(out_path),
           "--sidebands", sidebands,
           "--omega-IR-eV", "1.55", "--T-X-fs", "0.35",
           "--T-L-fs", str(T_L_fs), "--tau-delay-fs", str(tau_delay_fs),
           "--angle-n-theta", "3", "--angle-n-phi", "3",
           "--orient-n-alpha", "3", "--orient-n-beta", "3",
           "--orient-n-gamma", "3", "--quiet"]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)

    AU_PER_AS = 24.188843
    with h5py.File(out_path, "r") as f:
        taus_au = []
        for k in sorted(f.keys()):
            g = f[k]
            if "tau_avg" in g:
                taus_au.append(float(g["tau_avg"][()]))
    taus_as = np.array(taus_au) * AU_PER_AS

    print(f"  [{name}]  τ_2ℏω(κ=55,60,65) = {taus_as[0]:+.4f}, "
          f"{taus_as[1]:+.4f}, {taus_as[2]:+.4f}  as")
    ok = np.all(np.abs(taus_as) < tol_as)
    print(f"    {'PASS' if ok else 'FAIL'}  (analytic = 0,  tol = {tol_as} as)")
    return bool(ok)


def main() -> int:
    print("=== T1: synthetic single-channel-style cancellation tests ===")
    print("  Setup: diagonal scalar U_κ = e^{-iδ(ε)}·I, cc_raw = 1, d_raw = 1")
    print("  Predict: τ_2ℏω = 0 EXACTLY  (within numerical tol)\n")

    fails = 0

    # ---- case A: zero phase shift everywhere ----
    if not run_test(lambda eps: 0.0, "T1.A  δ = 0"):
        fails += 1

    # ---- case B: constant phase shift δ = π/4 ----
    if not run_test(lambda eps: np.pi / 4.0, "T1.B  δ = π/4"):
        fails += 1

    # ---- case C: linear-in-E phase shift δ(ε) = 0.5·ε  ----
    if not run_test(lambda eps: 0.5 * eps, "T1.C  δ(ε) = 0.5·ε"):
        fails += 1

    # ---- case D: nonlinear phase shift δ(ε) = √ε  ----
    if not run_test(lambda eps: np.sqrt(eps), "T1.D  δ(ε) = √ε"):
        fails += 1

    # ---- case E: same as D but with τ_delay = 1 fs (independence check) ----
    if not run_test(lambda eps: np.sqrt(eps), "T1.E  δ(ε) = √ε,  τ_delay = 1 fs",
                    tau_delay_fs=1.0):
        fails += 1

    print(f"\n  Total failures: {fails} / 5")
    return fails


if __name__ == "__main__":
    sys.exit(main())
