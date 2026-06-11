#!/usr/bin/env python3
"""
plot_final.py
-------------
Final plot using BW17 nomenclature.

Our pipeline computes BW17 Eq. 21:
    τ(2q, k̂, R̂_γ) = (1/2ω) · arg[ M^(2q-1)* · M^(2q+1) ]

which is the *full* effective two-photon RABBITT delay (sometimes called
the "atomic delay" τ_a or "molecular delay" τ_M).

BW17 Eq. 23 decomposes it as
    τ  =  τ_cc(2q)  +  τ_mol(2q, k̂, R̂_γ)

where:
    τ_cc  (Eq. 24) is the UNIVERSAL Coulomb-laser correction, depending only
                  on the residual-ion charge Z and the two harmonic energies.
    τ_mol (Eq. 25) is the MOLECULAR contribution carried by the b_{LM;(2q±1)}
                  coefficients (Eq. 19).

For the anion targets here (residual is neutral H₂ or H₂O), the asymptotic
factor A_κk (Eq. 14) has Z = 0 → its Coulomb phases vanish → arg(A_κ_-,k*
A_κ_+,k) is constant (0 or π modulo branch) → τ_cc ≈ 0.  Hence

    τ(2q, k̂, R̂_γ)  ≈  τ_mol(2q, k̂, R̂_γ)        (anion, Z = 0)

so the labeled curve is the *molecular* contribution per BW17 Eq. 25,
NOT the universal Coulomb-laser τ_cc.
"""
import h5py, numpy as np, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EV = 27.2114
AS = 24.188843


def collect(fn_template, kappas):
    rows = []
    for K in kappas:
        fn = fn_template.format(K)
        if not os.path.exists(fn): continue
        with h5py.File(fn, "r") as f:
            g = f["sideband_000"]
            E  = float(g["eps_kappa"][()]) * EV
            t  = float(g["tau_avg"][()]) * AS
            Ml = g["M_less"][:]
            Mg = g["M_greater"][:]
            W  = g["W_grid"][:]
            cross = (W * Ml.conj() * Mg).sum() / W.sum()
            rows.append((E, t, abs(cross)))
    return np.array(rows).T   # 3×N


def tauW(phase_a):
    with h5py.File(phase_a, "r") as f:
        dk = float(f.attrs["dk"])
        iks = sorted(int(g[2:]) for g in f["per_ik"].keys() if g.startswith("ik"))
        k = np.array(iks)*dk; E = k**2/2
        N = int(f.attrs["N_psi"])
        D = np.zeros((len(iks), 3*N), dtype=complex)
        for i, ik in enumerate(iks):
            g = f[f"per_ik/ik{ik:04d}"]
            for qi, q in enumerate(("x","y","z")):
                D[i, qi*N:(qi+1)*N] = g[f"D_ortho_len_{q}_re"][:] + 1j*g[f"D_ortho_len_{q}_im"][:]
    dD = np.empty_like(D)
    for j in range(D.shape[1]): dD[:, j] = np.gradient(D[:, j], k)/k
    num = (np.conj(D)*dD).sum(axis=1).imag
    den = (abs(D)**2).sum(axis=1)
    return E, -np.divide(num, den, out=np.zeros_like(num), where=den>1e-30)


def main():
    H2_kappas  = [50, 55, 60, 65, 66, 67, 68, 69, 70]
    H2O_kappas = [50, 55, 60, 65, 70]
    h2  = collect("/tmp/h2_fine_{}.h5",  H2_kappas)
    h2o = collect("/tmp/h2o_{}.h5",       H2O_kappas)
    E_W_h2,  tW_h2  = tauW(Path("h2_test/two_photon_me_h2_fine.h5"))
    E_W_h2o, tW_h2o = tauW(Path("h2o_test/two_photon_me_h2o_cc_delay.h5"))

    # Labels with proper BW17 nomenclature
    tau_W_label     = r"$\tau_W$ (Wigner-Smith, BW17 Eq. 6)"
    tau_mol_label   = r"$\tau_{\rm mol}$ (BW17 Eq. 25;  $\tau_{cc}\!\approx\!0$ for Z=0)"
    M_label         = r"$|\langle M^{(2q-1)*} M^{(2q+1)}\rangle|$"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # H2- delay
    ax = axes[0, 0]
    ax.plot(E_W_h2*EV, tW_h2*AS, "-", c="tab:blue", lw=2, label=tau_W_label)
    ax.plot(h2[0], h2[1], "o-", c="tab:red", ms=7, label=tau_mol_label)
    ax.axhline(0, c="gray", lw=0.5)
    ax.set_title(r"H$_2^-$ photodetachment (THF; SE-HF + PCM)")
    ax.set_xlabel(r"photoelectron kinetic energy $\varepsilon_\kappa$ (eV)")
    ax.set_ylabel(r"delay $\tau$ (as)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    # H2- amplitude
    ax = axes[0, 1]
    ax.plot(h2[0], h2[2], "o-", c="tab:purple", ms=7, label=M_label)
    ax.set_title(r"H$_2^-$: two-photon amplitude (no Fano-like peak)")
    ax.set_xlabel(r"$\varepsilon_\kappa$ (eV)")
    ax.set_ylabel(r"$|\langle M_<^* M_>\rangle|$ (a.u.)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # H2O- delay
    ax = axes[1, 0]
    ax.plot(E_W_h2o*EV, tW_h2o*AS, "-",  c="tab:blue", lw=2, label=tau_W_label)
    ax.plot(h2o[0], h2o[1], "s-", c="tab:red", ms=7, label=tau_mol_label)
    ax.axhline(0, c="gray", lw=0.5)
    ax.set_title(r"H$_2$O$^-$ photodetachment (THF; SE-HF + PCM)")
    ax.set_xlabel(r"$\varepsilon_\kappa$ (eV)")
    ax.set_ylabel(r"delay $\tau$ (as)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    # H2O- amplitude
    ax = axes[1, 1]
    ax.plot(h2o[0], h2o[2], "s-", c="tab:purple", ms=7, label=M_label)
    ax.set_title(r"H$_2$O$^-$: two-photon amplitude (modest bump at 4.1 eV)")
    ax.set_xlabel(r"$\varepsilon_\kappa$ (eV)")
    ax.set_ylabel(r"$|\langle M_<^* M_>\rangle|$ (a.u.)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle(
        r"Effective two-photon RABBITT delay (BW17 Eq. 21) $\equiv\tau_{\rm mol}$ for $Z\!=\!0$ anion residual."
        "\nFor short-range residual, BW17's universal $\\tau_{cc}$ (Eq. 24) vanishes; "
        "the curve is the molecular $\\tau_{\\rm mol}$ (Eq. 25).",
        fontsize=11)
    fig.tight_layout()
    out = Path("h2_test/h2_and_h2o_tau_mol.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
