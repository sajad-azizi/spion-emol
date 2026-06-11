#!/usr/bin/env python3
"""
plot_with_amplitude.py
----------------------
Per reviewer request: show BOTH the two-photon phase (τ_2ℏω) and the
two-photon amplitude |⟨M_<* M_>⟩| as functions of E_κ.  A real continuum
resonance should appear in BOTH (Fano-like).  An angular-interference
artifact appears only in the phase.
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
            ML = abs((W * Ml).sum() / W.sum())
            MG = abs((W * Mg).sum() / W.sum())
            rows.append((E, t, abs(cross), ML, MG))
    return np.array(rows).T   # 5×N


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

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # H2- delay
    ax = axes[0, 0]
    ax.plot(E_W_h2*EV, tW_h2*AS, "-",  c="tab:blue", lw=2, label=r"$\tau_W$")
    ax.plot(h2[0], h2[1], "o-", c="tab:red", ms=7, label=r"$\tau_{2\hbar\omega}$")
    ax.axhline(0, c="gray", lw=0.5)
    ax.set_title(r"H$_2^-$ photodetachment: delays")
    ax.set_xlabel("E$_\\kappa$ (eV)"); ax.set_ylabel(r"$\tau$ (as)")
    ax.grid(alpha=0.3); ax.legend(loc="best")

    # H2- amplitude
    ax = axes[0, 1]
    ax.plot(h2[0], h2[2], "o-", c="tab:purple", ms=7, label=r"$|\langle M_<^* M_>\rangle|$")
    ax.set_title(r"H$_2^-$: two-photon AMPLITUDE  (no resonance peak)")
    ax.set_xlabel("E$_\\kappa$ (eV)"); ax.set_ylabel(r"|M_<^* M_>| (arb)")
    ax.grid(alpha=0.3); ax.legend()

    # H2O- delay
    ax = axes[1, 0]
    ax.plot(E_W_h2o*EV, tW_h2o*AS, "-",  c="tab:blue", lw=2, label=r"$\tau_W$")
    ax.plot(h2o[0], h2o[1], "s-", c="tab:red", ms=7, label=r"$\tau_{2\hbar\omega}$")
    ax.axhline(0, c="gray", lw=0.5)
    ax.set_title(r"H$_2$O$^-$ photodetachment: delays")
    ax.set_xlabel("E$_\\kappa$ (eV)"); ax.set_ylabel(r"$\tau$ (as)")
    ax.grid(alpha=0.3); ax.legend(loc="best")

    # H2O- amplitude
    ax = axes[1, 1]
    ax.plot(h2o[0], h2o[2], "s-", c="tab:purple", ms=7, label=r"$|\langle M_<^* M_>\rangle|$")
    ax.set_title(r"H$_2$O$^-$: two-photon AMPLITUDE  (modest peak at 4.1 eV)")
    ax.set_xlabel("E$_\\kappa$ (eV)"); ax.set_ylabel(r"|M_<^* M_>| (arb)")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("Extended κ-sweep suggests a resonance-like two-photon phase structure;\n"
                 "the effect is NOT consistent with a simple continuum-continuum correction "
                 "(no τ_W feature; amplitude does not show a real resonance peak)",
                 fontsize=11)
    fig.tight_layout()
    out = Path("h2_test/h2_and_h2o_delay_with_amplitude.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
