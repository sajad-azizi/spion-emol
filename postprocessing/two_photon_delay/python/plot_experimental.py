#!/usr/bin/env python3
"""Final plot with EXPERIMENTAL RABBITT pulse parameters.

ω_IR = 1.55 eV  (800 nm Ti:Sa)
T_X  = 0.30 fs  (300 as XUV burst, typical APT)
T_L  = 30.0 fs  (community-standard IR pulse, ~11 cycles)

References: Klünder PRL 2011, Isinger Science 2017, Mauritsson PRL 2010,
Sabbar PRA 2015, Cattaneo Nature 2018, Vos Science 2018, Heuser PRA 2016.
"""
import h5py, numpy as np, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EV = 27.2114
AS = 24.188843


def load(dat):
    arr = np.loadtxt(dat)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    return arr


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
    h2  = load("/tmp/h2_experimental/two_photon_delay.dat")
    h2o = load("/tmp/h2o_experimental/two_photon_delay.dat")
    E_W_h2,  tW_h2  = tauW(Path("h2_test/two_photon_me_h2_fine.h5"))
    E_W_h2o, tW_h2o = tauW(Path("h2o_test/two_photon_me_h2o_cc_delay.h5"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # H2-
    ax = axes[0, 0]
    ax.plot(E_W_h2*EV, tW_h2*AS, "-", c="tab:blue", lw=2,
            label=r"$\tau_W$ (Wigner-Smith)")
    ax.plot(h2[:, 2], h2[:, 4], "o-", c="tab:red", ms=7,
            label=r"$\tau_{2\hbar\omega}$ (BW17 Eq. 21; $\approx\tau_{mol}$ for Z=0)")
    ax.axhline(0, c="gray", lw=0.5)
    ax.set_title(r"H$_2^-$ photodetachment")
    ax.set_xlabel(r"$\varepsilon_\kappa$ (eV)"); ax.set_ylabel(r"$\tau$ (as)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(h2[:, 2], h2[:, 5], "o-", c="tab:purple", ms=7)
    ax.set_title(r"H$_2^-$: two-photon amplitude")
    ax.set_xlabel(r"$\varepsilon_\kappa$ (eV)")
    ax.set_ylabel(r"$|\langle M_<^* M_>\rangle|$ (a.u.)")
    ax.grid(alpha=0.3)

    # H2O-
    ax = axes[1, 0]
    ax.plot(E_W_h2o*EV, tW_h2o*AS, "-", c="tab:blue", lw=2,
            label=r"$\tau_W$ (Wigner-Smith)")
    ax.plot(h2o[:, 2], h2o[:, 4], "s-", c="tab:red", ms=7,
            label=r"$\tau_{2\hbar\omega}$ (BW17 Eq. 21; $\approx\tau_{mol}$ for Z=0)")
    ax.axhline(0, c="gray", lw=0.5)
    ax.set_title(r"H$_2$O$^-$ photodetachment")
    ax.set_xlabel(r"$\varepsilon_\kappa$ (eV)"); ax.set_ylabel(r"$\tau$ (as)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(h2o[:, 2], h2o[:, 5], "s-", c="tab:purple", ms=7)
    ax.set_title(r"H$_2$O$^-$: two-photon amplitude")
    ax.set_xlabel(r"$\varepsilon_\kappa$ (eV)")
    ax.set_ylabel(r"$|\langle M_<^* M_>\rangle|$ (a.u.)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Effective two-photon RABBITT delay (BW17 Eq. 21) at EXPERIMENTAL pulse parameters:\n"
        r"$\omega_{IR}\!=\!1.55$ eV, $T_X\!=\!0.30$ fs, $T_L\!=\!30$ fs (Klünder/Isinger/Mauritsson/Sabbar/Cattaneo).  "
        r"Off-resonance $\tau_{2\hbar\omega} \!\to\! \mathcal{O}(10)$ as, matching Lindroth-Dahlström for Z=0.",
        fontsize=10)
    fig.tight_layout()
    out = Path("h2_test/h2_and_h2o_experimental_pulse.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
