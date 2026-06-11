#!/usr/bin/env python3
"""
plot_potential.py -- plot V(r) slices from the CSV produced by
dump_potential_slices. Four panels:
  (a) angular-averaged V(r) = V_{00,00}(r)
  (b) p-orbital diagonals  V_{1,m;1,m}(r)  with centrifugal overlay
  (c) centrifugal-subtracted diagonals  V_{l,m;l,m} - l(l+1)/(2 r^2)
  (d) off-diagonal couplings

Usage:  plot_potential.py <csv>  [--out fig.png]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="potential.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    # Header is a commented "#" line; use csv library after stripping
    with open(args.csv) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    hdr = lines[0].strip().split(",")
    rows = np.array([[float(x) for x in ln.strip().split(",")] for ln in lines[1:]])
    col = {name: i for i, name in enumerate(hdr)}
    r = rows[:, col["r"]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(args.title or args.csv, fontsize=11)

    # (a) angular average V_{00,00}
    ax = axes[0, 0]
    ax.plot(r, rows[:, col["V_00_00"]], color="tab:blue", lw=1.5,
            label=r"$V_{0,0;\,0,0}(r) \equiv \langle V(r,\hat n)\rangle_\Omega$")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$r$ (Bohr)");  ax.set_ylabel(r"Hartree")
    ax.set_title(r"Angular-averaged $V(r)$")
    ax.set_xlim(r.min(), r.max())
    ax.grid(alpha=0.3); ax.legend()

    # (b) p-orbital diagonals WITH centrifugal
    ax = axes[0, 1]
    ax.plot(r, rows[:, col["V_1m_1m"]], label=r"$V_{1,-1;1,-1}$  (p$_y$)", color="tab:green")
    ax.plot(r, rows[:, col["V_10_10"]], label=r"$V_{1, 0;1, 0}$  (p$_z$)", color="tab:blue")
    ax.plot(r, rows[:, col["V_1p_1p"]], label=r"$V_{1,+1;1,+1}$  (p$_x$)", color="tab:red")
    ax.plot(r, rows[:, col["centrif_1"]], '--', color="black", lw=0.8,
            label=r"$\ell(\ell+1)/(2 r^2),\,\ell=1$")
    ax.set_xlabel(r"$r$ (Bohr)"); ax.set_ylabel("Hartree")
    ax.set_title(r"$p$-diagonal $V_{l,m;l,m}(r)$")
    ax.set_xlim(r.min(), r.max())
    # Log-y where sensible (centrifugal diverges at r=0 so clip):
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

    # (c) centrifugal-subtracted diagonals (the "true" local potential)
    ax = axes[1, 0]
    ax.plot(r, rows[:, col["V_10_10_noCF"]],
            label=r"$V_{1,0;1,0} - \ell(\ell+1)/(2 r^2)$", color="tab:blue")
    ax.plot(r, rows[:, col["V_20_20_noCF"]],
            label=r"$V_{2,0;2,0} - \ell(\ell+1)/(2 r^2)$", color="tab:orange")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$r$ (Bohr)"); ax.set_ylabel("Hartree")
    ax.set_title("Non-centrifugal diagonal (local potential part)")
    ax.set_xlim(r.min(), r.max())
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (d) off-diagonal couplings
    ax = axes[1, 1]
    ax.plot(r, rows[:, col["V_00_10"]], label=r"$V_{0,0;1,0}$ (s-p$_z$)", color="tab:purple")
    ax.plot(r, rows[:, col["V_10_20"]], label=r"$V_{1,0;2,0}$ (p$_z$-d$_{z^2}$)", color="tab:brown")
    ax.plot(r, rows[:, col["V_1m_1p"]], label=r"$V_{1,-1;1,+1}$ (p$_y$-p$_x$)", color="tab:pink")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$r$ (Bohr)"); ax.set_ylabel("Hartree")
    ax.set_title("Off-diagonal couplings")
    ax.set_xlim(r.min(), r.max())
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"-> wrote {args.out}")


if __name__ == "__main__":
    main()
