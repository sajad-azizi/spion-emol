#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

AZI_FILE = "zpol_azimuthal_average.dat"
FULL_FILE = "zpol_full_emission_average.dat"
OUTFILE = "FIG_3.png"

MIRROR_THETA_TO_2PI = True  # theta in [0,pi] -> [0,2pi] by mirroring

# If your input delays are already in attoseconds, keep these at 1.0
TAU_FACTOR_TOP = 1.0
TAU_FACTOR_FULL = 1.0

A0SQ_TO_MB = 28.002852  # 1 bohr^2 = 28.002852 Mb

TAU_VMIN, TAU_VMAX = -200.0, 200.0

# These are in Mb, not bohr^2. Use None for automatic percentile scaling.
SIGMA_VMIN_MB, SIGMA_VMAX_MB = 1e-03*A0SQ_TO_MB, 5e+01*A0SQ_TO_MB #####None, None

# ---------------- styling ----------------
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6


def edges_from_centers(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("Need at least 2 points to build edges.")
    dx = np.diff(x)
    e = np.empty(x.size + 1, dtype=float)
    e[1:-1] = x[:-1] + 0.5 * dx
    e[0] = x[0] - 0.5 * dx[0]
    e[-1] = x[-1] + 0.5 * dx[-1]
    return e


def autoscale_lognorm(data, lo=1.0, hi=99.0):
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return 1e-12, 1.0
    vmin = np.percentile(x, lo)
    vmax = np.percentile(x, hi)
    if vmin <= 0:
        vmin = np.min(x[x > 0])
    if vmax <= vmin:
        vmax = vmin * 10.0
    return float(vmin), float(vmax)


def read_azimuthal_grid(fname):
    """
    Expected columns:
        E_kin(a.u.), theta(rad), tau_tilde(as), sigma_tilde(bohr^2)

    Returns:
        E(nE), theta(nT), tau(nE,nT), sigma(nE,nT)
    """
    d = np.loadtxt(fname, comments="#")
    if d.ndim != 2 or d.shape[1] < 4:
        raise ValueError(f"{fname}: expected >=4 columns (E_kin, theta, tau, sigma). Got {d.shape}")

    E = d[:, 0]
    th = d[:, 1]
    tau = d[:, 2]
    sig = d[:, 3]

    E_u = np.unique(E)
    th_u = np.unique(th)
    nE, nT = E_u.size, th_u.size
    if d.shape[0] != nE * nT:
        raise ValueError(
            f"{fname}: rows={d.shape[0]} but unique(E)*unique(theta)={nE}*{nT}={nE*nT}."
        )

    order = np.lexsort((th, E))
    E_s, th_s = E[order], th[order]
    tau_s, sig_s = tau[order], sig[order]

    E_grid = E_s.reshape(nE, nT)
    th_grid = th_s.reshape(nE, nT)
    if not np.allclose(E_grid, E_grid[:, [0]]) or not np.allclose(th_grid, th_grid[[0], :]):
        raise ValueError(f"{fname}: grid consistency check failed after sorting.")

    return E_grid[:, 0], th_grid[0, :], tau_s.reshape(nE, nT), sig_s.reshape(nE, nT)


def read_full_emission(fname):
    """
    Expected columns:
        E_kin(a.u.), tau_bar(as), sigma_bar(bohr^2)
    """
    d = np.loadtxt(fname, comments="#")
    if d.ndim != 2 or d.shape[1] < 3:
        raise ValueError(f"{fname}: expected >=3 columns. Got {d.shape}")
    return d[:, 0], d[:, 1], d[:, 2]


def mirror_theta_to_2pi(theta, Z):
    """
    theta: (nT,) in [0, pi]
    Z:     (nE, nT)
    Returns theta_full (nT_full,) strictly increasing from 0 to <2pi,
            Z_full (nE, nT_full)
    """
    theta = np.asarray(theta, dtype=float)
    Z = np.asarray(Z)

    if theta.ndim != 1 or Z.ndim != 2 or Z.shape[1] != theta.size:
        raise ValueError("mirror_theta_to_2pi: shape mismatch")

    idx = np.argsort(theta)
    theta = theta[idx]
    Z = Z[:, idx]

    theta_core = theta[1:-1]
    Z_core = Z[:, 1:-1]

    theta_m = 2.0 * np.pi - theta_core[::-1]
    Z_m = Z_core[:, ::-1]

    theta_full = np.concatenate([theta, theta_m])
    Z_full = np.concatenate([Z, Z_m], axis=1)

    order = np.argsort(theta_full)
    return theta_full[order], Z_full[:, order]


def format_polar(ax, kmax, grid_color="white"):
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    ax.set_thetagrids(angles_deg, labels=[f"{a}°" for a in angles_deg])
    ax.tick_params(axis="x", which="major", labelsize=6, pad=-7)

    for line in ax.xaxis.get_gridlines():
        line.set(linewidth=0.1, linestyle="--", color=grid_color)

    rmax = max(4.0, float(np.ceil(kmax)))
    radii = [1, 2, 3] if rmax <= 4.5 else list(range(1, int(rmax) + 1))
    ax.set_rgrids(radii, labels=[f"{r}" for r in radii], angle=7.4 * 22.5)
    ax.tick_params(axis="y", which="major", labelsize=4.3, pad=-3)
    ax.set_rmin(radii[0])
    ax.set_rmax(radii[-1])

    for line in ax.yaxis.get_gridlines():
        line.set(linewidth=0.15, linestyle="--", color=grid_color)

    ax.set_xticklabels([])


def put_degree_labels(ax):
    ax.set_xticklabels([])
    r = ax.get_rmax()*1.29
    ax.text(0, r - 0.6, r'0$^{\circ}$', ha='center', va='center', fontsize=5.5)
    ax.text(np.pi / 4, r - 0.5, r'45$^{\circ}$', ha='center', va='center', fontsize=5.5)
    ax.text(np.pi / 2, r - 0.6, r'90$^{\circ}$', ha='center', va='center', fontsize=5.5)
    ax.text(3 * np.pi / 4, r - 0.5, r'135$^{\circ}$', ha='center', va='center', fontsize=5.5)
    ax.text(np.pi, r - 0.2, r'180$^{\circ}$', ha='center', va='center', fontsize=5.5)
    ax.text(5 * np.pi / 4, r - 0.2, r'225$^{\circ}$', ha='center', va='center', fontsize=5.5)
    ax.text(3 * np.pi / 2, r - 0.5, r'270$^{\circ}$', ha='center', va='center', fontsize=5.5)
    ax.text(7 * np.pi / 4, r - 0.36, r'315$^{\circ}$', ha='center', va='center', fontsize=5.5)


def main():
    # ================== read data ==================
    E, theta, tau_tilde, sigma_tilde = read_azimuthal_grid(AZI_FILE)
    tau_tilde = tau_tilde * TAU_FACTOR_TOP

    E2, tau_bar, sigma_bar = read_full_emission(FULL_FILE)
    tau_bar = tau_bar * TAU_FACTOR_FULL

    if len(E) != len(E2) or not np.allclose(E, E2, rtol=0.0, atol=1e-12):
        raise ValueError("Energy grids in azimuthal and full-emission files do not match.")

    k = np.sqrt(2.0 * E)

    theta0 = theta.copy()
    if MIRROR_THETA_TO_2PI:
        theta_tau, tau_tilde = mirror_theta_to_2pi(theta0, tau_tilde)
        theta_sigma, sigma_tilde = mirror_theta_to_2pi(theta0, sigma_tilde)
        if theta_tau.shape != theta_sigma.shape or not np.allclose(theta_tau, theta_sigma):
            raise RuntimeError("Mirroring produced different theta grids for tau and sigma.")
        theta = theta_tau

    theta_edges = edges_from_centers(theta)
    k_edges = edges_from_centers(k)

    # Convert sigma to Mb before scaling or plotting
    sigma_tilde_mb = sigma_tilde * A0SQ_TO_MB
    sigma_bar_mb = sigma_bar * A0SQ_TO_MB

    # ================== figure ==================
    article_width_pt = 510 / 2
    article_width_in = article_width_pt / 72.27

    fig = plt.figure(figsize=(article_width_in, article_width_in))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0], projection="polar")
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    format_polar(ax1, k.max(), grid_color="white")
    format_polar(ax2, k.max(), grid_color="gray")
    put_degree_labels(ax1)
    put_degree_labels(ax2)

    ax1.text(np.pi / 4.5, ax1.get_rmax() + 1.1, r"(a)")
    ax2.text(np.pi / 4.5, ax2.get_rmax() + 1.1, r"(b)")

    # --- sigma lognorm in Mb ---
    sigma_plot_mb = np.ma.masked_where(~np.isfinite(sigma_tilde_mb) | (sigma_tilde_mb <= 0), sigma_tilde_mb)
    if SIGMA_VMIN_MB is None or SIGMA_VMAX_MB is None:
        vmin_auto, vmax_auto = autoscale_lognorm(sigma_plot_mb)
        vmin = vmin_auto if SIGMA_VMIN_MB is None else SIGMA_VMIN_MB
        vmax = vmax_auto if SIGMA_VMAX_MB is None else SIGMA_VMAX_MB
    else:
        vmin, vmax = SIGMA_VMIN_MB, SIGMA_VMAX_MB

    pcm1 = ax1.pcolormesh(
        theta_edges,
        k_edges,
        sigma_plot_mb,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="auto",
    )
    cax1 = inset_axes(ax1, width="5%", height="80%", loc="right", borderpad=-1.3)
    cbar1 = fig.colorbar(pcm1, cax=cax1)
    cbar1.set_label(r"average $\tilde{\sigma}(k,\vartheta)$ [Mb]", labelpad=1.2)
    cbar1.ax.tick_params(labelsize=6, pad=0.6)

    # --- tau ---
    tau_plot = np.ma.masked_where(~np.isfinite(tau_tilde), tau_tilde)
    pcm2 = ax2.pcolormesh(
        theta_edges,
        k_edges,
        tau_plot,
        cmap="bwr",
        vmin=TAU_VMIN,
        vmax=TAU_VMAX,
        shading="auto",
    )
    cax2 = inset_axes(ax2, width="5%", height="80%", loc="right", borderpad=-1.3)
    cbar2 = fig.colorbar(pcm2, cax=cax2)
    cbar2.set_label(r"average $\tilde{\tau}(k,\vartheta)$ [as]", labelpad=0.0)
    cbar2.ax.tick_params(labelsize=6, pad=0.6)

    plt.subplots_adjust(wspace=0.8)

    # ================== bottom row ==================
    ax3.plot(k[1:], sigma_bar_mb[1:], lw=1.5, color="blue")
    ax3.set_xlabel(r"$k$ [a.u.]", labelpad=0.2)
    ax3.set_ylabel(r"average $\bar{\sigma}(k)$ [Mb]", labelpad=1.4)
    ax3.set_yscale("log")
    ax3.set_xticks([0, 1, 2, 3])
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.tick_params(direction="in", which="major", length=4, width=0.8)
    ax3.tick_params(direction="in", which="minor", length=1, width=0.8)

    ax4.plot(k[1:], tau_bar[1:], lw=1.5, color="blue")
    ax4.set_xlabel(r"$k$ [a.u.]", labelpad=0.0)
    ax4.set_xticks([0, 1, 2, 3])
    ax4.xaxis.set_minor_locator(AutoMinorLocator())
    ax4.tick_params(direction="in", which="major", length=4, width=0.8)
    ax4.tick_params(direction="in", which="minor", length=1, width=0.8)

    ax4r = ax4.twinx()
    ax4r.set_ylim(ax4.get_ylim())
    ax4r.set_ylabel(r"average $\bar{\tau}(k)$ [as]", labelpad=1.4)
    ax4r.yaxis.set_minor_locator(AutoMinorLocator())
    ax4r.tick_params(direction="in", which="major", length=4, width=0.8)
    ax4r.tick_params(direction="in", which="minor", length=1, width=0.8)

    ax4.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax4.spines["left"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    left, width = 0.19, 0.95
    height = 0.3
    bottom_lower_row = 0.17
    ax3.set_position([left + 0.0, bottom_lower_row, width / 2 - 0.1, height])
    ax4.set_position([left + width / 2 + 0.1 - 0.15, bottom_lower_row, width / 2 - 0.1, height])

    # Re-sync the right axis after manual repositioning
    ax4r.set_position(ax4.get_position())
    ax4r.set_ylim(ax4.get_ylim())

    ax3.text(2.0, 200, r"(c)")
    ax4.text(2.0, 300, r"(d)")

    plt.savefig(OUTFILE, dpi=1200, bbox_inches="tight", facecolor="none", transparent=True)
    # plt.show()


if __name__ == "__main__":
    main()
