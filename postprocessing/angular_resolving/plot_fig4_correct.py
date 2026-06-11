#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# Input files from generate_data_fig4_correct.py
# Expected columns:
#   E_kin(a.u.)   theta(rad)   tau_pol(as)   sigma_pol(bohr^2)
# ============================================================
FILE_PHI_0   = "pol_avg_phi_0.dat"
FILE_PHI_PI4 = "pol_avg_phi_pi_4.dat"
FILE_PHI_PI3 = "pol_avg_phi_pi_3.dat"

OUTFILE_PNG = "FIG_4.png"
OUTFILE_PDF = "FIG_4.pdf"

A0SQ_TO_MB = 28.002852  # 1 bohr^2 = 28.002852 Mb

# Plot limits
TAU_VMIN, TAU_VMAX = -200.0, 200.0

# Sigma limits are in Mb. Use None for automatic positive-percentile scaling.
SIGMA_VMIN_MB, SIGMA_VMAX_MB = 1e-03*A0SQ_TO_MB, 5e+01*A0SQ_TO_MB #####None, None

# Mirror theta in [0, pi] to full [0, 2pi)
MIRROR_TO_2PI = True

# Font/style
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6


# ============================================================
# Helpers
# ============================================================
def edges_from_centers(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Need a 1D array with at least 2 points to build edges.")
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


def read_polar_grid(fname):
    """
    Read file with columns:
        E_kin(a.u.)   theta(rad)   tau(as)   sigma(bohr^2)

    Returns:
        E_au:    (nE,)
        theta:   (nT,)
        tau:     (nE, nT)
        sigma:   (nE, nT)
    """
    d = np.loadtxt(fname, comments="#")
    if d.ndim != 2 or d.shape[1] < 4:
        raise ValueError(f"{fname}: expected at least 4 columns, got shape {d.shape}")

    E = d[:, 0]
    theta = d[:, 1]
    tau = d[:, 2]
    sigma = d[:, 3]

    E_u = np.unique(E)
    th_u = np.unique(theta)
    nE, nT = E_u.size, th_u.size

    if d.shape[0] != nE * nT:
        raise ValueError(
            f"{fname}: rows={d.shape[0]} but unique(E)*unique(theta)={nE}*{nT}={nE*nT}"
        )

    # Sort lexicographically by (E, theta)
    order = np.lexsort((theta, E))
    E_s = E[order]
    th_s = theta[order]
    tau_s = tau[order]
    sigma_s = sigma[order]

    E_grid = E_s.reshape(nE, nT)
    th_grid = th_s.reshape(nE, nT)
    tau_grid = tau_s.reshape(nE, nT)
    sigma_grid = sigma_s.reshape(nE, nT)

    if not np.allclose(E_grid, E_grid[:, [0]], rtol=0.0, atol=1e-12):
        raise ValueError(f"{fname}: inconsistent E grid after sorting.")
    if not np.allclose(th_grid, th_grid[[0], :], rtol=0.0, atol=1e-12):
        raise ValueError(f"{fname}: inconsistent theta grid after sorting.")

    return E_grid[:, 0], th_grid[0, :], tau_grid, sigma_grid


def mirror_theta_to_2pi(theta, Z):
    """
    Mirror data defined on theta in [0, pi] to [0, 2pi].
    Avoid duplicating theta=pi.
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


def choose_radial_gridlines(e_eV):
    emax = float(np.max(e_eV))
    if emax <= 25:
        radii = [5, 10, 15, 20]
    elif emax <= 50:
        radii = [10, 20, 30, 40]
    elif emax <= 80:
        radii = [20, 40, 60]
    else:
        radii = [20, 40, 60, 80]
    radii = [r for r in radii if r < emax]
    if not radii:
        radii = [round(emax * 0.3, 1), round(emax * 0.6, 1), round(emax * 0.9, 1)]
    labels = [f"{r:g}" for r in radii]
    return radii, labels


def format_polar_axis(ax, radii, radii_labels, grid_color):
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(angles_deg, labels=[f"{a}°" for a in angles_deg])
    ax.tick_params(axis="x", which="major", labelsize=6, pad=-3)

    for line in ax.xaxis.get_gridlines():
        line.set(linewidth=0.1, linestyle="--", color=grid_color)
    for line in ax.yaxis.get_gridlines():
        line.set(linewidth=0.15, linestyle="--", color=grid_color)

    ax.set_rgrids(radii, labels=radii_labels, angle=7.4 * 22.5)
    ax.tick_params(axis="y", which="major", labelsize=4.5, pad=-3)


def add_custom_angle_labels(ax):
    ax.set_xticklabels([])
    fontsize = 6
    rmax = ax.get_rmax()
    ax.text(0, rmax + rmax * 0.10, r"0$^{\circ}$", ha="center", va="center", fontsize=fontsize)
    ax.text(np.pi / 4, rmax + rmax * 0.14, r"45$^{\circ}$", ha="center", va="center", fontsize=fontsize)
    ax.text(np.pi / 2, rmax + rmax * 0.10, r"90$^{\circ}$", ha="center", va="center", fontsize=fontsize)
    ax.text(3 * np.pi / 4, rmax + rmax * 0.18, r"135$^{\circ}$", ha="center", va="center", fontsize=fontsize)
    ax.text(np.pi, rmax + rmax * 0.22, r"180$^{\circ}$", ha="center", va="center", fontsize=fontsize)
    ax.text(5 * np.pi / 4, rmax + rmax * 0.20, r"225$^{\circ}$", ha="center", va="center", fontsize=fontsize)
    ax.text(3 * np.pi / 2, rmax + rmax * 0.12, r"270$^{\circ}$", ha="center", va="center", fontsize=fontsize)
    ax.text(7 * np.pi / 4, rmax + rmax * 0.18, r"315$^{\circ}$", ha="center", va="center", fontsize=fontsize)


# ============================================================
# Main
# ============================================================
def main():
    # ---------- read files ----------
    E0_au, theta0, tau0, sigma0 = read_polar_grid(FILE_PHI_0)
    E1_au, theta1, tau1, sigma1 = read_polar_grid(FILE_PHI_PI4)
    E2_au, theta2, tau2, sigma2 = read_polar_grid(FILE_PHI_PI3)

    # ---------- consistency checks ----------
    if not np.allclose(E0_au, E1_au, rtol=0.0, atol=1e-12) or not np.allclose(E0_au, E2_au, rtol=0.0, atol=1e-12):
        raise ValueError("Energy grids do not match across phi files.")
    if not np.allclose(theta0, theta1, rtol=0.0, atol=1e-12) or not np.allclose(theta0, theta2, rtol=0.0, atol=1e-12):
        raise ValueError("Theta grids do not match across phi files.")

    E_au = E0_au
    theta = theta0

    # Convert radial coordinate to eV
    E_eV = E_au * 27.211386245988

    # Convert sigma to Mb before choosing LogNorm limits
    sigma0_mb = sigma0 * A0SQ_TO_MB
    sigma1_mb = sigma1 * A0SQ_TO_MB
    sigma2_mb = sigma2 * A0SQ_TO_MB

    # ---------- mirror theta ----------
    if MIRROR_TO_2PI:
        theta_full, sigma0_mb = mirror_theta_to_2pi(theta, sigma0_mb)
        _,         sigma1_mb = mirror_theta_to_2pi(theta, sigma1_mb)
        _,         sigma2_mb = mirror_theta_to_2pi(theta, sigma2_mb)

        _, tau0 = mirror_theta_to_2pi(theta, tau0)
        _, tau1 = mirror_theta_to_2pi(theta, tau1)
        _, tau2 = mirror_theta_to_2pi(theta, tau2)
    else:
        theta_full = theta

    # ---------- edges for pcolormesh ----------
    theta_edges = edges_from_centers(theta_full)
    E_edges = edges_from_centers(E_eV)

    # ---------- radial grid lines ----------
    radii, radii_labels = choose_radial_gridlines(E_eV)

    # ---------- sigma color scale in Mb ----------
    sigma_all = np.concatenate([sigma0_mb.ravel(), sigma1_mb.ravel(), sigma2_mb.ravel()])
    if SIGMA_VMIN_MB is None or SIGMA_VMAX_MB is None:
        vmin_auto, vmax_auto = autoscale_lognorm(sigma_all, lo=1.0, hi=99.0)
        sigma_vmin = vmin_auto if SIGMA_VMIN_MB is None else SIGMA_VMIN_MB
        sigma_vmax = vmax_auto if SIGMA_VMAX_MB is None else SIGMA_VMAX_MB
    else:
        sigma_vmin = SIGMA_VMIN_MB
        sigma_vmax = SIGMA_VMAX_MB

    # ---------- figure ----------
    article_width_pt = 510
    article_width_in = article_width_pt / 72.27

    fig, axs = plt.subplots(
        2, 3,
        subplot_kw={"projection": "polar"},
        figsize=(article_width_in, article_width_in * 0.7)
    )

    ax1, ax3, ax5 = axs[0, 0], axs[0, 1], axs[0, 2]
    ax2, ax4, ax6 = axs[1, 0], axs[1, 1], axs[1, 2]

    # ---------- upper row: sigma ----------
    for ax, cs_data, title in zip(
        [ax1, ax3, ax5],
        [sigma0_mb, sigma1_mb, sigma2_mb],
        [r"$\varphi=0$", r"$\varphi=\pi/4$", r"$\varphi=\pi/3$"]
    ):
        pcm = ax.pcolormesh(
            theta_edges,
            E_edges,
            np.ma.masked_where(~np.isfinite(cs_data) | (cs_data <= 0), cs_data),
            norm=LogNorm(vmin=sigma_vmin, vmax=sigma_vmax),
            cmap="viridis",
            shading="auto"
        )
        cax = inset_axes(ax, width="5%", height="80%", loc="right", borderpad=-1.6)
        cbar = fig.colorbar(pcm, cax=cax)
        if ax is ax5:
            cbar.set_label(r"$\sigma_{\mathrm{pol}}(E,\vartheta)$ [Mb]", labelpad=1.4)
        cbar.ax.tick_params(labelsize=6, pad=0.6)

        format_polar_axis(ax, radii, radii_labels, grid_color="white")
        ##ax.set_title(title, pad=12, fontsize=8)

    # ---------- lower row: tau ----------
    for ax, tau_data, title in zip(
        [ax2, ax4, ax6],
        [tau0, tau1, tau2],
        [r"$\varphi=0$", r"$\varphi=\pi/4$", r"$\varphi=\pi/3$"]
    ):
        pcm = ax.pcolormesh(
            theta_edges,
            E_edges,
            np.ma.masked_where(~np.isfinite(tau_data), tau_data),
            cmap="bwr",
            vmin=TAU_VMIN,
            vmax=TAU_VMAX,
            shading="auto"
        )
        cax = inset_axes(ax, width="5%", height="80%", loc="right", borderpad=-1.6)
        cbar = fig.colorbar(pcm, cax=cax)
        if ax is ax6:
            cbar.set_label(r"$\tau_{\mathrm{pol}}(E,\vartheta)$ [as]", labelpad=0.0)
        cbar.ax.tick_params(labelsize=6, pad=0.6)

        format_polar_axis(ax, radii, radii_labels, grid_color="gray")
        ##ax.set_title(title, pad=12, fontsize=8)

    # ---------- layout ----------
    plt.subplots_adjust(wspace=0.8, hspace=1.8)

    # Move bottom row closer to top row
    positions_top = [ax1.get_position().bounds, ax3.get_position().bounds, ax5.get_position().bounds]
    positions_bottom = [ax2.get_position().bounds, ax4.get_position().bounds, ax6.get_position().bounds]

    gap_reduction_factor = 0.13
    new_positions_bottom = []
    for top, bottom in zip(positions_top, positions_bottom):
        new_bottom = top[1] - (top[3] + (top[1] - bottom[1] - bottom[3]) * gap_reduction_factor)
        new_positions_bottom.append([bottom[0], new_bottom, bottom[2], bottom[3]])

    for ax, pos in zip([ax2, ax4, ax6], new_positions_bottom):
        ax.set_position(pos)

    # Custom angle labels
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        add_custom_angle_labels(ax)

    # Panel labels
    pos_angle = np.pi / 4.5
    pos_E = E_eV.max() * 1.48
    ax1.text(pos_angle, pos_E, r"(a)")
    ax3.text(pos_angle, pos_E, r"(b)")
    ax5.text(pos_angle, pos_E, r"(c)")
    ax2.text(pos_angle, pos_E, r"(d)")
    ax4.text(pos_angle, pos_E, r"(e)")
    ax6.text(pos_angle, pos_E, r"(f)")

    plt.savefig(OUTFILE_PNG, dpi=1200, bbox_inches="tight", facecolor="none", transparent=True)
    ##plt.savefig(OUTFILE_PDF, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
    print(f"Saved {OUTFILE_PNG}")


if __name__ == "__main__":
    main()
