#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

INFILE = "avg_phi_pol_averaged.dat"
OUTFILE_PNG = "FIG_5.png"
OUTFILE_PDF = "FIG_5.pdf"

MIRROR_THETA_TO_2PI = True
RADIUS_MODE = "E_eV"   # "E_eV" or "k_au"
E_RADIUS_MAX_EV = 120.0

E_AU_TO_EV = 27.211386245988
TAU_FACTOR = 1.0
A0SQ_TO_MB = 28.002852

SIGMA_VMIN_MB, SIGMA_VMAX_MB = 1e-03*A0SQ_TO_MB, 5e+01*A0SQ_TO_MB #####None, None
TAU_VMIN, TAU_VMAX = -200.0, 200.0

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6


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


def edges_from_centers(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Need a 1D array with at least 2 points.")
    dx = np.diff(x)
    e = np.empty(x.size + 1, dtype=float)
    e[1:-1] = x[:-1] + 0.5 * dx
    e[0] = x[0] - 0.5 * dx[0]
    e[-1] = x[-1] + 0.5 * dx[-1]
    return e


def read_E_theta_grid(fname):
    d = np.loadtxt(fname, comments="#")
    if d.ndim != 2 or d.shape[1] < 4:
        raise ValueError(f"{fname}: expected >=4 columns (E, theta, tau, sigma). Got {d.shape}")

    E = d[:, 0]
    th = d[:, 1]
    tau = d[:, 2]
    sig = d[:, 3]

    E_u = np.unique(E)
    th_u = np.unique(th)
    nE, nT = E_u.size, th_u.size

    if d.shape[0] != nE * nT:
        raise ValueError(
            f"{fname}: rows={d.shape[0]} but unique(E)*unique(theta)={nE}*{nT}={nE*nT}"
        )

    order = np.lexsort((th, E))
    E_s, th_s = E[order], th[order]
    tau_s, sig_s = tau[order], sig[order]

    E_grid = E_s.reshape(nE, nT)
    th_grid = th_s.reshape(nE, nT)

    if not np.allclose(E_grid, E_grid[:, [0]], rtol=0.0, atol=1e-12):
        raise ValueError(f"{fname}: inconsistent E grid after sorting.")
    if not np.allclose(th_grid, th_grid[[0], :], rtol=0.0, atol=1e-12):
        raise ValueError(f"{fname}: inconsistent theta grid after sorting.")

    return E_grid[:, 0], th_grid[0, :], tau_s.reshape(nE, nT), sig_s.reshape(nE, nT)


def mirror_theta_to_2pi(theta, Z):
    theta = np.asarray(theta, float)
    Z = np.asarray(Z)

    if theta.ndim != 1 or Z.ndim != 2 or Z.shape[1] != theta.size:
        raise ValueError("mirror_theta_to_2pi: shape mismatch")

    idx = np.argsort(theta)
    theta = theta[idx]
    Z = Z[:, idx]

    core_theta = theta[1:-1]
    core_Z = Z[:, 1:-1]

    theta_m = 2.0 * np.pi - core_theta[::-1]
    Z_m = core_Z[:, ::-1]

    theta_full = np.concatenate([theta, theta_m])
    Z_full = np.concatenate([Z, Z_m], axis=1)

    order = np.argsort(theta_full)
    return theta_full[order], Z_full[:, order]


def choose_radial_ticks(rmax, mode):
    if mode == "E_eV":
        if rmax <= 25:
            radii = [5, 10, 15, 20]
        elif rmax <= 50:
            radii = [10, 20, 30, 40]
        elif rmax <= 80:
            radii = [20, 40, 60, 80]
        elif rmax <= 120:
            radii = [20, 40, 60, 100]
        else:
            radii = [20, 40, 60, 100]
        radii = [r for r in radii if r <= rmax + 1e-9]
        if not radii:
            radii = [round(0.3 * rmax, 1), round(0.6 * rmax, 1), round(0.9 * rmax, 1)]
    else:
        if rmax >= 4:
            radii = [1, 2, 3, 4]
        elif rmax >= 3:
            radii = [1, 2, 3]
        else:
            radii = [round(0.3 * rmax, 2), round(0.6 * rmax, 2), round(0.9 * rmax, 2)]
    return radii, [f"{x:g}" for x in radii]


def format_polar(ax, radii, radii_labels, rmax, grid_color):
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    ax.set_thetagrids(angles_deg, labels=[f"{a}°" for a in angles_deg])
    ax.tick_params(axis="x", which="major", labelsize=6, pad=-3)

    for line in ax.xaxis.get_gridlines():
        line.set(linewidth=0.1, linestyle="--", color=grid_color)

    ax.set_rgrids(radii, labels=radii_labels, angle=7.4 * 22.5)
    ax.tick_params(axis="y", which="major", labelsize=4.0, pad=-3)
    ax.set_rmin(radii[0])
    ax.set_rmax(rmax)

    for line in ax.yaxis.get_gridlines():
        line.set(linewidth=0.15, linestyle="--", color=grid_color)

    ax.set_xticklabels([])

    extra = 0.12 * ax.get_rmax()
    ax.text(0, ax.get_rmax() + 0.95 * extra, r"0$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(np.pi/4, ax.get_rmax() + 1.1 * extra, r"45$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(np.pi/2, ax.get_rmax() + 0.95 * extra, r"90$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(3*np.pi/4, ax.get_rmax() + 1.1 * extra, r"135$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(np.pi, ax.get_rmax() + 1.85 * extra, r"180$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(5*np.pi/4, ax.get_rmax() + 1.7 * extra, r"225$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(3*np.pi/2, ax.get_rmax() + 1. * extra, r"270$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(7*np.pi/4, ax.get_rmax() + 1.7 * extra, r"315$^{\circ}$", ha="center", va="center", fontsize=5.5)


def add_radius_scale_axis(fig, ax, xticks, xlabel, xmax, extra_shift=0.0):
    rect = ax.get_position()
    rect = (
        rect.xmin + 0.138 + extra_shift,
        rect.ymin + rect.height / 2 - 0.18,
        rect.width - 0.22,
        rect.height / 2,
    )
    scale_ax = fig.add_axes(rect)
    for loc in ["right", "top", "left"]:
        scale_ax.spines[loc].set_visible(False)
    scale_ax.tick_params(left=False, labelleft=False)
    scale_ax.patch.set_visible(False)
    scale_ax.set_xticks(xticks)
    scale_ax.set_xlim(0.0, xmax)
    scale_ax.xaxis.set_minor_locator(AutoMinorLocator())
    scale_ax.tick_params(direction="inout", which="major", length=5, width=0.8, pad=0.8)
    scale_ax.tick_params(direction="in", which="minor", length=1, width=0.8)

    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_coords(1.27, -0.1)


# Read data
E_au, theta0, tau, sigma = read_E_theta_grid(INFILE)
tau *= TAU_FACTOR

if RADIUS_MODE == "E_eV":
    r = E_au * E_AU_TO_EV
    r_label = r"$E$ [eV]"
    r_plot_max = E_RADIUS_MAX_EV
elif RADIUS_MODE == "k_au":
    r = np.sqrt(2.0 * E_au)
    r_label = r"$k$ [a.u.]"
    r_plot_max = float(np.max(r))
else:
    raise ValueError('RADIUS_MODE must be "E_eV" or "k_au"')

if MIRROR_THETA_TO_2PI:
    theta, tau = mirror_theta_to_2pi(theta0, tau)
    _, sigma = mirror_theta_to_2pi(theta0, sigma)
else:
    theta = theta0

sigma_mb = sigma * A0SQ_TO_MB

theta_edges = edges_from_centers(theta)
r_edges = edges_from_centers(r)

sigma_plot_mb = np.ma.masked_where(~np.isfinite(sigma_mb) | (sigma_mb <= 0), sigma_mb)
tau_plot = np.ma.masked_where(~np.isfinite(tau), tau)

if SIGMA_VMIN_MB is None or SIGMA_VMAX_MB is None:
    vmin_auto, vmax_auto = autoscale_lognorm(sigma_plot_mb)
    sig_vmin = vmin_auto if SIGMA_VMIN_MB is None else SIGMA_VMIN_MB
    sig_vmax = vmax_auto if SIGMA_VMAX_MB is None else SIGMA_VMAX_MB
else:
    sig_vmin, sig_vmax = SIGMA_VMIN_MB, SIGMA_VMAX_MB

article_width_pt = 510 / 2
article_width_in = article_width_pt / 72.27

fig, axs = plt.subplots(
    1, 2,
    subplot_kw={"projection": "polar"},
    figsize=(article_width_in, article_width_in)
)
ax1, ax2 = axs

radii, radii_labels = choose_radial_ticks(r_plot_max, RADIUS_MODE)

format_polar(ax1, radii, radii_labels, r_plot_max, grid_color="white")
format_polar(ax2, radii, radii_labels, r_plot_max, grid_color="gray")

pos_angle = np.pi / 4.6
pos_r = max(r_plot_max * 1.45, np.max(r) * 1.25)
ax1.text(pos_angle, pos_r, r"(a)")
ax2.text(pos_angle, pos_r, r"(b)")

pcm1 = ax1.pcolormesh(
    theta_edges,
    r_edges,
    sigma_plot_mb,
    norm=LogNorm(vmin=sig_vmin, vmax=sig_vmax),
    shading="auto",
)
cax1 = inset_axes(ax1, width="5%", height="80%", loc="right", borderpad=-1.3)
cbar1 = fig.colorbar(pcm1, cax=cax1)
cbar1.set_label(r"$\sigma_{\mathrm{avg}}(E,\vartheta)$ [Mb]", labelpad=1.2)
cbar1.ax.tick_params(labelsize=6, pad=0.6)

pcm2 = ax2.pcolormesh(
    theta_edges,
    r_edges,
    tau_plot,
    cmap="bwr",
    vmin=TAU_VMIN,
    vmax=TAU_VMAX,
    shading="auto",
)
cax2 = inset_axes(ax2, width="5%", height="80%", loc="right", borderpad=-1.3)
cbar2 = fig.colorbar(pcm2, cax=cax2)
cbar2.set_label(r"$\tau_{\mathrm{avg}}(E,\vartheta)$ [as]", labelpad=0.0)
cbar2.ax.tick_params(labelsize=6, pad=0.6)

if RADIUS_MODE == "E_eV":
    xt = [x for x in [0, 40, 80, 120] if x <= r_plot_max + 1e-9]
    if len(xt) < 2:
        xt = [0, round(r_plot_max, 1)]
else:
    xt = [x for x in [0, 1, 2, 3, 4] if x <= r_plot_max + 1e-9]
    if len(xt) < 2:
        xt = [0, round(r_plot_max, 2)]

add_radius_scale_axis(fig, ax1, xticks=xt, xlabel=r_label, xmax=r_plot_max, extra_shift=0.0)
add_radius_scale_axis(fig, ax2, xticks=xt, xlabel=r_label, xmax=r_plot_max, extra_shift=0.08)

plt.subplots_adjust(wspace=0.8)

plt.savefig(OUTFILE_PNG, dpi=1200, bbox_inches="tight", facecolor="none", transparent=True)
###plt.savefig(OUTFILE_PDF, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
print(f"Saved {OUTFILE_PNG}")
