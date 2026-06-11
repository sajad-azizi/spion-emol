#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

# =========================
# User settings
# =========================
INFILE  = "zpol_phi0_angular_resolved_phi0_len_z.dat"
OUTFILE = "FIG_2.png"

MIRROR_THETA_TO_2PI = True      # theta in [0,pi] -> fill [0,2pi] by mirroring
RADIUS_MODE = "Ekin_eV"       # "omega_eV", "Ekin_eV", or "k_au"

E_AU_TO_EV = 27.211386245988
TAU_FACTOR = 1.0                # file already stores tau in as
A02_TO_MB = 28.002852           # 1 bohr^2 = 28.002852 Mb

SIGMA_VMIN_MB, SIGMA_VMAX_MB = 1e-03*A02_TO_MB, None #### 1e-03*A02_TO_MB, 5e+01*A02_TO_MB   # None = autoscale from percentiles
TAU_VMIN, TAU_VMAX             = -200, 200

# =========================
# Plot style
# =========================
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6


# =========================
# Helpers
# =========================
def autoscale_lognorm(data, lo=1, hi=99):
    x = np.asarray(data)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return 1e-12, 1.0
    vmin = float(np.percentile(x, lo))
    vmax = float(np.percentile(x, hi))
    vmin = max(vmin, float(np.min(x)))
    vmax = max(vmax, vmin * 10.0)
    return vmin, vmax


def read_angular_resolved_grid(fname):
    """
    Read the corrected output produced by generate_data_fig2_correct.py.

    Expected columns:
      0 k [a.u.]
      1 E_kin [a.u.]
      2 omega [a.u.]
      3 theta [rad]
      4 Re D(theta,0)
      5 Im D(theta,0)
      6 sigma(theta,0) [bohr^2]
      7 tau(theta,0) [as]
      8 |D(theta,0)|^2

    Returns
    -------
    k1, e_kin1, omega1 : (nE,)
    theta1             : (nT,)
    sigma2d, tau2d     : (nE, nT)
    """
    d = np.loadtxt(fname, comments="#")
    if d.ndim != 2 or d.shape[1] < 8:
        raise ValueError(
            f"{fname}: expected >=8 columns with k, E_kin, omega, theta, ..., sigma, tau. Got {d.shape}"
        )

    k = d[:, 0]
    e_kin = d[:, 1]
    omega = d[:, 2]
    th = d[:, 3]
    sigma = d[:, 6]
    tau = d[:, 7]

    k_u = np.unique(k)
    th_u = np.unique(th)
    nE, nT = k_u.size, th_u.size

    if d.shape[0] != nE * nT:
        raise ValueError(
            f"{fname}: rows={d.shape[0]} but unique(k)*unique(theta)={nE}*{nT}={nE*nT}. "
            "File is not a complete regular grid."
        )

    order = np.lexsort((th, k))  # sort by k then theta
    k_s = k[order]
    e_s = e_kin[order]
    o_s = omega[order]
    th_s = th[order]
    sigma_s = sigma[order]
    tau_s = tau[order]

    k_grid = k_s.reshape(nE, nT)
    e_grid = e_s.reshape(nE, nT)
    o_grid = o_s.reshape(nE, nT)
    th_grid = th_s.reshape(nE, nT)

    if not (
        np.allclose(k_grid, k_grid[:, [0]])
        and np.allclose(e_grid, e_grid[:, [0]])
        and np.allclose(o_grid, o_grid[:, [0]])
        and np.allclose(th_grid, th_grid[[0], :])
    ):
        raise ValueError(f"{fname}: grid consistency check failed after sorting.")

    return (
        k_grid[:, 0],
        e_grid[:, 0],
        o_grid[:, 0],
        th_grid[0, :],
        sigma_s.reshape(nE, nT),
        tau_s.reshape(nE, nT),
    )


def mirror_theta_to_2pi(theta, Z):
    """
    theta: (nT,) in [0,pi]
    Z:     (nR, nT)
    Returns theta_full (nT_full,) and Z_full (nR, nT_full) filling [0,2pi].
    """
    theta = np.asarray(theta, float)
    Z = np.asarray(Z)
    if theta.ndim != 1 or Z.ndim != 2 or Z.shape[1] != theta.size:
        raise ValueError("mirror_theta_to_2pi: shape mismatch")

    idx = np.argsort(theta)
    theta = theta[idx]
    Z = Z[:, idx]

    # exclude endpoints to avoid duplicate rays
    core_theta = theta[1:-1]
    core_Z = Z[:, 1:-1]

    theta_m = 2.0 * np.pi - core_theta[::-1]
    Z_m = core_Z[:, ::-1]

    theta_full = np.concatenate([theta, theta_m])
    Z_full = np.concatenate([Z, Z_m], axis=1)
    return theta_full, Z_full


def format_polar(ax, radii, radii_labels, grid_color):
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    ax.set_thetagrids(angles_deg, labels=[f"{a}°" for a in angles_deg])
    ax.tick_params(axis="x", which="major", labelsize=6, pad=-3)

    for line in ax.xaxis.get_gridlines():
        line.set(linewidth=0.1, linestyle="--", color=grid_color)

    ax.set_rgrids(radii, labels=radii_labels, angle=7.4 * 22.5)
    ax.tick_params(axis="y", which="major", labelsize=3.3, pad=-3)
    ax.set_rmin(radii[0])
    ax.set_rmax(radii[-1])

    for line in ax.yaxis.get_gridlines():
        line.set(linewidth=0.15, linestyle="--", color=grid_color)

    ax.set_xticklabels([])

    extra = 0.8 * ax.get_rmax()
    ax.text(0, ax.get_rmax() + 0.55*extra, r"0$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(np.pi/4, ax.get_rmax() + 0.6*extra, r"45$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(np.pi/2, ax.get_rmax() + 0.55*extra, r"90$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(3*np.pi/4, ax.get_rmax() + 0.64*extra, r"135$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(np.pi, ax.get_rmax() + 0.75*extra, r"180$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(5*np.pi/4, ax.get_rmax() + 0.70*extra, r"225$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(3*np.pi/2, ax.get_rmax() + 0.6*extra, r"270$^{\circ}$", ha="center", va="center", fontsize=5.5)
    ax.text(7*np.pi/4, ax.get_rmax() + 0.70*extra, r"315$^{\circ}$", ha="center", va="center", fontsize=5.5)


def add_radius_scale_axis(fig, ax, xticks, xlabel, right_shift=0.0):
    rect = ax.get_position()
    rect = (
        rect.xmin + 0.138 + right_shift,
        rect.ymin + rect.height/2 - 0.18,
        rect.width - 0.22,
        rect.height/2,
    )
    scale_ax = fig.add_axes(rect)
    for loc in ["right", "top", "left"]:
        scale_ax.spines[loc].set_visible(False)
    scale_ax.tick_params(left=False, labelleft=False)
    scale_ax.patch.set_visible(False)
    scale_ax.set_xticks(xticks)
    scale_ax.xaxis.set_minor_locator(AutoMinorLocator())
    scale_ax.tick_params(direction="inout", which="major", length=5, width=0.8, pad=0.8)
    scale_ax.tick_params(direction="in", which="minor", length=1, width=0.8)

    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_coords(1.27, -0.1)


# =========================
# Read + prepare data
# =========================
k_au, Ekin_au, omega_au, theta, sigma_bohr2, tau_as = read_angular_resolved_grid(INFILE)
tau_as *= TAU_FACTOR

if RADIUS_MODE == "omega_eV":
    r = omega_au * E_AU_TO_EV
    r_label = r"$\hbar\omega\,$[eV]"
elif RADIUS_MODE == "Ekin_eV":
    r = Ekin_au * E_AU_TO_EV
    r_label = r"$E\,$[eV]"
elif RADIUS_MODE == "k_au":
    r = k_au
    r_label = r"$k\,$[a.u.]"
else:
    raise ValueError('RADIUS_MODE must be "omega_eV", "Ekin_eV", or "k_au"')

theta0 = theta.copy()
if MIRROR_THETA_TO_2PI:
    theta, tau_as = mirror_theta_to_2pi(theta0, tau_as)
    theta, sigma_bohr2 = mirror_theta_to_2pi(theta0, sigma_bohr2)

Theta, R = np.meshgrid(theta, r)

sigma_mb = sigma_bohr2 * A02_TO_MB
sigma_plot = np.ma.masked_where(~np.isfinite(sigma_mb) | (sigma_mb <= 0), sigma_mb)
if SIGMA_VMIN_MB is None or SIGMA_VMAX_MB is None:
    vmin_auto, vmax_auto = autoscale_lognorm(sigma_plot)
    sig_vmin = vmin_auto if SIGMA_VMIN_MB is None else SIGMA_VMIN_MB
    sig_vmax = vmax_auto if SIGMA_VMAX_MB is None else SIGMA_VMAX_MB
else:
    sig_vmin, sig_vmax = SIGMA_VMIN_MB, SIGMA_VMAX_MB

# =========================
# Figure
# =========================
article_width_pt = 510 / 2
article_width_in = article_width_pt / 72.27

fig, axs = plt.subplots(
    1, 2,
    subplot_kw={"projection": "polar"},
    figsize=(article_width_in, article_width_in)
)
ax1, ax2 = axs

pos_angle = np.pi / 4.5
pos_r = np.max(r) * 1.4
ax1.text(pos_angle, pos_r, r"(a)")
ax2.text(pos_angle, pos_r, r"(b)")

rmax = float(np.max(r))
if RADIUS_MODE in ("omega_eV", "Ekin_eV"):
    if rmax <= 20:
        radii = [5, 10, 15]
        xt = [0, 5, 10, 15, 20]
    elif rmax <= 40:
        radii = [10, 20, 30]
        xt = [0, 10, 20, 30, 40]
    elif rmax <= 80:
        radii = [20, 40, 60]
        xt = [0, 20, 40, 60, 80]
    else:
        step = 20
        radii = list(np.arange(step, step * 4, step))
        xt = list(np.arange(0, step * (int(rmax // step) + 2), step))
else:
    if rmax >= 4:
        radii = [1, 2, 3, 4]
        xt = [0, 1, 2, 3, 4]
    else:
        radii = [0.25*rmax, 0.5*rmax, 0.75*rmax]
        xt = [0, 0.5*rmax, rmax]

radii = [rv for rv in radii if 0 < rv < rmax]
if len(radii) < 2:
    radii = [0.25*rmax, 0.5*rmax, 0.75*rmax]
radii_labels = [f"{rv:g}" for rv in radii]

format_polar(ax1, radii, radii_labels, grid_color="white")
format_polar(ax2, radii, radii_labels, grid_color="gray")

pcm1 = ax1.pcolormesh(
    Theta,
    R,
    sigma_plot,
    norm=LogNorm(vmin=sig_vmin, vmax=sig_vmax),
    shading="auto",
)
cax1 = inset_axes(ax1, width="5%", height="80%", loc="right", borderpad=-1.3)
cbar1 = fig.colorbar(pcm1, cax=cax1)
cbar1.set_label(r"$\sigma(E,\vartheta)$ [Mb]", labelpad=1.2)
cbar1.ax.tick_params(labelsize=6, pad=0.6)

tau_plot = np.ma.masked_where(~np.isfinite(tau_as), tau_as)
pcm2 = ax2.pcolormesh(
    Theta,
    R,
    tau_plot,
    cmap="bwr",
    vmin=TAU_VMIN,
    vmax=TAU_VMAX,
    shading="auto",
)
cax2 = inset_axes(ax2, width="5%", height="80%", loc="right", borderpad=-1.3)
cbar2 = fig.colorbar(pcm2, cax=cax2)
cbar2.set_label(r"$\tau(E,\vartheta)$ [as]", labelpad=0.0)
cbar2.ax.tick_params(labelsize=6, pad=0.6)

add_radius_scale_axis(fig, ax1, xticks=xt, xlabel=r_label, right_shift=0.0)
add_radius_scale_axis(fig, ax2, xticks=xt, xlabel=r_label, right_shift=0.08)

plt.subplots_adjust(wspace=0.8)
plt.savefig(OUTFILE, dpi=1200, bbox_inches="tight", facecolor="none", transparent=True)
# plt.show()
