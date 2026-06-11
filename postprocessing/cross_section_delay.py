#!/usr/bin/env python3
"""
cross_section_delay.py
----------------------
Consume per-channel dipole .dat files (written by gather_dipoles.py) and
compute:

  * total photoionization cross section sigma(E) in length and velocity gauges
  * Wigner-type time delay tau(E) in both gauges
  * gauge diagnostics: Q_sigma(E) and RMS phase offset dphi_rms(E)
  * orientation-averaged quantities (x + y + z)/3

Physics (atomic units throughout).  Our ψ_β has UNIT-AMPLITUDE regular
asymptote (A → I, so ψ_lm(r→∞) = ĵ_ℓ(kr)).  d_lm = ⟨Ψ⁻_lm|r·ε̂|Φ_0⟩ in that
convention, NOT energy-normalized.  Conversion to the Dill–Dehmer
energy-normalized amplitude carries a √(2/πk) factor, leaving

    sigma^L  = (8pi * omega / (c * k)) * sum_{lm} |d_{lm}^L|^2
    sigma^V  = (8pi        / (c * k * omega)) * sum_{lm} |d_{lm}^V|^2

    tau(E)   = Im[ sum_{lm} d_{lm}^*  d/dE d_{lm} ] / sum_{lm} |d_{lm}|^2
               ( = d arg(D)/dE = ∂_E Φ; NO minus sign by default, matching
                 density_current.pdf / rtd_content.py / gauge_panels.py.
                 Pass --with-minus-sign to recover the legacy −Im sign. )

    Q_sigma  = sum |d^V|^2 / (omega^2 * sum |d^L|^2)     (= 1 for exact H)
    dphi_rms = weighted RMS of arg(d^V / d^L)            (= 0 for exact H)

    beta(E)  = asymmetry parameter for photoelectron angular distribution
               with linearly polarized light on a randomly oriented sample.
               Computed by direct numerical integration over solid angle,
               averaging over the three Cartesian polarization directions.
               β ∈ [-1, 2]; atomic s→p limit is β=2; isotropic is β=0.

Outputs:

    delay_xsec_len_homo.dat   columns described in the file header
    delay_xsec_vel_homo.dat
    gauge_diagnostics.dat
    cross_section_delay_both.png
    gauge_diagnostics.png
    gauge_overlay.png

Usage:
    python cross_section_delay.py <gathered_dir> [--output-dir DIR]
                                                 [--xaxis k|E|omega]
                                                 [--idx-start N]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.special import sph_harm_y
    _SCIPY_HAS_SPH_HARM_Y = True
    def _sph_harm(l, m, theta, phi):
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    _SCIPY_HAS_SPH_HARM_Y = False
    from scipy.special import sph_harm
    def _sph_harm(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)  # note swapped args


# --------------------------------------------------------------------------
# Numerically-stable normalised associated Legendre via 3-term recurrence.
# Replaces scipy.special.sph_harm for the real-Y_lm computation in
# compute_beta, because older scipy versions overflow in the factorial-
# based normalisation at high (l, |m|) -- the symptom is NaN in Y^R at
# l ≳ 70, which silently makes β = 0 through the σ_tot>1e-30 fallback.
#
# This recurrence is bit-stable up to l ~ 1000 and matches the Condon-
# Shortley convention used everywhere else in the code.
# --------------------------------------------------------------------------
def _norm_legendre_PtildeLM(l: int, m: int, theta) -> np.ndarray:
    """Fully-normalised associated Legendre  P̃_l^|m|(cos θ)  =
       sqrt((2l+1)/(4π) · (l-m)!/(l+m)!) · P_l^m(cos θ),  CS-phased.

    Stable for arbitrary (l, |m|) -- no factorials computed explicitly.
    """
    am = abs(m)
    if am > l:
        return np.zeros_like(theta)
    x = np.cos(theta)
    s = np.sin(theta)
    # seed:  P̃_0^0 = 1/(2 sqrt(π))
    P_mm = np.full_like(x, 1.0 / (2.0 * np.sqrt(np.pi)))
    # diagonal up:  P̃_k^k = -sqrt((2k+1)/(2k)) · sin θ · P̃_{k-1}^{k-1}
    for k in range(1, am + 1):
        P_mm = -np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * s * P_mm
    if l == am:
        return P_mm
    # column up:  P̃_{am+1}^am = sqrt(2 am + 3) · x · P̃_am^am
    P_prev2 = P_mm
    P_prev1 = np.sqrt(2.0 * am + 3.0) * x * P_mm
    if l == am + 1:
        return P_prev1
    for ll in range(am + 2, l + 1):
        a = np.sqrt((4.0 * ll * ll - 1.0) / (ll * ll - am * am))
        b = np.sqrt((2.0 * ll + 1.0) / (2.0 * ll - 3.0)
                    * ((ll - 1.0) * (ll - 1.0) - am * am)
                    / (ll * ll - am * am))
        P_cur = a * x * P_prev1 - b * P_prev2
        P_prev2 = P_prev1
        P_prev1 = P_cur
    return P_prev1


# ------------------------ constants + column layout ------------------------
C_AU       = 137.036          # speed of light in au
AU_TO_AS   = 24.18884         # 1 au of time  -> attoseconds
BOHR2_TO_MB = 28.003          # 1 bohr^2      -> megabarn
HA_TO_EV   = 27.2114

DIPOLE_PREFIX_LEN = "dipole_len_homo"
DIPOLE_PREFIX_VEL = "dipole_vel_homo"

COL_K, COL_EKIN, COL_OMEGA = 0, 1, 2
COL_RE, COL_IM             = 3, 4
COL_DSQ, COL_ARG           = 5, 6


# --------------------------- file discovery -------------------------------
def extract_mu(path: str) -> int:
    m = re.search(r"_(\d+)\.dat$", Path(path).name)
    if m is None:
        raise ValueError(f"cannot extract mu from filename: {path}")
    return int(m.group(1))


def discover_mu_list(input_dir: Path, prefix: str, pol: str) -> List[int]:
    files = sorted(glob.glob(str(input_dir / f"{prefix}_{pol}_*.dat")), key=extract_mu)
    if not files:
        raise FileNotFoundError(f"no files match {prefix}_{pol}_*.dat in {input_dir}")
    return [extract_mu(f) for f in files]


def infer_common_mu(input_dir: Path,
                    prefixes: Iterable[str],
                    pols: Iterable[str]) -> List[int]:
    reference = None
    for prefix in prefixes:
        for pol in pols:
            mu_list = discover_mu_list(input_dir, prefix, pol)
            if reference is None:
                reference = mu_list
            elif mu_list != reference:
                raise ValueError(
                    f"channel-list mismatch for prefix={prefix}, pol={pol}\n"
                    f"  expected: {reference[:10]}{' ...' if len(reference) > 10 else ''}\n"
                    f"  got:      {mu_list[:10]}{' ...' if len(mu_list) > 10 else ''}")
    assert reference is not None
    return reference


# ------------------------------ loading -----------------------------------
def load_channel_file(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 7:
        raise ValueError(f"{path}: expected 7 columns, got {arr.shape[1]}")
    return arr


def load_dipole(input_dir: Path, prefix: str, pol: str,
                mu_list: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (k, E_kin, omega, D[n_e, n_ch])."""
    first = load_channel_file(input_dir / f"{prefix}_{pol}_{mu_list[0]}.dat")
    k, e, o = first[:, COL_K], first[:, COL_EKIN], first[:, COL_OMEGA]
    D = np.zeros((len(k), len(mu_list)), dtype=complex)
    for idx, mu in enumerate(mu_list):
        d = load_channel_file(input_dir / f"{prefix}_{pol}_{mu}.dat")
        if len(d) != len(k):
            raise ValueError(f"row count mismatch at mu={mu}")
        if not (np.allclose(d[:, COL_K], k) and
                np.allclose(d[:, COL_EKIN], e) and
                np.allclose(d[:, COL_OMEGA], o)):
            raise ValueError(f"grid mismatch at mu={mu}")
        D[:, idx] = d[:, COL_RE] + 1j * d[:, COL_IM]
    return k, e, o, D


# ---------------------------- observables ---------------------------------
def sigma_from_dipole(k: np.ndarray, D: np.ndarray, omega: np.ndarray,
                      gauge: str) -> np.ndarray:
    """
    Total photoionization cross section [bohr²], summed over partial waves.

    Convention in this code:
      * ψ_β(r→∞) = A_{β·} ĵ_ℓ(kr) + B_{β·} ŷ_ℓ(kr) with A ≈ I (regular
        solutions have UNIT AMPLITUDE — NOT energy-normalized).
      * d_lm = ⟨Ψ⁻_lm | r·ε̂ | Φ_0⟩ with Ψ⁻ built from ψ_β via (A−iB)⁻†.

    Conversion to energy-normalized amplitudes: d_EN = √(2/(πk)) · d_ours,
    so |d_EN|² = (2/(πk)) · |d_ours|². The standard Dill–Dehmer formula
        σ = 4π²α ω · Σ |d_EN|²
    becomes, in our convention,
        σ_L = (8π ω /(c·k))      · Σ |d_lm^L|²
        σ_V = (8π   /(c·k·ω))    · Σ |d_lm^V|²
    (α = 1/c.  Gauge equivalence d^V = ω·d^L ⇒ σ_L = σ_V for exact eigenfn.)
    """
    sum_d2 = np.sum(np.abs(D) ** 2, axis=1)
    if gauge == "length":
        sigma = 8.0 * np.pi * omega / (C_AU * k) * sum_d2
    elif gauge == "velocity":
        sigma = 8.0 * np.pi / (C_AU * k * omega) * sum_d2
    else:
        raise ValueError(f"unknown gauge {gauge!r}")
    return sigma


def raw_strength(D: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(D) ** 2, axis=1)


def time_delay(k: np.ndarray, D: np.ndarray,
               with_minus: bool = False) -> np.ndarray:
    """
    Wigner photoionization time delay = energy derivative of the
    matrix-element phase.

    DEFAULT (with_minus=False) — the convention used everywhere else in
    the postprocessing suite (density_current.pdf, rtd_content.py,
    gauge_panels.py):

        tau_W(E) = ∂_E arg D = + Im[ sum d* dd/dE ] / sum |d|^2     (NO minus)

    This is τ_W = ∂_E Φ of the density-current note.  For the C8F8
    production dipoles it gives a POSITIVE delay at the trapping
    resonance (≈ +170 as near k ≈ 1.3 a.u.) — the physically correct
    sign — and it matches this module's own header/docstring formula.

    LEGACY (with_minus=True) — multiplies by −1:

        tau_W(E) = − Im[ sum d* dd/dE ] / sum |d|^2

    This matched an older fixture (the cube L = 2.94 a.u.) whose stored
    matrix element had phase arg D⁻ = −δ_l (the bra <Ψ⁻| conjugated the
    outgoing phase), so the minus was needed there to recover +dδ_l/dE.
    Kept as an opt-in for back-comparison only; do NOT mix the two
    conventions in one analysis.

    dE = k · dk for E = k²/2; differentiate D wrt k and divide by k.
    """
    dD_dE = np.empty_like(D)
    for i in range(D.shape[1]):
        dD_dk = np.gradient(D[:, i], k)
        dD_dE[:, i] = dD_dk / k
    num = np.sum(np.conj(D) * dD_dE, axis=1).imag
    den = np.sum(np.abs(D) ** 2, axis=1)
    sign = -1.0 if with_minus else 1.0
    return sign * np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)


def q_sigma(D_L: np.ndarray, D_V: np.ndarray, omega: np.ndarray) -> np.ndarray:
    denom = omega ** 2 * np.sum(np.abs(D_L) ** 2, axis=1)
    numer = np.sum(np.abs(D_V) ** 2, axis=1)
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 1e-30)


def dphi_rms(D_L: np.ndarray, D_V: np.ndarray) -> np.ndarray:
    dphi = np.angle(D_V * np.conj(D_L))
    w    = np.abs(D_L) * np.abs(D_V)
    s_w  = np.sum(w, axis=1)
    s_w_phi2 = np.sum(w * dphi ** 2, axis=1)
    return np.sqrt(np.divide(s_w_phi2, s_w, out=np.zeros_like(s_w_phi2), where=s_w > 1e-30))


# ---------------------------- asymmetry β(ω) ------------------------------
#
# For linearly-polarized light on a randomly oriented molecular sample, the
# photoelectron angular distribution is
#
#     I(θ_lab) = (σ/4π) · [1 + β P_2(cos θ_lab)]
#
# where θ_lab is the angle between photoelectron wavevector k̂_lab and the
# polarization ε̂_lab.  β ∈ [-1, 2]; β=0 is isotropic, β=2 is pure ∝ cos²θ
# (atomic s→p limit), β=-1 is pure ∝ sin²θ (d-wave with specific phase).
#
# We compute β by DIRECT NUMERICAL INTEGRATION over the molecular-frame
# solid angle for each of the three Cartesian polarizations q ∈ {x, y, z},
# then average over q to realize the orientation average:
#
#   β = 5 · ⟨P_2⟩ / σ
#
# where
#   σ     = (1/3) Σ_q ∫ dΩ |D^(q)(n̂)|²
#   ⟨P_2⟩ = (1/3) Σ_q ∫ dΩ P_2(n̂ · q̂) |D^(q)(n̂)|²
#   D^(q)(n̂) = Σ_{lm} d^(q)_{lm} · Y^R_{lm}(n̂)
#
# Uses Gauss-Legendre in cos θ (spectrally exact for our partial-wave
# expansion up to sufficient order) and uniform spacing in φ.  No external
# package beyond scipy.  Works for any molecular symmetry — no Wigner-3j
# machinery needed.
# --------------------------------------------------------------------------

def _real_Ylm(l: int, m: int, theta, phi):
    """Real spherical harmonic Y^R_{l,m} in our convention (x → q=+1,
    y → q=-1, z → q=0).  Returns real array; shape matches broadcast of
    theta/phi.

    Routes:
      * scipy ≥ 1.15  (has sph_harm_y)  →  use scipy directly (fast,
        reference implementation).
      * older scipy  (only sph_harm)    →  use the stable normalised-
        Legendre 3-term recurrence below, because older scipy overflows
        the factorial normaliser at high (l, |m|) (NaN at l ≳ 70) and
        silently poisons the β calculation through the σ_tot>1e-30
        fallback.
    """
    am = abs(m)
    if _SCIPY_HAS_SPH_HARM_Y:
        # Modern scipy: use the reference implementation directly.
        if m == 0:
            return np.real(_sph_harm(l, 0, theta, phi))
        elif m > 0:
            return np.sqrt(2.0) * ((-1) ** m) * np.real(_sph_harm(l, m, theta, phi))
        else:
            return np.sqrt(2.0) * ((-1) ** am) * np.imag(_sph_harm(l, am, theta, phi))
    # Older scipy: fall back to the stable recurrence.
    P_lm = _norm_legendre_PtildeLM(l, am, theta)
    if m == 0:
        return P_lm
    elif m > 0:
        return np.sqrt(2.0) * ((-1) ** m) * P_lm * np.cos(m * phi)
    else:
        return np.sqrt(2.0) * ((-1) ** am) * P_lm * np.sin(am * phi)


def _idx_to_lm(mu: int):
    l = int(np.floor(np.sqrt(mu)))
    while (l + 1) * (l + 1) <= mu:
        l += 1
    return l, mu - l * l - l


def compute_beta(D_x: np.ndarray, D_y: np.ndarray, D_z: np.ndarray,
                 mu_list: List[int],
                 n_theta: int = 40, n_phi: int = 80) -> np.ndarray:
    """β(ω) orientation-averaged, per-energy.  Bounded β ∈ [−1, 2].

    The previous implementation evaluated only diagonal terms,
        β_old·σ = (5/3) Σ_q ∫|D^(q)(k̂)|² P_2(k̂·ê_q) dΩ_k,
    which happens to be correct for spherical (atomic) targets -- the
    cross terms ⟨D^(x)*·D^(y)⟩ vanish by m-orthogonality of the Y_1^m --
    but for non-spherical targets (cube, H₂O, C8F8) the cross terms are
    NOT zero and the formula can leave the physical bound β ∈ [−1, 2].
    (The cube test showed peak β ≈ +2.95 with the old formula.)

    The corrected formula, derived by doing the full SO(3) molecular-
    orientation average analytically and then collapsing the polarisation
    sphere integral via ⟨n_a n_b⟩ = δ_{ab}/3 and
    ⟨n_a n_b n_c n_d⟩ = (δ_{ab}δ_{cd}+δ_{ac}δ_{bd}+δ_{ad}δ_{bc})/15, is

        β = N_1 / σ_tot  −  1,

    where
        σ_tot = (1/3) ∫ dΩ_k Σ_q |D^(q)(k̂)|²
        N_1   =     ∫ dΩ_k |Σ_q D^(q)(k̂) · k̂_q |²

    Bound check: Cauchy–Schwarz on ⟨D, k̂⟩ vs Σ|D|² · Σ k_q² = Σ|D|²·1
    gives N_1 ≤ 3·σ_tot → β ≤ 2.  And N_1 ≥ 0 → β ≥ −1.

    Atomic sanity: with D^(q)(k̂) = k̂_q (proper s→p amplitude),
    Σ_q D^(q) k_q = |k̂|² = 1, so N_1 = 4π and σ_tot = 4π/3 → β = 2. ✓

    Parameters
    ----------
    D_x, D_y, D_z : complex arrays, shape (n_energies, n_channels)
    mu_list       : ordered list of channel indices (length = n_channels).
    n_theta, n_phi: integration grid (40×80 = 3200 pts is plenty for
                    l_max ≤ 20; Gauss–Legendre is exact in cos θ to
                    degree 2·n_theta − 1).
    """
    # -- integration grid (cos θ by Gauss-Legendre; φ by trapezoid) --
    x, w = np.polynomial.legendre.leggauss(n_theta)   # x = cos θ ∈ (-1, 1)
    theta = np.arccos(x)                              # shape (n_theta,)
    phi = np.arange(n_phi) * (2.0 * np.pi / n_phi)    # shape (n_phi,)
    dphi = 2.0 * np.pi / n_phi
    TH, PH = np.meshgrid(theta, phi, indexing='ij')   # both (n_theta, n_phi)

    # -- precompute Y^R_{lm}(θ, φ) for each channel once --
    Y_table = np.empty((len(mu_list), n_theta, n_phi), dtype=np.float64)
    for idx, mu in enumerate(mu_list):
        l, m = _idx_to_lm(mu)
        Y_table[idx] = _real_Ylm(l, m, TH, PH)

    # -- direction cosines for each polarization axis (k̂ components) --
    nx = np.sin(TH) * np.cos(PH)
    ny = np.sin(TH) * np.sin(PH)
    nz = np.cos(TH)

    # -- per-energy integration --
    n_energies = D_x.shape[0]
    beta = np.zeros(n_energies)
    weights = w[:, None] * dphi    # shape (n_theta, n_phi), ∫dΩ weight
    for ie in range(n_energies):
        # D^(q)(k̂) = Σ_lm d^(q)_{lm} Y^R_{lm}(k̂) -- complex map on the
        # (θ, φ) grid for each Cartesian polarisation q.
        Dmap_x = np.tensordot(D_x[ie], Y_table, axes=([0], [0]))
        Dmap_y = np.tensordot(D_y[ie], Y_table, axes=([0], [0]))
        Dmap_z = np.tensordot(D_z[ie], Y_table, axes=([0], [0]))

        # σ_tot = (1/3) ∫ dΩ_k (|D^x|² + |D^y|² + |D^z|²).
        sigma_q_sum = (np.abs(Dmap_x) ** 2
                     + np.abs(Dmap_y) ** 2
                     + np.abs(Dmap_z) ** 2)
        sigma_tot = (1.0 / 3.0) * np.sum(sigma_q_sum * weights)

        # N_1 = ∫ dΩ_k |D^x · k_x + D^y · k_y + D^z · k_z|².
        D_dot_k = Dmap_x * nx + Dmap_y * ny + Dmap_z * nz   # complex
        N1 = np.sum(np.abs(D_dot_k) ** 2 * weights)

        if sigma_tot > 1e-30:
            beta[ie] = N1 / sigma_tot - 1.0
        else:
            beta[ie] = 0.0
    return beta


# ------------------------------- plots ------------------------------------
def get_xvals(k, E_kin, omega, choice: str) -> Tuple[np.ndarray, str]:
    if choice == "k":
        return k, r"Momentum $k$ (a.u.)"
    if choice == "E":
        return E_kin * HA_TO_EV, r"Kinetic energy $E$ (eV)"
    if choice == "omega":
        return omega * HA_TO_EV, r"Photon energy $\omega$ (eV)"
    raise ValueError(f"unknown xaxis={choice!r}")


def load_two_photon_delay(path: Path):
    """Load two_photon_delay.dat (written by postprocessing/two_photon_delay.py).
    Returns dict with E_eV, tau_as, M_amp (|<M_<* M_>>|), or None if absent."""
    if not path.exists():
        return None
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 7:
        raise ValueError(
            f"{path}: expected ≥7 columns (ik, E_au, E_eV, tau_au, tau_as, "
            f"|M|, arg(M)), got {arr.shape[1]}")
    return {
        "ik":      arr[:, 0].astype(int),
        "E_au":    arr[:, 1],
        "E_eV":    arr[:, 2],
        "tau_au":  arr[:, 3],
        "tau_as":  arr[:, 4],
        "M_amp":   arr[:, 5],
        "M_arg":   arr[:, 6],
    }


def plot_two_panel_per_gauge(xvals, xlabel,
                             sigma_L, tau_L, sigma_V, tau_V,
                             idx_start: int, path: Path,
                             two_photon=None, xaxis: str = "E",
                             beta_L=None, beta_V=None) -> None:
    """Composite plot for the main figure.  Rows:

        Row 0    : cross section σ(E)        — length | velocity   (2 cols)
        Row 1    : Wigner-Smith τ_W(E)       — length | velocity   (2 cols)
       [Row 2]   : asymmetry β(E)            — length | velocity   (2 cols)
                                                present iff beta_L/V given
       [Row last]: two-photon τ_2ℏω + |M·M*| — present iff two_photon arg
                                                                  (2 cols)

    Optional rows are only drawn when their inputs are provided; the
    two-photon row is ALWAYS at the bottom.
    """
    import matplotlib.gridspec as gridspec

    sl = slice(idx_start, None)
    has_beta = (beta_L is not None) and (beta_V is not None)
    has_tp   = two_photon is not None
    n_rows   = 2 + (1 if has_beta else 0) + (1 if has_tp else 0)

    fig = plt.figure(figsize=(14, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.32, wspace=0.20)
    # All rows have 2 columns -- build axes up-front.
    ax = [[fig.add_subplot(gs[i, j]) for j in (0, 1)]
          for i in range(n_rows)]
    beta_row = 2 if has_beta else None
    tp_row   = (2 + (1 if has_beta else 0)) if has_tp else None
    colors = {"x": "C0", "y": "C1", "z": "C2", "avg": "k"}
    gauges = {"Length": (sigma_L, tau_L), "Velocity": (sigma_V, tau_V)}

    # ---- rows 0, 1 : σ and τ per gauge ----
    for col, (label, (sig, tau)) in enumerate(gauges.items()):
        a = ax[0][col]
        for d in ("x", "y", "z", "avg"):
            a.plot(xvals[sl], sig[d][sl] * BOHR2_TO_MB,
                   color=colors[d],
                   lw=2 if d == "avg" else 1,
                   label=(r"$\bar{\sigma}$" if d == "avg" else rf"$\sigma_{d}$"))
        a.set_xlabel(xlabel); a.set_ylabel(r"$\sigma$ (Mb)")
        a.set_yscale("log"); a.legend(); a.grid(True, alpha=0.3)
        a.set_title(f"Cross section — {label}")

        a = ax[1][col]
        for d in ("x", "y", "z", "avg"):
            a.plot(xvals[sl], tau[d][sl] * AU_TO_AS, color=colors[d],
                   lw=2 if d == "avg" else 1.2,
                   label=(r"$\bar{\tau}$" if d == "avg" else rf"$\tau_{d}$"))
        a.set_xlabel(xlabel); a.set_ylabel(r"$\tau$ (as)")
        a.legend(); a.grid(True, alpha=0.3)
        a.set_title(f"Wigner time delay — {label}")

    # ---- optional β row (above the τ_2ℏω row) : β(E) per gauge (2 cols) ----
    if has_beta:
        # length
        a = ax[beta_row][0]
        a.plot(xvals[sl], beta_L[sl], "b-", lw=2, label=r"$\beta_L$")
        a.axhline( 0.0, color="gray", ls=":", lw=0.8, alpha=0.6)
        a.axhline( 2.0, color="gray", ls=":", lw=0.5, alpha=0.4)
        a.axhline(-1.0, color="gray", ls=":", lw=0.5, alpha=0.4)
        a.set_xlabel(xlabel); a.set_ylabel(r"$\beta(E)$")
        a.legend(loc="best"); a.grid(True, alpha=0.3)
        a.set_title(r"Asymmetry parameter $\beta(E)$ — Length"
                    r"   (orientation-averaged; bound $\beta\in[-1,2]$)")
        # velocity
        a = ax[beta_row][1]
        a.plot(xvals[sl], beta_V[sl], "r-", lw=2, label=r"$\beta_V$")
        a.axhline( 0.0, color="gray", ls=":", lw=0.8, alpha=0.6)
        a.axhline( 2.0, color="gray", ls=":", lw=0.5, alpha=0.4)
        a.axhline(-1.0, color="gray", ls=":", lw=0.5, alpha=0.4)
        a.set_xlabel(xlabel); a.set_ylabel(r"$\beta(E)$")
        a.legend(loc="best"); a.grid(True, alpha=0.3)
        a.set_title(r"Asymmetry parameter $\beta(E)$ — Velocity"
                    r"   (orientation-averaged; bound $\beta\in[-1,2]$)")

    # ---- optional bottom row : two-photon τ_2ℏω + amplitude ----
    if has_tp:
        x_tp = two_photon["E_eV"]
        if xaxis == "k":
            x_tp = np.sqrt(2.0 * two_photon["E_au"])
            xl_tp = r"Momentum $k$ (a.u.)"
        elif xaxis == "omega":
            xl_tp = r"Kinetic energy $E_\kappa$ (eV)"
        else:
            xl_tp = xlabel

        a = ax[tp_row][0]
        a.plot(x_tp, two_photon["tau_as"], "o-", color="C3", lw=2, ms=8,
               label=r"$\tau_{2\hbar\omega}$ (BW17 Eq. 21)")
        a.axhline(0, color="gray", lw=0.5)
        a.set_xlabel(xl_tp); a.set_ylabel(r"$\tau_{2\hbar\omega}$ (as)")
        a.legend(); a.grid(True, alpha=0.3)
        a.set_title(r"Effective two-photon delay (≈ $\tau_{mol}$ for Z=0)")

        a = ax[tp_row][1]
        a.plot(x_tp, two_photon["M_amp"], "s-", color="C4", lw=2, ms=8,
               label=r"$|\langle M^{(2q-1)*} M^{(2q+1)}\rangle|$")
        a.set_xlabel(xl_tp)
        a.set_ylabel(r"$|\langle M_<^*\,M_>\rangle|$ (a.u.)")
        a.legend(); a.grid(True, alpha=0.3)
        a.set_title("Two-photon amplitude (resonance check)")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(xvals, xlabel, Q, phi, idx_start: int, path: Path) -> None:
    sl = slice(idx_start, None)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    colors = {"x": "C0", "y": "C1", "z": "C2", "avg": "k"}

    a = ax[0]
    for d in ("x", "y", "z", "avg"):
        a.plot(xvals[sl], Q[d][sl], color=colors[d],
               lw=2 if d == "avg" else 1.2,
               label=(r"$\bar{Q}_\sigma$" if d == "avg" else rf"$Q_{{\sigma,{d}}}$"))
    a.axhline(1.0, color="r", ls="--", lw=1.5, label=r"$Q_\sigma=1$")
    a.set_xlabel(xlabel); a.set_ylabel(r"$Q_\sigma(E)$")
    a.legend(); a.grid(True, alpha=0.3)
    a.set_title(r"Gauge diagnostic: $Q_\sigma=\sum|d^V|^2 / [\omega^2\sum|d^L|^2]$")

    a = ax[1]
    for d in ("x", "y", "z", "avg"):
        a.plot(xvals[sl], phi[d][sl], color=colors[d],
               lw=2 if d == "avg" else 1.2,
               label=(r"$\overline{\Delta\phi}_{\rm rms}$" if d == "avg"
                      else rf"$\Delta\phi_{{\rm rms,{d}}}$"))
    a.axhline(0.0, color="r", ls="--", lw=1.5)
    a.set_xlabel(xlabel); a.set_ylabel(r"$\Delta\phi_{\rm rms}$ (rad)")
    a.legend(); a.grid(True, alpha=0.3)
    a.set_title("Phase consistency")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gauge_overlay(xvals, xlabel,
                       sigma_L, tau_L, sigma_V, tau_V,
                       idx_start: int, path: Path) -> None:
    sl = slice(idx_start, None)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    a = ax[0]
    a.plot(xvals[sl], sigma_L["avg"][sl] * BOHR2_TO_MB, "b-",  lw=2, label=r"$\bar{\sigma}$ Length")
    a.plot(xvals[sl], sigma_V["avg"][sl] * BOHR2_TO_MB, "r--", lw=2, label=r"$\bar{\sigma}$ Velocity")
    a.set_xlabel(xlabel); a.set_ylabel(r"$\sigma$ (Mb)")
    a.set_yscale("log"); a.legend(); a.grid(True, alpha=0.3)
    a.set_title("Orientation-averaged cross section")

    a = ax[1]
    a.plot(xvals[sl], tau_L["avg"][sl] * AU_TO_AS, "b-",  lw=2, label=r"$\bar{\tau}$ Length")
    a.plot(xvals[sl], tau_V["avg"][sl] * AU_TO_AS, "r--", lw=2, label=r"$\bar{\tau}$ Velocity")
    a.set_xlabel(xlabel); a.set_ylabel(r"$\tau$ (as)")
    a.legend(); a.grid(True, alpha=0.3)
    a.set_title("Orientation-averaged time delay")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------- main -----------------------------------
def resolve_input_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_absolute() and p.exists(): return p
    if p.exists():                     return p.resolve()
    work = os.environ.get("WORK")
    if work:
        q = Path(work) / arg
        if q.exists(): return q
    raise FileNotFoundError(f"input dir not found: {arg}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute cross sections and Wigner time delays from gathered dipole .dat files.")
    ap.add_argument("input_dir", help="Directory written by gather_dipoles.py (gathered_*).")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write delay_xsec_*.dat and .png plots. "
                         "Default: same as input_dir.")
    ap.add_argument("--xaxis", choices=["k", "E", "omega"], default="k")
    ap.add_argument("--idx-start", type=int, default=2,
                    help="Drop the first N energy points from plots/summary "
                         "to avoid gradient boundary artifacts (default 2).")
    ap.add_argument("--with-minus-sign", action="store_true",
                    help="Apply a leading minus to the time delay: "
                         "tau = -Im[sum d* dd/dE]/sum|d|^2 (legacy convention). "
                         "Default is the no-minus tau = +d arg(D)/dE, "
                         "consistent with rtd_content.py / gauge_panels.py.")
    ap.add_argument("--with-two-photon", action="store_true",
                    help="If set, look for two_photon_delay.dat in --output-dir "
                         "(written by postprocessing/two_photon_delay.py) and "
                         "add a 3rd row to the main plot showing τ_2ℏω(E_κ) "
                         "and the two-photon amplitude |<M_<* M_>>|.")
    ap.add_argument("--two-photon-dat", type=Path, default=None,
                    help="Override path to two_photon_delay.dat.  Defaults to "
                         "<output_dir>/two_photon_delay.dat.  Only used when "
                         "--with-two-photon is set.")
    args = ap.parse_args()

    input_dir = resolve_input_dir(args.input_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("cross_section_delay.py")
    print("=" * 72)
    print(f"  input_dir : {input_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  xaxis     : {args.xaxis}")
    print()

    mu_list = infer_common_mu(input_dir,
                              (DIPOLE_PREFIX_LEN, DIPOLE_PREFIX_VEL),
                              ("x", "y", "z"))
    print(f"  channels  : {len(mu_list)}   mu range [{mu_list[0]}, {mu_list[-1]}]")

    # Load all six (gauge, polarization) data sets, check grids agree.
    D_L: Dict[str, np.ndarray] = {}
    D_V: Dict[str, np.ndarray] = {}
    k = E_kin = omega = None
    for d in ("x", "y", "z"):
        k2, e2, o2, D_L[d] = load_dipole(input_dir, DIPOLE_PREFIX_LEN, d, mu_list)
        k,  E_kin, omega  = (k2, e2, o2) if k is None else (k, E_kin, omega)
        if not (np.allclose(k, k2) and np.allclose(E_kin, e2) and np.allclose(omega, o2)):
            raise ValueError(f"grid mismatch: length/{d}")
    for d in ("x", "y", "z"):
        k2, e2, o2, D_V[d] = load_dipole(input_dir, DIPOLE_PREFIX_VEL, d, mu_list)
        if not (np.allclose(k, k2) and np.allclose(E_kin, e2) and np.allclose(omega, o2)):
            raise ValueError(f"grid mismatch: velocity/{d}")

    if len(k) < 3:
        raise ValueError("need >= 3 energy points for stable numerical derivatives")
    print(f"  energies  : {len(k)}   k range [{k[0]:.4f}, {k[-1]:.4f}] au")
    print()

    # ----- sigma + raw strength -----
    sigma_L, sigma_V, raw_L, raw_V = {}, {}, {}, {}
    for d in ("x", "y", "z"):
        sigma_L[d] = sigma_from_dipole(k, D_L[d], omega, "length")
        sigma_V[d] = sigma_from_dipole(k, D_V[d], omega, "velocity")
        raw_L[d]   = raw_strength(D_L[d])
        raw_V[d]   = raw_strength(D_V[d])
    for bag in (sigma_L, sigma_V, raw_L, raw_V):
        bag["avg"] = (bag["x"] + bag["y"] + bag["z"]) / 3.0

    # ----- tau -----
    tau_L, tau_V = {}, {}
    for d in ("x", "y", "z"):
        tau_L[d] = time_delay(k, D_L[d], with_minus=args.with_minus_sign)
        tau_V[d] = time_delay(k, D_V[d], with_minus=args.with_minus_sign)
    # sigma-weighted average of tau (more physical than plain mean).
    for gauge, (sig, tau) in (("L", (sigma_L, tau_L)), ("V", (sigma_V, tau_V))):
        num = sum(sig[d] * tau[d] for d in ("x", "y", "z"))
        den = sum(sig[d]           for d in ("x", "y", "z"))
        tau["avg"] = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)

    # ----- gauge diagnostics -----
    Q, dphi = {}, {}
    for d in ("x", "y", "z"):
        Q[d]    = q_sigma(D_L[d], D_V[d], omega)
        dphi[d] = dphi_rms(D_L[d], D_V[d])
    D_L_all = np.concatenate([D_L[d] for d in ("x", "y", "z")], axis=1)
    D_V_all = np.concatenate([D_V[d] for d in ("x", "y", "z")], axis=1)
    Q["avg"]    = q_sigma(D_L_all, D_V_all, omega)
    dphi["avg"] = dphi_rms(D_L_all, D_V_all)

    # ----- asymmetry parameter β(ω), both gauges -----
    print("  computing β(ω) by numerical angle integration...")
    beta_L = compute_beta(D_L["x"], D_L["y"], D_L["z"], mu_list)
    beta_V = compute_beta(D_V["x"], D_V["y"], D_V["z"], mu_list)

    # ----- write tables -----
    def write_delay_xsec(path: Path, gauge_label: str, sigma, tau, raw_strength_bag):
        tau_sign = "-" if args.with_minus_sign else "+"
        hdr = (
            f"{gauge_label} gauge  |  IP = {-(-omega[0] + E_kin[0]):.6f} au\n"
            "sigma = 8pi * (omega | 1/omega) / (c * k) * sum|d_lm|^2  [bohr^2]\n"
            f"tau   = {tau_sign}Im[sum d* dd/dE] / sum|d|^2  [a.u. of time]"
            f"   ({'legacy -Im (--with-minus-sign)' if args.with_minus_sign else 'default +d argD/dE'})\n"
            "sigma_raw = sum|d_lm(E)|^2\n"
            "Columns: k  E_kin  omega  "
            "tau_x tau_y tau_z tau_avg  "
            "sigma_x sigma_y sigma_z sigma_avg  "
            "sigma_raw_x sigma_raw_y sigma_raw_z sigma_raw_avg"
        )
        stack = [k, E_kin, omega,
                 tau["x"], tau["y"], tau["z"], tau["avg"],
                 sigma["x"], sigma["y"], sigma["z"], sigma["avg"],
                 raw_strength_bag["x"], raw_strength_bag["y"],
                 raw_strength_bag["z"], raw_strength_bag["avg"]]
        np.savetxt(path, np.column_stack(stack), header=hdr, fmt="%.10e")

    out_L = output_dir / "delay_xsec_len_homo.dat"
    out_V = output_dir / "delay_xsec_vel_homo.dat"
    write_delay_xsec(out_L, "LENGTH", sigma_L, tau_L, raw_L)
    write_delay_xsec(out_V, "VELOCITY", sigma_V, tau_V, raw_V)

    out_diag = output_dir / "gauge_diagnostics.dat"
    hdr_diag = (
        "Gauge consistency diagnostics\n"
        "Q_sigma  = sum|d^V|^2 / (omega^2 * sum|d^L|^2)   (= 1 for exact H)\n"
        "dphi_rms = weighted RMS of arg(d^V / d^L) in rad (= 0 for exact H)\n"
        "Columns: k  E_kin  omega  Q_x Q_y Q_z Q_avg  dphi_x dphi_y dphi_z dphi_avg"
    )
    np.savetxt(out_diag, np.column_stack([
        k, E_kin, omega,
        Q["x"], Q["y"], Q["z"], Q["avg"],
        dphi["x"], dphi["y"], dphi["z"], dphi["avg"],
    ]), header=hdr_diag, fmt="%.10e")

    # β(ω) for both gauges
    out_beta = output_dir / "beta_asymmetry.dat"
    hdr_beta = (
        "Asymmetry parameter β(ω) for photoelectron angular distribution,\n"
        "orientation-averaged over random molecular orientation with linearly\n"
        "polarized light.  β ∈ [-1, 2]; atomic s→p limit is β=2; isotropic β=0.\n"
        "Computed by direct numerical angle integration (40 θ × 80 φ) of\n"
        "  β = 5 · ⟨P_2⟩ / σ\n"
        "over the three Cartesian polarization directions.\n"
        "Columns: k  E_kin  omega  beta_L  beta_V"
    )
    np.savetxt(out_beta, np.column_stack([
        k, E_kin, omega, beta_L, beta_V,
    ]), header=hdr_beta, fmt="%.10e")

    # ----- two-photon delay (optional) -----
    two_photon = None
    if args.with_two_photon:
        tp_path = (args.two_photon_dat
                   if args.two_photon_dat is not None
                   else output_dir / "two_photon_delay.dat")
        two_photon = load_two_photon_delay(tp_path)
        if two_photon is None:
            print(f"  WARN: --with-two-photon set but {tp_path} not found; "
                  f"plot will skip the 3rd row.\n"
                  f"  Run postprocessing/two_photon_delay.py first.")
        else:
            print(f"  two-photon : loaded {len(two_photon['ik'])} sidebands "
                  f"from {tp_path}")

    # ----- plots -----
    xvals, xlabel = get_xvals(k, E_kin, omega, args.xaxis)
    plot_two_panel_per_gauge(xvals, xlabel, sigma_L, tau_L, sigma_V, tau_V,
                             args.idx_start, output_dir / "cross_section_delay_both.png",
                             two_photon=two_photon, xaxis=args.xaxis,
                             beta_L=beta_L, beta_V=beta_V)
    plot_diagnostics(xvals, xlabel, Q, dphi, args.idx_start,
                     output_dir / "gauge_diagnostics.png")
    plot_gauge_overlay(xvals, xlabel, sigma_L, tau_L, sigma_V, tau_V,
                       args.idx_start, output_dir / "gauge_overlay.png")

    # β(ω) plot
    fig_b, ax_b = plt.subplots(figsize=(10, 4.2))
    sl_b = slice(args.idx_start, None)
    ax_b.plot(xvals[sl_b], beta_L[sl_b], "b-",  lw=2, label=r"$\beta^L$")
    ax_b.plot(xvals[sl_b], beta_V[sl_b], "r--", lw=2, label=r"$\beta^V$")
    ##ax_b.axhline( 2.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax_b.axhline( 0.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ##ax_b.axhline(-1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax_b.set_xlabel(xlabel); ax_b.set_ylabel(r"$\beta(\omega)$")
    ax_b.set_title("Asymmetry parameter (orientation-averaged)")
    ######ax_b.set_ylim(-1.3, 2.3)
    ax_b.legend(); ax_b.grid(True, alpha=0.3)
    fig_b.tight_layout()
    fig_b.savefig(output_dir / "beta_asymmetry.png", dpi=150, bbox_inches="tight")
    plt.close(fig_b)

    # ----- summary -----
    sl = slice(args.idx_start, None)
    print("Summary")
    print("-" * 72)
    for d in ("x", "y", "z", "avg"):
        print(f"  sigma_L_{d:>3s}: max = {np.max(sigma_L[d][sl]) * BOHR2_TO_MB:.4e} Mb")
    for d in ("x", "y", "z", "avg"):
        print(f"  sigma_V_{d:>3s}: max = {np.max(sigma_V[d][sl]) * BOHR2_TO_MB:.4e} Mb")
    for d in ("x", "y", "z", "avg"):
        t = tau_L[d][sl] * AU_TO_AS
        print(f"  tau_L_{d:>3s}  : [{t.min():.2f}, {t.max():.2f}] as")
    for d in ("x", "y", "z", "avg"):
        t = tau_V[d][sl] * AU_TO_AS
        print(f"  tau_V_{d:>3s}  : [{t.min():.2f}, {t.max():.2f}] as")
    for d in ("x", "y", "z", "avg"):
        q = Q[d][sl]
        q = q[np.isfinite(q)]
        if len(q):
            print(f"  Q_sigma_{d:>3s}: mean = {q.mean():.4f}, "
                  f"range [{q.min():.3f}, {q.max():.3f}]")
    for d in ("x", "y", "z", "avg"):
        print(f"  dphi_rms_{d:>3s}: mean = {dphi[d][sl].mean():.4f} rad, "
              f"max = {dphi[d][sl].max():.4f} rad")
    print(f"  beta_L     : [{beta_L[sl].min():.3f}, {beta_L[sl].max():.3f}]  "
          f"mean = {beta_L[sl].mean():.3f}")
    print(f"  beta_V     : [{beta_V[sl].min():.3f}, {beta_V[sl].max():.3f}]  "
          f"mean = {beta_V[sl].mean():.3f}")

    print("\nSaved:")
    for p in (out_L, out_V, out_diag, out_beta,
              output_dir / "cross_section_delay_both.png",
              output_dir / "gauge_diagnostics.png",
              output_dir / "gauge_overlay.png",
              output_dir / "beta_asymmetry.png"):
        print(f"  {p}")
    print("\nDone.")


if __name__ == "__main__":
    main()
