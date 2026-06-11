#!/usr/bin/env python3
"""
plot_cube_planes.py
===================
Read a Gaussian .cube file and render three 2-D panels side-by-side:
the xy plane (perpendicular to ẑ), the xz plane (perp ŷ), and the yz
plane (perp x̂).

Two reduction modes:
  --mode integrate  (default)
        2-D map = ∫ f(x, y, z) d(perp axis) · d(perp).  Reveals the
        global structure summed along the line of sight.
  --mode slice
        2-D map = f at z=0 (or y=0, x=0).  Reveals the central slice
        — sharpest features for symmetric data.

Color scaling:  AsinhNorm (signed, smooth log-like), same data-driven
linthresh as the polar plots (--lo-pct of |f|).  Each row of three
panels shares vlim and linthresh so colors are directly comparable.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
try:
    from matplotlib.colors import AsinhNorm
    _HAS_ASINH = True
except Exception:
    _HAS_ASINH = False
import numpy as np


# --------------------------- I/O ---------------------------------
def load_cube(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, str]:
    """Parse a Gaussian .cube and return (x, y, z, data, title).

    Assumes:
      * Diagonal voxel basis (cardinal axes only — what polar_fft.py writes).
      * Atomic-units cube (positive Nvoxels per axis).
    """
    title_lines = []
    with open(path) as fh:
        # First two lines: title / description.
        title_lines.append(fh.readline().rstrip("\n"))
        title_lines.append(fh.readline().rstrip("\n"))

        toks = fh.readline().split()
        natoms = int(toks[0])
        origin = np.array([float(toks[1]), float(toks[2]), float(toks[3])])
        # Sign of `natoms` (negative = data block has an extra header line).
        has_value_header = natoms < 0
        natoms = abs(natoms)

        voxel = np.zeros((3, 4))
        for i in range(3):
            t = fh.readline().split()
            voxel[i] = [int(t[0]), float(t[1]), float(t[2]), float(t[3])]
        Nx, Ny, Nz = int(voxel[0, 0]), int(voxel[1, 0]), int(voxel[2, 0])
        # Diagonal-basis check.
        if (abs(voxel[0, 2]) + abs(voxel[0, 3])
            + abs(voxel[1, 1]) + abs(voxel[1, 3])
            + abs(voxel[2, 1]) + abs(voxel[2, 2])) > 1e-10:
            raise ValueError(
                f"{path}: non-diagonal voxel basis not supported "
                f"by this reader.")
        dx, dy, dz = voxel[0, 1], voxel[1, 2], voxel[2, 3]

        # Skip the per-atom lines and the optional value-count header.
        for _ in range(natoms):
            fh.readline()
        if has_value_header:
            fh.readline()

        # All remaining tokens are floats.
        flat = np.fromstring(fh.read(), sep=" ")

    expected = Nx * Ny * Nz
    if flat.size != expected:
        raise ValueError(
            f"{path}: read {flat.size} values, expected Nx·Ny·Nz = {expected}")

    # Innermost loop in cube = z.  Shape (Nx, Ny, Nz) in C-order.
    data = flat.reshape(Nx, Ny, Nz)
    x = origin[0] + dx * np.arange(Nx)
    y = origin[1] + dy * np.arange(Ny)
    z = origin[2] + dz * np.arange(Nz)
    return x, y, z, data, " | ".join(title_lines)


# --------------------------- reductions --------------------------
def reduce_planes(data: np.ndarray,
                   x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xy_map, xz_map, yz_map).  Each is a 2-D array on the
    (first × second) axis pair indicated in the name.
    """
    if mode == "integrate":
        dx = x[1] - x[0]; dy = y[1] - y[0]; dz = z[1] - z[0]
        xy = data.sum(axis=2) * dz   # ∫ f dz  → on the (x, y) grid
        xz = data.sum(axis=1) * dy   # ∫ f dy  → on the (x, z) grid
        yz = data.sum(axis=0) * dx   # ∫ f dx  → on the (y, z) grid
    elif mode == "slice":
        # Mid-index ≈ 0 for even-N centered cubes (rx[N/2] = 0).
        kx = np.argmin(np.abs(x))
        ky = np.argmin(np.abs(y))
        kz = np.argmin(np.abs(z))
        xy = data[:, :, kz]
        xz = data[:, ky, :]
        yz = data[kx, :, :]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return xy, xz, yz


# --------------------------- color scaling -----------------------
def autoscale_signed(arr: np.ndarray, hi_pct: float = 99.0,
                     lo_pct: float = 1.0) -> Tuple[float, float]:
    a = np.abs(arr[np.isfinite(arr) & (arr != 0)])
    if a.size == 0:
        return 1.0, 1e-3
    vlim = float(np.percentile(a, hi_pct))
    lin  = float(np.percentile(a, lo_pct))
    if not np.isfinite(lin) or lin <= 0:
        lin = max(vlim * 1e-8, 1e-30)
    lin = min(lin, vlim * 0.5)
    return vlim, lin


def make_norm(vlim: float, linthresh: float, kind: str, linscale: float):
    if kind == "asinh" and _HAS_ASINH:
        return AsinhNorm(linear_width=linthresh, vmin=-vlim, vmax=+vlim)
    if kind == "log":
        return LogNorm(vmin=max(linthresh, 1e-30), vmax=vlim)
    return SymLogNorm(linthresh=linthresh, linscale=linscale,
                      vmin=-vlim, vmax=+vlim, base=10)


# --------------------------- plot --------------------------------
def plot_three_planes(x, y, z, data, *, mode: str, norm_kind: str,
                       lo_pct: float, linscale: float, cmap: str,
                       output: Path, title: str):
    xy, xz, yz = reduce_planes(data, x, y, z, mode)

    # Joint color scale across the three panels for direct comparability.
    joint = np.concatenate([xy.ravel(), xz.ravel(), yz.ravel()])
    vlim, linthresh = autoscale_signed(joint, lo_pct=lo_pct)
    norm = make_norm(vlim, linthresh, norm_kind, linscale)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    extents = [
        (x[0], x[-1], y[0], y[-1]),   # xy
        (x[0], x[-1], z[0], z[-1]),   # xz
        (y[0], y[-1], z[0], z[-1]),   # yz
    ]
    panels = [
        ("xy plane" + (r"  $\int f\,dz$" if mode == "integrate" else r"  $f(x,y,0)$"),
         xy.T, "x [bohr]", "y [bohr]"),
        ("xz plane" + (r"  $\int f\,dy$" if mode == "integrate" else r"  $f(x,0,z)$"),
         xz.T, "x [bohr]", "z [bohr]"),
        ("yz plane" + (r"  $\int f\,dx$" if mode == "integrate" else r"  $f(0,y,z)$"),
         yz.T, "y [bohr]", "z [bohr]"),
    ]
    for ax, (pname, pdata, xl, yl), ext in zip(axes, panels, extents):
        im = ax.imshow(pdata, origin="lower", extent=ext,
                       cmap=cmap, norm=norm, aspect="equal",
                       interpolation="nearest")
        ax.set_title(pname, fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.tick_params(labelsize=8)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.axvline(0, color="k", lw=0.4, alpha=0.4)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)

    fig.suptitle(f"{title}\n"
                 f"mode={mode}   norm={norm_kind}   vlim=±{vlim:.3e}   "
                 f"linthresh={linthresh:.3e}   lo_pct={lo_pct:g}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output}")


# --------------------------- main --------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True, type=Path,
        help="Input .cube file.")
    ap.add_argument("--output", required=True, type=Path,
        help="Output image (PNG / PDF / etc.).")
    ap.add_argument("--mode", choices=("integrate", "slice"),
        default="integrate",
        help="2-D reduction: 'integrate' (∫ along the perpendicular axis "
             "— default) or 'slice' (central slice f(x,y,0), etc.).")
    ap.add_argument("--norm", choices=("asinh", "symlog", "log"),
        default="asinh",
        help="Color-scale: asinh (default, signed-log without white band), "
             "symlog (classic linear-near-zero + log), or log (positive "
             "data only).")
    ap.add_argument("--lo-pct", type=float, default=1.0,
        help="Percentile of |f| for the near-zero transition (default 1).")
    ap.add_argument("--linscale", type=float, default=0.5,
        help="(symlog only) colormap-budget share for the linear band.")
    ap.add_argument("--cmap", default="seismic",
        help="Matplotlib colormap (default: seismic).")
    args = ap.parse_args()

    print("=" * 72)
    print(f" plot_cube_planes.py   {args.cube}")
    print("=" * 72)
    x, y, z, data, title = load_cube(args.cube)
    print(f"  shape        : {data.shape}    "
          f"range x=[{x[0]:.3f},{x[-1]:.3f}]  "
          f"y=[{y[0]:.3f},{y[-1]:.3f}]  z=[{z[0]:.3f},{z[-1]:.3f}]")
    print(f"  cube min/max : {data.min():.3e} / {data.max():.3e}  "
          f"|max|={np.abs(data).max():.3e}")
    plot_three_planes(x, y, z, data,
                       mode=args.mode, norm_kind=args.norm,
                       lo_pct=args.lo_pct, linscale=args.linscale,
                       cmap=args.cmap, output=args.output,
                       title=f"{args.cube.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
