#!/usr/bin/env python
"""
snapshot_tb_profiles.py
-----------------------
For each sampled MCS system, create a combined figure showing:
  - Left  : Tb spatial snapshot (from YYYYMMDD_TbExpand.nc)
             with the system footprint overlaid
  - Right : vertical profiles of latheating and eddymoistening
             per GPM category (convective / non-convective / no-rain)
             (from YYYYMMDD.gpm_select-toocan-subset.nc)

Usage
-----
  python snapshot_tb_profiles.py                  # uses defaults below
  python snapshot_tb_profiles.py 20140320         # specific date
  python snapshot_tb_profiles.py 20140320 --n 3   # first 3 systems that day
"""

import argparse
import os
import sys
import glob
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
TOOCAN_DIR  = "/glade/work/addisus/GPM_TOOCAN_data/gpm_select_toocan"
TBEXPAND_DIR = "/glade/work/addisus/GPM_TOOCAN_data/gpm_select_TbExpand"
ERA5_DIR    = "/glade/work/addisus/GPM_TOOCAN_data/era5_select"
OUTPUT_DIR  = "/glade/work/addisus/GPM_TOOCAN_data/plots/snapshots"

# Number of systems to plot (per date)
N_SYSTEMS_DEFAULT = 3

# Extra margin around the system bbox (grid boxes)
EXTRA_MARGIN = 5   # on top of the 20-box belt already baked into TbExpand

# nc1 category colors and labels
CAT_COLORS = {0: "royalblue", 1: "tomato", 2: "mediumseagreen"}
CAT_LABELS = {0: "GPM_cnv", 1: "GPM_noncnv", 2: "GPM_norain"}

# Missing-value threshold
MISSING_THRESH = -1000.0


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def clean(arr):
    """Replace -9999 / extreme-negative values with NaN in a numpy array."""
    out = arr.astype(float)
    out[out < MISSING_THRESH] = np.nan
    return out


def find_file_pair(date_str):
    """Return (toocan_path, tbexpand_path) for YYYYMMDD, or raise."""
    toocan_pattern  = os.path.join(TOOCAN_DIR,   f"{date_str}.gpm_select-toocan-subset.nc")
    tbexpand_pattern = os.path.join(TBEXPAND_DIR, f"{date_str}_TbExpand.nc")

    if not os.path.isfile(toocan_pattern):
        raise FileNotFoundError(f"TOOCAN file not found: {toocan_pattern}")
    if not os.path.isfile(tbexpand_pattern):
        raise FileNotFoundError(f"TbExpand file not found: {tbexpand_pattern}")

    return toocan_pattern, tbexpand_pattern


def find_era5_file(date_str):
    """Return ERA5 path for YYYYMMDD, or None if not found."""
    path = os.path.join(ERA5_DIR,
                        f"{date_str}.era5_select-toocan-subset_pm6hrsSearch.nc")
    return path if os.path.isfile(path) else None


def list_available_dates():
    """Return sorted list of YYYYMMDD strings that have both file types."""
    toocan_files = glob.glob(os.path.join(TOOCAN_DIR, "*.gpm_select-toocan-subset.nc"))
    dates = []
    for f in sorted(toocan_files):
        m = re.search(r"(\d{8})\.", os.path.basename(f))
        if m:
            d = m.group(1)
            tb_f = os.path.join(TBEXPAND_DIR, f"{d}_TbExpand.nc")
            if os.path.isfile(tb_f):
                dates.append(d)
    return dates


def get_systems(toocan_ds):
    """
    Return list of (ti, si) index pairs where DCS_number != 0,
    sorted by DCS_number (arbitrary but reproducible).
    """
    dcs = toocan_ds["DCS_number"].values   # (nt, nsysmax)
    idx = list(zip(*np.argwhere(dcs != 0).T))  # list of (ti, si)
    # de-duplicate: keep first occurrence of each DCS_number
    seen = {}
    unique = []
    for ti, si in idx:
        d = int(dcs[ti, si])
        if d not in seen:
            seen[d] = (ti, si)
            unique.append((ti, si, d))
    # sort by DCS_number for reproducibility
    unique.sort(key=lambda x: x[2])
    return unique  # list of (ti, si, dcs_number)


# -----------------------------------------------------------------------
# Core plotting function
# -----------------------------------------------------------------------

def make_snapshot(date_str, ti, si, dcs_number,
                  toocan_ds, tb_ds,
                  out_dir, era5_ds=None):
    """
    Create and save one snapshot figure for a single system.

    Parameters
    ----------
    date_str   : YYYYMMDD string
    ti, si     : time and system indices in toocan_ds
    dcs_number : integer DCS_number
    toocan_ds  : opened TOOCAN subset Dataset
    tb_ds      : opened TbExpand Dataset (lazy / chunked)
    out_dir    : output directory
    """

    # ---- 1. Extract spatial Tb snapshot ---------------------------------
    # Load DCS mask at time step ti (one 2D slice ~ 72 MB for float64)
    dcs_spatial = tb_ds["DCS_number"].isel(nt=ti).load().values   # (nlat, nlon)
    tb_spatial  = tb_ds["Tb"].isel(nt=ti).load().values           # (nlat, nlon)
    lat_arr     = tb_ds["lat"].values   # (nlat,)
    lon_arr     = tb_ds["lon"].values   # (nlon,)

    # Find pixels belonging to this system
    sys_mask = (dcs_spatial == dcs_number)
    sys_rows, sys_cols = np.where(sys_mask)

    if len(sys_rows) == 0:
        print(f"  Warning: DCS {dcs_number} not found in TbExpand at ti={ti}; skipping.")
        return

    # Bounding box with extra margin
    r0 = max(sys_rows.min() - EXTRA_MARGIN, 0)
    r1 = min(sys_rows.max() + EXTRA_MARGIN + 1, len(lat_arr))
    c0 = max(sys_cols.min() - EXTRA_MARGIN, 0)
    c1 = min(sys_cols.max() + EXTRA_MARGIN + 1, len(lon_arr))

    tb_sub    = tb_spatial[r0:r1, c0:c1]
    mask_sub  = sys_mask[r0:r1, c0:c1]
    lat_sub   = lat_arr[r0:r1]
    lon_sub   = lon_arr[c0:c1]

    # ---- 2. Extract vertical profiles -----------------------------------
    heights = toocan_ds["nlvl"].values            # (nlvl,) in km
    ncat    = toocan_ds.sizes["nc1"]

    lath_profiles  = {}   # cat_idx -> 1-D array (nlvl,)
    eddy_profiles  = {}

    for cat in range(ncat):
        lath = toocan_ds["latheating"].values[cat, ti, si, :]   # (nlvl,)
        eddy = toocan_ds["eddymoistening"].values[cat, ti, si, :]

        lath_profiles[cat] = clean(lath)
        eddy_profiles[cat] = clean(eddy)

    # ---- 2b. Extract ERA5 T / q profiles --------------------------------
    era5_tprof = era5_qprof = era5_plev = None
    era5_t_p25 = era5_t_p75 = None
    era5_q_p25 = era5_q_p75 = None

    if era5_ds is not None:
        # Select the ERA5 time with the smallest |deltatime| at this ti
        dt = era5_ds["deltatime"].values[:, ti]   # (nc2,)
        abs_dt = np.where(np.isfinite(dt), np.abs(dt), np.inf)
        best_nc2 = int(np.argmin(abs_dt))

        # tprof / qprof / plev: (nc2, nc1, nt, nsysmax, nlvl)
        # nc1=0 mean, nc1=1 25th, nc1=3 75th percentile
        era5_tprof = clean(era5_ds["tprof"].values[best_nc2, 0, ti, si, :])
        era5_t_p25 = clean(era5_ds["tprof"].values[best_nc2, 1, ti, si, :])
        era5_t_p75 = clean(era5_ds["tprof"].values[best_nc2, 3, ti, si, :])
        era5_qprof = clean(era5_ds["qprof"].values[best_nc2, 0, ti, si, :])
        era5_q_p25 = clean(era5_ds["qprof"].values[best_nc2, 1, ti, si, :])
        era5_q_p75 = clean(era5_ds["qprof"].values[best_nc2, 3, ti, si, :])
        era5_plev  = clean(era5_ds["plev"].values[best_nc2, 0, ti, si, :])

    # System info
    lc_lon = float(toocan_ds["lc_lon"].values[ti, si])
    lc_lat = float(toocan_ds["lc_lat"].values[ti, si])
    time_h = float(toocan_ds["time"].values[ti])

    # ---- 3. Build figure ------------------------------------------------
    fig = plt.figure(figsize=(20, 8))
    gs  = GridSpec(2, 4, figure=fig,
                   width_ratios=[2.2, 1, 1, 1],
                   hspace=0.35, wspace=0.40)

    ax_tb     = fig.add_subplot(gs[:, 0])   # Tb snapshot spans both rows
    ax_lath   = fig.add_subplot(gs[0, 1])   # latheating profile
    ax_eddy   = fig.add_subplot(gs[1, 1])   # eddymoistening profile
    ax_T      = fig.add_subplot(gs[0, 2])   # ERA5 temperature profile
    ax_q      = fig.add_subplot(gs[1, 2])   # ERA5 moisture profile
    ax_legend = fig.add_subplot(gs[:, 3])   # legend / info panel

    # --- Tb map ---
    # IR colormap: cold (deep convection) = dark purple/blue; warm = white/yellow
    tb_cmap  = plt.cm.RdYlBu_r
    tb_vmin, tb_vmax = 180, 310

    LON2D, LAT2D = np.meshgrid(lon_sub, lat_sub)
    im = ax_tb.pcolormesh(LON2D, LAT2D, tb_sub,
                          cmap=tb_cmap, vmin=tb_vmin, vmax=tb_vmax,
                          shading="nearest")
    plt.colorbar(im, ax=ax_tb, label="Tb (K)", fraction=0.046, pad=0.04)

    # Overlay system footprint as contour
    if mask_sub.any():
        ax_tb.contour(LON2D, LAT2D, mask_sub.astype(int),
                      levels=[0.5], colors="black", linewidths=1.5,
                      linestyles="--", zorder=5)

    # System center
    ax_tb.plot(lc_lon, lc_lat, marker="*", color="white",
               markersize=14, markeredgecolor="black", markeredgewidth=1,
               zorder=6, label="System centre")

    ax_tb.set_xlabel("Longitude (°E)", fontsize=12)
    ax_tb.set_ylabel("Latitude (°N)", fontsize=12)
    ax_tb.set_title(f"Tb – {date_str}  {time_h:.1f} UTC h\n"
                    f"DCS #{dcs_number}  "
                    f"({lc_lon:.2f}°E, {lc_lat:.2f}°N)",
                    fontsize=11)
    ax_tb.tick_params(labelsize=10)

    # --- Profile panels ---
    profile_info = [
        (ax_lath, lath_profiles, "Latent Heating (K hr⁻¹)", "latheating"),
        (ax_eddy, eddy_profiles, "Eddy Moistening (g kg⁻¹ day⁻¹)", "eddymoistening"),
    ]

    for ax, profiles, xlabel, varname in profile_info:
        for cat in range(ncat):
            prof = profiles[cat]
            valid = ~np.isnan(prof)
            if valid.any():
                ax.plot(prof, heights, color=CAT_COLORS[cat],
                        linewidth=2, label=CAT_LABELS[cat],
                        marker="o", markersize=3)

        ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Height (km)", fontsize=10)
        ax.set_title(varname, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)

    # --- ERA5 T and q profiles ---
    for ax, mean_prof, p25, p75, xlabel, title in [
        (ax_T, era5_tprof, era5_t_p25, era5_t_p75,
         "Temperature (K)",           "ERA5 Temperature"),
        (ax_q, era5_qprof, era5_q_p25, era5_q_p75,
         "Water Vapor (g kg⁻¹)",      "ERA5 Moisture"),
    ]:
        if era5_plev is not None and mean_prof is not None:
            valid = np.isfinite(era5_plev) & np.isfinite(mean_prof)
            if valid.any():
                ax.plot(mean_prof[valid], era5_plev[valid],
                        color="darkorange", linewidth=2,
                        marker="o", markersize=3, label="mean")
                # IQR shading where all three are finite
                iqr_valid = valid & np.isfinite(p25) & np.isfinite(p75)
                if iqr_valid.any():
                    ax.fill_betweenx(era5_plev[iqr_valid],
                                     p25[iqr_valid], p75[iqr_valid],
                                     color="darkorange", alpha=0.2,
                                     label="25–75 %ile")
        else:
            ax.text(0.5, 0.5, "ERA5\nnot available",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")

        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Pressure (hPa)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)
        if era5_plev is not None and mean_prof is not None:
            ax.legend(fontsize=8, loc="lower right")

    # --- Legend / info panel ---
    ax_legend.axis("off")
    legend_patches = [
        mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c])
        for c in range(ncat)
    ]
    legend_patches.append(
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label="System boundary")
    )
    legend_patches.append(
        plt.Line2D([0], [0], marker="*", color="white", linestyle="None",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=12, label="System centre")
    )
    ax_legend.legend(handles=legend_patches, loc="center",
                     fontsize=11, framealpha=0.9, title="Legend",
                     title_fontsize=11)

    info_text = (
        f"Date : {date_str}\n"
        f"Time : {time_h:.1f} UTC h\n"
        f"DCS  : {dcs_number}\n"
        f"Lon  : {lc_lon:.2f}°E\n"
        f"Lat  : {lc_lat:.2f}°N\n"
        f"#pixels : {sys_mask.sum():d}"
    )
    ax_legend.text(0.5, 0.15, info_text,
                   transform=ax_legend.transAxes,
                   ha="center", va="center", fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                             alpha=0.8))

    fig.suptitle(
        f"MCS snapshot: {date_str}  T={time_h:.1f} h UTC  |  DCS #{dcs_number}",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ---- 4. Save --------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    fname = f"snapshot_{date_str}_DCS{dcs_number}_ti{ti:02d}.png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Make Tb + profile snapshots for TOOCAN-matched MCS systems."
    )
    parser.add_argument("date", nargs="?", default=None,
                        help="Date YYYYMMDD (default: first available date).")
    parser.add_argument("--n", type=int, default=N_SYSTEMS_DEFAULT,
                        help=f"Number of systems to plot (default: {N_SYSTEMS_DEFAULT}).")
    parser.add_argument("--outdir", default=OUTPUT_DIR,
                        help="Output directory for PNG files.")
    args = parser.parse_args()

    # Resolve date
    if args.date is None:
        dates = list_available_dates()
        if not dates:
            sys.exit("No matching file pairs found.")
        date_str = dates[0]
        print(f"No date specified; using first available: {date_str}")
    else:
        date_str = args.date

    print(f"\nProcessing date: {date_str}")
    toocan_path, tbexpand_path = find_file_pair(date_str)
    print(f"  TOOCAN  : {toocan_path}")
    print(f"  TbExpand: {tbexpand_path}")

    # Open TOOCAN (small, load fully)
    toocan_ds = xr.open_dataset(toocan_path)

    # Open TbExpand lazily (10 GB; we read one time slice at a time)
    tb_ds = xr.open_dataset(tbexpand_path, chunks={"nt": 1})

    # Open ERA5 if available
    era5_path = find_era5_file(date_str)
    era5_ds = None
    if era5_path:
        print(f"  ERA5    : {era5_path}")
        era5_ds = xr.open_dataset(era5_path)
    else:
        print(f"  ERA5    : not found for {date_str}")

    # Find systems
    systems = get_systems(toocan_ds)
    if not systems:
        sys.exit("No systems (non-zero DCS_number) found in TOOCAN file.")

    n_plot = min(args.n, len(systems))
    print(f"Found {len(systems)} unique systems; plotting first {n_plot}.\n")

    for i, (ti, si, dcs_number) in enumerate(systems[:n_plot]):
        print(f"[{i+1}/{n_plot}] DCS={dcs_number}  ti={ti}  si={si}")
        make_snapshot(
            date_str, ti, si, dcs_number,
            toocan_ds, tb_ds,
            out_dir=args.outdir,
            era5_ds=era5_ds,
        )

    toocan_ds.close()
    tb_ds.close()
    if era5_ds is not None:
        era5_ds.close()

    print(f"\nDone. Plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()
