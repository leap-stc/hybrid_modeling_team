#!/usr/bin/env python
"""
snapshot_tb_compare.py
-----------------------
For N_SYSTEMS (default 5) sampled MCS systems, create ONE combined figure with:
  - Panels 1-5 (3×2 grid, positions 0-4): Tb spatial snapshots with system
    footprint overlaid.
  - Panel 6 (bottom-right): table comparing three convective metrics across
    all 5 systems:
      1. conv_area_frac  = ngrids(GPM_cnv) / ngrids(total)
      2. conv_intensity  = rainrate(GPM_cnv)               [mm hr⁻¹]
      3. z_top           = highest height (km) where latheating(GPM_cnv) > threshold

Usage
-----
  python snapshot_tb_compare.py                  # first available date, 5 systems
  python snapshot_tb_compare.py 20140320         # specific date
  python snapshot_tb_compare.py 20140320 --n 5   # explicit count (max 5)
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
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
TOOCAN_DIR   = "/glade/work/addisus/GPM_TOOCAN_data/gpm_select_toocan"
TBEXPAND_DIR = "/glade/work/addisus/GPM_TOOCAN_data/gpm_select_TbExpand"
OUTPUT_DIR   = "/glade/work/addisus/GPM_TOOCAN_data/plots/snapshots_compare"

N_SYSTEMS_DEFAULT = 5    # panels 1-5; max 5 for the 3×2 layout

# Latent-heating threshold for z_top calculation [K hr⁻¹]
LATH_THRESHOLD = 0.5

# Extra margin around each system bounding box (grid boxes)
EXTRA_MARGIN = 5

# Category index for GPM convective (nc1 axis)
CAT_CNV = 0   # GPM_cnv
CAT_LABELS = {0: "GPM_cnv", 1: "GPM_noncnv", 2: "GPM_norain"}

MISSING_THRESH = -1000.0

# Variable names in the TOOCAN subset file — adjust if they differ
VAR_NGRIDS   = "ngrids"       # shape (nc1, nt, nsysmax)  or (nt, nsysmax, nc1)
VAR_RAINRATE = "rainrate"     # shape (nc1, nt, nsysmax)  or similar
VAR_LATH     = "latheating"   # shape (nc1, nt, nsysmax, nlvl)
VAR_HEIGHTS  = "nlvl"         # 1-D heights in km


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def clean(arr):
    """Replace fill values (< MISSING_THRESH) with NaN."""
    out = arr.astype(float)
    out[out < MISSING_THRESH] = np.nan
    return out


def find_file_pair(date_str):
    toocan_path   = os.path.join(TOOCAN_DIR,   f"{date_str}.gpm_select-toocan-subset.nc")
    tbexpand_path = os.path.join(TBEXPAND_DIR,  f"{date_str}_TbExpand.nc")
    if not os.path.isfile(toocan_path):
        raise FileNotFoundError(f"TOOCAN file not found: {toocan_path}")
    if not os.path.isfile(tbexpand_path):
        raise FileNotFoundError(f"TbExpand file not found: {tbexpand_path}")
    return toocan_path, tbexpand_path


def list_available_dates():
    toocan_files = glob.glob(os.path.join(TOOCAN_DIR, "*.gpm_select-toocan-subset.nc"))
    dates = []
    for f in sorted(toocan_files):
        m = re.search(r"(\d{8})\.", os.path.basename(f))
        if m:
            d = m.group(1)
            if os.path.isfile(os.path.join(TBEXPAND_DIR, f"{d}_TbExpand.nc")):
                dates.append(d)
    return dates


def get_systems(toocan_ds):
    """Return list of (ti, si, dcs_number), de-duplicated and sorted."""
    dcs  = toocan_ds["DCS_number"].values
    idx  = list(zip(*np.argwhere(dcs != 0).T))
    seen, unique = {}, []
    for ti, si in idx:
        d = int(dcs[ti, si])
        if d not in seen:
            seen[d] = (ti, si)
            unique.append((ti, si, d))
    unique.sort(key=lambda x: x[2])
    return unique


# -----------------------------------------------------------------------
# Metric extraction
# -----------------------------------------------------------------------

def compute_metrics(toocan_ds, ti, si, heights):
    """
    Return (conv_area_frac, conv_intensity, z_top) for one system.
    Missing / unavailable values are returned as np.nan.
    """
    ncat = toocan_ds.sizes.get("nc1", 3)

    # --- 1. conv_area_frac -----------------------------------------------
    conv_area_frac = np.nan
    if VAR_NGRIDS in toocan_ds:
        ng = toocan_ds[VAR_NGRIDS].values  # try (nc1, nt, nsysmax)
        try:
            if ng.ndim == 3:
                # assume (nc1, nt, nsysmax)
                grids_per_cat = clean(ng[:, ti, si])
            else:
                # try (nt, nsysmax, nc1)
                grids_per_cat = clean(ng[ti, si, :])
            total = np.nansum(grids_per_cat)
            if total > 0:
                conv_area_frac = float(grids_per_cat[CAT_CNV]) / float(total)
        except (IndexError, TypeError):
            pass

    # --- 2. conv_intensity (rainrate of GPM_cnv) -------------------------
    conv_intensity = np.nan
    if VAR_RAINRATE in toocan_ds:
        rr = toocan_ds[VAR_RAINRATE].values
        try:
            if rr.ndim == 3:
                val = clean(rr[[CAT_CNV], ti, si] if rr.shape[0] == ncat
                            else rr[ti, si, [CAT_CNV]])
            else:
                val = clean(rr[CAT_CNV, ti, si])
            conv_intensity = float(np.nanmean(val))
        except (IndexError, TypeError):
            pass

    # --- 3. z_top (highest height where GPM_cnv latheating > threshold) --
    z_top = np.nan
    if VAR_LATH in toocan_ds:
        lath = clean(toocan_ds[VAR_LATH].values[CAT_CNV, ti, si, :])
        above = np.where(lath > LATH_THRESHOLD)[0]
        if len(above) > 0:
            z_top = float(heights[above[-1]])

    return conv_area_frac, conv_intensity, z_top


# -----------------------------------------------------------------------
# Snapshot panel helper
# -----------------------------------------------------------------------

def draw_tb_panel(ax, tb_ds, toocan_ds, ti, si, dcs_number, date_str, panel_label):
    """Draw one Tb snapshot into ax. Returns True on success."""
    dcs_spatial = tb_ds["DCS_number"].isel(nt=ti).load().values
    tb_spatial  = tb_ds["Tb"].isel(nt=ti).load().values
    lat_arr     = tb_ds["lat"].values
    lon_arr     = tb_ds["lon"].values

    sys_mask  = (dcs_spatial == dcs_number)
    sys_rows, sys_cols = np.where(sys_mask)
    if len(sys_rows) == 0:
        ax.text(0.5, 0.5, f"DCS {dcs_number}\nnot found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray")
        return False

    r0 = max(sys_rows.min() - EXTRA_MARGIN, 0)
    r1 = min(sys_rows.max() + EXTRA_MARGIN + 1, len(lat_arr))
    c0 = max(sys_cols.min() - EXTRA_MARGIN, 0)
    c1 = min(sys_cols.max() + EXTRA_MARGIN + 1, len(lon_arr))

    tb_sub   = tb_spatial[r0:r1, c0:c1]
    mask_sub = sys_mask[r0:r1, c0:c1]
    lat_sub  = lat_arr[r0:r1]
    lon_sub  = lon_arr[c0:c1]

    LON2D, LAT2D = np.meshgrid(lon_sub, lat_sub)
    im = ax.pcolormesh(LON2D, LAT2D, tb_sub,
                       cmap=plt.cm.RdYlBu_r, vmin=180, vmax=310,
                       shading="nearest")
    plt.colorbar(im, ax=ax, label="Tb (K)", fraction=0.046, pad=0.04)

    if mask_sub.any():
        ax.contour(LON2D, LAT2D, mask_sub.astype(int),
                   levels=[0.5], colors="black", linewidths=1.5,
                   linestyles="--", zorder=5)

    lc_lon = float(toocan_ds["lc_lon"].values[ti, si])
    lc_lat = float(toocan_ds["lc_lat"].values[ti, si])
    time_h = float(toocan_ds["time"].values[ti])

    ax.plot(lc_lon, lc_lat, marker="*", color="white",
            markersize=10, markeredgecolor="black", markeredgewidth=1, zorder=6)

    ax.set_xlabel("Lon (°E)", fontsize=8)
    ax.set_ylabel("Lat (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"{panel_label})  DCS #{dcs_number}  {date_str}\n"
        f"T={time_h:.0f} h  ({lc_lon:.1f}°E, {lc_lat:.1f}°N)",
        fontsize=9, fontweight="bold", pad=4
    )
    return True


# -----------------------------------------------------------------------
# Table panel helper
# -----------------------------------------------------------------------

def draw_table_panel(ax, table_rows, panel_label):
    """
    Draw a formatted comparison table into ax.

    table_rows : list of dicts with keys:
        dcs, date, time_h, conv_area_frac, conv_intensity, z_top
    """
    ax.axis("off")

    col_labels = ["DCS #", "Date", "T (h)",
                  "conv_area_frac\n(GPM_cnv/total)",
                  "conv_intensity\n(mm hr⁻¹)",
                  "z_top\n(km)"]

    cell_text = []
    for r in table_rows:
        def fmt(v, dec=3):
            return f"{v:.{dec}f}" if np.isfinite(v) else "—"
        cell_text.append([
            str(r["dcs"]),
            r["date"],
            f"{r['time_h']:.0f}",
            fmt(r["conv_area_frac"], 3),
            fmt(r["conv_intensity"], 2),
            fmt(r["z_top"],          1),
        ])

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c4f7c")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(1, len(cell_text) + 1):
        fc = "#eaf2fb" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fc)

    ax.set_title(
        f"{panel_label})  Convective metrics comparison\n"
        f"(lath threshold for z_top = {LATH_THRESHOLD} K hr⁻¹)",
        fontsize=9, fontweight="bold", pad=4
    )


# -----------------------------------------------------------------------
# Main figure builder
# -----------------------------------------------------------------------

def make_compare_figure(date_str, systems, toocan_ds, tb_ds, out_dir):
    """
    Build and save one 3×2 figure.
    """
    heights = toocan_ds[VAR_HEIGHTS].values   # (nlvl,) in km
    n       = len(systems)                    # 1..5

    PANEL_LABELS = ["a", "b", "c", "d", "e", "f"]

    fig = plt.figure(figsize=(22, 13))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes_map = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_table = fig.add_subplot(gs[1, 2])

    table_rows = []

    for idx, (ti, si, dcs_number) in enumerate(systems):
        ax = axes_map[idx]
        draw_tb_panel(ax, tb_ds, toocan_ds, ti, si, dcs_number,
                      date_str, PANEL_LABELS[idx])

        # Compute metrics
        caf, ci, zt = compute_metrics(toocan_ds, ti, si, heights)
        time_h = float(toocan_ds["time"].values[ti])
        table_rows.append(dict(
            dcs=dcs_number, date=date_str, time_h=time_h,
            conv_area_frac=caf, conv_intensity=ci, z_top=zt
        ))
        print(f"  DCS {dcs_number:>6}  conv_area_frac={caf:.3f}  "
              f"conv_intensity={ci:.2f} mm/hr  z_top={zt:.1f} km")

    # Hide any unused map panels (if fewer than 5 systems)
    for idx in range(n, 5):
        axes_map[idx].axis("off")

    # Table panel
    draw_table_panel(ax_table, table_rows, PANEL_LABELS[5])

    fig.suptitle(
        f"MCS Convective Comparison  |  {date_str}  |  {n} systems",
        fontsize=14, fontweight="bold", y=1.01
    )

    os.makedirs(out_dir, exist_ok=True)
    fname = f"compare_{date_str}_N{n}.png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fpath}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3×2 MCS Tb snapshot + convective metrics table."
    )
    parser.add_argument("date", nargs="?", default=None,
                        help="Date YYYYMMDD (default: first available).")
    parser.add_argument("--n", type=int, default=N_SYSTEMS_DEFAULT,
                        help=f"Systems to plot, max 5 (default: {N_SYSTEMS_DEFAULT}).")
    parser.add_argument("--outdir", default=OUTPUT_DIR,
                        help="Output directory.")
    args = parser.parse_args()

    n_plot = min(args.n, 5)   # layout supports at most 5 snapshot panels

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

    toocan_ds = xr.open_dataset(toocan_path)
    tb_ds     = xr.open_dataset(tbexpand_path, chunks={"nt": 1})

    # Report which metric variables were found
    for v in [VAR_NGRIDS, VAR_RAINRATE, VAR_LATH, VAR_HEIGHTS]:
        status = "FOUND" if v in toocan_ds else "NOT FOUND (will show —)"
        print(f"  {v:15s}: {status}")

    systems = get_systems(toocan_ds)
    if not systems:
        sys.exit("No systems found in TOOCAN file.")

    selected = systems[:n_plot]
    print(f"\nFound {len(systems)} systems; plotting {len(selected)}.\n")

    make_compare_figure(date_str, selected, toocan_ds, tb_ds, out_dir=args.outdir)

    toocan_ds.close()
    tb_ds.close()
    print(f"Done. Output: {args.outdir}")


if __name__ == "__main__":
    main()
