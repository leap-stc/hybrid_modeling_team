#!/usr/bin/env python
"""
lh_mh_compare.py
-----------------
For a given date, select up to 5 DCS systems and compare their vertical
profiles side-by-side in a 3×2 figure:

  a) Latent heating    – GPM_cnv      b) Latent heating    – GPM_noncnv
  c) Eddy moistening   – GPM_cnv      d) Eddy moistening   – GPM_noncnv
  e) ERA5 temperature                 f) ERA5 moisture

Each panel overlays one line per DCS system (up to 5), using a consistent
colour palette so the same DCS is identifiable across all panels.

Usage
-----
  python lh_mh_compare.py                   # first available date, 5 systems
  python lh_mh_compare.py 20140320          # specific date
  python lh_mh_compare.py 20140320 --n 3   # first 3 systems that day
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
ERA5_DIR     = "/glade/work/addisus/GPM_TOOCAN_data/era5_select"
OUTPUT_DIR   = "/glade/work/addisus/GPM_TOOCAN_data/plots/lh_mh_compare"

N_SYSTEMS_DEFAULT = 5   # max 5 to keep the legend readable
MISSING_THRESH    = -1000.0

# Category indices on the nc1 axis
CAT_CNV    = 0   # GPM_cnv
CAT_NONCNV = 1   # GPM_noncnv

# 5 visually distinct colours – one per DCS system
DCS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
DCS_STYLES = ["-", "--", "-.", ":", (0, (5, 1))]

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def clean(arr):
    out = arr.astype(float)
    out[out < MISSING_THRESH] = np.nan
    return out


def find_file_pair(date_str):
    tp = os.path.join(TOOCAN_DIR,   f"{date_str}.gpm_select-toocan-subset.nc")
    tb = os.path.join(TBEXPAND_DIR,  f"{date_str}_TbExpand.nc")
    if not os.path.isfile(tp):
        raise FileNotFoundError(f"TOOCAN not found: {tp}")
    if not os.path.isfile(tb):
        raise FileNotFoundError(f"TbExpand not found: {tb}")
    return tp, tb


def find_era5_file(date_str):
    path = os.path.join(ERA5_DIR,
                        f"{date_str}.era5_select-toocan-subset_pm6hrsSearch.nc")
    return path if os.path.isfile(path) else None


def list_available_dates():
    files = glob.glob(os.path.join(TOOCAN_DIR, "*.gpm_select-toocan-subset.nc"))
    dates = []
    for f in sorted(files):
        m = re.search(r"(\d{8})\.", os.path.basename(f))
        if m:
            d = m.group(1)
            if os.path.isfile(os.path.join(TBEXPAND_DIR, f"{d}_TbExpand.nc")):
                dates.append(d)
    return dates


def get_systems(toocan_ds):
    dcs    = toocan_ds["DCS_number"].values
    idx    = list(zip(*np.argwhere(dcs != 0).T))
    seen, unique = {}, []
    for ti, si in idx:
        d = int(dcs[ti, si])
        if d not in seen:
            seen[d] = (ti, si)
            unique.append((ti, si, d))
    unique.sort(key=lambda x: x[2])
    return unique


# -----------------------------------------------------------------------
# Profile extraction
# -----------------------------------------------------------------------

def extract_profiles(toocan_ds, era5_ds, ti, si):
    """
    Extract all six profile types for one system.

    Returns dict with keys:
      lath_cnv, lath_noncnv           – 1-D array on heights (km)
      eddy_cnv, eddy_noncnv           – 1-D array on heights (km)
      era5_t, era5_q, era5_plev       – 1-D arrays on pressure (hPa)
      era5_t_p25, era5_t_p75          – IQR bounds
      era5_q_p25, era5_q_p75          – IQR bounds
    All missing/invalid values are NaN.
    """
    def lath(cat):
        return clean(toocan_ds["latheating"].values[cat, ti, si, :])

    def eddy(cat):
        return clean(toocan_ds["eddymoistening"].values[cat, ti, si, :])

    out = dict(
        lath_cnv    = lath(CAT_CNV),
        lath_noncnv = lath(CAT_NONCNV),
        eddy_cnv    = eddy(CAT_CNV),
        eddy_noncnv = eddy(CAT_NONCNV),
        era5_t=None, era5_q=None, era5_plev=None,
        era5_t_p25=None, era5_t_p75=None,
        era5_q_p25=None, era5_q_p75=None,
    )

    if era5_ds is not None:
        dt       = era5_ds["deltatime"].values[:, ti]        # (nc2,)
        abs_dt   = np.where(np.isfinite(dt), np.abs(dt), np.inf)
        best_nc2 = int(np.argmin(abs_dt))

        # tprof/qprof/plev: (nc2, nc1, nt, nsysmax, nlvl)
        # nc1 index: 0=mean, 1=p25, 3=p75
        def era5_var(v, nc1_idx):
            return clean(era5_ds[v].values[best_nc2, nc1_idx, ti, si, :])

        out.update(
            era5_t    = era5_var("tprof", 0),
            era5_t_p25= era5_var("tprof", 1),
            era5_t_p75= era5_var("tprof", 3),
            era5_q    = era5_var("qprof", 0),
            era5_q_p25= era5_var("qprof", 1),
            era5_q_p75= era5_var("qprof", 3),
            era5_plev = era5_var("plev",  0),
        )

    return out


# -----------------------------------------------------------------------
# Individual panel drawers
# -----------------------------------------------------------------------

def draw_height_profile(ax, title, xlabel,
                        profiles_list, heights, legend_labels,
                        colors, styles, add_legend=False):
    """
    Overlay up to 5 profiles on a height (km) y-axis.

    profiles_list : list of 1-D arrays, one per DCS
    """
    has_data = False
    for prof, label, col, ls in zip(profiles_list, legend_labels,
                                    colors, styles):
        if prof is None:
            continue
        valid = ~np.isnan(prof)
        if valid.any():
            ax.plot(prof[valid], heights[valid],
                    color=col, linestyle=ls, linewidth=2.0, label=label)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Height (km)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=9)
    if add_legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)


def draw_pressure_profile(ax, title, xlabel,
                          means_list, p25_list, p75_list, plev_list,
                          legend_labels, colors, styles, add_legend=False):
    """
    Overlay up to 5 ERA5 profiles on a pressure (hPa) y-axis with IQR shading.
    """
    has_data = False
    for mean, p25, p75, plev, label, col, ls in zip(
            means_list, p25_list, p75_list, plev_list,
            legend_labels, colors, styles):

        if mean is None or plev is None:
            continue
        valid = np.isfinite(plev) & np.isfinite(mean)
        if not valid.any():
            continue

        ax.plot(mean[valid], plev[valid],
                color=col, linestyle=ls, linewidth=2.0, label=label)

        if p25 is not None and p75 is not None:
            iqr_v = valid & np.isfinite(p25) & np.isfinite(p75)
            if iqr_v.any():
                ax.fill_betweenx(plev[iqr_v], p25[iqr_v], p75[iqr_v],
                                 color=col, alpha=0.12)
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "ERA5 not available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray")

    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Pressure (hPa)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=9)
    if add_legend:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)


# -----------------------------------------------------------------------
# Main figure builder
# -----------------------------------------------------------------------

def make_lh_mh_figure(date_str, systems, toocan_ds, era5_ds, out_dir):
    """
    Build and save one 3×2 figure comparing profiles of 5 DCS systems.
    """
    heights = toocan_ds["nlvl"].values    # (nlvl,) in km
    n       = len(systems)

    # Collect profiles for each system
    all_profiles   = []
    legend_labels  = []
    colors_used    = []
    styles_used    = []

    for idx, (ti, si, dcs_number) in enumerate(systems):
        lc_lon = float(toocan_ds["lc_lon"].values[ti, si])
        lc_lat = float(toocan_ds["lc_lat"].values[ti, si])
        time_h = float(toocan_ds["time"].values[ti])

        profs = extract_profiles(toocan_ds, era5_ds, ti, si)
        all_profiles.append(profs)

        label = (f"DCS {dcs_number}  "
                 f"({lc_lon:.1f}°E, {lc_lat:.1f}°N)  "
                 f"T={time_h:.0f} h")
        legend_labels.append(label)
        colors_used.append(DCS_COLORS[idx % len(DCS_COLORS)])
        styles_used.append(DCS_STYLES[idx % len(DCS_STYLES)])
        print(f"  [{idx+1}] DCS {dcs_number}  ti={ti}  si={si}")

    # Helper to pull a named field from all profiles
    def gather(key):
        return [p.get(key) for p in all_profiles]

    fig = plt.figure(figsize=(20, 13))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])   # a) lath GPM_cnv
    ax_b = fig.add_subplot(gs[0, 1])   # b) lath GPM_noncnv
    ax_c = fig.add_subplot(gs[0, 2])   # c) eddy GPM_cnv
    ax_d = fig.add_subplot(gs[1, 0])   # d) eddy GPM_noncnv
    ax_e = fig.add_subplot(gs[1, 1])   # e) ERA5 temperature
    ax_f = fig.add_subplot(gs[1, 2])   # f) ERA5 moisture

    # --- a) Latent heating – GPM_cnv ---
    draw_height_profile(
        ax_a,
        title="a)  Latent Heating – GPM_cnv",
        xlabel="Latent Heating (K hr⁻¹)",
        profiles_list=gather("lath_cnv"),
        heights=heights,
        legend_labels=legend_labels,
        colors=colors_used, styles=styles_used,
        add_legend=False,
    )

    # --- b) Latent heating – GPM_noncnv ---
    draw_height_profile(
        ax_b,
        title="b)  Latent Heating – GPM_noncnv",
        xlabel="Latent Heating (K hr⁻¹)",
        profiles_list=gather("lath_noncnv"),
        heights=heights,
        legend_labels=legend_labels,
        colors=colors_used, styles=styles_used,
        add_legend=False,
    )

    # --- c) Eddy moistening – GPM_cnv ---
    draw_height_profile(
        ax_c,
        title="c)  Eddy Moistening – GPM_cnv",
        xlabel="Eddy Moistening (g kg⁻¹ day⁻¹)",
        profiles_list=gather("eddy_cnv"),
        heights=heights,
        legend_labels=legend_labels,
        colors=colors_used, styles=styles_used,
        add_legend=False,
    )

    # --- d) Eddy moistening – GPM_noncnv ---
    draw_height_profile(
        ax_d,
        title="d)  Eddy Moistening – GPM_noncnv",
        xlabel="Eddy Moistening (g kg⁻¹ day⁻¹)",
        profiles_list=gather("eddy_noncnv"),
        heights=heights,
        legend_labels=legend_labels,
        colors=colors_used, styles=styles_used,
        add_legend=True,    # legend in bottom-left panel
    )

    # --- e) ERA5 temperature ---
    draw_pressure_profile(
        ax_e,
        title="e)  ERA5 Temperature",
        xlabel="Temperature (K)",
        means_list=gather("era5_t"),
        p25_list=gather("era5_t_p25"),
        p75_list=gather("era5_t_p75"),
        plev_list=gather("era5_plev"),
        legend_labels=legend_labels,
        colors=colors_used, styles=styles_used,
        add_legend=False,
    )

    # --- f) ERA5 moisture ---
    draw_pressure_profile(
        ax_f,
        title="f)  ERA5 Moisture",
        xlabel="Water Vapor (g kg⁻¹)",
        means_list=gather("era5_q"),
        p25_list=gather("era5_q_p25"),
        p75_list=gather("era5_q_p75"),
        plev_list=gather("era5_plev"),
        legend_labels=legend_labels,
        colors=colors_used, styles=styles_used,
        add_legend=False,
    )

    # Shared legend at the top of the figure
    legend_handles = [
        mpatches.Patch(color=colors_used[i], label=legend_labels[i])
        for i in range(n)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(n, 3),
        fontsize=9,
        framealpha=0.9,
        title=f"DCS systems  |  shading = 25–75th percentile (ERA5)",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 1.03),
    )

    era5_status = "with ERA5" if era5_ds is not None else "ERA5 not available"
    fig.suptitle(
        f"LH / MH Profile Comparison  |  {date_str}  |  "
        f"{n} DCS systems  |  {era5_status}",
        fontsize=13, fontweight="bold", y=1.07,
    )

    os.makedirs(out_dir, exist_ok=True)
    fname = f"lh_mh_compare_{date_str}_N{n}.png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fpath}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3×2 LH/MH profile comparison for up to 5 DCS systems."
    )
    parser.add_argument("date", nargs="?", default=None,
                        help="Date YYYYMMDD (default: first available).")
    parser.add_argument("--n", type=int, default=N_SYSTEMS_DEFAULT,
                        help=f"Systems to plot, max 5 (default: {N_SYSTEMS_DEFAULT}).")
    parser.add_argument("--outdir", default=OUTPUT_DIR,
                        help="Output directory.")
    args = parser.parse_args()

    n_plot = min(args.n, 5)

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
    toocan_path, _ = find_file_pair(date_str)
    print(f"  TOOCAN  : {toocan_path}")

    era5_path = find_era5_file(date_str)
    era5_ds   = None
    if era5_path:
        print(f"  ERA5    : {era5_path}")
        era5_ds = xr.open_dataset(era5_path)
    else:
        print(f"  ERA5    : not found for {date_str} (panels e/f will be blank)")

    toocan_ds = xr.open_dataset(toocan_path)

    systems = get_systems(toocan_ds)
    if not systems:
        sys.exit("No systems found in TOOCAN file.")

    selected = systems[:n_plot]
    print(f"Found {len(systems)} systems; comparing first {len(selected)}.\n")

    make_lh_mh_figure(date_str, selected, toocan_ds, era5_ds, out_dir=args.outdir)

    toocan_ds.close()
    if era5_ds is not None:
        era5_ds.close()

    print(f"Done. Output: {args.outdir}")


if __name__ == "__main__":
    main()
