#!/usr/bin/env python

import glob
import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# -----------------------------
# Helper to parse category labels
# -----------------------------
def parse_category_description(desc_str):
    """
    Parse strings like:
    "{0}GPM_cnv;{1}GPM_noncnv;{2}GPM_norain"
    into a dict: {0: "GPM_cnv", 1: "GPM_noncnv", 2: "GPM_norain"}
    """
    mapping = {}
    if not desc_str:
        return mapping
    for part in desc_str.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            idx_str, name = part.split("}")
            idx = int(idx_str.replace("{", "").strip())
            mapping[idx] = name.strip()
        except Exception:
            continue
    return mapping


def clean_data(data, missing_values=[-9999, -9999.0]):
    """
    Replace missing value indicators with np.nan.
    Also replaces extreme negative values (< -1000) as they're likely missing data.
    """
    data_clean = data.copy()
    
    # Replace specific missing value indicators
    for mv in missing_values:
        data_clean = xr.where(data_clean == mv, np.nan, data_clean)
    
    # Replace extreme negative values (likely missing data)
    data_clean = xr.where(data_clean < -1000, np.nan, data_clean)
    
    return data_clean


# -----------------------------
# Configuration
# -----------------------------
# Directory containing the extracted gpm_select-toocan NetCDF files
DATA_DIR = "gpm_select_toocan"  

# Glob pattern for the files
FILE_PATTERN = os.path.join(DATA_DIR, "*.nc")

# Variables for which to compute vertical profiles
PROFILE_VARS = [
    "latheating",      # Latent heating (K/hr)
    "eddyheating",     # Eddy heating (K/hr)
    "eddymoistening",  # Eddy moistening
    "swheating",       # Shortwave heating
    "lwheating",       # Longwave heating
]

OUTPUT_FILE = "toocan_vertical_profiles_by_year.nc"
PLOT_OUTPUT_DIR = "plots"


# -----------------------------
# Process TOOCAN data
# -----------------------------
def process_toocan_data():
    """
    Process GPM-TOOCAN data and create yearly vertical profiles with variance.
    
    Uses online statistics (Welford's method) to handle files with different 
    nsysmax dimensions without loading all data into memory.
    """
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found with pattern: {FILE_PATTERN}")

    print(f"Found {len(files)} TOOCAN files")

    year_data = {}  # Store all samples for variance calculation
    category_names = None
    levels = None

    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.match(r"^(\d{4})(\d{2})(\d{2})", fname)
        if not m:
            print(f"Skipping file (cannot parse date from name): {fname}")
            continue
        year = int(m.group(1))

        print(f"Processing TOOCAN file: {fname} (year={year})")

        ds = xr.open_dataset(fpath)

        if category_names is None:
            nc1_desc = ds["nc1"].attrs.get("description", "")
            nc1_map = parse_category_description(nc1_desc)
            ncat = ds.sizes["nc1"]
            category_names = [nc1_map.get(i, f"cat_{i}") for i in range(ncat)]
            print("Categories (nc1):")
            for i, name in enumerate(category_names):
                print(f"  {i}: {name}")

        if levels is None:
            levels = ds["nlvl"].values

        if "DCS_number" in ds:
            dcs_valid = (ds["DCS_number"] != 0)
            mask = dcs_valid.expand_dims(
                {"nc1": ds.sizes["nc1"], "nlvl": ds.sizes["nlvl"]},
                axis=(0, 3),
            )
        else:
            mask = None

        if year not in year_data:
            year_data[year] = {var: {"sum": None, "sum_sq": None, "count": None} 
                              for var in PROFILE_VARS}

        for var_name in PROFILE_VARS:
            if var_name not in ds:
                print(f"  Warning: {var_name} not in dataset; skipping for this file.")
                continue

            var = ds[var_name]
            
            # Clean the data - replace -9999 and extreme negatives with NaN
            var = clean_data(var)
            
            if mask is not None:
                var = var.where(mask)

            # Compute statistics for this file without storing full arrays
            # Sum over time and system dimensions
            var_sum = var.sum(dim=["nt", "nsysmax"], skipna=True)
            var_sum_sq = (var ** 2).sum(dim=["nt", "nsysmax"], skipna=True)
            var_count = (~np.isnan(var)).sum(dim=["nt", "nsysmax"])
            
            # Accumulate statistics
            if year_data[year][var_name]["sum"] is None:
                year_data[year][var_name]["sum"] = var_sum
                year_data[year][var_name]["sum_sq"] = var_sum_sq
                year_data[year][var_name]["count"] = var_count
            else:
                year_data[year][var_name]["sum"] += var_sum
                year_data[year][var_name]["sum_sq"] += var_sum_sq
                year_data[year][var_name]["count"] += var_count

        ds.close()

    # Calculate mean and std for each year
    years_sorted = sorted(year_data.keys())
    ncat = len(category_names)
    nlev = len(levels)

    year_coord = xr.DataArray(years_sorted, dims=("year",), name="year")
    cat_coord = xr.DataArray(category_names, dims=("category",), name="category")
    lev_coord = xr.DataArray(levels, dims=("nlvl",), name="nlvl")

    data_vars = {}

    for var_name in PROFILE_VARS:
        mean_arr = np.full((len(years_sorted), ncat, nlev), np.nan, dtype=np.float64)
        std_arr = np.full((len(years_sorted), ncat, nlev), np.nan, dtype=np.float64)

        for yi, year in enumerate(years_sorted):
            if var_name not in year_data[year]:
                continue
            
            stats = year_data[year][var_name]
            
            if stats["sum"] is None or stats["count"] is None:
                continue
            
            # Calculate mean
            count = stats["count"].where(stats["count"] > 0)
            mean = stats["sum"] / count
            
            # Calculate standard deviation using: std = sqrt(E[X^2] - E[X]^2)
            mean_sq = stats["sum_sq"] / count
            variance = mean_sq - (mean ** 2)
            variance = variance.where(variance >= 0, 0)  # Handle numerical errors
            std = np.sqrt(variance)
            
            mean_arr[yi, :, :] = mean.values
            std_arr[yi, :, :] = std.values

        # Create mean data variable
        data_vars[var_name] = xr.DataArray(
            mean_arr,
            dims=("year", "category", "nlvl"),
            coords={
                "year": year_coord,
                "category": cat_coord,
                "nlvl": lev_coord,
            },
            name=var_name,
        )
        
        # Create std data variable
        data_vars[f"{var_name}_std"] = xr.DataArray(
            std_arr,
            dims=("year", "category", "nlvl"),
            coords={
                "year": year_coord,
                "category": cat_coord,
                "nlvl": lev_coord,
            },
            name=f"{var_name}_std",
        )

    out_ds = xr.Dataset(data_vars=data_vars, coords={
        "year": year_coord,
        "category": cat_coord,
        "nlvl": lev_coord,
    })

    out_ds["category"].attrs["description"] = "Convective categories"
    out_ds["nlvl"].attrs["long_name"] = "Height of vertical levels"
    out_ds["nlvl"].attrs["units"] = "km"

    print(f"\nSaving TOOCAN yearly vertical profiles to: {OUTPUT_FILE}")
    out_ds.to_netcdf(OUTPUT_FILE)
    return out_ds


# -----------------------------
# Plotting functions
# -----------------------------
def plot_combined_vertical_profiles(toocan_ds):
    """
    Create a single figure with three subplots showing vertical profiles 
    for convective, non-convective, and no-rain categories with variance shading.
    """
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    # Average over all years
    toocan_mean = toocan_ds.mean(dim="year")

    # Find the three categories
    categories = toocan_mean["category"].values
    conv_idx = None
    nonconv_idx = None
    norain_idx = None
    
    for i, cat in enumerate(categories):
        cat_lower = cat.lower()
        if "norain" in cat_lower or "no_rain" in cat_lower:
            norain_idx = i
        elif "cnv" in cat_lower and "non" not in cat_lower:
            conv_idx = i
        elif "noncnv" in cat_lower or "non" in cat_lower:
            nonconv_idx = i

    if conv_idx is None or nonconv_idx is None or norain_idx is None:
        print("Warning: Could not identify all three categories")
        print(f"Found categories: {categories}")
        return

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    heights = toocan_mean["nlvl"].values
    
    # Category information
    cat_info = [
        (conv_idx, "Convective", axes[0], 'blue'),
        (nonconv_idx, "Non-Convective", axes[1], 'red'),
        (norain_idx, "No Rain", axes[2], 'green')
    ]
    
    for cat_idx, cat_name, ax, color in cat_info:
        mean_profile = toocan_mean["latheating"].isel(category=cat_idx)
        std_profile = toocan_mean["latheating_std"].isel(category=cat_idx)
        
        # Plot mean line
        ax.plot(mean_profile, heights, color=color, linewidth=2.5, 
                label='Mean', zorder=3)
        
        # Add shaded region for ±1 standard deviation
        ax.fill_betweenx(heights, 
                         mean_profile - std_profile, 
                         mean_profile + std_profile,
                         alpha=0.3, color=color, label='±1σ', zorder=2)
        
        # Formatting
        ax.set_xlabel('Latent Heating (K/hr)', fontsize=13, fontweight='bold')
        ax.set_title(cat_name, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=11, loc='best')
        
        # Set x-axis limits for better visibility
        mean_vals = mean_profile.values[~np.isnan(mean_profile.values)]
        if len(mean_vals) > 0:
            x_range = np.max(np.abs(mean_vals)) * 1.2
            ax.set_xlim(-x_range, x_range)
    
    # Set y-axis label only on the first subplot
    axes[0].set_ylabel('Height (km)', fontsize=13, fontweight='bold')
    
    # Add overall title
    fig.suptitle('TOOCAN Latent Heating Vertical Profiles by Category', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_path = os.path.join(PLOT_OUTPUT_DIR, "toocan_latent_heating_all_categories.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined plot saved to: {output_path}")
    plt.close()


def plot_all_variables_combined(toocan_ds):
    """
    Create plots for all available heating/moistening variables.
    """
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    # Average over all years
    toocan_mean = toocan_ds.mean(dim="year")
    
    # Find the three categories
    categories = toocan_mean["category"].values
    conv_idx = None
    nonconv_idx = None
    norain_idx = None
    
    for i, cat in enumerate(categories):
        cat_lower = cat.lower()
        if "norain" in cat_lower or "no_rain" in cat_lower:
            norain_idx = i
        elif "cnv" in cat_lower and "non" not in cat_lower:
            conv_idx = i
        elif "noncnv" in cat_lower or "non" in cat_lower:
            nonconv_idx = i
    
    if conv_idx is None or nonconv_idx is None or norain_idx is None:
        print("Warning: Could not identify all three categories")
        return
    
    heights = toocan_mean["nlvl"].values
    
    # Get all variables (excluding _std variables)
    variables = [var for var in toocan_ds.data_vars if not var.endswith("_std")]
    
    # Create a plot for each variable
    for var_name in variables:
        if var_name not in toocan_mean:
            continue
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        cat_info = [
            (conv_idx, "Convective", axes[0], 'blue'),
            (nonconv_idx, "Non-Convective", axes[1], 'red'),
            (norain_idx, "No Rain", axes[2], 'green')
        ]
        
        for cat_idx, cat_name, ax, color in cat_info:
            mean_profile = toocan_mean[var_name].isel(category=cat_idx)
            std_var = f"{var_name}_std"
            
            if std_var in toocan_mean:
                std_profile = toocan_mean[std_var].isel(category=cat_idx)
                
                # Plot mean line
                ax.plot(mean_profile, heights, color=color, linewidth=2.5, 
                       label='Mean', zorder=3)
                
                # Add shaded region
                ax.fill_betweenx(heights, 
                               mean_profile - std_profile, 
                               mean_profile + std_profile,
                               alpha=0.3, color=color, label='±1σ', zorder=2)
            else:
                # Just plot mean if no std available
                ax.plot(mean_profile, heights, color=color, linewidth=2.5, 
                       label='Mean', zorder=3)
            
            # Formatting
            ax.set_xlabel(f'{var_name}', fontsize=13, fontweight='bold')
            ax.set_title(cat_name, fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.legend(fontsize=11, loc='best')
            
            # Set x-axis limits
            mean_vals = mean_profile.values[~np.isnan(mean_profile.values)]
            if len(mean_vals) > 0:
                x_range = np.max(np.abs(mean_vals)) * 1.2
                ax.set_xlim(-x_range, x_range)
        
        axes[0].set_ylabel('Height (km)', fontsize=13, fontweight='bold')
        
        fig.suptitle(f'TOOCAN {var_name} Vertical Profiles by Category', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        output_path = os.path.join(PLOT_OUTPUT_DIR, f"toocan_{var_name}_all_categories.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    print("="*60)
    print("TOOCAN VERTICAL PROFILE ANALYSIS")
    print("="*60)
    
    # Process TOOCAN data
    print("\n" + "="*60)
    print("PROCESSING TOOCAN DATA")
    print("="*60)
    toocan_ds = process_toocan_data()
    
    # Create comparison plots
    print("\n" + "="*60)
    print("CREATING PLOTS")
    print("="*60)
    plot_combined_vertical_profiles(toocan_ds)
    
    # Create plots for all variables
    print("\nCreating plots for all variables...")
    plot_all_variables_combined(toocan_ds)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Data: {OUTPUT_FILE}")
    print(f"  - Plots: {PLOT_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
