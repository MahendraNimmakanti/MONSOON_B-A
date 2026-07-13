"""
Supplementary Figures — Monsoon Break Events (1981–2025)
=========================================================
Figure A : JJAS daily all-India rainfall bar plots for each break year
           with 1991–2020 smoothed climatology overlaid and break period shaded.
Figure B : Spatial mean rainfall anomaly panels for each break event
           (event mean – 1991–2020 DOY climatology), plotted over India.

Author: Mahendra (Purdue / Matthew Huber lab)
Date  : 2026-06-18
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMD_DIR   = "/home/mahendra/Downloads/IMD_Rainfall"
OUT_DIR   = "/home/mahendra"
CLIM_YRS  = (1991, 2020)
SMOOTH_W  = 15   # days for climatology smoothing

# Break events (from v2 analysis)
break_events = [
    dict(year=1982, start="1982-06-27", end="1982-07-08", aligned=True),
    dict(year=1986, start="1986-08-25", end="1986-09-08", aligned=False),
    dict(year=1992, start="1992-09-15", end="1992-09-25", aligned=True),
    dict(year=1994, start="1994-09-21", end="1994-09-30", aligned=True),
    dict(year=2002, start="2002-07-06", end="2002-07-17", aligned=True),
    dict(year=2002, start="2002-07-24", end="2002-08-02", aligned=False),
    dict(year=2004, start="2004-08-27", end="2004-09-05", aligned=True),
    dict(year=2009, start="2009-06-13", end="2009-06-25", aligned=True),
    dict(year=2009, start="2009-07-31", end="2009-08-12", aligned=True),
    dict(year=2023, start="2023-08-08", end="2023-08-17", aligned=True),
]
for ev in break_events:
    ev["start"] = pd.Timestamp(ev["start"])
    ev["end"]   = pd.Timestamp(ev["end"])
    ev["dur"]   = (ev["end"] - ev["start"]).days + 1

unique_years = sorted(set(ev["year"] for ev in break_events))

# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD ALL-INDIA TIME SERIES + SPATIAL DATASET (1981–2025)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading all IMD files for JJAS …")

ai_list    = []   # all-India spatial mean
ds_by_year = {}   # spatial datasets keyed by year

all_needed_years = set(range(CLIM_YRS[0], CLIM_YRS[1]+1)) | set(unique_years)

for yr in sorted(all_needed_years):
    fpath = f"{IMD_DIR}/IMD_rain_{yr}.nc"
    if not os.path.exists(fpath):
        continue
    ds   = xr.open_dataset(fpath)
    rain = ds["rain"].sel(time=ds["rain"].time.dt.month.isin([6,7,8,9]))
    # All-India daily mean
    ai   = rain.mean(dim=["lat","lon"], skipna=True)
    df   = ai.to_dataframe(name="rain").reset_index()
    df["time"] = pd.to_datetime(df["time"])
    ai_list.append(df)
    # Store full spatial data for break years
    if yr in set(unique_years):
        ds_by_year[yr] = rain.load()
    ds.close()

ai_df = pd.concat(ai_list).rename(columns={"time":"date"}).set_index("date").sort_index()
print(f"  All-India series: {len(ai_df)} JJAS days")

# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE 1991–2020 CLIMATOLOGY (all-India + spatial)
# ─────────────────────────────────────────────────────────────────────────────
print(f"Computing {CLIM_YRS[0]}–{CLIM_YRS[1]} climatology …")

# All-India DOY climatology
clim_mask = ((ai_df.index.year >= CLIM_YRS[0]) &
             (ai_df.index.year <= CLIM_YRS[1]))
ai_clim   = ai_df[clim_mask].copy()
ai_clim["doy"] = ai_clim.index.day_of_year

doy_mean_raw = ai_clim.groupby("doy")["rain"].mean()
doy_std_raw  = ai_clim.groupby("doy")["rain"].std()
# 15-day rolling smooth
doy_mean = doy_mean_raw.rolling(SMOOTH_W, center=True, min_periods=5).mean().fillna(doy_mean_raw)
doy_std  = doy_std_raw.rolling(SMOOTH_W, center=True, min_periods=5).mean().fillna(doy_std_raw)

# Spatial DOY climatology (only needed for break-event DOYs)
print("  Building spatial climatology grid …")
clim_years_list = [yr for yr in range(CLIM_YRS[0], CLIM_YRS[1]+1)
                   if yr in set(range(CLIM_YRS[0], CLIM_YRS[1]+1))]

# Load climatology spatial data once
clim_ds_list = []
for yr in range(CLIM_YRS[0], CLIM_YRS[1]+1):
    fpath = f"{IMD_DIR}/IMD_rain_{yr}.nc"
    if not os.path.exists(fpath): continue
    ds   = xr.open_dataset(fpath)
    rain = ds["rain"].sel(time=ds["rain"].time.dt.month.isin([6,7,8,9]))
    clim_ds_list.append(rain.load())
    ds.close()

clim_concat = xr.concat(clim_ds_list, dim="time")   # (N_days_total, lat, lon)

# Day-of-year climatological mean (spatial) — raw, then smoothed below in the loop
print(f"  Spatial climatology: {clim_concat.sizes['time']} total JJAS days from {len(clim_ds_list)} years")

def spatial_doy_clim(doy_list):
    """Return smoothed spatial climatological mean for a list of DOYs."""
    # Gather all clim-period days matching any of the target DOYs ±SMOOTH_W//2
    half = SMOOTH_W // 2
    sel_times = []
    for doy in doy_list:
        for offset in range(-half, half+1):
            d = doy + offset
            sel_times.append(d % 366 or 366)   # wrap
    sel_times = list(set(sel_times))
    mask = np.isin(clim_concat.time.dt.dayofyear.values, sel_times)
    return clim_concat.isel(time=mask).mean(dim="time", skipna=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE A — JJAS daily rainfall + climatology per break year
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure A: JJAS daily rainfall + climatology …")

TARGET_COLOR = "#c0392b"
OTHER_COLOR  = "#2980b9"
CLIM_COLOR   = "k"

fig_a, axes_a = plt.subplots(len(unique_years), 1,
                              figsize=(16, 3.8*len(unique_years)),
                              sharex=False)
if len(unique_years) == 1:
    axes_a = [axes_a]

fig_a.suptitle(
    "All-India Daily Rainfall — JJAS  (mm/day)\n"
    f"Bars: observed  |  Line: {CLIM_YRS[0]}–{CLIM_YRS[1]} smoothed climatology  |"
    "  Shaded: break event (red = MJO-aligned, grey = not aligned)",
    fontsize=12, fontweight="bold")

for ax, yr in zip(axes_a, unique_years):
    # JJAS daily data for this year
    yr_data = ai_df[ai_df.index.year == yr].copy()
    yr_data["doy"]  = yr_data.index.day_of_year
    yr_data["clim"] = yr_data["doy"].map(doy_mean)
    yr_data["std"]  = yr_data["doy"].map(doy_std)
    yr_data["anom"] = yr_data["rain"] - yr_data["clim"]

    dates = yr_data.index
    rain  = yr_data["rain"].values
    clim  = yr_data["clim"].values
    std   = yr_data["std"].values

    # Bars coloured by anomaly sign
    bar_cols = np.where(rain >= clim, "#3498db", "#e74c3c")
    ax.bar(dates, rain, width=0.9, color=bar_cols, alpha=0.75, zorder=2, label="Observed")

    # Climatology ± 1 std
    ax.fill_between(dates, clim - std, clim + std,
                    color="grey", alpha=0.18, zorder=1, label="Clim ± 1σ")
    ax.plot(dates, clim, color=CLIM_COLOR, lw=2.0, zorder=3,
            label=f"Clim {CLIM_YRS[0]}–{CLIM_YRS[1]}")
    ax.axhline(0, color="k", lw=0.4)

    # Shade each break event in this year
    yr_events = [ev for ev in break_events if ev["year"] == yr]
    for ev in yr_events:
        shade_col = TARGET_COLOR if ev["aligned"] else "gray"
        ax.axvspan(ev["start"], ev["end"] + pd.Timedelta(days=1),
                   color=shade_col, alpha=0.22, zorder=4, lw=0)
        # Dashed outline
        ax.axvline(ev["start"], color=shade_col, lw=1.5, ls="--", zorder=5)
        ax.axvline(ev["end"] + pd.Timedelta(days=1), color=shade_col, lw=1.5, ls="--", zorder=5)
        # Label: dates + duration
        mid = ev["start"] + (ev["end"] - ev["start"]) / 2
        ax.text(mid, 0.97,
                f"{ev['start'].strftime('%b %d')}–{ev['end'].strftime('%b %d')}\n({ev['dur']}d)",
                ha="center", va="top", fontsize=7.5, color=shade_col,
                fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_xlim(dates[0] - pd.Timedelta(days=1),
                dates[-1] + pd.Timedelta(days=2))
    ax.set_ylabel("Rain (mm/day)", fontsize=9)
    ax.set_title(f"{yr}  JJAS", fontsize=10, fontweight="bold", loc="left")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, ls=":", alpha=0.4)

    # Legend (first panel only)
    if yr == unique_years[0]:
        extra = [
            mpatches.Patch(color=TARGET_COLOR, alpha=0.35, label="Break: MJO 7/8/1 aligned"),
            mpatches.Patch(color="gray",        alpha=0.35, label="Break: not aligned"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles+extra, fontsize=8, loc="upper right", ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUT_DIR}/v2_JJAS_daily_rainfall_byYear.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/v2_JJAS_daily_rainfall_byYear.png", dpi=150, bbox_inches="tight")
print("  Saved: v2_JJAS_daily_rainfall_byYear.pdf/.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURE B — Spatial anomaly panels for each break event
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure B: spatial rainfall anomaly panels …")

n_ev  = len(break_events)
ncols = 4
nrows = int(np.ceil(n_ev / ncols))

proj  = ccrs.PlateCarree()
fig_b = plt.figure(figsize=(5.5*ncols, 5.0*nrows))
fig_b.suptitle(
    "Mean Rainfall Anomaly During Monsoon Break Events ≥10 Days\n"
    f"Anomaly = event mean − {CLIM_YRS[0]}–{CLIM_YRS[1]} DOY climatology (mm/day)  |"
    "  ★ = MJO phases 7/8/1 aligned",
    fontsize=13, fontweight="bold", y=1.01)

# Shared colormap limits — compute across all events first
print("  Computing spatial anomalies …")
anom_list = []
for ev in break_events:
    yr   = ev["year"]
    s, e = ev["start"], ev["end"]
    doys = list(range(s.day_of_year, e.day_of_year + 1))
    if yr not in ds_by_year:
        anom_list.append(None); continue
    event_rain = ds_by_year[yr].sel(time=slice(s, e)).mean(dim="time", skipna=True)
    clim_rain  = spatial_doy_clim(doys)
    anom       = (event_rain - clim_rain).values
    anom_list.append(anom)

# Symmetric colorbar across all events
all_vals = np.concatenate([a.ravel()[~np.isnan(a.ravel())] for a in anom_list if a is not None])
vlim = np.percentile(np.abs(all_vals), 97)
vlim = np.round(vlim / 2) * 2   # round to even number

# Get lat/lon from one dataset
sample_ds = ds_by_year[unique_years[0]]
lats = sample_ds.lat.values
lons = sample_ds.lon.values

for idx, (ev, anom) in enumerate(zip(break_events, anom_list)):
    ax = fig_b.add_subplot(nrows, ncols, idx+1, projection=proj)

    if anom is None:
        ax.set_title("No data"); continue

    # Plot anomaly
    im = ax.pcolormesh(
        lons, lats, anom,
        transform=proj,
        cmap="BrBG",
        vmin=-vlim, vmax=vlim
    )

    # Stipple significance: |anom| > 0.5 * local std
    # Compute local std over the same DOYs
    doys = list(range(ev["start"].day_of_year, ev["end"].day_of_year + 1))
    half = SMOOTH_W // 2
    all_doys = list(set((d + off) % 366 or 366
                        for d in doys for off in range(-half, half+1)))
    mask = np.isin(clim_concat.time.dt.dayofyear.values, all_doys)
    local_std = clim_concat.isel(time=mask).std(dim="time", skipna=True).values
    sig_mask  = np.abs(anom) > local_std
    yy, xx = np.meshgrid(lats[::4], lons[::4], indexing="ij")
    sig_sub = sig_mask[::4, ::4]
    ax.scatter(xx[sig_sub], yy[sig_sub], s=1.5, c="k",
               transform=proj, alpha=0.6, zorder=5)

    # Map features
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),   lw=0.5, linestyle=":")
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   lw=0.3, edgecolor="gray", facecolor="none")
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="none", edgecolor="none")
    ax.set_extent([66, 100, 6, 39], crs=proj)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4,
                      xlocs=[70, 80, 90, 100], ylocs=[10, 20, 30])
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}

    flag = " ★" if ev["aligned"] else ""
    # Mean all-india anom for this event
    ai_anom = float(np.nanmean(anom))
    ax.set_title(
        f"{ev['year']}  {ev['start'].strftime('%b %d')}–{ev['end'].strftime('%b %d')}"
        f"  ({ev['dur']}d){flag}\n"
        f"All-India mean: {ai_anom:+.2f} mm/day",
        fontsize=8.5, fontweight="bold",
        color="#c0392b" if ev["aligned"] else "black"
    )

# Shared colorbar
cbar_ax = fig_b.add_axes([0.15, -0.02, 0.70, 0.018])
sm = plt.cm.ScalarMappable(cmap="BrBG", norm=plt.Normalize(-vlim, vlim))
sm.set_array([])
cbar = fig_b.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Rainfall Anomaly (mm/day)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Hide unused panels
total_slots = nrows * ncols
for j in range(n_ev, total_slots):
    fig_b.add_subplot(nrows, ncols, j+1).set_visible(False)

plt.tight_layout(rect=[0, 0.02, 1, 1.0])
plt.savefig(f"{OUT_DIR}/v2_break_spatial_anomaly.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/v2_break_spatial_anomaly.png", dpi=150, bbox_inches="tight")
print("  Saved: v2_break_spatial_anomaly.pdf/.png")
plt.close()

print(f"\nAll output saved to: {OUT_DIR}")
print("  v2_JJAS_daily_rainfall_byYear.pdf/png")
print("  v2_break_spatial_anomaly.pdf/png")
