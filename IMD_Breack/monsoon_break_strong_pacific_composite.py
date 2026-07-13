"""
Strong Equatorial-Pacific-Phase MJO -- Monsoon Break Composite (1981-2025)
===========================================================================
Continuation of monsoon_break_mjo_analysis_v2.py / monsoon_break_plots_v2.py
(same directory). Re-derives the identical break-event catalogue (IMD
criteria: all-India z < -1, >=3 consecutive days; long breaks >=10 days;
climatology 1981-2010) and then asks a sharper question of it:

  Which long break events occurred under a STRONG, equatorial-Pacific-phase
  MJO -- i.e. suppressed convection over India because the active MJO
  envelope was over the Western Pacific / Western Hemisphere / Western
  Indian Ocean (RMM phases 7, 8, 1), AND the MJO signal was strong
  (event-mean RMM amplitude >= 1.0), not just weakly/ambiguously placed?

Selection criteria (both required):
  (1) MJO-aligned : > 50% of the event's break days fall in RMM phases 7/8/1
  (2) MJO strong  : event-mean RMM amplitude >= 1.0

Two "all-events-in-one-picture" composites are produced for this subset:
  Figure 1 : ONE combined RMM phase-space diagram -- every qualifying
             event's trajectory overlaid on a single WH04 phase wheel
             (vs. the earlier one-panel-per-event grid in v2).
  Figure 2 : ONE composite (event-of-events mean) rainfall anomaly map,
             with block-bootstrap significance stippling (1000 resamples
             of random JJAS periods of matching duration, two-sided,
             p < 0.05) -- i.e. is the composite anomaly bigger than what
             random JJAS spells of the same lengths would give by chance?

Data
----
IMD gridded daily rainfall  : /media/mahendra/T7/IMD_Rainfall/IMD_rain_YYYY.nc
                              (0.25 deg ~ 25 km, Pai et al. 2014), 1901-2025.
                              Only 1981-2025 used here (climatology + events).
RMM (MJO) index             : /home/mahendra/Downloads/rmm.74toRealtime.txt
                              daily, 1974-06-01 to present.

Author : Mahendra (Purdue / Matthew Huber lab)
Date   : 2026-07-13
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import timedelta
import os, warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------------
IMD_DIR    = "/media/mahendra/T7/IMD_Rainfall"
RMM_FILE   = "/home/mahendra/Downloads/rmm.74toRealtime.txt"
OUT_DIR    = "/media/mahendra/T7/IMD_Breack"

CLIM_YRS   = (1981, 2010)   # climatology base period (same as v2)
EVENT_YRS  = (1981, 2025)   # years searched for break events / bootstrap pool
MIN_CONSEC = 3              # IMD minimum consecutive days for a break
MIN_DUR    = 10             # minimum duration kept for MJO analysis (same as v2)

TARGET_PHASES = {7, 8, 1}   # "equatorial Pacific -> India" dry sequence (unchanged from v2)
STRONG_AMP    = 1.0         # event-mean RMM amplitude threshold for "strong" MJO
N_BOOT        = 1000        # block-bootstrap resamples
ALPHA         = 0.05        # two-sided significance level (2.5 / 97.5 pctile)
SMOOTH_HALF   = 7           # +/- days for smoothing the spatial DOY climatology
RNG_SEED      = 42          # reproducibility

PHASE_COLORS = {
    1: "#e41a1c", 2: "#ff7f00", 3: "#d4aa00", 4: "#a65628",
    5: "#4daf4a", 6: "#377eb8", 7: "#984ea3", 8: "#f781bf"
}

rng = np.random.default_rng(RNG_SEED)

# ===========================================================================
# 1. ALL-INDIA DAILY JJAS RAINFALL SERIES (1981-2025) -- identical to v2
# ===========================================================================
print("=" * 78)
print("STEP 1: Building all-India JJAS daily rainfall series (1981-2025) ...")
print("=" * 78)

years = range(EVENT_YRS[0], EVENT_YRS[1] + 1)
ai_list = []
for yr in years:
    fpath = f"{IMD_DIR}/IMD_rain_{yr}.nc"
    if not os.path.exists(fpath):
        print(f"  WARNING: missing {yr}")
        continue
    ds   = xr.open_dataset(fpath)
    rain = ds["rain"]
    jjas = rain.sel(time=rain.time.dt.month.isin([6, 7, 8, 9]))
    ai   = jjas.mean(dim=["lat", "lon"], skipna=True)
    df   = ai.to_dataframe(name="rain").reset_index()
    df["time"] = pd.to_datetime(df["time"])
    ai_list.append(df)
    ds.close()

ai_df = pd.concat(ai_list, ignore_index=True).rename(columns={"time": "date"})
ai_df = ai_df.set_index("date").sort_index()
print(f"  All-India JJAS series: {ai_df.index.min().date()} -> {ai_df.index.max().date()}, "
      f"{len(ai_df)} days")

# ===========================================================================
# 2. JJAS CLIMATOLOGY (1981-2010) & STANDARDISED ANOMALY -- identical to v2
# ===========================================================================
print(f"\nSTEP 2: Computing all-India climatology ({CLIM_YRS[0]}-{CLIM_YRS[1]}) ...")

clim_mask = (ai_df.index.year >= CLIM_YRS[0]) & (ai_df.index.year <= CLIM_YRS[1])
clim_df   = ai_df[clim_mask].copy()
clim_df["doy"] = clim_df.index.dayofyear
doy_mean = clim_df.groupby("doy")["rain"].mean()
doy_std  = clim_df.groupby("doy")["rain"].std()
doy_mean_s = doy_mean.rolling(window=15, center=True, min_periods=7).mean().fillna(doy_mean)
doy_std_s  = doy_std.rolling(window=15, center=True, min_periods=7).mean().fillna(doy_std)

ai_df["doy"]  = ai_df.index.dayofyear
ai_df["mean"] = ai_df["doy"].map(doy_mean_s)
ai_df["std"]  = ai_df["doy"].map(doy_std_s)
ai_df["zscore"] = (ai_df["rain"] - ai_df["mean"]) / ai_df["std"]

# ===========================================================================
# 3. BREAK-DAY / BREAK-EVENT DETECTION -- identical to v2
# ===========================================================================
print("\nSTEP 3: Identifying break events ...")
ai_df["break_day"] = ai_df["zscore"] < -1.0

def flush_event(consec_days, ai_df, events, min_consec):
    if len(consec_days) < min_consec:
        return
    events.append({
        "start_date" : consec_days[0],
        "end_date"   : consec_days[-1],
        "duration"   : len(consec_days),
        "mean_rain"  : round(ai_df.loc[consec_days[0]:consec_days[-1], "rain"].mean(), 2),
        "mean_zscore": round(ai_df.loc[consec_days[0]:consec_days[-1], "zscore"].mean(), 2),
    })

events, consec_days = [], []
for date, row in ai_df.iterrows():
    if row["break_day"]:
        if consec_days and (date - consec_days[-1]).days != 1:
            flush_event(consec_days, ai_df, events, MIN_CONSEC)
            consec_days = []
        consec_days.append(date)
    else:
        if consec_days:
            flush_event(consec_days, ai_df, events, MIN_CONSEC)
            consec_days = []
if consec_days:
    flush_event(consec_days, ai_df, events, MIN_CONSEC)

all_breaks = pd.DataFrame(events)
all_breaks["year"] = all_breaks["start_date"].dt.year
long_breaks = all_breaks[all_breaks["duration"] >= MIN_DUR].reset_index(drop=True)
print(f"  All break events (>={MIN_CONSEC} consecutive days): {len(all_breaks)}")
print(f"  Long break events (>={MIN_DUR} days)              : {len(long_breaks)}")

# ===========================================================================
# 4. LOAD RMM (MJO) INDEX -- identical to v2
# ===========================================================================
print("\nSTEP 4: Loading RMM (MJO) index ...")
rmm = pd.read_csv(
    RMM_FILE, skiprows=2, sep=r'\s+',
    names=["year", "month", "day", "RMM1", "RMM2", "phase", "amplitude", "method"],
    na_values=["1.E36", "999"], engine="python"
)
rmm["date"] = pd.to_datetime(rmm[["year", "month", "day"]])
rmm = rmm.dropna(subset=["phase", "amplitude"])
rmm["phase"] = rmm["phase"].astype(int)
rmm = rmm.set_index("date").sort_index()
print(f"  RMM: {rmm.index.min().date()} -> {rmm.index.max().date()}")

# ===========================================================================
# 5. PER-EVENT MJO ALIGNMENT + AMPLITUDE -- identical to v2
# ===========================================================================
print("\nSTEP 5: Checking MJO phase alignment + amplitude for each long break ...")
results = []
for _, row in long_breaks.iterrows():
    start, end = row["start_date"], row["end_date"]
    slc = rmm.loc[start:end]
    if len(slc) == 0:
        continue
    dom_phase   = int(slc["phase"].mode()[0])
    frac_target = round((slc["phase"].isin(TARGET_PHASES)).mean(), 2)
    mean_amp    = round(slc["amplitude"].mean(), 2)
    results.append({
        "year": row["year"], "start_date": start, "end_date": end,
        "duration": int(row["duration"]), "mean_rain": row["mean_rain"],
        "mean_zscore": row["mean_zscore"], "dominant_phase": dom_phase,
        "frac_phase781": frac_target, "mean_amplitude": mean_amp,
        "aligned_781": frac_target > 0.5,
    })
res = pd.DataFrame(results)

# ===========================================================================
# 6. SELECT "STRONG EQUATORIAL-PACIFIC-PHASE" BREAK EVENTS  <-- NEW
# ===========================================================================
print("\nSTEP 6: Selecting strong equatorial-Pacific-phase break events ...")
strong = res[res["aligned_781"] & (res["mean_amplitude"] >= STRONG_AMP)].reset_index(drop=True)

print(f"\n{'='*90}")
print(f"All long breaks (>={MIN_DUR}d)            : {len(res)}")
print(f"  ... MJO-aligned (phases 7/8/1 >50%)    : {int(res['aligned_781'].sum())}")
print(f"  ... aligned AND amplitude >= {STRONG_AMP:<4}  : {len(strong)}   <-- COMPOSITE SET")
print(f"{'='*90}")
cols = ["year", "start_date", "end_date", "duration", "dominant_phase",
        "frac_phase781", "mean_amplitude", "mean_rain", "mean_zscore"]
print(strong[cols].to_string(index=False))

if len(strong) < 2:
    print("\n  WARNING: fewer than 2 qualifying events -- composite/bootstrap will be "
          "very noisy. Consider relaxing STRONG_AMP.")

strong_csv = strong.copy()
strong_csv["start_date"] = strong_csv["start_date"].dt.strftime("%Y-%m-%d")
strong_csv["end_date"]   = strong_csv["end_date"].dt.strftime("%Y-%m-%d")
strong_csv.to_csv(f"{OUT_DIR}/strong_pacific_break_events.csv", index=False)
print(f"\n  Saved event table: {OUT_DIR}/strong_pacific_break_events.csv")

# ===========================================================================
# 7. LOAD FULL SPATIAL RECORD (1981-2025, JJAS only) INTO A PER-YEAR CACHE
# ===========================================================================
print("\nSTEP 7: Loading gridded rainfall (1981-2025, JJAS) for spatial composite ...")
year_data = {}   # year -> dict(dates=np.datetime64[], doy=int[], rain=(ndays,nlat,nlon))
lats = lons = None
for yr in years:
    fpath = f"{IMD_DIR}/IMD_rain_{yr}.nc"
    if not os.path.exists(fpath):
        continue
    ds   = xr.open_dataset(fpath)
    rain = ds["rain"].sel(time=ds["rain"].time.dt.month.isin([6, 7, 8, 9])).load()
    if lats is None:
        lats = rain.lat.values.copy()
        lons = rain.lon.values.copy()
    year_data[yr] = {
        "dates": pd.to_datetime(rain.time.values),
        "doy"  : rain.time.dt.dayofyear.values,
        "rain" : rain.values.astype(np.float32),   # (ndays, nlat, nlon)
    }
    ds.close()
print(f"  Loaded {len(year_data)} years, grid = {len(lats)} lat x {len(lons)} lon")

# --- Precompute smoothed spatial DOY climatology (1981-2010) once, cached ---
print(f"  Building smoothed spatial DOY climatology ({CLIM_YRS[0]}-{CLIM_YRS[1]}, "
      f"+/-{SMOOTH_HALF}d window) ...")
clim_doy_list, clim_rain_list = [], []
for yr in range(CLIM_YRS[0], CLIM_YRS[1] + 1):
    if yr not in year_data:
        continue
    clim_doy_list.append(year_data[yr]["doy"])
    clim_rain_list.append(year_data[yr]["rain"])
clim_doy_arr  = np.concatenate(clim_doy_list)
clim_rain_arr = np.concatenate(clim_rain_list, axis=0)

doy_clim_mean = {}
for d in range(140, 285):   # generous buffer around JJAS (152-273 nominal)
    window = set(range(d - SMOOTH_HALF, d + SMOOTH_HALF + 1))
    mask = np.isin(clim_doy_arr, list(window))
    if mask.sum() == 0:
        continue
    doy_clim_mean[d] = np.nanmean(clim_rain_arr[mask], axis=0)

def clim_for_doys(doy_arr):
    """Mean of the precomputed smoothed per-day climatology over a set of DOYs."""
    maps = [doy_clim_mean[int(d)] for d in doy_arr if int(d) in doy_clim_mean]
    return np.nanmean(np.stack(maps), axis=0)

def event_anomaly(year, start_date, end_date):
    """Mean anomaly map (event mean rain - smoothed DOY climatology) for a real event."""
    dat  = year_data[year]
    mask = (dat["dates"] >= start_date) & (dat["dates"] <= end_date)
    block_rain = np.nanmean(dat["rain"][mask], axis=0)
    clim       = clim_for_doys(dat["doy"][mask])
    return block_rain - clim

def synthetic_block_anomaly(duration, avail_years):
    """Random contiguous JJAS block of `duration` days (block bootstrap null draw)."""
    for _ in range(50):   # retry guard in case a short year is picked
        yr  = avail_years[rng.integers(0, len(avail_years))]
        dat = year_data[yr]
        n   = len(dat["dates"])
        if n <= duration:
            continue
        start_idx = rng.integers(0, n - duration + 1)
        idx = slice(start_idx, start_idx + duration)
        block_rain = np.nanmean(dat["rain"][idx], axis=0)
        clim       = clim_for_doys(dat["doy"][idx])
        return block_rain - clim
    raise RuntimeError("Could not draw a synthetic block -- check year_data.")

avail_years = np.array(sorted(year_data.keys()))

# ===========================================================================
# 8. REAL COMPOSITE (mean of the qualifying events' anomaly maps)
# ===========================================================================
print("\nSTEP 8: Computing real composite anomaly (event-of-events mean) ...")
real_anoms = []
durations  = []
for _, ev in strong.iterrows():
    a = event_anomaly(int(ev["year"]), ev["start_date"], ev["end_date"])
    real_anoms.append(a)
    durations.append(int(ev["duration"]))
composite_real = np.nanmean(np.stack(real_anoms), axis=0)

# ===========================================================================
# 9. BLOCK BOOTSTRAP SIGNIFICANCE TEST (N_BOOT resamples)
# ===========================================================================
print(f"\nSTEP 9: Block-bootstrap significance test ({N_BOOT} resamples, "
      f"two-sided p<{ALPHA}) ...")
boot_composites = np.empty((N_BOOT, len(lats), len(lons)), dtype=np.float32)
for b in range(N_BOOT):
    block_anoms = [synthetic_block_anomaly(d, avail_years) for d in durations]
    boot_composites[b] = np.nanmean(np.stack(block_anoms), axis=0)
    if (b + 1) % 200 == 0:
        print(f"    ... {b+1}/{N_BOOT} resamples done")

lower = np.nanpercentile(boot_composites, 100 * ALPHA / 2, axis=0)
upper = np.nanpercentile(boot_composites, 100 * (1 - ALPHA / 2), axis=0)
sig_mask = (composite_real < lower) | (composite_real > upper)
n_land   = np.sum(~np.isnan(composite_real))
n_sig    = np.sum(sig_mask & ~np.isnan(composite_real))
print(f"  Significant land grid points: {n_sig} / {n_land} "
      f"({100*n_sig/max(n_land,1):.1f}%)")

# ===========================================================================
# 10. FIGURE 1 -- COMBINED MJO PHASE-SPACE DIAGRAM (all events, ONE picture)
# ===========================================================================
print("\nSTEP 10: Plotting combined MJO phase-space diagram ...")

fig1, ax1 = plt.subplots(figsize=(8, 8))

# WH04 8-phase background sectors
for ph in range(1, 9):
    a0 = ((ph + 3) % 8) * np.pi / 4
    a1 = a0 + np.pi / 4
    th = np.linspace(a0, a1, 60)
    r  = 4.0
    ax1.fill(np.concatenate([[0], r*np.cos(th), [0]]),
              np.concatenate([[0], r*np.sin(th), [0]]),
              color=PHASE_COLORS[ph],
              alpha=0.35 if ph in TARGET_PHASES else 0.08, zorder=0)
    mid = a0 + np.pi / 8
    ax1.text(3.3*np.cos(mid), 3.3*np.sin(mid), str(ph),
             ha="center", va="center", fontsize=11, fontweight="bold",
             color=PHASE_COLORS[ph])
ax1.add_patch(plt.Circle((0, 0), 1.0, color="gray", fill=False, lw=1.2, ls="--"))

geo_labels = {
    (1, 2): ("W. Indian\nOcean", 225.0),
    (3,)  : ("India /\nI. Ocean", 292.5),
    (4, 5): ("Maritime\nContinent", 337.5 + 22.5),
    (6, 7): ("Western\nPacific", 67.5 + 22.5),
    (8,)  : ("W. Hem. /\nAfrica", 157.5),
}
for phs, (lbl, ang_deg) in geo_labels.items():
    rad = np.radians(ang_deg)
    ax1.text(3.75*np.cos(rad), 3.75*np.sin(rad), lbl,
             ha="center", va="center", fontsize=7.5, color="dimgray", style="italic")

event_cmap = plt.cm.tab10(np.linspace(0, 1, max(len(strong), 1)))
for i, (_, ev) in enumerate(strong.iterrows()):
    slc = rmm.loc[ev["start_date"]:ev["end_date"]]
    if len(slc) == 0:
        continue
    col = event_cmap[i % 10]
    label = (f"{int(ev['year'])} {ev['start_date'].strftime('%b %d')}-"
             f"{ev['end_date'].strftime('%b %d')} ({int(ev['duration'])}d, "
             f"amp={ev['mean_amplitude']:.2f})")
    ax1.plot(slc["RMM1"], slc["RMM2"], color=col, lw=2.2, alpha=0.9, zorder=5, label=label)
    ax1.scatter(slc["RMM1"].iloc[0], slc["RMM2"].iloc[0], marker="^", s=90,
                color=col, edgecolors="k", lw=0.6, zorder=6)
    ax1.scatter(slc["RMM1"].iloc[-1], slc["RMM2"].iloc[-1], marker="s", s=90,
                color=col, edgecolors="k", lw=0.6, zorder=6)

ax1.set_xlim(-4.5, 4.5); ax1.set_ylim(-4.5, 4.5); ax1.set_aspect("equal")
ax1.axhline(0, color="k", lw=0.6); ax1.axvline(0, color="k", lw=0.6)
ax1.set_xlabel("RMM1", fontsize=11); ax1.set_ylabel("RMM2", fontsize=11)
ax1.set_title(
    "Combined MJO Phase-Space Trajectories\n"
    "Strong Equatorial-Pacific-Phase Monsoon Break Events (1981-2025)\n"
    f">50% of break days in phases 7/8/1  &  mean amplitude >= {STRONG_AMP}"
    "  |  ▲ start  ■ end",
    fontsize=11, fontweight="bold")
ax1.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
           title="Break event", frameon=True)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/strong_pacific_mjo_wheel_combined.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/strong_pacific_mjo_wheel_combined.png", dpi=150, bbox_inches="tight")
print("  Saved: strong_pacific_mjo_wheel_combined.pdf/.png")
plt.close(fig1)

# ===========================================================================
# 11. FIGURE 2 -- COMPOSITE RAINFALL ANOMALY MAP WITH SIGNIFICANCE STIPPLING
# ===========================================================================
print("\nSTEP 11: Plotting composite rainfall anomaly with significance stippling ...")

vlim = np.nanpercentile(np.abs(composite_real), 97)
vlim = max(np.round(vlim / 2) * 2, 2)

proj = ccrs.PlateCarree()
fig2 = plt.figure(figsize=(9, 8))
ax2  = fig2.add_subplot(1, 1, 1, projection=proj)

im = ax2.pcolormesh(lons, lats, composite_real, transform=proj,
                     cmap="BrBG", vmin=-vlim, vmax=vlim, shading="auto")

# Stipple where the bootstrap test flags significance (p < ALPHA, two-sided)
yy, xx = np.meshgrid(lats[::3], lons[::3], indexing="ij")
sig_sub = sig_mask[::3, ::3]
ax2.scatter(xx[sig_sub], yy[sig_sub], s=3, c="k", transform=proj,
            alpha=0.65, zorder=5, label=f"p < {ALPHA} (bootstrap, n={N_BOOT})")

ax2.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7)
ax2.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5, linestyle=":")
ax2.add_feature(cfeature.STATES.with_scale("50m"), lw=0.3, edgecolor="gray", facecolor="none")
ax2.set_extent([66, 100, 6, 39], crs=proj)

gl = ax2.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4,
                    xlocs=[70, 80, 90, 100], ylocs=[10, 20, 30])
gl.top_labels = False; gl.right_labels = False
gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}

ai_anom = float(np.nanmean(composite_real))
ax2.set_title(
    f"Composite Rainfall Anomaly -- Strong Equatorial-Pacific-Phase\n"
    f"Monsoon Break Events (n={len(strong)}, 1981-2025)\n"
    f"Anomaly = event-mean rain minus {CLIM_YRS[0]}-{CLIM_YRS[1]} DOY climatology "
    f"(mm/day)  |  All-India mean: {ai_anom:+.2f} mm/day\n"
    f"Stippling: block-bootstrap significant at p<{ALPHA} ({N_BOOT} resamples)",
    fontsize=10.5, fontweight="bold")

cbar = fig2.colorbar(im, ax=ax2, orientation="horizontal", pad=0.08, shrink=0.85)
cbar.set_label("Rainfall Anomaly (mm/day)", fontsize=10)
cbar.ax.tick_params(labelsize=8)
ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/strong_pacific_rainfall_composite_significance.pdf",
            dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/strong_pacific_rainfall_composite_significance.png",
            dpi=150, bbox_inches="tight")
print("  Saved: strong_pacific_rainfall_composite_significance.pdf/.png")
plt.close(fig2)

# ===========================================================================
print(f"\nAll output saved to: {OUT_DIR}")
print("  strong_pacific_break_events.csv")
print("  strong_pacific_mjo_wheel_combined.pdf/png")
print("  strong_pacific_rainfall_composite_significance.pdf/png")
print("\nDone.")
