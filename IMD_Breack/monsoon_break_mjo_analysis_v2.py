"""
Monsoon Break Events vs MJO Phase Analysis — v2 (1981–2025)
============================================================
Break detection uses IMD's official criteria applied to all-India
gridded rainfall (Pai et al. 2014 dataset, 0.25° resolution):
  • Spatial mean over ALL land grid points (non-NaN, 6.5–38.5°N, 66.5–100°E)
  • Break day  : standardised anomaly (z-score) < -1.0  [JJAS climatology 1981–2010]
  • Break event: ≥ 3 CONSECUTIVE break days (IMD minimum persistence criterion)
  • Long break : event duration ≥ 10 days (for MJO analysis)

MJO alignment criterion: > 50% of break days in phases 7, 8, or 1
(suppressed convection over Indian Ocean / India)

Author: Mahendra (Purdue / Matthew Huber lab)
Date  : 2026-06-18
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from datetime import timedelta
import glob, os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
RMM_FILE  = "/home/mahendra/Downloads/rmm.74toRealtime.txt"
IMD_DIR   = "/home/mahendra/Downloads/IMD_Rainfall"
OUT_DIR   = "/home/mahendra"
CLIM_YRS  = (1981, 2010)   # climatology base period
MIN_CONSEC = 3             # IMD minimum consecutive days for a break
MIN_DUR    = 10            # minimum duration to keep for MJO analysis

# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD ALL-INDIA DAILY RAINFALL TIME SERIES (JJAS, 1981–2025)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading IMD daily rainfall — all India (land points only) …")

years = range(1981, 2026)
ai_list = []

for yr in years:
    fpath = f"{IMD_DIR}/IMD_rain_{yr}.nc"
    if not os.path.exists(fpath):
        print(f"  WARNING: missing {yr}")
        continue
    ds   = xr.open_dataset(fpath)
    rain = ds["rain"]
    # JJAS only
    jjas = rain.sel(time=rain.time.dt.month.isin([6, 7, 8, 9]))
    # All-India spatial mean (NaN = ocean, skipna averages land only)
    ai   = jjas.mean(dim=["lat", "lon"], skipna=True)
    df   = ai.to_dataframe(name="rain").reset_index()
    df["time"] = pd.to_datetime(df["time"])
    ai_list.append(df)
    ds.close()

ai_df = pd.concat(ai_list, ignore_index=True).rename(columns={"time": "date"})
ai_df = ai_df.set_index("date").sort_index()
print(f"  All-India JJAS series: {ai_df.index.min().date()} → {ai_df.index.max().date()}, "
      f"{len(ai_df)} days")

# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE JJAS CLIMATOLOGY (1981–2010) AND STANDARDISE
# ─────────────────────────────────────────────────────────────────────────────
print(f"Computing climatology ({CLIM_YRS[0]}–{CLIM_YRS[1]}) …")

clim_mask = (ai_df.index.year >= CLIM_YRS[0]) & (ai_df.index.year <= CLIM_YRS[1])
clim_df   = ai_df[clim_mask].copy()

# Day-of-year climatological mean and std (smoothed with 15-day rolling window
# to remove sampling noise in rare DOYs)
clim_df["doy"] = clim_df.index.day_of_year
doy_mean = clim_df.groupby("doy")["rain"].mean()
doy_std  = clim_df.groupby("doy")["rain"].std()

# Smooth with a 15-day centred rolling window to stabilise DOY statistics
doy_mean_s = doy_mean.rolling(window=15, center=True, min_periods=7).mean()
doy_std_s  = doy_std.rolling(window=15, center=True, min_periods=7).mean()
doy_mean_s = doy_mean_s.fillna(doy_mean)
doy_std_s  = doy_std_s.fillna(doy_std)

# Standardise full series
ai_df["doy"]  = ai_df.index.day_of_year
ai_df["mean"] = ai_df["doy"].map(doy_mean_s)
ai_df["std"]  = ai_df["doy"].map(doy_std_s)
ai_df["zscore"] = (ai_df["rain"] - ai_df["mean"]) / ai_df["std"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. IDENTIFY BREAK DAYS (z < -1) AND GROUP INTO EVENTS
# ─────────────────────────────────────────────────────────────────────────────
print("Identifying break events …")

ai_df["break_day"] = ai_df["zscore"] < -1.0

def flush_event(consec_days, ai_df, events, min_consec):
    """Record an event only if it has ≥ min_consec CALENDAR-consecutive days."""
    if len(consec_days) < min_consec:
        return
    events.append({
        "start_date" : consec_days[0],
        "end_date"   : consec_days[-1],
        "duration"   : len(consec_days),
        "mean_rain"  : round(ai_df.loc[consec_days[0]:consec_days[-1], "rain"].mean(), 2),
        "mean_zscore": round(ai_df.loc[consec_days[0]:consec_days[-1], "zscore"].mean(), 2),
        "min_zscore" : round(ai_df.loc[consec_days[0]:consec_days[-1], "zscore"].min(), 2),
        "clim_rain"  : round(ai_df.loc[consec_days[0]:consec_days[-1], "mean"].mean(), 2),
    })

# Group calendar-consecutive break days
events      = []
consec_days = []

for date, row in ai_df.iterrows():
    if row["break_day"]:
        # Start new streak OR continue only if calendar-consecutive (1-day gap)
        if consec_days and (date - consec_days[-1]).days != 1:
            flush_event(consec_days, ai_df, events, MIN_CONSEC)
            consec_days = []
        consec_days.append(date)
    else:
        if consec_days:
            flush_event(consec_days, ai_df, events, MIN_CONSEC)
            consec_days = []

# Close any open event at end of series
if consec_days:
    flush_event(consec_days, ai_df, events, MIN_CONSEC)

all_breaks = pd.DataFrame(events)
all_breaks["year"] = all_breaks["start_date"].dt.year
all_breaks["month"] = all_breaks["start_date"].dt.month

print(f"  All break events (≥{MIN_CONSEC} consecutive days): {len(all_breaks)}")
long_breaks = all_breaks[all_breaks["duration"] >= MIN_DUR].reset_index(drop=True)
print(f"  Long break events (≥{MIN_DUR} days)              : {len(long_breaks)}")
print()
print(long_breaks[["year","start_date","end_date","duration","mean_rain","clim_rain","mean_zscore"]].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD RMM DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading RMM data …")
rmm = pd.read_csv(
    RMM_FILE,
    skiprows=2,
    sep=r'\s+',
    names=["year","month","day","RMM1","RMM2","phase","amplitude","method"],
    na_values=["1.E36","999"],
    engine="python"
)
rmm["date"] = pd.to_datetime(rmm[["year","month","day"]])
rmm = rmm.dropna(subset=["phase","amplitude"])
rmm["phase"] = rmm["phase"].astype(int)
rmm = rmm.set_index("date").sort_index()
print(f"  RMM: {rmm.index.min().date()} → {rmm.index.max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CHECK MJO PHASE ALIGNMENT FOR EACH LONG BREAK
# ─────────────────────────────────────────────────────────────────────────────
TARGET_PHASES = {7, 8, 1}

results = []
for _, row in long_breaks.iterrows():
    start = row["start_date"]
    end   = row["end_date"]
    slc   = rmm.loc[start:end]

    if len(slc) == 0:
        results.append({**row, "dominant_phase": np.nan, "frac_phase781": 0.0,
                        "mean_amplitude": np.nan, "aligned_781": False})
        continue

    dom_phase    = int(slc["phase"].mode()[0])
    frac_target  = round((slc["phase"].isin(TARGET_PHASES)).mean(), 2)
    mean_amp     = round(slc["amplitude"].mean(), 2)
    is_aligned   = frac_target > 0.5

    results.append({
        "year"          : row["year"],
        "start_date"    : start,
        "end_date"      : end,
        "duration"      : row["duration"],
        "mean_rain"     : row["mean_rain"],
        "clim_rain"     : row["clim_rain"],
        "mean_zscore"   : row["mean_zscore"],
        "dominant_phase": dom_phase,
        "frac_phase781" : frac_target,
        "mean_amplitude": mean_amp,
        "aligned_781"   : is_aligned,
    })

res = pd.DataFrame(results)
aligned     = res[res["aligned_781"]].reset_index(drop=True)
not_aligned = res[~res["aligned_781"]].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"All Break Events ≥ {MIN_DUR} days | All-India Rainfall | IMD criteria (z < -1, ≥{MIN_CONSEC} consec. days)")
print(f"{'='*90}")
print(f"{'Year':<6} {'Start':>12} {'End':>12} {'Dur':>4} {'DomPh':>6} "
      f"{'%781':>6} {'Amp':>5} {'Rain':>5} {'Clim':>5} {'Z':>6} {'Aligned':>10}")
print("-"*90)
for _, r in res.sort_values("start_date").iterrows():
    flag = "  YES ★" if r["aligned_781"] else "  no"
    print(f"{int(r['year']):<6} {str(r['start_date'].date()):>12} {str(r['end_date'].date()):>12} "
          f"{int(r['duration']):>4} {int(r['dominant_phase']):>6} "
          f"{r['frac_phase781']*100:>5.0f}% {r['mean_amplitude']:>5.2f} "
          f"{r['mean_rain']:>5.2f} {r['clim_rain']:>5.2f} {r['mean_zscore']:>6.2f} {flag}")
print("-"*90)
print(f"Total: {len(res)} break events ≥{MIN_DUR} days | "
      f"Aligned with phases 7/8/1: {res['aligned_781'].sum()} "
      f"({res['aligned_781'].mean()*100:.0f}%)")
print()
print("ALIGNED EVENTS:")
cols = ["year","start_date","end_date","duration","dominant_phase","frac_phase781","mean_amplitude","mean_rain","mean_zscore"]
print(aligned[cols].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 1: Duration bars + % days in Ph7/8/1
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures …")

PHASE_COLORS = {
    1: "#e41a1c", 2: "#ff7f00", 3: "#d4aa00", 4: "#a65628",
    5: "#4daf4a", 6: "#377eb8", 7: "#984ea3", 8: "#f781bf"
}
TARGET_COLOR = "#d62728"
OTHER_COLOR  = "#1f77b4"

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(
    "Indian Monsoon Break Events (≥10 days, 1981–2025)\n"
    "All-India Rainfall | IMD Criteria: z < −1, ≥3 Consecutive Days",
    fontsize=13, fontweight="bold")

bar_colors = [TARGET_COLOR if a else OTHER_COLOR for a in res["aligned_781"]]
xlabels    = [f"{int(r.year)}\n{r.start_date.strftime('%b %d')}" for _, r in res.iterrows()]

# Panel A
ax = axes[0]
ax.bar(range(len(res)), res["duration"], color=bar_colors, edgecolor="k", lw=0.6, width=0.7)
for i, (_, r) in enumerate(res.iterrows()):
    ax.text(i, r["duration"] + 0.15, f"Ph{int(r['dominant_phase'])}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(range(len(res)))
ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Duration (days)", fontsize=10)
ax.set_title("(a) Duration of each break event  |  Red = MJO phases 7/8/1 dominant", fontsize=10)
ax.axhline(10, color="gray", ls="--", lw=1)
ax.set_ylim(0, res["duration"].max() + 4)
ax.legend(handles=[
    mpatches.Patch(color=TARGET_COLOR, label="Aligned with phases 7/8/1 (>50% days)"),
    mpatches.Patch(color=OTHER_COLOR,  label="Not aligned")
], fontsize=9, loc="upper right")

# Panel B
ax2 = axes[1]
ax2.bar(range(len(res)), res["frac_phase781"]*100, color=bar_colors, edgecolor="k", lw=0.6, width=0.7)
ax2.axhline(50, color="gray", ls="--", lw=1.2, label="50% threshold")
ax2.set_xticks(range(len(res)))
ax2.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8.5)
ax2.set_ylabel("% break days in MJO phases 7, 8, 1", fontsize=10)
ax2.set_title("(b) Fraction of break days in MJO phases 7, 8, 1 (suppressed convection over India)",
              fontsize=10)
ax2.set_ylim(0, 108)
ax2.legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/v2_break_mjo_alignment.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/v2_break_mjo_alignment.png", dpi=150, bbox_inches="tight")
print("  Saved: v2_break_mjo_alignment.pdf/.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 2: RMM phase-wheel for each long break
# ─────────────────────────────────────────────────────────────────────────────
def plot_rmm_wheel(ax, start, end, title=""):
    slc = rmm.loc[start:end].copy()
    if len(slc) == 0:
        ax.set_title(title + "\n(no RMM data)"); return

    # Correct WH04 sector boundaries: Phase 1 at 180°-225°, going CCW
    # Formula: ang_start = ((ph + 3) % 8) * π/4
    for ph in range(1, 9):
        a0 = ((ph + 3) % 8) * np.pi / 4
        a1 = a0 + np.pi / 4
        th = np.linspace(a0, a1, 60)
        r  = 4.0
        ax.fill(np.concatenate([[0], r*np.cos(th), [0]]),
                np.concatenate([[0], r*np.sin(th), [0]]),
                color=PHASE_COLORS[ph],
                alpha=0.45 if ph in TARGET_PHASES else 0.13, zorder=0)
        mid = a0 + np.pi / 8
        ax.text(3.2*np.cos(mid), 3.2*np.sin(mid), str(ph),
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=PHASE_COLORS[ph])

    ax.add_patch(plt.Circle((0,0), 1.0, color="gray", fill=False, lw=1, ls="--"))

    # Geographic region labels (WH04 standard)
    geo_labels = {
        (1, 2)  : ("W.Ind.\nOcean", 225.0),
        (3,)    : ("India /\nI.Ocean",292.5),
        (4, 5)  : ("Mar.\nCont.",   337.5+22.5),
        (6, 7)  : ("W.\nPacific",   67.5+22.5),
        (8,)    : ("W.Hem /\nAfrica",157.5),
    }
    for phs, (lbl, ang_deg) in geo_labels.items():
        rad = np.radians(ang_deg)
        ax.text(3.7*np.cos(rad), 3.7*np.sin(rad), lbl,
                ha="center", va="center", fontsize=5, color="dimgray",
                style="italic")

    n = len(slc)
    cmap_vals = plt.cm.plasma(np.linspace(0, 1, n))
    for i in range(n-1):
        ax.plot(slc["RMM1"].iloc[i:i+2], slc["RMM2"].iloc[i:i+2],
                color=cmap_vals[i], lw=2.0, alpha=0.85)
    ax.scatter(slc["RMM1"], slc["RMM2"], c=np.arange(n), cmap="plasma",
               s=30, zorder=5, edgecolors="k", lw=0.3)
    ax.scatter(slc["RMM1"].iloc[0], slc["RMM2"].iloc[0], s=90, marker="^",
               color="green", zorder=10, label="Start")
    ax.scatter(slc["RMM1"].iloc[-1], slc["RMM2"].iloc[-1], s=90, marker="s",
               color="red", zorder=10, label="End")
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("RMM1", fontsize=7); ax.set_ylabel("RMM2", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8, fontweight="bold")

n_ev  = len(res)
ncols = 4
nrows = int(np.ceil(n_ev / ncols))
fig2, ax2s = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
flat = ax2s.flatten() if n_ev > 1 else [ax2s]

fig2.suptitle(
    "MJO Phase-Space Trajectories During All-India Monsoon Break Events (≥10 days)\n"
    "Shaded: phases 7/8/1 | ▲ start  ■ end  (dark→light = early→late in event)",
    fontsize=10, fontweight="bold", y=1.01)

for i, (_, row) in enumerate(res.iterrows()):
    flag  = " ★" if row["aligned_781"] else ""
    title = (f"{int(row['year'])}  {row['start_date'].strftime('%b %d')}–"
             f"{row['end_date'].strftime('%b %d')}  ({int(row['duration'])}d)\n"
             f"DomPh={int(row['dominant_phase'])}  Ph781={row['frac_phase781']*100:.0f}%  "
             f"z={row['mean_zscore']:.2f}{flag}")
    plot_rmm_wheel(flat[i], row["start_date"], row["end_date"], title=title)

for j in range(n_ev, len(flat)):
    flat[j].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/v2_break_rmm_wheels.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/v2_break_rmm_wheels.png", dpi=150, bbox_inches="tight")
print("  Saved: v2_break_rmm_wheels.pdf/.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURE 3: Daily timeline — all-India rain + z-score + MJO phase shading
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(len(res), 1, figsize=(15, 3.8*len(res)), squeeze=False)
fig3.suptitle(
    "Daily MJO Phase, All-India Rainfall & Standardised Anomaly (z-score)\n"
    "During Monsoon Break Events ≥ 10 Days (1981–2025)  |  IMD Criteria: z < −1, ≥3 Consec. Days",
    fontsize=11, fontweight="bold")

for idx, (_, row) in enumerate(res.iterrows()):
    ax_main = axes3[idx, 0]
    start   = row["start_date"]
    end     = row["end_date"]
    # ±5 day padding for context
    pad     = timedelta(days=5)
    p_start = start - pad
    p_end   = end   + pad

    # Pull data with padding
    slc_ai  = ai_df.loc[p_start:p_end]
    slc_rmm = rmm.loc[p_start:p_end][["phase","amplitude"]]

    ax_main.set_xlim(p_start, p_end)

    # Phase background shading (only over the break period)
    for d, ph in zip(slc_rmm.index, slc_rmm["phase"]):
        col   = PHASE_COLORS.get(int(ph), "gray")
        alpha = 0.55 if int(ph) in TARGET_PHASES else 0.18
        ax_main.axvspan(d, d + timedelta(days=1), color=col, alpha=alpha, lw=0)

    # Shade the official break period
    ax_main.axvspan(start, end + timedelta(days=1), color="none",
                    edgecolor="black", lw=2, linestyle="--", zorder=3)
    ax_main.axvline(start, color="k", lw=1.5, ls="--", zorder=4)
    ax_main.axvline(end + timedelta(days=1), color="k", lw=1.5, ls="--", zorder=4)

    # All-India rainfall bars
    ax_rain = ax_main.twinx()
    ax_rain.bar(slc_ai.index, slc_ai["rain"], width=0.85,
                color="steelblue", alpha=0.50, label="All-India Rain (mm/day)", zorder=2)
    ax_rain.plot(slc_ai.index, slc_ai["mean"], color="navy", lw=1.5,
                 ls="-", label="Climatology", zorder=3)
    ax_rain.set_ylabel("Rain (mm/day)", fontsize=8, color="steelblue")
    ax_rain.tick_params(axis="y", colors="steelblue", labelsize=7)
    ax_rain.set_ylim(0, max(slc_ai["rain"].max() * 1.6, 12))

    # Z-score line
    ax_z = ax_main.twinx()
    ax_z.spines["right"].set_position(("outward", 58))
    ax_z.plot(slc_ai.index, slc_ai["zscore"], color="darkred", lw=2.0,
              label="Z-score", zorder=5)
    ax_z.axhline(-1.0, color="darkred", lw=1.0, ls=":", alpha=0.7, label="z = -1 (break threshold)")
    ax_z.axhline(0.0,  color="gray",    lw=0.8, ls="--", alpha=0.5)
    ax_z.set_ylabel("Z-score", fontsize=8, color="darkred")
    ax_z.tick_params(axis="y", colors="darkred", labelsize=7)
    zlim = max(abs(slc_ai["zscore"].min()), abs(slc_ai["zscore"].max()), 2.5)
    ax_z.set_ylim(-zlim*1.4, zlim*1.4)

    # MJO amplitude
    ax_amp = ax_main.twinx()
    ax_amp.spines["right"].set_position(("outward", 115))
    ax_amp.plot(slc_rmm.index, slc_rmm["amplitude"], "k-", lw=1.5,
                alpha=0.7, label="MJO Amplitude", zorder=4)
    ax_amp.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.5)
    ax_amp.set_ylabel("MJO Amp.", fontsize=7, color="k")
    ax_amp.tick_params(labelsize=6)
    ax_amp.set_ylim(0, max(slc_rmm["amplitude"].max()*1.3, 2.5))

    # Phase numbers along top
    for d, ph in zip(slc_rmm.index, slc_rmm["phase"]):
        ax_main.text(d + timedelta(hours=12), 0.96, str(int(ph)),
                     ha="center", va="top", fontsize=6, fontweight="bold",
                     color="white" if int(ph) in TARGET_PHASES else "dimgray",
                     transform=ax_main.get_xaxis_transform(), zorder=6)

    flag_str = " ★ Aligned" if row["aligned_781"] else ""
    ax_main.set_title(
        f"{int(row['year'])}  {start.strftime('%b %d')} – {end.strftime('%b %d')}  "
        f"({int(row['duration'])} days)  |  Mean rain = {row['mean_rain']:.2f} mm/d  "
        f"|  Mean z = {row['mean_zscore']:.2f}  "
        f"|  Dom. Phase = {int(row['dominant_phase'])}  "
        f"|  Ph7/8/1 fraction = {row['frac_phase781']*100:.0f}%{flag_str}",
        fontsize=9, fontweight="bold",
        color=TARGET_COLOR if row["aligned_781"] else "black")
    ax_main.set_yticks([])

    # Legend (first panel only)
    if idx == 0:
        ph_handles = [mpatches.Patch(color=PHASE_COLORS[p], label=f"Ph{p}",
                                     alpha=0.7) for p in range(1,9)]
        ax_main.legend(handles=ph_handles, loc="lower left", fontsize=6.5,
                       ncol=8, title="MJO Phase (shading)", framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.975])
plt.savefig(f"{OUT_DIR}/v2_break_daily_timeline.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/v2_break_daily_timeline.png", dpi=150, bbox_inches="tight")
print("  Saved: v2_break_daily_timeline.pdf/.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 10. FIGURE 4: All-India JJAS rainfall anomaly time series with break events
# ─────────────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(18, 5))
ax4.fill_between(ai_df.index, ai_df["zscore"], 0,
                 where=ai_df["zscore"] < 0, color="tomato",  alpha=0.4, label="Below normal")
ax4.fill_between(ai_df.index, ai_df["zscore"], 0,
                 where=ai_df["zscore"] > 0, color="royalblue", alpha=0.4, label="Above normal")
ax4.axhline(-1.0, color="darkred", lw=1.2, ls="--", label="z = −1 (break threshold)")
ax4.axhline( 1.0, color="steelblue", lw=0.8, ls="--", alpha=0.5)
ax4.axhline( 0.0, color="k", lw=0.6)

# Mark long break events
for _, r in res.iterrows():
    col  = TARGET_COLOR if r["aligned_781"] else "gray"
    ax4.axvspan(r["start_date"], r["end_date"] + timedelta(days=1),
                color=col, alpha=0.3, zorder=3)
    mid_date = r["start_date"] + (r["end_date"] - r["start_date"]) / 2
    ax4.text(mid_date, ax4.get_ylim()[0] if ax4.get_ylim()[0] != 0 else -3.5,
             f"{int(r['year'])}\n{r['start_date'].strftime('%b')}",
             ha="center", va="bottom", fontsize=7, fontweight="bold",
             color=col, zorder=4)

ax4.set_xlim(ai_df.index.min(), ai_df.index.max())
ax4.set_ylabel("Standardised Anomaly (z-score)", fontsize=10)
ax4.set_title("All-India JJAS Daily Rainfall — Standardised Anomaly (1981–2025)\n"
              "Red shading: break events ≥10 days aligned with MJO 7/8/1  |  "
              "Grey: break events not aligned", fontsize=10)
ax4.legend(fontsize=9, loc="lower right")
ax4.set_ylim(-4.5, 4.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/v2_allIndia_zscore_timeseries.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/v2_allIndia_zscore_timeseries.png", dpi=150, bbox_inches="tight")
print("  Saved: v2_allIndia_zscore_timeseries.pdf/.png")
plt.close()

print(f"\nAll output saved to: {OUT_DIR}")
print("  v2_break_mjo_alignment.pdf/png")
print("  v2_break_rmm_wheels.pdf/png")
print("  v2_break_daily_timeline.pdf/png")
print("  v2_allIndia_zscore_timeseries.pdf/png")
