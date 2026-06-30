"""
anth_in_q_dataviz: numerical consistency check for the anth_in_q formulation.

Loads runs B, C, D from exp_anth_test and compares CDR signals:
    old CDR signal = delCT_B           (old baseline is all zeros)
    new CDR signal = delCT_D - delCT_C  (subtract new baseline carrying Canth)

If the two formulations are numerically equivalent these signals should be
identical. Any discrepancy indicates that passing dCanth/dt through the
implicit Euler solver (subject to transport T and air-sea exchange A) changes
the result relative to the current approach of silently shifting the background.

Set output_path and date_tag below, then run cells sequentially.
"""

# %%
import glob
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from oae_tmm import loaders
from oae_tmm.grid import make_3d
from experiments.base import rho

output_path = './outputs/'
data_path   = './data/'
date_tag    = '2026-06-29'        # set to e.g. '2026-06-29' to pin a specific run date

scenario = 'ssp245'


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(label):
    """Load a single-chunk dataset for an anth_test run by label."""
    pattern = os.path.join(output_path, f'anth_test_{date_tag}_{label}_{scenario}*.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files matched: {pattern}')
    if len(files) > 1:
        return xr.open_mfdataset(files, combine='by_coords')
    return xr.open_dataset(files[0])


def _vol_weighted_total(da, cell_vol_3d):
    """Global volume-weighted sum [Pmol] from a 4D DataArray (time,lat,lon,depth)."""
    return np.nansum(da.values * cell_vol_3d[np.newaxis] * rho, axis=(1, 2, 3)) / 1e21


def _add_colorbar(cf, ax):
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    cb = plt.colorbar(cf, ax=ax, shrink=0.6, format=fmt)
    cb.ax.yaxis.get_offset_text().set_x(4)


def _map_panel(ax, lons, lats, data_2d, title, vmin, vmax, cmap='RdBu_r'):
    masked = np.ma.masked_invalid(data_2d)
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(lons, lats, masked, levels=levels, cmap=cmap, extend='both')
    _add_colorbar(cf, ax)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.tick_params(labelsize=7)


def _section_panel(ax, lats, depths, data_latdepth, title, vmin, vmax, cmap='RdBu_r'):
    masked = np.ma.masked_invalid(data_latdepth)
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(lats, depths, masked.T, levels=levels, cmap=cmap, extend='both')
    _add_colorbar(cf, ax)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-90, 90)
    ax.tick_params(labelsize=7)


# ── load data ─────────────────────────────────────────────────────────────────

# %%
ds_B = _load('B_old_maxAT')
ds_C = _load('C_new_baseline')
ds_D = _load('D_new_maxAT')

grid        = loaders.load_ocim(data_path)
ocnmask     = grid['ocnmask']
cell_vol_3d = make_3d(grid['cell_volume'], ocnmask)
lats        = grid['latitude']
lons        = grid['longitude']
depths      = grid['depth']
lon_idx     = int(np.argmin(np.abs(lons - 180)))

time = ds_B['time'].values

print(f'Loaded {len(time)} timesteps  ({time[0]:.2f} → {time[-1]:.2f})')
print(f'  B: {os.path.basename(ds_B.encoding.get("source", "?"))}')
print(f'  C: {os.path.basename(ds_C.encoding.get("source", "?"))}')
print(f'  D: {os.path.basename(ds_D.encoding.get("source", "?"))}')


# ── CDR signals ───────────────────────────────────────────────────────────────

# %%
# old: state in B is purely CDR (old baseline ≈ 0)
old_delCT  = ds_B['delCT']
old_delAT  = ds_B['delAT']
old_delxCO2 = ds_B['delxCO2'].values   # ppm, scalar per timestep

# new: subtract baseline (C) which carries accumulated Canth increment
new_delCT  = ds_D['delCT']  - ds_C['delCT']
new_delAT  = ds_D['delAT']  - ds_C['delAT']
new_delxCO2 = ds_D['delxCO2'].values - ds_C['delxCO2'].values  # ppm

# difference between formulations (should be ≈ 0 if equivalent)
diff_delCT  = new_delCT  - old_delCT
diff_delAT  = new_delAT  - old_delAT
diff_delxCO2 = new_delxCO2 - old_delxCO2


# ── Figure 1: global time series ──────────────────────────────────────────────

# %%
old_CT_pmol  = _vol_weighted_total(old_delCT,  cell_vol_3d)
new_CT_pmol  = _vol_weighted_total(new_delCT,  cell_vol_3d)
diff_CT_pmol = _vol_weighted_total(diff_delCT, cell_vol_3d)

old_AT_pmol  = _vol_weighted_total(old_delAT,  cell_vol_3d)
new_AT_pmol  = _vol_weighted_total(new_delAT,  cell_vol_3d)
diff_AT_pmol = _vol_weighted_total(diff_delAT, cell_vol_3d)

fig1, axes = plt.subplots(3, 2, figsize=(9, 6), sharex=True)
fig1.suptitle('anth_in_q equivalence test — CDR signal comparison', fontsize=11)

axes[0, 0].plot(time, old_CT_pmol,  label='old (B)',      lw=1.5)
axes[0, 0].plot(time, new_CT_pmol,  label='new (D − C)',  lw=1.5, ls='--')
axes[0, 0].set_ylabel('∆CT [Pmol]')
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_title('CDR signal — ∆CT')

axes[0, 1].plot(time, diff_CT_pmol, color='C2', lw=1.5)
axes[0, 1].axhline(0, color='k', lw=0.5, ls='--')
axes[0, 1].set_ylabel('difference [Pmol]')
axes[0, 1].set_title('new − old  (∆CT)')

axes[1, 0].plot(time, old_AT_pmol,  label='old (B)',      lw=1.5)
axes[1, 0].plot(time, new_AT_pmol,  label='new (D − C)',  lw=1.5, ls='--')
axes[1, 0].set_ylabel('∆AT [Pmol]')
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title('CDR signal — ∆AT')

axes[1, 1].plot(time, diff_AT_pmol, color='C2', lw=1.5)
axes[1, 1].axhline(0, color='k', lw=0.5, ls='--')
axes[1, 1].set_ylabel('difference [Pmol]')
axes[1, 1].set_title('new − old  (∆AT)')

axes[2, 0].plot(time, old_delxCO2,  label='old (B)',      lw=1.5)
axes[2, 0].plot(time, new_delxCO2,  label='new (D − C)',  lw=1.5, ls='--')
axes[2, 0].set_ylabel('∆xCO₂ [ppm]')
axes[2, 0].set_xlabel('year')
axes[2, 0].legend(fontsize=8)
axes[2, 0].set_title('CDR signal — ∆xCO₂')

axes[2, 1].plot(time, diff_delxCO2, color='C2', lw=1.5)
axes[2, 1].axhline(0, color='k', lw=0.5, ls='--')
axes[2, 1].set_ylabel('difference [ppm]')
axes[2, 1].set_xlabel('year')
axes[2, 1].set_title('new − old  (∆xCO₂)')

for ax in axes.flat:
    ax.tick_params(labelsize=8)
plt.tight_layout()
plt.show()


# ── Figure 2: run C alone (Canth accumulation verification) ───────────────────

# %%
# Run C has no CDR, anth_in_q=True: state should accumulate Canth(t) - Canth_start.
# Check that delCT_C (global total) rises monotonically and has reasonable magnitude.
C_CT_pmol  = _vol_weighted_total(ds_C['delCT'], cell_vol_3d)
C_xCO2     = ds_C['delxCO2'].values

fig2, axes2 = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
fig2.suptitle('Run C (new baseline, no CDR) — Canth accumulation', fontsize=11)

axes2[0].plot(time, C_CT_pmol, lw=1.5)
axes2[0].set_ylabel('∆CT [Pmol]')
axes2[0].set_title('global ∆CT — should rise with Canth(t) − Canth_start')
axes2[0].axhline(0, color='k', lw=0.5, ls='--')

axes2[1].plot(time, C_xCO2, lw=1.5)
axes2[1].set_ylabel('∆xCO₂ [ppm]')
axes2[1].set_xlabel('year')
axes2[1].set_title('∆xCO₂ — should rise with atmospheric CO₂ scenario')
axes2[1].axhline(0, color='k', lw=0.5, ls='--')

for ax in axes2:
    ax.tick_params(labelsize=8)
plt.tight_layout()
plt.show()


# ── Figure 3: surface maps at final timestep ──────────────────────────────────

# %%
final = len(time) - 1
yr    = float(time[final])

old_surf  = old_delCT.isel(time=final).values[:, :, 0]
new_surf  = new_delCT.isel(time=final).values[:, :, 0]
diff_surf = diff_delCT.isel(time=final).values[:, :, 0]

absmax_cdr  = float(np.nanpercentile(np.abs(np.concatenate([old_surf.ravel(), new_surf.ravel()])), 98))
absmax_diff = float(np.nanpercentile(np.abs(diff_surf[~np.isnan(diff_surf)]), 98)) if np.any(~np.isnan(diff_surf)) else 1.0

fig3, axes3 = plt.subplots(1, 3, figsize=(10, 3))
fig3.suptitle(f'Surface ∆CT (µmol kg⁻¹) at t={yr:.1f}', fontsize=11)

_map_panel(axes3[0], lons, lats, old_surf,  'old CDR signal (B)',   -absmax_cdr,  absmax_cdr)
_map_panel(axes3[1], lons, lats, new_surf,  'new CDR signal (D−C)', -absmax_cdr,  absmax_cdr)
_map_panel(axes3[2], lons, lats, diff_surf, 'new − old',            -absmax_diff, absmax_diff)

plt.tight_layout()
plt.show()


# ── Figure 4: lat-depth sections of CDR signal and difference ─────────────────

# %%
old_sec  = old_delCT.isel(time=final).values[:, lon_idx, :]
new_sec  = new_delCT.isel(time=final).values[:, lon_idx, :]
diff_sec = diff_delCT.isel(time=final).values[:, lon_idx, :]

absmax_diff_sec = float(np.nanpercentile(np.abs(diff_sec[~np.isnan(diff_sec)]), 98)) if np.any(~np.isnan(diff_sec)) else 1.0

fig4, axes4 = plt.subplots(1, 3, figsize=(10, 3))
fig4.suptitle(f'∆CT lat–depth section at ~180°E, t={yr:.1f}', fontsize=11)

_section_panel(axes4[0], lats, depths, old_sec,  'old CDR signal (B)',   -absmax_cdr,      absmax_cdr)
_section_panel(axes4[1], lats, depths, new_sec,  'new CDR signal (D−C)', -absmax_cdr,      absmax_cdr)
_section_panel(axes4[2], lats, depths, diff_sec, 'new − old',            -absmax_diff_sec, absmax_diff_sec)

for ax in axes4:
    ax.set_xlabel('lat (°N)', fontsize=8)
axes4[0].set_ylabel('depth (m)', fontsize=8)

plt.tight_layout()
plt.show()


# ── numerical summary ─────────────────────────────────────────────────────────

# %%
print('=== Numerical consistency summary (final timestep) ===')
print(f'  old ∆CT global total:   {old_CT_pmol[-1]:.6f} Pmol')
print(f'  new ∆CT global total:   {new_CT_pmol[-1]:.6f} Pmol')
print(f'  difference:             {diff_CT_pmol[-1]:.3e} Pmol  '
      f'({abs(diff_CT_pmol[-1] / old_CT_pmol[-1]) * 100:.4f}% of old signal)')
print()
print(f'  old ∆xCO₂:             {old_delxCO2[-1]:.6f} ppm')
print(f'  new ∆xCO₂:             {new_delxCO2[-1]:.6f} ppm')
print(f'  difference:             {diff_delxCO2[-1]:.3e} ppm')

# %%
