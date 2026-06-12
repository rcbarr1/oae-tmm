"""
Quick inspection of oae-tmm experiment output files.

Set filepath and data_path in the first cell, then run cells sequentially.
Prints metadata, variable summary at the final timestep, and an initial
condition check. Shows three figures: time series of volume-weighted ocean
totals, and surface + lat-depth transect maps at first/middle/final timestep
for both delAT and delCT.
"""

# %%
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from oae_tmm import loaders
from oae_tmm.grid import make_3d

test_files = ['exp22_2026-06-12_test_ssp126_000.nc',
              'exp22_TEST_000.nc', # before refactor
              'exp23_2026-06-12_test_ssp126_000.nc',
              'exp23_TEST_000.nc', # before refactor
              'exp24_2026-06-12_t-mixed_00000_000.nc',
              'exp24_TEST_000.nc', # before refactor
              'exp25_2026-06-12_t-mixed_00000_000.nc',
              'exp25_TEST_000.nc', # before refactor
              'exp26_2026-06-12_t1_00000_000.nc',
              'exp26_TEST_000.nc', # before refactor
              'LCA1_2026-06-12_nearshore_colombia_1ton_2050_000.nc',
              'LCA1_TEST_000.nc'] # before refactor

filepath  = './outputs/' + test_files[11]
data_path = './data/'

# ── helpers ───────────────────────────────────────────────────────────────────

def _print_stats(label, arr):
    valid = arr[~np.isnan(arr)]
    n_nan = int(np.sum(np.isnan(arr)))
    if len(valid) == 0:
        print(f'    {label:<28s}  all NaN')
        return
    print(f'    {label:<28s}  min={valid.min():12.4e}  max={valid.max():12.4e}  '
          f'mean={valid.mean():12.4e}  NaN={n_nan}')


def _add_colorbar(cf, ax):
    _fmt = mticker.ScalarFormatter(useMathText=True)
    _fmt.set_scientific(True); _fmt.set_powerlimits((0, 0))
    cb = plt.colorbar(cf, ax=ax, shrink=0.6, format=_fmt)
    cb.ax.yaxis.get_offset_text().set_x(4)  # shift exponent right of colorbar


def _map_panel(ax, lons, lats, data_2d, title, vmin, vmax, cmap):
    masked = np.ma.masked_invalid(data_2d)
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(lons, lats, masked, levels=levels, cmap=cmap, extend='both')
    _add_colorbar(cf, ax)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.tick_params(labelsize=7)


def _section_panel(ax, lats, depths, data_latdepth, title, vmin, vmax, cmap):
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


#%% metadata
ds = xr.open_dataset(filepath)

# handle both old (lat/lon, delDIC) and current (latitude/longitude, delCT) naming
lat_key = 'latitude' if 'latitude' in ds.coords else 'lat'
lon_key = 'longitude' if 'longitude' in ds.coords else 'lon'
ct_key  = 'delCT'    if 'delCT'    in ds else 'delDIC'
ct_add  = 'CT_added' if 'CT_added' in ds else 'DIC_added'

lats   = ds[lat_key].values
lons   = ds[lon_key].values
depths = ds['depth'].values
times  = ds['time'].values

print(f'=== {os.path.basename(filepath)} ===')
for k in ('experiment', 'scenario', 'tag'):
    if k in ds.attrs:
        print(f'  {k}: {ds.attrs[k]}')
print(f'  time:  {times[0]:.4f} → {times[-1]:.4f} yr  ({len(times)} steps)')
print(f'  grid:  {len(lats)} lat × {len(lons)} lon × {len(depths)} depth')
print(f'  size:  {os.path.getsize(filepath) / 1024**2:.1f} MB')


#%% variable summary + IC check
final = len(times) - 1
print('Variable summary (final timestep):')
_print_stats('delAT (µmol kg⁻¹)',     ds['delAT'].isel(time=final).values)
_print_stats(f'{ct_key} (µmol kg⁻¹)', ds[ct_key].isel(time=final).values)
_print_stats('AT_added (µmol kg⁻¹)',  ds['AT_added'].isel(time=final).values)
_print_stats(f'{ct_add} (µmol kg⁻¹)', ds[ct_add].isel(time=final).values)
print(f'    {"delxCO2 (ppm)":<28s}  {float(ds["delxCO2"].isel(time=final)):.6f}')

print('\nInitial condition check (t=0):')
at_ok  = np.all(np.nan_to_num(ds['delAT'].isel(time=0).values) == 0)
ct_ok  = np.all(np.nan_to_num(ds[ct_key].isel(time=0).values) == 0)
xco_ok = float(ds['delxCO2'].isel(time=0)) == 0.0
status = 'PASS' if (at_ok and ct_ok and xco_ok) else 'FAIL'
print(f'  {status}  (delAT=0: {at_ok}, {ct_key}=0: {ct_ok}, delxCO2=0: {xco_ok})')


#%% load grid
grid        = loaders.load_ocim(data_path)
ocnmask     = grid['ocnmask']
rho         = grid['rho']
cell_vol_3d = make_3d(grid['cell_volume'], ocnmask)   # (n_lat, n_lon, n_depth)
longitude   = grid['longitude']
latitude    = grid['latitude']
depth       = grid['depth']
lon_idx     = int(np.argmin(np.abs(longitude - 180)))  # ~180°E


#%% Figure 1: time series
# µmol/kg * kg/m³ * m³ = µmol; / 1e21 = Pmol
delAT_pmol   = np.nansum(ds['delAT'].values    * cell_vol_3d[np.newaxis] * rho,
                        axis=(1, 2, 3)) / 1e21
delCT_pmol   = np.nansum(ds[ct_key].values    * cell_vol_3d[np.newaxis] * rho,
                        axis=(1, 2, 3)) / 1e21
AT_added_pmol = np.nansum(ds['AT_added'].values * cell_vol_3d[np.newaxis] * rho,
                          axis=(1, 2, 3)) / 1e21
CT_added_pmol = np.nansum(ds[ct_add].values    * cell_vol_3d[np.newaxis] * rho,
                          axis=(1, 2, 3)) / 1e21

fig1, axes1 = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
fig1.suptitle(os.path.basename(filepath), fontsize=10)
axes1[0].plot(times, delAT_pmol);             axes1[0].set_ylabel('∆AT [Pmol]')
axes1[1].plot(times, delCT_pmol);             axes1[1].set_ylabel(f'∆{ct_key[3:]} [Pmol]')
axes1[2].plot(times, ds['delxCO2'].values);   axes1[2].set_ylabel('∆xCO₂ [ppm]')
axes1[3].plot(times, AT_added_pmol);          axes1[3].set_ylabel('AT_added [Pmol]')
axes1[4].plot(times, CT_added_pmol);          axes1[4].set_ylabel(f'{ct_add} [Pmol]')
axes1[4].set_xlabel('year')
for ax in axes1:
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.tick_params(labelsize=8)
plt.tight_layout()
plt.show()


#%% Figure 2: delAT maps (surface + transect)
t_idxs = [0, len(times) // 2, len(times) - 1]
t_lbls = ['first', 'mid', 'final']

arr = ds['delAT'].values
absmax = float(np.percentile(np.abs(arr[~np.isnan(arr)]), 98))
vmin, vmax = -absmax, absmax

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 6))
fig2.suptitle(f'∆AT (µmol kg⁻¹) — {os.path.basename(filepath)}', fontsize=10)
for col, (ti, tlbl) in enumerate(zip(t_idxs, t_lbls)):
    yr = float(times[ti])
    _map_panel(axes2[0, col], lons, lats, arr[ti, :, :, 0],
               f'{tlbl} (t={yr:.1f}) — surface', vmin, vmax, 'RdBu_r')
    _section_panel(axes2[1, col], latitude, depth, arr[ti, :, lon_idx, :],
                   f'{tlbl} (t={yr:.1f}) — ~180°E', vmin, vmax, 'RdBu_r')
axes2[1, 0].set_xlabel('lat (°N)', fontsize=8)
axes2[1, 0].set_ylabel('depth (m)', fontsize=8)
plt.tight_layout()
plt.show()


#%% Figure 3: delCT maps (surface + transect)
arr = ds[ct_key].values
absmax = float(np.percentile(np.abs(arr[~np.isnan(arr)]), 98))
vmin, vmax = -absmax, absmax

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 6))
fig3.suptitle(f'∆{ct_key[3:]} (µmol kg⁻¹) — {os.path.basename(filepath)}', fontsize=10)
for col, (ti, tlbl) in enumerate(zip(t_idxs, t_lbls)):
    yr = float(times[ti])
    _map_panel(axes3[0, col], lons, lats, arr[ti, :, :, 0],
               f'{tlbl} (t={yr:.1f}) — surface', vmin, vmax, 'RdBu_r')
    _section_panel(axes3[1, col], latitude, depth, arr[ti, :, lon_idx, :],
                   f'{tlbl} (t={yr:.1f}) — ~180°E', vmin, vmax, 'RdBu_r')
axes3[1, 0].set_xlabel('lat (°N)', fontsize=8)
axes3[1, 0].set_ylabel('depth (m)', fontsize=8)
plt.tight_layout()
plt.show()


#%% Figure 4: AT_added maps (surface + transect)
arr = ds['AT_added'].values
absmax = float(np.nanmax(np.abs(arr)))
vmin, vmax = -absmax, absmax
t_idxs[0] = 1

fig4, axes4 = plt.subplots(2, 3, figsize=(15, 6))
fig4.suptitle(f'AT_added (µmol kg⁻¹) — {os.path.basename(filepath)}', fontsize=10)
for col, (ti, tlbl) in enumerate(zip(t_idxs, t_lbls)):
    yr = float(times[ti])
    _map_panel(axes4[0, col], lons, lats, arr[ti, :, :, 0],
               f'{tlbl} (t={yr:.1f}) — surface', vmin, vmax, 'RdBu_r')
    _section_panel(axes4[1, col], latitude, depth, arr[ti, :, lon_idx, :],
                   f'{tlbl} (t={yr:.1f}) — ~180°E', vmin, vmax, 'RdBu_r')
axes4[1, 0].set_xlabel('lat (°N)', fontsize=8)
axes4[1, 0].set_ylabel('depth (m)', fontsize=8)
plt.tight_layout()
plt.show()


#%% Figure 5: CT_added maps (surface + transect)
arr = ds[ct_add].values
absmax = float(np.nanmax(np.abs(arr)))
vmin, vmax = -absmax, absmax
t_idxs[0] = 1

fig5, axes5 = plt.subplots(2, 3, figsize=(15, 6))
fig5.suptitle(f'{ct_add} (µmol kg⁻¹) — {os.path.basename(filepath)}', fontsize=10)
for col, (ti, tlbl) in enumerate(zip(t_idxs, t_lbls)):
    yr = float(times[ti])
    _map_panel(axes5[0, col], lons, lats, arr[ti, :, :, 0],
               f'{tlbl} (t={yr:.1f}) — surface', vmin, vmax, 'RdBu_r')
    _section_panel(axes5[1, col], latitude, depth, arr[ti, :, lon_idx, :],
                   f'{tlbl} (t={yr:.1f}) — ~180°E', vmin, vmax, 'RdBu_r')
axes5[1, 0].set_xlabel('lat (°N)', fontsize=8)
axes5[1, 0].set_ylabel('depth (m)', fontsize=8)
plt.tight_layout()
plt.show()

# %%
