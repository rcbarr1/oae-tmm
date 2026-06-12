"""
Visually and numerically check all regridded input data for oae-tmm.

For each data source (GLODAP, NCEP/NOAA, TRACE), shows side-by-side surface
maps of the original (native grid) and regridded (OCIM2-48L) fields so that
spatial artifacts from the regridding are immediately visible.

Use this as a before/after baseline when modifying regrid functions.
"""
#%%
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from oae_tmm import loaders, trace

data_path = './data/'
grid = loaders.load_ocim(data_path)

def _print_stats(label, arr, lats=None):
    valid_mask = ~np.isnan(arr)
    valid = arr[valid_mask]
    n_nan = int(np.sum(~valid_mask))
    if lats is not None:
        # cell area ∝ cos(lat) on a regular lon/lat grid; broadcast to (n_lat, n_lon)
        # and zero out NaN cells so they don't contribute to the denominator
        w = np.cos(np.deg2rad(lats)).reshape((-1,) + (1,) * (arr.ndim - 1)) * np.ones(arr.shape)
        mean = np.nansum(arr * w) / np.sum(w[valid_mask])
    else:
        mean = valid.mean()
    print(f'    {label:<28s}  '
          f'min={valid.min():8.2f}  max={valid.max():8.2f}  '
          f'mean={mean:8.2f}  NaN={n_nan}')


def _surface_panel(ax, lons, lats, data, title, cmap='RdYlBu_r', vmin=None, vmax=None,
                   xlim=(0, 360)):
    masked = np.ma.masked_invalid(data)
    if vmin is None: vmin = float(np.nanpercentile(data, 2))
    if vmax is None: vmax = float(np.nanpercentile(data, 98))
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(lons, lats, masked, levels=levels, cmap=cmap, extend='both')
    plt.colorbar(cf, ax=ax, shrink=0.7)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(*xlim); ax.set_ylim(-90, 90)
    ax.set_xlabel('lon (°E)', fontsize=7); ax.set_ylabel('lat (°N)', fontsize=7)
    ax.tick_params(labelsize=7)


#%% ── GLODAP ───────────────────────────────────────────────────────────────────

_GLODAP_VARS = [
    # (raw_var,    raw_suffix,    reg_var,       label,              vmin,  vmax,  cmap)
    ('TCO2',       'TCO2',        'CT',          'CT (µmol kg⁻¹)',   1900,  2300,  'viridis'),
    ('TAlk',       'TAlk',        'AT',          'AT (µmol kg⁻¹)',   2200,  2500,  'viridis'),
    ('temperature','temperature', 'temperature', 'Temperature (°C)', -2,    30,    'RdYlBu_r'),
    ('salinity',   'salinity',    'salinity',    'Salinity',         33,    37,    'viridis'),
    ('silicate',   'silicate',    'silicate',    'Silicate (µmol kg⁻¹)', 0, 120,  'plasma'),
    ('PO4',        'PO4',         'phosphate',   'Phosphate (µmol kg⁻¹)', 0, 3,  'plasma'),
]

print('\n=== GLODAP ===')
latitude  = grid['latitude']
longitude = grid['longitude']
base = data_path + 'GLODAPv2.2016b.MappedProduct/'

for raw_var, raw_suffix, reg_var, label, vmin, vmax, cmap in _GLODAP_VARS:
    print(f'\n  {label}')
    with xr.open_dataset(base + f'GLODAPv2.2016b.{raw_suffix}.nc') as raw_ds:
        raw_lat = raw_ds['lat'].to_numpy()
        raw_lon = raw_ds['lon'].to_numpy()   # native 20–380°E convention
        raw_surf = raw_ds[raw_var].transpose('lat', 'lon', 'depth_surface').values[:, :, 0]
    with xr.open_dataset(base + f'{reg_var}.nc') as reg_ds:
        reg_surf = reg_ds[reg_var].transpose('latitude', 'longitude', 'depth').values[:, :, 0]

    _print_stats('raw (native grid, surface)',  raw_surf, lats=raw_lat)
    _print_stats('regridded (OCIM2-48L, surface)', reg_surf, lats=latitude)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'GLODAP {label} — original vs regridded (surface layer)', fontsize=10)
    _surface_panel(axes[0], raw_lon, raw_lat, raw_surf,
                    f'Original ({raw_var}, native 1°×1° grid, 20–380°E)',
                    vmin=vmin, vmax=vmax, cmap=cmap, xlim=(20, 380))
    _surface_panel(axes[1], longitude, latitude, reg_surf,
                    f'Regridded ({reg_var}, OCIM2-48L)',
                    vmin=vmin, vmax=vmax, cmap=cmap)
    plt.tight_layout()

plt.show()


#%% ── NCEP/NOAA ─────────────────────────────────────────────────────────────────

_NCEP_FIELDS = [
    # (var_key, raw_path_rel, reg_path_rel, var_name, label, vmin, vmax, cmap, use_cftime, time_slice)
    ('icec', 'NCEP_DOE_Reanalysis_II/icec.sfc.mon.ltm.1991-2020.nc',
             'NCEP_DOE_Reanalysis_II/icec.nc',
             'icec', 'Sea ice fraction', 0, 1, 'Blues', True, None),
    ('wspd', 'NCEP_DOE_Reanalysis_II/wspd.10m.mon.mean.nc',
             'NCEP_DOE_Reanalysis_II/wspd.nc',
             'wspd', 'Wind speed at 10 m (m s⁻¹)', 0, 15, 'viridis', False, slice(552, 924)),
    ('sst',  'NOAA_Extended_Reconstruction_SST_V5/sst.mon.ltm.1991-2020.nc',
             'NOAA_Extended_Reconstruction_SST_V5/sst.nc',
             'sst', 'SST (°C)', -2, 30, 'RdYlBu_r', True, None),
]

print('\n=== NCEP/NOAA ===')
latitude  = grid['latitude']
longitude = grid['longitude']

for var_key, raw_rel, reg_rel, var_name, label, vmin, vmax, cmap, use_cftime, tslice in _NCEP_FIELDS:
    print(f'\n  {label}')
    with xr.open_dataset(data_path + raw_rel,
                         decode_times=xr.coders.CFDatetimeCoder(use_cftime=use_cftime)) as raw_ds:
        raw_lat = raw_ds['lat'].to_numpy()
        raw_lon = raw_ds['lon'].to_numpy()
        da = raw_ds[var_name]
        if tslice is not None:
            da = da.isel(time=tslice)
        raw_data = da.mean(dim='time', skipna=True).values
    with xr.open_dataset(data_path + reg_rel) as reg_ds:
        reg_data = reg_ds[var_name].to_numpy()

    _print_stats('raw (native grid)',  raw_data, lats=raw_lat)
    _print_stats('regridded (OCIM2-48L)', reg_data, lats=latitude)
    if var_key == 'wspd' and np.any(reg_data < 0):
        print('  WARNING: negative wind speeds in regridded data')
    if var_key == 'icec' and np.any(reg_data > 1):
        print('  WARNING: ice fraction > 1 in regridded data')

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'NCEP/NOAA {label} — original vs regridded', fontsize=10)
    _surface_panel(axes[0], raw_lon, raw_lat, raw_data,
                    'Original (native grid)', vmin=vmin, vmax=vmax, cmap=cmap)
    _surface_panel(axes[1], longitude, latitude, reg_data,
                    'Regridded (OCIM2-48L)', vmin=vmin, vmax=vmax, cmap=cmap)
    plt.tight_layout()

plt.show()

#%% ── TRACE ─────────────────────────────────────────────────────────────────────

# choose scenario and year to test
year = 2052
scenario = 'ssp434'

print(f'\n=== TRACE (year={year}, scenario={scenario}) ===')
ocnmask   = grid['ocnmask']
latitude  = grid['latitude']
longitude = grid['longitude']
depth     = grid['depth']

scenarios = { # have not yet built gridded product for REMIND scenario
    'none': 1, 'ssp119': 2, 'ssp126': 3, 'ssp245': 4, 'ssp370': 5,
    'ssp370_lowNTCF': 6, 'ssp434': 7, 'ssp460': 8, 'ssp534_OS': 9,
}
scen_idx = scenarios.get(scenario, 1)
if 2020 <= year <= 2100:
    raw_path = data_path + f'TRACE_gridded/OCIM_CanthFromTRACECO2Pathway{scen_idx}.nc'
elif year > 2100:
    raw_path = data_path + f'TRACE_gridded/CanthFromTRACECO2Pathway{scen_idx}.nc'
else:
    raw_path = data_path + f'TRACE_gridded/CanthFromTRACECO2Pathway1.nc'

print(f'  Loading raw TRACE file: {os.path.basename(raw_path)}')
with xr.open_dataset(raw_path, decode_times=False) as raw_ds:
    raw_lat = raw_ds['lat'].to_numpy()
    raw_lon = raw_ds['lon'].to_numpy()
    raw_at_year = raw_ds.interp(time=year)['canth']
    # normalize to (lat, lon, depth)
    raw_canth = raw_at_year.transpose('lat', 'lon', 'depth').values

# Call interp_trace — pass copies so we can detect in-place mutation
lon_copy_before = longitude.copy()
canth_regrid = trace.interp_trace(
    data_path, year, scenario,
    latitude.copy(), longitude.copy(), depth.copy(), ocnmask,
)
if not np.array_equal(longitude, lon_copy_before):
    print('  WARNING: interp_trace mutated the longitude array')

print('\n  Numerical summary (surface):')
_print_stats('raw Canth',        raw_canth[:, :, 0],    lats=raw_lat)
_print_stats('interp_trace',     canth_regrid[:, :, 0], lats=latitude)

print('\n  Numerical summary (all depths):')
_print_stats('raw Canth',        raw_canth,    lats=raw_lat)
_print_stats('interp_trace',     canth_regrid, lats=latitude)

ocn_vals = canth_regrid[ocnmask == 1]
n_nan = int(np.sum(np.isnan(ocn_vals)))
print(f'    Ocean cells: {len(ocn_vals)}, NaN={n_nan}')
if n_nan == 0:
    print('    OK: no NaN values in ocean cells')
else:
    print(f'    WARNING: {n_nan} NaN values in ocean cells')

# Surface map: original vs regridded
vmin = float(np.nanpercentile(raw_canth[:, :, 0], 2))
vmax = float(np.nanpercentile(raw_canth[:, :, 0], 98))
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(f'TRACE Canth at {year}, {scenario} — original vs interp_trace (surface)', fontsize=10)
_surface_panel(axes[0], raw_lon, raw_lat, raw_canth[:, :, 0],
                'Original (native grid)', vmin=vmin, vmax=vmax, cmap='plasma', xlim=(0, 380))
_surface_panel(axes[1], longitude, latitude, canth_regrid[:, :, 0],
                'interp_trace output (OCIM2-48L)', vmin=vmin, vmax=vmax, cmap='plasma')
plt.tight_layout()

# Depth section
lon_idx = int(np.argmin(np.abs(longitude - 180)))   # ~180°E
sec_vmin = 0.0
sec_vmax = float(np.nanpercentile(canth_regrid, 95))
section = np.ma.masked_invalid(canth_regrid[:, lon_idx, :].T)
levels = np.linspace(sec_vmin, sec_vmax, 100)
fig4, ax4 = plt.subplots(figsize=(10, 7))
cf = ax4.contourf(latitude, depth, section, levels=levels, cmap='plasma', extend='both')
cb = fig4.colorbar(cf, ax=ax4)
ax4.invert_yaxis()
ax4.set_xlabel('latitude (°N)', fontsize=9)
ax4.set_ylabel('depth (m)', fontsize=9)
ax4.set_title(f'TRACE Canth at {year}, {scenario} — lat-depth section at ~180°E', fontsize=10)
ax4.set_xlim(-90, 90)
ax4.set_ylim(depth.max(), 0)

# Depth profiles
fig3, ax3 = plt.subplots(figsize=(6, 7))
for loc_name, loc_lat, loc_lon in [
    ('Tropical Pacific (0°N, 200°E)',  0,  200),
    ('N. Atlantic (60°N, 330°E)',      60,  330),
    ('Southern Ocean (-50°N, 180°E)', -50,  180),
]:
    lat_i = int(np.argmin(np.abs(latitude - loc_lat)))
    lon_i = int(np.argmin(np.abs(longitude - loc_lon)))
    profile = canth_regrid[lat_i, lon_i, :]
    if not np.all(np.isnan(profile)):
        ax3.plot(profile, depth, label=loc_name)
ax3.invert_yaxis()
ax3.set_xlabel('Canth (µmol kg⁻¹)', fontsize=9)
ax3.set_ylabel('depth (m)', fontsize=9)
ax3.set_title(f'TRACE Canth at {year} — depth profiles', fontsize=10)
ax3.legend(fontsize=8)
plt.tight_layout()

plt.show()

#%%