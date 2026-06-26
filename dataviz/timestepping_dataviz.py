"""
Created on Tue Jun 23 2026

DATAVIZ FOR TIMESTEPPING COMPARISON
Covers both max_AT and impulse_response timestepping experiments.

@author: Reese C. Barrett
"""
#%%
from pathlib import Path
import glob
import os

from dataviz.dataviz import broadcast_to_dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

data_path   = './data/'
max_AT_path = '/Volumes/LaCie/outputs/max_AT/'
ir_path     = './outputs/'   # impulse_response --test output; change if run elsewhere

# Set to the tag dates used when the experiments were run
max_AT_date = '2026-06-18'
ir_date     = '2026-06-23'

model_data  = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask     = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()
latitude    = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()
longitude   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()
cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy()
model_data.close()
rho = 1025  # seawater density [kg m-3]


# ============================================================================ #
#%%  MAX AT TIMESTEPPING                                                        #
# ============================================================================ #

max_AT_names = [
    f'max_AT_{max_AT_date}_annually_none',
    f'max_AT_{max_AT_date}_monthly_none',
    f'max_AT_{max_AT_date}_dekadal_none',
    f'max_AT_{max_AT_date}_pentadal_none',
    f'max_AT_{max_AT_date}_daily_none',
    f'max_AT_{max_AT_date}_hourly_none',
    f'max_AT_{max_AT_date}_mixed_none',
]
max_AT_labels = ['Annual', 'Monthly', 'Dekadal', 'Pentadal', 'Daily', 'Hourly', 'Mixed']

#%% compute or load cached max_AT time series
max_AT_cache_path = max_AT_path + 'max_AT_timestepping_cache.nc'


def _load_max_AT(name, label):
    time_dim = f'time_{label}'
    exp_vars = {}
    with xr.open_mfdataset(
            max_AT_path + name + '_*.nc',
            combine='by_coords', chunks={'time': 10}, parallel=True) as ds:

        cv = broadcast_to_dataset(cell_volume, ds)

        AT_added = ds['AT_added'] * cv * rho * 1e-6
        exp_vars[f'AT_added_cum_{label}'] = (
            AT_added.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                    .cumsum(dim='time')
                    .rename({'time': time_dim})
        )
        exp_vars[f'delxCO2_{label}'] = ds['delxCO2'].rename({'time': time_dim})

        delCT = ds['delCT'] * cv * rho * 1e-6
        exp_vars[f'delCT_{label}'] = (
            delCT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                 .rename({'time': time_dim})
        )
        delAT = ds['delAT'] * cv * rho * 1e-6
        exp_vars[f'delAT_{label}'] = (
            delAT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                 .rename({'time': time_dim})
        )
    return exp_vars


if os.path.exists(max_AT_cache_path):
    max_AT_cache = xr.open_dataset(max_AT_cache_path).load()
    missing = [(n, l) for n, l in zip(max_AT_names, max_AT_labels)
               if f'AT_added_cum_{l}' not in max_AT_cache]
    if missing:
        new_vars = {}
        for name, label in tqdm(missing, desc='Computing missing max_AT experiments'):
            new_vars.update(_load_max_AT(name, label))
        with TqdmCallback(desc='Computing missing max_AT cache entries'):
            max_AT_cache = xr.merge([max_AT_cache, xr.Dataset(new_vars).compute()])
        max_AT_cache.to_netcdf(max_AT_cache_path)
else:
    cache_vars = {}
    for name, label in tqdm(zip(max_AT_names, max_AT_labels), total=len(max_AT_labels),
                            desc='Loading max_AT experiments'):
        cache_vars.update(_load_max_AT(name, label))
    with TqdmCallback(desc='Computing max_AT cache'):
        max_AT_cache = xr.Dataset(cache_vars).compute()
    max_AT_cache.to_netcdf(max_AT_cache_path)

#%% plot max_AT: cumulative AT added
fig = plt.figure(figsize=(5, 5), dpi=200)
ax  = fig.gca()
for label in max_AT_labels:
    var = max_AT_cache[f'AT_added_cum_{label}']
    ax.plot(var[f'time_{label}'].values, var.values, label=label)
plt.xlabel('Year')
plt.ylabel(r'Cumulative $A_{\mathbf{T}}$ added to mixed layer (mol)')
plt.title('max_AT: timestepping comparison')
plt.legend()

for label in max_AT_labels:
    if label != 'Hourly':
        var  = max_AT_cache[f'AT_added_cum_{label}']
        pct  = (var.sel({f'time_{label}': 2035}, method='nearest') /
                max_AT_cache['AT_added_cum_Hourly'].sel(time_Hourly=2035, method='nearest') * 100)
        print(f'{label} : {pct:.2f}% of hourly AT added')

#%% plot max_AT: delxCO2
fig = plt.figure(figsize=(5, 5), dpi=200)
ax  = fig.gca()
for label in max_AT_labels:
    var = max_AT_cache[f'delxCO2_{label}']
    ax.plot(var[f'time_{label}'].values, var.values, label=label)
plt.xlabel('Year')
plt.ylabel('Change in atmospheric CO$_{2}$ (ppm)')
plt.title('max_AT: timestepping comparison')
plt.legend()

for label in max_AT_labels:
    if label != 'Hourly':
        var  = max_AT_cache[f'delxCO2_{label}']
        pct  = (var.sel({f'time_{label}': 2035}, method='nearest') /
                max_AT_cache['delxCO2_Hourly'].sel(time_Hourly=2035, method='nearest') * 100)
        print(f'{label} : {pct:.2f}% of hourly CO2 drawdown')

#%% plot max_AT: delCT
fig = plt.figure(figsize=(5, 5), dpi=200)
ax  = fig.gca()
for label in max_AT_labels:
    var = max_AT_cache[f'delCT_{label}']
    ax.plot(var[f'time_{label}'].values, var.values, label=label)
plt.xlabel('Year')
plt.ylabel(r'Change in $C_{\mathbf{T}}$ (mol)')
plt.title('max_AT: timestepping comparison')
plt.legend()

for label in max_AT_labels:
    if label != 'Hourly':
        var  = max_AT_cache[f'delCT_{label}']
        pct  = (var.sel({f'time_{label}': 2035}, method='nearest') /
                max_AT_cache['delCT_Hourly'].sel(time_Hourly=2035, method='nearest') * 100)
        print(f'{label} : {pct:.2f}% of hourly change in CT')

#%% plot max_AT: delAT
fig = plt.figure(figsize=(5, 5), dpi=200)
ax  = fig.gca()
for label in max_AT_labels:
    var = max_AT_cache[f'delAT_{label}']
    ax.plot(var[f'time_{label}'].values, var.values, label=label)
plt.xlabel('Year')
plt.ylabel(r'Change in $A_{\mathbf{T}}$ (mol)')
plt.title('max_AT: timestepping comparison')
plt.legend()

for label in max_AT_labels:
    if label != 'Hourly':
        var  = max_AT_cache[f'delAT_{label}']
        pct  = (var.sel({f'time_{label}': 2035}, method='nearest') /
                max_AT_cache['delAT_Hourly'].sel(time_Hourly=2035, method='nearest') * 100)
        print(f'{label} : {pct:.2f}% of hourly change in AT')


# ============================================================================ #
#%%  IMPULSE RESPONSE TIMESTEPPING                                              #
# ============================================================================ #

#%% plot impulse_response: --test cell locations on OCIM grid
surf_mask_2d  = ocnmask[:, :, 0]
ocn_idxs_surf = np.argwhere(surf_mask_2d == 1)
n             = len(ocn_idxs_surf)
test_indices  = [758, 5291, 7965, 8810]

fig = plt.figure(figsize=(10, 4), dpi=200)
ax  = fig.gca()
ax.pcolormesh(longitude, latitude, surf_mask_2d, cmap='Blues', alpha=0.4)
for cell_num, color in zip(test_indices, ['C0', 'C1', 'C2', 'C3']):
    lat_idx, lon_idx = ocn_idxs_surf[cell_num, 0], ocn_idxs_surf[cell_num, 1]
    ax.scatter(longitude[lon_idx], latitude[lat_idx], color=color, s=60, zorder=5,
               label=f'cell {cell_num}')
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('Impulse response: --test cell locations')
ax.legend()

ir_t_names = ['annually', 'monthly', 'dekadal', 'pentadal', 'daily']
ir_labels  = ['Annual',   'Monthly', 'Dekadal', 'Pentadal', 'Daily']

# Discover which cell numbers were run by globbing for the daily files
ir_daily_files = sorted(glob.glob(ir_path + f'impulse_response_{ir_date}_daily_none_*_000.nc'))
ir_cell_nums   = [int(Path(f).stem.split('_')[-2]) for f in ir_daily_files]

#%% compute or load cached impulse_response time series
ir_cache_path = ir_path + f'ir_timestepping_cache_{ir_date}.nc'


def _load_ir(t_name, label, cell_num):
    time_dim = f'time_{label}_{cell_num}'
    files    = sorted(glob.glob(
        ir_path + f'impulse_response_{ir_date}_{t_name}_none_{cell_num:05d}_*.nc'))
    if not files:
        return {}
    exp_vars = {}
    with xr.open_mfdataset(files, combine='by_coords', chunks={'time': 10}, parallel=True) as ds:
        cv = broadcast_to_dataset(cell_volume, ds)

        AT_added = ds['AT_added'] * cv * rho * 1e-6
        exp_vars[f'AT_added_cum_{label}_{cell_num}'] = (
            AT_added.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                    .cumsum(dim='time')
                    .rename({'time': time_dim})
        )
        exp_vars[f'delxCO2_{label}_{cell_num}']  = ds['delxCO2'].rename({'time': time_dim})

        delCT = ds['delCT'] * cv * rho * 1e-6
        exp_vars[f'delCT_{label}_{cell_num}'] = (
            delCT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                 .rename({'time': time_dim})
        )
        delAT = ds['delAT'] * cv * rho * 1e-6
        exp_vars[f'delAT_{label}_{cell_num}'] = (
            delAT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                 .rename({'time': time_dim})
        )
    return exp_vars


combos = [(t, l, c) for t, l in zip(ir_t_names, ir_labels) for c in ir_cell_nums]

if os.path.exists(ir_cache_path):
    ir_cache = xr.open_dataset(ir_cache_path).load()
    missing  = [(t, l, c) for t, l, c in combos
                if f'AT_added_cum_{l}_{c}' not in ir_cache]
    if missing:
        new_vars = {}
        for t_name, label, cell_num in tqdm(missing, desc='Computing missing IR experiments'):
            new_vars.update(_load_ir(t_name, label, cell_num))
        with TqdmCallback(desc='Computing missing IR cache entries'):
            ir_cache = xr.merge([ir_cache, xr.Dataset(new_vars).compute()])
        ir_cache.to_netcdf(ir_cache_path)
else:
    cache_vars = {}
    for t_name, label, cell_num in tqdm(combos, desc='Loading IR experiments'):
        cache_vars.update(_load_ir(t_name, label, cell_num))
    with TqdmCallback(desc='Computing IR cache'):
        ir_cache = xr.Dataset(cache_vars).compute()
    ir_cache.to_netcdf(ir_cache_path)

#%% plot impulse_response: cumulative AT added (four-panel)
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
for ax, cell_num in zip(axes.flat, ir_cell_nums):
    for label in ir_labels:
        var      = ir_cache[f'AT_added_cum_{label}_{cell_num}']
        time_dim = f'time_{label}_{cell_num}'
        ax.plot(var[time_dim].values, var.values, label=label)
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Cumulative $A_{\mathbf{T}}$ added (mol)')
    ax.set_title(f'Cell {cell_num}')
    ax.legend()
fig.suptitle('Impulse response: AT added — timestepping comparison')
plt.tight_layout()

_col_w  = 12
_ref    = 'Daily'
_header = f"{'':12}" + ''.join(f"{'Cell '+str(c):>{_col_w}}" for c in ir_cell_nums)
print('\nImpulse response: AT added (% of daily at 2027)')
print(_header)
for label in ir_labels:
    if label != _ref:
        _row = f"{label:<12}"
        for cell_num in ir_cell_nums:
            ref_val = ir_cache[f'AT_added_cum_{_ref}_{cell_num}'].sel(
                {f'time_{_ref}_{cell_num}': 2027}, method='nearest').values
            val = ir_cache[f'AT_added_cum_{label}_{cell_num}'].sel(
                {f'time_{label}_{cell_num}': 2027}, method='nearest').values
            _row += f"{val / ref_val * 100:>{_col_w}.2f}%"
        print(_row)

#%% plot impulse_response: delxCO2 (four-panel)
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
for ax, cell_num in zip(axes.flat, ir_cell_nums):
    for label in ir_labels:
        var      = ir_cache[f'delxCO2_{label}_{cell_num}']
        time_dim = f'time_{label}_{cell_num}'
        ax.plot(var[time_dim].values, var.values, label=label)
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Change in atmospheric CO$_{2}$ (ppm)')
    ax.set_title(f'Cell {cell_num}')
    ax.legend()
fig.suptitle('Impulse response: CO$_2$ drawdown — timestepping comparison')
plt.tight_layout()

print('\nImpulse response: CO2 drawdown (% of daily at 2027)')
print(_header)
for label in ir_labels:
    if label != _ref:
        _row = f"{label:<12}"
        for cell_num in ir_cell_nums:
            ref_val = ir_cache[f'delxCO2_{_ref}_{cell_num}'].sel(
                {f'time_{_ref}_{cell_num}': 2027}, method='nearest').values
            val = ir_cache[f'delxCO2_{label}_{cell_num}'].sel(
                {f'time_{label}_{cell_num}': 2027}, method='nearest').values
            _row += f"{val / ref_val * 100:>{_col_w}.2f}%"
        print(_row)

#%% plot impulse_response: delCT (four-panel)
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
for ax, cell_num in zip(axes.flat, ir_cell_nums):
    for label in ir_labels:
        var      = ir_cache[f'delCT_{label}_{cell_num}']
        time_dim = f'time_{label}_{cell_num}'
        ax.plot(var[time_dim].values, var.values, label=label)
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Change in $C_{\mathbf{T}}$ (mol)')
    ax.set_title(f'Cell {cell_num}')
    ax.legend()
fig.suptitle(r'Impulse response: $C_T$ change — timestepping comparison')
plt.tight_layout()

print('\nImpulse response: CT change (% of daily at 2027)')
print(_header)
for label in ir_labels:
    if label != _ref:
        _row = f"{label:<12}"
        for cell_num in ir_cell_nums:
            ref_val = ir_cache[f'delCT_{_ref}_{cell_num}'].sel(
                {f'time_{_ref}_{cell_num}': 2027}, method='nearest').values
            val = ir_cache[f'delCT_{label}_{cell_num}'].sel(
                {f'time_{label}_{cell_num}': 2027}, method='nearest').values
            _row += f"{val / ref_val * 100:>{_col_w}.2f}%"
        print(_row)

#%% plot impulse_response: delAT (four-panel)
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
for ax, cell_num in zip(axes.flat, ir_cell_nums):
    for label in ir_labels:
        var      = ir_cache[f'delAT_{label}_{cell_num}']
        time_dim = f'time_{label}_{cell_num}'
        ax.plot(var[time_dim].values, var.values, label=label)
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Change in $A_{\mathbf{T}}$ (mol)')
    ax.set_title(f'Cell {cell_num}')
    ax.legend()
fig.suptitle(r'Impulse response: $A_T$ change — timestepping comparison')
plt.tight_layout()

print('\nImpulse response: AT change (% of daily at 2027)')
print(_header)
for label in ir_labels:
    if label != _ref:
        _row = f"{label:<12}"
        for cell_num in ir_cell_nums:
            ref_val = ir_cache[f'delAT_{_ref}_{cell_num}'].sel(
                {f'time_{_ref}_{cell_num}': 2027}, method='nearest').values
            val = ir_cache[f'delAT_{label}_{cell_num}'].sel(
                {f'time_{label}_{cell_num}': 2027}, method='nearest').values
            _row += f"{val / ref_val * 100:>{_col_w}.2f}%"
        print(_row)

# %%
