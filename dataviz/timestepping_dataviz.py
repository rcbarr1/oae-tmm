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

from dataviz.dataviz import broadcast_to_dataset, load_ocim_grid, apply_style
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

data_path   = './data/'
max_AT_path = '/Volumes/LaCie/outputs/max_AT/'
ir_path     = './outputs/'   # impulse_response --test output; change if run elsewhere

# Set to the tag dates used when the experiments were run
max_AT_date = '2026-06-18'
ir_date     = '2026-06-23'

grid        = load_ocim_grid(data_path)
ocnmask     = grid['ocnmask']
latitude    = grid['latitude']
longitude   = grid['longitude']
cell_volume = grid['cell_volume']
rho = 1025  # seawater density [kg m-3]

textcolor, fontweight = apply_style()
_fs = 13
_display = {'Dekadal': '10-Day', 'Pentadal': '5-Day'}


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

        AT_added = ds['AT_added'] * cv * rho * 1e-6 # mol
        exp_vars[f'AT_added_cum_{label}'] = (
            AT_added.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                    .cumsum(dim='time')
                    .rename({'time': time_dim})
        )
        exp_vars[f'delxCO2_{label}'] = ds['delxCO2'].rename({'time': time_dim})

        delCT = ds['delCT'] * cv * rho * 1e-6 # mol
        exp_vars[f'delCT_{label}'] = (
            delCT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                 .rename({'time': time_dim})
        )
        delAT = ds['delAT'] * cv * rho * 1e-6 # mol
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

#%% plot max_AT: combined 2×2 publication figure
_panel_vars = [
    ('AT_added_cum', r'Cumulative ${A_{\mathrm{T}}}^{\prime}$ added (Pmol)', 1e-15),
    ('delxCO2',      r'CO$_2$ drawdown (ppm)',                               1.0),
    ('delCT',        r"${C_{\mathrm{T}}}^{\prime}$ (Pmol)",                 1e-15),
    ('delAT',        r"${A_{\mathrm{T}}}^{\prime}$ (Pmol)",                 1e-15),
]
# (lbl, x, ha): (b) floated right to avoid line overlap, rest top-left
_panel_labels = [('(a)', 0.02, 'left'), ('(b)', 0.98, 'right'),
                 ('(c)', 0.02, 'left'), ('(d)', 0.02, 'left')]

fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=200)
fig.subplots_adjust(hspace=0.32, wspace=0.32)

_legend_handles = []
_legend_text    = []

for ax, (var_key, ylabel, scale), (panel_lbl, px, pa) in zip(axes.flat, _panel_vars, _panel_labels):
    for label in max_AT_labels:
        var      = max_AT_cache[f'{var_key}_{label}']
        time_dim = f'time_{label}'
        line,    = ax.plot(var[time_dim].values, var.values * scale, label=label)
        if panel_lbl == '(a)':
            _legend_handles.append(line)
            _legend_text.append(_display.get(label, label))
    ax.set_xlabel('Year', fontsize=_fs)
    ax.set_ylabel(ylabel, fontsize=_fs)
    ax.text(px, 0.97, panel_lbl, transform=ax.transAxes,
            fontsize=_fs, va='top', ha=pa)
    for side in ('top', 'bottom', 'left', 'right'):
        ax.spines[side].set_color(textcolor)

fig.legend(_legend_handles, _legend_text,
           loc='lower center', ncol=4,
           fontsize=_fs, bbox_to_anchor=(0.5, -0.02),
           frameon=False)
plt.show()

_tbl_vars = [
    ('AT_added_cum', "AT' added"),
    ('delxCO2',      'CO2 drawdown'),
    ('delCT',        "CT'"),
    ('delAT',        "AT'"),
]
_lw  = 10
_vw  = 14
_yr  = 2035
_sep = '=' * (_lw + _vw * len(_tbl_vars))

print(f'\nmax_AT timestepping: % of hourly response at {_yr}')
print(_sep)
print(f"{'':>{_lw}}" + ''.join(f"{h:>{_vw}}" for _, h in _tbl_vars))
print('-' * (_lw + _vw * len(_tbl_vars)))
for label in max_AT_labels:
    if label != 'Hourly':
        _row = f"{_display.get(label, label):<{_lw}}"
        for var_key, _ in _tbl_vars:
            val     = max_AT_cache[f'{var_key}_{label}'].sel(
                        {f'time_{label}': _yr}, method='nearest')
            ref_val = max_AT_cache[f'{var_key}_Hourly'].sel(
                        time_Hourly=_yr, method='nearest')
            _row   += f"{float(val / ref_val * 100):>{_vw - 1}.2f}%"
        print(_row)
print(_sep)

max_AT_cache.close()

# ============================================================================ #
#%%  IMPULSE RESPONSE TIMESTEPPING                                              #
# ============================================================================ #

ir_t_names = ['annually', 'monthly', 'dekadal', 'pentadal', 'daily']
ir_labels  = ['Annual',   'Monthly', 'Dekadal', 'Pentadal', 'Daily']

# Discover which cell numbers were run by globbing for the daily files
ir_daily_files = sorted(glob.glob(ir_path + f'impulse_response_{ir_date}_daily_none_*_000.nc'))
ir_cell_nums   = [int(Path(f).stem.split('_')[-2]) for f in ir_daily_files]

#%% plot impulse_response: --test cell locations on OCIM grid
surf_mask_2d  = ocnmask[:, :, 0]
ocn_idxs_surf = np.argwhere(surf_mask_2d == 1)
n             = len(ocn_idxs_surf)
test_indices  = [758, 5291, 7965, 8810]

data_crs = ccrs.PlateCarree()
map_proj = ccrs.EqualEarth(central_longitude=200)

fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=200,
                       subplot_kw={'projection': map_proj})
ax.set_global()
ax.set_facecolor('#b0cfe0')
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
ax.coastlines(linewidth=0.5)
for cell_num, color in zip(test_indices, ['C0', 'C1', 'C2', 'C3']):
    lat_idx, lon_idx = ocn_idxs_surf[cell_num, 0], ocn_idxs_surf[cell_num, 1]
    ax.scatter(longitude[lon_idx], latitude[lat_idx], color=color, s=60, zorder=5,
               label=f'Cell {cell_num}', transform=data_crs)
ax.legend(fontsize=_fs, frameon=False, loc='lower center',
          bbox_to_anchor=(0.5, -0.01), ncol=len(test_indices),
          bbox_transform=fig.transFigure)
plt.tight_layout()
plt.show()

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

        AT_added = ds['AT_added'] * cv * rho * 1e-6 # mol
        exp_vars[f'AT_added_cum_{label}_{cell_num}'] = (
            AT_added.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                    .cumsum(dim='time')
                    .rename({'time': time_dim})
        )
        exp_vars[f'delxCO2_{label}_{cell_num}']  = ds['delxCO2'].rename({'time': time_dim})

        delCT = ds['delCT'] * cv * rho * 1e-6 # mol
        exp_vars[f'delCT_{label}_{cell_num}'] = (
            delCT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                 .rename({'time': time_dim})
        )
        delAT = ds['delAT'] * cv * rho * 1e-6 # mol
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

#%% plot impulse_response: combined 4×4 publication figure
# rows = variables (AT added, CO2, CT, AT), columns = cells
_ir_row_vars = [
    ('AT_added_cum', r'Cumulative ${A_{\mathrm{T}}}^{\prime}$ added (Pmol)', 1e-15),
    ('delxCO2',      r'CO$_2$ drawdown (ppm)',                               1.0),
    ('delCT',        r"${C_{\mathrm{T}}}^{\prime}$ (Pmol)",                 1e-15),
    ('delAT',        r"${A_{\mathrm{T}}}^{\prime}$ (Pmol)",                 1e-15),
]
_n_rows = len(_ir_row_vars)
_n_cols = len(ir_cell_nums)
_ir_linestyles = {
    'Annual':   '-',
    'Monthly':  '--',
    'Dekadal':  ':',
    'Pentadal': '-.',
    'Daily':    (0, (3, 1, 1, 1)),
}

fig, axes = plt.subplots(_n_rows, _n_cols,
                         figsize=(2.6 * _n_cols, 2.6 * _n_rows),
                         dpi=200, squeeze=False)
fig.subplots_adjust(bottom=0.08, hspace=0.18, wspace=0.18)

_ir_legend_handles = []
_ir_legend_text    = []

for row, (var_key, row_ylabel, scale) in enumerate(_ir_row_vars):
    for col, cell_num in enumerate(ir_cell_nums):
        ax = axes[row, col]
        for label in ir_labels:
            var      = ir_cache[f'{var_key}_{label}_{cell_num}']
            time_dim = f'time_{label}_{cell_num}'
            line,    = ax.plot(var[time_dim].values, var.values * scale, label=label,
                              linestyle=_ir_linestyles[label])
            if row == 0 and col == 0:
                _ir_legend_handles.append(line)
                _ir_legend_text.append(_display.get(label, label))
        if row == 0:
            ax.set_title(f'Cell {cell_num}', fontsize=_fs, color=textcolor)
        if col == 0:
            ax.set_ylabel(row_ylabel, fontsize=_fs)
        else:
            ax.set_yticklabels([])
        if row == _n_rows - 1:
            ax.set_xlabel('Year', fontsize=_fs)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        for side in ('top', 'bottom', 'left', 'right'):
            ax.spines[side].set_color(textcolor)

fig.legend(_ir_legend_handles, _ir_legend_text,
           loc='lower center', ncol=len(ir_labels),
           fontsize=_fs, bbox_to_anchor=(0.5, -0.02),
           frameon=False)
plt.show()

_col_w  = 12
_ref    = 'Daily'
_header = f"{'':12}" + ''.join(f"{'Cell '+str(c):>{_col_w}}" for c in ir_cell_nums)
print('\nImpulse response: AT added (% of daily at 2027)')
print(_header)
for label in ir_labels:
    if label != _ref:
        _row = f"{_display.get(label, label):<12}"
        for cell_num in ir_cell_nums:
            ref_val = ir_cache[f'AT_added_cum_{_ref}_{cell_num}'].sel(
                {f'time_{_ref}_{cell_num}': 2027}, method='nearest').values
            val = ir_cache[f'AT_added_cum_{label}_{cell_num}'].sel(
                {f'time_{label}_{cell_num}': 2027}, method='nearest').values
            _row += f"{val / ref_val * 100:>{_col_w}.2f}%"
        print(_row)

print('\nImpulse response: CO2 drawdown (% of daily at 2027)')
print(_header)
for label in ir_labels:
    if label != _ref:
        _row = f"{_display.get(label, label):<12}"
        for cell_num in ir_cell_nums:
            ref_val = ir_cache[f'delxCO2_{_ref}_{cell_num}'].sel(
                {f'time_{_ref}_{cell_num}': 2027}, method='nearest').values
            val = ir_cache[f'delxCO2_{label}_{cell_num}'].sel(
                {f'time_{label}_{cell_num}': 2027}, method='nearest').values
            _row += f"{val / ref_val * 100:>{_col_w}.2f}%"
        print(_row)

print('\nImpulse response: CT change (% of daily at 2027)')
print(_header)
for label in ir_labels:
    if label != _ref:
        _row = f"{_display.get(label, label):<12}"
        for cell_num in ir_cell_nums:
            ref_val = ir_cache[f'delCT_{_ref}_{cell_num}'].sel(
                {f'time_{_ref}_{cell_num}': 2027}, method='nearest').values
            val = ir_cache[f'delCT_{label}_{cell_num}'].sel(
                {f'time_{label}_{cell_num}': 2027}, method='nearest').values
            _row += f"{val / ref_val * 100:>{_col_w}.2f}%"
        print(_row)

print('\nImpulse response: AT change (% of daily at 2027)')
print(_header)
for label in ir_labels:
    if label != _ref:
        _row = f"{_display.get(label, label):<12}"
        for cell_num in ir_cell_nums:
            ref_val = ir_cache[f'delAT_{_ref}_{cell_num}'].sel(
                {f'time_{_ref}_{cell_num}': 2027}, method='nearest').values
            val = ir_cache[f'delAT_{label}_{cell_num}'].sel(
                {f'time_{label}_{cell_num}': 2027}, method='nearest').values
            _row += f"{val / ref_val * 100:>{_col_w}.2f}%"
        print(_row)

ir_cache.close()

# %%
