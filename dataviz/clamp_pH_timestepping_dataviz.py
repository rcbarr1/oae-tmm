"""
Created on Wed Aug  6 2026

DATAVIZ FOR TIMESTEPPING COMPARISON — clamp_pH experiment

@author: Reese C. Barrett
"""
#%%
import os

from dataviz.dataviz import broadcast_to_dataset, load_ocim_grid, apply_style
import xarray as xr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

data_path      = './data/'
clamp_pH_path  = './outputs/'

# Set to the tag date used when the experiments were run (--test --exp-id 0-6)
clamp_pH_date = '2026-08-04'

grid        = load_ocim_grid(data_path)
cell_volume = grid['cell_volume']
rho = 1025  # seawater density [kg m-3]

textcolor, fontweight = apply_style()
_fs = 13
_display = {'Dekadal': '10-Day', 'Pentadal': '5-Day'}


# ============================================================================ #
#%%  CLAMP_PH TIMESTEPPING                                                      #
# ============================================================================ #

clamp_pH_names = [
    f'clamp_pH_{clamp_pH_date}_annually_none',
    f'clamp_pH_{clamp_pH_date}_monthly_none',
    f'clamp_pH_{clamp_pH_date}_dekadal_none',
    f'clamp_pH_{clamp_pH_date}_pentadal_none',
    f'clamp_pH_{clamp_pH_date}_daily_none',
    f'clamp_pH_{clamp_pH_date}_hourly_none',
    f'clamp_pH_{clamp_pH_date}_mixed_none',
]
clamp_pH_labels = ['Annual', 'Monthly', 'Dekadal', 'Pentadal', 'Daily', 'Hourly', 'Mixed']

#%% compute or load cached clamp_pH time series
clamp_pH_cache_path = clamp_pH_path + 'clamp_pH_timestepping_cache.nc'

def _load_clamp_pH(name, label):
    time_dim = f'time_{label}'
    exp_vars = {}
    with xr.open_mfdataset(
            clamp_pH_path + name + '_*.nc',
            combine='by_coords', chunks={'time': 10}, parallel=True) as ds:

        cv = broadcast_to_dataset(cell_volume, ds)

        AT_added = ds['AT_added'] * cv * rho * 1e-6  # mol
        exp_vars[f'AT_added_cum_{label}'] = (
            AT_added.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                    .cumsum(dim='time')
                    .rename({'time': time_dim})
        )
        exp_vars[f'delxCO2_{label}'] = ds['delxCO2'].rename({'time': time_dim})

        delCT = ds['delCT'] * cv * rho * 1e-6  # mol
        exp_vars[f'delCT_{label}'] = (
            delCT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                 .rename({'time': time_dim})
        )
        delAT = ds['delAT'] * cv * rho * 1e-6  # mol
        exp_vars[f'delAT_{label}'] = (
            delAT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                 .rename({'time': time_dim})
        )
    return exp_vars


if os.path.exists(clamp_pH_cache_path):
    clamp_pH_cache = xr.open_dataset(clamp_pH_cache_path).load()
    missing = [(n, l) for n, l in zip(clamp_pH_names, clamp_pH_labels)
               if f'AT_added_cum_{l}' not in clamp_pH_cache]
    if missing:
        new_vars = {}
        for name, label in tqdm(missing, desc='Computing missing clamp_pH experiments'):
            new_vars.update(_load_clamp_pH(name, label))
        with TqdmCallback(desc='Computing missing clamp_pH cache entries'):
            clamp_pH_cache = xr.merge([clamp_pH_cache, xr.Dataset(new_vars).compute()])
        clamp_pH_cache.to_netcdf(clamp_pH_cache_path)
else:
    cache_vars = {}
    for name, label in tqdm(zip(clamp_pH_names, clamp_pH_labels), total=len(clamp_pH_labels),
                            desc='Loading clamp_pH experiments'):
        cache_vars.update(_load_clamp_pH(name, label))
    with TqdmCallback(desc='Computing clamp_pH cache'):
        clamp_pH_cache = xr.Dataset(cache_vars).compute()
    clamp_pH_cache.to_netcdf(clamp_pH_cache_path)

#%% plot clamp_pH: combined 2×2 publication figure
_panel_vars = [
    ('AT_added_cum', r'Cumulative net ${A_{\mathrm{T}}}^{\prime}$ (Pmol)', 1e-15),
    ('delxCO2',      r'CO$_2$ drawdown (ppm)',                             1.0),
    ('delCT',        r"${C_{\mathrm{T}}}^{\prime}$ (PgC)",                1e-15 * 12.011),
    ('delAT',        r"${A_{\mathrm{T}}}^{\prime}$ (Pmol)",               1e-15),
]
_panel_labels = [('(a)', 0.02, 'left'), ('(b)', 0.98, 'right'),
                 ('(c)', 0.02, 'left'), ('(d)', 0.02, 'left')]

fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=200)
fig.subplots_adjust(hspace=0.32, wspace=0.32)

_legend_handles = []
_legend_text    = []

for ax, (var_key, ylabel, scale), (panel_lbl, px, pa) in zip(axes.flat, _panel_vars, _panel_labels):
    for label in clamp_pH_labels:
        var      = clamp_pH_cache[f'{var_key}_{label}']
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
    ('AT_added_cum', "net AT'"),
    ('delxCO2',      'CO2 drawdown'),
    ('delCT',        "CT'"),
    ('delAT',        "AT'"),
]
_lw  = 10
_vw  = 14
_yr  = 2035
_sep = '=' * (_lw + _vw * len(_tbl_vars))

print(f'\nclamp_pH timestepping: % of hourly response at {_yr}')
print(_sep)
print(f"{'':>{_lw}}" + ''.join(f"{h:>{_vw}}" for _, h in _tbl_vars))
print('-' * (_lw + _vw * len(_tbl_vars)))
for label in clamp_pH_labels:
    if label != 'Hourly':
        _row = f"{_display.get(label, label):<{_lw}}"
        for var_key, _ in _tbl_vars:
            val     = clamp_pH_cache[f'{var_key}_{label}'].sel(
                        {f'time_{label}': _yr}, method='nearest')
            ref_val = clamp_pH_cache[f'{var_key}_Hourly'].sel(
                        time_Hourly=_yr, method='nearest')
            _row   += f"{float(val / ref_val * 100):>{_vw - 1}.2f}%"
        print(_row)
print(_sep)

clamp_pH_cache.close()

# %%
