"""
Created on Wed Jul 02 2026

DATAVIZ FOR IMPULSE RESPONSE: Global OAE efficiency map
Assembles per-cell η = ΔC_T / ΔA_T at 5 and 15 years into a 2×3 surface map
figure across three SSP scenarios.

Reference: Zhou et al. (2025), Nature Climate Change, 15, 59–65.

@author: Reese C. Barrett
"""
#%%
import glob
import os

from dataviz.dataviz import broadcast_to_dataset
from oae_tmm.grid import flatten
from oae_tmm.trace import interp_trace
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm.auto import tqdm
from geopy.distance import geodesic

data_path = './data/'
ir_path   = '/Volumes/LaCie/outputs/impulse_response/'
ir_path   = './outputs/'
ir_date   = '2026-07-01'  # set to the tag date used when production runs were submitted

fontweight = 'normal'
textcolor = '#000000'
mpl.rcParams['font.family']     = 'Calibri'
mpl.rcParams['font.weight']     = fontweight
mpl.rcParams['text.color']      = textcolor
mpl.rcParams['axes.labelcolor'] = textcolor
mpl.rcParams['xtick.color']     = textcolor
mpl.rcParams['ytick.color']     = textcolor
_fs = 13

model_data  = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask     = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()
latitude    = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()
longitude   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()
depth       = model_data['tz'].isel(longitude=0, latitude=0).to_numpy()
cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy()
model_data.close()
rho = 1025  # seawater density [kg m-3]

surf_mask_2d  = ocnmask[:, :, 0]
ocn_idxs_surf = np.argwhere(surf_mask_2d == 1)  # (N_cells, 2)

scenarios = ['none', 'ssp126', 'ssp534_OS']
YEAR_5YR  = 2027   # 5 years after CDR start (2022)
YEAR_15YR = 2037   # 15 years after CDR start


#%% validity check

def is_valid(ds):
    """Return True if ds ran to completion and has no corrupt final state.

    Checks that (1) the last time coordinate equals YEAR_15YR and (2) the
    final delCT field contains no NaN, which would indicate a truncated or
    solver-failed run.
    """
    if float(ds.time.values[-1]) != YEAR_15YR:
        return False
    if not np.any(np.isfinite(ds['delCT'].isel(time=-1).values)):
        return False
    return True


#%% compute or load cached time series

final_cache_path = ir_path + f'ir_efficiency_cache_{ir_date}.nc'


def _npy_path(scenario, horizon):
    return ir_path + f'ir_eta_{scenario}_{horizon}_{ir_date}.npy'


def _load_final_cache():
    """Return dict of 6 eta arrays from the .nc cache, or None if incomplete."""
    if not os.path.exists(final_cache_path):
        return None
    expected = [f'eta_{h}_{s}' for s in scenarios for h in ('5yr', '15yr')]
    with xr.open_dataset(final_cache_path) as ds:
        if not all(v in ds for v in expected):
            return None
        cache = {v: ds[v].values for v in expected}
        if all(np.all(np.isnan(v)) for v in cache.values()):
            print('  Final cache exists but is all-NaN — recomputing')
            return None
        return cache


def _compute_scenario(scenario):
    """Loop over all surface cells and compute η at 5 yr and 15 yr.

    Returns (eta_5yr, eta_15yr), each a 2D array (n_lat, n_lon) with np.nan
    for land cells and for ocean cells whose output is missing or invalid.
    """
    eta_5yr  = np.full(surf_mask_2d.shape, np.nan)
    eta_15yr = np.full(surf_mask_2d.shape, np.nan)

    for cell_num, (lat_idx, lon_idx) in enumerate(
            tqdm(ocn_idxs_surf, desc=f'Computing η [{scenario}]')):

        pattern = ir_path + f'impulse_response_{ir_date}_{scenario}_{cell_num:05d}_*.nc'
        files   = sorted(glob.glob(pattern))
        if not files:
            continue

        try:
            with xr.open_mfdataset(files, combine='by_coords') as ds:
                if not is_valid(ds):
                    continue

                cv = broadcast_to_dataset(cell_volume, ds)

                with np.errstate(invalid='ignore', divide='ignore'):
                    delCT_total = (ds['delCT'] * rho * cv * 1e-6).sum(
                        dim=['latitude', 'longitude', 'depth'], skipna=True)
                    delAT_total = (ds['delAT'] * rho * cv * 1e-6).sum(
                        dim=['latitude', 'longitude', 'depth'], skipna=True)
                    eta = delCT_total / delAT_total

                eta_5yr[lat_idx, lon_idx] = float(
                    eta.sel(time=YEAR_5YR,  method='nearest', tolerance=0.5).values)
                eta_15yr[lat_idx, lon_idx] = float(
                    eta.sel(time=YEAR_15YR, method='nearest', tolerance=0.5).values)

        except Exception as e:
            print(f'  Failed cell {cell_num}: {e}')

    return eta_5yr, eta_15yr


def _load_or_compute_scenario(scenario):
    """Return (eta_5yr, eta_15yr) from .npy intermediates or by computing."""
    p5  = _npy_path(scenario, '5yr')
    p15 = _npy_path(scenario, '15yr')

    if os.path.exists(p5) and os.path.exists(p15):
        eta_5yr, eta_15yr = np.load(p5), np.load(p15)
        if np.any(np.isfinite(eta_5yr)) or np.any(np.isfinite(eta_15yr)):
            return eta_5yr, eta_15yr
        print(f'  [{scenario}] cached .npy files are all-NaN — recomputing')

    eta_5yr, eta_15yr = _compute_scenario(scenario)
    np.save(p5,  eta_5yr)
    np.save(p15, eta_15yr)
    return eta_5yr, eta_15yr


eta_cache = _load_final_cache()

if eta_cache is None:
    eta_cache = {}
    for scenario in scenarios:
        eta_5, eta_15 = _load_or_compute_scenario(scenario)
        eta_cache[f'eta_5yr_{scenario}']  = eta_5
        eta_cache[f'eta_15yr_{scenario}'] = eta_15

    ds_out = xr.Dataset(
        {k: xr.DataArray(v, dims=['latitude', 'longitude'],
                         coords={'latitude': latitude, 'longitude': longitude})
         for k, v in eta_cache.items()}
    )
    ds_out.to_netcdf(final_cache_path)


#%% figure: 2×3 global efficiency map

_scenario_labels = {
    'none':      'No SSP',
    'ssp126':    'SSP1-2.6',
    'ssp534_OS': 'SSP5-3.4 OS',
}
_row_labels = [r'$\eta$ at 5 years', r'$\eta$ at 15 years']
_horizons   = [('5yr', YEAR_5YR), ('15yr', YEAR_15YR)]
_panel_ids  = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

data_crs = ccrs.PlateCarree()
map_proj = ccrs.EqualEarth(central_longitude=200)

fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 4.5),
    dpi=200,
    subplot_kw={'projection': map_proj},
)
fig.subplots_adjust(hspace=0.08, wspace=0.04, top=0.88, bottom=0.12, left=0.06, right=0.88)

_vmin, _vmax = 0.0, 1.0
im = None

for row, (horizon, _) in enumerate(_horizons):
    for col, scenario in enumerate(scenarios):
        ax    = axes[row, col]
        data  = eta_cache[f'eta_{horizon}_{scenario}']
        panel = _panel_ids[row * 3 + col]

        ax.set_global()

        im = ax.pcolormesh(
            longitude, latitude, data,
            cmap='viridis', vmin=_vmin, vmax=_vmax,
            transform=data_crs, zorder=1,
        )

        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
        ax.coastlines(linewidth=0.5, zorder=3)

        ax.text(0.01, 0.97, panel,
                transform=ax.transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

        for spine in ax.spines.values():
            spine.set_color(textcolor)

        if row == 0:
            ax.set_title(_scenario_labels[scenario], fontsize=_fs, fontweight=fontweight, color=textcolor, pad=6)

        if col == 0:
            ax.text(-0.05, 0.5, _row_labels[row],
                    transform=ax.transAxes,
                    fontsize=_fs, va='center', ha='right',
                    rotation=90, color=textcolor)

cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
cbar    = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$\eta$ (mol ${C_{\mathrm{T}}}^{\prime}$ / mol ${A_{\mathrm{T}}}^{\prime}$)', fontsize=_fs, fontweight=fontweight)
cbar.set_ticks(np.linspace(_vmin, _vmax, 6))
cbar.ax.yaxis.label.set_color(textcolor)
cbar.ax.tick_params(colors=textcolor, labelsize=_fs)

plt.show()

#%%  statistics

# ice-free efficiency range, average, and standard deviation
_tbl_vars = ['Mean', 'Std. Dev', 'Min', 'Max']
_lw  = 16
_vw  = 12
_sep = '=' * (_lw + _vw * len(_tbl_vars))

f_ice = xr.open_dataset(data_path + 'ncep_doe_reanalysis_ii/icec.nc')['icec'].transpose('latitude', 'longitude').values
icemask = flatten((f_ice <= 0.05).astype(int), ocnmask[:, :, 0])
weights = flatten(cell_volume[:, :, 0], ocnmask[:, :, 0])

print('Ice-Free OAE Efficiency Statistics (eta, %)')
print(_sep)
print(f"{'':>{_lw}}" + ''.join(f"{h:>{_vw}}" for h in _tbl_vars))
for horizon, _ in _horizons:
    for scenario in scenarios:
        _row = f"{horizon}, {scenario}\t"
        data  = flatten(eta_cache[f'eta_{horizon}_{scenario}'], ocnmask[:, :, 0])
        mask  = icemask.astype(bool) & np.isfinite(data)
        d, w  = data[mask], weights[mask]
        _mean = np.average(d, weights=w)
        _std  = np.sqrt(np.average((d - _mean) ** 2, weights=w))
        _low  = np.min(d)
        _high = np.max(d)
        _vars = [_mean, _std, _low, _high]
        for _var in _vars:
            _row += f"{float(_var * 100):>{_vw - 1}.2f}%"
        print(_row)
print(_sep)

# North Pacific example
data = eta_cache[f'eta_15yr_ssp126'] 
print(f"\neta at (49.5 ºN, 201 ºE): {data[70, 100] * 100:.2f}%")
print(f"eta at (41.5 ºN, 201 ºE): {data[66, 100] * 100:.2f}%")
distance = geodesic((latitude[70], longitude[100]), (latitude[66], longitude[100])).km
print(f'distance between these coords: {distance:.2f} (km)')

# total anthropogenic C added to ocean (Pmol)
_lw  = 6
_vw  = 12
_sep = '=' * (_lw + _vw * len(_tbl_vars))

print('Anthropogenic Carbon Statistics (Pmol accumulated)')
print(_sep)
print(f"{'':>{_lw}}" + ''.join(f"{s:>{_vw}}" for s in scenarios))

for horizon, year in _horizons:
    _row = f"{horizon}\t"
    for scenario in scenarios:
        if scenario == 'none':
            _Canth = 0
        else:
            data  = interp_trace(data_path, year, scenario, latitude, longitude, depth, ocnmask) - interp_trace(data_path, 2022, scenario, latitude, longitude, depth, ocnmask)
            _Canth = np.nansum(data * rho * cell_volume * 1e-6) * 1e-15 # Pmol
        _row += f"{float(_Canth):>{_vw - 1}.2f}"
    print(_row)
print(_sep)

#%%