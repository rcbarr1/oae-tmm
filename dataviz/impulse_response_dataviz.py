"""
Created on Wed Jul 02 2026

DATAVIZ FOR IMPULSE RESPONSE: Global OAE efficiency map
Assembles per-cell η = ΔC_T / ΔA_T at 5 and 15 years into a 2×3 surface map
figure across three SSP scenarios.

Reference: Zhou et al. (2025), Nature Climate Change, 15, 59–65.

@author: Reese C. Barrett
"""
#%%
import argparse
import glob
import os

from dataviz.dataviz import broadcast_to_dataset, load_ocim_grid, apply_style
from scipy.spatial import KDTree
from oae_tmm.grid import flatten
from oae_tmm.trace import interp_trace
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm.auto import tqdm
from geopy.distance import geodesic
from scipy.stats import pearsonr

data_path = './data/'
# ir_path   = '/Volumes/LaCie/outputs/impulse_response/'
ir_path   = './outputs/'
ir_date   = '2026-07-24'  # set to the tag date used when production runs were submitted

textcolor, fontweight = apply_style()
_fs = 13

grid        = load_ocim_grid(data_path)
ocnmask     = grid['ocnmask']
latitude    = grid['latitude']
longitude   = grid['longitude']
depth       = grid['depth']
cell_volume = grid['cell_volume']
rho = 1025  # seawater density [kg m-3]

surf_mask_2d  = ocnmask[:, :, 0]
ocn_idxs_surf = np.argwhere(surf_mask_2d == 1)  # (N_cells, 2)

scenarios = ['none', 'ssp245', 'ssp534_OS']
YEAR_5YR  = 2027   # 5 years after CDR start (2022)
YEAR_15YR = 2037   # 15 years after CDR start
YEAR_50YR = 2072   # 50 years after CDR start

_parser = argparse.ArgumentParser()
_parser.add_argument('--scenario', default=None, choices=scenarios,
                     help='compute only this scenario\'s .npy intermediates and exit '
                          '(none, ssp245, ssp534_OS); omit to run all scenarios and generate figure')
_args, _ = _parser.parse_known_args()
_scenario_filter = _args.scenario


#%% validity check

def is_valid(ds):
    """Return True if ds ran to completion and has no corrupt final state.

    Checks that (1) the last time coordinate equals YEAR_15YR and (2) the
    final delCT field contains no NaN, which would indicate a truncated or
    solver-failed run.
    """
    if float(ds.time.values[-1]) < YEAR_50YR - 0.5:
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
    expected = [f'eta_{h}_{s}' for s in scenarios for h in ('5yr', '15yr', '50yr')]
    with xr.open_dataset(final_cache_path) as ds:
        if not all(v in ds for v in expected):
            return None
        cache = {v: ds[v].values for v in expected}
        if any(np.all(np.isnan(v)) for v in cache.values()):
            print('  Final cache incomplete (some scenarios all-NaN) — recomputing')
            return None
        return cache


def _compute_scenario(scenario):
    """Loop over all surface cells and compute η at 5 yr and 15 yr.

    Returns (eta_5yr, eta_15yr), each a 2D array (n_lat, n_lon) with np.nan
    for land cells and for ocean cells whose output is missing or invalid.
    """
    eta_5yr  = np.full(surf_mask_2d.shape, np.nan)
    eta_15yr = np.full(surf_mask_2d.shape, np.nan)
    eta_50yr = np.full(surf_mask_2d.shape, np.nan)

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
                if float(ds.time.values[-1]) >= YEAR_50YR:
                    eta_50yr[lat_idx, lon_idx] = float(
                        eta.sel(time=YEAR_50YR, method='nearest', tolerance=0.5).values)

        except Exception as e:
            print(f'  Failed cell {cell_num}: {e}')

    return eta_5yr, eta_15yr, eta_50yr


def _load_or_compute_scenario(scenario):
    """Return (eta_5yr, eta_15yr, eta_50yr) from .npy intermediates or by computing."""
    p5  = _npy_path(scenario, '5yr')
    p15 = _npy_path(scenario, '15yr')
    p50 = _npy_path(scenario, '50yr')

    if os.path.exists(p5) and os.path.exists(p15) and os.path.exists(p50):
        eta_5yr, eta_15yr, eta_50yr = np.load(p5), np.load(p15), np.load(p50)
        if np.any(np.isfinite(eta_5yr)) or np.any(np.isfinite(eta_15yr)):
            return eta_5yr, eta_15yr, eta_50yr
        print(f'  [{scenario}] cached .npy files are all-NaN — recomputing')

    eta_5yr, eta_15yr, eta_50yr = _compute_scenario(scenario)
    np.save(p5,  eta_5yr)
    np.save(p15, eta_15yr)
    np.save(p50, eta_50yr)
    return eta_5yr, eta_15yr, eta_50yr


eta_cache = _load_final_cache()

if eta_cache is None:
    if _scenario_filter is not None:
        # Single-scenario mode: compute .npy intermediates for one scenario and exit.
        # Re-run without --scenario once all scenarios are done to assemble the
        # final cache and generate the figure.
        _load_or_compute_scenario(_scenario_filter)
        import sys; sys.exit(0)

    eta_cache = {}
    for scenario in scenarios:
        eta_5, eta_15, eta_50 = _load_or_compute_scenario(scenario)
        eta_cache[f'eta_5yr_{scenario}']  = eta_5
        eta_cache[f'eta_15yr_{scenario}'] = eta_15
        eta_cache[f'eta_50yr_{scenario}'] = eta_50

    ds_out = xr.Dataset(
        {k: xr.DataArray(v, dims=['latitude', 'longitude'],
                         coords={'latitude': latitude, 'longitude': longitude})
         for k, v in eta_cache.items()}
    )
    ds_out.to_netcdf(final_cache_path)


#%% figure: 2×3 global efficiency map

_scenario_labels = {
    'none':      'Fixed Atm. CO$_{2}$',
    'ssp245':    'SSP2-4.5',
    'ssp534_OS': 'SSP5-3.4 OS',
}
_row_labels = [r'$\eta$ at 5 years', r'$\eta$ at 15 years', r'$\eta$ at 50 years']
_horizons   = [('5yr', YEAR_5YR), ('15yr', YEAR_15YR), ('50yr', YEAR_50YR)]
_panel_ids  = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

data_crs = ccrs.PlateCarree()
map_proj = ccrs.EqualEarth(central_longitude=200)

fig, axes = plt.subplots(
    3, 3,
    figsize=(12, 4.5),
    dpi=200,
    subplot_kw={'projection': map_proj},
)
fig.subplots_adjust(hspace=0.08, wspace=0.04, top=0.88, bottom=0.12, left=0.06, right=0.88)

_vmin, _vmax = 0, 100
im = None

for row, (horizon, _) in enumerate(_horizons):
    for col, scenario in enumerate(scenarios):
        ax    = axes[row, col]
        data  = eta_cache[f'eta_{horizon}_{scenario}']
        panel = _panel_ids[row * 3 + col]

        ax.set_global()

        im = ax.pcolormesh(
            longitude, latitude, data * 100, # as a percentage
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
cbar.set_label(r'$\eta$ (%, mol ${C_{\mathrm{T}}}^{\prime}$ / mol ${A_{\mathrm{T}}}^{\prime}$)', fontsize=_fs, fontweight=fontweight)
cbar.set_ticks(np.linspace(_vmin, _vmax, 6))
cbar.ax.yaxis.label.set_color(textcolor)
cbar.ax.tick_params(colors=textcolor, labelsize=_fs)

plt.show()

#%%  statistics
# spatial correlation coefficient
_lw_corr  = 32
_vw_corr  = 10
_sep_corr = '=' * (_lw_corr + _vw_corr)

print('Pairwise Spatial Correlation (Pearson r) Between Scenarios')
print(_sep_corr)
print(f"{'Pair':>{_lw_corr}}{'r':>{_vw_corr}}")
for horizon, _ in _horizons:
    for i, s1 in enumerate(scenarios):
        for s2 in scenarios[i + 1:]:
            d1   = flatten(eta_cache[f'eta_{horizon}_{s1}'], ocnmask[:, :, 0])
            d2   = flatten(eta_cache[f'eta_{horizon}_{s2}'], ocnmask[:, :, 0])
            mask = np.isfinite(d1) & np.isfinite(d2)
            r_val, _ = pearsonr(d1[mask], d2[mask])
            label = f'{horizon}: {s1} vs {s2}'
            print(f"{label:>{_lw_corr}}{r_val:>{_vw_corr - 1}.6f}")
print(_sep_corr)

# ice-free efficiency range, average, and standard deviation
_tbl_vars = ['Mean', 'Std. Dev', 'Min', 'Max']
_lw  = 16
_vw  = 12
_sep = '=' * (_lw + _vw * len(_tbl_vars))

f_ice = xr.open_dataset(data_path + 'ncep_doe_reanalysis_ii/icec.nc')['icec'].transpose('latitude', 'longitude').values
icemask = flatten((f_ice <= 0.05).astype(int), ocnmask[:, :, 0])
weights = flatten(cell_volume[:, :, 0], ocnmask[:, :, 0]) * 0 + 1

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
data = eta_cache[f'eta_15yr_ssp245'] 
print(f"\neta at (49.5 ºN, 201 ºE): {data[70, 100] * 100:.2f}%")
print(f"eta at (41.5 ºN, 201 ºE): {data[66, 100] * 100:.2f}%")
distance = geodesic((latitude[70], longitude[100]), (latitude[66], longitude[100])).km
print(f'distance between these coords: {distance:.2f} (km)')

# total anthropogenic C added to ocean (PgC)
_lw  = 6
_vw  = 12
_sep = '=' * (_lw + _vw * len(_tbl_vars))

print('Anthropogenic Carbon Statistics (PgC accumulated)')
print(_sep)
print(f"{'':>{_lw}}" + ''.join(f"{s:>{_vw}}" for s in scenarios))

for horizon, year in _horizons:
    _row = f"{horizon}\t"
    for scenario in scenarios:
        if scenario == 'none':
            _Canth = 0
        else:
            data  = interp_trace(data_path, year, scenario, latitude, longitude, depth, ocnmask) - interp_trace(data_path, 2022, scenario, latitude, longitude, depth, ocnmask)
            _Canth = np.nansum(data * rho * cell_volume * 1e-6) * 1e-15 * 12.011 # PgC
        _row += f"{float(_Canth):>{_vw - 1}.2f}"
    print(_row)
print(_sep)

# spatial variation within 500 km (≈ equivalent circular radius of a median Zhou et al. open-ocean patch)
R_EARTH_KM = 6371.0
RADIUS_KM  = 500.0
chord_dist = 2 * np.sin(RADIUS_KM / (2 * R_EARTH_KM))

lats_r = np.radians(latitude[ocn_idxs_surf[:, 0]])
lons_r = np.radians(longitude[ocn_idxs_surf[:, 1]])
xyz = np.column_stack([
    np.cos(lats_r) * np.cos(lons_r),
    np.cos(lats_r) * np.sin(lons_r),
    np.sin(lats_r),
])
tree = KDTree(xyz)
neighbor_lists = tree.query_ball_point(xyz, chord_dist)

ice_flat = (f_ice <= 0.05)[ocn_idxs_surf[:, 0], ocn_idxs_surf[:, 1]]

_slices = [
    ('5yr',  eta_cache['eta_5yr_ssp245']),
    ('15yr', eta_cache['eta_15yr_ssp245']),
    ('50yr', eta_cache['eta_50yr_ssp245']),
]

_lw3   = 6
_vw3   = 16
_hdrs3 = ['Max range', '95th pct range', 'Mean range']
_sep3  = '=' * (_lw3 + _vw3 * len(_hdrs3))

print(f'\nSpatial variation within {RADIUS_KM:.0f} km radius (SSP2-4.5, ice-free, %)')
print(f'(≈ equivalent circular radius of Zhou et al. median open-ocean patch, area = 7.2×10⁵ km²)')
print(_sep3)
print(f"{'':>{_lw3}}" + ''.join(f"{h:>{_vw3}}" for h in _hdrs3))
for horizon, eta_2d in _slices:
    eta_flat = eta_2d[ocn_idxs_surf[:, 0], ocn_idxs_surf[:, 1]]
    ranges = np.array([
        (np.nanmax(eta_flat[nb]) - np.nanmin(eta_flat[nb]))
        if np.any(np.isfinite(eta_flat[nb])) else np.nan
        for nb in neighbor_lists
    ])
    valid = ice_flat & np.isfinite(ranges)
    r = ranges[valid]
    _row  = f"{horizon:>{_lw3}}"
    _row += f"{float(np.max(r) * 100):>{_vw3 - 1}.2f}%"
    _row += f"{float(np.percentile(r, 95) * 100):>{_vw3 - 1}.2f}%"
    _row += f"{float(np.mean(r) * 100):>{_vw3 - 1}.2f}%"
    print(_row)
print(_sep3)

#%%