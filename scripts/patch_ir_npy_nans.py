"""
Patch NaN holes in ir_eta_*_*_2026-07-24.npy using newly rerun .nc files.

Only processes cells that are currently NaN in the existing .npy files,
so no .nc files are needed for cells that already have valid data.

CLI usage:
    python -m dataviz.patch_ir_npy_nans
"""
#%%
import glob
import numpy as np
import xarray as xr

from dataviz.dataviz import broadcast_to_dataset, load_ocim_grid

data_path = './data/'
ir_path   = './outputs/'
ir_date   = '2026-07-24'
scenario  = 'ssp534_OS'
YEAR_5YR  = 2027
YEAR_15YR = 2037
YEAR_50YR = 2072
rho = 1025

grid          = load_ocim_grid(data_path)
ocnmask       = grid['ocnmask']
cell_volume   = grid['cell_volume']
surf_mask_2d  = ocnmask[:, :, 0]
ocn_idxs_surf = np.argwhere(surf_mask_2d == 1)


def _npy_path(horizon):
    return ir_path + f'ir_eta_{scenario}_{horizon}_{ir_date}.npy'


eta_5yr  = np.load(_npy_path('5yr'))
eta_15yr = np.load(_npy_path('15yr'))
eta_50yr = np.load(_npy_path('50yr'))

nan_cells = [
    (cell_num, lat_idx, lon_idx)
    for cell_num, (lat_idx, lon_idx) in enumerate(ocn_idxs_surf)
    if np.isnan(eta_5yr[lat_idx, lon_idx])
    or np.isnan(eta_15yr[lat_idx, lon_idx])
    or np.isnan(eta_50yr[lat_idx, lon_idx])
]
_all_scenarios   = ['none', 'ssp245', 'ssp534_OS']
_scenario_offset = _all_scenarios.index(scenario) * len(ocn_idxs_surf)
print(f'Found {len(nan_cells)} NaN cells to patch')
print('Cell numbers:', [cell_num for cell_num, _, _ in nan_cells])
print('Experiment IDs:', [_scenario_offset + cell_num for cell_num, _, _ in nan_cells])

#%%
filled = 0
for cell_num, lat_idx, lon_idx in nan_cells:
    pattern = ir_path + f'impulse_response_{ir_date}_{scenario}_{cell_num:05d}_*.nc'
    files   = sorted(glob.glob(pattern))
    if not files:
        continue
    try:
        with xr.open_mfdataset(files, combine='by_coords') as ds:
            cv = broadcast_to_dataset(cell_volume, ds)
            with np.errstate(invalid='ignore', divide='ignore'):
                delCT_total = (ds['delCT'] * rho * cv * 1e-6).sum(
                    dim=['latitude', 'longitude', 'depth'], skipna=True)
                delAT_total = (ds['delAT'] * rho * cv * 1e-6).sum(
                    dim=['latitude', 'longitude', 'depth'], skipna=True)
                eta = delCT_total / delAT_total

            eta_5yr[lat_idx, lon_idx]  = float(
                eta.sel(time=YEAR_5YR,  method='nearest', tolerance=0.5).values)
            eta_15yr[lat_idx, lon_idx] = float(
                eta.sel(time=YEAR_15YR, method='nearest', tolerance=0.5).values)
            if float(ds.time.values[-1]) >= YEAR_50YR:
                eta_50yr[lat_idx, lon_idx] = float(
                    eta.sel(time=YEAR_50YR, method='nearest', tolerance=0.5).values)
        filled += 1
    except Exception as e:
        print(f'  Failed cell {cell_num}: {e}')

print(f'Patched {filled} / {len(nan_cells)} cells')
np.save(_npy_path('5yr'),  eta_5yr)
np.save(_npy_path('15yr'), eta_15yr)
np.save(_npy_path('50yr'), eta_50yr)
print('Done')

# %%
