#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 2026

DATA VIZ FOR EXP25: Attempting to replicate map from Yamamoto et al., 2024
- load in each .nc file (one experiment per .nc file), calculate maximum cumulative additionality
- maximum cumulative additionality =  MAX( (delxCO2 * Ma) / (CT_added_total * cell_volume * rho) )
- save the additionalities to the appropriate place in the OCIM grid
- plot grids

@author: Reese C. Barrett
"""
#%%
from dataviz.dataviz import plot_surface2d, broadcast_to_dataset
import xarray as xr
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import warnings

# load model architecture
data_path = './data/'
output_path = '/Volumes/LaCie/outputs/exp25/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

latitude    = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
longitude   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
depth       = model_data['tz'].isel(longitude=0, latitude=0).to_numpy()   # m below sea surface
cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

model_data.close()
rho = 1025   # seawater density [kg m-3]
Ma  = 1.8e26 # micromoles of air in atmosphere [µmol air]

# suppress divide-by-NaN warnings from land boxes
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

#%% pull in all experiments (CT release from an individual grid cell across all grid cells)
experiment_names = ['exp25_2026-02-05_t-mixed_' + str(i) for i in range(8400, 8601)]

# set up array to save maximum cumulative additionality in
max_alphas = np.full(ocnmask[:, :, 0].shape, np.nan)
failed_experiments = []

# calculate max_alpha for each experiment
for experiment_name in tqdm(experiment_names):
    try:
        with xr.open_mfdataset(
                output_path + experiment_name + '_*.nc',
                combine='by_coords',
                chunks={'time': 10},
                parallel=True) as ds:

            # check that there is data until model year 100
            ds.delxCO2.sel(time=2102).values

            cell_volume_xr = broadcast_to_dataset(cell_volume, ds)
            cum_CT_added = ds.CT_added.cumsum(dim='time')

            # alpha = (µmol air * (1e-6 mol/µmol) * (µmol CO2 / mol air)) / (µmol CO2 kg-1 * m3 * kg m-3) * 100%
            alpha = (Ma * 1e-6 * ds.delxCO2) / (cum_CT_added * cell_volume_xr * rho).sum(dim=['latitude', 'longitude', 'depth']) * 100

            # find lat and lon of CT release, store max_alpha at correct grid location
            CT_location = np.argwhere(ds.CT_added.isel(time=1).transpose('latitude', 'longitude', 'depth').values < 0)
            lats, lons, _ = CT_location[0]
            max_alphas[lats, lons] = float(alpha.max())

    except Exception as e:
        print(f"Failed: {experiment_name} -> {e}")
        failed_experiments.append(experiment_name)

#%% used to combine two separate runs into one output array
# max_alphas_old  = np.load(output_path + 'max_alphas.npy')
# max_alphas_full = np.nansum(np.dstack((max_alphas, max_alphas_old)), 2)
# np.save(output_path + 'max_alphas.npy', max_alphas_full)

np.save(output_path + 'max_alphas.npy', max_alphas)

#%% plot additionality
max_alphas = np.load(output_path + 'max_alphas.npy')

colors = ['#5d4e9f', '#5e67a2', '#607ba4', '#6393a7', '#64a9ac',
          '#65c0ae', '#8bd0b1', '#b2dfb4', '#daefb7', '#feffbb',
          '#fee2a3', '#fcc58d', '#faa974', '#f78b5d', '#f56e46',
          '#e45744', '#d24244', '#c12e43', '#af1843', '#9d0142']
cmap = ListedColormap(colors, name='yamamoto')

plot_surface2d(latitude, longitude, max_alphas, 0, 100, cmap, 'maximum cumulative additionality')

# %% watch what happens with single time step
ds = xr.open_dataset('./outputs/exp25_2026-02-03_t-mixed_290_000.nc')

for t_idx in tqdm(range(0, 4)):
    plot_surface2d(latitude, longitude, ds.delCT.isel(time=t_idx, depth=0).values, 0, 100, 'viridis', 'delCT at t = ' + str(ds.time.isel(time=t_idx).values))

ds.close()
# %%
