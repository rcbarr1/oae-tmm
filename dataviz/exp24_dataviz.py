#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 2026

DATA VIZ FOR EXP24: Attempting to replicate map from Zhou et al., 2025
- load in each .nc file (one experiment per .nc file), calculate efficiency
--> total delCT / total delAT at 5 and 15 years
- save the efficiencies (nu) to the appropriate place in the OCIM grid
- plot grids

@author: Reese C. Barrett
"""
#%%
from dataviz.dataviz import plot_surface2d, plot_longitude3d, broadcast_to_dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# load model architecture
data_path = './data/'
#output_path = './outputs/'
output_path = '/Volumes/LaCie/outputs/exp24/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

latitude    = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
longitude   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
depth       = model_data['tz'].isel(longitude=0, latitude=0).to_numpy()   # m below sea surface
cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

model_data.close()
rho = 1025 # seawater density for volume to mass [kg m-3]

#%% pull in all experiments (AT release from an individual grid cell across all grid cells)
experiment_names = []
for i in range(4206, 10442):
    experiment_names.append('exp24_2026-02-12_t-mixed_' + f'{i:05d}')

# set up array to save nu in
nus_5years = np.full(ocnmask[:, :, 0].shape, np.nan)
nus_15years = np.full(ocnmask[:, :, 0].shape, np.nan)
failed_experiments = []

# calculate nu for each experiment
for experiment_name in tqdm(experiment_names):
    try:
        with xr.open_mfdataset(
                output_path + experiment_name + '_*.nc',
                combine='by_coords',
                chunks={'time': 10},
                parallel=True) as ds:

            cell_volume_xr = broadcast_to_dataset(cell_volume, ds)

            # convert delCT and delAT from µmol kg-1 to mol, sum over ocean volume
            delCT = (ds.delCT * rho * cell_volume_xr * 1e-6).sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
            delAT  = (ds.delAT  * rho * cell_volume_xr * 1e-6).sum(dim=['latitude', 'longitude', 'depth'], skipna=True)

            nu = delCT / delAT
            nu_5years  = nu.sel(time=2007).values
            nu_15years = nu.sel(time=2017).values

            # find lat and lon of alkalinity release, store nu at correct grid location
            alk_location = np.argwhere(ds.AT_added.isel(time=1).transpose('latitude', 'longitude', 'depth').values > 0)
            lats, lons, _ = alk_location[0]
            nus_5years[lats, lons]  = nu_5years
            nus_15years[lats, lons] = nu_15years

    except Exception as e:
        print(f"Failed: {experiment_name} -> {e}")
        failed_experiments.append(experiment_name)
        continue

#%% used to combine two separate runs into one output array
# nus_5years_old  = np.load(output_path + 'nus5yrs_dtmixed_PT1.npy')
# nus_15years_old = np.load(output_path + 'nus15yrs_dtmixed_PT1.npy')

# nus_5years_full  = np.nansum(np.dstack((nus_5years,  nus_5years_old)),  2)
# nus_15years_full = np.nansum(np.dstack((nus_15years, nus_15years_old)), 2)

# np.save(output_path + 'nus15yrs_dtmixed.npy', nus_15years_full)
# np.save(output_path + 'nus5yrs_dtmixed.npy',  nus_5years_full)

#%% plot efficiency to match zhou map
nus_5years_full  = np.load(output_path + 'nus5yrs_dtmixed.npy')
nus_15years_full = np.load(output_path + 'nus15yrs_dtmixed.npy')

vmin = 0
vmax = 1
fig, axs = plt.subplots(2, 1, dpi=200, figsize=(6.2, 8))

# rotate lons to start at 20ºE to match Zhou et al. map
split_idx    = np.where(longitude >= 20)[0][0]
longitude_rot = np.concatenate((longitude[split_idx:], longitude[:split_idx] + 360))
nus_5years_rot   = np.concatenate((nus_5years_full[:,  split_idx:], nus_5years_full[:,  :split_idx]), axis=1)
nus_15years_rot  = np.concatenate((nus_15years_full[:, split_idx:], nus_15years_full[:, :split_idx]), axis=1)

nu_5years_masked  = np.ma.masked_where(nus_5years_rot  == 0, nus_5years_rot)
nu_15years_masked = np.ma.masked_where(nus_15years_rot == 0, nus_15years_rot)

levels = np.linspace(vmin - 0.001, vmax, 50)
cntr0 = axs[0].contourf(longitude_rot, latitude, nu_5years_masked,  levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
cntr1 = axs[1].contourf(longitude_rot, latitude, nu_15years_masked, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
c = fig.colorbar(cntr1, ax=list(axs), orientation='horizontal', pad=0.09)
c.set_ticks(np.round(np.linspace(vmin, vmax, 11), 2).tolist())
c.set_label('Mean (η)')

axs[0].get_xaxis().set_visible(False)
axs[0].set_ylabel('Latitude (ºN)')
axs[1].set_ylabel('Latitude (ºN)')
axs[1].set_xlabel('Longitude (ºE)')
axs[0].set_ylim((-80, 80))
axs[1].set_ylim((-80, 80))

# %% watch what happens with single time step
ds = xr.open_dataset('./outputs/exp24_TEST_000.nc')

alk_location = np.argwhere(ds.AT_added.isel(time=1).transpose('latitude', 'longitude', 'depth').values > 0)
AT_lat, AT_lon, _ = alk_location[0]

for t_idx in tqdm(range(0, len(ds.time.values))):
    plot_surface2d(latitude, longitude, ds.delCT.isel(time=t_idx, depth=0).values + 0.0001, 0, 0.1, 'viridis', 'delCT at t = ' + str(ds.time.isel(time=t_idx).values))
    # plot_longitude3d(latitude, depth, ds.delAT.isel(time=t_idx)+0.0001, AT_lon, 0, 10, 'viridis', 'delAT at t = ' + str(ds.time.isel(time=t_idx).values))

# %%
