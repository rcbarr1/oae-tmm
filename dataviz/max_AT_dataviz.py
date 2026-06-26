"""
Created on Mon Jun 22 2026

DATAVIZ FOR MAX_AT: Maximum alkalinity calculation

@author: Reese C. Barrett
"""
#%%
from dataviz.dataviz import broadcast_to_dataset, get_co2_scenario
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

# load model architecture
data_path = './data/'
output_path = '/Volumes/LaCie/outputs/max_AT/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

latitude    = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
longitude   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

model_data.close()
rho = 1025  # seawater density [kg m-3]

#%% pull timestepping comparison experiments (generated with --test --exp-id 0-6)
experiment_names = ['max_AT_2026-06-24_long_none',
                    'max_AT_2026-06-24_long_ssp126',
                    'max_AT_2026-06-24_long_ssp245',
                    'max_AT_2026-06-24_long_ssp534']

labels = ['None', 'SSP1-2.6', 'SSP2-4.5', 'SSP5-3.4 OS']
scenarios = ['none', 'ssp126', 'ssp245', 'ssp534_OS']
colors = ['#003f5c', '#83517c', '#d86c6a', '#ffa600']

#%% compute or load cached time series
cache_path = output_path + 'max_AT_cache.nc'

def _load_experiment(experiment_name, label):
    time_dim = f'time_{label}'
    exp_vars = {}
    with xr.open_mfdataset(
            output_path + experiment_name + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        cell_volume_xr = broadcast_to_dataset(cell_volume, ds)

        AT_added = ds['AT_added'] * cell_volume_xr * rho * 1e-6
        exp_vars[f'AT_added_cum_{label}'] = (
            AT_added.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
                    .cumsum(dim='time')
                    .rename({'time': time_dim})
        )

        exp_vars[f'delxCO2_{label}'] = ds['delxCO2'].rename({'time': time_dim})

        delCT = ds['delCT'] * cell_volume_xr * rho * 1e-6
        exp_vars[f'delCT_{label}'] = delCT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True).rename({'time': time_dim})

        delAT = ds['delAT'] * cell_volume_xr * rho * 1e-6
        exp_vars[f'delAT_{label}'] = delAT.sum(dim=['latitude', 'longitude', 'depth'], skipna=True).rename({'time': time_dim})
    return exp_vars

if os.path.exists(cache_path):
    cache = xr.open_dataset(cache_path).load()
    missing = [(exp, lbl) for exp, lbl in zip(experiment_names, labels)
               if f'AT_added_cum_{lbl}' not in cache]
    if missing:
        new_vars = {}
        for exp_name, label in tqdm(missing, desc='Computing missing experiments'):
            new_vars.update(_load_experiment(exp_name, label))
        with TqdmCallback(desc='Computing missing cache entries'):
            cache = xr.merge([cache, xr.Dataset(new_vars).compute()])
        cache.to_netcdf(cache_path)
else:
    cache_vars = {}
    for exp_name, label in tqdm(zip(experiment_names, labels), total=len(experiment_names), desc='Loading experiments'):
        cache_vars.update(_load_experiment(exp_name, label))
    with TqdmCallback(desc='Computing cache'):
        cache = xr.Dataset(cache_vars).compute()
    cache.to_netcdf(cache_path)

#%% plot AT added over time for timestepping comparison
fig = plt.figure(figsize=(5, 5), dpi=200)
ax = fig.gca()

for label in labels:
    var = cache[f'AT_added_cum_{label}']
    ax.plot(var[f'time_{label}'].values, var.values, label=label)

plt.xlabel('Year')
plt.ylabel(r'Cumulative $A_{\mathbf{T}}$ added to mixed layer (mol)')
plt.legend()

#%% plot change in atmospheric CO2 over time for timestepping comparison
fig = plt.figure(figsize=(5, 5), dpi=200)
ax = fig.gca()

for label in labels:
    var = cache[f'delxCO2_{label}']
    ax.plot(var[f'time_{label}'].values, var.values, label=label)

plt.xlabel('Year')
plt.ylabel('Change in atmospheric CO$_{2}$ (ppm)')
plt.legend()

#%% plot change in CT over time for timestepping comparison
fig = plt.figure(figsize=(5, 5), dpi=200)
ax = fig.gca()

for label in labels:
    var = cache[f'delCT_{label}']
    ax.plot(var[f'time_{label}'].values, var.values, label=label)

plt.xlabel('Year')
plt.ylabel(r'Change in $C_{\mathbf{T}}$ (mol)')
plt.legend()

#%% plot change in AT over time for timestepping comparison
fig = plt.figure(figsize=(5, 5), dpi=200)
ax = fig.gca()

for label in labels:
    var = cache[f'delAT_{label}']
    ax.plot(var[f'time_{label}'].values, var.values, label=label)

plt.xlabel('Year')
plt.ylabel(r'Change in $A_{\mathbf{T}}$ (mol)')
plt.legend()

# %% four-panel full-ocean totals figure for paper
# a. cumulative AT added over time
# b. change in atmospheric CO2 over time
# c. oae efficiency over time (delCT / delAT)
fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=200)

pre_time = np.arange(2020, 2030, 1)
pre_zeros = np.zeros(len(pre_time))

for label, color in zip(labels, colors):
    time = cache[f'AT_added_cum_{label}'][f'time_{label}'].values
    axes[0][0].plot(np.concatenate([pre_time, time]),
                    np.concatenate([pre_zeros, cache[f'AT_added_cum_{label}'].values]), 
                    label=label, c=color)
axes[0][0].set_xlabel('Year')
axes[0][0].set_ylabel(r'Cumulative $A_{\mathbf{T}}$ added (mol)')
axes[0][0].set_xlim([2020, 2100])
axes[0][0].legend()

for label, color in zip(labels, colors):
    time = cache[f'delCT_{label}'][f'time_{label}'].values
    delAT_vals = cache[f'delAT_{label}'].values
    with np.errstate(invalid='ignore', divide='ignore'):
        efficiency = np.where(delAT_vals != 0, cache[f'delCT_{label}'].values / delAT_vals * 100, np.nan)
    axes[0][1].plot(time, efficiency, label=label, c=color)
axes[0][1].set_xlabel('Year')
axes[0][1].set_ylabel(r'OAE efficiency: ${C_{\mathbf{T}}}^{\prime}$ / ${A_{\mathbf{T}}}^{\prime}$ (%)')
axes[0][1].set_xlim([2020, 2100])

for label, color in zip(labels, colors):
    time = cache[f'delxCO2_{label}'][f'time_{label}'].values
    axes[1][0].plot(np.concatenate([pre_time, time]),
                    np.concatenate([pre_zeros, cache[f'delxCO2_{label}'].values]),
                    label=label, c=color)
axes[1][0].set_xlabel('Year')
axes[1][0].set_ylabel(r'Change in atmospheric CO$_2$ (ppm)')
axes[1][0].set_xlim([2020, 2100])

for label, scenario, color in zip(labels, scenarios, colors):
    time = cache[f'delxCO2_{label}'][f'time_{label}'].values
    time_extended = np.concatenate([np.arange(2020, 2030, 1), time])
    atmospheric_co2 = get_co2_scenario(scenario, time_extended)
    axes[1][1].plot(time, cache[f'delxCO2_{label}'].values + atmospheric_co2[10:], label=label, c=color)
    axes[1][1].plot(time_extended, atmospheric_co2, label=label, ls=':', c=color)
axes[1][1].set_xlabel('Year')
axes[1][1].set_ylabel(r'Atmospheric CO$_2$ (ppm)')
axes[1][1].set_xlim([2020, 2100])

plt.tight_layout()

# %% pH changes visualization figure 


