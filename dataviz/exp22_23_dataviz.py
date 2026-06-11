#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:43:49 2025

DATA VIZ FOR EXP22 / EXP23: Maximum alkalinity calculation
- exp22: uses direct pyTRACE calculation for Canth
- exp23: uses interpolated TRACE for Canth

@author: Reese C. Barrett
"""
#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oae_tmm.grid import flatten, make_3d, get_depth_idx
from oae_tmm.trace import calculate_canth, interp_trace
from dataviz.dataviz import get_co2_scenario, plot_surface2d, plot_surface3d, make_surf_animation, broadcast_to_dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.colors import Normalize
import PyCO2SYS as pyco2
from tqdm import tqdm

# load model architecture
data_path = './data/'
# output_path = './outputs/'
output_path = '/Volumes/LaCie/outputs/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

model_lat   = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
model_lon   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
model_depth = model_data['tz'].isel(longitude=0, latitude=0).to_numpy()   # m below sea surface
model_vols  = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

rho      = 1025 # seawater density for volume to mass [kg m-3]
surf_idx = get_depth_idx(ocnmask, 0) # indices of surface grid cells in flattened array

model_data.close()

new_layer_idx = np.cumsum([int(np.nansum(ocnmask[:, :, i])) for i in range(len(model_depth))])

#%% set experiments we are interested in plotting

# --- exp22/exp23 comparison (uncomment to use) ---
# experiment_names = ['exp22_2026-02-06_t0_none',
#                     'exp22_2026-02-06_t0_ssp126',
#                     'exp22_2026-02-06_t0_ssp245',
#                     'exp22_2026-02-06_t0_ssp534_OS',
#                     'exp22_2026-02-06_t1_none',
#                     'exp22_2026-02-06_t1_ssp126',
#                     'exp22_2026-02-06_t1_ssp245',
#                     'exp22_2026-02-06_t1_ssp534_OS',]
#
# experiment_names = ['exp23_2026-02-06_t0_none',
#                     'exp23_2026-02-06_t0_ssp126',
#                     'exp23_2026-02-06_t0_ssp245',
#                     'exp23_2026-02-06_t0_ssp534_OS',
#                     'exp23_2026-02-06_t1_none',
#                     'exp23_2026-02-06_t1_ssp126',
#                     'exp23_2026-02-06_t1_ssp245',
#                     'exp23_2026-02-06_t1_ssp534_OS',]
#
# scenarios = ['none', 'ssp126', 'ssp245', 'ssp534_OS',
#              'none', 'ssp126', 'ssp245', 'ssp534_OS',
#              'none', 'ssp126', 'ssp245', 'ssp534_OS',]
#
# labels = ['dt=1yr, no_ssp', 'dt=1yr, ssp126', 'dt=1yr, ssp245', 'dt=1yr, ssp534_OS',
#           'dt=1mon, no_ssp', 'dt=1mon, ssp126', 'dt=1mon, ssp245', 'dt=1mon, ssp534_OS',
#           'dt=1day, no_ssp', 'dt=1day, ssp126', 'dt=1day, ssp245', 'dt=1day, ssp534_OS']
#
# linestyles = ['-','-','-','-','--','--','--','--',':',':',':',':']
# linecolors = ['#023880', '#96adcf', '#145a6a', '#2eceb7',
#               '#023880', '#96adcf', '#145a6a', '#2eceb7',
#               '#023880', '#96adcf', '#145a6a', '#2eceb7',]
# ncol = 2
# start_year = 2015
# textcolor = '#595959'

# --- TRACE interpolation comparison (uncomment to use) ---
# experiment_names = ['exp23_2026-03-27_NEWTRACE_ssp534_OS',
#                     'exp23_2026-03-27_OLDTRACE_ssp534_OS',
#                     'exp22_2026-03-27_CALC_ssp534_OS',]
#
# scenarios = ['ssp534_OS', 'ssp534_OS', 'ssp534_OS']
#
# linestyles = ['-', '-', '-', '-']
# linecolors = ['#00429d', '#bf89b6', '#ffb0de']
# labels = ['High res gridded', 'Low res gridded', 'Calculated Canth']
# ncol = 1
# start_year = 2020
# mpl.rcParams['font.family'] = 'Calibri'
# textcolor = '#595959'
# mpl.rcParams['text.color'] = textcolor
# mpl.rcParams['axes.labelcolor'] = textcolor
# mpl.rcParams['xtick.color'] = textcolor
# mpl.rcParams['ytick.color'] = textcolor
# mpl.rcParams['font.weight'] = 'bold'

# --- OSM talk (active) ---
experiment_names = ['exp22_2026-02-11_t1_none_ML',
                    'exp22_2026-02-11_t1_ssp126_ML',
                    'exp22_2026-02-11_t1_ssp245_ML',
                    'exp22_2026-02-11_t1_ssp534_OS_ML',]

scenarios = ['none',
             'ssp126',
             'ssp245',
             'ssp534_OS']

linestyles = ['-', '-', '-', '-']
linecolors = ['#00429d', '#7a64a8', '#bf89b6', '#ffb0de']
labels = ['No emissions', 'SSP1-2.6', 'SSP2-4.5', 'SSP5-3.4OS']
ncol = 1
start_year = 2030
mpl.rcParams['font.family'] = 'Calibri'
textcolor = '#595959'
mpl.rcParams['text.color'] = textcolor
mpl.rcParams['axes.labelcolor'] = textcolor
mpl.rcParams['xtick.color'] = textcolor
mpl.rcParams['ytick.color'] = textcolor
mpl.rcParams['font.weight'] = 'bold'

#%% pull in preindustrial baselines

# get GLODAP data
DIC_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/DIC.npy') # dissolved inorganic carbon [µmol kg-1]
AT_3D  = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')  # total alkalinity [µmol kg-1]
T_3D   = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy') # temperature [ºC]
S_3D   = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy')    # salinity [unitless]
Si_3D  = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy')    # silicate [µmol kg-1]
P_3D   = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy')         # phosphate [µmol kg-1]

S  = flatten(S_3D,  ocnmask)
T  = flatten(T_3D,  ocnmask)
Si = flatten(Si_3D, ocnmask)
P  = flatten(P_3D,  ocnmask)

# get TRACE data
Canth_2002_3D = calculate_canth('none', 2002, T_3D, S_3D, ocnmask, model_lat, model_lon, model_depth)
if start_year != 2002:
    Canth_3D = calculate_canth('none', start_year, T_3D, S_3D, ocnmask, model_lat, model_lon, model_depth)
else:
    Canth_3D = Canth_2002_3D

# calculate preindustrial DIC by subtracting anthropogenic carbon
DIC_preind_3D = DIC_3D - Canth_2002_3D
DIC_preind    = flatten(DIC_preind_3D, ocnmask)

DIC_start_3D = DIC_preind_3D + Canth_3D

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[0], ocnmask.shape[1])).transpose([1, 2, 0])
pressure    = flatten(pressure_3D, ocnmask)

# calculate preindustrial pH assuming steady state alkalinity
co2sys = pyco2.sys(dic=DIC_preind,
                   alkalinity=flatten(AT_3D, ocnmask),
                   salinity=flatten(S_3D, ocnmask),
                   temperature=flatten(T_3D, ocnmask),
                   pressure=flatten(pressure_3D, ocnmask),
                   total_silicate=flatten(Si_3D, ocnmask),
                   total_phosphate=flatten(P_3D, ocnmask))

pH_preind    = co2sys['pH']
pH_preind_3D = make_3d(pH_preind, ocnmask)

#%% calculate anthropogenic carbon at each time step
Canth_all_scenarios = []
interp = 0

# open dataset with relevant time fields
ds = xr.open_mfdataset(
        output_path + experiment_names[0] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)

t = ds.time.values

# with interpolation
if interp:
    for scenario in scenarios:
        Canth_all_idx = []
        for idx in tqdm(range(len(t))):
            if scenario != 'none':
                Canth_idx_3D = interp_trace(data_path, t[idx], scenario, model_lat, model_lon, model_depth, ocnmask)
            else:
                Canth_idx_3D = Canth_3D
            Canth_all_idx.append(flatten(Canth_idx_3D, ocnmask))
        Canth_all_scenarios.append(Canth_all_idx)

# without interpolation
else:
    for scenario in scenarios:
        Canth_all_idx = []
        for idx in tqdm(range(len(t))):
            if scenario != 'none':
                Canth_idx_3D = calculate_canth(scenario, t[idx], T_3D, S_3D, ocnmask, model_lat, model_lon, model_depth)
            else:
                Canth_idx_3D = Canth_3D
            Canth_all_idx.append(flatten(Canth_idx_3D, ocnmask))
        Canth_all_scenarios.append(Canth_all_idx)

np.save(output_path + 'Canth_all_scenarios_calculated_2030-2080.npy', Canth_all_scenarios)

#%% cumulative AT added
fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
ax = fig.gca()

for exp_idx in range(len(experiment_names)):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        model_vols_xr = broadcast_to_dataset(model_vols, ds)
        AT_added = ds['AT_added'] * model_vols_xr * rho * 1e-6
        AT_added = AT_added.sum(dim=['lat', 'lon', 'depth'], skipna=True)
        AT_added_cum = AT_added.cumsum(dim='time')

        ax.plot(ds['time'].values, AT_added_cum.compute().values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

# plt.legend(loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, -0.39))
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel(r'Cumulative $A_{\mathbf{T}}$ added to mixed layer (mol)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)
#plt.ylim([0, 6.5e16])

#%% normal AT added
fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
ax = fig.gca()

for exp_idx in range(len(experiment_names)):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        model_vols_xr = broadcast_to_dataset(model_vols, ds)
        AT_added = ds['AT_added'] * model_vols_xr * rho * 1e-6
        AT_added = AT_added.sum(dim=['lat', 'lon', 'depth'], skipna=True)

        ax.plot(ds['time'].values, AT_added.compute().values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

plt.legend(loc='upper right', ncol=ncol, fontsize=11.5)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel(r'Instantaneous $A_{\mathbf{T}}$ added to mixed layer (mol)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% surface AT added
for exp_idx in range(1):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        for idx in np.arange(0, len(ds.time), 12):
            DIC = flatten(ds['delDIC'].isel(time=idx).values, ocnmask) + flatten(DIC_start_3D, ocnmask)
            AT  = flatten(ds['delAT'].isel(time=idx).values,  ocnmask) + flatten(AT_3D, ocnmask)
            co2sys = pyco2.sys(
                    alkalinity=AT,
                    dic=DIC,
                    salinity=S,
                    temperature=T,
                    pressure=pressure,
                    total_silicate=Si,
                    total_phosphate=P)

            co2sys_start = pyco2.sys(
                    alkalinity=AT,
                    dic=flatten(DIC_start_3D, ocnmask),
                    salinity=S,
                    temperature=T,
                    pressure=pressure,
                    total_silicate=Si,
                    total_phosphate=P)

            delpH = make_3d(co2sys['pH'] - co2sys_start['pH'], ocnmask)
            plot_surface3d(model_lat, model_lon, delpH, 0, -0.2, 0.2, 'RdBu', 'delpH in ' + str(ds['time'].isel(time=idx).values))

#%% change in atmospheric CO2
fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
ax = fig.gca()

for exp_idx in range(len(experiment_names)):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        ax.plot(ds['time'].values, ds['delxCO2'].values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

# plt.legend(loc='upper right', ncol=ncol)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Change in atmospheric CO$_{2}$ (ppm)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)
#plt.ylim([-90, 0])

#%% total atmospheric CO2 over time
fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
ax = fig.gca()

for exp_idx in range(len(experiment_names)):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        atmospheric_CO2 = get_co2_scenario(scenarios[exp_idx], ds['time'].values)

        ax.plot(ds['time'].values, ds['delxCO2'].values + atmospheric_CO2, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])
        ax.plot(ds['time'].values, atmospheric_CO2, label=labels[exp_idx], c=linecolors[exp_idx], ls=':')

#plt.legend(loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, -0.39))
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Atmospheric CO$_{2}$ (ppm)', fontsize=11.5, weight='bold')
#plt.title('atmospheric CO2 with maximum OAE')
#plt.ylim([310, 560])
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% change in DIC (surface)
fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
ax = fig.gca()

ax.axhline(float(np.average(DIC_preind[surf_idx], weights=flatten(model_vols, ocnmask)[surf_idx])), c='black', linestyle='--', label='Preindustrial DIC')

for exp_idx in tqdm(range(len(experiment_names))):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        DIC_modeled_3D = ds['delDIC'] + broadcast_to_dataset(DIC_start_3D, ds)
        DIC_weighted_mean = DIC_modeled_3D.isel(depth=0).weighted(
            broadcast_to_dataset(model_vols, ds).isel(depth=0)
        ).mean(dim=['lat', 'lon'], skipna=True)

        ax.plot(ds['time'].values, DIC_weighted_mean.values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

#plt.legend(loc='lower right', ncol=ncol)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Average surface ocean DIC (µmol kg$^{-1}$)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% change in DIC (full ocean)
fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
ax = fig.gca()

ax.axhline(float(np.average(DIC_preind, weights=flatten(model_vols, ocnmask))), c='black', linestyle='--', label='Preindustrial DIC')

for exp_idx in tqdm(range(len(experiment_names))):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        DIC_modeled_3D = ds['delDIC'] + broadcast_to_dataset(DIC_start_3D, ds)
        DIC_weighted_mean = DIC_modeled_3D.weighted(
            broadcast_to_dataset(model_vols, ds)
        ).mean(dim=['lat', 'lon', 'depth'])

        ax.plot(ds['time'].values, DIC_weighted_mean.values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

ax.legend(bbox_to_anchor=(1, 0.05), loc='lower right', ncol=ncol)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Average ocean DIC (µmol kg$^{-1}$)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% change in pH (surface)
Canth_all_scenarios = np.load(output_path + 'Canth_all_scenarios_calculated_2030-2080.npy')

fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
ax = fig.gca()

AT_flat         = flatten(AT_3D, ocnmask)
DIC_preind_flat = flatten(DIC_preind_3D, ocnmask)
surf_weights    = flatten(model_vols, ocnmask)[surf_idx]

ax.axhline(float(np.average(pH_preind[surf_idx], weights=flatten(model_vols, ocnmask)[surf_idx])), c='black', linestyle='--', label='Preindustrial pH')

for exp_idx in tqdm(range(len(experiment_names))):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        n_t            = len(ds['time'])
        Canth_for_exp  = Canth_all_scenarios[exp_idx]  # shape (n_t, n_flat)

        AT_surf  = np.stack([flatten(ds['delAT'].isel(time=idx).values,  ocnmask)[surf_idx] for idx in range(n_t)]) + AT_flat[surf_idx]
        DIC_surf = np.stack([flatten(ds['delDIC'].isel(time=idx).values, ocnmask)[surf_idx] for idx in range(n_t)]) + DIC_preind_flat[surf_idx] + Canth_for_exp[:, surf_idx]

        co2sys = pyco2.sys(
            alkalinity=AT_surf,
            dic=DIC_surf,
            salinity=S[surf_idx],
            temperature=T[surf_idx],
            pressure=pressure[surf_idx],
            total_silicate=Si[surf_idx],
            total_phosphate=P[surf_idx])

        avg_pH_modeled_surf = (co2sys['pH'] * surf_weights).sum(axis=1) / surf_weights.sum()

        ax.plot(ds['time'].values, avg_pH_modeled_surf, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Average surface ocean pH', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% change in pH (full ocean)
fig = plt.figure(figsize=(3.5, 3.5), dpi=200)
ax = fig.gca()

AT_flat         = flatten(AT_3D, ocnmask)
DIC_preind_flat = flatten(DIC_preind_3D, ocnmask)
all_weights     = flatten(model_vols, ocnmask)

ax.axhline(float(np.average(pH_preind, weights=flatten(model_vols, ocnmask))), c='black', linestyle='--', label='Preindustrial pH')

for exp_idx in tqdm(range(len(experiment_names))):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        n_t           = len(ds['time'])
        Canth_for_exp = Canth_all_scenarios[exp_idx]  # shape (n_t, n_flat)

        AT_all  = np.stack([flatten(ds['delAT'].isel(time=idx).values,  ocnmask) for idx in range(n_t)]) + AT_flat
        DIC_all = np.stack([flatten(ds['delDIC'].isel(time=idx).values, ocnmask) for idx in range(n_t)]) + DIC_preind_flat + Canth_for_exp

        co2sys = pyco2.sys(
            alkalinity=AT_all,
            dic=DIC_all,
            salinity=S,
            temperature=T,
            pressure=pressure,
            total_silicate=Si,
            total_phosphate=P)

        avg_pH_modeled = (co2sys['pH'] * all_weights).sum(axis=1) / all_weights.sum()

        ax.plot(ds['time'].values, avg_pH_modeled, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

plt.legend(loc='lower right', ncol=ncol)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Average ocean pH', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% change in revelle factors
# plot RF in 2030 and 2080 for all four scenarios at surface and in ocean basins:
# Atlantic (25.5°W), Pacific (150.5ºW), and Indian (90.5°E)

Canth_all_scenarios = np.load(output_path + 'Canth_all_scenarios_calculated_2030-2080.npy')

# num experiments x num time steps of interest x length of flattened ocnmask
t_idxs = [0, -1]
revelle_factors = np.zeros((len(experiment_names), len(t_idxs), len(S)))

for exp_idx in tqdm(range(len(experiment_names))):
    with xr.open_mfdataset(
            output_path + experiment_names[exp_idx] + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        AT_modeled_3D = ds['delAT'] + broadcast_to_dataset(AT_3D, ds)

        for t_idx in t_idxs:
            Canth_3d      = make_3d(Canth_all_scenarios[exp_idx, t_idx, :], ocnmask)
            DIC_modeled_3D = ds['delDIC'].isel(time=t_idx) + broadcast_to_dataset(DIC_preind_3D, ds) + broadcast_to_dataset(Canth_3d, ds)

            AT_modeled  = flatten(AT_modeled_3D.isel(time=t_idx).values, ocnmask)
            DIC_modeled = flatten(DIC_modeled_3D.values, ocnmask)

            co2sys = pyco2.sys(
                alkalinity=AT_modeled,
                dic=DIC_modeled,
                salinity=S,
                temperature=T,
                pressure=pressure,
                total_silicate=Si,
                total_phosphate=P)

            revelle_factors[exp_idx, t_idx, :] = co2sys['revelle_factor']

#%% make surface plots: three columns (2030, 2080, difference) and four rows (one for each)
fig, axes = plt.subplots(4, 3, figsize=(10, 12), dpi=200)

rf_norm      = Normalize(vmin=8,  vmax=20)
rf_diff_norm = Normalize(vmin=-5, vmax=5)

# plot 2030
for exp_idx in range(len(experiment_names)):
    ax = axes[exp_idx, 0]
    RF_to_plot = make_3d(revelle_factors[exp_idx, 0, :], ocnmask)[:, :, 0]
    ax.contourf(model_lon, model_lat, RF_to_plot, cmap='viridis', norm=rf_norm)
    if exp_idx == 0:
        ax.set_title('2030', fontsize=12)
    ax.set_ylabel(scenarios[exp_idx], fontsize=10)

# plot 2080
for exp_idx in range(len(experiment_names)):
    ax = axes[exp_idx, 1]
    RF_to_plot = make_3d(revelle_factors[exp_idx, 1, :], ocnmask)[:, :, 0]
    ax.contourf(model_lon, model_lat, RF_to_plot, cmap='viridis', norm=rf_norm)
    if exp_idx == 0:
        ax.set_title('2080', fontsize=12)

# plot difference
for exp_idx in range(len(experiment_names)):
    ax = axes[exp_idx, 2]
    RF_to_plot = make_3d(revelle_factors[exp_idx, 1, :], ocnmask)[:, :, 0] - make_3d(revelle_factors[exp_idx, 0, :], ocnmask)[:, :, 0]
    ax.contourf(model_lon, model_lat, RF_to_plot, cmap='seismic', norm=rf_diff_norm)
    if exp_idx == 0:
        ax.set_title('difference', fontsize=12)

fig.colorbar(plt.cm.ScalarMappable(norm=rf_norm,      cmap='viridis'), ax=axes[:, :2], fraction=0.03, pad=0.02, label='Revelle factor')
fig.colorbar(plt.cm.ScalarMappable(norm=rf_diff_norm, cmap='seismic'), ax=axes[:, 2],  fraction=0.03, pad=0.02, label='Difference')

#%% make transect plots: three columns (2030, 2080, difference) and four rows (one for each)
fig, axes = plt.subplots(4, 3, figsize=(12, 12), dpi=200)

# model_lon[105] = 149 ºW (Pacific)
# model_lon[167] = 25 ºW (Atlantic)
# model_lon[45]  = 91 ºE (Indian)
lon_idx = 105

rf_norm      = Normalize(vmin=8,  vmax=20)
rf_diff_norm = Normalize(vmin=-5, vmax=5)

# plot 2030
for exp_idx in range(len(experiment_names)):
    ax = axes[exp_idx, 0]
    RF_to_plot = make_3d(revelle_factors[exp_idx, 0, :], ocnmask)[:, lon_idx, :]
    ax.contourf(model_lat, model_depth, RF_to_plot.T, cmap='viridis', norm=rf_norm)
    if exp_idx == 0:
        ax.set_title('2030', fontsize=12)
    ax.set_ylabel(scenarios[exp_idx], fontsize=10)
    ax.invert_yaxis()

# plot 2080
for exp_idx in range(len(experiment_names)):
    ax = axes[exp_idx, 1]
    RF_to_plot = make_3d(revelle_factors[exp_idx, 1, :], ocnmask)[:, lon_idx, :]
    ax.contourf(model_lat, model_depth, RF_to_plot.T, cmap='viridis', norm=rf_norm)
    if exp_idx == 0:
        ax.set_title('2080', fontsize=12)
    ax.invert_yaxis()

# plot difference
for exp_idx in range(len(experiment_names)):
    ax = axes[exp_idx, 2]
    RF_to_plot = make_3d(revelle_factors[exp_idx, 1, :], ocnmask)[:, lon_idx, :] - make_3d(revelle_factors[exp_idx, 0, :], ocnmask)[:, lon_idx, :]
    ax.contourf(model_lat, model_depth, RF_to_plot.T, cmap='seismic', norm=rf_diff_norm)
    if exp_idx == 0:
        ax.set_title('difference', fontsize=12)
    ax.invert_yaxis()

fig.colorbar(plt.cm.ScalarMappable(norm=rf_norm,      cmap='viridis'), ax=axes[:, :2], fraction=0.03, pad=0.02, label='Revelle factor')
fig.colorbar(plt.cm.ScalarMappable(norm=rf_diff_norm, cmap='seismic'), ax=axes[:, 2],  fraction=0.03, pad=0.02, label='Difference')

#%% line plot of pressure by index
vmin = -50
vmax = 6000
plt.plot(flatten(pressure_3D, ocnmask), c='gray')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('pressure (dbar)')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plot of salinity by index
vmin = 14
vmax = 41
plt.plot(flatten(S_3D, ocnmask), c='skyblue')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('salinity')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plot of temperature by index
vmin = -5
vmax = 35
plt.plot(flatten(T_3D, ocnmask), c='salmon')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('temperature (ºC)')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plot of silicate by index
vmin = -10
vmax = 300
plt.plot(flatten(Si_3D, ocnmask), c='plum')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('silicate (µmol kg-1)')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plot of phosphate by index
vmin = -0.5
vmax = 3.6
plt.plot(flatten(P_3D, ocnmask), c='mediumaquamarine')
plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
plt.title('phosphate (µmol kg-1)')
plt.xlim([-1000, np.sum(ocnmask)+1000])
plt.ylim([vmin, vmax])
plt.show()

#%% line plots of DIC by index
vmin = -500
vmax = 2500
for t_idx in range(0, len(ds.time)):
    plt.plot(flatten(ds.isel(time=t_idx).delDIC.values, ocnmask) + flatten(DIC_3D, ocnmask), c='steelblue')
    plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
    plt.title('DIC (µmol kg-1) at t = ' + str(t_idx))
    plt.xlim([-1000, np.sum(ocnmask)+1000])
    plt.ylim([vmin, vmax])
    plt.show()

#%% line plots of AT by index
vmin = -6000
vmax = 4000
for t_idx in range(0, len(ds.time)):
    plt.plot(flatten(ds.isel(time=t_idx).delAT.values, ocnmask) + flatten(AT_3D, ocnmask), c='goldenrod')
    plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
    plt.title('AT (µmol kg-1) at t = ' + str(t_idx))
    plt.xlim([-1000, np.sum(ocnmask)+1000])
    plt.ylim([vmin, vmax])
    plt.show()

#%% line plots of pH by index
vmin = 2
vmax = 10
n_t = len(ds.time)
AT_all  = np.stack([flatten(ds.isel(time=idx).delAT.values,  ocnmask) for idx in range(n_t)]) + flatten(AT_3D,  ocnmask)
DIC_all = np.stack([flatten(ds.isel(time=idx).delDIC.values, ocnmask) for idx in range(n_t)]) + flatten(DIC_3D, ocnmask)
co2sys = pyco2.sys(
    alkalinity=AT_all,
    dic=DIC_all,
    salinity=S,
    temperature=T,
    pressure=pressure,
    total_silicate=Si,
    total_phosphate=P)

for t_idx in range(n_t):
    plt.plot(co2sys['pH'][t_idx], c='lightpink')
    plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
    plt.title('pH at t = ' + str(t_idx))
    plt.xlim([-1000, np.sum(ocnmask)+1000])
    plt.ylim([vmin, vmax])
    plt.show()

#%% plot surface pH
co2sys_surf = pyco2.sys(
    alkalinity=flatten(ds.isel(time=1).delAT.values,  ocnmask) + flatten(AT_3D,  ocnmask),
    dic=flatten(ds.isel(time=1).delDIC.values, ocnmask) + flatten(DIC_3D, ocnmask),
    salinity=S,
    temperature=T,
    pressure=pressure,
    total_silicate=Si,
    total_phosphate=P)

pH_plot    = co2sys_surf['pH']
pH_plot_3D = make_3d(pH_plot, ocnmask)
plot_surface2d(model_lat, model_lon, pH_plot_3D[:, :, 0], 2, 10, 'viridis', 'pH (t = 15, depth = ' + str(np.round(model_depth[4], 2)) + ' m)')
plt.show()

#%% line plots of AT added by index
vmin = -50
vmax = 250
for t_idx in range(0, len(ds.time)):
    plt.plot(flatten(ds.isel(time=t_idx).AT_added.values, ocnmask), c='darkgoldenrod')
    plt.vlines(new_layer_idx, vmin, vmax, colors='gainsboro', ls=':')
    plt.title('AT added (µmol kg-1) at t = ' + str(t_idx))
    plt.xlim([-1000, np.sum(ocnmask)+1000])
    plt.ylim([vmin, vmax])
    plt.show()

#%% animations of tracers
# AT added
ds = xr.open_dataset('/Volumes/LaCie/outputs/exp22_2026-02-06_t1_none_000.nc')
make_surf_animation(ds.AT_added, 'AT Added (µmol kg-1)', model_lat, model_lon, ds.time.values, len(ds.time.values), 0, 150, 'viridis', './movies/exp22_2026-02-06_t1_none_AT_added.mp4')

# delDIC
ds = xr.open_dataset('/Volumes/LaCie/outputs/exp22_2026-02-06_t1_none_000.nc')
make_surf_animation(ds.delDIC, 'Change in DIC (µmol kg-1)', model_lat, model_lon, ds.time.values, len(ds.time.values), -250, 250, 'RdBu', './movies/exp22_2026-02-06_t1_none_delDIC.mp4')

#%% animation of deviation from preindustrial pH
ds = xr.open_dataset('/Volumes/LaCie/outputs/exp22_2026-02-11_t1_none_ML_000.nc')
ds = ds.isel(time=slice(0,10))

n_t            = len(ds['time'])
AT_flat        = flatten(AT_3D, ocnmask)
DIC_start_flat = flatten(DIC_start_3D, ocnmask)
vmin, vmax     = -0.2, 0.2
levels         = np.linspace(vmin, vmax, 100)

def get_pH_deviation(idx):
    result = pyco2.sys(
        alkalinity=flatten(ds['delAT'].isel(time=idx).values, ocnmask) + AT_flat,
        dic=flatten(ds['delDIC'].isel(time=idx).values, ocnmask) + DIC_start_flat,
        salinity=S, temperature=T, pressure=pressure,
        total_silicate=Si, total_phosphate=P)
    return make_3d(result['pH'] - pH_preind, ocnmask)[:, :, 0]

fig, ax = plt.subplots(figsize=(10, 7))
cntr = ax.contourf(model_lon, model_lat, get_pH_deviation(0), levels=levels, cmap='RdBu', vmin=vmin, vmax=vmax)
fig.colorbar(cntr, ax=ax, label='Deviation of pH from preindustrial')
ax.set_xlabel('Longitude (ºE)')
ax.set_ylabel('Latitude (ºN)')
ax.set_title(f't = {ds.time.values[0]:.3f} yr')

def update_frame(idx):
    ax.clear()
    ax.contourf(model_lon, model_lat, get_pH_deviation(idx), levels=levels, cmap='RdBu', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Longitude (ºE)')
    ax.set_ylabel('Latitude (ºN)')
    ax.set_title(f't = {ds.time.values[idx]:.3f} yr')
    return []

ani = animation.FuncAnimation(fig, update_frame, frames=n_t, interval=100, blit=False)
ani.save('./movies/exp22_2026-02-11_t1_none_ML_pH_deviation_preind.mp4', writer=animation.FFMpegWriter(fps=10), dpi=200)

#%% 