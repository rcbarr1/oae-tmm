#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 2026

DATA VIZ FOR LCA1: Adding AT to four zones to get efficiency

@author: Reese C. Barrett
"""
#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oae_tmm.loaders import load_mat
from oae_tmm.grid import flatten, make_3d, get_depth_idx
from oae_tmm.trace import calculate_canth, interp_trace
from dataviz.dataviz import get_co2_scenario, plot_surface2d, plot_surface3d, make_surf_animation, make_surf_animation_pH
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import PyCO2SYS as pyco2
from tqdm import tqdm

# load model architecture
data_path = './data/'
output_path = './outputs/'
# output_path = '/Volumes/LaCie/outputs/'

# load transport matrix (OCIM2-48L, from Holzer et al., 2021)
# transport matrix is referred to as "A" vector in John et al., 2020 (AWESOME OCIM)
TR = load_mat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
TR = TR['TR']

# open up rest of data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

model_lat = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
model_lon = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
model_depth = model_data['tz'].isel(longitude=0, latitude=0).to_numpy() # m below sea surface
model_vols = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

# some other important numbers
grid_cell_depth = model_data['wz'].transpose('latitude', 'longitude', 'depth').to_numpy() # depth of model layers (need bottom of grid cell, not middle) [m]
z1 = grid_cell_depth[0, 0, 1] # depth of first model layer [m]
rho = 1025 # seawater density for volume to mass [kg m-3]
surf_idx = get_depth_idx(ocnmask,0) # indicies of surface grid cells in 3D array flattened by flatten()

model_data.close()

# rules for saving files
t_per_file = 2000 # number of time steps 

# calculate when new layers start (for line plots)
new_layer_idx = np.zeros(len(model_depth))
for i in range(len(model_depth)):
    new_layer_idx[i] = int(np.nansum(ocnmask[i,:,:]))
new_layer_idx = np.cumsum(new_layer_idx)

#%% set experiments we are interested in plotting

experiment_names = ['LCA1_2026-03-12_1ton_00000',
                    'LCA1_2026-03-12_5ton_00000',
                    'LCA1_2026-03-12_1ton_00001',
                    'LCA1_2026-03-12_5ton_00001',
                    'LCA1_2026-03-12_1ton_00002',
                    'LCA1_2026-03-12_5ton_00002',
                    'LCA1_2026-03-12_1ton_00003',
                    'LCA1_2026-03-12_5ton_00003']

scenarios = ['REMIND']

labels = ['Nearshore Col. 1 ton',
          'Nearshore Col. 5 ton',
          'Offshore Col. 1 ton',
          'Offshore Col. 5 ton',
          'Nearshore Nor. 1 ton',
          'Nearshore Nor. 5 ton',
          'Offshore Nor. 1 ton',
          'Offshore Nor. 5 ton',]

experiment_names = ['LCA1_2026-03-12_1ton_00000',
                    'LCA1_2026-03-16_1ton_2050_00000',
                    'LCA1_2026-03-12_1ton_00001',
                    'LCA1_2026-03-16_1ton_2050_00001',
                    'LCA1_2026-03-12_1ton_00002',
                    'LCA1_2026-03-16_1ton_2050_00002',
                    'LCA1_2026-03-12_1ton_00003',
                    'LCA1_2026-03-16_1ton_2050_00003']

labels = ['Nearshore Col. 2026',
          'Nearshore Col. 2050',
          'Offshore Col. 2026',
          'Offshore Col. 2050',
          'Nearshore Nor. 2026',
          'Nearshore Nor. 2050',
          'Offshore Nor. 2026',
          'Offshore Nor. 2050',]

start_years = [2026, 2050, 2026, 2050, 2026, 2050, 2026, 2050]

linestyles = [(0, (5,10)), (0, (1, 1)), (0, (5,10)), (0, (1, 1)),
              (0, (5,10)), (0, (1, 1)), (0, (5,10)), (0, (1, 1))]

linecolors = ['#00429d', '#00429d', '#7a64a8', '#7a64a8',
              '#bf89b6', '#bf89b6', '#ffb0de', '#ffb0de']

ncol = 1
start_year = 2050

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
AT_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/TA.npy')   # total alkalinity [µmol kg-1]
T_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy') # temperature [ºC]
S_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy') # salinity [unitless]
Si_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/silicate.npy') # silicate [µmol kg-1]
P_3D = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/PO4.npy') # phosphate [µmol kg-1]

S = flatten(S_3D, ocnmask)
T = flatten(T_3D, ocnmask)
Si = flatten(Si_3D, ocnmask)
P = flatten(P_3D, ocnmask)

# get TRACE data (switch to interpolation for speed?)
Canth_2002_3D = calculate_canth('REMIND', 2002, T_3D, S_3D, ocnmask, model_lat, model_lon, model_depth)
if start_year != 2002:
    Canth_3D = calculate_canth('REMIND', start_year, T_3D, S_3D, ocnmask, model_lat, model_lon, model_depth)
else:
    Canth_3D = Canth_2002_3D

# calculate preindustrial DIC by subtracting anthropogenic carbon
DIC_preind_3D = DIC_3D - Canth_2002_3D
DIC_preind = flatten(DIC_preind_3D, ocnmask)

DIC_start_3D = DIC_preind_3D + Canth_3D

# create "pressure" array by broadcasting depth array
pressure_3D = np.tile(model_depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[0], ocnmask.shape[1])).transpose([1, 2, 0])
pressure = flatten(pressure_3D, ocnmask) 

# calculate preindustrial pH assuming steady state alkalinity
co2sys = pyco2.sys(dic=DIC_preind,
                   alkalinity=flatten(AT_3D, ocnmask),
                   salinity=flatten(S_3D,ocnmask),
                   temperature=flatten(T_3D,ocnmask),
                   pressure=flatten(pressure_3D,ocnmask),
                   total_silicate=flatten(Si_3D,ocnmask),
                   total_phosphate=flatten(P_3D,ocnmask))

pH_preind = co2sys['pH']
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

np.save(output_path + 'Canth_LCA_start2050.npy', Canth_all_scenarios)
#%% load in Canth calculated
Canth_start2026 = np.load(output_path + 'Canth_LCA_start2026.npy')
Canth_start2050 = np.load(output_path + 'Canth_LCA_start2050.npy')

#%% calculate eta over time for each location
# eta = mol C / mol AT = delDIC / delAT

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # broadcast model_vols to convert ∆AT from per kg to total
    model_vols_xr = xr.DataArray(model_vols, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
    
    delAT_mol = ds['delAT'] * model_vols_xr * rho * 1e-6
    delDIC_mol = ds['delDIC'] * model_vols_xr * rho * 1e-6

    delAT_mol_total = delAT_mol.sum(dim=['lat', 'lon', 'depth'], skipna=True)
    delDIC_mol_total = delDIC_mol.sum(dim=['lat', 'lon', 'depth'], skipna=True)

    eta = delDIC_mol_total / delAT_mol_total

    # only actually pull values into memory needed for plotting
    ax.plot(ds['time'].values - start_years[exp_idx], eta.values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

plt.legend(bbox_to_anchor=(0.97, -0.1), ncol=2)
plt.xlabel('Year Since A$_{T}$ Pulse')
plt.ylabel('η (∆ mol C per ∆ mol A$_{T}$)')

#%% cumulative AT added
fig = plt.figure(figsize=(3.5,3.5), dpi=200)
ax = fig.gca()

# use xarray to open metadata of files of interest
for exp_idx in tqdm(range(len(experiment_names))):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # broadcast model_vols to convert ∆AT from per kg to total
    model_vols_xr = xr.DataArray(model_vols, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
    
    AT_added = ds['AT_added'] * model_vols_xr * rho * 1e-6
    AT_added = AT_added.sum(dim=['lat', 'lon', 'depth'], skipna=True)
    AT_added_cum = AT_added.cumsum(dim='time')
    
    # only actually pull values into memory needed for plotting
    ax.plot(ds['time'].values, AT_added_cum.compute().values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])
    
# plt.legend(loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, -0.39))
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Cumulative $A_{\mathbf{T}}$ added to mixed layer (mol)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)
#plt.ylim([0, 6.5e16])
#%% normal AT added
fig = plt.figure(figsize=(3.5,3.5), dpi=200)
ax = fig.gca()

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # broadcast model_vols to convert ∆AT from per kg to total
    model_vols_xr = xr.DataArray(model_vols, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
    
    AT_added = ds['AT_added'] * model_vols_xr * rho * 1e-6
    AT_added = AT_added.sum(dim=['lat', 'lon', 'depth'], skipna=True)
    
    ax.plot(ds['time'].values, AT_added.compute().values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])
    
plt.legend(loc='upper right', ncol=ncol, fontsize=11.5)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Instantaneous $A_{\mathbf{T}}$ added to mixed layer (mol)', fontsize=11.5, weight='bold') 
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% surface AT added
# use xarray to open metadata of files of interest
# for exp_idx in range(len(experiment_names)):
for exp_idx in range(1):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    for idx in np.arange(0, len(ds.time), 12):
        DIC = flatten(ds['delDIC'].isel(time=idx).values, ocnmask) + flatten(DIC_start_3D, ocnmask) 
        AT = flatten(ds['delAT'].isel(time=idx).values, ocnmask) + flatten(AT_3D, ocnmask)  
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
fig = plt.figure(figsize=(3.5,3.5), dpi=200)
ax = fig.gca()

# use xarray to open metadata of files of interest
#for exp_idx in range(len(experiment_names)):
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # only actually pull values into memory needed for plotting
    ax.plot(ds['time'].values, ds['delxCO2'].values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])
    
plt.legend(loc='upper right', ncol=ncol)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Change in atmospheric CO$_{2}$ (ppm)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)
#plt.ylim([-90, 0])

#%% total atmospheric CO2 over time
fig = plt.figure(figsize=(3.5,3.5), dpi=200)
ax = fig.gca()

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)

    # get background atmospheric CO2 based on relevant CO2 scenario
    atmospheric_CO2 = get_co2_scenario(scenarios[exp_idx], ds['time'].values)

    # only actually pull values into memory needed for plotting
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
fig = plt.figure(figsize=(3.5,3.5), dpi=200)
ax = fig.gca()

# plot preindustrial baseline
ax.axhline(np.average(DIC_preind[surf_idx], weights=flatten(model_vols,ocnmask)[surf_idx]), c='black', linestyle='--', label='Preindustrial DIC')

# store DIC and model_vols in xarray for broadcasting
DIC_start_ds = xr.DataArray(DIC_start_3D, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
model_vols_ds = xr.DataArray(model_vols, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})

# use xarray to open metadata of files of interest
for exp_idx in tqdm(range(len(experiment_names))):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # wrap GLODAP DIC in xarray dataset to convert ∆DIC to total DIC over time
    DIC_modeled_3D = ds['delDIC'] + DIC_start_ds
    
    # wrap model_vols in xarray dataset to convert from concentration to amount or use in weighted average
    DIC_weighted_mean = DIC_modeled_3D.isel(depth=0).weighted(model_vols_xr.isel(depth=0)).mean(dim=['lat', 'lon'], skipna=True)

    ax.plot(ds['time'].values, DIC_weighted_mean.values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

#plt.legend(loc='lower right', ncol=ncol)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Average surface ocean DIC (µmol kg$^{-1}$)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% change in DIC (full ocean)
fig = plt.figure(figsize=(3.5,3.5), dpi=200)
ax = fig.gca()
DIC_start_ds = xr.DataArray(DIC_start_3D, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})

# plot preindustrial baseline
ax.axhline(np.average(DIC_preind, weights=flatten(model_vols,ocnmask)), c='black', linestyle='--', label='Preindustrial DIC')

# use xarray to open metadata of files of interest
for exp_idx in tqdm(range(len(experiment_names))):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # wrap GLODAP DIC in xarray dataset to convert ∆DIC to total DIC over time
    DIC_modeled_3D = ds['delDIC'] + DIC_start_ds
    
    # wrap model_vols in xarray dataset to convert from concentration to amount or use in weighted average
    model_vols_ds = xr.DataArray(model_vols, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
    DIC_weighted_mean = DIC_modeled_3D.weighted(model_vols_xr).mean(dim=['lat','lon', 'depth'])

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

fig = plt.figure(figsize=(3.5,3.5), dpi=200)
ax = fig.gca()
DIC_preind_ds = xr.DataArray(DIC_preind_3D, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
AT_ds = xr.DataArray(AT_3D, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})

# plot preindustrial baseline
ax.axhline(np.average(pH_preind[surf_idx], weights=flatten(model_vols,ocnmask)[surf_idx]), c='black', linestyle='--', label='Preindustrial pH')

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # same for ∆AT to AT
    AT_modeled_3D = ds['delAT'] + AT_ds

    avg_pH_modeled_surf = np.zeros(len(ds['time']))

    for idx in tqdm(range(len(ds['time']))):
        # wrap Canth in xarray dataset to convert ∆DIC to total DIC over time
        Canth_ds = xr.DataArray(make_3d(Canth_all_scenarios[exp_idx][idx], ocnmask), dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
        DIC_modeled_3D = ds['delDIC'] + DIC_preind_ds + Canth_ds
        
        AT_modeled = flatten(AT_modeled_3D.isel(time=idx).values,ocnmask)[surf_idx]
        DIC_modeled = flatten(DIC_modeled_3D.isel(time=idx).values,ocnmask)[surf_idx]

        #  call co2sys to calculate pH
        co2sys = pyco2.sys(
            alkalinity=AT_modeled,
            dic=DIC_modeled,
            salinity=S[surf_idx],
            temperature=T[surf_idx],
            pressure=pressure[surf_idx],
            total_silicate=Si[surf_idx],
            total_phosphate=P[surf_idx])
    
        avg_pH_modeled_surf[idx] = np.average(co2sys['pH'], weights=flatten(model_vols,ocnmask)[surf_idx])
        
    ax.plot(ds['time'].values, avg_pH_modeled_surf, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Average surface ocean pH (µmol kg$^{-1}$)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% change in pH (full ocean)
fig = plt.figure(figsize=(3.5,3.5), dpi=200)
ax = fig.gca()
DIC_preind_ds = xr.DataArray(DIC_preind_3D, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
AT_ds = xr.DataArray(AT_3D, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})

# plot preindustrial baseline
ax.axhline(np.average(pH_preind, weights=flatten(model_vols,ocnmask)), c='black', linestyle='--', label='Preindustrial pH')

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # same for ∆AT to AT
    AT_modeled_3D = ds['delAT'] + AT_ds
    
    avg_pH_modeled = np.zeros(len(ds['time']))
    
    for idx in tqdm(range(len(ds['time']))):
        # wrap Canth in xarray dataset to convert ∆DIC to total DIC over time
        Canth_ds = xr.DataArray(make_3d(Canth_all_scenarios[exp_idx][idx], ocnmask), dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
        DIC_modeled_3D = ds['delDIC'] + DIC_preind_ds + Canth_ds
        
        AT_modeled = flatten(AT_modeled_3D.isel(time=idx).values, ocnmask)
        DIC_modeled = flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)

        #  call co2sys to calculate pH
        co2sys = pyco2.sys(
            alkalinity=AT_modeled,
            dic=DIC_modeled,
            salinity=S,
            temperature=T,
            pressure=pressure,
            total_silicate=Si,
            total_phosphate=P)
        
        avg_pH_modeled[idx] = np.average(co2sys['pH'], weights=flatten(model_vols,ocnmask))
    
    ax.plot(ds['time'].values, avg_pH_modeled, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

plt.legend(loc='lower right', ncol=ncol)
plt.xlabel('Year', fontsize=11.5, weight='bold')
plt.ylabel('Average ocean pH (µmol kg$^{-1}$)', fontsize=11.5, weight='bold')
ax.spines['bottom'].set_color(textcolor)
ax.spines['top'].set_color(textcolor)
ax.spines['left'].set_color(textcolor)
ax.spines['right'].set_color(textcolor)

#%% plot surface pH
co2sys = pyco2.sys(
    alkalinity=flatten(ds.isel(time=1).delAT.values,ocnmask) + flatten(AT_3D,ocnmask),
    dic=flatten(ds.isel(time=1).delDIC.values,ocnmask) + flatten(DIC_3D,ocnmask),
    salinity=S,
    temperature=T,
    pressure=pressure,
    total_silicate=Si,
    total_phosphate=P)

pH_plot = co2sys['pH']
pH_plot_3D = make_3d(pH_plot,ocnmask)
plot_surface2d(model_lon, model_lat, pH_plot_3D[0,:,:], 2, 10, 'viridis', 'pH (t = 15, depth = ' + str(np.round(model_depth[4], 2)) + ' m)')
#plt.ylim([-500, 2500])
plt.show()

#%% animations of tracers
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)

# AT added
make_surf_animation(ds.AT_added, 'AT Added (µmol kg-1)', model_lat, model_lon, ds.time.values, len(ds.time.values), 0, 0.0002, 'viridis', 'LCA1_2026-03-12_dt-1mon_00000_AT_added.mp4')

# delDIC
make_surf_animation(ds.delDIC, 'Change in DIC (µmol kg-1)', model_lat, model_lon, ds.time.values, len(ds.time.values), -0.00001, 0.00001, 'RdBu', 'LCA1_2026-03-12_dt-1mon_00000_delDIC.mp4')

#%% animation of deviation from preindustrial pH
ds = xr.open_dataset('/Volumes/LaCie/outputs/exp22_2026-02-06_t1_none_000.nc')
DIC_start_ds = xr.DataArray(DIC_start_3D, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
AT_ds = xr.DataArray(AT_3D, dims=["lat", "lon", "depth"], coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})

# wrap GLODAP DIC in xarray dataset to convert ∆DIC to total DIC over time
DIC_modeled_3D = ds['delDIC'] + DIC_start_ds
    
# same for ∆AT to AT
AT_modeled_3D = ds['delAT'] + AT_ds
    
pH_modeled = []
    
for idx in tqdm(range(len(ds['time']))):
        
    AT_modeled = flatten(AT_modeled_3D.isel(time=idx).values, ocnmask)
    DIC_modeled = flatten(DIC_modeled_3D.isel(time=idx).values, ocnmask)

    #  call co2sys to calculate pH
    co2sys = pyco2.sys(
        alkalinity=AT_modeled,
        dic=DIC_modeled,
        salinity=S,
        temperature=T,
        pressure=pressure,
        total_silicate=Si,
        total_phosphate=P)
        
    pH_modeled.append(co2sys['pH'] - pH_preind)

make_surf_animation_pH(pH_modeled, 'Deviation of pH from preindustrial', model_lat, model_lon, ds.time.values, len(ds.time.values), ocnmask, -0.2, 0.2, 'RdBu', 'exp22_2026-02-06_t1_none_pH_deviation_preind.mp4')

#%% Canth


#%%