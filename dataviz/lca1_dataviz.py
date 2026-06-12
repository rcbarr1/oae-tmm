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
from oae_tmm.grid import flatten, make_3d, get_depth_idx
from oae_tmm.trace import calculate_canth, interp_trace
from dataviz.dataviz import get_co2_scenario, plot_surface2d, plot_surface3d, broadcast_to_dataset, make_surf_animation
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

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

latitude    = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
longitude   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
depth       = model_data['tz'].isel(longitude=0, latitude=0).to_numpy()   # m below sea surface
cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

rho = 1025  # seawater density [kg m-3]
surf_idx = get_depth_idx(ocnmask, 0)

model_data.close()

#%% set experiments we are interested in plotting

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

scenarios = ['REMIND']

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
_glodap = data_path + 'GLODAPv2.2016b.MappedProduct/'
CT_3d          = xr.open_dataset(_glodap + 'CT.nc')['CT'].values                    # dissolved inorganic carbon [µmol kg-1]
AT_3d          = xr.open_dataset(_glodap + 'AT.nc')['AT'].values                    # total alkalinity [µmol kg-1]
temperature_3d = xr.open_dataset(_glodap + 'temperature.nc')['temperature'].values  # temperature [ºC]
salinity_3d    = xr.open_dataset(_glodap + 'salinity.nc')['salinity'].values        # salinity [unitless]
silicate_3d    = xr.open_dataset(_glodap + 'silicate.nc')['silicate'].values        # silicate [µmol kg-1]
phosphate_3d   = xr.open_dataset(_glodap + 'phosphate.nc')['phosphate'].values      # phosphate [µmol kg-1]

salinity    = flatten(salinity_3d,    ocnmask)
temperature = flatten(temperature_3d, ocnmask)
silicate    = flatten(silicate_3d,    ocnmask)
phosphate   = flatten(phosphate_3d,   ocnmask)

# get TRACE data
Canth_2002_3d = calculate_canth('REMIND', 2002, temperature_3d, salinity_3d, ocnmask, latitude, longitude, depth)
if start_year != 2002:
    Canth_3d = calculate_canth('REMIND', start_year, temperature_3d, salinity_3d, ocnmask, latitude, longitude, depth)
else:
    Canth_3d = Canth_2002_3d

# calculate preindustrial CT by subtracting anthropogenic carbon
CT_preind_3d = CT_3d - Canth_2002_3d
CT_preind    = flatten(CT_preind_3d, ocnmask)

CT_start_3d = CT_preind_3d + Canth_3d

# create "pressure" array by broadcasting depth array
pressure_3d = np.tile(depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[0], ocnmask.shape[1])).transpose([1, 2, 0])
pressure    = flatten(pressure_3d, ocnmask)

# calculate preindustrial pH assuming steady state alkalinity
co2sys = pyco2.sys(dic=CT_preind,
                   alkalinity=flatten(AT_3d, ocnmask),
                   salinity=flatten(salinity_3d, ocnmask),
                   temperature=flatten(temperature_3d, ocnmask),
                   pressure=flatten(pressure_3d, ocnmask),
                   total_silicate=flatten(silicate_3d, ocnmask),
                   total_phosphate=flatten(phosphate_3d, ocnmask))

pH_preind    = co2sys['pH']
pH_preind_3d = make_3d(pH_preind, ocnmask)

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
                Canth_idx_3d = interp_trace(data_path, t[idx], scenario, latitude, longitude, depth, ocnmask)
            else:
                Canth_idx_3d = Canth_3d
            Canth_all_idx.append(flatten(Canth_idx_3d, ocnmask))
        Canth_all_scenarios.append(Canth_all_idx)

# without interpolation
else:
    for scenario in scenarios:
        Canth_all_idx = []
        for idx in tqdm(range(len(t))):
            if scenario != 'none':
                Canth_idx_3d = calculate_canth(scenario, t[idx], temperature_3d, salinity_3d, ocnmask, latitude, longitude, depth)
            else:
                Canth_idx_3d = Canth_3d
            Canth_all_idx.append(flatten(Canth_idx_3d, ocnmask))
        Canth_all_scenarios.append(Canth_all_idx)

np.save(output_path + 'Canth_LCA_start2050.npy', Canth_all_scenarios)

#%% load in Canth calculated
Canth_start2026 = np.load(output_path + 'Canth_LCA_start2026.npy')
Canth_start2050 = np.load(output_path + 'Canth_LCA_start2050.npy')

#%% calculate eta over time for each location
# eta = mol C / mol AT = delCT / delAT

fig = plt.figure(figsize=(5,5), dpi=200)
ax = fig.gca()

for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)

    cell_volume_xr = broadcast_to_dataset(cell_volume, ds)

    delAT_mol  = ds['delAT']  * cell_volume_xr * rho * 1e-6
    delCT_mol = ds['delCT'] * cell_volume_xr * rho * 1e-6

    delAT_mol_total = delAT_mol.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)
    delCT_mol_total = delCT_mol.sum(dim=['latitude', 'longitude', 'depth'], skipna=True)

    eta = delCT_mol_total / delAT_mol_total

    ax.plot(ds['time'].values - start_years[exp_idx], eta.values, label=labels[exp_idx], c=linecolors[exp_idx], ls=linestyles[exp_idx])

plt.legend(bbox_to_anchor=(0.97, -0.1), ncol=2)
plt.xlabel('Year Since A$_{T}$ Pulse')
plt.ylabel('η (∆ mol C per ∆ mol A$_{T}$)')

#%%