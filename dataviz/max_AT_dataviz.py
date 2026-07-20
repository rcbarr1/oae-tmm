"""
Created on Mon Jun 22 2026

DATAVIZ FOR MAX_AT: Maximum alkalinity calculation

@author: Reese C. Barrett
"""
#%%
from dataviz.dataviz import broadcast_to_dataset, get_co2_scenario
from oae_tmm.grid import flatten, make_3d
from oae_tmm.trace import calculate_canth
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import PyCO2SYS as pyco2
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

# load model architecture
data_path = './data/'
output_path = '/Volumes/LaCie/outputs/max_AT/'
#output_path = './outputs/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

latitude    = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
longitude   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
depth       = model_data['tz'].isel(longitude=0, latitude=0).to_numpy()   # m below sea surface
cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

model_data.close()
rho = 1025  # seawater density [kg m-3]

fontweight = 'normal'
textcolor = '#000000'
mpl.rcParams['font.family']     = 'Calibri'
mpl.rcParams['font.weight']     = fontweight
mpl.rcParams['text.color']      = textcolor
mpl.rcParams['axes.labelcolor'] = textcolor
mpl.rcParams['xtick.color']     = textcolor
mpl.rcParams['ytick.color']     = textcolor
_fs = 13

#%% pull timestepping comparison experiments (generated with --test --exp-id 0-6)
experiment_names = ['max_AT_2026-07-01_long_none',
                    'max_AT_2026-07-01_long_ssp126',
                    'max_AT_2026-07-01_long_ssp245',
                    'max_AT_2026-07-01_long_ssp534']

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

# %% figure: four-panel full-ocean totals for paper
# a. cumulative AT added over time
# b. oae efficiency over time (delCT / delAT)
# c. change in atmospheric CO2 over time
# d. deviation from non max AT co2 trajectories

fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=200)

pre_time = np.arange(2020, 2030, 1)
pre_zeros = np.zeros(len(pre_time))

for label, color in zip(labels, colors):
    time = cache[f'AT_added_cum_{label}'][f'time_{label}'].values
    axes[0][0].plot(np.concatenate([pre_time, time]),
                    np.concatenate([pre_zeros, cache[f'AT_added_cum_{label}'].values]), 
                    label=label, c=color)
axes[0][0].set_xlabel('Year', fontsize=_fs)
axes[0][0].set_ylabel(r'Cumulative ${A_{\mathrm{T}}}$ Added (mol)', fontsize=_fs)
axes[0][0].set_xlim([2020, 2100])
axes[0][0].legend(fontsize=_fs, frameon=False)
for side in ('top', 'bottom', 'left', 'right'):
    axes[0][0].spines[side].set_color(textcolor)
axes[0][0].tick_params(labelsize=_fs)
axes[0][0].text(0.02, 0.97, '(a)', transform=axes[0][0].transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

for label, scenario, color in zip(labels, scenarios, colors):
    time = cache[f'delxCO2_{label}'][f'time_{label}'].values
    time_extended = np.concatenate([np.arange(2020, 2030, 1), time])
    atmospheric_co2 = get_co2_scenario(scenario, time_extended)
    axes[0][1].plot(time, cache[f'delxCO2_{label}'].values + atmospheric_co2[10:], label=label, c=color)
    axes[0][1].plot(time_extended, atmospheric_co2, label=label, ls=':', c=color)
axes[0][1].set_xlabel('Year', fontsize=_fs)
axes[0][1].set_ylabel(r'Atmospheric CO$_2$ (ppm)', fontsize=_fs)
axes[0][1].set_xlim([2020, 2100])
for side in ('top', 'bottom', 'left', 'right'):
    axes[0][1].spines[side].set_color(textcolor)
axes[0][1].tick_params(labelsize=_fs)
axes[0][1].text(0.02, 0.97, '(b)', transform=axes[0][1].transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

for label, color in zip(labels, colors):
    time = cache[f'delCT_{label}'][f'time_{label}'].values
    axes[1][0].plot(np.concatenate([pre_time, time]),
                    np.concatenate([pre_zeros, cache[f'delCT_{label}'].values]),
                    label=label, c=color)
axes[1][0].set_xlabel('Year', fontsize=_fs)
axes[1][0].set_ylabel(r'Change in $C_{\mathrm{T}}$ Content (mol)', fontsize=_fs)
axes[1][0].set_xlim([2020, 2100])
for side in ('top', 'bottom', 'left', 'right'):
    axes[1][0].spines[side].set_color(textcolor)
axes[1][0].tick_params(labelsize=_fs)
axes[1][0].text(0.02, 0.97, '(c)', transform=axes[1][0].transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

for label, color in zip(labels, colors):
    time = cache[f'delCT_{label}'][f'time_{label}'].values
    delAT_vals = cache[f'delAT_{label}'].values
    with np.errstate(invalid='ignore', divide='ignore'):
        efficiency = np.where(delAT_vals != 0, cache[f'delCT_{label}'].values / delAT_vals * 100, np.nan)
    axes[1][1].plot(time, efficiency, label=label, c=color)
axes[1][1].set_xlabel('Year', fontsize=_fs)
axes[1][1].set_ylabel(r'OAE Efficiency: ${C_{\mathrm{T}}}^{\prime}$ / ${A_{\mathrm{T}}}^{\prime}$ (%)', fontsize=_fs)
axes[1][1].set_xlim([2020, 2100])
for side in ('top', 'bottom', 'left', 'right'):
    axes[1][1].spines[side].set_color(textcolor)
axes[1][1].tick_params(labelsize=_fs)
axes[1][1].text(0.02, 0.97, '(d)', transform=axes[1][1].transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

plt.tight_layout()

#%% statistics: ocean total changes

# for each scenario, print AT added, change in CT, change in atmospheric CO2, total atmospheric CO2, % change in atmospheric CO2 w/OAE (compared to anthropogenic increase)

_tbl_vars = ['AT_added (Pmol)', 'delCT (Pmol)', 'delxCO2 (ppm)', 'xCO2 (ppm)', 'rel_delxCO2 (%)', 'eta (%)']
_lw  = 16
_vw  = 16
_sep = '=' * (_lw + _vw * len(_tbl_vars))

year = 2075
start_year = 2020
start_CDR = 2030
labels = ['None', 'SSP1-2.6', 'SSP2-4.5', 'SSP5-3.4 OS']
scenarios = ['none', 'ssp126', 'ssp245', 'ssp534_OS']

print(f'Total Ocean Changes Statistics {year}')
print(_sep)
print(f"{'':>{_lw}}" + ''.join(f"{h:>{_vw}}" for h in _tbl_vars))

for label, scenario in zip(labels, scenarios):
    _row = f"{label:<{_lw}}"
    AT_added = cache[f'AT_added_cum_{label}'].sel({f'time_{label}': year}, method='nearest', tolerance=0.5).values * 1e-15
    delAT = cache[f'delAT_{label}'].sel({f'time_{label}': year}, method='nearest', tolerance=0.5).values * 1e-15
    delCT = cache[f'delCT_{label}'].sel({f'time_{label}': year}, method='nearest', tolerance=0.5).values * 1e-15
    delxCO2 = cache[f'delxCO2_{label}'].sel({f'time_{label}': year}, method='nearest', tolerance=0.5).values
    if scenario == 'none': xCO2 = delxCO2 + get_co2_scenario(scenario, [start_year])
    else: xCO2 = delxCO2 + get_co2_scenario(scenario, [year])
    if scenario == 'none': relative_delxCO2 = np.nan 
    else: relative_delxCO2 = np.abs(delxCO2) / (get_co2_scenario(scenario, [year]) - get_co2_scenario(scenario, [start_CDR])) * 100
    eta = delCT/delAT * 100
    _vars = [AT_added, delCT, delxCO2, xCO2, relative_delxCO2, eta]
    for _var in _vars:
        _row += f"{float(_var):>{_vw}.2f}"
    print(_row)
print(_sep)

#%% load data for pH calculations

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

# create "pressure" array by broadcasting depth array
pressure_3d = np.tile(depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[0], ocnmask.shape[1])).transpose([1, 2, 0])
pressure    = flatten(pressure_3d, ocnmask)

#%% calculate pH (no OAE) on OCIM grid in 2050 and 2100 using SSP2-4.5

# get TRACE data
Canth_2002_3d = calculate_canth('none', 2002, temperature_3d, salinity_3d, ocnmask, latitude, longitude, depth)
Canth_2050_3d = calculate_canth('ssp245', 2050, temperature_3d, salinity_3d, ocnmask, latitude, longitude, depth)
Canth_2100_3d = calculate_canth('ssp245', 2100, temperature_3d, salinity_3d, ocnmask, latitude, longitude, depth)

# calculate preindustrial CT by subtracting anthropogenic carbon
CT_preind_3d = CT_3d - Canth_2002_3d

CT_2050_3d = CT_preind_3d + Canth_2050_3d
CT_2100_3d = CT_preind_3d + Canth_2100_3d

# calculate pH assuming steady state alkalinity
co2sys_2050 = pyco2.sys(dic=flatten(CT_2050_3d, ocnmask),
                        alkalinity=flatten(AT_3d, ocnmask),
                        salinity=salinity, temperature=temperature, pressure=pressure,
                        total_silicate=silicate, total_phosphate=phosphate)
pH_2050     = co2sys_2050['pH']
RC_2050     = co2sys_2050['revelle_factor']
omegaA_2050 = co2sys_2050['saturation_aragonite']
pCO2_2050   = co2sys_2050['pCO2']

pH_2050_3d     = make_3d(pH_2050, ocnmask)
RC_2050_3d     = make_3d(RC_2050, ocnmask)
omegaA_2050_3d = make_3d(omegaA_2050, ocnmask)
pCO2_2050_3d   = make_3d(pCO2_2050, ocnmask)

co2sys_2100 = pyco2.sys(dic=flatten(CT_2100_3d, ocnmask),
                        alkalinity=flatten(AT_3d, ocnmask),
                        salinity=salinity, temperature=temperature, pressure=pressure,
                        total_silicate=silicate, total_phosphate=phosphate)
pH_2100     = co2sys_2100['pH']
RC_2100     = co2sys_2100['revelle_factor']
omegaA_2100 = co2sys_2100['saturation_aragonite']
pCO2_2100   = co2sys_2100['pCO2']

pH_2100_3d     = make_3d(pH_2100, ocnmask)
RC_2100_3d     = make_3d(RC_2100, ocnmask)
omegaA_2100_3d = make_3d(omegaA_2100, ocnmask)
pCO2_2100_3d   = make_3d(pCO2_2100, ocnmask)

#%% calculate pH (max OAE) on OCIM grid in 2050 and 2100 using SSP2-4.5

with xr.open_mfdataset(
        output_path + experiment_names[2] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True) as ds:
    
    CT_OAE_2050_3d = CT_preind_3d + Canth_2050_3d + ds['delCT'].sel(time=2050, method='nearest', tolerance=0.5).values
    CT_OAE_2100_3d = CT_preind_3d + Canth_2100_3d + ds['delCT'].sel(time=2100, method='nearest', tolerance=0.5).values

    AT_OAE_2050_3d = AT_3d + ds['delAT'].sel(time=2050, method='nearest', tolerance=0.5).values
    AT_OAE_2100_3d = AT_3d + ds['delAT'].sel(time=2100, method='nearest', tolerance=0.5).values

# calculate pH assuming steady state alkalinity
co2sys_OAE_2050 = pyco2.sys(dic=flatten(CT_OAE_2050_3d, ocnmask),
                            alkalinity=flatten(AT_OAE_2050_3d, ocnmask),
                            salinity=salinity, temperature=temperature, pressure=pressure,
                            total_silicate=silicate, total_phosphate=phosphate)
pH_OAE_2050       = co2sys_OAE_2050['pH']
RC_OAE_2050       = co2sys_OAE_2050['revelle_factor']
omegaA_OAE_2050   = co2sys_OAE_2050['saturation_aragonite']
pCO2_OAE_2050     = co2sys_OAE_2050['pCO2']

pH_OAE_2050_3d     = make_3d(pH_OAE_2050, ocnmask)
RC_OAE_2050_3d     = make_3d(RC_OAE_2050, ocnmask)
omegaA_OAE_2050_3d = make_3d(omegaA_OAE_2050, ocnmask)
pCO2_OAE_2050_3d   = make_3d(pCO2_OAE_2050, ocnmask)

co2sys_OAE_2100 = pyco2.sys(dic=flatten(CT_OAE_2100_3d, ocnmask),
                            alkalinity=flatten(AT_OAE_2100_3d, ocnmask),
                            salinity=salinity, temperature=temperature, pressure=pressure,
                            total_silicate=silicate, total_phosphate=phosphate)
pH_OAE_2100       = co2sys_OAE_2100['pH']
RC_OAE_2100       = co2sys_OAE_2100['revelle_factor']
omegaA_OAE_2100   = co2sys_OAE_2100['saturation_aragonite']
pCO2_OAE_2100     = co2sys_OAE_2100['pCO2']

pH_OAE_2100_3d     = make_3d(pH_OAE_2100, ocnmask)
RC_OAE_2100_3d     = make_3d(RC_OAE_2100, ocnmask)
omegaA_OAE_2100_3d = make_3d(omegaA_OAE_2100, ocnmask)
pCO2_OAE_2100_3d   = make_3d(pCO2_OAE_2100, ocnmask)

del_pH_2050_3d     = pH_OAE_2050_3d - pH_2050_3d
del_RC_2050_3d     = RC_OAE_2050_3d - RC_2050_3d
del_omegaA_2050_3d = omegaA_OAE_2050_3d - omegaA_2050_3d
del_pCO2_2050_3d   = pCO2_OAE_2050_3d - pCO2_2050_3d

del_pH_2100_3d     = pH_OAE_2100_3d - pH_2100_3d
del_RC_2100_3d     = RC_OAE_2100_3d - RC_2100_3d
del_omegaA_2100_3d = omegaA_OAE_2100_3d - omegaA_2100_3d
del_pCO2_2100_3d   = pCO2_OAE_2100_3d - pCO2_2100_3d

# %% figures: carbonate chemistry changes visualizations (SSP2-4.5, max OAE vs. no OAE)
# rows: 2050 (top), 2100 (bottom)
# cols: surface map | Pacific (209°E) | Atlantic (335°E) | Indian Ocean (91°E)

pac_idx, atl_idx, ind_idx = 104, 167, 45
section_lons = [longitude[pac_idx], longitude[atl_idx], longitude[ind_idx]]
section_lat_lims = [(59, -75), (67, -76), (21, -66)]  # (north, south) to match N-to-S latitude axis

data_crs = ccrs.PlateCarree()
map_proj = ccrs.EqualEarth(central_longitude=200)

# surface maps of change in pH, Revelle factor, Ω_A (3 rows × 2 cols)
surface_vars = [
    (r'$\mathrm{pH}$', del_pH_2050_3d[:, :, 0],     del_pH_2100_3d[:, :, 0]),
    (r'${R_C}$',      del_RC_2050_3d[:, :, 0],     del_RC_2100_3d[:, :, 0]),
    (r'${\Omega_A}$',  del_omegaA_2050_3d[:, :, 0], del_omegaA_2100_3d[:, :, 0]),
]

fig3, axes3 = plt.subplots(3, 2, figsize=(12, 9), dpi=200,
                            subplot_kw={'projection': map_proj})

im_list = []
for row_idx, (var_label, data_2050, data_2100) in enumerate(surface_vars):
    vmax_row = max(np.nanmax(np.abs(data_2050)), np.nanmax(np.abs(data_2100)))
    for col_idx, (year, data) in enumerate([(2050, data_2050), (2100, data_2100)]):
        axes3[row_idx, col_idx].set_global()
        axes3[row_idx, col_idx].set_facecolor('#b0cfe0')
        im = axes3[row_idx, col_idx].pcolormesh(longitude, latitude, data,
                                                cmap='RdBu', vmin=-vmax_row, vmax=vmax_row,
                                                transform=data_crs)
        axes3[row_idx, col_idx].add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
        axes3[row_idx, col_idx].coastlines(linewidth=0.5)
        for sec_lon, lat_lim in zip(section_lons, section_lat_lims):
            axes3[row_idx, col_idx].plot([sec_lon, sec_lon], list(lat_lim), color='yellow',
                                         linewidth=0.8, transform=data_crs)
        if row_idx == 0:
            axes3[row_idx, col_idx].set_title(str(year), fontsize=_fs, color=textcolor)
        axes3[row_idx, col_idx].text(0.02, 0.97, f'({chr(ord("a") + row_idx * 2 + col_idx)})',
                                     transform=axes3[row_idx, col_idx].transAxes,
                                     fontsize=_fs, va='top', ha='left', color=textcolor)
    im_list.append((im, var_label))

plt.tight_layout()
fig3.subplots_adjust(right=0.85, hspace=0.35)
for row_idx, (im, var_label) in enumerate(im_list):
    ax_pos  = axes3[row_idx, 1].get_position()
    cbar_ax = fig3.add_axes([0.87, ax_pos.y0, 0.015, ax_pos.height])
    cbar3   = fig3.colorbar(im, cax=cbar_ax)
    cbar3.set_label(f'Change in {var_label}', fontsize=_fs)
    cbar3.ax.yaxis.label.set_color(textcolor)
    cbar3.ax.tick_params(colors=textcolor, labelsize=_fs)

sec_cmap = plt.cm.RdBu.copy()
sec_cmap.set_bad('lightgray')

# pCO2 interior sections (3 rows × 3 cols)
fig2, sec_axes = plt.subplots(2, 3, figsize=(12, 7), dpi=200)

sec_col_titles = ['Pacific', 'Atlantic', 'Indian Ocean']
del_pCO2_by_year = {
    2050: [del_pCO2_2050_3d[:, :, 0],
           del_pCO2_2050_3d[:, pac_idx, :],
           del_pCO2_2050_3d[:, atl_idx, :],
           del_pCO2_2050_3d[:, ind_idx, :]],
    2100: [del_pCO2_2100_3d[:, :, 0],
           del_pCO2_2100_3d[:, pac_idx, :],
           del_pCO2_2100_3d[:, atl_idx, :],
           del_pCO2_2100_3d[:, ind_idx, :]],
}

vmax = max(np.nanmax(np.abs(d)) for row in del_pCO2_by_year.values() for d in row)

for row_idx, year in enumerate([2050, 2100]):
    for col_idx, (section, lat_lim, title) in enumerate(
            zip(del_pCO2_by_year[year][1:], section_lat_lims, sec_col_titles)):
        im = sec_axes[row_idx, col_idx].pcolormesh(latitude, depth, section.T,
                                                   cmap=sec_cmap, vmin=-vmax, vmax=vmax)
        sec_axes[row_idx, col_idx].set_xlabel('Latitude (°N)', fontsize=_fs)
        sec_axes[row_idx, col_idx].set_ylabel('Depth (m)', fontsize=_fs)
        sec_axes[row_idx, col_idx].set_xlim(lat_lim[1], lat_lim[0])
        sec_axes[row_idx, col_idx].set_ylim(0, 2500)
        sec_axes[row_idx, col_idx].invert_yaxis()
        sec_axes[row_idx, col_idx].set_title(f'{title}, {year}', fontsize=_fs, color=textcolor)
        sec_axes[row_idx, col_idx].text(0.02, 0.97, f'({chr(ord("a") + row_idx * 3 + col_idx)})',
                                        transform=sec_axes[row_idx, col_idx].transAxes,
                                        fontsize=_fs, va='top', ha='left', color=textcolor)
        for side in ('top', 'bottom', 'left', 'right'):
            sec_axes[row_idx, col_idx].spines[side].set_color(textcolor)
        sec_axes[row_idx, col_idx].tick_params(labelsize=_fs)

plt.tight_layout()
fig2.subplots_adjust(right=0.87)
cbar_ax = fig2.add_axes([0.89, 0.1, 0.015, 0.8])
cbar2   = fig2.colorbar(im, cax=cbar_ax)
cbar2.set_label(r'Change in $\mathrm{pCO_2}$ (µatm)', fontsize=_fs)
cbar2.ax.yaxis.label.set_color(textcolor)
cbar2.ax.tick_params(colors=textcolor, labelsize=_fs)

#%% statistics: carbonate chemistry changes

# for SSP2-4.5, average change in surface ocean pH, Revelle factor, Ω_A, and pCO2 by 2100
_labels = ['pH', 'RC', 'omegaA', 'pCO2']
_vars   = [del_pH_2100_3d, del_RC_2100_3d, del_omegaA_2100_3d, del_pCO2_2100_3d]

for var, label in zip(_vars, _labels):
    data    = flatten(var[:,:,0], ocnmask[:,:,0]), 
    weights = flatten(cell_volume[:,:,0], ocnmask[:,:,0]), 
    print(f'Average change in surface ocean {label} (2100):\t{np.average(data, weights=weights):.2f}')

# maximum surface and subsurface pCO2 changes
print(f'Maximum change in surface ocean pCO2 (2100):\t{np.nanmin(del_pCO2_2100_3d[:,:,0]):.2f}')
print(f'Maximum change in subsurface ocean pCO2 (2100):\t{np.nanmin(del_pCO2_2100_3d[:,:,1::]):.2f}')

# sample location in North Pacific (53 ºN, 151 ºW)
depth_idx = 12
print(f'Surface change in pCO2 at 53ºN, 151ºW (2100):\t{del_pCO2_2100_3d[72, 104, 0]:.2f}')
print(f'{depth[depth_idx]:.0f} m change in pCO2 at 53ºN, 151ºW (2100):\t{del_pCO2_2100_3d[72, 104, depth_idx]:.2f}')
print(f'Depth difference (m):\t\t\t\t{depth[depth_idx] - depth[0]:.2f}')
#%%