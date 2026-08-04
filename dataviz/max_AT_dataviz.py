"""
Created on Mon Jun 22 2026

DATAVIZ FOR MAX_AT: Maximum alkalinity calculation

@author: Reese C. Barrett
"""
#%%
from dataviz.dataviz import broadcast_to_dataset, get_co2_scenario, load_ocim_grid, load_glodap, apply_style
from oae_tmm.grid import flatten, make_3d
from oae_tmm.trace import calculate_canth, interp_trace
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import PyCO2SYS as pyco2
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback
from gsw import SA_from_SP, pt0_from_t

# load model architecture
data_path = './data/'
#output_path = '/Volumes/LaCie/outputs/max_AT/'
output_path = './outputs/'

grid        = load_ocim_grid(data_path)
ocnmask     = grid['ocnmask']
mldmask     = grid['mldmask']
latitude    = grid['latitude']
longitude   = grid['longitude']
depth       = grid['depth']
cell_volume = grid['cell_volume']
rho = 1025  # seawater density [kg m-3]

textcolor, fontweight = apply_style()
_fs = 13

plot_start       = 2020   # pre-CDR period start for plots
start_simulation = 2030   # simulation start year
end_year         = 2100   # simulation end year

#%% pull timestepping comparison experiments (generated with --test --exp-id 0-6)
experiment_names = ['max_AT_2026-07-30_long_none',
                    'max_AT_2026-07-30_long_ssp126',
                    'max_AT_2026-07-30_long_ssp245',
                    'max_AT_2026-07-30_long_ssp534']

labels = ['None', 'SSP1-2.6', 'SSP2-4.5', 'SSP5-3.4 OS']
legend_labels = ['Fixed Atm. CO₂', 'SSP1-2.6', 'SSP2-4.5', 'SSP5-3.4 OS']
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

_expected_end = float(np.arange(start_simulation, end_year, 1/360)[-1])

if os.path.exists(cache_path):
    cache = xr.open_dataset(cache_path).load()
    stale = [(exp, lbl) for exp, lbl in zip(experiment_names, labels)
             if f'AT_added_cum_{lbl}' not in cache
             or float(cache[f'time_{lbl}'].values[-1]) < _expected_end - 0.01]
    if stale:
        stale_labels = [lbl for _, lbl in stale]
        drop_vars   = [v  for v  in cache.data_vars if any(v.endswith(f'_{lbl}')  for lbl in stale_labels)]
        drop_coords = [co for co in cache.coords    if any(co == f'time_{lbl}'    for lbl in stale_labels)]
        cache = cache.drop_vars(drop_vars + drop_coords, errors='ignore')
        new_vars = {}
        for exp_name, label in tqdm(stale, desc='Computing stale/missing experiments'):
            new_vars.update(_load_experiment(exp_name, label))
        with TqdmCallback(desc='Computing cache entries'):
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
# b. change in atmospheric CO2 over time & deviation from co2 scenarios
# c. change in ocean CT content
# d. oae efficiency over time (delCT / delAT)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=200)

pre_time = np.arange(plot_start, start_simulation, 1)
pre_zeros = np.zeros(len(pre_time))

for label, legend_label, color in zip(labels, legend_labels, colors):
    time = cache[f'AT_added_cum_{label}'][f'time_{label}'].values
    axes[0][0].plot(np.concatenate([pre_time, time]),
                    np.concatenate([pre_zeros, cache[f'AT_added_cum_{label}'].values * 1e-15]),
                    label=legend_label, c=color)
axes[0][0].set_xlabel('Year', fontsize=_fs)
axes[0][0].set_ylabel(r'Cumulative ${A_{\mathrm{T}}}$ Added (Pmol)', fontsize=_fs)
axes[0][0].set_xlim([plot_start, end_year])
axes[0][0].legend(fontsize=_fs, frameon=False)
for side in ('top', 'bottom', 'left', 'right'):
    axes[0][0].spines[side].set_color(textcolor)
axes[0][0].tick_params(labelsize=_fs)
axes[0][0].text(0.02, 0.97, '(a)', transform=axes[0][0].transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

for label, legend_label, scenario, color in zip(labels, legend_labels, scenarios, colors):
    time = cache[f'delxCO2_{label}'][f'time_{label}'].values
    time_extended = np.concatenate([pre_time, time])
    atmospheric_co2 = get_co2_scenario(scenario, time_extended)
    axes[0][1].plot(time, cache[f'delxCO2_{label}'].values + atmospheric_co2[len(pre_time):], label=legend_label, c=color)
    axes[0][1].plot(time_extended, atmospheric_co2, label=legend_label, ls=':', c=color)
axes[0][1].set_xlabel('Year', fontsize=_fs)
axes[0][1].set_ylabel(r'Atmospheric CO$_2$ (ppm)', fontsize=_fs)
axes[0][1].set_xlim([plot_start, end_year])
for side in ('top', 'bottom', 'left', 'right'):
    axes[0][1].spines[side].set_color(textcolor)
axes[0][1].tick_params(labelsize=_fs)
axes[0][1].text(0.02, 0.97, '(b)', transform=axes[0][1].transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

for label, legend_label, color in zip(labels, legend_labels, colors):
    time = cache[f'delCT_{label}'][f'time_{label}'].values
    axes[1][0].plot(np.concatenate([pre_time, time]),
                    np.concatenate([pre_zeros, cache[f'delCT_{label}'].values * 1e-15 * 12.011]),
                    label=legend_label, c=color)
axes[1][0].set_xlabel('Year', fontsize=_fs)
axes[1][0].set_ylabel(r'Change in $C_{\mathrm{T}}$ Content (PgC)', fontsize=_fs)
axes[1][0].set_xlim([plot_start, end_year])
for side in ('top', 'bottom', 'left', 'right'):
    axes[1][0].spines[side].set_color(textcolor)
axes[1][0].tick_params(labelsize=_fs)
axes[1][0].text(0.02, 0.97, '(c)', transform=axes[1][0].transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

for label, legend_label, color in zip(labels, legend_labels, colors):
    time = cache[f'delCT_{label}'][f'time_{label}'].values
    delAT_vals = cache[f'delAT_{label}'].values
    with np.errstate(invalid='ignore', divide='ignore'):
        efficiency = np.where(delAT_vals != 0, cache[f'delCT_{label}'].values / delAT_vals * 100, np.nan)
    axes[1][1].plot(time, efficiency, label=legend_label, c=color)
axes[1][1].set_xlabel('Year', fontsize=_fs)
axes[1][1].set_ylabel(r'OAE Efficiency: ${C_{\mathrm{T}}}^{\prime}$ / ${A_{\mathrm{T}}}^{\prime}$ (%)', fontsize=_fs)
axes[1][1].set_xlim([plot_start, end_year])
for side in ('top', 'bottom', 'left', 'right'):
    axes[1][1].spines[side].set_color(textcolor)
axes[1][1].tick_params(labelsize=_fs)
axes[1][1].text(0.02, 0.97, '(d)', transform=axes[1][1].transAxes,
                fontsize=_fs, va='top', ha='left', color=textcolor)

plt.tight_layout()

#%% statistics: ocean total changes

# for each scenario, print AT added, change in CT, change in atmospheric CO2, total atmospheric CO2, % change in atmospheric CO2 w/OAE (compared to anthropogenic increase)

_tbl_vars = ['AT_added (Pmol)', 'delCT (PgC)', 'delxCO2 (ppm)', 'xCO2 (ppm)', 'CO2 equiv. yr', 'eta (%)']
_fmts = ['.2f', '.2f', '.2f', '.2f', '.0f', '.2f']
_lw  = 16
_vw  = 16
_sep = '=' * (_lw + _vw * len(_tbl_vars))

_traj_data = np.loadtxt('./pyTRACE/pyTRACE/data/CO2Trajectories.txt')
_traj_years = _traj_data[:, 0]
_traj_scenario_cols = {'none': 1, 'ssp119': 2, 'ssp126': 3, 'ssp245': 4, 'ssp370': 5,
                       'ssp370_lowNTCF': 6, 'ssp434': 7, 'ssp460': 8, 'ssp534_OS': 9, 'REMIND': 10}

labels = ['None', 'SSP1-2.6', 'SSP2-4.5', 'SSP5-3.4 OS']
scenarios = ['none', 'ssp126', 'ssp245', 'ssp534_OS']

print(f'Total Ocean Changes Statistics {end_year}')
print(_sep)
print(f"{'':>{_lw}}" + ''.join(f"{h:>{_vw}}" for h in _tbl_vars))

avg_delxCO2 = 0
avg_eta = 0

for label, scenario in zip(labels, scenarios):
    _row = f"{label:<{_lw}}"
    AT_added = cache[f'AT_added_cum_{label}'].sel({f'time_{label}': end_year}, method='nearest', tolerance=0.5).values * 1e-15
    delAT = cache[f'delAT_{label}'].sel({f'time_{label}': end_year}, method='nearest', tolerance=0.5).values * 1e-15
    delCT = cache[f'delCT_{label}'].sel({f'time_{label}': end_year}, method='nearest', tolerance=0.5).values * 1e-15
    delxCO2 = cache[f'delxCO2_{label}'].sel({f'time_{label}': end_year}, method='nearest', tolerance=0.5).values
    if scenario == 'none': xCO2 = delxCO2 + get_co2_scenario(scenario, [plot_start])
    else: xCO2 = delxCO2 + get_co2_scenario(scenario, [end_year])
    _equiv_mask = (_traj_years <= 2022) if scenario == 'none' else slice(None)
    _equiv_co2  = _traj_data[_equiv_mask, _traj_scenario_cols[scenario]]
    _equiv_yrs  = _traj_years[_equiv_mask]
    hist_yr_equiv = np.interp(float(xCO2), _equiv_co2, _equiv_yrs)
    eta = delCT/delAT * 100
    delCT = delCT * 12.011
    _vars = [AT_added, delCT, delxCO2, xCO2, hist_yr_equiv, eta]
    avg_delxCO2 += delxCO2
    avg_eta += eta
    for _var, _fmt in zip(_vars, _fmts):
        _row += f"{float(_var):>{_vw}{_fmt}}"
    print(_row)
print(_sep)
avg_delxCO2 /= len(scenarios)
avg_eta /= len(scenarios)

print(f'Average decrease in atmospheric CO2 by 2100:\t{avg_delxCO2:.2f} ppm')
print(f'Average eta by 2100:\t{avg_eta:.2f} %')

# %% convert Caserini et al. (2022) limestone reserves value to mol
# https://doi.org/10.1029/2021GB007246
limestone_Gt = 15000
g_per_mol_limestone = 40.078 + 12.011 + 3*15.999
# Gt -> metric ton -> grams -> moles -> Pmol
limestone_Pmol = limestone_Gt * 1e9 * 1e6 / g_per_mol_limestone * 1e-15
print(f'Nearshore-ish limestone reserves in Pmol:\t {limestone_Pmol}')


#%% load data for pH calculations

glodap         = load_glodap(data_path)
CT_3d          = glodap['CT_3d']
AT_3d          = glodap['AT_3d']
temperature_3d = glodap['temperature_3d']
salinity_3d    = glodap['salinity_3d']
silicate_3d    = glodap['silicate_3d']
phosphate_3d   = glodap['phosphate_3d']

salinity    = flatten(salinity_3d,    ocnmask)
temperature = flatten(temperature_3d, ocnmask)
silicate    = flatten(silicate_3d,    ocnmask)
phosphate   = flatten(phosphate_3d,   ocnmask)

# create "pressure" array by broadcasting depth array
pressure_3d = np.tile(depth[:, np.newaxis, np.newaxis], (1, ocnmask.shape[0], ocnmask.shape[1])).transpose([1, 2, 0])
pressure      = flatten(pressure_3d, ocnmask)
latitude_flat  = flatten(np.broadcast_to(latitude[:, np.newaxis, np.newaxis],  ocnmask.shape), ocnmask)
longitude_flat = flatten(np.broadcast_to(longitude[np.newaxis, :, np.newaxis], ocnmask.shape), ocnmask)

# make surface temperature array for calculation of pCO2
abs_salinity = SA_from_SP(salinity, pressure, longitude_flat, latitude_flat)
potential_temperature = pt0_from_t(abs_salinity, temperature, pressure) 

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
co2sys_2050         = pyco2.sys(dic=flatten(CT_2050_3d, ocnmask),
                                alkalinity=flatten(AT_3d, ocnmask),
                                salinity=salinity, temperature=temperature, pressure=pressure,
                                total_silicate=silicate, total_phosphate=phosphate)
co2sys_2050_surf    = pyco2.sys(dic=flatten(CT_2050_3d, ocnmask),
                                alkalinity=flatten(AT_3d, ocnmask),
                                salinity=salinity, temperature=potential_temperature, pressure=0,
                                total_silicate=silicate, total_phosphate=phosphate)
pH_2050     = co2sys_2050['pH']
RC_2050     = co2sys_2050['revelle_factor']
omegaA_2050 = co2sys_2050['saturation_aragonite']
pCO2_2050   = co2sys_2050_surf['pCO2']

pH_2050_3d     = make_3d(pH_2050, ocnmask)
RC_2050_3d     = make_3d(RC_2050, ocnmask)
omegaA_2050_3d = make_3d(omegaA_2050, ocnmask)
pCO2_2050_3d   = make_3d(pCO2_2050, ocnmask)

co2sys_2100         = pyco2.sys(dic=flatten(CT_2100_3d, ocnmask),
                                alkalinity=flatten(AT_3d, ocnmask),
                                salinity=salinity, temperature=temperature, pressure=pressure,
                                total_silicate=silicate, total_phosphate=phosphate)
co2sys_2100_surf    = pyco2.sys(dic=flatten(CT_2100_3d, ocnmask),
                                alkalinity=flatten(AT_3d, ocnmask),
                                salinity=salinity, temperature=potential_temperature, pressure=0,
                                total_silicate=silicate, total_phosphate=phosphate)
pH_2100     = co2sys_2100['pH']
RC_2100     = co2sys_2100['revelle_factor']
omegaA_2100 = co2sys_2100['saturation_aragonite']
pCO2_2100   = co2sys_2100_surf['pCO2']

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
co2sys_OAE_2050         = pyco2.sys(dic=flatten(CT_OAE_2050_3d, ocnmask),
                                    alkalinity=flatten(AT_OAE_2050_3d, ocnmask),
                                    salinity=salinity, temperature=temperature, pressure=pressure,
                                    total_silicate=silicate, total_phosphate=phosphate)
co2sys_OAE_2050_surf    = pyco2.sys(dic=flatten(CT_OAE_2050_3d, ocnmask),
                                    alkalinity=flatten(AT_OAE_2050_3d, ocnmask),
                                    salinity=salinity, temperature=potential_temperature, pressure=0,
                                    total_silicate=silicate, total_phosphate=phosphate)
pH_OAE_2050       = co2sys_OAE_2050['pH']
RC_OAE_2050       = co2sys_OAE_2050['revelle_factor']
omegaA_OAE_2050   = co2sys_OAE_2050['saturation_aragonite']
pCO2_OAE_2050     = co2sys_OAE_2050_surf['pCO2']

pH_OAE_2050_3d     = make_3d(pH_OAE_2050, ocnmask)
RC_OAE_2050_3d     = make_3d(RC_OAE_2050, ocnmask)
omegaA_OAE_2050_3d = make_3d(omegaA_OAE_2050, ocnmask)
pCO2_OAE_2050_3d   = make_3d(pCO2_OAE_2050, ocnmask)

co2sys_OAE_2100         = pyco2.sys(dic=flatten(CT_OAE_2100_3d, ocnmask),
                                    alkalinity=flatten(AT_OAE_2100_3d, ocnmask),
                                    salinity=salinity, temperature=temperature, pressure=pressure,
                                    total_silicate=silicate, total_phosphate=phosphate)
co2sys_OAE_2100_surf    = pyco2.sys(dic=flatten(CT_OAE_2100_3d, ocnmask),
                                    alkalinity=flatten(AT_OAE_2100_3d, ocnmask),
                                    salinity=salinity, temperature=potential_temperature, pressure=0,
                                    total_silicate=silicate, total_phosphate=phosphate)
pH_OAE_2100       = co2sys_OAE_2100['pH']
RC_OAE_2100       = co2sys_OAE_2100['revelle_factor']
omegaA_OAE_2100   = co2sys_OAE_2100['saturation_aragonite']
pCO2_OAE_2100     = co2sys_OAE_2100_surf['pCO2']

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
    (r'$R_C$',       del_RC_2050_3d[:, :, 0],     del_RC_2100_3d[:, :, 0]),
    (r'${\Omega_A}$',  del_omegaA_2050_3d[:, :, 0], del_omegaA_2100_3d[:, :, 0]),
]

fig3, axes3 = plt.subplots(3, 2, figsize=(12, 9), dpi=200,
                           subplot_kw={'projection': map_proj})

im_list = []
for row_idx, (var_label, data_2050, data_2100) in enumerate(surface_vars):
    vmax_row = max(np.nanmax(np.abs(data_2050)), np.nanmax(np.abs(data_2100)))
    for col_idx, (year, data) in enumerate([(2050, data_2050), (2100, data_2100)]):
        axes3[row_idx, col_idx].set_global()
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
cbar2.set_label(r'Change in $p_{\mathrm{CO_2}}$ (µatm)', fontsize=_fs)
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
print(f'{(depth[depth_idx] - depth[0]):.0f} m change in pCO2 at 53ºN, 151ºW (2100):\t{del_pCO2_2100_3d[72, 104, depth_idx]:.2f}')
print(f'Depth difference (m):\t\t\t\t{depth[depth_idx] - depth[0]:.2f}')

#%% statistics: subsurface ocean pH changes

# calculate preindustrial pH
co2sys_preind = pyco2.sys(dic=flatten(CT_preind_3d, ocnmask),
                          alkalinity=flatten(AT_3d, ocnmask),
                          salinity=salinity, temperature=temperature, pressure=pressure,
                          total_silicate=silicate, total_phosphate=phosphate)
pH_preind     = co2sys_preind['pH']
pH_preind_3d  = make_3d(pH_preind, ocnmask)

print(f'whole ocean preindustrial pH:\t{np.average(flatten(pH_preind_3d, ocnmask), weights=flatten(cell_volume, ocnmask)):.2f}')
avg_surf_pH_preind = np.average(flatten(pH_preind_3d, mldmask), weights=flatten(cell_volume, mldmask))
print(f'surface ocean mixed layer preindustrial pH:\t{avg_surf_pH_preind:.2f}')

# weighted avg mixed layer pH under SSP2-4.5 max OAE over time
salinity_mld    = flatten(salinity_3d,    mldmask)
temperature_mld = flatten(temperature_3d, mldmask)
silicate_mld    = flatten(silicate_3d,    mldmask)
phosphate_mld   = flatten(phosphate_3d,   mldmask)
pressure_mld    = flatten(pressure_3d,    mldmask)
weights_mld     = flatten(cell_volume,    mldmask)
CT_preind_mld   = flatten(CT_preind_3d,   mldmask)
AT_mld          = flatten(AT_3d,          mldmask)

surf_pH_cache_path = output_path + 'max_AT_surf_pH_cache.nc'

if os.path.exists(surf_pH_cache_path):
    _surf_pH_cache = xr.open_dataset(surf_pH_cache_path).load()
    time_surf_pH = _surf_pH_cache['time'].values
    avg_surf_pH_OAE_ts = _surf_pH_cache['avg_surf_pH_OAE'].values
    if float(time_surf_pH[-1]) < _expected_end - 0.01:
        os.remove(surf_pH_cache_path)
        raise RuntimeError('Surface pH cache is stale — delete it and rerun to recompute.')
else:
    with xr.open_mfdataset(
            output_path + experiment_names[2] + '_*.nc',
            combine='by_coords',
            chunks={'time': 1},
            parallel=True) as ds:
        time_surf_pH = ds['time'].values
        avg_surf_pH_OAE_ts = np.full(len(time_surf_pH), np.nan)
        for i, t in enumerate(tqdm(time_surf_pH, desc='Computing surface pH time series')):
            delCT_t  = ds['delCT'].isel(time=i).values
            delAT_t  = ds['delAT'].isel(time=i).values
            Canth_t  = interp_trace(data_path, t, 'ssp245', latitude, longitude, depth, ocnmask)
            CT_t_mld = CT_preind_mld + flatten(Canth_t, mldmask) + flatten(delCT_t, mldmask)
            AT_t_mld = AT_mld + flatten(delAT_t, mldmask)
            co2sys_t = pyco2.sys(dic=CT_t_mld, alkalinity=AT_t_mld,
                                 salinity=salinity_mld, temperature=temperature_mld, pressure=pressure_mld,
                                 total_silicate=silicate_mld, total_phosphate=phosphate_mld)
            avg_surf_pH_OAE_ts[i] = np.average(co2sys_t['pH'], weights=weights_mld)
    xr.Dataset({'avg_surf_pH_OAE': ('time', avg_surf_pH_OAE_ts)},
               coords={'time': time_surf_pH}).to_netcdf(surf_pH_cache_path)

within_5pct  = np.abs(avg_surf_pH_OAE_ts - avg_surf_pH_preind) / avg_surf_pH_preind < 0.05
crossing_idx = np.where(within_5pct)[0]
if len(crossing_idx) > 0:
    recovery_year = time_surf_pH[crossing_idx[0]]
    print(f'Year surface pH within 5% of preindustrial: {recovery_year:.0f}')
else:
    recovery_year = None
    print('Surface pH never recovers to within 5% of preindustrial within simulation')

fig_surf_pH, ax_surf_pH = plt.subplots(figsize=(8, 4), dpi=200)
ax_surf_pH.plot(time_surf_pH, avg_surf_pH_OAE_ts, label='SSP2-4.5 max OAE', c=colors[2])
ax_surf_pH.axhline(avg_surf_pH_preind, color=textcolor, ls='--', label='Preindustrial')
if recovery_year is not None:
    ax_surf_pH.axvline(recovery_year, color='gray', ls=':', label=f'Within 5% (year {recovery_year:.0f})')
ax_surf_pH.set_xlabel('Year', fontsize=_fs)
ax_surf_pH.set_ylabel('Avg Mixed Layer pH', fontsize=_fs)
ax_surf_pH.set_xlim([time_surf_pH[0], time_surf_pH[-1]])
ax_surf_pH.legend(fontsize=_fs, frameon=False)
ax_surf_pH.tick_params(labelsize=_fs)
for side in ('top', 'bottom', 'left', 'right'):
    ax_surf_pH.spines[side].set_color(textcolor)
plt.tight_layout()

# calculate
avg_subsurf_pH_2100_OAE = np.average(flatten(pH_OAE_2100_3d, ocnmask.astype(bool) & ~mldmask.astype(bool)), weights=flatten(cell_volume, ocnmask.astype(bool) & ~mldmask.astype(bool)))
avg_subsurf_pH_preind = np.average(flatten(pH_preind_3d, ocnmask.astype(bool) & ~mldmask.astype(bool)), weights=flatten(cell_volume, ocnmask.astype(bool) & ~mldmask.astype(bool)))
print(f'subsurface pH in 2100 (SSP2-4.5 w/max OAE):\t{avg_subsurf_pH_2100_OAE:.2f}')
print(f'subsurface preindustrial pH:\t{avg_subsurf_pH_preind:.2f}')
print(f'preindustrial - 2100:\t{(avg_subsurf_pH_preind - avg_subsurf_pH_2100_OAE):.2f}')

# %% statistics: climate relevance

# atmospheric CO2 decline by 2100 from simulation start (2030)
start_CDR = 2030
_tbl2_vars = ['delxCO2_OAE (ppm)', 'xCO2_2100-xCO2_2030 (ppm)']
_lw2  = 16
_vw2  = 24
_sep2 = '=' * (_lw2 + _vw2 * len(_tbl2_vars))

print('Atmospheric CO2 Changes by 2100')
print(_sep2)
print(f"{'':>{_lw2}}" + ''.join(f"{h:>{_vw2}}" for h in _tbl2_vars))

for label, scenario in zip(labels, scenarios):
    delxCO2_2100 = cache[f'delxCO2_{label}'].sel({f'time_{label}': 2100}, method='nearest', tolerance=0.5).values
    xCO2_2100    = delxCO2_2100 + get_co2_scenario(scenario, [2100])
    xCO2_2030    = get_co2_scenario(scenario, [start_CDR])
    total_change = xCO2_2100 - xCO2_2030
    print(f"{label:<{_lw2}}{float(delxCO2_2100):>{_vw2}.2f}{float(total_change):>{_vw2}.2f}")
print(_sep2)

# atmospheric CO2 levels at 2100 with max OAE
print('Atmospheric CO2 at 2100 with max OAE')
for label, scenario in zip(labels[1:], scenarios[1:]):
    delxCO2_2100 = cache[f'delxCO2_{label}'].sel({f'time_{label}': 2100}, method='nearest', tolerance=0.5).values
    xCO2_2100    = delxCO2_2100 + get_co2_scenario(scenario, [2100])
    print(f'  {label}:\t{float(xCO2_2100):.2f} ppm')

# additional C accumulated in ocean by 2100 for ssp2-4.5
delCT_2100 = cache['delCT_SSP2-4.5'].sel({'time_SSP2-4.5': 2100}, method='nearest', tolerance=0.5).values
print(f'delCT at 2100 (SSP2-4.5):\t{delCT_2100 * 12.011 * 1e-15:.2f} PgC')

# amount of CT accumulated as of 2022 (using TRACE)
Canth_2022 = calculate_canth('none', 2022, temperature_3d, salinity_3d, ocnmask, latitude, longitude, depth) * cell_volume * rho * 1e-6 # mol 
print(f'Canth accumulated as of 2022:\t{np.nansum(Canth_2022) * 1e-15 * 12.011:.2e} PgC')

#%%