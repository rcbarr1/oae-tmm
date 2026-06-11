"""
Created on Wed May 13 12:26 2026

DATA VIZ FOR EXP26: monte carlo simulation testing air-sea gas exchange parameterization

@author: Reese C. Barrett
"""
#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oae_tmm.loaders import load_mat
from oae_tmm.grid import get_depth_idx
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import PyCO2SYS as pyco2
from tqdm import tqdm

# load model architecture
data_path = './data/'
# output_path = './outputs/'
output_path = '/Volumes/LaCie/outputs/'

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
surf_idx = get_depth_idx(ocnmask,0) # indicies of surface grid cells in 3D array flattened by p2.flatten()

model_data.close()

# rules for saving files
t_per_file = 2000 # number of time steps 

# calculate when new layers start (for line plots)
new_layer_idx = np.zeros(len(model_depth))
for i in range(len(model_depth)):
    new_layer_idx[i] = int(np.nansum(ocnmask[i,:,:]))
new_layer_idx = np.cumsum(new_layer_idx)

#%% set experiments we are interested in plotting
num_mc = 144
experiment_names = []

for i in range(num_mc):
    experiment_name = 'exp26_2026-03-28_t1_' + f'{i:05d}'
    experiment_names.append(experiment_name)

# %% calculate global ocean average CDR efficiency (eta = delDIC / delAT) for each run after 20 years
etas = np.zeros(num_mc)

# use xarray to open metadata of files of interest
for exp_idx in range(len(experiment_names)):
    ds = xr.open_mfdataset(
        output_path + experiment_names[exp_idx] + '_*.nc',
        combine='by_coords',
        chunks={'time': 10},
        parallel=True)
    
    # sum across all grid cells to get total change in AT and DIC
    model_vols_xr = xr.DataArray(model_vols, # broadcast model volumes to xarray to convert from concentration in per kg to total
                                 dims=["lat", "lon", "depth"],
                                 coords={"lat": ds.lat, "lon": ds.lon, "depth": ds.depth})
    delDIC = ds['delDIC'] * model_vols_xr * rho # convert from µmol/kg to µmol
    delAT = ds['delAT'] * model_vols_xr * rho # convert from µmol/kg to µmol

    eta_full_ocean = delDIC.isel(time=-1).sum() / delAT.isel(time=-1).sum()
    etas[exp_idx] = eta_full_ocean * 100

eta_avg = np.mean(etas)
eta_std = np.std(etas)

# plot histogram of etas
fig = plt.figure(figsize=(8, 8), dpi=200)
ax = fig.gca()

ax.hist(etas, bins=20, edgecolor='black', alpha=0.7)

ax.set_xlabel('CDR Efficiency (η = ∆DIC/∆AT * 100%)')
ax.set_ylabel('Number of MC Simulations')
ax.set_title('Average global ocean CDR efficiency 20 years after AT pulse\nvarying gas transfer velocity (k) with standard deviation of 0.2')
ax.axvline(eta_avg, color='red', linestyle='--', linewidth=2, label=f'Mean: {eta_avg:.2f}%')
ax.axvline(eta_avg - eta_std, color='orange', linestyle=':', linewidth=2, label=f'Std Dev: ±{eta_std:.2f}%')
ax.axvline(eta_avg + eta_std, color='orange', linestyle=':', linewidth=2)
ax.legend()
plt.tight_layout()
plt.show()

#%% map of average oae efficiency and standard deviation of oae efficiency at each surface location for full surface pulse

# THIS DOESN'T MAKE SENSE FOR FULL SURFACE RELEASE SINCE ADVECTION AND SUCH WOULD MAKE THE IMPACT OF AT ADDED TO EACH CELL MOVE BEYOND THAT CELL BY 20 YEARS
# INSTEAD, TO SHOW SPATIALLY, WOULD NEED TO REPEAT FULL MC SIMULATION FOR A PULSE ADDITION AT EACH GRID CELL
# THIS WOULD BE COOL BUT WOULD TAKE A LONG TIME (STILL POSSIBLY FEASIBLE THOUGH?)

# surf_ocn_etas = np.full((num_mc, ocnmask.shape[0], ocnmask.shape[1]), np.nan)

# # use xarray to open metadata of files of interest
# for exp_idx in range(len(experiment_names)):
#     ds = xr.open_mfdataset(
#         output_path + experiment_names[exp_idx] + '_*.nc',
#         combine='by_coords',
#         chunks={'time': 10},
#         parallel=True)
    
#     eta_surf = ds['delDIC'].isel(depth=0, time=-1) / ds['delAT'].isel(depth=0, time=-1)
#     surf_ocn_etas[exp_idx, :, :] = eta_surf * 100

# eta_surf_avg = np.nanmean(surf_ocn_etas, axis=0)
# eta_surf_std = np.nanstd(surf_ocn_etas, axis=0)

# # plot map of surface ocean average and std eta
# p2.plot_surface2d(model_lat, model_lon, eta_surf_avg, 0, 100, 'viridis', 'Average CDR efficiency 20 years after AT pulse\nvarying gas transfer velocity (k) with std = 0.2')
# p2.plot_surface2d(model_lat, model_lon, eta_surf_std, 0, 15, 'magma', 'Std. deviation of CDR efficiency 20 years after AT pulse\nvarying gas transfer velocity (k) with std = 0.2')


#%%



