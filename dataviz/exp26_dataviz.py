"""
Created on Wed May 13 12:26 2026

DATA VIZ FOR EXP26: monte carlo simulation testing air-sea gas exchange parameterization

@author: Reese C. Barrett
"""
#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataviz.dataviz import broadcast_to_dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# load model architecture
data_path = './data/'
output_path = './outputs/'
# output_path = '/Volumes/LaCie/outputs/'

# open data associated with transport matrix
model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
ocnmask = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()

latitude    = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()    # ºN
longitude   = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()     # ºE
cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy() # m^3

model_data.close()
rho = 1025  # seawater density [kg m-3]

#%% set experiments we are interested in plotting
num_mc = 144
experiment_names = ['exp26_2026-03-28_t1_' + f'{i:05d}' for i in range(num_mc)]

# %% calculate global ocean average CDR efficiency (eta = delCT / delAT) for each run after 20 years
etas = np.zeros(num_mc)

for exp_idx, experiment_name in enumerate(experiment_names):
    with xr.open_mfdataset(
            output_path + experiment_name + '_*.nc',
            combine='by_coords',
            chunks={'time': 10},
            parallel=True) as ds:

        cell_volume_xr = broadcast_to_dataset(cell_volume, ds)
        delCT = ds['delCT'] * cell_volume_xr * rho  # µmol/kg → µmol
        delAT  = ds['delAT']  * cell_volume_xr * rho  # µmol/kg → µmol

        eta_full_ocean = delCT.isel(time=-1).sum() / delAT.isel(time=-1).sum()
        etas[exp_idx] = float(eta_full_ocean) * 100

eta_avg = float(np.mean(etas))
eta_std = float(np.std(etas))

# plot histogram of etas
fig = plt.figure(figsize=(8, 8), dpi=200)
ax = fig.gca()

ax.hist(etas, bins=20, edgecolor='black', alpha=0.7)

ax.set_xlabel('CDR Efficiency (η = ∆CT/∆AT * 100%)')
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

#     eta_surf = ds['delCT'].isel(depth=0, time=-1) / ds['delAT'].isel(depth=0, time=-1)
#     surf_ocn_etas[exp_idx, :, :] = eta_surf * 100

# eta_surf_avg = np.nanmean(surf_ocn_etas, axis=0)
# eta_surf_std = np.nanstd(surf_ocn_etas, axis=0)

# # plot map of surface ocean average and std eta
# p2.plot_surface2d(latitude, longitude, eta_surf_avg, 0, 100, 'viridis', 'Average CDR efficiency 20 years after AT pulse\nvarying gas transfer velocity (k) with std = 0.2')
# p2.plot_surface2d(latitude, longitude, eta_surf_std, 0, 15, 'magma', 'Std. deviation of CDR efficiency 20 years after AT pulse\nvarying gas transfer velocity (k) with std = 0.2')


#%%



