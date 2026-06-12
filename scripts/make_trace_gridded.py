"""
Generate the TRACE gridded Canth product on the OCIM2-48L grid at higher
temporal resolution.

Calls oae_tmm.trace.calculate_canth for every scenario and year from 2000
to 2100 and saves results as NetCDF. Run once to produce the files used
by interp_trace at experiment time:

    data/TRACE_gridded/OCIM_CanthFromTRACECO2Pathway{1..10}.nc
"""

import numpy as np
import xarray as xr
from tqdm import tqdm

from oae_tmm.loaders import load_ocim
from oae_tmm.trace import calculate_canth

data_path = './data/'
output_path = './data/TRACE_gridded/'

ocim = load_ocim(data_path)
ocnmask   = ocim['ocnmask']
latitude  = ocim['latitude']
longitude = ocim['longitude']
depth     = ocim['depth']

temperature_3d = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/temperature.npy')
salinity_3d    = np.load(data_path + 'GLODAPv2.2016b.MappedProduct/salinity.npy')

scenarios = {
    'none': 1, 'ssp119': 2, 'ssp126': 3, 'ssp245': 4, 'ssp370': 5,
    'ssp370_lowNTCF': 6, 'ssp434': 7, 'ssp460': 8, 'ssp534_OS': 9, 'REMIND': 10,
}

years = np.arange(2000, 2101)

for scenario_name, scenario_idx in scenarios.items():
    print(f'Processing scenario: {scenario_name}')
    canth_time_series = []

    for year in tqdm(years):
        canth_3d = calculate_canth(
            scenario_name, year, temperature_3d, salinity_3d, ocnmask, latitude, longitude, depth,
        )
        canth_time_series.append(canth_3d)

    canth_xr = xr.DataArray(
        np.array(canth_time_series),
        dims=['time', 'latitude', 'longitude', 'depth'],
        coords={
            'time': years,
            'latitude': latitude,
            'longitude': longitude,
            'depth': depth,
        },
        name='Canth',
    )

    output_filename = output_path + f'OCIM_CanthFromTRACECO2Pathway{scenario_idx}.nc'
    canth_xr.to_dataset().to_netcdf(output_filename)
    print(f'Saved: {output_filename}')
