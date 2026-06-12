"""
Regrid observational data products to the OCIM2-48L grid.

Run once to produce the .nc files in data/GLODAPv2.2016b.MappedProduct/
and the NCEP/NOAA reanalysis arrays used by experiments at runtime.
"""

from oae_tmm.loaders import load_ocim
from oae_tmm.regrid import regrid_glodap, regrid_ncep_noaa

data_path = './data/'

ocim = load_ocim(data_path)
ocnmask   = ocim['ocnmask']
latitude  = ocim['latitude']
longitude = ocim['longitude']
depth     = ocim['depth']

# regrid GLODAPv2.2016b mapped fields
for var in ['TCO2', 'TAlk', 'pHtsinsitutp', 'temperature', 'salinity', 'silicate', 'PO4']:
    regrid_glodap(data_path, var, latitude, longitude, depth, ocnmask)

# regrid NCEP/DOE reanalysis II surface fields
for var in ['icec', 'wspd', 'sst']:
    regrid_ncep_noaa(data_path, var, latitude, longitude, ocnmask)
