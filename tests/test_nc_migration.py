"""
Migration verification: compare .nc outputs against original .npy files.

Run this after regenerating data with the new regrid.py code, while the old
.npy files still exist on disk. All checks should pass before deleting .npy files.

Usage:
    python tests/test_nc_migration.py [data_path]

    data_path defaults to './data/' if not provided.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr


def _check(cond, msg):
    status = 'PASS' if cond else 'FAIL'
    print(f'  {status}  {msg}')
    return cond


def run(data_path: str) -> bool:
    checks = [
        # (npy_rel_path, nc_rel_path, variable_name)
        ('GLODAPv2.2016b.MappedProduct/CT.npy',          'GLODAPv2.2016b.MappedProduct/CT.nc',          'CT'),
        ('GLODAPv2.2016b.MappedProduct/AT.npy',          'GLODAPv2.2016b.MappedProduct/AT.nc',          'AT'),
        ('GLODAPv2.2016b.MappedProduct/temperature.npy', 'GLODAPv2.2016b.MappedProduct/temperature.nc', 'temperature'),
        ('GLODAPv2.2016b.MappedProduct/salinity.npy',    'GLODAPv2.2016b.MappedProduct/salinity.nc',    'salinity'),
        ('GLODAPv2.2016b.MappedProduct/silicate.npy',    'GLODAPv2.2016b.MappedProduct/silicate.nc',    'silicate'),
        ('GLODAPv2.2016b.MappedProduct/phosphate.npy',   'GLODAPv2.2016b.MappedProduct/phosphate.nc',   'phosphate'),
        ('NCEP_DOE_Reanalysis_II/icec.npy',              'NCEP_DOE_Reanalysis_II/icec.nc',              'icec'),
        ('NCEP_DOE_Reanalysis_II/wspd.npy',              'NCEP_DOE_Reanalysis_II/wspd.nc',              'wspd'),
        ('NOAA_Extended_Reconstruction_SST_V5/sst.npy',  'NOAA_Extended_Reconstruction_SST_V5/sst.nc',  'sst'),
    ]

    passed = True
    for npy_rel, nc_rel, var in checks:
        npy_path = data_path + npy_rel
        nc_path  = data_path + nc_rel
        if not os.path.exists(npy_path):
            print(f'  SKIP  {npy_rel} (file not found)')
            continue
        if not os.path.exists(nc_path):
            ok = _check(False, f'{nc_rel} [{var}] — .nc file not found')
            passed = False
            continue
        npy_data = np.load(npy_path)
        nc_data  = xr.open_dataset(nc_path)[var].values
        ok = _check(
            np.allclose(npy_data, nc_data, equal_nan=True),
            f'{nc_rel} [{var}]',
        )
        passed = passed and ok

    return passed


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else './data/'
    if not data_path.endswith('/'):
        data_path += '/'
    print(f'\n--- nc migration check (data_path={data_path}) ---')
    ok = run(data_path)
    print('\n  ALL PASS' if ok else '\n  SOME CHECKS FAILED')
    if not ok:
        sys.exit(1)
