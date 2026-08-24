"""
One-time script to copy timestepping cache files and add units attributes.
Writes to *_with_units.nc copies; inspect before replacing the originals.

Variable units (derived from oae_tmm/output.py and timestepping_dataviz.py):
  AT_added_cum_*  mol    (AT_added [µmol/kg] × cell_volume [m³] × ρ [kg/m³] × 1e-6, cumulative)
  delxCO2_*       ppm    (taken directly from model output)
  delCT_*         mol    (delCT [µmol/kg] × cell_volume × ρ × 1e-6, global sum)
  delAT_*         mol    (delAT [µmol/kg] × cell_volume × ρ × 1e-6, global sum)
"""
import xarray as xr
from pathlib import Path

outputs = Path('./outputs')

_unit_map = {
    'AT_added_cum': 'mol',
    'delxCO2':      'ppm',
    'delCT':        'mol',
    'delAT':        'mol',
}

def _add_units(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for var in ds.data_vars:
        for prefix, unit in _unit_map.items():
            if var.startswith(prefix):
                ds[var].attrs['units'] = unit
                break
    return ds


for src in [
    outputs / 'max_AT_timestepping_cache.nc',
    outputs / 'ir_timestepping_cache_2026-06-23.nc',
]:
    if not src.exists():
        print(f'SKIP (not found): {src}')
        continue
    dst = src.with_stem(src.stem + '_with_units')
    ds = xr.open_dataset(src).load()
    ds = _add_units(ds)
    ds.to_netcdf(dst)
    ds.close()
    print(f'Written: {dst}')
    # Spot-check
    ds2 = xr.open_dataset(dst)
    sample = list(ds2.data_vars)[0]
    print(f'  sample var {sample!r}: units={ds2[sample].attrs.get("units", "MISSING")}')
    ds2.close()
