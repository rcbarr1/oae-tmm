"""
One-time data preprocessing functions for oae-tmm.

These functions convert raw observational data products to the OCIM2-48L grid
and save the results as .npy files. They are called once from
scripts/generate_input_data.py, not during experiment runs. The functions
serve as documentation of exactly how each input dataset was processed.

Data sources:
  - GLODAPv2.2016b: https://glodap.info/index.php/mapped-data-product/
  - WOA18: https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/
  - NCEP/DOE Reanalysis II: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html
  - NOAA ERSSTv5: https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html
  - COBALT: GFDL ocean biogeochemistry model output
"""

import time
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from oae_tmm.grid import inpaint_nans_3d, inpaint_nans_2d


def regrid_glodap(data_path: str, glodap_var: str, latitude: np.ndarray,
                  longitude: np.ndarray, depth: np.ndarray,
                  ocnmask: np.ndarray) -> None:
    """Regrid a GLODAPv2.2016b mapped variable to the OCIM grid and save as .npy.

    Interpolates from the GLODAP lat/lon/depth grid to the OCIM2-48L grid
    using linear interpolation, then fills remaining NaNs with iterative
    neighbor averaging. The result is saved to the GLODAPv2.2016b.MappedProduct/
    subdirectory of data_path.

    GLODAP uses longitudes from 20E to 380E (i.e., shifted by 20 degrees),
    so model longitudes are temporarily shifted to match before interpolating.

    Parameters
    ----------
    data_path : str
        Path to the data directory GLODAPv2.2016b.MappedProduct/
    glodap_var : str
        Variable name as it appears in the GLODAP NetCDF file (e.g., 'TCO2',
        'TAlk', 'temperature', 'salinity', 'silicate', 'PO4').
    latitude : np.ndarray
        1D array of OCIM2-48L latitude values [degrees N].
    longitude : np.ndarray
        1D array of OCIM2-48L longitude values [degrees E, 0-360].
    depth : np.ndarray
        1D array of OCIM2-48L depth values [m].
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    """
    print('begin regrid of ' + glodap_var)
    start_time = time.time()

    glodap_data = xr.open_dataset(
        data_path + 'GLODAPv2.2016b.MappedProduct/GLODAPv2.2016b.' + glodap_var + '.nc'
    )

    glodap_lat   = glodap_data['lat'].to_numpy()
    glodap_lon   = glodap_data['lon'].to_numpy()
    glodap_depth = glodap_data['Depth'].to_numpy()

    # transpose GLODAP dimensions from (depth, lat, lon) to (lat, lon, depth) to match OCIM
    var = glodap_data[glodap_var].transpose('lat', 'lon', 'depth_surface').copy().values

    interp = RegularGridInterpolator(
        (glodap_lat, glodap_lon, glodap_depth), var, bounds_error=False, fill_value=None  # type: ignore[arg-type]
    )

    # GLODAP longitudes run from 20E to 380E; shift model lons to match
    longitude[longitude < 20] += 360

    lat_grid, lon_grid, depth_grid = np.meshgrid(latitude, longitude, depth, indexing='ij')
    query_points = np.array([lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel()]).T
    var = interp(query_points).reshape(depth_grid.shape)

    var = inpaint_nans_3d(var, mask=ocnmask)

    # restore model longitudes
    longitude[longitude > 360] -= 360

    # silicate and phosphate can go slightly negative due to interpolation near zero, set to zero if needed
    if glodap_var in ('PO4', 'silicate'):
        var[var < 0] = 0

    # save with consistent naming (GLODAP uses 'TCO2' and 'TAlk'; we store as 'CT' and 'AT')
    if glodap_var == 'TCO2':
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/CT.npy', var)
    elif glodap_var == 'TAlk':
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/AT.npy', var)
    else:
        np.save(data_path + 'GLODAPv2.2016b.MappedProduct/' + glodap_var + '.npy', var)

    print('\tregrid complete in ' + str(round(time.time() - start_time, 3)) + ' s')


def regrid_woa(data_path: str, woa_var: str, latitude: np.ndarray,
               longitude: np.ndarray, depth: np.ndarray,
               ocnmask: np.ndarray) -> None:
    """Regrid a World Ocean Atlas 2018 variable to the OCIM grid and save as .npy.

    Interpolates from the WOA18 lat/lon/depth grid to the OCIM2-48L grid,
    then fills NaNs with iterative neighbor averaging. WOA18 longitudes are
    in -180 to 180; they are converted to 0-360 before interpolating.

    Parameters
    ----------
    data_path : str
        Path to the data directory WOA18/
    woa_var : str
        Variable to regrid. One of: 'S' (salinity), 'T' (temperature),
        'Si' (silicate), 'P' (phosphate).
    latitude : np.ndarray
        1D array of OCIM2-48L latitude values [degrees N].
    longitude : np.ndarray
        1D array of OCIM2-48L longitude values [degrees E, 0-360].
    depth : np.ndarray
        1D array of OCIM2-48L depth values [m].
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    """
    if woa_var == 'S':
        data = xr.open_dataset(data_path + 'WOA18/woa18_decav81B0_s00_01.nc', decode_times=False)
    elif woa_var == 'T':
        data = xr.open_dataset(data_path + 'WOA18/woa18_decav81B0_t00_01.nc', decode_times=False)
    elif woa_var == 'Si':
        data = xr.open_dataset(data_path + 'WOA18/woa18_all_i00_01.nc', decode_times=False)
    elif woa_var == 'P':
        data = xr.open_dataset(data_path + 'WOA18/woa18_all_p00_01.nc', decode_times=False)
    else:
        print("WOA data not found. Choose from woa_var = 'S', 'T', 'Si', 'P'")
        return

    print('begin regrid of ' + woa_var)
    start_time = time.time()

    # convert WOA18 longitudes from -180-180 to 0-360 to match OCIM
    data['lon'] = (data['lon'] + 360) % 360
    data = data.sortby('lon')

    data_lat   = data['lat'].to_numpy()
    data_lon   = data['lon'].to_numpy()
    data_depth = data['depth'].to_numpy()

    # transpose to (lat, lon, depth) to match OCIM dimension order
    if woa_var == 'S':
        var = data.s_an.isel(time=0).transpose('lat', 'lon', 'depth').values
    elif woa_var == 'T':
        var = data.t_an.isel(time=0).transpose('lat', 'lon', 'depth').values
    elif woa_var == 'Si':
        var = data.i_an.isel(time=0).transpose('lat', 'lon', 'depth').values
    elif woa_var == 'P':
        var = data.p_an.isel(time=0).transpose('lat', 'lon', 'depth').values

    interp = RegularGridInterpolator(
        (data_lat, data_lon, data_depth), var, bounds_error=False, fill_value=None  # type: ignore[arg-type]
    )

    lat_grid, lon_grid, depth_grid = np.meshgrid(latitude, longitude, depth, indexing='ij')
    query_points = np.array([lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel()]).T
    var = interp(query_points).reshape(depth_grid.shape)

    var = inpaint_nans_3d(var, mask=ocnmask)

    if woa_var == 'S':
        np.save(data_path + 'WOA18/S.npy', var)
    elif woa_var == 'T':
        np.save(data_path + 'WOA18/T.npy', var)
    elif woa_var == 'Si':
        np.save(data_path + 'WOA18/Si.npy', var)
    elif woa_var == 'P':
        np.save(data_path + 'WOA18/P.npy', var)

    print('\tregrid complete in ' + str(round(time.time() - start_time, 3)) + ' s')


def regrid_ncep_noaa(data_path: str, ncep_var: str, latitude: np.ndarray,
                     longitude: np.ndarray, ocnmask: np.ndarray) -> None:
    """Regrid a NCEP/DOE or NOAA SST surface field to the OCIM grid and save as .npy.

    Computes the annual mean from the monthly climatology, interpolates to the
    OCIM2-48L surface grid, and fills NaNs. Wind speed is averaged over
    1994-2024; ice fraction and SST use the 1991-2020 long-term mean.

    Parameters
    ----------
    data_path : str
        Path to the data directory (must contain NCEP_DOE_Reanalysis_II/ and
        NOAA_Extended_Reconstruction_SST_V5/).
    ncep_var : str
        Variable to regrid. One of: 'icec' (ice concentration), 'wspd' (wind
        speed at 10 m), 'sst' (sea surface temperature).
    latitude : np.ndarray
        1D array of OCIM2-48L latitude values [degrees N].
    longitude : np.ndarray
        1D array of OCIM2-48L longitude values [degrees E].
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    """
    if ncep_var == 'icec':
        data = xr.open_dataset(data_path + 'NCEP_DOE_Reanalysis_II/icec.sfc.mon.ltm.1991-2020.nc')
        var = data.icec.mean(dim='time', skipna=True).values
    elif ncep_var == 'wspd':
        data = xr.open_dataset(data_path + 'NCEP_DOE_Reanalysis_II/wspd.10m.mon.mean.nc')
        # time indices 552-924 correspond to 1994-01-01 through 2024-01-01
        var = data.wspd.isel(time=slice(552, 924)).mean(dim='time', skipna=True).values
    elif ncep_var == 'sst':
        data = xr.open_dataset(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.mon.ltm.1991-2020.nc')
        var = data.sst.mean(dim='time', skipna=True).values
    else:
        print('NCEP/NOAA data not found.')
        return

    print('begin regrid of ' + ncep_var)
    start_time = time.time()

    data_lat = data['lat'].to_numpy()
    data_lon = data['lon'].to_numpy()

    interp = RegularGridInterpolator(
        (data_lat, data_lon), var, bounds_error=False, fill_value=None  # type: ignore[arg-type]
    )

    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing='ij')
    query_points = np.array([lat_grid.ravel(), lon_grid.ravel()]).T
    var = interp(query_points).reshape(lon_grid.shape)

    var = inpaint_nans_2d(var, mask=ocnmask[:, :, 0])

    if ncep_var == 'icec':
        np.save(data_path + 'NCEP_DOE_Reanalysis_II/icec.npy', var)
    elif ncep_var == 'wspd':
        np.save(data_path + 'NCEP_DOE_Reanalysis_II/wspd.npy', var)
    elif ncep_var == 'sst':
        np.save(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.npy', var)

    print('\tregrid complete in ' + str(round(time.time() - start_time, 3)) + ' s')


def regrid_cobalt(cobalt_vrbl, latitude: np.ndarray, longitude: np.ndarray,
                  depth: np.ndarray, ocnmask: np.ndarray,
                  data_path: str) -> None:
    """Regrid a COBALT biogeochemistry variable to the OCIM grid and save as .npy.

    Averages across the COBALT time dimension, converts longitudes from the
    COBALT convention (-300 to +60) to 0-360, interpolates to the OCIM2-48L
    grid, and fills NaNs.

    Parameters
    ----------
    cobalt_vrbl : xarray.DataArray
        COBALT model variable with dimensions (time, zl, yh, xh).
    latitude : np.ndarray
        1D array of OCIM2-48L latitude values [degrees N].
    longitude : np.ndarray
        1D array of OCIM2-48L longitude values [degrees E, 0-360].
    depth : np.ndarray
        1D array of OCIM2-48L depth values [m].
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    data_path : str
        Path where COBALT_regridded/ output directory exists.
    """
    cobalt_var = cobalt_vrbl.copy()
    var_name = cobalt_var.name
    print('begin regrid of ' + var_name)
    start_time = time.time()
    
    cobalt_var = cobalt_var.where(cobalt_var != 1e20)  # replace fill value with NaN
    cobalt_var = cobalt_var.mean(dim='time', skipna=True)
    
    # convert COBALT longitudes from -300-+60 to 0-360 to match OCIM
    cobalt_var['xh'] = (cobalt_var['xh'] + 360) % 360
    cobalt_var = cobalt_var.sortby('xh')

    cobalt_lat   = cobalt_var['yh'].to_numpy()
    cobalt_lon   = cobalt_var['xh'].to_numpy()
    cobalt_depth = cobalt_var['zl'].to_numpy()

    # transpose to (lat, lon, depth) to match OCIM dimension order
    var = cobalt_var.transpose('yh', 'xh', 'zl').values
    
    interp = RegularGridInterpolator(
        (cobalt_lat, cobalt_lon, cobalt_depth), var, method='linear',
        bounds_error=False, fill_value=None  # type: ignore[arg-type]
    )

    lat_grid, lon_grid, depth_grid = np.meshgrid(latitude, longitude, depth, indexing='ij')
    query_points = np.array([lat_grid.ravel(), lon_grid.ravel(), depth_grid.ravel()]).T
    var_interped = interp(query_points).reshape(depth_grid.shape)

    var_inpainted = inpaint_nans_3d(var_interped, mask=ocnmask)

    np.save(data_path + 'COBALT_regridded/' + var_name + '.npy', var_inpainted)
    print('\tregrid complete in ' + str(round(time.time() - start_time, 3)) + ' s')
