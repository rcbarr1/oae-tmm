"""
Experiment-time data loading functions for oae-tmm.

These functions load pre-processed data from disk at the start of each
experiment run. They expect that scripts/generate_input_data.py (which calls
oae_tmm.regrid) has already been run to produce the .nc files on disk.

All functions return plain dicts of numpy arrays so that experiments can
unpack only the fields they need.
"""

import numpy as np
import xarray as xr
import scipy.io as spio
from oae_tmm.grid import flatten, get_depth_idx


def load_mat(filename: str) -> dict:
    """Load a MATLAB .mat file and return its contents as a nested Python dict.

    scipy.io.loadmat returns MATLAB structs as opaque mat_struct objects.
    This function recursively converts them to plain Python dicts so that
    fields can be accessed with standard bracket notation.

    From https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

    Parameters
    ----------
    filename : str
        Path to the .mat file.

    Returns
    -------
    dict
        Contents of the .mat file with all MATLAB structs converted to dicts
        and cell arrays converted to lists.
    """
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def load_ocim(data_path: str) -> dict:
    """Load the OCIM2-48L transport matrix and grid metadata.

    Reads the sparse transport matrix from a MATLAB .mat file and the
    associated grid variables (ocean mask, coordinates, volumes) from a
    NetCDF file.

    Parameters
    ----------
    data_path : str
        Path to the data directory where OCIM2-48L matrix is stored in OCIM2_48L_base/

    Returns
    -------
    dict with keys:
        TR          : scipy sparse matrix (m x m), ocean transport operator
        ocnmask     : np.ndarray (n_lat, n_lon, n_depth), 1 = ocean, 0 = land
        mldmask     : np.ndarray (n_lat, n_lon, n_depth), 1 = ocean cell within MLD, 0 otherwise
        latitude    : np.ndarray (n_lat,), latitude of grid cell centers [degrees N]
        longitude   : np.ndarray (n_lon,), longitude of grid cell centers [degrees E]
        depth       : np.ndarray (n_depth,), depth of layer centers [m]
        cell_volume : np.ndarray (m,), grid cell volumes [m^3]
        cell_area   : np.ndarray (m,), horizontal area of each grid cell [m^2]
        pressure    : np.ndarray (m,), pressure at each ocean cell [dbar]
        mld         : np.ndarray (n_lat, n_lon), annual mean mixed layer depth [m]
        z1          : float, thickness of the surface model layer [m]
        surf_idx    : np.ndarray (n_surface_cells, 1), flat indices of surface ocean cells
        rho         : float, reference seawater density [kg m^-3]
    """
    # transport matrix (Holzer et al. 2021)
    mat = load_mat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
    TR = mat['TR']

    # grid metadata
    model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
    ocnmask   = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()
    latitude  = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()     # degrees N
    longitude = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()      # degrees E
    depth     = model_data['tz'].isel(longitude=0, latitude=0).to_numpy()    # m below sea surface
    cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy()  # m^3
    cell_area   = model_data['area'].transpose('latitude', 'longitude', 'depth').to_numpy()  # m^2

    # wz: depth of the top of each layer (W-grid interfaces); wz[0]=0 (sea surface),
    # z1 = wz[1] = top of cell 1 = bottom of cell 0 = surface layer thickness
    cell_top_depth_3d    = model_data['wz'].transpose('latitude', 'longitude', 'depth').to_numpy()  # m
    cell_bottom_depth_3d = cell_top_depth_3d + cell_volume / cell_area  # m
    z1 = cell_top_depth_3d[0, 0, 1]
    mld = model_data['mld'].transpose('latitude', 'longitude').to_numpy()  # m
    mldmask = ((cell_bottom_depth_3d < mld[:, :, None]) * ocnmask).astype(int)

    surf_idx = get_depth_idx(ocnmask, 0)  # indices of surface grid cells in flattened ocean vector
    rho = 1025  # reference seawater density [kg m^-3]

    # pressure [dbar ≈ m]: broadcast depth to 3D, then flatten
    depth_3d = np.broadcast_to(depth[np.newaxis, np.newaxis, :], ocnmask.shape)

    return {
        'TR':          TR,
        'ocnmask':     ocnmask,
        'mldmask':     mldmask,
        'latitude':    latitude,
        'longitude':   longitude,
        'depth':       depth,
        'cell_volume': flatten(cell_volume, ocnmask),
        'cell_area':   flatten(cell_area, ocnmask),
        'pressure':    flatten(depth_3d, ocnmask),
        'mld':         mld,
        'z1':          z1,
        'surf_idx':    surf_idx,
        'rho':         rho,
    }


def load_glodap(data_path: str, ocnmask: np.ndarray) -> dict:
    """Load pre-regridded GLODAPv2 fields from disk.

    Reads the .nc files produced by regrid.regrid_glodap() and flattens each
    field to a 1D ocean-only vector using the OCIM2-48L ocean mask.

    Parameters
    ----------
    data_path : str
        Path to the data directory where data is stored in GLODAPv2.2016b.MappedProduct/
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.

    Returns
    -------
    dict with keys:
        temperature : np.ndarray (m,), temperature [degrees C]
        salinity    : np.ndarray (m,), salinity [unitless]
        CT          : np.ndarray (m,), dissolved inorganic carbon [µmol kg^-1]
        AT          : np.ndarray (m,), total alkalinity [µmol kg^-1]
        silicate    : np.ndarray (m,), silicate [µmol kg^-1]
        phosphate   : np.ndarray (m,), phosphate [µmol kg^-1]
    """
    base = data_path + 'GLODAPv2.2016b.MappedProduct/'
    result = {}
    for var in ('temperature', 'salinity', 'CT', 'AT', 'silicate', 'phosphate'):
        with xr.open_dataset(base + f'{var}.nc') as ds:
            result[var] = flatten(ds[var].transpose('latitude', 'longitude', 'depth').values, ocnmask)
    return result


def load_ncep_noaa(data_path: str, ocnmask: np.ndarray) -> dict:
    """Load pre-regridded NCEP/DOE and NOAA surface fields from disk.

    Reads the .nc files produced by regrid.regrid_ncep_noaa(). Each 2D
    surface field (n_lat, n_lon) is placed into the surface layer of a 3D
    array (zero at all other depths) and flattened to a 1D ocean-only vector.
    Subsurface values are 0, not NaN, so the vectors can be used directly in
    build_A_matrix without further masking.

    Parameters
    ----------
    data_path : str
        Path to the data directory where data is stored in NCEP_DOE_Reanalysis_II/ and
        NOAA_Extended_Reconstruction_SST_V5/
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.

    Returns
    -------
    dict with keys:
        f_ice : np.ndarray (m,), annual mean ice fraction [0-1], 0 at depth
        wspd  : np.ndarray (m,), annual mean wind speed at 10 m [m s^-1], 0 at depth
        sst   : np.ndarray (m,), annual mean sea surface temperature [degrees C], 0 at depth
    """
    def _surf_to_flat(field_2d):
        field_3d = np.zeros(ocnmask.shape)
        field_3d[:, :, 0] = field_2d
        return flatten(field_3d, ocnmask)

    return {
        'f_ice': _surf_to_flat(xr.open_dataset(data_path + 'NCEP_DOE_Reanalysis_II/icec.nc')['icec'].transpose('latitude', 'longitude').values),
        'wspd':  _surf_to_flat(xr.open_dataset(data_path + 'NCEP_DOE_Reanalysis_II/wspd.nc')['wspd'].transpose('latitude', 'longitude').values),
        'sst':   _surf_to_flat(xr.open_dataset(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.nc')['sst'].transpose('latitude', 'longitude').values),
    }
