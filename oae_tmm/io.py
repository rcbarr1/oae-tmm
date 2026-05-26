"""
Experiment-time data loading functions for oae-tmm.

These functions load pre-processed data from disk at the start of each
experiment run. They expect that scripts/generate_input_data.py (which calls
oae_tmm.regrid) has already been run to produce the .npy files on disk.

All functions return plain dicts of numpy arrays so that experiments can
unpack only the fields they need.
"""

import numpy as np
import xarray as xr
import scipy.io as spio
import geopandas as gpd
from shapely.geometry import Point
from oae_tmm.grid import get_depth_idx


def load_mat(filename: str) -> dict:
    """Load a MATLAB .mat file and return its contents as a nested Python dict.

    scipy.io.loadmat returns MATLAB structs as opaque mat_struct objects.
    This function recursively converts them to plain Python dicts so that
    fields can be accessed with standard bracket notation.

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
    NetCDF file. This block of loading code is identical across all
    experiments and is encapsulated here to avoid repetition.

    Parameters
    ----------
    data_path : str
        Path to the data directory (must contain OCIM2_48L_base/).

    Returns
    -------
    dict with keys:
        TR              : scipy sparse matrix (m x m), ocean transport operator
        ocnmask         : np.ndarray (n_lat, n_lon, n_depth), 1 = ocean, 0 = land
        model_lat       : np.ndarray (n_lat,), latitude of grid cell centers [degrees N]
        model_lon       : np.ndarray (n_lon,), longitude of grid cell centers [degrees E]
        model_depth     : np.ndarray (n_depth,), depth of layer centers [m]
        model_vols      : np.ndarray (n_lat, n_lon, n_depth), grid cell volumes [m^3]
        grid_cell_depth : np.ndarray (n_lat, n_lon, n_depth), depth of layer bottoms [m]
        z1              : float, thickness of the surface model layer [m]
        surf_idx        : np.ndarray (n_surface_cells, 1), flat indices of surface ocean cells
        rho             : float, reference seawater density [kg m^-3]
    """
    # transport matrix (Holzer et al. 2021)
    mat = load_mat(data_path + 'OCIM2_48L_base/OCIM2_48L_base_transport.mat')
    TR = mat['TR']

    # grid metadata
    model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
    ocnmask         = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()
    model_lat       = model_data['tlat'].isel(depth=0, longitude=0).to_numpy()     # degrees N
    model_lon       = model_data['tlon'].isel(depth=0, latitude=0).to_numpy()      # degrees E
    model_depth     = model_data['tz'].isel(longitude=0, latitude=0).to_numpy()    # m below surface
    model_vols      = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy()  # m^3

    # wz gives the bottom depth of each layer; z1 is the thickness of the surface layer
    grid_cell_depth = model_data['wz'].transpose('latitude', 'longitude', 'depth').to_numpy()  # m
    z1 = grid_cell_depth[0, 0, 1]

    surf_idx = get_depth_idx(ocnmask, 0)
    rho = 1025  # reference seawater density [kg m^-3]

    return {
        'TR': TR,
        'ocnmask': ocnmask,
        'model_lat': model_lat,
        'model_lon': model_lon,
        'model_depth': model_depth,
        'model_vols': model_vols,
        'grid_cell_depth': grid_cell_depth,
        'z1': z1,
        'surf_idx': surf_idx,
        'rho': rho,
    }


def load_glodap(data_path: str) -> dict:
    """Load pre-regridded GLODAPv2 fields from disk.

    Reads the .npy files produced by regrid.regrid_glodap(). All fields are
    on the OCIM2-48L grid (n_lat, n_lon, n_depth) with land cells as NaN.

    Parameters
    ----------
    data_path : str
        Path to the data directory (must contain GLODAPv2.2016b.MappedProduct/).

    Returns
    -------
    dict with keys:
        T_3D   : np.ndarray, temperature [degrees C]
        S_3D   : np.ndarray, salinity [unitless]
        DIC_3D : np.ndarray, dissolved inorganic carbon [µmol kg^-1]
        AT_3D  : np.ndarray, total alkalinity [µmol kg^-1]
        Si_3D  : np.ndarray, silicate [µmol kg^-1]
        P_3D   : np.ndarray, phosphate [µmol kg^-1]
    """
    base = data_path + 'GLODAPv2.2016b.MappedProduct/'
    return {
        'T_3D':   np.load(base + 'temperature.npy'),
        'S_3D':   np.load(base + 'salinity.npy'),
        'DIC_3D': np.load(base + 'DIC.npy'),
        'AT_3D':  np.load(base + 'TA.npy'),
        'Si_3D':  np.load(base + 'silicate.npy'),
        'P_3D':   np.load(base + 'PO4.npy'),
    }


def load_ncep_noaa(data_path: str) -> dict:
    """Load pre-regridded NCEP/DOE and NOAA surface fields from disk.

    Reads the .npy files produced by regrid.regrid_ncep_noaa(). All fields
    are annual means on the OCIM2-48L surface grid (n_lat, n_lon).

    Parameters
    ----------
    data_path : str
        Path to the data directory (must contain NCEP_DOE_Reanalysis_II/ and
        NOAA_Extended_Reconstruction_SST_V5/).

    Returns
    -------
    dict with keys:
        f_ice_2D : np.ndarray (n_lat, n_lon), annual mean ice fraction [0-1]
        wspd_2D  : np.ndarray (n_lat, n_lon), annual mean wind speed at 10 m [m s^-1]
        sst_2D   : np.ndarray (n_lat, n_lon), annual mean sea surface temperature [degrees C]
    """
    return {
        'f_ice_2D': np.load(data_path + 'NCEP_DOE_Reanalysis_II/icec.npy'),
        'wspd_2D':  np.load(data_path + 'NCEP_DOE_Reanalysis_II/wspd.npy'),
        'sst_2D':   np.load(data_path + 'NOAA_Extended_Reconstruction_SST_V5/sst.npy'),
    }


def build_lme_masks(shp_path: str, ocnmask: np.ndarray,
                    lats: np.ndarray, lons: np.ndarray) -> tuple:
    """Build binary masks for each of the 66 Large Marine Ecosystems (LMEs).

    Reads the LME shapefile, determines which OCIM2-48L surface grid cells
    fall within each LME polygon, and returns a mask per LME. Several manual
    corrections are applied to fix holes and overlaps that arise from
    converting the LME boundaries to the coarse OCIM grid.

    Parameters
    ----------
    shp_path : str
        Path to the LMEs66.shp shapefile.
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    lats : np.ndarray
        1D array of OCIM2-48L latitude values [degrees N].
    lons : np.ndarray
        1D array of OCIM2-48L longitude values [degrees E, 0-360].

    Returns
    -------
    lme_masks : dict
        Keys are LME_NUMBER (int), values are boolean arrays of shape
        (n_lon, n_lat) that are True where the surface cell belongs to that LME.
    lme_id_to_name : dict
        Maps LME_NUMBER (int) to LME_NAME (str).
    """
    lmes = gpd.read_file(shp_path)
    if lmes.crs != "EPSG:4326":
        lmes = lmes.to_crs(epsg=4326)

    # convert lons from 0-360 to -180-180 for spatial intersection with shapefile
    lons_for_test = ((lons + 180) % 360) - 180
    lon_grid, lat_grid = np.meshgrid(lons_for_test, lats)

    points = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

    lme_id_grid = np.zeros(lon_grid.shape, dtype=int).T
    lme_id_to_name = {}
    lme_masks = {}

    for idx, row in lmes.iterrows():
        lme_id = int(row["LME_NUMBER"])
        name = row["LME_NAME"]
        mask_flat = points_gdf.within(row.geometry)
        mask = mask_flat.to_numpy().reshape(lat_grid.shape).T
        mask = np.logical_and(mask, ocnmask[:, :, 0].astype(bool))

        # manual corrections for grid cells that overlap LME boundaries or
        # fall in gaps created by the coarse OCIM grid resolution
        if lme_id == 22:
            mask[177, 76] = False
        if lme_id == 3:
            mask[119, 63] = True
        elif lme_id == 9:
            mask[152:154, 69] = True
        elif lme_id == 11:
            mask[131, 53] = True
        elif lme_id == 12:
            mask[138, 56] = True
        elif lme_id == 13:
            mask[141, 39] = True
            mask[144, 36] = True
            mask[144, 31] = True
        elif lme_id == 17:
            mask[152, 48] = True
        elif lme_id == 18:
            mask[146, 80] = True
            mask[148, 79] = True
            mask[153, 81] = True
            mask[154, 77] = True
        elif lme_id == 20:
            mask[7, 84] = True
            mask[9, 85] = True
            mask[30, 86] = True
        elif lme_id == 25:
            mask[175, 65] = True
            mask[176, 67] = True
        elif lme_id == 26:
            mask[176, 64] = True
            mask[177:180, 63:65] = True
            mask[0:7, 63:65] = True
            mask[5, 62] = True
        elif lme_id == 27:
            mask[173, 59] = True
        elif lme_id == 28:
            mask[0, 48] = True
        elif lme_id == 32:
            mask[25, 51] = True
        elif lme_id == 34:
            mask[47, 53] = True
        elif lme_id == 36:
            mask[52, 50] = True
        elif lme_id == 37:
            mask[60, 48] = True
        elif lme_id == 39:
            mask[65, 39] = True
        elif lme_id == 43:
            mask[69, 27] = True
        elif lme_id == 45:
            mask[57, 34] = True
        elif lme_id == 54:
            mask[89, 81] = True
            mask[95, 77] = True
            mask[101, 81] = True
        elif lme_id == 58:
            mask[47, 85:87] = True
            mask[45, 86] = True
            mask[32, 80] = True
            mask[33, 81] = True
        elif lme_id == 59:
            mask[169, 78] = True
        elif lme_id == 61:
            mask[0, 9] = True
            mask[1:15, 10] = True
            mask[16, 11] = True
            mask[17, 10] = True
            mask[20:25, 11] = True
            mask[25:29, 12] = True
            mask[29, 11] = True
            mask[34, 10] = True
            mask[38, 10] = True
            mask[41, 11] = True
            mask[42:46, 12] = True
            mask[51:72, 12] = True
            mask[72:79, 11] = True
            mask[81:84, 10] = True
            mask[84, 8] = True
            mask[81, 6] = True
            mask[102, 6] = True
            mask[104:107, 7] = True
            mask[110:129, 8] = True
            mask[168, 8] = True
            mask[170:176, 9] = True
            mask[176:179, 10] = True
        elif lme_id == 66:
            mask[123:125, 83] = True
            mask[127, 85] = True
            mask[133:137, 86] = True
            mask[142, 85] = True
            mask[144, 84] = True
            mask[144, 87] = True
            mask[157, 87] = True

        if np.any(mask):
            lme_id_grid[mask] = lme_id
            lme_masks[lme_id] = mask
            lme_id_to_name[lme_id] = name

    return lme_masks, lme_id_to_name
