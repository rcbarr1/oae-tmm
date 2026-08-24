import warnings

import matplotlib as mpl
import numpy as np
import xarray as xr


def get_co2_scenario(scenario, time):
    """Return atmospheric CO2 [ppm] for a given SSP scenario at the requested times.

    Reads from the pyTRACE CO2TrajectoriesAdjusted.txt file (University of Melbourne
    greenhouse gas dataset) and interpolates linearly to the requested times.

    Parameters
    ----------
    scenario : str
        Emissions scenario. One of: 'none', 'ssp119', 'ssp126', 'ssp245', 'ssp370',
        'ssp370_lowNTCF', 'ssp434', 'ssp460', 'ssp534_OS', 'REMIND'.
        'none' holds CO2 fixed at the value of time[0] (historical extrapolation
        from 2012–2022); a warning is raised if time[0] > 2022.
    time : np.ndarray
        1D array of decimal years at which to evaluate CO2 [yr CE].

    Returns
    -------
    np.ndarray
        Atmospheric CO2 [ppm], same length as time.
    """
    scenarios = {
        'none': 1, 'ssp119': 2, 'ssp126': 3, 'ssp245': 4, 'ssp370': 5,
        'ssp370_lowNTCF': 6, 'ssp434': 7, 'ssp460': 8, 'ssp534_OS': 9,
        'REMIND': 10,
    }

    if scenario not in scenarios:
        raise ValueError(
            f"Invalid scenario {scenario!r}. Must be one of: {', '.join(scenarios.keys())}"
        )

    data_file = './pyTRACE/pyTRACE/data/CO2Trajectories.txt'
    data = np.loadtxt(data_file)
    co2_years = data[:, 0]
    co2_values = data[:, scenarios[scenario]]

    if scenario != 'none':
        return np.interp(time, co2_years, co2_values)

    if time[0] > 2022:
        warnings.warn(
            "'none' scenario chosen, but time > 2022 selected. "
            "CO2 is based on a linear extrapolation from 2012–2022."
        )
    return np.interp(time[0], co2_years, co2_values) * np.ones_like(time)


def broadcast_to_dataset(array, ds):
    """Wrap a numpy array in an xarray DataArray aligned to a dataset's lat/lon/depth coords.

    Convenience wrapper for the common pattern of broadcasting a static 3D numpy
    array (e.g. model volumes, GLODAP climatology) against a time-varying dataset
    so that xarray can apply arithmetic with automatic dimension alignment.

    Parameters
    ----------
    array : np.ndarray
        3D array of shape (n_lat, n_lon, n_depth).
    ds : xarray.Dataset
        Dataset whose lat, lon, and depth coordinates are used.

    Returns
    -------
    xarray.DataArray
        DataArray with dims ['latitude', 'longitude', 'depth'] and coordinates from ds.
    """
    return xr.DataArray(array, dims=['latitude', 'longitude', 'depth'],
                        coords={'latitude': ds.latitude, 'longitude': ds.longitude, 'depth': ds.depth})


def load_ocim_grid(data_path):
    """Load OCIM2-48L grid variables from the base dataset.

    Parameters
    ----------
    data_path : str
        Path to the data directory containing OCIM2_48L_base/.

    Returns
    -------
    dict with keys:
        'ocnmask'     : np.ndarray, shape (n_lat, n_lon, n_depth), 1=ocean 0=land
        'mldmask'     : np.ndarray, shape (n_lat, n_lon, n_depth), 1=ocean cell within MLD 0=otherwise
        'latitude'    : np.ndarray [°N]
        'longitude'   : np.ndarray [°E]
        'depth'       : np.ndarray [m below sea surface]
        'cell_volume' : np.ndarray [m³]
    """
    model_data = xr.open_dataset(data_path + 'OCIM2_48L_base/OCIM2_48L_base_data.nc')
    ocnmask     = model_data['ocnmask'].transpose('latitude', 'longitude', 'depth').to_numpy()
    cell_volume = model_data['vol'].transpose('latitude', 'longitude', 'depth').to_numpy()
    cell_area   = model_data['area'].transpose('latitude', 'longitude', 'depth').to_numpy()
    cell_top_depth_3d    = model_data['wz'].transpose('latitude', 'longitude', 'depth').to_numpy()
    cell_bottom_depth_3d = cell_top_depth_3d + cell_volume / cell_area
    mld     = model_data['mld'].transpose('latitude', 'longitude').to_numpy()
    mldmask = ((cell_bottom_depth_3d < mld[:, :, None]) * ocnmask).astype(int)
    grid = {
        'ocnmask':     ocnmask,
        'mldmask':     mldmask,
        'latitude':    model_data['tlat'].isel(depth=0, longitude=0).to_numpy(),
        'longitude':   model_data['tlon'].isel(depth=0, latitude=0).to_numpy(),
        'depth':       model_data['tz'].isel(longitude=0, latitude=0).to_numpy(),
        'cell_volume': cell_volume,
    }
    model_data.close()
    return grid


def load_glodap(data_path):
    """Load GLODAP v2.2016b mapped climatology fields.

    Parameters
    ----------
    data_path : str
        Path to the data directory containing GLODAPv2.2016b.MappedProduct/.

    Returns
    -------
    dict with keys:
        'CT_3d'          : np.ndarray, dissolved inorganic carbon [µmol kg-1]
        'AT_3d'          : np.ndarray, total alkalinity [µmol kg-1]
        'temperature_3d' : np.ndarray, temperature [ºC]
        'salinity_3d'    : np.ndarray, salinity [unitless]
        'silicate_3d'    : np.ndarray, silicate [µmol kg-1]
        'phosphate_3d'   : np.ndarray, phosphate [µmol kg-1]
    """
    glodap_path = data_path + 'GLODAPv2.2016b.MappedProduct/'
    return {
        'CT_3d':          xr.open_dataset(glodap_path + 'CT.nc')['CT'].values,
        'AT_3d':          xr.open_dataset(glodap_path + 'AT.nc')['AT'].values,
        'temperature_3d': xr.open_dataset(glodap_path + 'temperature.nc')['temperature'].values,
        'salinity_3d':    xr.open_dataset(glodap_path + 'salinity.nc')['salinity'].values,
        'silicate_3d':    xr.open_dataset(glodap_path + 'silicate.nc')['silicate'].values,
        'phosphate_3d':   xr.open_dataset(glodap_path + 'phosphate.nc')['phosphate'].values,
    }


def apply_style():
    """Apply standard plot style with black text.

    Sets font family, weight, and all text/tick/label colors globally via
    matplotlib rcParams.

    Returns
    -------
    textcolor : str
        Text color '#000000', for use in explicit color= kwargs.
    fontweight : str
        Font weight 'normal', for use in explicit fontweight= kwargs.
    """
    textcolor  = '#000000'
    fontweight = 'normal'
    mpl.rcParams['font.family']     = 'DejaVu Sans'
    mpl.rcParams['font.weight']     = fontweight
    mpl.rcParams['text.color']      = textcolor
    mpl.rcParams['axes.labelcolor'] = textcolor
    mpl.rcParams['xtick.color']     = textcolor
    mpl.rcParams['ytick.color']     = textcolor
    return textcolor, fontweight
