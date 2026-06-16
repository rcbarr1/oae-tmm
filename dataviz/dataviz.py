import copy
import warnings

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import xarray as xr

from oae_tmm.grid import make_3d


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

    data_file = './pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt'
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


def plot_surface2d(lats, lons, variable, vmin, vmax, cmap, title):
    """Plot a filled-contour surface map of a 2D lat/lon field.

    Zero-valued cells are masked and rendered in black (intended for land cells).

    Parameters
    ----------
    lats, lons : np.ndarray
        1D arrays of latitudes [°N] and longitudes [°E].
    variable : np.ndarray
        2D array of shape (n_lat, n_lon).
    vmin, vmax : float
        Colorscale limits.
    cmap : str or matplotlib.colors.Colormap
        Colormap name or object.
    title : str
        Plot title.
    """
    variable_masked = np.ma.masked_where(variable == 0, variable)

    if not isinstance(cmap, mcolors.Colormap):
        cmap = mpl.colormaps[cmap].copy()
    else:
        cmap = copy.copy(cmap)
    cmap.set_bad(color='black')

    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.linspace(vmin - 0.1, vmax, 100)
    cntr = ax.contourf(lons, lats, variable_masked, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = fig.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10), 2))
    ax.set_xlabel('longitude (ºE)')
    ax.set_ylabel('latitude (ºN)')
    ax.set_title(title)
    ax.set_xlim((0, 360))
    ax.set_ylim((-90, 90))


def plot_surface3d(lats, lons, variable, depth_level, vmin, vmax, cmap, title,
                   logscale=False, lon_lims=None):
    """Plot a filled-contour surface map of one depth level from a 3D lat/lon/depth field.

    Parameters
    ----------
    lats, lons : np.ndarray
        1D arrays of latitudes [°N] and longitudes [°E].
    variable : np.ndarray
        3D array of shape (n_lat, n_lon, n_depth).
    depth_level : int
        Depth index to plot.
    vmin, vmax : float
        Colorscale limits.
    cmap : str
        Matplotlib colormap name.
    title : str
        Plot title.
    logscale : bool, optional
        If True, use LogNorm colorscale (vmin/vmax are ignored). Default False.
    lon_lims : tuple of float, optional
        (lon_min, lon_max) x-axis limits. If None, defaults to [0, 360].

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

    if logscale:
        cntr = ax.contourf(lons, lats, variable[:, :, depth_level],
                           norm=LogNorm(), cmap=cmap)
        fig.colorbar(cntr, ax=ax)
    else:
        levels = np.linspace(vmin - 1e-7, vmax, 100)
        cntr = ax.contourf(lons, lats, variable[:, :, depth_level],
                           levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        c = fig.colorbar(cntr, ax=ax)
        c.set_ticks(np.round(np.linspace(vmin, vmax, 10), 2))
    ax.set_xlabel('longitude (ºE)')
    ax.set_ylabel('latitude (ºN)')
    ax.set_title(title)
    ax.set_ylim((-90, 90))
    ax.set_xlim((0, 360) if lon_lims is None else lon_lims)

    return fig


def plot_longitude3d(lats, depths, variable, longitude, vmin, vmax, cmap, title):
    """Plot a filled-contour latitude–depth section at a fixed longitude index.

    Parameters
    ----------
    lats : np.ndarray
        1D array of latitudes [°N].
    depths : np.ndarray
        1D array of depth levels [m].
    variable : np.ndarray
        3D array of shape (n_lat, n_lon, n_depth).
    longitude : int
        Longitude index to slice.
    vmin, vmax : float
        Colorscale limits.
    cmap : str
        Matplotlib colormap name.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.linspace(vmin - 1e-7, vmax, 100)
    cntr = ax.contourf(lats, depths, variable[:, longitude, :].T,
                       levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = fig.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10), 2))
    ax.invert_yaxis()
    ax.set_xlabel('latitude (ºN)')
    ax.set_ylabel('depth (m)')
    ax.set_title(title)
    ax.set_xlim((-90, 90))
    ax.set_ylim((depths.max(), 0))


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


def make_surf_animation(variable, colorbar_label, latitude, longitude, t, nt,
                        vmin, vmax, cmap, filename):
    """Animate a surface (depth=0) field over time and save to an mp4 file.

    Parameters
    ----------
    variable : xarray.DataArray
        4D DataArray with dims (time, latitude, longitude, depth); depth=0 slice is plotted.
    colorbar_label : str
        Label for the colorbar.
    latitude, longitude : np.ndarray
        1D arrays of OCIM2-48L latitudes [°N] and longitudes [°E].
    t : np.ndarray
        1D array of time values (decimal years) for frame titles.
    nt : int
        Number of frames to render.
    vmin, vmax : float
        Colorscale limits.
    cmap : str
        Matplotlib colormap name.
    filename : str
        Output mp4 path.
    """
    levels = np.linspace(vmin, vmax, 100)
    fig, ax = plt.subplots(figsize=(10, 7))

    cntr = ax.contourf(longitude, latitude, variable.isel(time=0).values[:, :, 0],
                       levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cntr, ax=ax, label=colorbar_label)
    ax.set_xlabel('Longitude (ºE)')
    ax.set_ylabel('Latitude (ºN)')
    ax.set_title(f't = {t[0]:.3f} yr')

    def update_frame(idx):
        ax.clear()
        ax.contourf(longitude, latitude, variable.isel(time=idx).values[:, :, 0],
                    levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude (ºE)')
        ax.set_ylabel('Latitude (ºN)')
        ax.set_title(f't = {t[idx]:.3f} yr')
        return []

    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    ani.save(filename, writer=animation.FFMpegWriter(fps=10), dpi=200)


def make_section_animation(variable, colorbar_label, depth, latitude, t, nt,
                           vmin, vmax, cmap, filename):
    """Animate a latitude–depth section at lon index 90 (≈181°E) over time and save to mp4.

    Parameters
    ----------
    variable : xarray.DataArray
        4D DataArray with dims (time, latitude, longitude, depth).
    colorbar_label : str
        Label for the colorbar.
    depth : np.ndarray
        1D array of OCIM2-48L depth levels [m].
    latitude : np.ndarray
        1D array of OCIM2-48L latitudes [°N].
    t : np.ndarray
        1D array of time values (decimal years) for frame titles.
    nt : int
        Number of frames to render.
    vmin, vmax : float
        Colorscale limits.
    cmap : str
        Matplotlib colormap name.
    filename : str
        Output mp4 path.
    """
    levels = np.linspace(vmin, vmax, 100)
    fig, ax = plt.subplots(figsize=(10, 7))

    cntr = ax.contourf(latitude, depth, variable.isel(time=0).values[:, 90, :].T,
                       levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cntr, ax=ax, label=colorbar_label)
    ax.invert_yaxis()
    ax.set_xlabel('Latitude (ºN)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f't = {t[0]:.3f} yr at 181ºE')

    def update_frame(idx):
        ax.clear()
        ax.contourf(latitude, depth, variable.isel(time=idx).values[:, 90, :].T,
                    levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xlabel('Latitude (ºN)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f't = {t[idx]:.3f} yr at 181ºE')
        return []

    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    ani.save(filename, writer=animation.FFMpegWriter(fps=10), dpi=200)
