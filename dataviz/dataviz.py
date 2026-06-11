#%%
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Point

from oae_tmm.grid import make_3d as make_3D


def get_co2_scenario(scenario, times):
    """Return atmospheric CO2 [ppm] for a given SSP scenario at the requested times.

    Data source: pyTRACE CO2TrajectoriesAdjusted.txt (University of Melbourne
    greenhouse gas dataset). Scenarios: 'none', 'ssp119', 'ssp126', 'ssp245',
    'ssp370', 'ssp370_lowNTCF', 'ssp434', 'ssp460', 'ssp534_OS'.
    """
    scenarios = {'none': 1, 'ssp119': 2, 'ssp126': 3, 'ssp245': 4, 'ssp370': 5,
                 'ssp370_lowNTCF': 6, 'ssp434': 7, 'ssp460': 8, 'ssp534_OS': 9}

    if scenario not in scenarios:
        raise ValueError(f"Invalid value: {scenario!r}. Must be one of: {', '.join(scenarios.keys())}")

    data_file = './pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt'
    data = np.loadtxt(data_file)
    CO2_data_years = data[:, 0]
    CO2_data = data[:, scenarios[scenario]]

    if scenario != 'none':
        atmospheric_CO2 = np.interp(times, CO2_data_years, CO2_data)
    else:
        if times[0] >= 2020:
            warnings.warn("'none' scenario chosen, but time > 2022 selected. "
                          "Canth is based on a linear extrapolation from 2012-2022 in this case.")
        atmospheric_CO2 = np.interp(times[0], CO2_data_years, CO2_data) * np.ones_like(times)

    return atmospheric_CO2


def plot_surface2d(lats, lons, variable, vmin, vmax, cmap, title):

    # mask out zero values
    variable_masked = np.ma.masked_where(variable == 0, variable)

    # create colormap copy, set masked to black
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color='black')

    # main plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca()
    levels = np.linspace(vmin-0.1, vmax, 100)
    cntr = plt.contourf(lons, lats, variable_masked, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))

    plt.xlabel('longitude (ºE)')
    plt.ylabel('latitude (ºN)')
    plt.title(title)
    plt.xlim([0, 360]), plt.ylim([-90,90])


def plot_surface3d(lats, lons, variable, depth_level, vmin, vmax, cmap, title, logscale=None, lon_lims=None):
    fig = plt.figure(figsize=(10,7), dpi=200)
    ax = fig.gca()

    if logscale:
        cntr = plt.contourf(lons, lats, variable[:, :, depth_level], norm=LogNorm(), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        levels = np.linspace(vmin-1e-7, vmax, 100)
        cntr = plt.contourf(lons, lats, variable[:, :, depth_level], levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))
    plt.xlabel('longitude (ºE)')
    plt.ylabel('latitude (ºN)')
    plt.title(title)

    if lon_lims==None:
        plt.xlim([0, 360]), plt.ylim([-90,90])
    else:
        plt.ylim([-90,90])
        plt.xlim(lon_lims)

    return fig


def plot_longitude3d(lats, depths, variable, longitude, vmin, vmax, cmap, title):
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca()
    levels = np.linspace(vmin-1e-7, vmax, 100)
    cntr = plt.contourf(lats, depths, variable[:, longitude, :].T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    c = plt.colorbar(cntr, ax=ax)
    c.set_ticks(np.round(np.linspace(vmin, vmax, 10),2))
    ax.invert_yaxis()
    plt.xlabel('latitude (ºN)')
    plt.ylabel('depth (m)')
    plt.title(title)
    plt.xlim([-90, 90]), plt.ylim([depths.max(), 0])


def plot_lmes(lme_masks, ocnmask, lats, lons):
    # convert lons to -180 to 180 for plotting
    lons_shifted = np.where(lons > 180, lons - 360, lons)

    # create an array to hold lme ids
    id_grid = np.full((len(lons), len(lats)), np.nan)
    centers = []

    for idx, (lme_id, mask) in enumerate(lme_masks.items(), start=1):
        id_grid[mask] = int(lme_id)

        if np.any(mask):
            lat_center = np.mean(lats[np.any(mask, axis=0)])
            lon_center = np.mean(lons[np.any(mask, axis=1)])
            if lon_center > 180:
                lon_center -= 360
            centers.append((lon_center, lat_center, int(lme_id)))

    fig = plt.figure(figsize=(14, 8), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()

    ax.pcolormesh(
        lons_shifted, lats, ocnmask[0, :, :].T,
        transform=ccrs.PlateCarree(),
        cmap='Greys_r', shading='nearest'
    )

    hsv_colors = [(i / len(lme_masks), 0.75, 0.85) for i in range(len(lme_masks)+4)]
    rgb_colors = [mcolors.hsv_to_rgb(c) for c in hsv_colors]
    random.Random(48).shuffle(rgb_colors)
    cmap = mcolors.ListedColormap(rgb_colors)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(0.5, len(lme_masks) + 4 + 1.5, 1),
        ncolors=len(lme_masks)+4
    )

    ax.pcolormesh(
        lons_shifted, lats, id_grid.T,
        transform=ccrs.PlateCarree(),
        cmap=cmap, alpha=0.8, norm=norm, shading='nearest'
    )

    for lon_c, lat_c, idx in centers:
        ax.text(lon_c, lat_c, str(idx),
                transform=ccrs.PlateCarree(),
                fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    plt.title("Large Marine Ecosystems (62 out of 66 can be represented on OCIM grid)")
    plt.show()


def make_surf_animation(variable, colorbar_label, model_lat, model_lon, t, nt, vmin, vmax, cmap, filename):
    fig, ax = plt.subplots(figsize=(10,7))

    cntr = ax.contourf(model_lon, model_lat,
                       variable.isel(time=0).values[:,:,0],
                       levels=np.linspace(vmin, vmax, 100),
                       cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cntr, ax=ax, label=colorbar_label)
    ax.set_xlabel('Longitude (ºE)')
    ax.set_ylabel('Latitude (ºN)')
    ax.set_title('t = ' + f'{t[0]:.3f}' + ' yr')

    def update_frame(idx):
        ax.clear()
        ax.contourf(model_lon, model_lat,
                    variable.isel(time=idx).values[:,:,0],
                    levels=np.linspace(vmin, vmax, 100),
                    cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude (ºE)')
        ax.set_ylabel('Latitude (ºN)')
        ax.set_title('t = ' + f'{t[idx]:.3f}' + ' yr')
        return []

    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(filename, writer=writer, dpi=200)


def make_surf_animation_pH(pH, colorbar_label, model_lat, model_lon, t, nt, ocnmask, vmin, vmax, cmap, filename):
    fig, ax = plt.subplots(figsize=(10,7))

    pH_3D = make_3D(pH[0], ocnmask)
    cntr = ax.contourf(model_lon, model_lat,
                       pH_3D[:,:,0],
                       levels=np.linspace(vmin, vmax, 100),
                       cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cntr, ax=ax, label=colorbar_label)
    ax.set_xlabel('Longitude (ºE)')
    ax.set_ylabel('Latitude (ºN)')
    ax.set_title('t = ' + f'{t[0]:.3f}' + ' yr')

    def update_frame(idx):
        ax.clear()
        pH_3D = make_3D(pH[idx], ocnmask)
        ax.contourf(model_lon, model_lat,
                    pH_3D[:,:,0],
                    levels=np.linspace(vmin, vmax, 100),
                    cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude (ºE)')
        ax.set_ylabel('Latitude (ºN)')
        ax.set_title('t = ' + f'{t[idx]:.3f}' + ' yr')
        return []

    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(filename, writer=writer, dpi=200)


def make_section_animation(variable, colorbar_label, model_depth, model_lat, t, nt, vmin, vmax, cmap, filename):
    fig, ax = plt.subplots(figsize=(10,7))

    cntr = ax.contourf(model_lat, model_depth,
                       variable.isel(time=0).values[:,90,:].T,
                       levels=np.linspace(vmin, vmax, 100),
                       cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cntr, ax=ax, label=colorbar_label)
    ax.invert_yaxis()
    ax.set_xlabel('Latitude (ºN)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('t = ' + f'{t[0]:.3f}' + 'yr at 181ºE')

    def update_frame(idx):
        ax.clear()
        ax.contourf(model_lat, model_depth,
                    variable.isel(time=idx).values[:,90,:].T,
                    levels=np.linspace(vmin, vmax, 100),
                    cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xlabel('Latitude (ºN)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('t = ' + f'{t[idx]:.3f}' + ' yr at 181 ºE')
        return []

    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(filename, writer=writer, dpi=200)


def make_section_animation_pH(pH, colorbar_label, model_depth, model_lat, t, nt, ocnmask, vmin, vmax, cmap, filename):
    fig, ax = plt.subplots(figsize=(10,7))

    pH_3D = make_3D(pH[0], ocnmask)
    cntr = ax.contourf(model_lat, model_depth,
                       pH_3D[:,90,:].T,
                       levels=np.linspace(vmin, vmax, 100),
                       cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cntr, ax=ax, label=colorbar_label)
    ax.invert_yaxis()
    ax.set_xlabel('Latitude (ºN)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('t = ' + f'{t[0]:.3f}' + ' yr at 181ºE')

    def update_frame(idx):
        pH_3D = make_3D(pH[idx], ocnmask)
        ax.clear()
        ax.contourf(model_lat, model_depth,
                    pH_3D[:,90,:].T,
                    levels=np.linspace(vmin, vmax, 100),
                    cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xlabel('Latitude (ºN)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('t = ' + f'{t[idx]:.3f}' + ' yr at 181 ºE')
        return []

    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=100, blit=False)
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(filename, writer=writer, dpi=200)
