"""
NetCDF output helpers for oae-tmm.

Provides two public functions for streaming output (one timestep at a time):

  open_simulation_output  — creates and initializes a NETCDF4 file, returns
      the open Dataset. Caller writes to it incrementally with
      write_simulation_step and is responsible for closing it.

  write_simulation_step   — appends one timestep to an already-open Dataset.
      Designed to be called inside the time-stepping loop so that only the
      current c and q vectors (not the full history) need to be in memory.

Typical usage:

    with output.open_simulation_output(path, lat, lon, depth, ocnmask) as ds:
        for i, (c, q, t) in enumerate(simulation):
            output.write_simulation_step(ds, c, q * dt, t, ocnmask)
            if i % 20 == 0:
                ds.sync()   # flush to disk periodically

Output variable conventions
---------------------------
All variables use float32 to keep file sizes manageable.

  delDIC, delAT       [µmol kg⁻¹]  perturbation state (from c)
  DIC_added, AT_added [µmol kg⁻¹]  integrated source per timestep (from q·dt)
  delxCO2, xCO2_added [ppm]        atmospheric CO₂ perturbation (from c and
                                   q·dt, converted from mol mol^-1 * 1e6)

Chunking: one time slice per chunk (1, n_lat, n_lon, n_depth) so that
reading a single timestep never loads the entire file. zlib complevel=4.
"""

import numpy as np
from netCDF4 import Dataset

from oae_tmm.grid import make_3d


def open_simulation_output(
    path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    depth: np.ndarray,
    ocnmask: np.ndarray,
    attrs: dict = None,
) -> Dataset:
    """Create and initialize a NETCDF4 simulation output file.

    Sets up all dimensions, coordinate variables, and tracer variables.
    The time dimension is unlimited so timesteps can be appended one at a
    time with write_simulation_step. Returns the open Dataset — use as a
    context manager so the file is closed on exit:

        with open_simulation_output(path, ...) as ds:
            write_simulation_step(ds, c, q_dt, time, ocnmask)

    Parameters
    ----------
    path : str
        Output file path (including .nc extension).
    lat : np.ndarray
        1D array of OCIM2-48L latitudes [°N].
    lon : np.ndarray
        1D array of OCIM2-48L longitudes [°E].
    depth : np.ndarray
        1D array of OCIM2-48L depth levels [m].
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    attrs : dict, optional
        Experiment metadata written as global attributes, e.g.
        {'scenario': 'ssp245', 'AT_added_umol_kg': 1.0}.

    Returns
    -------
    netCDF4.Dataset
        Open, writable Dataset. Call ds.close() or use as a context manager.
    """
    n_lat, n_lon, n_depth = ocnmask.shape

    ds = Dataset(path, 'w', format='NETCDF4')
    if attrs:
        ds.setncatts(attrs)

    ds.createDimension('time', None)   # unlimited
    ds.createDimension('lat', n_lat)
    ds.createDimension('lon', n_lon)
    ds.createDimension('depth', n_depth)

    tv = ds.createVariable('time', 'f8', ('time',))
    tv.units = 'year'

    lav = ds.createVariable('lat', 'f4', ('lat',))
    lav.units = 'degrees_north'
    lav[:] = lat

    lov = ds.createVariable('lon', 'f4', ('lon',))
    lov.units = 'degrees_east'
    lov[:] = lon

    dv = ds.createVariable('depth', 'f4', ('depth',))
    dv.units = 'meters'
    dv[:] = depth

    chunk4d = (1, n_lat, n_lon, n_depth)
    kw4d = dict(zlib=True, complevel=4, chunksizes=chunk4d)
    chunk1d = (1,)
    kw1d = dict(zlib=True, complevel=4, chunksizes=chunk1d)

    v = ds.createVariable('delDIC',     'f4', ('time', 'lat', 'lon', 'depth'), **kw4d)
    v.units = 'umol kg-1'
    v = ds.createVariable('DIC_added',  'f4', ('time', 'lat', 'lon', 'depth'), **kw4d)
    v.units = 'umol kg-1'
    v = ds.createVariable('delAT',      'f4', ('time', 'lat', 'lon', 'depth'), **kw4d)
    v.units = 'umol kg-1'
    v = ds.createVariable('AT_added',   'f4', ('time', 'lat', 'lon', 'depth'), **kw4d)
    v.units = 'umol kg-1'
    v = ds.createVariable('delxCO2',    'f4', ('time',), **kw1d)
    v.units = 'ppm'
    v = ds.createVariable('xCO2_added', 'f4', ('time',), **kw1d)
    v.units = 'ppm'

    return ds


def write_simulation_step(
    ds: Dataset,
    c: np.ndarray,
    q_dt: np.ndarray,
    time: float,
    ocnmask: np.ndarray,
) -> None:
    """Append one timestep to an open simulation output file.

    Partitions c and q_dt into their tracer components (∆xCO2, ∆DIC, ∆AT),
    reshapes flat ocean-only vectors to 3D with make_3d, and appends to the
    unlimited time dimension.

    Parameters
    ----------
    ds : netCDF4.Dataset
        Open dataset returned by open_simulation_output().
    c : np.ndarray
        State vector [∆xCO2, ∆DIC (m), ∆AT (m)], shape (2m+1,).
    q_dt : np.ndarray
        Source/sink vector * timestep [tracer units], shape (2m+1,).
        Pass q * dt (not the raw flux rate q) so stored values are integrated
        amounts per timestep.
    time : float
        Calendar year for this timestep [decimal years CE].
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    """
    m = (c.shape[0] - 1) // 2
    i = len(ds.variables['time'])   # append at next position in unlimited dim

    ds.variables['time'][i]        = time
    ds.variables['delxCO2'][i]     = np.float32(c[0] * 1e6)
    ds.variables['xCO2_added'][i]  = np.float32(q_dt[0] * 1e6)
    ds.variables['delDIC'][i]      = make_3d(c[1:(m+1)], ocnmask).astype('float32')
    ds.variables['DIC_added'][i]   = make_3d(q_dt[1:(m+1)], ocnmask).astype('float32')
    ds.variables['delAT'][i]       = make_3d(c[(m+1):], ocnmask).astype('float32')
    ds.variables['AT_added'][i]    = make_3d(q_dt[(m+1):], ocnmask).astype('float32')
