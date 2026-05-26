"""
Grid utility functions for oae-tmm.

These are pure array-transformation functions: they reshape, fill, or index
numpy arrays. They do not load data from disk, run chemistry, or call the
solver. Because they are stateless (output depends only on inputs), they are
easy to unit-test with small synthetic arrays.

OCIM2-48L array conventions used throughout:
  - 3D arrays have shape (n_lat, n_lon, n_depth) in Fortran (column-major) order
  - Flattened 1D arrays omit land grid cells; ocean-only cells are retained
    in the same Fortran order
  - ocnmask: integer array of shape (n_lat, n_lon, n_depth), 1 = ocean, 0 = land
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve, uniform_filter
from tqdm import tqdm


def flatten(field_3d: np.ndarray, ocnmask: np.ndarray) -> np.ndarray:
    """Flatten a 3D field to a 1D ocean-only vector.

    Ravels the array in Fortran (column-major) order and removes all land
    grid cells, matching the indexing convention of the OCIM2-48L transport
    matrix. The inverse operation is make_3d().

    Parameters
    ----------
    field_3d : np.ndarray
        3D array of shape (n_lat, n_lon, n_depth).
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.

    Returns
    -------
    np.ndarray
        1D array of length m, where m = number of ocean grid cells.
    """
    return field_3d.flatten(order='F')[ocnmask.flatten(order='F').astype(bool)]


def make_3d(field_flat: np.ndarray, ocnmask: np.ndarray) -> np.ndarray:
    """Expand a 1D ocean-only vector back to a 3D array.

    Inverse of flatten(). Land grid cells are filled with NaN. Uses Fortran
    (column-major) ordering to match the OCIM2-48L convention.

    Parameters
    ----------
    field_flat : np.ndarray
        1D array of length m (ocean grid cells only).
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.

    Returns
    -------
    np.ndarray
        3D array of shape (n_lat, n_lon, n_depth) with NaN at land cells.
    """
    field_3d = np.full(ocnmask.shape, np.nan)
    flat_mask = ocnmask.flatten(order='F').astype(bool)

    field_3d_flat = field_3d.flatten(order='F')
    field_3d_flat[flat_mask] = field_flat
    field_3d = field_3d_flat.reshape(ocnmask.shape, order='F')

    return field_3d


def smooth_tracer(field_flat: np.ndarray, ocnmask: np.ndarray) -> np.ndarray:
    """Smooth a tracer field by averaging each cell with its 3D neighborhood.

    Applies a 5x5x5 uniform (box) filter across all three spatial dimensions.
    Only ocean cells contribute to each neighborhood average — land cells are
    zeroed out before filtering and the count of valid ocean neighbors is tracked
    separately so the average is not diluted by land. The longitude axis is
    padded with a wrap to correctly handle the periodic boundary of the global
    ocean (i.e., the dateline).

    Parameters
    ----------
    field_flat : np.ndarray
        1D ocean-only tracer vector of length m.
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.

    Returns
    -------
    np.ndarray
        1D ocean-only smoothed tracer vector of length m.
    """
    field_3d = make_3d(field_flat, ocnmask)
    field_3d = np.nan_to_num(field_3d, nan=0.0)  # land cells = 0 for averaging

    # pad longitude axis with a wrap so the filter sees the correct neighbors
    # at the dateline (0/360 boundary)
    field_pad = np.pad(field_3d, ((0, 0), (2, 2), (0, 0)), mode='wrap')
    ocnmask_pad = np.pad(ocnmask, ((0, 0), (2, 2), (0, 0)), mode='wrap')

    # sum tracer values in each 5x5x5 neighborhood, weighted by ocean mask
    field_sum_pad = uniform_filter(field_pad * ocnmask_pad, size=(5, 5, 5), mode='constant', cval=0)

    # count ocean cells in each neighborhood to compute true average
    ocnmask_pad_count = uniform_filter(ocnmask_pad.astype(float), size=(5, 5, 5), mode='constant', cval=0)

    ocnmask_pad_count[ocnmask_pad_count == 0] = np.nan  # avoid division by zero at isolated cells
    field_smooth_pad = field_sum_pad / ocnmask_pad_count

    # remove longitude padding and return as 1D
    field_smooth_3d = field_smooth_pad[:, 2:-2, :]
    return flatten(field_smooth_3d, ocnmask)


def get_depth_idx(ocnmask: np.ndarray, depth_level: int) -> np.ndarray:
    """Return the flat indices of ocean cells at the surface depth level.

    Identifies all ocean grid cells at depth index 0 (the surface layer) and
    returns their positions within the flattened 1D ocean-only vector produced
    by flatten(). Note: the depth_level parameter is accepted for API
    consistency but currently the function always returns surface (depth
    index 0) indices.

    Parameters
    ----------
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    depth_level : int
        Depth index of interest (currently unused; surface layer is always
        returned).

    Returns
    -------
    np.ndarray
        2D array of shape (n_surface_cells, 1) containing the flat indices of
        surface ocean cells within the 1D ocean-only vector.
    """
    surf_mask = np.zeros_like(ocnmask)
    surf_mask[:, :, 0] = 1  # select only the top depth layer

    ocn_surf_mask = ocnmask * surf_mask
    return np.argwhere(flatten(ocn_surf_mask, ocnmask) == 1)


def inpaint_nans_3d(array_3d: np.ndarray, iterations: int = 10,
                    mask: np.ndarray = None) -> np.ndarray:
    """Fill NaN values in a 3D array by iterative neighbor averaging.

    Uses a 6-connected stencil (one neighbor in each of +/-x, +/-y, +/-z) to
    iteratively replace NaN cells with the mean of their valid neighbors.
    NaNs are initialized to the global array mean before the first iteration
    so every cell has a starting value. If a land mask is provided, land cells
    are excluded from the fill and restored to NaN at the end.

    This approach is used instead of scipy's interpolation because the OCIM
    grid is irregular (ocean-only), and simple iterative averaging avoids
    extrapolating into land or across bathymetric discontinuities.

    Parameters
    ----------
    array_3d : np.ndarray
        3D array of shape (n_lat, n_lon, n_depth) with NaN values to fill.
    iterations : int, optional
        Number of averaging iterations. More iterations propagates fill values
        further from valid data. Default is 10.
    mask : np.ndarray, optional
        Boolean or integer array of shape (n_lat, n_lon, n_depth); True/1 =
        ocean cell that can receive a fill value. Land cells (False/0) remain
        NaN after filling.

    Returns
    -------
    np.ndarray
        3D array with NaN values filled at ocean cells.
    """
    interpolated = array_3d.copy()

    # 6-connected stencil: one neighbor in each axis direction, no diagonals
    kernel = np.zeros((3, 3, 3))
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1  # +/-depth
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1  # +/-lat
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1  # +/-lon

    nan_mask = np.isnan(interpolated)
    interpolated[nan_mask] = np.nanmean(interpolated)  # initialize NaNs to global mean

    if mask is not None:
        land_mask = ~mask.astype(bool)  # True where land
    else:
        land_mask = np.zeros_like(array_3d, dtype=bool)

    for _ in range(iterations):
        valid = ~np.isnan(interpolated)
        neighbor_sum = convolve(np.nan_to_num(interpolated), kernel, mode='wrap')
        neighbor_count = convolve(valid.astype(float), kernel, mode='wrap')

        with np.errstate(invalid='ignore', divide='ignore'):
            new_vals = neighbor_sum / neighbor_count

        # only update cells that were originally NaN, are ocean, and have at least one valid neighbor
        update_mask = nan_mask & ~land_mask & (neighbor_count > 0)
        interpolated[update_mask] = new_vals[update_mask]

    if mask is not None:
        interpolated[~mask.astype(bool)] = np.nan  # restore land to NaN

    return interpolated


def inpaint_nans_2d(array_2d: np.ndarray, iterations: int = 10,
                    mask: np.ndarray = None) -> np.ndarray:
    """Fill NaN values in a 2D array by iterative neighbor averaging.

    2D analogue of inpaint_nans_3d(), using a 4-connected stencil (+/-x, +/-y).
    Used for surface-only fields such as wind speed, ice fraction, and SST.
    A tqdm progress bar is shown because this function can be slow for large
    grids with many NaN cells (e.g., sea-ice-covered regions).

    Parameters
    ----------
    array_2d : np.ndarray
        2D array of shape (n_lat, n_lon) with NaN values to fill.
    iterations : int, optional
        Number of averaging iterations. Default is 10.
    mask : np.ndarray, optional
        Boolean or integer array of shape (n_lat, n_lon); True/1 = ocean cell
        that can receive a fill value.

    Returns
    -------
    np.ndarray
        2D array with NaN values filled at ocean cells.
    """
    interpolated = array_2d.copy()

    # 4-connected stencil: up, down, left, right — no diagonals
    kernel = np.zeros((3, 3))
    kernel[0, 1] = 1  # up
    kernel[2, 1] = 1  # down
    kernel[1, 0] = 1  # left
    kernel[1, 2] = 1  # right

    nan_mask = np.isnan(interpolated)
    interpolated[nan_mask] = np.nanmean(interpolated)  # initialize NaNs to global mean

    if mask is not None:
        land_mask = ~mask.astype(bool)  # True where land
    else:
        land_mask = np.zeros_like(array_2d, dtype=bool)

    for _ in tqdm(range(iterations), desc="inpainting"):
        valid = ~np.isnan(interpolated)
        neighbor_sum = convolve(np.nan_to_num(interpolated), kernel, mode='wrap')
        neighbor_count = convolve(valid.astype(float), kernel, mode='wrap')

        with np.errstate(invalid='ignore', divide='ignore'):
            new_vals = neighbor_sum / neighbor_count

        update_mask = nan_mask & ~land_mask & (neighbor_count > 0)
        interpolated[update_mask] = new_vals[update_mask]

    if mask is not None:
        interpolated[~mask.astype(bool)] = np.nan  # restore land to NaN

    return interpolated


def find_mld(model_lat: np.ndarray, model_lon: np.ndarray, ocnmask: np.ndarray,
             MLD_da: np.ndarray, latm: np.ndarray, lonm: np.ndarray,
             type_flag: int) -> np.ndarray:
    """Interpolate the Holte et al. mixed layer depth climatology to the OCIM grid.

    Reads a monthly climatology of mixed layer depth (MLD) from Holte et al.
    and interpolates it onto the OCIM2-48L lat/lon grid. Can return either the
    maximum monthly MLD (type_flag=0) or the mean monthly MLD (type_flag=1)
    across the 12-month climatology. NaNs remaining after interpolation (e.g.,
    in sea-ice regions) are filled using inpaint_nans_2d().

    The Holte et al. data uses longitudes in the range -180 to 180, while OCIM
    uses 0 to 360, so longitudes are converted before interpolating.

    Parameters
    ----------
    model_lat : np.ndarray
        1D array of OCIM2-48L latitude values [degrees N].
    model_lon : np.ndarray
        1D array of OCIM2-48L longitude values [degrees E, 0-360].
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    MLD_da : np.ndarray
        Mixed layer depth from Holte et al. density algorithm; shape
        (n_months, n_lon_holte, n_lat_holte) [m].
    latm : np.ndarray
        2D latitude grid from Holte et al. [degrees N].
    lonm : np.ndarray
        2D longitude grid from Holte et al. [degrees E, -180 to 180].
    type_flag : int
        0 = return maximum monthly MLD across all months.
        1 = return mean monthly MLD across all months.

    Returns
    -------
    np.ndarray
        2D array of shape (n_lat, n_lon) with MLD values [m] on the OCIM grid.
    """
    if type_flag == 0:
        MLDs = np.nanmax(MLD_da, axis=0)   # deepest MLD across all months
    elif type_flag == 1:
        MLDs = np.nanmean(MLD_da, axis=0)  # average MLD across all months
    else:
        print('ERROR: type_flag should be specified as 0 or 1')

    # convert Holte et al. longitudes from -180-180 to 0-360 to match OCIM
    lonm[lonm <= 0] += 360

    # sort ascending along longitude so RegularGridInterpolator requirements are met
    lonm_1d = lonm[:, 0]
    sort_idx = np.argsort(lonm_1d)
    lonm_1d = lonm_1d[sort_idx]
    lonm = lonm[sort_idx, :]
    MLDs = MLDs[sort_idx, :]

    # pad lon and lat boundaries so interpolation does not fail at the edges
    # (wrapping the globe and extending slightly past the poles)
    lonm = np.vstack([lonm[-1, :] - 360, lonm, lonm[0, :] + 360])
    latm = np.vstack([latm[-1, :], latm, latm[0, :]])
    MLDs = np.vstack([MLDs[-1, :], MLDs, MLDs[0, :]])

    latm = np.hstack([latm[:, 0:1] - 1, latm, latm[:, -1:] + 1])
    lonm = np.hstack([lonm[:, 0:1], lonm, lonm[:, -1:]])
    MLDs = np.hstack([MLDs[:, 0:1], MLDs, MLDs[:, -1:]])

    interp = RegularGridInterpolator(
        (latm[:, 0], lonm[0, :]), MLDs, bounds_error=False, fill_value=None
    )

    # build query points on the OCIM lat/lon grid
    lat, lon = np.meshgrid(model_lat, model_lon, indexing='ij')
    query_points = np.array([lat.ravel(), lon.ravel()]).T
    var = interp(query_points).reshape(lon.shape)

    # fill any remaining NaNs (common in sea-ice regions) using neighbor averaging
    interp_MLDs = inpaint_nans_2d(var, mask=ocnmask[:, :, 0].astype(bool))

    return interp_MLDs
