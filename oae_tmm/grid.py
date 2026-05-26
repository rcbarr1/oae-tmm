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
from scipy.ndimage import convolve
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


def get_depth_idx(ocnmask: np.ndarray, depth_level: int) -> np.ndarray:
    """Return the flat indices of ocean cells at a given depth level.

    Identifies all ocean grid cells at the specified depth index and returns
    their positions within the flattened 1D ocean-only vector produced by
    flatten(). All experiments currently call this with depth_level=0 to
    select surface grid cells.

    Parameters
    ----------
    ocnmask : np.ndarray
        Integer mask of shape (n_lat, n_lon, n_depth); 1 = ocean, 0 = land.
    depth_level : int
        Index into the depth dimension of ocnmask. 0 = surface layer.

    Returns
    -------
    np.ndarray
        2D array of shape (n_cells, 1) containing the flat indices of ocean
        cells at depth_level within the 1D ocean-only vector.
    """
    depth_mask = np.zeros_like(ocnmask)
    depth_mask[:, :, depth_level] = 1

    ocn_depth_mask = ocnmask * depth_mask
    return np.argwhere(flatten(ocn_depth_mask, ocnmask) == 1)


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

