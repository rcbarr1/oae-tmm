"""
Regression test: compare src/expXX.py output against experiments/expXX.py.

Both files must be pre-computed with identical simulation parameters (same
times, scenario, start_year, start_CDR) before running this script.

Usage:
    python tests/refactor_test.py

Edit OLD_OUTPUT and NEW_OUTPUT at the top to point to the pre-computed files.

Note on time coordinate: src/expXX.py writes initial conditions at t[1] (the
first loop iteration) rather than t[0], so its time axis starts one step later
than experiments/expXX.py, which writes at t[0]. The time coordinate check will
fail unless both outputs were sliced to a common time range before comparison.
This slicing has been implemented in the code below.
"""
#%%
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

OLD_OUTPUT = './outputs/exp23_TEST_none_000.nc'                    # from src/exp23.py
NEW_OUTPUT = './outputs/exp23_2026-06-09_test_none_000.nc'  # from experiments/exp23.py

RTOL = 1e-4   # relative tolerance for tracer comparison (float32 precision)
ATOL = 1e-3   # absolute tolerance [µmol kg-1]; handles near-zero values where
              # rtol alone would flag tiny absolute differences as large relative errors

VIZ_LON_IDX = 90   # ~180°E (Pacific) for latitude-depth section plots

COORDS     = ['lat', 'lon', 'depth']
# time is checked separately after slicing away the IC record (src/ writes IC at t[1], new at t[0])
TRACERS_4D = ['delDIC', 'delAT', 'DIC_added', 'AT_added']
TRACERS_1D = ['delxCO2', 'xCO2_added']


def _pass(msg): print(f'  PASS  {msg}')
def _fail(msg): print(f'  FAIL  {msg}')


def check_coordinates(old, new):
    print('\n--- Coordinates ---')
    passed = True

    # time: slice away IC record before comparing (src/ writes IC at t[1], new code at t[0])
    ov = old['time'].values[1:]
    nv = new['time'].values[1:]
    if ov.shape != nv.shape:
        _fail(f'time: shape mismatch after slicing — old {ov.shape}, new {nv.shape}')
        passed = False
    elif not np.array_equal(ov, nv):
        _fail(f'time: values differ after slicing (max |diff| = {np.max(np.abs(ov - nv)):.3e})')
        passed = False
    else:
        _pass(f'time: exact match after slicing  shape={ov.shape}')

    for coord in COORDS:
        for label, ds in [('old', old), ('new', new)]:
            if coord not in ds.coords:
                _fail(f'{coord}: missing from {label} output')
                passed = False

        if not passed:
            continue

        ov, nv = old[coord].values, new[coord].values
        if ov.shape != nv.shape:
            _fail(f'{coord}: shape mismatch — old {ov.shape}, new {nv.shape}')
            passed = False
        elif not np.array_equal(ov, nv):
            _fail(f'{coord}: values differ (max |diff| = {np.max(np.abs(ov - nv)):.3e})')
            passed = False
        else:
            _pass(f'{coord}: exact match  shape={ov.shape}')
    return passed


def check_nan_structure(old, new):
    """NaN marks land cells in 4D fields; both outputs should share the same pattern."""
    print('\n--- NaN structure (4D tracers) ---')
    passed = True
    for var in TRACERS_4D:
        old_nan = np.isnan(old[var].values)
        new_nan = np.isnan(new[var].values)
        if np.array_equal(old_nan, new_nan):
            _pass(f'{var}: NaN pattern matches  ({np.sum(~old_nan)} ocean values)')
        else:
            _fail(f'{var}: NaN pattern differs at {np.sum(old_nan != new_nan)} cells')
            passed = False

    for var in TRACERS_1D:
        old_nan = np.any(np.isnan(old[var].values))
        new_nan = np.any(np.isnan(new[var].values))
        if old_nan or new_nan:
            _fail(f'{var}: unexpected NaNs (old={old_nan}, new={new_nan})')
            passed = False
        else:
            _pass(f'{var}: no NaNs')
    return passed


def check_tracers(old, new):
    print('\n--- Tracer values ---')
    passed = True
    for var in TRACERS_4D + TRACERS_1D:
        if 'time' in old[var].dims:
            ov = old[var].isel(time=slice(1, None)).values
        else:
            ov = old[var].values

        if 'time' in new[var].dims:
            nv = new[var].isel(time=slice(1, None)).values
        else:
            nv = new[var].values

        if ov.shape != nv.shape:
            _fail(f'{var}: shape mismatch — old {ov.shape}, new {nv.shape}')
            passed = False
            continue

        # compare ocean cells only (where both are finite)
        finite = np.isfinite(ov) & np.isfinite(nv)
        if not np.any(finite):
            _fail(f'{var}: no finite values to compare')
            passed = False
            continue

        o, n = ov[finite], nv[finite]
        abs_diff = np.abs(n - o)
        # relative diff: fall back to absolute when denominator is near zero
        rel_diff = abs_diff / np.where(np.abs(o) > 1e-30, np.abs(o), 1.0)

        stats = (f'errors: mean abs={np.mean(abs_diff):.2e}, max abs={np.max(abs_diff):.2e}'
                 f'  |  mean rel={np.mean(rel_diff):.2e}, max rel={np.max(rel_diff):.2e}')

        try:
            np.testing.assert_allclose(n, o, rtol=RTOL, atol=ATOL, equal_nan=False)
            _pass(f'{var}: within rtol={RTOL}, atol={ATOL}')
        except AssertionError:
            _fail(f'{var}: exceeds rtol={RTOL}, atol={ATOL}')
            passed = False
        print(stats)
    return passed


def _contour_sf(ax, lats, lons, data3d, vmin, vmax, title):
    """Surface (depth=0) filled-contour on ax; data3d shape (lat, lon, depth)."""
    var = np.ma.masked_invalid(data3d[:, :, 0])
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.6')
    cf = ax.contourf(lons, lats, var, levels=np.linspace(vmin, vmax, 60),
                     cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('lon (°E)', fontsize=11)
    ax.set_ylabel('lat (°N)', fontsize=11)
    ax.tick_params(labelsize=10)
    return cf


def _contour_sec(ax, lats, depths, data3d, vmin, vmax, title):
    """Latitude-depth section at VIZ_LON_IDX; data3d shape (lat, lon, depth)."""
    var = np.ma.masked_invalid(data3d[:, VIZ_LON_IDX, :].T)   # → (depth, lat)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.6')
    cf = ax.contourf(lats, depths, var, levels=np.linspace(vmin, vmax, 60),
                     cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('lat (°N)', fontsize=11)
    ax.set_ylabel('depth (m)', fontsize=11)
    ax.tick_params(labelsize=10)
    return cf


def visualize_tracers(old, new):
    """Surface maps, latitude-depth sections, and 1D time series for all tracers."""
    lats   = old['lat'].values
    lons   = old['lon'].values
    depths = old['depth'].values
    times  = old['time'].values[1:]   # drop IC record (matches slicing in check_tracers)

    # ── 4D tracers: 2 rows (first / last timestep) × 4 cols (surf old, surf new, sect old, sect new)
    # constrained_layout reserves colorbar space automatically, preventing overlap
    for var in TRACERS_4D:
        ov = old[var].isel(time=slice(1, None))
        nv = new[var].isel(time=slice(1, None))

        fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
        fig.suptitle(var, fontsize=18, fontweight='bold')

        for row, (label, tidx) in enumerate([('first', 0), ('last', -1)]):
            o3d = ov.isel(time=tidx).values
            n3d = nv.isel(time=tidx).values
            t_yr = times[tidx]
            vm = max(float(np.nanmax(np.abs(o3d))), 1e-10)

            cf_sf = _contour_sf(axes[row, 0], lats, lons, o3d, -vm, vm,
                                f'surface  old  {label}  t={t_yr:.1f}')
            _contour_sf(axes[row, 1], lats, lons, n3d, -vm, vm,
                        f'surface  new  {label}  t={t_yr:.1f}')
            cf_sc = _contour_sec(axes[row, 2], lats, depths, o3d, -vm, vm,
                                 f'section (~180°E)  old  {label}  t={t_yr:.1f}')
            _contour_sec(axes[row, 3], lats, depths, n3d, -vm, vm,
                         f'section (~180°E)  new  {label}  t={t_yr:.1f}')

            fig.colorbar(cf_sf, ax=axes[row, :2].tolist(), shrink=0.8, label='µmol kg⁻¹')
            fig.colorbar(cf_sc, ax=axes[row, 2:].tolist(), shrink=0.8, label='µmol kg⁻¹')

        plt.show()

    # ── 1D tracers: time series for old and new
    fig, axes = plt.subplots(1, len(TRACERS_1D), figsize=(12, 5), constrained_layout=True)
    for ax, var in zip(axes, TRACERS_1D):
        ov = old[var].isel(time=slice(1, None)).values
        nv = new[var].isel(time=slice(1, None)).values
        ax.plot(times, ov, label='old', linewidth=2)
        ax.plot(times, nv, label='new', linewidth=2, linestyle='--')
        ax.set_title(var, fontsize=14)
        ax.set_xlabel('year', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11)
    plt.show()


if __name__ == '__main__':
    print(f'old: {OLD_OUTPUT}')
    print(f'new: {NEW_OUTPUT}')

    old = xr.open_dataset(OLD_OUTPUT)
    new = xr.open_dataset(NEW_OUTPUT)

    coord_ok  = check_coordinates(old, new)
    nan_ok    = check_nan_structure(old, new)
    tracer_ok = check_tracers(old, new)
    visualize_tracers(old, new)

    old.close()
    new.close()

    print('\n--- Summary ---')
    if coord_ok and nan_ok and tracer_ok:
        print('All checks passed.')
    else:
        print('Some checks failed — see above.')
        sys.exit(1)

# %%
