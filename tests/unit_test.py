"""
Unit tests for oae_tmm modules: grid, chemistry, transport, output, and base.

Tests cover pure functions with no I/O or external file dependencies.
All tests use small synthetic arrays to verify numerical correctness,
shape invariants, and edge-case handling.

Coverage:
  grid.flatten / grid.make_3d     — roundtrip, Fortran order, land cells → NaN
  grid.get_depth_idx              — correct ocean-cell indices per depth level
  grid.inpaint_nans_2d            — NaN filling, land mask respected, edge cases
  grid.inpaint_nans_3d            — NaN filling in 3D, land mask respected, edge cases
  chemistry.schmidt_number        — polynomial values, shape, ValueError on unknown gas
  chemistry.calc_piston_velocity  — zero wind → zero k, U² proportionality, known value
  transport.build_A_matrix        — block shape/format, zero air-sea blocks when k=0
  output.write_simulation_step    — ×1e6 unit conversion, c-vector slicing, float32 storage
  base.BaseExperiment._output_path — single-file passthrough, multi-file suffix formatting

Usage:
    python tests/unit_test.py
"""
#%%
import os
import sys
import tempfile
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import sparse

from netCDF4 import Dataset

from oae_tmm.grid import flatten, make_3d, get_depth_idx, inpaint_nans_2d, inpaint_nans_3d
from oae_tmm.chemistry import schmidt_number, calc_piston_velocity
from oae_tmm.transport import build_A_matrix
from oae_tmm.output import open_simulation_output, write_simulation_step
from experiments.base import BaseExperiment, ExperimentConfig


def _pass(msg): print(f'  PASS  {msg}')
def _fail(msg): print(f'  FAIL  {msg}')


def _check(cond, msg):
    if cond:
        _pass(msg)
        return True
    _fail(msg)
    return False


# ── grid.flatten / grid.make_3d ──────────────────────────────────────────────

def test_flatten_make3d():
    print('\n--- grid.flatten / grid.make_3d ---')
    passed = True

    # 3x2x3 mask with 4 ocean cells at known positions
    # F-order flat indices: (0,0,0)→0, (1,0,0)→1, (0,0,1)→6, (1,1,1)→10
    # After removing land cells, ocean-only order is: (0,0,0), (1,0,0), (0,0,1), (1,1,1)
    mask = np.zeros((3, 2, 3), dtype=int)
    mask[0, 0, 0] = 1
    mask[1, 0, 0] = 1
    mask[0, 0, 1] = 1
    mask[1, 1, 1] = 1

    arr = np.full((3, 2, 3), np.nan)
    arr[0, 0, 0] = 1.0
    arr[1, 0, 0] = 2.0
    arr[0, 0, 1] = 3.0
    arr[1, 1, 1] = 4.0

    flat = flatten(arr, mask)
    passed &= _check(flat.shape == (4,), f'flatten output shape: expected (4,), got {flat.shape}')
    passed &= _check(np.array_equal(flat, [1.0, 2.0, 3.0, 4.0]),
                     f'flatten values in F-order: expected [1,2,3,4], got {flat}')

    # Roundtrip: flatten then make_3d recovers ocean values; land cells → NaN
    reconstructed = make_3d(flat, mask)
    passed &= _check(reconstructed.shape == (3, 2, 3),
                     f'make_3d output shape: expected (3,2,3), got {reconstructed.shape}')
    passed &= _check(reconstructed[0, 0, 0] == 1.0, 'make_3d: ocean value at (0,0,0) correct')
    passed &= _check(reconstructed[1, 0, 0] == 2.0, 'make_3d: ocean value at (1,0,0) correct')
    passed &= _check(reconstructed[0, 0, 1] == 3.0, 'make_3d: ocean value at (0,0,1) correct')
    passed &= _check(reconstructed[1, 1, 1] == 4.0, 'make_3d: ocean value at (1,1,1) correct')
    passed &= _check(np.all(np.isnan(reconstructed[mask == 0])),
                     'make_3d: all land cells are NaN')

    # All-ocean mask: flatten matches F-order ravel
    mask_all = np.ones((2, 2, 2), dtype=int)
    arr_all = np.arange(8.0).reshape((2, 2, 2))
    passed &= _check(np.array_equal(flatten(arr_all, mask_all), arr_all.flatten(order='F')),
                     'flatten with all-ocean mask matches F-order ravel')

    # Single ocean cell edge case
    mask_one = np.zeros((2, 2, 2), dtype=int)
    mask_one[1, 0, 1] = 1
    arr_one = np.zeros((2, 2, 2))
    arr_one[1, 0, 1] = 99.0
    flat_one = flatten(arr_one, mask_one)
    passed &= _check(flat_one.shape == (1,) and flat_one[0] == 99.0,
                     'single ocean cell: flatten returns array of length 1 with correct value')

    return passed


# ── grid.get_depth_idx ───────────────────────────────────────────────────────

def test_get_depth_idx():
    print('\n--- grid.get_depth_idx ---')
    passed = True

    # Same mask as above — 4 ocean cells
    # Ocean-only vector: [cell0=(0,0,0), cell1=(1,0,0), cell2=(0,0,1), cell3=(1,1,1)]
    mask = np.zeros((3, 2, 3), dtype=int)
    mask[0, 0, 0] = 1
    mask[1, 0, 0] = 1
    mask[0, 0, 1] = 1
    mask[1, 1, 1] = 1

    idx_0 = get_depth_idx(mask, 0)
    passed &= _check(idx_0.shape == (2, 1),
                     f'get_depth_idx(depth=0) shape: expected (2,1), got {idx_0.shape}')
    passed &= _check(set(idx_0.ravel()) == {0, 1},
                     f'get_depth_idx(depth=0) indices: expected {{0,1}}, got {set(idx_0.ravel())}')

    idx_1 = get_depth_idx(mask, 1)
    passed &= _check(idx_1.shape == (2, 1),
                     f'get_depth_idx(depth=1) shape: expected (2,1), got {idx_1.shape}')
    passed &= _check(set(idx_1.ravel()) == {2, 3},
                     f'get_depth_idx(depth=1) indices: expected {{2,3}}, got {set(idx_1.ravel())}')

    # No ocean cells at depth 2 → empty result
    idx_2 = get_depth_idx(mask, 2)
    passed &= _check(idx_2.shape == (0, 1),
                     f'get_depth_idx(depth=2) shape: expected (0,1), got {idx_2.shape}')

    return passed


# ── grid.inpaint_nans_2d ─────────────────────────────────────────────────────

def test_inpaint_nans_2d():
    print('\n--- grid.inpaint_nans_2d ---')
    passed = True

    # NaN center cell surrounded by uniform 1.0 → should fill to ~1.0
    arr = np.ones((3, 3))
    arr[1, 1] = np.nan
    result = inpaint_nans_2d(arr, iterations=20)
    passed &= _check(not np.isnan(result[1, 1]), 'NaN surrounded by 1.0 gets filled')
    passed &= _check(np.isclose(result[1, 1], 1.0, atol=1e-3),
                     f'filled value close to 1.0: got {result[1, 1]:.6f}')

    # Non-NaN cells should be unchanged
    non_nan_mask = np.ones((3, 3), dtype=bool)
    non_nan_mask[1, 1] = False
    passed &= _check(np.allclose(result[non_nan_mask], 1.0), 'non-NaN cells unchanged')

    # Land mask: land cell remains NaN even when surrounded by valid ocean
    arr2 = np.ones((3, 3))
    arr2[1, 1] = np.nan
    ocean_mask = np.ones((3, 3), dtype=int)
    ocean_mask[1, 1] = 0
    result2 = inpaint_nans_2d(arr2, iterations=20, mask=ocean_mask)
    passed &= _check(np.isnan(result2[1, 1]), 'land cell (mask=0) remains NaN after filling')

    # No-NaN input: array returned unchanged
    arr3 = np.arange(9.0).reshape((3, 3))
    result3 = inpaint_nans_2d(arr3, iterations=10)
    passed &= _check(np.allclose(result3, arr3), 'no-NaN input unchanged')

    # Gradient field: NaN at one end fills towards the gradient boundary value
    arr4 = np.array([[1.0, 2.0, 3.0],
                     [1.0, np.nan, 3.0],
                     [1.0, 2.0, 3.0]])
    result4 = inpaint_nans_2d(arr4, iterations=50)
    passed &= _check(np.isclose(result4[1, 1], 2.0, atol=0.1),
                     f'NaN in gradient field fills to midpoint: got {result4[1, 1]:.4f}')

    return passed


# ── grid.inpaint_nans_3d ─────────────────────────────────────────────────────

def test_inpaint_nans_3d():
    print('\n--- grid.inpaint_nans_3d ---')
    passed = True

    # NaN center cell in a uniform 5.0 cube → should fill to ~5.0
    arr = np.full((3, 3, 3), 5.0)
    arr[1, 1, 1] = np.nan
    result = inpaint_nans_3d(arr, iterations=20)
    passed &= _check(not np.isnan(result[1, 1, 1]), '3D NaN surrounded by 5.0 gets filled')
    passed &= _check(np.isclose(result[1, 1, 1], 5.0, atol=1e-3),
                     f'filled 3D value close to 5.0: got {result[1, 1, 1]:.6f}')

    # Land mask: land cell remains NaN
    arr2 = np.full((3, 3, 3), 5.0)
    arr2[1, 1, 1] = np.nan
    ocean_mask = np.ones((3, 3, 3), dtype=int)
    ocean_mask[1, 1, 1] = 0
    result2 = inpaint_nans_3d(arr2, iterations=20, mask=ocean_mask)
    passed &= _check(np.isnan(result2[1, 1, 1]), 'land cell (mask=0) remains NaN after 3D filling')

    # No-NaN input unchanged
    arr3 = np.random.default_rng(42).random((3, 3, 3))
    result3 = inpaint_nans_3d(arr3, iterations=10)
    passed &= _check(np.allclose(result3, arr3), 'no-NaN 3D input unchanged')

    # Multiple isolated NaN cells: all should be filled when surrounded by valid neighbors
    arr4 = np.full((5, 5, 5), 2.0)
    arr4[2, 2, 2] = np.nan
    arr4[1, 3, 1] = np.nan
    result4 = inpaint_nans_3d(arr4, iterations=30)
    passed &= _check(not np.any(np.isnan(result4)),
                     'multiple isolated NaN cells all filled when surrounded by valid neighbors')

    return passed


# ── chemistry.schmidt_number ─────────────────────────────────────────────────

def test_schmidt_number():
    print('\n--- chemistry.schmidt_number ---')
    passed = True

    # Unknown gas raises ValueError
    try:
        schmidt_number('H2O', 20.0)
        _fail('unknown gas: expected ValueError, got none')
        passed = False
    except ValueError:
        _pass('unknown gas raises ValueError')

    # Scalar input → scalar (or 0-d array) output
    sc = schmidt_number('CO2', 20.0)
    passed &= _check(np.ndim(sc) == 0, f'scalar input → scalar output: got ndim={np.ndim(sc)}')

    # Array input → same shape output
    temps = np.array([0.0, 10.0, 20.0, 30.0])
    sc_arr = schmidt_number('CO2', temps)
    passed &= _check(sc_arr.shape == temps.shape,
                     f'array input → same shape: expected {temps.shape}, got {sc_arr.shape}')

    # Known polynomial value at 20°C for CO2 (Wanninkhof 2014 Table 1)
    # Sc = a + b*T + c*T² + d*T³ + e*T⁴
    for gas, coeffs in [
        ('CO2', (2116.8, -136.25,  4.7353,  -0.092307,  0.0007555)),
        ('O2',  (1920.4, -135.6,   5.2122,  -0.10939,   0.00093777)),
        ('N2',  (2304.8, -162.75,  6.2557,  -0.13129,   0.0011255)),
        ('Ar',  (2078.1, -146.74,  5.6403,  -0.11838,   0.0010148)),
    ]:
        a, b, c, d, e = coeffs
        T = 20.0
        expected = a + b*T + c*T**2 + d*T**3 + e*T**4
        computed = schmidt_number(gas, T)
        passed &= _check(np.isclose(computed, expected, rtol=1e-9),
                         f'{gas} Sc at 20°C: expected {expected:.4f}, got {computed:.4f}')

    # Monotonically decreasing with temperature over the ocean range (-2 to 30°C)
    temps_range = np.linspace(-2, 30, 200)
    for gas in ['CO2', 'O2', 'N2', 'Ar']:
        sc_range = schmidt_number(gas, temps_range)
        passed &= _check(np.all(np.diff(sc_range) < 0),
                         f'{gas} Sc decreases monotonically over -2 to 30°C')

    # All gases return positive Sc over ocean temperatures
    for gas in ['CO2', 'O2', 'N2', 'Ar']:
        sc_range = schmidt_number(gas, temps_range)
        passed &= _check(np.all(sc_range > 0), f'{gas} Sc > 0 over -2 to 30°C')

    return passed


# ── chemistry.calc_piston_velocity ───────────────────────────────────────────

def test_calc_piston_velocity():
    print('\n--- chemistry.calc_piston_velocity ---')
    passed = True

    # Zero wind speed → zero piston velocity everywhere
    sst = np.array([[15.0, 20.0], [5.0, 10.0]])
    k_zero = calc_piston_velocity(sst, np.zeros_like(sst))
    passed &= _check(np.all(k_zero == 0.0), 'zero wind speed → zero piston velocity')

    # Non-negative for positive wind
    wspd = np.array([[5.0, 10.0], [7.0, 3.0]])
    k = calc_piston_velocity(sst, wspd)
    passed &= _check(np.all(k >= 0), 'positive wind speed → non-negative piston velocity')

    # Output shape matches input shape
    sst_big = np.ones((5, 7)) * 15.0
    k_big = calc_piston_velocity(sst_big, np.ones((5, 7)) * 8.0)
    passed &= _check(k_big.shape == (5, 7),
                     f'output shape matches input: expected (5,7), got {k_big.shape}')

    # Known value at SST=20°C, wspd=10 m/s
    T, U = 20.0, 10.0
    Sc = schmidt_number('CO2', T)
    k_expected = 0.251 * U**2 * (Sc / 660)**-0.5 * (24 * 365.25 / 100)  # m yr⁻¹
    k_computed = calc_piston_velocity(np.array([[T]]), np.array([[U]]))[0, 0]
    passed &= _check(np.isclose(k_computed, k_expected, rtol=1e-9),
                     f'k at 20°C, 10 m/s: expected {k_expected:.2f} m yr⁻¹, got {k_computed:.2f}')

    # k ∝ U²: doubling wind quadruples piston velocity (at fixed SST)
    sst1 = np.array([[20.0]])
    k5  = calc_piston_velocity(sst1, np.array([[5.0]]))[0, 0]
    k10 = calc_piston_velocity(sst1, np.array([[10.0]]))[0, 0]
    passed &= _check(np.isclose(k10 / k5, 4.0, rtol=1e-9),
                     f'k ∝ U²: k(10 m/s) / k(5 m/s) = {k10/k5:.6f}, expected 4.0')

    return passed


# ── transport.build_A_matrix ─────────────────────────────────────────────────

def test_build_A_matrix():
    print('\n--- transport.build_A_matrix ---')
    passed = True

    m = 4
    rng = np.random.default_rng(0)

    # Tiny synthetic TR (column-stochastic: columns sum to ~0 for mass conservation)
    TR_dense = rng.random((m, m)) * 0.01
    np.fill_diagonal(TR_dense, -TR_dense.sum(axis=0))
    TR = sparse.csr_matrix(TR_dense)

    # Generic positive-definite physical parameters
    R_C        = rng.random(m) * 10 + 5
    R_A        = rng.random(m) * (-10) - 5
    DIC        = rng.random(m) * 100 + 1900
    AT         = rng.random(m) * 100 + 2200
    aqueous_CO2 = rng.random(m) * 10 + 5
    K0         = rng.random(m) * 1e-2 + 1e-3
    V          = rng.random(m) * 1e10 + 1e9
    z1         = 36.0

    # --- Shape and CSR format ---
    k = rng.random(m) * 5.0
    f_ice = rng.random(m) * 0.3
    A = build_A_matrix(TR, k, f_ice, V, R_C, R_A, DIC, AT, aqueous_CO2, K0, z1)
    passed &= _check(A.shape == (2*m + 1, 2*m + 1),
                     f'A shape: expected ({2*m+1},{2*m+1}), got {A.shape}')
    passed &= _check(A.format == 'csr', f'A format: expected csr, got {A.format!r}')

    # --- k=0 → air-sea exchange vanishes; only TR remains in diagonal blocks ---
    A0 = build_A_matrix(TR, np.zeros(m), np.zeros(m), V, R_C, R_A, DIC, AT, aqueous_CO2, K0, z1)
    Ad = A0.toarray()

    # Row 0 (∆xCO2 equation) should be all zeros (no air-sea flux)
    passed &= _check(np.allclose(Ad[0, :], 0.0), 'k=0: xCO2 row (row 0) is all zeros')

    # Column 0 (∆xCO2 → ocean coupling) should be all zeros
    passed &= _check(np.allclose(Ad[:, 0], 0.0), 'k=0: xCO2 column (col 0) is all zeros')

    # DIC-DIC block [1:m+1, 1:m+1] should equal TR exactly
    passed &= _check(np.allclose(Ad[1:m+1, 1:m+1], TR_dense),
                     'k=0: DIC-DIC block equals TR')

    # DIC-AT cross block [1:m+1, m+1:] should be zero
    passed &= _check(np.allclose(Ad[1:m+1, m+1:], 0.0),
                     'k=0: DIC-AT cross block is zeros')

    # AT-AT block [m+1:, m+1:] should equal TR exactly
    passed &= _check(np.allclose(Ad[m+1:, m+1:], TR_dense),
                     'k=0: AT-AT block equals TR')

    # AT-DIC cross block [m+1:, 1:m+1] should always be zero (AT not influenced by DIC)
    passed &= _check(np.allclose(Ad[m+1:, 1:m+1], 0.0),
                     'k=0: AT-DIC cross block is zeros')

    # --- With nonzero k: A00 should be negative (∆xCO2 is damped by gas exchange) ---
    passed &= _check(A.toarray()[0, 0] < 0,
                     'nonzero k: A[0,0] is negative (∆xCO2 damped by air-sea exchange)')

    return passed


# ── grid.inpaint_nans edge cases ─────────────────────────────────────────────

def test_inpaint_edge_cases():
    print('\n--- inpaint_nans edge cases ---')
    passed = True

    # All-NaN 2D: emits UserWarning; result stays all-NaN.
    arr_nan = np.full((4, 4), np.nan)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = inpaint_nans_2d(arr_nan, iterations=10)
    passed &= _check(np.all(np.isnan(result)), 'all-NaN 2D input: result is all-NaN')
    passed &= _check(any(issubclass(x.category, UserWarning) for x in w),
                     'all-NaN 2D input: UserWarning raised')

    arr_nan3 = np.full((3, 3, 3), np.nan)
    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter('always')
        result3 = inpaint_nans_3d(arr_nan3, iterations=10)
    passed &= _check(np.all(np.isnan(result3)), 'all-NaN 3D input: result is all-NaN')
    passed &= _check(any(issubclass(x.category, UserWarning) for x in w3),
                     'all-NaN 3D input: UserWarning raised')

    # iterations=0: emits UserWarning; NaN cells replaced by global mean; valid cells unchanged.
    arr = np.array([[1., 2., np.nan],
                    [4., np.nan, 6.],
                    [np.nan, 8., 9.]])
    global_mean = np.nanmean(arr)   # (1+2+4+6+8+9)/6 = 5.0
    with warnings.catch_warnings(record=True) as w0:
        warnings.simplefilter('always')
        result0 = inpaint_nans_2d(arr.copy(), iterations=0)
    passed &= _check(any(issubclass(x.category, UserWarning) for x in w0),
                     'iterations=0 2D: UserWarning raised')
    passed &= _check(np.isclose(result0[0, 2], global_mean),
                     f'iterations=0: NaN replaced by global mean ({global_mean}): got {result0[0,2]:.4f}')
    passed &= _check(result0[0, 0] == 1.0 and result0[2, 2] == 9.0,
                     'iterations=0: valid cells unchanged')

    # Single valid cell in a 2D field: all NaNs initialised to that cell's value,
    # so the result is uniformly that value after any number of iterations.
    arr_single = np.full((5, 5), np.nan)
    arr_single[2, 2] = 10.0
    result_single = inpaint_nans_2d(arr_single, iterations=20)
    passed &= _check(not np.any(np.isnan(result_single)),
                     'single valid cell 2D: all NaNs filled after 20 iterations')
    passed &= _check(np.allclose(result_single, 10.0),
                     'single valid cell 2D: all values converge to 10.0')

    return passed


# ── output.write_simulation_step ─────────────────────────────────────────────

def test_output_write_step():
    """Verify ×1e6 unit conversion and c-vector slicing via a temp NetCDF file."""
    print('\n--- output.write_simulation_step ---')
    passed = True

    # 3×2×2 mask with 4 ocean cells (F-order: (0,0,0), (1,0,0), (0,1,0), (0,0,1))
    ocnmask = np.zeros((3, 2, 2), dtype=int)
    ocnmask[0, 0, 0] = 1
    ocnmask[1, 0, 0] = 1
    ocnmask[0, 1, 0] = 1
    ocnmask[0, 0, 1] = 1
    lat   = np.array([0., 1., 2.])
    lon   = np.array([0., 1.])
    depth = np.array([10., 50.])

    # Use integer-like values so float32 conversion is exact
    c    = np.array([5e-7, 1., 2., 3., 4., 10., 20., 30., 40.])   # length 2m+1 = 9
    q_dt = np.array([2e-7, 0.5, 1.5, 2.5, 3.5, 5., 15., 25., 35.])

    fd, tmp = tempfile.mkstemp(suffix='.nc')
    os.close(fd)
    try:
        ds = open_simulation_output(tmp, lat, lon, depth, ocnmask)
        write_simulation_step(ds, c, q_dt, time=2020.5, ocnmask=ocnmask)
        ds.close()

        with Dataset(tmp, 'r') as nc:
            # ×1e6 unit conversion (dimensionless → ppm)
            passed &= _check(np.isclose(nc['delxCO2'][0],    c[0]    * 1e6, rtol=1e-5),
                             f'delxCO2 = c[0]×1e6: expected {c[0]*1e6:.4f}, got {float(nc["delxCO2"][0]):.4f}')
            passed &= _check(np.isclose(nc['xCO2_added'][0], q_dt[0] * 1e6, rtol=1e-5),
                             f'xCO2_added = q_dt[0]×1e6: expected {q_dt[0]*1e6:.4f}, got {float(nc["xCO2_added"][0]):.4f}')

            # DIC slicing: c[1:m+1] maps to ocean cells in F-order
            delDIC = nc['delDIC'][0]   # shape (3, 2, 2)
            passed &= _check(np.isclose(delDIC[0, 0, 0], c[1], rtol=1e-5),
                             f'delDIC ocean cell 0 → (0,0,0): expected {c[1]}, got {delDIC[0,0,0]:.4f}')
            passed &= _check(np.isclose(delDIC[1, 0, 0], c[2], rtol=1e-5),
                             f'delDIC ocean cell 1 → (1,0,0): expected {c[2]}, got {delDIC[1,0,0]:.4f}')
            passed &= _check(np.isclose(delDIC[0, 1, 0], c[3], rtol=1e-5),
                             f'delDIC ocean cell 2 → (0,1,0): expected {c[3]}, got {delDIC[0,1,0]:.4f}')
            passed &= _check(np.isclose(delDIC[0, 0, 1], c[4], rtol=1e-5),
                             f'delDIC ocean cell 3 → (0,0,1): expected {c[4]}, got {delDIC[0,0,1]:.4f}')
            passed &= _check(np.isnan(delDIC[2, 0, 0]),
                             'delDIC at land cell (2,0,0) is NaN')

            # AT slicing: c[m+1:] = c[5:]
            delAT = nc['delAT'][0]
            passed &= _check(np.isclose(delAT[0, 0, 0], c[5], rtol=1e-5),
                             f'delAT ocean cell 0 → (0,0,0): expected {c[5]}, got {delAT[0,0,0]:.4f}')
            passed &= _check(np.isclose(delAT[1, 0, 0], c[6], rtol=1e-5),
                             f'delAT ocean cell 1 → (1,0,0): expected {c[6]}, got {delAT[1,0,0]:.4f}')

            # Time coordinate written correctly
            passed &= _check(float(nc['time'][0]) == 2020.5,
                             f'time coordinate: expected 2020.5, got {float(nc["time"][0])}')
    finally:
        os.unlink(tmp)

    return passed


# ── base.BaseExperiment._output_path ─────────────────────────────────────────

def test_output_path():
    print('\n--- base.BaseExperiment._output_path ---')
    passed = True

    def _exp(output_path, max_steps=0):
        cfg = ExperimentConfig(
            data_path='./data/', output_path=output_path,
            scenario='ssp245', start_year=2020.0,
            times=np.array([0., 1.]), max_steps_per_file=max_steps,
        )
        return BaseExperiment(cfg)

    # max_steps_per_file=0: always return the original path unchanged
    exp = _exp('./outputs/test.nc', max_steps=0)
    passed &= _check(exp._output_path(0) == './outputs/test.nc',
                     'max_steps=0: returns original path for file 0')
    passed &= _check(exp._output_path(99) == './outputs/test.nc',
                     'max_steps=0: returns original path for any file number')

    # max_steps_per_file > 0: strip .nc, append _{NNN}.nc with zero-padded 3 digits
    exp2 = _exp('./outputs/test.nc', max_steps=1000)
    passed &= _check(exp2._output_path(0)  == './outputs/test_000.nc',
                     'multi-file, file 0:  → _000.nc')
    passed &= _check(exp2._output_path(5)  == './outputs/test_005.nc',
                     'multi-file, file 5:  → _005.nc')
    passed &= _check(exp2._output_path(42) == './outputs/test_042.nc',
                     'multi-file, file 42: → _042.nc')

    # Path without .nc extension: no spurious stripping, suffix appended cleanly
    exp3 = _exp('./outputs/run', max_steps=1000)
    passed &= _check(exp3._output_path(0) == './outputs/run_000.nc',
                     'path without .nc: appends _000.nc without double-suffix')

    return passed


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('flatten / make_3d',      test_flatten_make3d),
        ('get_depth_idx',          test_get_depth_idx),
        ('inpaint_nans_2d',        test_inpaint_nans_2d),
        ('inpaint_nans_3d',        test_inpaint_nans_3d),
        ('inpaint_nans edge cases',test_inpaint_edge_cases),
        ('schmidt_number',         test_schmidt_number),
        ('calc_piston_velocity',   test_calc_piston_velocity),
        ('build_A_matrix',         test_build_A_matrix),
        ('write_simulation_step',  test_output_write_step),
        ('_output_path',           test_output_path),
    ]

    results = {}
    for name, fn in tests:
        results[name] = fn()

    print('\n--- Summary ---')
    all_passed = all(results.values())
    for name, ok in results.items():
        status = 'PASS' if ok else 'FAIL'
        print(f'  {status}  {name}')

    if not all_passed:
        sys.exit(1)

# %%
