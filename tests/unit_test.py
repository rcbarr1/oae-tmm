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
  ImpulseResponse.make_q          — overlap-fraction logic conserves total AT across all dt

Usage:
    python tests/unit_test.py
"""
#%%
import os
import sys
import tempfile
import warnings

import numpy as np
from scipy import sparse

from netCDF4 import Dataset  # type: ignore[import-untyped]

from oae_tmm.grid import flatten, make_3d, get_depth_idx, inpaint_nans_2d, inpaint_nans_3d
from oae_tmm.chemistry import schmidt_number, calc_piston_velocity
from oae_tmm.transport import build_A_matrix
from oae_tmm.output import open_simulation_output, write_simulation_step
from experiments.base import BaseExperiment, ExperimentConfig, rho, Patm, Ma
from experiments.impulse_response import ImpulseResponse


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

    # Land-contamination fix: a NaN ocean cell flanked by land on 3 sides and two valid
    # ocean neighbors (100.0 and 0.0) should fill to their mean (50.0), not be pulled
    # toward the global mean of the ocean cells (~25.0).
    arr5 = np.full((5, 5, 1), np.nan)
    arr5[0, 1, 0] = 100.0
    arr5[2, 1, 0] = 0.0
    arr5[3, 1, 0] = 0.0
    arr5[4, 1, 0] = 0.0
    ocean_mask5 = np.zeros((5, 5, 1), dtype=int)
    ocean_mask5[:, 1, 0] = 1   # single column of ocean; all other cells are land
    result5 = inpaint_nans_3d(arr5, iterations=5, mask=ocean_mask5)
    # [1,1,0] has valid ocean neighbors [0,1,0]=100 and [2,1,0]=0; land on 3 sides
    passed &= _check(np.isclose(result5[1, 1, 0], 50.0, atol=1e-3),
                     f'land-contamination: fills to ocean-neighbor mean (50.0), got {result5[1,1,0]:.4f}')

    # Global-mean pre-fill fix: a NaN patch surrounded by low values (100.0) in an ocean
    # with a high global mean (~2000.0) should fill toward the boundary value, not the mean.
    arr6 = np.full((10, 10, 1), 2000.0)
    arr6[3:7, 3:7, 0] = np.nan
    arr6[2, 3:7, 0] = 100.0   # boundary: top
    arr6[7, 3:7, 0] = 100.0   # boundary: bottom
    arr6[3:7, 2, 0] = 100.0   # boundary: left
    arr6[3:7, 7, 0] = 100.0   # boundary: right
    result6 = inpaint_nans_3d(arr6)
    passed &= _check(not np.any(np.isnan(result6[3:7, 3:7, 0])),
                     'local-min: all NaN cells in patch filled')
    passed &= _check(np.all(np.abs(result6[3:7, 3:7, 0] - 100.0) < 1.0),
                     f'local-min: patch fills toward boundary value (100.0), not global mean (~2000.0); '
                     f'max deviation: {np.max(np.abs(result6[3:7, 3:7, 0] - 100.0)):.4f}')

    # Convergence: a 20-cell chain fills completely from one end via convergence check.
    chain_len = 20
    arr7 = np.full((chain_len, 3, 1), np.nan)
    arr7[0, 1, 0] = 50.0
    ocean_mask7 = np.zeros((chain_len, 3, 1), dtype=int)
    ocean_mask7[:, 1, 0] = 1
    result7 = inpaint_nans_3d(arr7, mask=ocean_mask7)
    passed &= _check(not np.any(np.isnan(result7[:, 1, 0])),
                     f'large-gap: all {chain_len} cells in chain filled by convergence')
    passed &= _check(np.allclose(result7[:, 1, 0], 50.0),
                     f'large-gap: all filled cells equal 50.0; '
                     f'max deviation: {np.max(np.abs(result7[:, 1, 0] - 50.0)):.6f}')

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
    CT         = rng.random(m) * 100 + 1900
    AT         = rng.random(m) * 100 + 2200
    aqueous_CO2 = rng.random(m) * 10 + 5
    K0         = rng.random(m) * 1e-2 + 1e-3
    V          = rng.random(m) * 1e10 + 1e9
    z1         = 36.0

    # --- Shape and CSR format ---
    k = rng.random(m) * 5.0
    f_ice = rng.random(m) * 0.3
    A = build_A_matrix(TR, k, f_ice, V, R_C, R_A, CT, AT, aqueous_CO2, K0, z1, rho, Patm, Ma)
    passed &= _check(A.shape == (2*m + 1, 2*m + 1),
                     f'A shape: expected ({2*m+1},{2*m+1}), got {A.shape}')
    passed &= _check(A.format == 'csr', f'A format: expected csr, got {A.format!r}')

    # --- k=0 → air-sea exchange vanishes; only TR remains in diagonal blocks ---
    A0 = build_A_matrix(TR, np.zeros(m), np.zeros(m), V, R_C, R_A, CT, AT, aqueous_CO2, K0, z1, rho, Patm, Ma)
    Ad = A0.toarray()

    # Row 0 (∆xCO2 equation) should be all zeros (no air-sea flux)
    passed &= _check(np.allclose(Ad[0, :], 0.0), 'k=0: xCO2 row (row 0) is all zeros')

    # Column 0 (∆xCO2 → ocean coupling) should be all zeros
    passed &= _check(np.allclose(Ad[:, 0], 0.0), 'k=0: xCO2 column (col 0) is all zeros')

    # CT-CT block [1:m+1, 1:m+1] should equal TR exactly
    passed &= _check(np.allclose(Ad[1:m+1, 1:m+1], TR_dense),
                     'k=0: CT-CT block equals TR')

    # CT-AT cross block [1:m+1, m+1:] should be zero
    passed &= _check(np.allclose(Ad[1:m+1, m+1:], 0.0),
                     'k=0: CT-AT cross block is zeros')

    # AT-AT block [m+1:, m+1:] should equal TR exactly
    passed &= _check(np.allclose(Ad[m+1:, m+1:], TR_dense),
                     'k=0: AT-AT block equals TR')

    # AT-CT cross block [m+1:, 1:m+1] should always be zero (AT not influenced by CT)
    passed &= _check(np.allclose(Ad[m+1:, 1:m+1], 0.0),
                     'k=0: AT-CT cross block is zeros')

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

    # iterations=0: emits UserWarning; NaN cells remain NaN; valid cells unchanged.
    arr = np.array([[1., 2., np.nan],
                    [4., np.nan, 6.],
                    [np.nan, 8., 9.]])
    with warnings.catch_warnings(record=True) as w0:
        warnings.simplefilter('always')
        result0 = inpaint_nans_2d(arr.copy(), iterations=0)
    passed &= _check(any(issubclass(x.category, UserWarning) for x in w0),
                     'iterations=0 2D: UserWarning raised')
    passed &= _check(np.isnan(result0[0, 2]) and np.isnan(result0[1, 1]) and np.isnan(result0[2, 0]),
                     'iterations=0: NaN cells remain NaN')
    passed &= _check(result0[0, 0] == 1.0 and result0[2, 2] == 9.0,
                     'iterations=0: valid cells unchanged')

    # Single valid cell in a 2D field: fill propagates outward from the single valid
    # cell; all NaNs reach 10.0 after enough iterations.
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

            # CT slicing: c[1:m+1] maps to ocean cells in F-order
            delCT = nc['delCT'][0]   # shape (3, 2, 2)
            passed &= _check(np.isclose(delCT[0, 0, 0], c[1], rtol=1e-5),
                             f'delCT ocean cell 0 → (0,0,0): expected {c[1]}, got {delCT[0,0,0]:.4f}')
            passed &= _check(np.isclose(delCT[1, 0, 0], c[2], rtol=1e-5),
                             f'delCT ocean cell 1 → (1,0,0): expected {c[2]}, got {delCT[1,0,0]:.4f}')
            passed &= _check(np.isclose(delCT[0, 1, 0], c[3], rtol=1e-5),
                             f'delCT ocean cell 2 → (0,1,0): expected {c[3]}, got {delCT[0,1,0]:.4f}')
            passed &= _check(np.isclose(delCT[0, 0, 1], c[4], rtol=1e-5),
                             f'delCT ocean cell 3 → (0,0,1): expected {c[4]}, got {delCT[0,0,1]:.4f}')
            passed &= _check(np.isnan(delCT[2, 0, 0]),
                             'delCT at land cell (2,0,0) is NaN')

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


# ── MLD mask logic ───────────────────────────────────────────────────────────

def test_mld_mask():
    """mldmask: 1 where cell bottom depth (vol/area + top) < MLD, 0 otherwise.

    Uses actual cell bottom depths rather than nan-padded shifted tops, so the
    deepest ocean cell in each column is correctly included when the MLD extends
    to or past the seafloor.
    """
    print('\n--- MLD mask logic ---')
    passed = True

    # ── Standard cases ────────────────────────────────────────────────────────
    # 1×2 spatial grid, 4 depth levels, all ocean
    ocnmask_std = np.ones((1, 2, 4), dtype=int)
    cell_bottom_depth_3d_std = np.array([[[  50., 100., 200., 400.],   # lon 0
                                          [  30.,  80., 180., 360.]]])  # lon 1
    mld_std = np.array([[75., 95.]])
    mldmask_std = ((cell_bottom_depth_3d_std < mld_std[:, :, None]) * ocnmask_std).astype(int)

    # lon 0: bottoms=[50,100,200,400], MLD=75 → only cell 0 (bottom=50) qualifies
    passed &= _check(list(mldmask_std[0, 0]) == [1, 0, 0, 0],
                     f'lon 0 (MLD=75m): expected [1,0,0,0], got {list(mldmask_std[0, 0])}')

    # lon 1: bottoms=[30,80,180,360], MLD=95 → cells 0 and 1 qualify (bottoms 30 and 80 < 95)
    passed &= _check(list(mldmask_std[0, 1]) == [1, 1, 0, 0],
                     f'lon 1 (MLD=95m): expected [1,1,0,0], got {list(mldmask_std[0, 1])}')

    # ── Strict inequality ─────────────────────────────────────────────────────
    ocnmask_eq = np.ones((1, 1, 4), dtype=int)
    mld_exact  = np.array([[50.]])
    cell_bottom_exact = np.array([[[50., 100., 200., 400.]]])
    mldmask_exact = ((cell_bottom_exact < mld_exact[:, :, None]) * ocnmask_eq).astype(int)
    passed &= _check(mldmask_exact[0, 0, 0] == 0,
                     f'strict <: bottom=MLD=50 excluded (got {mldmask_exact[0, 0, 0]})')

    # ── Edge case: MLD extends to or past the actual seafloor ─────────────────
    # lon 0: 2 cells deep, seafloor at 100 m; lon 1: 1 cell deep, seafloor at 100 m
    ocnmask_edge = np.zeros((1, 2, 4), dtype=int)
    ocnmask_edge[0, 0, :2] = 1
    ocnmask_edge[0, 1, :1] = 1
    cell_bottom_depth_3d_edge = np.full((1, 2, 4), np.nan)
    cell_bottom_depth_3d_edge[0, 0, :2] = [50., 100.]
    cell_bottom_depth_3d_edge[0, 1, :1] = [100.]

    mld_deep = np.array([[120., 120.]])
    mldmask_deep = ((cell_bottom_depth_3d_edge < mld_deep[:, :, None]) * ocnmask_edge).astype(int)

    # lon 0: MLD=120m past seafloor at 100m → both cells included
    passed &= _check(list(mldmask_deep[0, 0]) == [1, 1, 0, 0],
                     f'MLD past seafloor, lon 0: expected [1,1,0,0], got {list(mldmask_deep[0, 0])}')

    # lon 1: single-cell column, MLD=120m past seafloor at 100m → cell included
    passed &= _check(mldmask_deep[0, 1, 0] == 1,
                     f'MLD past seafloor, single-cell col: expected 1, got {mldmask_deep[0, 1, 0]}')

    # lon 0: MLD=80m, deepest cell (bottom=100m) not reached → excluded
    mld_shallow = np.array([[80., 80.]])
    mldmask_shallow = ((cell_bottom_depth_3d_edge < mld_shallow[:, :, None]) * ocnmask_edge).astype(int)
    passed &= _check(mldmask_shallow[0, 0, 0] == 1 and mldmask_shallow[0, 0, 1] == 0,
                     f'MLD=80m, lon 0: expected [1,0,...], got {list(mldmask_shallow[0, 0])}')
    passed &= _check(mldmask_shallow[0, 1, 0] == 0,
                     f'MLD=80m, single-cell col: expected 0, got {mldmask_shallow[0, 1, 0]}')

    return passed


# ── base.BaseExperiment._output_path ─────────────────────────────────────────

def test_output_path():
    print('\n--- base.BaseExperiment._output_path ---')
    passed = True

    class _StubExperiment(BaseExperiment):
        def make_q(self, time_current, chem, dt):
            return np.zeros(1)

    def _exp(output_path, max_steps=0):
        cfg = ExperimentConfig(
            data_path='./data/', output_path=output_path,
            scenario='ssp245',
            time=np.array([2020.0, 2021.0]), max_steps_per_file=max_steps,
        )
        return _StubExperiment(cfg)

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


# ── ImpulseResponse.make_q overlap logic ─────────────────────────────────────

def test_impulse_make_q():
    """ImpulseResponse.make_q: overlap-fraction logic conserves total AT across all dt.

    Verifies three properties:
      1. Total AT added (sum of q_AT * dt over all steps) equals rate * impulse_end
         for every timestep resolution, because the overlap fraction scales rate so
         the contribution from each step is proportional to how much of that step
         falls within the 30-day impulse window.
      2. q is exactly zero for all steps whose window lies entirely after impulse_end.
      3. AT flux lands only on the designated cell; xCO2 and CT components are zero.
    """
    print('\n--- ImpulseResponse.make_q overlap logic ---')
    passed = True

    # Minimal stub: set only the attributes make_q touches (no data loading).
    m           = 6
    target_cell = 3
    z1          = 36.0

    cfg = ExperimentConfig(
        data_path='./', output_path='./test.nc',
        scenario='none', time=np.array([2022.0]),
    )
    exp              = ImpulseResponse(cfg)
    exp.m            = m
    exp.grid         = {'z1': z1}
    exp._q_AT_mask   = np.zeros(m)
    exp._q_AT_mask[target_cell] = 1.0

    rate           = 10 * 1e6 / z1 / rho   # [µmol AT kg⁻¹ yr⁻¹]
    impulse_end    = 30 / 360
    expected_total = rate * impulse_end

    def total_AT_added(time):
        total = 0.0
        for dt_i, t_cur in zip(np.diff(time), time[1:]):
            q = exp.make_q(t_cur, {}, dt_i)
            total += q[m + 1 + target_cell] * dt_i
        return total

    # 1. Total AT is conserved across all timestep resolutions.
    for name, time in [
        ('annual',   np.arange(2022, 2028,     1.0   )),
        ('monthly',  np.arange(2022, 2027.084, 1/12  )),
        ('dekadal',  np.arange(2022, 2027.028, 1/36  )),
        ('pentadal', np.arange(2022, 2027.014, 1/72  )),
        ('daily',    np.arange(2022, 2027.003, 1/360 )),
    ]:
        tot = total_AT_added(time)
        passed &= _check(
            np.isclose(tot, expected_total, rtol=1e-9),
            f'{name}: total AT = {tot:.6f} µmol kg⁻¹, expected {expected_total:.6f}',
        )

    # 2. q is zero for every daily step whose window starts after impulse_end.
    time_d = np.arange(2022, 2027.003, 1/360)
    any_nonzero = False
    for dt_i, t_cur in zip(np.diff(time_d), time_d[1:]):
        t_off = t_cur - 2022.0
        if t_off - dt_i >= impulse_end:   # step window lies entirely after impulse
            q = exp.make_q(t_cur, {}, dt_i)
            if not np.allclose(q, 0.0):
                any_nonzero = True
                break
    passed &= _check(not any_nonzero, 'q is zero for all daily steps after impulse window')

    # 3. AT lands only on target cell; xCO2 and CT are zero (NaOH assumption).
    q = exp.make_q(2022.0 + 1/360, {}, 1/360)
    at_vec = q[(m + 1):]
    passed &= _check(at_vec[target_cell] > 0, 'target cell receives positive AT flux')
    passed &= _check(np.allclose(at_vec[np.arange(m) != target_cell], 0.0),
                     'non-target cells receive zero AT flux')
    passed &= _check(q[0] == 0.0 and np.allclose(q[1:(m + 1)], 0.0),
                     'xCO2 and CT components are zero (NaOH)')

    return passed


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('flatten / make_3d',         test_flatten_make3d),
        ('get_depth_idx',             test_get_depth_idx),
        ('inpaint_nans_2d',           test_inpaint_nans_2d),
        ('inpaint_nans_3d',           test_inpaint_nans_3d),
        ('inpaint_nans edge cases',   test_inpaint_edge_cases),
        ('schmidt_number',            test_schmidt_number),
        ('calc_piston_velocity',      test_calc_piston_velocity),
        ('build_A_matrix',            test_build_A_matrix),
        ('write_simulation_step',     test_output_write_step),
        ('_output_path',              test_output_path),
        ('MLD mask logic',            test_mld_mask),
        ('ImpulseResponse.make_q',    test_impulse_make_q),
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
