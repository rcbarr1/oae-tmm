"""
One-time comparison: scripts/  vs  src/utils/  for all three preprocessing scripts.

Each test runs the old and new script, mocks all file writes to capture outputs
in memory, compares them numerically, and produces no output files on disk.

  generate_input_data:  np.save mocked    — compares all .npy arrays
  make_trace_gridded:   xr.Dataset.to_netcdf mocked — compares Canth DataArrays
                        (limited to 1 year, 1 scenario via source patching)
  interp_REMIND:        np.savetxt mocked — compares CO2 trajectory columns

The old make_trace_gridded.py reads its own output files at the end for
plotting; xr.open_dataset is also mocked to serve the captured datasets back,
preventing file-not-found errors.

Usage:
    python tests/scripts_test.py
"""
#%%
import os
import runpy
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')   # suppress figure display from old scripts' plot sections
import numpy as np
from tqdm import tqdm
import xarray as xr

DATA_PATH = './data/'
RTOL = 1e-4
ATOL = 1e-3


def _pass(msg): print(f'  PASS  {msg}')
def _fail(msg): print(f'  FAIL  {msg}')


def _stats(o, n):
    finite = np.isfinite(np.asarray(o)) & np.isfinite(np.asarray(n))
    if not np.any(finite):
        print('  (no finite values to compare)')
        return
    abs_diff = np.abs(np.asarray(n)[finite] - np.asarray(o)[finite])
    print(f'  values (old): mean={np.nanmean(o):.3e},  min={np.nanmin(o):.3e},  max={np.nanmax(o):.3e}')
    print(f'  errors:  mean abs={np.mean(abs_diff):.2e},  max abs={np.max(abs_diff):.2e}')


# ── generate_input_data ───────────────────────────────────────────────────────

def test_generate_input_data():
    """Run both generate_input_data scripts; compare every .npy array written."""
    print('\n--- generate_input_data ---')

    def _run(script_path):
        saved = {}
        def _mock(path, arr, *_a, **_kw):
            saved[path] = np.array(arr).copy()
        with patch('numpy.save', side_effect=_mock):
            runpy.run_path(script_path, run_name='__main__')
        return saved

    print('  running old...')
    old_saved = _run('src/utils/generate_input_data.py')
    print('  running new...')
    new_saved = _run('scripts/generate_input_data.py')

    all_keys = sorted(set(old_saved) | set(new_saved))
    passed = True
    for path in tqdm(all_keys, desc='  comparing arrays'):
        name = path.split('/')[-1]
        if path not in old_saved:
            _fail(f'{name}: missing from old output')
            passed = False
            continue
        if path not in new_saved:
            _fail(f'{name}: missing from new output')
            passed = False
            continue

        o, n = old_saved[path], new_saved[path]
        print(f'\n  {name}')
        _stats(o, n)
        try:
            np.testing.assert_allclose(n, o, rtol=RTOL, atol=ATOL, equal_nan=True)
            _pass(f'{name}: within rtol={RTOL}, atol={ATOL}')
        except AssertionError:
            _fail(f'{name}: exceeds rtol={RTOL}, atol={ATOL}')
            passed = False

    return passed


# ── make_trace_gridded ────────────────────────────────────────────────────────

def _patch_trace_src(src, is_old):
    """Limit make_trace_gridded to 1 year and 1 scenario by text substitution."""
    src = src.replace('years = np.arange(2000, 2101)', 'years = np.array([2020])')
    if is_old:
        src = src.replace(
            "scenario_dict = {'none' : 1, 'ssp119': 2, 'ssp126' : 3, 'ssp245' : 4, 'ssp370' : 5,\n"
            "                 'ssp360_NTCF' : 6, 'ssp434' : 7, 'ssp460' : 8, 'ssp534_OS' : 9, 'REMIND' : 10}",
            "scenario_dict = {'ssp119': 2}",
        )
    else:
        src = src.replace(
            "scenarios = {\n"
            "    'none': 1, 'ssp119': 2, 'ssp126': 3, 'ssp245': 4, 'ssp370': 5,\n"
            "    'ssp370_lowNTCF': 6, 'ssp434': 7, 'ssp460': 8, 'ssp534_OS': 9, 'REMIND': 10,\n"
            "}",
            "scenarios = {'ssp119': 2}",
        )
    return src


def _run_trace_gridded(script_path, is_old):
    with open(script_path) as f:
        src = f.read()

    src = _patch_trace_src(src, is_old)

    nc_saved = {}
    real_open_dataset = xr.open_dataset   # capture reference before patching

    def mock_to_netcdf(self, path=None, *_a, **_kw):
        if path is not None:
            nc_saved[path] = self.copy(deep=True)

    def mock_open_dataset(path, **kw):
        if path in nc_saved:
            return nc_saved[path]
        return real_open_dataset(path, **kw)

    ns = {'__name__': '__main__', '__file__': script_path, '__builtins__': __builtins__}
    with patch.object(xr.Dataset, 'to_netcdf', mock_to_netcdf), \
         patch('xarray.open_dataset', mock_open_dataset):
        exec(compile(src, script_path, 'exec'), ns)  # noqa: S102

    return nc_saved


def test_make_trace_gridded():
    """Run both make_trace_gridded scripts (1 year, ssp119); compare Canth."""
    print('\n--- make_trace_gridded  (ssp119, 2020)  — ~5 min ---')

    print('  running old...')
    old_saved = _run_trace_gridded('src/utils/make_trace_gridded.py', is_old=True)
    print('  running new...')
    new_saved = _run_trace_gridded('scripts/make_trace_gridded.py', is_old=False)

    key = DATA_PATH + 'TRACE_gridded/OCIM_CanthFromTRACECO2Pathway2.nc'
    if key not in old_saved or key not in new_saved:
        _fail(f'expected output {key} not captured (check scenario/years patching)')
        return False

    o = old_saved[key]['Canth'].values
    n = new_saved[key]['Canth'].values
    _stats(o, n)

    try:
        np.testing.assert_allclose(n, o, rtol=RTOL, atol=ATOL, equal_nan=True)
        _pass(f'Canth (ssp119, 2020): within rtol={RTOL}, atol={ATOL}')
        return True
    except AssertionError:
        _fail(f'Canth (ssp119, 2020): exceeds rtol={RTOL}, atol={ATOL}')
        return False


# ── interp_REMIND ─────────────────────────────────────────────────────────────

def test_interp_REMIND():
    """Run both interp_REMIND scripts; compare CO2TrajectoriesAdjusted REMIND column."""
    print('\n--- interp_REMIND  (CO2TrajectoriesAdjusted REMIND column) ---')

    def _run(script_path):
        saved = {}
        def _mock(path, arr, *_a, **_kw):
            saved[path] = np.array(arr, dtype=float)
        with patch('numpy.savetxt', side_effect=_mock):
            runpy.run_path(script_path, run_name='__main__')
        return saved

    old_saved = _run('src/utils/interp_REMIND.py')
    new_saved = _run('scripts/interp_REMIND.py')

    old_adj = old_saved['./pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt']
    new_adj = new_saved['./pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt']
    old_raw = old_saved['./pyTRACE/pyTRACE/data/CO2Trajectories.txt']
    new_raw = new_saved['./pyTRACE/pyTRACE/data/CO2Trajectories.txt']

    passed = True
    for label, o_col, n_col in [
        ('adjusted REMIND', old_adj[:, -1], new_adj[:, -1]),
        ('unadjusted REMIND', old_raw[:, 10], new_raw[:, 10]),
    ]:
        _stats(o_col, n_col)
        try:
            np.testing.assert_array_equal(n_col, o_col)
            _pass(f'{label}: exact match')
        except AssertionError:
            _fail(f'{label}: values differ')
            passed = False

    return passed


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('generate_input_data', test_generate_input_data),
        ('make_trace_gridded',  test_make_trace_gridded),
        ('interp_REMIND',       test_interp_REMIND),
    ]
    results = []
    for name, fn in tqdm(tests, desc='tests'):
        results.append(fn())

    r1, r2, r3 = results
    print('\n--- Summary ---')
    if all([r1, r2, r3]):
        print('All checks passed.')
    else:
        print('Some checks failed — see above.')
        sys.exit(1)

# %%
