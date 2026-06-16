"""
Scientific invariant tests for oae_tmm.

Tests physics that must hold regardless of grid resolution or experiment:

  1. Zero-forcing stability          — c=0, q=0 → c stays zero
  2. CT mass conservation  (k=0)   — sum(∆CT·V·ρ) = sum(q_CT·V·ρ)·N·dt exactly
  3. Total carbon conservation (k>0) — ∆xCO2·Ma + sum(∆CT·V·ρ) = sum(q_CT·V·ρ)·N·dt
  4. AT mass conservation            — sum(∆AT·V·ρ) = sum(q_AT·V·ρ)·N·dt, for any k
  5. Linearity                       — frozen A: 2×q → 2×c; c(q1+q2) = c(q1)+c(q2)

All tests use a small synthetic grid with a hand-constructed mass-conserving
transport matrix — no data files required.

Mathematical basis
------------------
If V^T · TR = 0 (TR conserves mass with volumes V), then:

    [Ma, V·ρ·1_m,  0    ] · A = 0   (total carbon is neutral)
    [0,  0,        V·ρ·1_m] · A = 0  (AT mass is neutral)

These follow from cancellation of gammax/gammaC terms and V^T·TR=0. For the
implicit Euler step (I - dt·A)·c_{n+1} = c_n + dt·q, multiplying both sides
by either weight vector gives exact conservation at every step, up to the
tolerance of the linear solver.

Usage:
    python tests/invariant_test.py
"""
#%%
import sys

import numpy as np
from scipy import sparse

from oae_tmm.transport import build_A_matrix, solve_timestep
from experiments.base import rho, Patm, Ma


def _pass(msg): print(f'  PASS  {msg}')
def _fail(msg): print(f'  FAIL  {msg}')


def _check(cond, msg):
    if cond:
        _pass(msg)
        return True
    _fail(msg)
    return False


# ── Shared synthetic grid ─────────────────────────────────────────────────────

m      = 6    # ocean cells
m_surf = 2    # first m_surf cells are surface (k > 0); rest subsurface (k = 0)

# Bidirectional circular-chain TR: each cell exchanges equally with its two
# neighbors → column sums = 0 → V^T · TR = 0 for any uniform volume vector.
alpha    = 0.05                    # exchange rate [yr^-1]
TR_dense = np.zeros((m, m))
for _i in range(m):
    _j = (_i + 1) % m
    TR_dense[_j, _i] += alpha;  TR_dense[_i, _i] -= alpha   # _i → _j
    TR_dense[_i, _j] += alpha;  TR_dense[_j, _j] -= alpha   # _j → _i
TR = sparse.csr_matrix(TR_dense)

# Grid constants
V    = np.full(m, 1e13)   # equal cell volumes [m^3]
rho  = 1025.0             # seawater density [kg m^-3]
Ma   = 1.8e26             # total moles of atmosphere [µmol air]
z1   = 36.0               # surface layer thickness [m]

# Carbonate chemistry (realistic, uniform across cells)
R_C         = np.full(m, 10.0)    # Revelle factor (R_C > 0)
R_A         = np.full(m, -15.0)   # alkalinity sensitivity (R_A < 0)
CT          = np.full(m, 2000.0)  # [µmol kg^-1]
AT          = np.full(m, 2300.0)  # [µmol kg^-1]
aqueous_CO2 = np.full(m, 10.0)    # [µmol kg^-1]
K0          = np.full(m, aqueous_CO2[0] / 400.0 * rho)   # from pCO2 = 400 µatm

# Piston velocity: nonzero only at surface cells
k     = np.zeros(m);  k[:m_surf] = 20.0   # [m yr^-1]
f_ice = np.zeros(m)

dt      = 1 / 12   # one month [yr]
N_steps = 10

# Build A matrices once (frozen chemistry — constant for all tests)
A_k0 = build_A_matrix(TR, np.zeros(m), f_ice, V, R_C, R_A, CT, AT, aqueous_CO2, K0, z1, rho, Patm, Ma)
A    = build_A_matrix(TR, k,           f_ice, V, R_C, R_A, CT, AT, aqueous_CO2, K0, z1, rho, Patm, Ma)


def _run(A_mat, q, n=N_steps):
    """Advance from c=0 for n implicit Euler steps with frozen A."""
    c = np.zeros(A_mat.shape[0])
    for _ in range(n):
        c = solve_timestep(A_mat, c, q, dt)
    return c


def _ct_mass(c):
    """Total ocean CT perturbation mass [µmol]."""
    return float(np.dot(c[1:m+1], V * rho))

def _at_mass(c):
    """Total ocean AT perturbation mass [µmol]."""
    return float(np.dot(c[m+1:], V * rho))

def _atm_mass(c):
    """Total atmospheric CO2 perturbation mass [µmol]."""
    return float(c[0] * Ma)


# ── Test 1: zero-forcing stability ───────────────────────────────────────────

def test_zero_forcing():
    """c=0, q=0 → c stays exactly zero (tests PETSc trivial-solution path)."""
    print('\n--- zero-forcing stability ---')
    passed = True

    c = _run(A, np.zeros(2*m + 1))
    max_abs = float(np.max(np.abs(c)))

    passed &= _check(max_abs < 1e-20,
                     f'c=0, q=0 → c stays zero: max|c| = {max_abs:.2e}')
    return passed


# ── Test 2: CT mass conservation (k=0) ──────────────────────────────────────

def test_ct_conservation_no_exchange():
    """With k=0, ocean CT mass = integral of source; ∆xCO2 stays zero."""
    print('\n--- CT mass conservation (k=0) ---')
    passed = True

    q = np.zeros(2*m + 1)
    q[1:m+1] = 1.0   # uniform CT source [µmol kg^-1 yr^-1]

    c        = _run(A_k0, q)
    expected = N_steps * dt * float(np.dot(q[1:m+1], V * rho))
    computed = _ct_mass(c)
    rtol     = abs(computed - expected) / abs(expected)

    print(f'  expected CT mass:  {expected:.6e} µmol')
    print(f'  computed CT mass:  {computed:.6e} µmol')
    print(f'  relative error:     {rtol:.2e}')
    passed &= _check(rtol < 1e-6,
                     f'CT mass conserved to rtol < 1e-6: rtol = {rtol:.2e}')

    # With k=0, xCO2 row/col of A are all zeros → ∆xCO2 must stay at 0
    passed &= _check(abs(c[0]) < 1e-20,
                     f'∆xCO2 stays zero when k=0: |∆xCO2| = {abs(c[0]):.2e}')

    return passed


# ── Test 3: total carbon conservation (k>0) ──────────────────────────────────

def test_total_carbon_conservation():
    """∆xCO2·Ma + sum(∆CT·V·ρ) equals CDR input for all time, even with air-sea exchange."""
    print('\n--- total carbon conservation (k>0) ---')
    passed = True

    q = np.zeros(2*m + 1)
    q[1:m+1] = 1.0   # CT source; no direct atmospheric injection (q[0]=0)

    c              = _run(A, q)
    expected_input = N_steps * dt * float(np.dot(q[1:m+1], V * rho))
    computed_total = _ct_mass(c) + _atm_mass(c)
    rtol           = abs(computed_total - expected_input) / abs(expected_input)

    print(f'  CDR input:             {expected_input:.6e} µmol')
    print(f'  ocean ∆CT mass:       {_ct_mass(c):.6e} µmol')
    print(f'  atm   ∆xCO2 mass:      {_atm_mass(c):.6e} µmol')
    print(f'  total (ocean + atm):   {computed_total:.6e} µmol')
    print(f'  relative error:        {rtol:.2e}')
    passed &= _check(rtol < 1e-6,
                     f'total carbon conserved to rtol < 1e-6: rtol = {rtol:.2e}')

    # Physical sanity: some CT must have outgassed → ∆xCO2 > 0 and ∆CT < CDR input
    passed &= _check(_atm_mass(c) > 0,
                     'air-sea exchange active: ∆xCO2 > 0 after CT injection')
    passed &= _check(_ct_mass(c) < expected_input,
                     'air-sea exchange active: ocean ∆CT < total input (some outgassed)')

    return passed


# ── Test 4: AT mass conservation ─────────────────────────────────────────────

def test_at_conservation():
    """AT mass is exactly the integral of the AT source, for any value of k.

    With k=0: ∆CT and ∆xCO2 stay zero (no coupling).
    With k>0: AT injection draws down atmospheric CO2 (OAE effect; ∆xCO2 < 0)
              and ∆xCO2·Ma + sum(∆CT·V·ρ) = 0 (carbon is transferred, not created).
    """
    print('\n--- AT mass conservation ---')
    passed = True

    q = np.zeros(2*m + 1)
    q[m+1:] = 1.0   # uniform AT source [µmol kg^-1 yr^-1]

    c_k0 = _run(A_k0, q)
    c_k  = _run(A,    q)
    expected = N_steps * dt * float(np.dot(q[m+1:], V * rho))

    for label, c in [('k=0', c_k0), ('k>0', c_k)]:
        computed = _at_mass(c)
        rtol     = abs(computed - expected) / abs(expected)
        print(f'  [{label}] expected AT mass: {expected:.6e} µmol')
        print(f'  [{label}] computed AT mass: {computed:.6e} µmol')
        print(f'  [{label}] relative error:   {rtol:.2e}')
        passed &= _check(rtol < 1e-6,
                         f'[{label}] AT mass conserved to rtol < 1e-6: rtol = {rtol:.2e}')

    # k=0: all coupling terms vanish — ∆xCO2 and ∆CT must stay exactly zero
    passed &= _check(abs(c_k0[0]) < 1e-20,
                     f'[k=0] ∆xCO2 stays zero with only AT source: |∆xCO2| = {abs(c_k0[0]):.2e}')
    passed &= _check(abs(_ct_mass(c_k0)) < 1e-6,
                     f'[k=0] ∆CT mass stays zero with only AT source: {abs(_ct_mass(c_k0)):.2e}')

    # k>0: alkalinity addition draws down atmospheric CO2 (OAE effect)
    # A02 = gammax·ρ·R_A/β_A < 0 (since R_A < 0) → rising ∆AT drives ∆xCO2 negative
    passed &= _check(c_k[0] < 0,
                     f'[k>0] ∆xCO2 < 0 after AT injection (OAE CO2 drawdown): ∆xCO2 = {c_k[0]:.4e}')

    # k>0: carbon is transferred from atmosphere to ocean, not created
    # → ∆xCO2·Ma + sum(∆CT·V·ρ) = 0 exactly (no CT/xCO2 source term)
    ct_plus_atm = _ct_mass(c_k) + _atm_mass(c_k)
    norm        = max(abs(_ct_mass(c_k)), abs(_atm_mass(c_k)))
    rtol_carbon = abs(ct_plus_atm) / norm
    print(f'  [k>0] ocean ∆CT: {_ct_mass(c_k):.4e}, atm ∆xCO2: {_atm_mass(c_k):.4e}, sum: {ct_plus_atm:.4e}')
    passed &= _check(rtol_carbon < 1e-6,
                     f'[k>0] ∆xCO2·Ma + sum(∆CT·V·ρ) = 0 to rtol < 1e-6: rtol = {rtol_carbon:.2e}')

    return passed


# ── Test 5: linearity ─────────────────────────────────────────────────────────

def test_linearity():
    """With frozen A, the system is linear: 2×q → 2×c; c(q1+q2) = c(q1)+c(q2)."""
    print('\n--- linearity ---')
    passed = True

    # Scaling: 2×q → 2×c
    q1 = np.zeros(2*m + 1)
    q1[2]     = 5.0   # CT source at cell 1
    q1[m + 3] = 3.0   # AT source at cell 2

    c1 = _run(A, q1)
    c2 = _run(A, 2.0 * q1)

    nonzero = np.abs(2 * c1) > 1e-30
    max_scale_err = float(np.max(np.abs(c2[nonzero] - 2*c1[nonzero]) / np.abs(2*c1[nonzero])))
    print(f'  max relative error (2×q → 2×c):            {max_scale_err:.2e}')
    passed &= _check(max_scale_err < 1e-5,
                     f'2×q → 2×c to rtol < 1e-5: max rel err = {max_scale_err:.2e}')

    # Superposition: c(q1+q2) = c(q1) + c(q2)
    q2 = np.zeros(2*m + 1)
    q2[4]     = 2.0   # CT source at cell 3
    q2[m + 1] = 7.0   # AT source at cell 0

    c_sum = _run(A, q1 + q2)
    c_q1  = _run(A, q1)
    c_q2  = _run(A, q2)

    combined    = c_q1 + c_q2
    nonzero_sup = np.abs(combined) > 1e-30
    max_sup_err = float(np.max(np.abs((c_sum - combined)[nonzero_sup] / combined[nonzero_sup])))
    print(f'  max relative error (superposition):         {max_sup_err:.2e}')
    passed &= _check(max_sup_err < 1e-5,
                     f'superposition c(q1+q2)=c(q1)+c(q2) to rtol < 1e-5: max rel err = {max_sup_err:.2e}')

    return passed


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('zero-forcing stability',          test_zero_forcing),
        ('CT mass conservation (k=0)',     test_ct_conservation_no_exchange),
        ('total carbon conservation (k>0)', test_total_carbon_conservation),
        ('AT mass conservation',            test_at_conservation),
        ('linearity',                       test_linearity),
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
