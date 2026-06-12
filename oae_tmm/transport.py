"""
Transport matrix assembly and implicit Euler solver for oae-tmm.

Provides two public functions:

  build_A_matrix — assembles the 2m+1 × 2m+1 sparse block matrix A that
      describes ocean transport and linearized air-sea CO2 exchange for one
      timestep. The carbonate chemistry parameters (R_C, R_A, CT, AT, aqueous_CO2,
      K0) change each timestep as CT and AT evolve, so this function is
      called inside the time loop by each experiment.

  solve_timestep — advances the state vector by one implicit Euler step using
      PETSc LGMRES.

Governing equations encoded in A (units in brackets):

    1. d(∆xCO2)/dt = ∆q_sea-air,xCO2                              [µmol CO2 (µmol air)^-1 yr^-1] (same as [µatm CO2 (µatm air)^-1 yr^-1])
    2. d(∆CT)/dt   = TR * ∆CT + ∆q_air-sea,CT + ∆q_CDR,CT        [µmol CT (kg seawater)^-1 yr^-1]
    3. d(∆AT)/dt   = TR * ∆AT  + ∆q_CDR,AT                        [µmol AT (kg seawater)^-1 yr^-1]

Air-sea fluxes depend on ∆c (not on external forcing), so they fold into A
rather than the source vector q:

    ∆q_sea-air,xCO2 = gammax * (rho * R_C * ∆CT/beta_C + rho * R_A * ∆AT/beta_A - K0 * Patm * ∆xCO2)
    ∆q_air-sea,CT   = gammaC * (R_C * ∆CT/beta_C + R_A * ∆AT/beta_A - K0 * Patm/rho * ∆xCO2)

    where gammax = k * V * (1 - f_ice) / Ma / z1
          gammaC = -k * (1 - f_ice) / z1

The time-stepping loop itself lives in experiments/base.py rather than here
because rebuilding A requires the current carbonate chemistry state, which
is computed at the experiment level.

Carbonate chemistry linearization similar to Nowicki et al. (2024).
Transport matrix setup similar to Yamamoto et al. (2024).
"""

from typing import Optional

import numpy as np
from scipy import sparse
from petsc4py import PETSc


def build_A_matrix(
    TR,
    k: np.ndarray,
    f_ice: np.ndarray,
    V: np.ndarray,
    R_C: np.ndarray,
    R_A: np.ndarray,
    CT: np.ndarray,
    AT: np.ndarray,
    aqueous_CO2: np.ndarray,
    K0: np.ndarray,
    z1: float,
    rho: float = 1025.0,
    Patm: float = 1e6,
    Ma: float = 1.8e26,
) -> sparse.csr_matrix:
    """Assemble the 2m+1 × 2m+1 sparse block matrix A.

    Encodes ocean carbon transport (TR) and linearized air-sea CO2 exchange
    for state vector c = [∆xCO2 [µmol CO2 (µmol air)^-1], ∆CT (m cells) [µmol kg^-1], ∆AT (m cells) [µmol kg^-1]]:

        A = [-gammax * K0 * Patm       | gammax * rho * R_C / beta_C | gammax * rho * R_A / beta_A]
            [-gammaC * K0 * Patm / rho | TR + gammaC * R_C / beta_C  | gammaC * R_A / beta_A      ]
            [0                         | 0                           | TR                         ]

    where beta_C = CT/aqueous_CO2 and beta_A = AT/aqueous_CO2 are computed
    internally (unitless).

    gammax = k * V * (1 - f_ice) / Ma / z1 and gammaC = -k * (1 - f_ice) / z1
    are local variables that group the air-sea exchange terms; they have no
    physical meaning beyond simplifying the algebra and are not returned.

    k and f_ice must be zero in subsurface cells so that air-sea exchange only
    acts on surface cells.
    
    All input arrays (except TR) are flattened, i.e. using grid.flatten(),
    such that they are of shape (m,), not the 3D verisons with shape (nlat,
    nlon, ndepth).

    Parameters
    ----------
    TR : scipy.sparse matrix
        OCIM2-48L transport matrix, shape (m, m) [yr^-1].
    k : np.ndarray
        CO2 piston velocity [m yr^-1], shape (m,). Zero for subsurface cells.
    f_ice : np.ndarray
        Sea ice fraction [0–1], shape (m,). Zero for subsurface cells.
    V : np.ndarray
        Ocean cell volumes [m^3], shape (m,).
    R_C : np.ndarray
        Revelle buffer factor (dpCO2/pCO2)/(dCT/CT) [unitless], shape (m,).
    R_A : np.ndarray
        Alkalinity buffer factor (dpCO2/pCO2)/(dAT/AT) [unitless], shape (m,).
    CT : np.ndarray
        Dissolved inorganic carbon [µmol kg^-1], shape (m,).
    AT : np.ndarray
        Total alkalinity [µmol kg^-1], shape (m,).
    aqueous_CO2 : np.ndarray
        Aqueous CO2 concentration [µmol kg^-1], shape (m,).
    K0 : np.ndarray
        CO2 solubility [µmol CO2 m^-3 (µatm CO2)^-1], shape (m,).
    z1 : float
        Depth of the first model layer [m].
    rho : float, optional
        Seawater density [kg m^-3]. Default 1025.
    Patm : float, optional
        Atmospheric pressure [µatm]. Default 1e6.
    Ma : float, optional
        Micromoles of air in the atmosphere [µmol air]. Default 1.8e26.

    Returns
    -------
    scipy.sparse.csr_matrix
        Block matrix A, shape (2m+1, 2m+1).
    """
    m = TR.shape[0]

    beta_C = CT / aqueous_CO2
    beta_A = AT / aqueous_CO2

    # air-sea exchange coefficients (zero for subsurface cells since k=0 there)
    gammax = k * V * (1 - f_ice) / Ma / z1
    gammaC = -k * (1 - f_ice) / z1

    # Block structure — c = [∆xCO2 (1) [µmol CO2 (µmol air)^-1], ∆CT (m) [µmol kg^-1], ∆AT (m) [µmol kg^-1]], total length 2m+1:
    #
    #   A = [ A00: 1×1  | A01: 1×m  | A02: 1×m  ]  ← calculates d(∆xCO2)/dt
    #       [ A10: m×1  | A11: m×m  | A12: m×m  ]  ← calculates d(∆CT)/dt
    #       [ A20: m×1  | A21: m×m  | A22: m×m  ]  ← calculates d(∆AT)/dt
    #
    #   A00 = -gammax * K0 * Patm             scalar (summed over surface cells)
    #   A01 =  gammax * rho * R_C / beta_C    1×m row vector
    #   A02 =  gammax * rho * R_A / beta_A    1×m row vector
    #   A10 = -gammaC * K0 * Patm / rho       m×1 column vector
    #   A11 =  TR + diag(gammaC * R_C / beta_C)   m×m sparse
    #   A12 =  diag(gammaC * R_A / beta_A)         m×m sparse
    #   A20 =  0                               m×1 zeros
    #   A21 =  0                               m×m zeros
    #   A22 =  TR                              m×m sparse

    # --- row 0: ∆xCO2 equation ---
    A00 = -Patm * np.sum(gammax * K0)                      # scalar
    A01 = gammax * rho * R_C / beta_C                      # (m,)
    A02 = gammax * rho * R_A / beta_A                      # (m,)

    A0_ = np.empty(1 + 2 * m)
    A0_[0] = A00
    A0_[1:(m+1)] = A01
    A0_[(m+1):] = A02

    # --- row 1: ∆CT equation ---
    A10 = -gammaC * K0 * Patm / rho                                          # (m,)
    A11 = TR + sparse.diags(gammaC * R_C / beta_C, format='csr')
    A12 = sparse.diags(gammaC * R_A / beta_A)

    A1_ = sparse.hstack([
        sparse.csr_matrix(A10[:, np.newaxis]), A11, A12
    ])

    # --- row 2: ∆AT equation (pure transport, no air-sea exchange) ---
    A2_ = sparse.hstack([
        sparse.csr_matrix(np.zeros((m, 1))), 0 * TR, TR
    ])

    return sparse.vstack([  # type: ignore[return-value]
        sparse.csr_matrix(A0_[np.newaxis, :]),
        A1_,
        A2_,
    ], format='csr')


_DEFAULT_SOLVER_OPTS = {
    'type': 'lgmres',
    'restart': 30,
    'preconditioner_type': 'bjacobi',
    'rtol': 1e-8,
    'atol': 1e-10,
    'max_it': 1000,
}


def solve_timestep(
    A: sparse.csr_matrix,
    c_prev: np.ndarray,
    q: np.ndarray,
    dt: float,
    solver_opts: Optional[dict] = None,
) -> np.ndarray:
    """Advance the state vector by one implicit Euler step.

    Solves (I - dt*A) * c_next = c_prev + dt*q using PETSc LGMRES with
    block Jacobi preconditioning. Uses c_prev as the initial guess.
    This is the Euler backward method. 

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Block matrix from build_A_matrix, shape (2m+1, 2m+1).
    c_prev : np.ndarray
        State vector at the previous time step, shape (2m+1,).
    q : np.ndarray
        Source/sink flux vector, shape (2m+1,). q[0] [µmol CO2 (µmol air)^-1 yr^-1],
        q[1:(m+1)] [µmol CT kg^-1 yr^-1], q[(m+1):] [µmol AT kg^-1 yr^-1].
    dt : float
        Time step [yr].
    solver_opts : dict, optional
        Override default PETSc solver settings. Recognised keys:
        'type', 'restart', 'preconditioner_type', 'rtol', 'atol', 'max_it'.

    Returns
    -------
    np.ndarray
        State vector at the next time step, shape (2m+1,).

    Raises
    ------
    RuntimeError
        If PETSc LGMRES fails to converge.
    """
    opts = {**_DEFAULT_SOLVER_OPTS, **(solver_opts or {})}

    # set up left and right hand sides according to Euler backward
    LHS = sparse.eye(A.shape[0], format='csr') - dt * A
    RHS = c_prev + dt * q

    # convert matricies from SciPy sparse to PETSc to parallelize
    LHS_petsc = PETSc.Mat().createAIJ(  # type: ignore[attr-defined]
        size=LHS.shape, csr=(LHS.indptr, LHS.indices, LHS.data)
    )
    RHS_petsc = PETSc.Vec().createWithArray(RHS)  # type: ignore[attr-defined]

    # set up PETSc solver
    ksp = PETSc.KSP().create()  # type: ignore[attr-defined]
    ksp.setOperators(LHS_petsc)
    ksp.setType(opts['type'])
    ksp.setGMRESRestart(opts['restart'])

    # set up preconditioner
    ksp.getPC().setType(opts['preconditioner_type'])

    # set convergence tolerances
    ksp.setTolerances(rtol=opts['rtol'], atol=opts['atol'], max_it=opts['max_it'])

    # set up output array (PETSc vector object) and initial guess
    c_next_petsc = LHS_petsc.createVecRight()
    c_next_petsc.setArray(c_prev.copy())
    ksp.setInitialGuessNonzero(True)

    ksp.solve(RHS_petsc, c_next_petsc)

    # check for convergence
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise RuntimeError(
            f'PETSc LGMRES failed to converge: reason={reason}, '
            f'iterations={ksp.getIterationNumber()}, '
            f'residual={ksp.getResidualNorm():.2e}'
        )

    return c_next_petsc.array.copy()
