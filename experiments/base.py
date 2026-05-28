"""
Base experiment class for oae-tmm.

Defines ExperimentConfig (a dataclass for all shared simulation parameters)
and BaseExperiment (a class encapsulating all shared setup and time-stepping
logic). Individual experiment files subclass BaseExperiment and override
make_q() to supply the CDR source/sink schedule.

Usage pattern:

    from experiments.base import BaseExperiment, ExperimentConfig

    class MyExperiment(BaseExperiment):
        def make_q(self, t_current, c, chem, dt):
            q = np.zeros(1 + 2 * self.m)
            # ... experiment-specific CDR source logic ...
            return q

    cfg = ExperimentConfig(
        data_path   = './data/',
        output_path = './outputs/my_experiment.nc',
        scenario    = 'ssp245',
        start_year  = 2020.0,
        times       = np.arange(0, 20, 1/12),   # 20 years, monthly steps
        output_freq = 1,
        attrs       = {'experiment': 'my_experiment', 'AT_added_umol_kg': 50.0},
    )
    MyExperiment(cfg).run()

The time-stepping loop in run() does the following at each step:
  1. Updates Canth from TRACE (if scenario != 'none').
  2. Recomputes carbonate chemistry from the current state vector c.
  3. Rebuilds the A matrix (chemistry parameters change each step as DIC/AT evolve).
  4. Calls make_q() to get the CDR source/sink for this step.
  5. Advances c by one implicit Euler step (via transport.solve_timestep).
  6. Accumulates output at the frequency specified by output_freq.
Output is written once to output_path after the loop completes.

Carbonate chemistry linearization similar to Nowicki et al. (2024).
Transport matrix setup similar to Yamamoto et al. (2024).
"""

import dataclasses
import gc

import numpy as np
import PyCO2SYS as pyco2
import jax
from tqdm import tqdm

from oae_tmm import chemistry, loaders, output, trace, transport
from oae_tmm.grid import flatten


@dataclasses.dataclass
class ExperimentConfig:
    """Shared configuration for all oae-tmm experiments.

    Attributes
    ----------
    data_path : str
        Path to the directory containing OCIM2-48L_base/, TRACE_gridded/, and
        other pre-processed data products.
    output_path : str
        Path for the output NetCDF file (including .nc extension). When
        max_steps_per_file > 0, this is treated as a base path: the .nc
        suffix is stripped and files are named {base}_{000}.nc, {base}_{001}.nc,
        etc.
    scenario : str
        Emissions scenario for both atmospheric CO2 forcing and TRACE Canth
        interpolation. One of: 'none', 'ssp119', 'ssp126', 'ssp245', 'ssp370',
        'ssp370_lowNTCF', 'ssp434', 'ssp460', 'ssp534_OS'.
        'none' holds atmospheric CO2 fixed at start_year and uses the TRACE
        historical (scenario 1) product for Canth.
    start_year : float
        Calendar year corresponding to times[0] (decimal years CE).
    times : np.ndarray
        1D array of time offsets from start_year [yr], e.g.
        np.arange(0, 20, 1/12) for 20 years at monthly resolution.
        The simulation starts at times[0] (initial conditions) and steps
        forward through times[1:].
    output_freq : int, optional
        Write output every N timesteps (default 1 = every step).
    max_steps_per_file : int, optional
        Maximum number of output records per NetCDF file (default 0 = no
        limit, single file). Set to e.g. 2000 for long hourly runs to keep
        individual file sizes manageable and limit data loss on job failure.
    attrs : dict, optional
        Experiment metadata written as global attributes to the output NetCDF
        file, e.g. {'scenario': 'ssp245', 'AT_added_umol_kg': 50.0}.
    """
    data_path:           str
    output_path:         str
    scenario:            str
    start_year:          float
    times:               np.ndarray
    output_freq:         int  = 1
    max_steps_per_file:  int  = 0
    attrs:               dict = None


class BaseExperiment:
    """Shared setup and time-stepping logic for all oae-tmm experiments.

    Subclasses must override make_q() to define the CDR perturbation. All
    other simulation infrastructure (data loading, preindustrial chemistry,
    A-matrix assembly, PETSc solve, NetCDF output) is inherited from here.

    After setup() runs, the following attributes are available in make_q():

    self.m          int             number of ocean grid cells
    self.grid       dict            OCIM grid (see loaders.load_ocim)
    self.surf       dict            surface fields (see loaders.load_ncep_noaa)
    self.AT         np.ndarray (m,) GLODAP total alkalinity [µmol kg^-1]
    self.T          np.ndarray (m,) temperature [°C]
    self.S          np.ndarray (m,) salinity [unitless]
    self.Si         np.ndarray (m,) silicate [µmol kg^-1]
    self.P          np.ndarray (m,) phosphate [µmol kg^-1]
    self.pressure   np.ndarray (m,) pressure [dbar ≈ m below surface]
    self.DIC_preind np.ndarray (m,) preindustrial DIC [µmol kg^-1]
    self.Canth      np.ndarray (m,) anthropogenic carbon [µmol kg^-1]
                                    (updated each step in run() when scenario != 'none')
    self.pH_preind  np.ndarray (m,) preindustrial pH [total scale]
    self.k          np.ndarray (m,) piston velocity, zero at depth [m yr^-1]
    self.f_ice      np.ndarray (m,) ice fraction, zero at depth [0–1]
    self.V          np.ndarray (m,) cell volumes [m^3]
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg   = config
        self.m          = None
        self.grid       = None
        self.surf       = None
        self.AT         = None
        self.T          = None
        self.S          = None
        self.Si         = None
        self.P          = None
        self.pressure   = None
        self.DIC_preind = None
        self.Canth      = None
        self.pH_preind  = None
        self.k          = None
        self.f_ice      = None
        self.V          = None

    def setup(self):
        """Load data and compute preindustrial chemistry.

        Populates all self.* fields needed by run() and make_q().
        Called automatically by run() — no need to call it separately unless
        you want to inspect the setup state before running.
        """
        self.grid = loaders.load_ocim(self.cfg.data_path)
        glodap    = loaders.load_glodap(self.cfg.data_path)
        self.surf = loaders.load_ncep_noaa(self.cfg.data_path)

        ocnmask = self.grid['ocnmask']
        self.m  = self.grid['TR'].shape[0]

        # flatten GLODAP fields to ocean-only 1D vectors (m,)
        self.AT = flatten(glodap['AT_3D'], ocnmask)
        self.T  = flatten(glodap['T_3D'],  ocnmask)
        self.S  = flatten(glodap['S_3D'],  ocnmask)
        self.Si = flatten(glodap['Si_3D'], ocnmask)
        self.P  = flatten(glodap['P_3D'],  ocnmask)

        # pressure [dbar]: broadcast depth to 3D then flatten
        depth_3D = np.tile(
            self.grid['model_depth'][np.newaxis, np.newaxis, :],
            (ocnmask.shape[0], ocnmask.shape[1], 1),
        )
        self.pressure = flatten(depth_3D, ocnmask)

        # piston velocity and surface fields expanded to flattened 3D vectors
        # (subsurface cells are 0, not NaN, as required by build_A_matrix)
        k_2D       = chemistry.calc_piston_velocity(self.surf['sst_2D'], self.surf['wspd_2D'])
        self.k     = self._expand_surf_to_flat(k_2D)
        self.f_ice = self._expand_surf_to_flat(self.surf['f_ice_2D'])
        self.V     = flatten(self.grid['model_vols'], ocnmask)

        # preindustrial DIC: GLODAP DIC minus Canth at 2002 (TRACE historical)
        Canth_2002_3D = trace.interp_trace(
            self.cfg.data_path, 2002, 'none',
            self.grid['model_lat'], self.grid['model_lon'],
            self.grid['model_depth'], ocnmask,
        )
        self.DIC_preind = flatten(glodap['DIC_3D'] - Canth_2002_3D, ocnmask)

        # Canth at the simulation start year (updated each step during run)
        Canth_3D = trace.interp_trace(
            self.cfg.data_path, self.cfg.start_year, self.cfg.scenario,
            self.grid['model_lat'], self.grid['model_lon'],
            self.grid['model_depth'], ocnmask,
        )
        self.Canth = flatten(Canth_3D, ocnmask)

        # preindustrial pH from GLODAP DIC_preind + GLODAP AT
        # (used in subclasses for CDR targeting)
        co2sys_preind = pyco2.sys(
            dic=self.DIC_preind, alkalinity=self.AT,
            salinity=self.S, temperature=self.T, pressure=self.pressure,
            total_silicate=self.Si, total_phosphate=self.P,
        )
        self.pH_preind = co2sys_preind['pH']
        del co2sys_preind
        gc.collect()
        jax.clear_caches()

    def _expand_surf_to_flat(self, surf_2D: np.ndarray) -> np.ndarray:
        """Expand a 2D surface field to a flattened 3D vector, zero at depth."""
        ocnmask = self.grid['ocnmask']
        field_3D = np.zeros(ocnmask.shape)
        field_3D[:, :, 0] = surf_2D
        return flatten(field_3D, ocnmask)

    def _calc_step_chemistry(self, c: np.ndarray) -> dict:
        """Compute carbonate chemistry for the current state vector c.

        Runs PyCO2SYS twice: once for all cells (to get pCO2, aqueous_CO2,
        R_C), and once for surface cells only with a small AT perturbation
        (to get R_A via numerical differentiation). Returns all quantities
        needed by build_A_matrix and make_q.

        Parameters
        ----------
        c : np.ndarray
            Current state vector [∆xCO2, ∆DIC (m), ∆AT (m)], shape (2m+1,).

        Returns
        -------
        dict with keys:
            pCO2        : np.ndarray (m,)  current pCO2 [µatm]
            aqueous_CO2 : np.ndarray (m,)  aqueous CO2 [µmol kg^-1]
            R_C         : np.ndarray (m,)  Revelle factor w.r.t. DIC
            R_A         : np.ndarray (m,)  Revelle factor w.r.t. AT (0 at depth)
            K0          : np.ndarray (m,)  CO2 solubility [µmol m^-3 µatm^-1]
            DIC_current : np.ndarray (m,)  DIC_preind + ∆DIC + Canth [µmol kg^-1]
            AT_current  : np.ndarray (m,)  AT + ∆AT [µmol kg^-1]
        """
        m = self.m

        AT_current  = self.AT + c[m + 1:]
        DIC_current = self.DIC_preind + c[1:m + 1] + self.Canth

        co2sys = pyco2.sys(
            dic=DIC_current, alkalinity=AT_current,
            salinity=self.S, temperature=self.T, pressure=self.pressure,
            total_silicate=self.Si, total_phosphate=self.P,
        )
        pCO2        = co2sys['pCO2']           # [µatm]
        aqueous_CO2 = co2sys['CO2']            # [µmol kg^-1]
        R_C         = co2sys['revelle_factor']
        del co2sys

        # R_A: (dpCO2/pCO2) / (dAT/AT) — numerical differentiation at surface only
        # subsurface cells remain NaN (air-sea exchange does not act on them)
        surf_idx = self.grid['surf_idx']
        co2sys_pert = pyco2.sys(
            dic            = DIC_current[surf_idx],
            alkalinity     = AT_current[surf_idx] + 1e-6,
            salinity       = self.S[surf_idx],
            temperature    = self.T[surf_idx],
            pressure       = self.pressure[surf_idx],
            total_silicate = self.Si[surf_idx],
            total_phosphate= self.P[surf_idx],
        )
        pCO2_pert = co2sys_pert['pCO2']
        del co2sys_pert

        # initialize to 0 (not NaN) at depth: air-sea exchange is zero there,
        # so 0 is the correct value when build_A_matrix multiplies R_A by gammaC
        R_A = np.zeros(R_C.shape)
        R_A[surf_idx] = (
            (pCO2_pert - pCO2[surf_idx]) / pCO2[surf_idx]
        ) / (1e-6 / self.AT[surf_idx])

        # Henry's law constant [µmol CO2 m^-3 µatm^-1]
        K0 = aqueous_CO2 / pCO2 * self.grid['rho']

        gc.collect()
        jax.clear_caches()

        return {
            'pCO2':        pCO2,
            'aqueous_CO2': aqueous_CO2,
            'R_C':         R_C,
            'R_A':         R_A,
            'K0':          K0,
            'DIC_current': DIC_current,
            'AT_current':  AT_current,
        }

    def make_q(
        self,
        t_current: float,
        c: np.ndarray,
        chem: dict,
        dt: float,
    ) -> np.ndarray:
        """Return the CDR source/sink flux vector for this timestep.

        Must be overridden in each experiment subclass.

        Parameters
        ----------
        t_current : float
            Current calendar year (start_year + time offset) [decimal years CE].
        c : np.ndarray
            Current state vector [∆xCO2, ∆DIC (m), ∆AT (m)], shape (2m+1,).
        chem : dict
            Carbonate chemistry dict from _calc_step_chemistry. Keys:
            pCO2, aqueous_CO2, R_C, R_A, K0, DIC_current, AT_current.
        dt : float
            Current timestep [yr].

        Returns
        -------
        np.ndarray
            Source/sink flux vector [tracer units yr^-1], shape (2m+1,):
              q[0]        : ∆xCO2 rate [µmol CO2 (µmol air)^-1 yr^-1]
              q[1:m+1]    : ∆DIC rate  [µmol DIC kg^-1 yr^-1]
              q[m+1:2m+1] : ∆AT rate   [µmol AT kg^-1 yr^-1]
        """
        raise NotImplementedError

    def _output_path(self, file_number: int) -> str:
        """Return the output path for a given file number.

        When max_steps_per_file == 0, always returns output_path as-is
        (single file, no suffix). When max_steps_per_file > 0, strips the
        .nc extension and appends _{file_number:03d}.nc so files are named
        {base}_000.nc, {base}_001.nc, etc.
        """
        if self.cfg.max_steps_per_file == 0:
            return self.cfg.output_path
        base = self.cfg.output_path
        if base.endswith('.nc'):
            base = base[:-3]
        return f'{base}_{file_number:03d}.nc'

    def run(self):
        """Run the full simulation and write NetCDF output.

        Calls setup() then steps through self.cfg.times[1:]. At each step,
        Canth is updated from TRACE (if scenario != 'none'), carbonate
        chemistry is recomputed from the current state, the A matrix is
        rebuilt, make_q() is called for the CDR source, and c is advanced
        one implicit Euler step. Only c (current state) and q (current
        source/sink) are held in memory at any time — output is written
        directly to disk at output_freq rather than accumulated in arrays.

        When max_steps_per_file > 0, output is split across multiple files
        named {base}_000.nc, {base}_001.nc, etc. Each file is synced to
        disk every 20 writes to limit data loss on job failure.
        """
        self.setup()

        ocnmask = self.grid['ocnmask']
        times   = self.cfg.times
        dts     = np.diff(times, prepend=np.nan)

        c = np.zeros(1 + 2 * self.m)

        file_number        = 0
        write_count_in_file = 0
        ds = output.open_simulation_output(
            self._output_path(file_number),
            self.grid['model_lat'],
            self.grid['model_lon'],
            self.grid['model_depth'],
            ocnmask,
            attrs=self.cfg.attrs,
        )
        try:
            for i in tqdm(range(1, len(times))):
                dt        = dts[i]
                t_current = times[i] + self.cfg.start_year

                # update Canth from TRACE for the current calendar year
                if self.cfg.scenario != 'none':
                    self.Canth = flatten(
                        trace.interp_trace(
                            self.cfg.data_path, t_current, self.cfg.scenario,
                            self.grid['model_lat'], self.grid['model_lon'],
                            self.grid['model_depth'], ocnmask,
                        ),
                        ocnmask,
                    )

                # recompute carbonate chemistry from current state
                chem = self._calc_step_chemistry(c)

                # rebuild A (chemistry parameters change each step)
                A = transport.build_A_matrix(
                    self.grid['TR'], self.k, self.f_ice, self.V,
                    chem['R_C'], chem['R_A'],
                    chem['DIC_current'], chem['AT_current'],
                    chem['aqueous_CO2'], chem['K0'],
                    self.grid['z1'],
                    rho=self.grid['rho'],
                )

                q = self.make_q(t_current, c, chem, dt)
                c = transport.solve_timestep(A, c, q, dt)

                if i % self.cfg.output_freq == 0:
                    # roll over to a new file if this one is full
                    if (self.cfg.max_steps_per_file > 0 and
                            write_count_in_file >= self.cfg.max_steps_per_file):
                        ds.close()
                        file_number += 1
                        write_count_in_file = 0
                        ds = output.open_simulation_output(
                            self._output_path(file_number),
                            self.grid['model_lat'],
                            self.grid['model_lon'],
                            self.grid['model_depth'],
                            ocnmask,
                            attrs=self.cfg.attrs,
                        )

                    output.write_simulation_step(ds, c, q * dt, t_current, ocnmask)
                    write_count_in_file += 1
                    if write_count_in_file % 20 == 0:
                        ds.sync()
        finally:
            ds.close()
