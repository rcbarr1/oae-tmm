"""
Base experiment class for oae-tmm.

Defines ExperimentConfig (a dataclass for all shared simulation parameters)
and BaseExperiment (a class encapsulating all shared setup and time-stepping
logic). Individual experiment files subclass BaseExperiment and override
make_q() to supply the CDR source/sink schedule.

Usage pattern:

    from experiments.base import BaseExperiment, ExperimentConfig

    class MyExperiment(BaseExperiment):
        def make_q(self, t_current, chem, dt):
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

import argparse
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
    start_CDR : float, optional
        Calendar year at which CDR deployment begins. make_q() returns zeros
        before this year. Default None means CDR is active from the first step.
    q_AT_mask : np.ndarray, optional
        Pre-flattened (m,) mask controlling where AT is added (1 = add, 0 = skip).
        Built in build_experiments() from MLD or region shapefiles and passed
        through config so all experiment parameters live in one place.
    attrs : dict, optional
        Experiment metadata written as global attributes to the output NetCDF
        file, e.g. {'scenario': 'ssp245', 'AT_added_umol_kg': 50.0}.
    """
    data_path:           str
    output_path:         str
    scenario:            str
    start_year:          float
    times:               np.ndarray
    output_freq:         int   = 1
    max_steps_per_file:  int   = 0
    start_CDR:           float      = None
    q_AT_mask:           np.ndarray = None
    attrs:               dict       = None


class BaseExperiment:
    """Shared setup and time-stepping logic for all oae-tmm experiments.

    Subclasses must override make_q() to define the CDR perturbation. All
    other simulation infrastructure (data loading, preindustrial chemistry,
    A-matrix assembly, PETSc solve, NetCDF output) is inherited from here.

    After setup() runs, the following attributes are available in make_q():

    self.m          int             number of ocean grid cells
    self.grid       dict            OCIM grid (see loaders.load_ocim)
    self.surf       dict            surface fields (m,) vectors, 0 at depth
                                    (see loaders.load_ncep_noaa)
    self.AT         np.ndarray (m,) GLODAP gridded total alkalinity [µmol kg^-1]
    self.T          np.ndarray (m,) temperature [°C]
    self.S          np.ndarray (m,) salinity [unitless]
    self.Si         np.ndarray (m,) silicate [µmol kg^-1]
    self.P          np.ndarray (m,) phosphate [µmol kg^-1]
    self.DIC_preind np.ndarray (m,) preindustrial DIC [µmol kg^-1]
    self.Canth      np.ndarray (m,) anthropogenic carbon [µmol kg^-1]
                                    (updated each step in run() when scenario != 'none')
    self.pH_preind  np.ndarray (m,) preindustrial pH [total scale]
    self.k          np.ndarray (m,) piston velocity, zero at depth [m yr^-1]
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg        = config
        self.m          = None
        self.grid       = None
        self.surf       = None
        self.AT         = None
        self.T          = None
        self.S          = None
        self.Si         = None
        self.P          = None
        self.DIC_preind = None
        self.Canth      = None
        self.pH_preind  = None
        self.k          = None

    def setup(self):
        """Load data and compute preindustrial chemistry.

        Populates all self.* fields needed by run() and make_q().
        Called automatically by run() — no need to call it separately unless
        you want to inspect the setup state before running.
        """
        self.grid = loaders.load_ocim(self.cfg.data_path)
        ocnmask   = self.grid['ocnmask']
        self.m    = self.grid['TR'].shape[0]

        glodap    = loaders.load_glodap(self.cfg.data_path, ocnmask)
        self.surf = loaders.load_ncep_noaa(self.cfg.data_path, ocnmask)

        # GLODAP fields — already flattened to (m,) by load_glodap
        self.AT = glodap['AT']
        self.T  = glodap['T']
        self.S  = glodap['S']
        self.Si = glodap['Si']
        self.P  = glodap['P']

        # piston velocity: calc_piston_velocity receives (m,) vectors with 0 at depth;
        # result is also (m,) with 0 at non-surface cells
        self.k     = chemistry.calc_piston_velocity(self.surf['sst'], self.surf['wspd'])

        # preindustrial DIC: GLODAP DIC minus Canth at 2002 (TRACE historical)
        Canth_2002 = flatten(
            trace.interp_trace(
                self.cfg.data_path, 2002, 'none',
                self.grid['model_lat'], self.grid['model_lon'],
                self.grid['model_depth'], ocnmask,
            ),
            ocnmask,
        )
        self.DIC_preind = glodap['DIC'] - Canth_2002

        # Canth at the simulation start year (updated each step during run)
        self.Canth = flatten(
            trace.interp_trace(
                self.cfg.data_path, self.cfg.start_year, self.cfg.scenario,
                self.grid['model_lat'], self.grid['model_lon'],
                self.grid['model_depth'], ocnmask,
            ),
            ocnmask,
        )

        # preindustrial pH from GLODAP DIC_preind + GLODAP AT
        # (used in subclasses for CDR targeting)
        co2sys_preind = pyco2.sys(
            dic=self.DIC_preind, alkalinity=self.AT,
            salinity=self.S, temperature=self.T, pressure=self.grid['pressure'],
            total_silicate=self.Si, total_phosphate=self.P,
        )
        self.pH_preind = co2sys_preind['pH']
        del co2sys_preind
        gc.collect()
        jax.clear_caches()


    def _calc_step_chemistry(self, c: np.ndarray) -> dict:
        """Compute carbonate chemistry for the current state vector c.

        Runs PyCO2SYS twice across all ocean cells: once for the unperturbed
        state (pCO2, aqueous_CO2, R_C) and once with a small AT perturbation
        (R_A via numerical differentiation). Surface-only air-sea gas exchange
        is enforced downstream through terms k and f_ice which are zero at depth.

        Parameters
        ----------
        c : np.ndarray
            Current state vector [∆xCO2, ∆DIC (m), ∆AT (m)], shape (2m+1,).

        Returns
        -------
        dict with keys:
            pCO2        : np.ndarray (m,)  current pCO2 [µatm]
            aqueous_CO2 : np.ndarray (m,)  aqueous CO2 [µmol kg^-1]
            R_C         : np.ndarray (m,)  Revelle buffer factor
            R_A         : np.ndarray (m,)  Alkalinity buffer factor
            K0          : np.ndarray (m,)  CO2 solubility [(µmol CO2) m^-3 (µatm CO2)^-1]
            DIC_current : np.ndarray (m,)  DIC_preind + ∆DIC + Canth [µmol kg^-1]
            AT_current  : np.ndarray (m,)  AT + ∆AT [µmol kg^-1]
        """
        m = self.m

        AT_current  = self.AT + c[(m+1):]
        DIC_current = self.DIC_preind + c[1:(m+1)] + self.Canth

        co2sys = pyco2.sys(
            dic=DIC_current, alkalinity=AT_current,
            salinity=self.S, temperature=self.T, pressure=self.grid['pressure'],
            total_silicate=self.Si, total_phosphate=self.P,
        )
        pCO2        = co2sys['pCO2']           # [µatm]
        aqueous_CO2 = co2sys['CO2']            # [µmol kg^-1]
        R_C         = co2sys['revelle_factor']
        del co2sys

        # R_A: (dpCO2/pCO2) / (dAT/AT) — numerical differentiation;
        co2sys_pert = pyco2.sys(
            dic=DIC_current, alkalinity=AT_current + 1e-6,
            salinity=self.S, temperature=self.T, pressure=self.grid['pressure'],
            total_silicate=self.Si, total_phosphate=self.P,
        )
        pCO2_pert = co2sys_pert['pCO2']
        del co2sys_pert

        R_A = ((pCO2_pert - pCO2) / pCO2) / (1e-6 / self.AT)

        # Solubility constant [(µmol CO2) m^-3 (µatm CO2)^-1]
        # Based on Weiss (1974) https://doi.org/10.1016/0304-4203(74)90015-2
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

    def make_q(self, t_current: float, chem: dict, dt: float) -> np.ndarray:
        """CDR source/sink flux vector, shape (2m+1,). Must be overridden.

        q[0] [µmol CO2 (µmol air)^-1 yr^-1], q[1:(m+1)] [µmol DIC kg^-1 yr^-1],
        q[(m+1):] [µmol AT kg^-1 yr^-1].
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

        file_number         = 0
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
            # write initial conditions (all zeros) at t=0
            output.write_simulation_step(ds, c, np.zeros_like(c), times[0] + self.cfg.start_year, ocnmask)
            write_count_in_file += 1

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
                    self.grid['TR'], self.k, self.surf['f_ice'], self.grid['model_vols'],
                    chem['R_C'], chem['R_A'],
                    chem['DIC_current'], chem['AT_current'],
                    chem['aqueous_CO2'], chem['K0'],
                    self.grid['z1'],
                    rho=self.grid['rho'],
                )

                if self.cfg.start_CDR is None or t_current >= self.cfg.start_CDR:
                    q = self.make_q(t_current, chem, dt)
                else:
                    q = np.zeros(1 + 2 * self.m)
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


def run_cli(build_experiments, description: str = ''):
    """Shared CLI entry point for all experiment modules.

    Handles --list, --test, and --exp-id (with range syntax, e.g. 0 2 5-10).
    Call from each experiment's main():

        def main():
            run_cli(build_experiments, 'Exp23: ...')
    """
    def _parse_exp_ids(values):
        ids = set()
        for v in values:
            if '-' in v:
                start, end = map(int, v.split('-'))
                ids.update(range(start, end + 1))
            else:
                ids.add(int(v))
        return sorted(ids)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--exp-id', nargs='+', help='experiment index or range (e.g. 0 2 5-10)')
    parser.add_argument('--list',   action='store_true', help='list all experiments and exit')
    parser.add_argument('--test',   action='store_true', help='run a short test experiment')
    args = parser.parse_args()

    data_path   = './data/'
    output_path = './outputs/'

    experiments = build_experiments(data_path, output_path, test=args.test)

    if args.list:
        print(f'total experiments: {len(experiments)}')
        for i, exp in enumerate(experiments):
            print(f'  {i}: {exp.cfg.output_path}')
        return

    if args.test:
        experiments[0].run()
        return

    if args.exp_id is None:
        parser.error('--exp-id is required; use --list to see all experiments')

    exp_ids = _parse_exp_ids(args.exp_id)
    invalid = [i for i in exp_ids if not (0 <= i < len(experiments))]
    if invalid:
        parser.error(f'invalid exp-id(s): {invalid}; must be 0 to {len(experiments) - 1}')

    for exp_id in exp_ids:
        experiments[exp_id].run()
