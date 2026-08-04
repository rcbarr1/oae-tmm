"""
clamp_pH: Clamp surface pH at preindustrial by adding or removing AT.

At each timestep, solves for the AT needed to hold every surface cell
(within the mixed layer) at its preindustrial pH and applies that as the
CDR flux. Unlike max_AT, cells where pH exceeds preindustrial (due to AT
mixing from adjacent cells) have AT removed to hold pH at the preindustrial
target. NaOH/HCl is assumed (no CT added). Supports multiple time-step
resolutions and SSP scenarios.

CLI usage:
    python -m experiments.clamp_pH --exp-id 0
    python -m experiments.clamp_pH --list
    python -m experiments.clamp_pH --test --list
    python -m experiments.clamp_pH --test --exp-id 0-6
    python -m experiments.clamp_pH --exp-id 2 3 --resume --date 2026-07-24
"""

import gc
from datetime import datetime

import jax
import numpy as np
import PyCO2SYS as pyco2

from experiments.base import BaseExperiment, ExperimentConfig, run_cli
from oae_tmm import loaders
from oae_tmm.grid import flatten


class ClampPH(BaseExperiment):
    """Clamp surface pH at preindustrial by adding or removing AT.

    At each timestep after start_CDR, solves for the AT required to hold
    each masked surface cell at preindustrial pH given the current CT, then
    applies that as a flux. Unlike MaxAT, cells where AT_desired < AT_current
    (pH above preindustrial) have AT removed to pull pH back to target. No CT
    is added or removed (NaOH/HCl assumption).
    """

    def make_q(self, time_current: float, chem: dict, dt: float) -> np.ndarray:
        """Add or remove AT to hold masked cells at preindustrial pH; no CT change."""
        q = np.zeros(1 + 2 * self.m)
        co2sys_desired = pyco2.sys(
            dic=chem['CT_current'], pH=self.pH_preind,
            salinity=self.salinity, temperature=self.temperature, pressure=self.grid['pressure'],
            total_silicate=self.silicate, total_phosphate=self.phosphate,
        )
        AT_desired = co2sys_desired['alkalinity']
        del co2sys_desired
        gc.collect()
        jax.clear_caches()

        AT_to_add = (AT_desired - chem['AT_current']) * self.cfg.q_AT_mask
        q[(self.m+1):] = AT_to_add / dt
        return q


def build_experiments(data_path: str, output_path: str, test: bool = False, tag_date: str = None) -> list:
    """Return a list of ClampPH instances covering all parameter combinations.

    In test mode: 7 timestep resolutions × scenario='none', 5-year runs.
    Used to compare how timestep size affects the result (use --test --list
    to see all experiments; --test --exp-id 0-6 to run).

    In production mode: long (daily for 70 years), for each SSP scenario.
    """
    grid    = loaders.load_ocim(data_path)
    ocnmask = grid['ocnmask']

    q_AT_mask = flatten(grid['mldmask'], ocnmask)

    start_CDR = 2030.0

    if tag_date is None:
        tag_date = datetime.now().strftime('%Y-%m-%d')

    if test:
        t0 = np.arange(2030, 2030.25, 1/360)           # daily, first 90 days
        t1 = np.arange(2030.25, 2035.084, 1/12)        # monthly
        mixed = np.concatenate((t0, t1))

        t_configs = [
            ('annually', np.arange(2030, 2036,      1.0   )),
            ('monthly',  np.arange(2030, 2035.084,  1/12  )),
            ('dekadal',  np.arange(2030, 2035.028,  1/36  )),
            ('pentadal', np.arange(2030, 2035.014,  1/72  )),
            ('daily',    np.arange(2030, 2035.003,  1/360 )),
            ('hourly',   np.arange(2030, 2035.0002, 1/8640)),
            ('mixed',    mixed),
        ]
        scenarios = ['none']
    else:
        t_configs = [
            ('long', np.arange(2030, 2100, 1/360)),
        ]
        scenarios = ['none', 'ssp126', 'ssp245', 'ssp534_OS']

    experiments = []
    for t_name, time in t_configs:
        for scenario in scenarios:
            tag = f'{tag_date}_{t_name}_{scenario}'
            cfg = ExperimentConfig(
                data_path          = data_path,
                output_path        = output_path + f'clamp_pH_{tag}.nc',
                scenario           = scenario,
                time               = time,
                max_steps_per_file = 2000,
                start_CDR          = start_CDR,
                q_AT_mask          = q_AT_mask,
                attrs              = {'experiment': 'clamp_pH', 'scenario': scenario, 'tag': tag},
            )
            experiments.append(ClampPH(cfg))
    return experiments


def main():
    run_cli(build_experiments, 'ClampPH: clamp surface pH at preindustrial by adding or removing AT')


if __name__ == '__main__':
    main()
