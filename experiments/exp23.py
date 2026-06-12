"""
Exp23: Maximum alkalinity addition targeting preindustrial surface pH.

At each timestep, solves for the AT needed to return every surface cell
(within the mixed layer) to its preindustrial pH and applies that as the
CDR flux. NaOH is assumed (no CT added). Supports multiple time-step
resolutions and SSP scenarios.

CLI usage:
    python -m experiments.exp23 --exp-id 0
    python -m experiments.exp23 --list
    python -m experiments.exp23 --test
"""

import gc
from datetime import datetime

import jax
import numpy as np
import PyCO2SYS as pyco2

from experiments.base import BaseExperiment, ExperimentConfig, run_cli
from oae_tmm import loaders
from oae_tmm.grid import flatten, make_3d


class Exp23(BaseExperiment):
    """Maximum alkalinity addition to restore preindustrial surface pH.

    At each timestep after start_CDR, solves for the AT required to return
    each masked surface cell to preindustrial pH given the current CT, then
    applies that as a flux. Cells where AT_desired < AT_current are skipped
    (no AT removal). No CT is added (NaOH assumption).
    """

    def make_q(self, t_current: float, chem: dict, dt: float) -> np.ndarray:
        """Add AT to restore preindustrial pH at masked cells; no CT change (NaOH)."""
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
        AT_to_add[AT_to_add < 0] = 0
        q[(self.m+1):] = AT_to_add / dt
        return q


def build_experiments(data_path: str, output_path: str, test: bool = False) -> list:
    """Return a list of Exp23 instances covering all parameter combinations.

    Loads the minimal OCIM grid data needed to construct the mixed-layer
    addition mask, then builds one Exp23 per (time resolution, scenario) pair.
    """
    grid    = loaders.load_ocim(data_path)
    ocnmask = grid['ocnmask']

    # mixed-layer mask: 1 where cell bottom depth < local MLD (cells fully within mixed layer)
    cell_top_depth_3d = make_3d(grid['cell_top_depth'], ocnmask)
    cell_bottom_depth_3d = np.concatenate(
        [cell_top_depth_3d[:, :, 1:], np.full((*cell_top_depth_3d.shape[:2], 1), np.inf)],
        axis=2,
    )
    mldmask    = (cell_bottom_depth_3d < grid['mld'][:, :, None]).astype(int)
    q_AT_mask  = flatten(mldmask * ocnmask, ocnmask)

    start_year = 2020.0
    start_CDR  = 2020.0  # same as start_year: CDR begins immediately

    if test:
        time_configs = [('test', np.arange(0, 6, 1.0))]
        scenarios    = ['ssp126']
        start_year   = 2002
        start_CDR    = 2002
    else:
        time_configs = [
            ('t0', np.arange(0, 20, 1.0)),    # annual steps
            ('t1', np.arange(0, 20, 1/12)),   # monthly steps
        ]
        scenarios = ['none', 'ssp126', 'ssp245', 'ssp534_OS']

    tag_date = datetime.now().strftime('%Y-%m-%d')
    experiments = []
    for t_name, times in time_configs:
        for scenario in scenarios:
            tag = f'{tag_date}_{t_name}_{scenario}'
            cfg = ExperimentConfig(
                data_path          = data_path,
                output_path        = output_path + f'exp23_{tag}.nc',
                scenario           = scenario,
                start_year         = start_year,
                times              = times,
                max_steps_per_file = 2000,
                start_CDR          = start_CDR,
                q_AT_mask          = q_AT_mask,
                attrs              = {'experiment': 'exp23', 'scenario': scenario, 'tag': tag},
            )
            experiments.append(Exp23(cfg))
    return experiments


def main():
    run_cli(build_experiments, 'Exp23: max AT addition to restore preindustrial pH')


if __name__ == '__main__':
    main()
