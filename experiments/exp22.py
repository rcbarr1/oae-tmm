"""
Exp22: Maximum alkalinity addition targeting preindustrial surface pH.

Same CDR logic as Exp23 but calls pyTRACE directly at every ocean grid cell
to compute anthropogenic carbon, rather than interpolating the pre-computed
TRACE gridded product.

CLI usage:
    python -m experiments.exp22 --exp-id 0
    python -m experiments.exp22 --list
    python -m experiments.exp22 --test
"""
import gc
from datetime import datetime

import jax
import numpy as np
import PyCO2SYS as pyco2
import xarray as xr

from experiments.base import BaseExperiment, ExperimentConfig, run_cli
from oae_tmm import loaders, trace
from oae_tmm.grid import flatten


class Exp22(BaseExperiment):
    """Maximum alkalinity addition with direct pyTRACE calls for Canth.

    Identical CDR logic to Exp23 (restore preindustrial surface pH via NaOH
    addition within the mixed layer), but computes anthropogenic carbon by
    calling the pyTRACE neural network directly at each timestep rather than
    reading the pre-computed TRACE gridded product.
    """

    def setup(self):
        """Load temperature_3d and salinity_3d for pyTRACE, then delegate to BaseExperiment.setup()."""
        base = self.cfg.data_path + 'GLODAPv2.2016b.MappedProduct/'
        self.temperature_3d = xr.open_dataset(base + 'temperature.nc')['temperature'].values
        self.salinity_3d = xr.open_dataset(base + 'salinity.nc')['salinity'].values
        super().setup()

    def _calc_canth(self, year: float, scenario: str) -> np.ndarray:
        """Compute Canth by calling pyTRACE directly at every ocean grid cell."""
        return trace.calculate_canth(
            scenario, year, self.temperature_3d, self.salinity_3d,
            self.grid['ocnmask'],
            self.grid['latitude'], self.grid['longitude'],
            self.grid['depth'],
        )

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
    """Return a list of Exp22 instances covering all parameter combinations."""
    grid    = loaders.load_ocim(data_path)
    ocnmask = grid['ocnmask']

    q_AT_mask = flatten(grid['mldmask'], ocnmask)

    start_year = 2020.0
    start_CDR  = 2020.0

    if test:
        time_configs = [('test', np.arange(0, 6, 1.0))]
        scenarios    = ['ssp126']
        start_year   = 2002
        start_CDR    = 2002
    else:
        time_configs = [
            ('t0', np.arange(0, 20, 1.0)),
            ('t1', np.arange(0, 20, 1/12)),
        ]
        scenarios = ['none', 'ssp126', 'ssp245', 'ssp534_OS']

    tag_date = datetime.now().strftime('%Y-%m-%d')
    experiments = []
    for t_name, times in time_configs:
        for scenario in scenarios:
            tag = f'{tag_date}_{t_name}_{scenario}'
            cfg = ExperimentConfig(
                data_path          = data_path,
                output_path        = output_path + f'exp22_{tag}.nc',
                scenario           = scenario,
                start_year         = start_year,
                times              = times,
                max_steps_per_file = 2000,
                start_CDR          = start_CDR,
                q_AT_mask          = q_AT_mask,
                attrs              = {'experiment': 'exp22', 'scenario': scenario, 'tag': tag},
            )
            experiments.append(Exp22(cfg))
    return experiments


def main():
    run_cli(build_experiments, 'Exp22: max AT addition to restore preindustrial pH (direct pyTRACE)')


if __name__ == '__main__':
    main()
