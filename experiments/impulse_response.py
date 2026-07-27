"""
impulse_response: Impulse-response air-sea gas exchange efficiency map.

Replicates the approach of Zhou et al. (2025): adds 10 mol AT m-2 yr-1
to one surface ocean grid cell for one month (impulse), then lets the
system equilibrate. Repeated independently for every surface cell to
produce a global map.

Reference: Zhou, M. et al. (2025). Mapping the global variation in the
efficiency of ocean alkalinity enhancement for carbon dioxide removal.
Nature Climate Change, 15, 59-65. https://doi.org/10.1038/s41558-024-02179-9.

CDR assumption: NaOH (no CT change). Production runs use annual time-stepping
over 15 years to capture the long-tail equilibration of the air-sea CO2 flux.

The AT flux in make_q uses overlap-fraction scaling so that the total AT
added over the 30-day impulse window is rate*(30/360) regardless of dt.
This allows any timestep size to represent the impulse correctly.

CLI usage:
    python -m experiments.impulse_response --exp-id 0
    python -m experiments.impulse_response --list
    python -m experiments.impulse_response --test --list
    python -m experiments.impulse_response --test --exp-id 0-19
"""

import numpy as np
from datetime import datetime

from experiments.base import BaseExperiment, ExperimentConfig, run_cli, rho
from oae_tmm import loaders
from oae_tmm.grid import flatten


class ImpulseResponse(BaseExperiment):
    """Impulse-response CDR at a single surface cell (NaOH, one month of AT addition).

    Adds 10 mol AT m-2 yr-1 to the one masked surface cell for the first ~30 days
    of the simulation, then zero thereafter.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._q_AT_mask = None

    def _get_q_AT_mask(self) -> np.ndarray:
        if self._q_AT_mask is None:
            ocnmask = self.grid['ocnmask']
            mask_3d = np.zeros(ocnmask.shape)
            attrs = self.cfg.attrs
            assert attrs is not None
            mask_3d[attrs['cell_lat_idx'], attrs['cell_lon_idx'], 0] = 1
            self._q_AT_mask = flatten(mask_3d, ocnmask)
        return self._q_AT_mask

    def make_q(self, time_current: float, chem: dict, dt: float) -> np.ndarray:
        q = np.zeros(1 + 2 * self.m)
        time_offset = time_current - self.cfg.time[0]
        rate = 10 * 1e6 / self.grid['z1'] / rho  # [µmol AT kg-1 yr-1]
        q_AT_mask = self._get_q_AT_mask()

        # overlap fraction of this timestep that falls within the 30-day impulse window;
        # scales rate so total AT added is rate*impulse_end regardless of dt
        impulse_end = 30 / 360
        overlap = max(0.0, min(time_offset, impulse_end) - max(time_offset - dt, 0.0))
        if overlap > 0:
            q[(self.m + 1):] = q_AT_mask * rate * (overlap / dt)
        return q


def build_experiments(data_path: str, output_path: str, test: bool = False) -> list:
    """Return a list of ImpulseResponse instances.

    In test mode: 5 timestep resolutions × 4 representative cells, 5-year runs
    with scenario='none'. Used to compare how timestep size affects the impulse
    response (use --test --list to see all experiments; --test --exp-id 0-N to run).

    In production mode: one per surface ocean grid cell, mixed time-stepping
    (daily for 90 days then monthly to year 15), for each SSP scenario.
    """
    grid    = loaders.load_ocim(data_path)
    ocnmask = grid['ocnmask']

    surf_mask = ocnmask.copy()
    surf_mask[:, :, 1:] = 0
    ocn_idxs = np.argwhere(surf_mask == 1)  # shape (n_surface_cells, 3)

    tag_date    = datetime.now().strftime('%Y-%m-%d')
    experiments = []

    if test:
        test_indices = [758, 5291, 7965, 8810]
        t_configs = [
            ('annually', np.arange(2022, 2028,     1.0  )),
            ('monthly',  np.arange(2022, 2027.084, 1/12 )),
            ('dekadal',  np.arange(2022, 2027.028, 1/36 )),
            ('pentadal', np.arange(2022, 2027.014, 1/72 )),
            ('daily',    np.arange(2022, 2027.003, 1/360)),
        ]
        for t_name, time in t_configs:
            for cell_num in test_indices:
                ocn_idx = ocn_idxs[cell_num]
                tag     = f'{tag_date}_{t_name}_none_{cell_num:05d}'
                cfg     = ExperimentConfig(
                    data_path          = data_path,
                    output_path        = output_path + f'impulse_response_{tag}.nc',
                    scenario           = 'none',
                    time               = time,
                    max_steps_per_file = 2000,
                    start_CDR          = 2022,
                    attrs              = {
                        'experiment':   'impulse_response',
                        'tag':          tag,
                        'cell_lat_idx': int(ocn_idx[0]),
                        'cell_lon_idx': int(ocn_idx[1]),
                        't_name':       t_name,
                    },
                )
                experiments.append(ImpulseResponse(cfg))
    else:
        time   = np.arange(2022, 2073, 1)

        for scenario in ['none', 'ssp245', 'ssp534_OS']:
            for cell_num, ocn_idx in zip(range(len(ocn_idxs)), ocn_idxs):
                tag = f'{tag_date}_{scenario}_{cell_num:05d}'
                cfg = ExperimentConfig(
                    data_path          = data_path,
                    output_path        = output_path + f'impulse_response_{tag}.nc',
                    scenario           = scenario,
                    time               = time,
                    max_steps_per_file = 2000,
                    start_CDR          = 2022,
                    attrs              = {
                        'experiment':   'impulse_response',
                        'tag':          tag,
                        'cell_lat_idx': int(ocn_idx[0]),
                        'cell_lon_idx': int(ocn_idx[1]),
                    },
                )
                experiments.append(ImpulseResponse(cfg))

    return experiments


def main():
    run_cli(build_experiments, 'ImpulseResponse: impulse-response air-sea gas exchange efficiency map')


if __name__ == '__main__':
    main()
