"""
Exp24: Impulse-response air-sea gas exchange efficiency map (Zhou et al., 2025).

Replicates the approach of https://www.nature.com/articles/s41558-024-02179-9: adds
10 mol AT m-2 yr-1 to one surface ocean grid cell for one month (impulse), then
lets the system equilibrate. Repeated independently for every surface cell to
produce a global map.

CDR assumption: NaOH (no CT change). Mixed time-stepping (daily for 90 days,
monthly to year 5, annual to year 15) captures fast air-sea re-equilibration while
keeping long-tail transport tractable.

CLI usage:
    python -m experiments.exp24 --exp-id 0
    python -m experiments.exp24 --list
    python -m experiments.exp24 --test
"""

import numpy as np
from datetime import datetime

from experiments.base import BaseExperiment, ExperimentConfig, run_cli, rho
from oae_tmm import loaders
from oae_tmm.grid import flatten


class Exp24(BaseExperiment):
    """Impulse-response CDR at a single surface cell (NaOH, one month of AT addition).

    Adds 10 mol AT m-2 yr-1 to the one masked surface cell for the first month
    of the simulation, then zero thereafter. The time-step detection mirrors the
    original logic: at annual steps the monthly-equivalent flux is applied (÷12);
    at monthly steps it is applied only for the first month; at daily steps only
    for the first ~30 days.
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

        if abs(dt - 1) < 0.01 and time_offset <= 1:               # annual step, first year
            q[(self.m + 1):] = q_AT_mask * rate / 12
        elif 0.5/12 < dt < 1.5/12 and time_offset <= 31/360:      # monthly step, first month
            q[(self.m + 1):] = q_AT_mask * rate
        elif 0.5/360 < dt < 1.5/360 and time_offset <= 30.5/360:  # daily step, first ~30 days
            q[(self.m + 1):] = q_AT_mask * rate
        return q


def build_experiments(data_path: str, output_path: str, test: bool = False) -> list:
    """Return a list of Exp24 instances, one per surface ocean grid cell.

    Loads the OCIM grid and identifies all surface ocean cells, then builds
    one Exp24 per cell (or just one cell in test mode).
    """
    grid    = loaders.load_ocim(data_path)
    ocnmask = grid['ocnmask']

    # surface-only mask (first depth layer)
    surf_mask = ocnmask.copy()
    surf_mask[:, :, 1:] = 0
    ocn_idxs = np.argwhere(surf_mask == 1)  # shape (n_surface_cells, 3)

    # mixed time-step schedule
    t0 = np.arange(2002, 2002.25, 1/360)   # daily, 2002–2002.25 (first 90 days)
    t1 = np.arange(2002.25, 2007,  1/12)   # monthly, 2002.25–2007
    t2 = np.arange(2007,   2018,   1)      # annual, 2007–2018
    mixed_time = np.concatenate((t0, t1, t2))

    if test:
        cells   = [ocn_idxs[0]]
        indices = [0]
        time = np.arange(2002, 2008, 1.0)
    else:
        cells   = ocn_idxs
        indices = range(len(ocn_idxs))
        time = mixed_time

    tag_date = datetime.now().strftime('%Y-%m-%d')
    experiments = []

    for cell_num, ocn_idx in zip(indices, cells):
        tag = f'{tag_date}_t-mixed_{cell_num:05d}'
        cfg = ExperimentConfig(
            data_path          = data_path,
            output_path        = output_path + f'exp24_{tag}.nc',
            scenario           = 'none',
            time               = time,
            max_steps_per_file = 2000,
            start_CDR          = 2002,
            attrs              = {
                'experiment':   'exp24',
                'tag':          tag,
                'cell_lat_idx': int(ocn_idx[0]),
                'cell_lon_idx': int(ocn_idx[1]),
            },
        )
        experiments.append(Exp24(cfg))

    return experiments


def main():
    run_cli(build_experiments, 'Exp24: impulse-response air-sea gas exchange efficiency map')


if __name__ == '__main__':
    main()
