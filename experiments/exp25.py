"""
Exp25: CT-removal efficiency map replicating Yamamoto et al., 2024.

Replicates the relative efficiency map from:
https://iopscience.iop.org/article/10.1088/1748-9326/ad7477/meta

For each surface ocean grid cell: removes CT at 1 μmol kg-1 yr-1 for the
first 30 days, then lets the system equilibrate for 100 years. Mixed
time-stepping (daily for 90 days, monthly to year 5, annual to year 100).

CDR assumption: CT removal only (no AT change).

CLI usage:
    python -m experiments.exp25 --exp-id 0
    python -m experiments.exp25 --list
    python -m experiments.exp25 --test
"""

import numpy as np
from datetime import datetime

from experiments.base import BaseExperiment, ExperimentConfig, run_cli
from oae_tmm import loaders
from oae_tmm.grid import flatten


class Exp25(BaseExperiment):
    """CT-removal impulse response at a single surface cell.

    Removes CT at 1 μmol kg-1 yr-1 from one masked surface cell for the
    first 30 days of the simulation, then zero thereafter. No AT change.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._q_CT_mask = None

    def _get_q_CT_mask(self) -> np.ndarray:
        if self._q_CT_mask is None:
            ocnmask = self.grid['ocnmask']
            mask_3d = np.zeros(ocnmask.shape)
            mask_3d[self.cfg.attrs['cell_lat_idx'], self.cfg.attrs['cell_lon_idx'], 0] = 1
            self._q_CT_mask = flatten(mask_3d, ocnmask)
        return self._q_CT_mask

    def make_q(self, t_current: float, _chem: dict, dt: float) -> np.ndarray:
        q = np.zeros(1 + 2 * self.m)
        t_offset = t_current - self.cfg.start_year
        if t_offset < 30.5 / 360:
            q[1:(self.m + 1)] = self._get_q_CT_mask() * -1  # [µmol CT kg-1 yr-1]
        return q


def build_experiments(data_path: str, output_path: str, test: bool = False) -> list:
    """Return a list of Exp25 instances, one per surface ocean grid cell."""
    grid    = loaders.load_ocim(data_path)
    ocnmask = grid['ocnmask']

    surf_mask = ocnmask.copy()
    surf_mask[:, :, 1:] = 0
    ocn_idxs = np.argwhere(surf_mask == 1)

    # mixed time-step schedule matching Yamamoto et al., 2024
    t0 = np.arange(0, 0.25,  1/360)  # daily for first 90 days
    t1 = np.arange(0.25, 5,  1/12)   # monthly until year 5
    t2 = np.arange(5,   101, 1)       # annual until year 100
    mixed_times = np.concatenate((t0, t1, t2))

    start_year = 2002
    start_CDR  = 2002

    if test:
        cells   = [ocn_idxs[0]]
        indices = [0]
        t0 = np.arange(0, 0.25, 1/360)  # daily for 90 days (covers 30-day CDR window)
        t1 = np.arange(0.25, 1, 1/12)   # monthly for the rest of year 1
        times = np.concatenate((t0, t1))
    else:
        cells   = ocn_idxs
        indices = range(len(ocn_idxs))
        times   = mixed_times

    tag_date = datetime.now().strftime('%Y-%m-%d')
    experiments = []

    for cell_num, ocn_idx in zip(indices, cells):
        tag = f'{tag_date}_t-mixed_{cell_num:05d}'
        cfg = ExperimentConfig(
            data_path          = data_path,
            output_path        = output_path + f'exp25_{tag}.nc',
            scenario           = 'none',
            start_year         = start_year,
            times              = times,
            max_steps_per_file = 2000,
            start_CDR          = start_CDR,
            attrs              = {
                'experiment':   'exp25',
                'tag':          tag,
                'cell_lat_idx': int(ocn_idx[0]),
                'cell_lon_idx': int(ocn_idx[1]),
            },
        )
        experiments.append(Exp25(cfg))

    return experiments


def main():
    run_cli(build_experiments, 'Exp25: CT-removal efficiency map (Yamamoto et al., 2024)')


if __name__ == '__main__':
    main()
