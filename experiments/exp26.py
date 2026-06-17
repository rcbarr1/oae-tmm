"""
Exp26: Monte Carlo sensitivity test of air-sea gas exchange efficiency.

Runs 1000 independent simulations with the piston velocity k scaled by a
random factor drawn from N(1, 0.2). Each run adds a pulse of 1 µmol AT kg-1
to every surface ocean cell at the first time step, then lets the system
equilibrate over 20 years at monthly resolution.

CDR assumption: NaOH (no CT change). 'none' emissions scenario.

CLI usage:
    python -m experiments.exp26 --exp-id 0
    python -m experiments.exp26 --list
    python -m experiments.exp26 --exp-id 0-99
"""

# NOTE ABOUT HOW TO DO THIS IN THE FUTURE
# see 2026-05-21 notes from meeting with Brendan (notability)
# need to do uncertainty in glodap variables plus uncertainty in air-sea gas
# exchange parameterization
#
# uncertainty in glodap variables: U = sqrt(u_meas^2 + u_mapping^2)
# can pull out mapping uncertainty by unsumming in quadrature using standard
# measurement uncertainties (i.e. 2 umol kg-1 for AT)
# then, adjust u_meas via normal MC method (normal distribution) and recalculate U
#
# for air-sea gas exchange: 20% global value probably won't cut it, instead need
# to vary U and Sc



import numpy as np
from datetime import datetime

from experiments.base import BaseExperiment, ExperimentConfig, run_cli
from oae_tmm import loaders
from oae_tmm.grid import flatten


class Exp26(BaseExperiment):
    """Global AT pulse with k scaled by a Monte Carlo factor.

    Adds 1 µmol AT kg-1 to every surface ocean cell at the first time step
    (pulse, NaOH). The piston velocity k is scaled by k_scale_factor drawn
    from N(1, 0.2) to quantify sensitivity to air-sea gas exchange
    parameterization.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        self._at_pulse_done = False

    def setup(self):
        super().setup()
        attrs = self.cfg.attrs
        assert attrs is not None
        self.k = self.k * attrs['k_scale_factor']

    def make_q(self, time_current: float, chem: dict, dt: float) -> np.ndarray:
        q = np.zeros(1 + 2 * self.m)
        if not self._at_pulse_done:
            q_AT_mask = self.cfg.q_AT_mask
            assert q_AT_mask is not None
            q[(self.m + 1):] = q_AT_mask / dt  # [µmol AT kg-1 yr-1]
            self._at_pulse_done = True
        return q


def build_experiments(data_path: str, output_path: str, test: bool = False) -> list:
    """Return a list of 1000 Exp26 instances, one per Monte Carlo k_scale_factor.

    k_scale_factors are drawn fresh from N(1, 0.2) each call. The value is
    stored in cfg.attrs['k_scale_factor'] and written to the output file so
    individual runs can be identified after the fact.
    """
    grid    = loaders.load_ocim(data_path)
    ocnmask = grid['ocnmask']

    surf_mask = ocnmask.copy()
    surf_mask[:, :, 1:] = 0
    q_AT_mask = flatten(surf_mask, ocnmask)

    time = np.arange(2020, 2040, 1/12)

    num_mc_sims     = 1000
    k_scale_factors = np.random.normal(loc=1.0, scale=0.2, size=num_mc_sims)

    if test:
        k_scale_factors = [1.0]
        time = np.arange(2020, 2025, 1/12)

    tag_date    = datetime.now().strftime('%Y-%m-%d')
    experiments = []

    for k_idx, k_sf in enumerate(k_scale_factors):
        tag = f'{tag_date}_t1_{k_idx:05d}'
        cfg = ExperimentConfig(
            data_path          = data_path,
            output_path        = output_path + f'exp26_{tag}.nc',
            scenario           = 'none',
            time               = time,
            max_steps_per_file = 2000,
            q_AT_mask          = q_AT_mask,
            attrs              = {
                'experiment':    'exp26',
                'tag':           tag,
                'k_scale_factor': float(k_sf),
            },
        )
        experiments.append(Exp26(cfg))

    return experiments


def main():
    run_cli(build_experiments, 'Exp26: Monte Carlo sensitivity test of air-sea gas exchange efficiency')


if __name__ == '__main__':
    main()
