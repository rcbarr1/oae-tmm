"""
exp_anth_test: numerical equivalence test for the anth_in_q formulation.

Runs three experiments to compare the old formulation (Canth only updates
the chemistry background) against the new formulation (dCanth/dt and
d(xCO2_anth)/dt enter via q each step):

    B  anth_in_q=False  MaxAT CDR  old CDR run   (CDR signal = c_B directly,
                                                   since old baseline is all zeros)
    C  anth_in_q=True   no CDR     new baseline   (c accumulates Canth increment)
    D  anth_in_q=True   MaxAT CDR  new CDR run

CDR signal comparison:
    old: c_B[1:(m+1)]
    new: c_D[1:(m+1)] - c_C[1:(m+1)]

If the two formulations are numerically equivalent these signals should match.

CLI usage:
    python -m experiments.exp_anth_test --list
    python -m experiments.exp_anth_test --exp-id 0-2
"""

import gc
from datetime import datetime

import jax
import numpy as np
import PyCO2SYS as pyco2

from experiments.base import BaseExperiment, ExperimentConfig, run_cli
from oae_tmm import loaders
from oae_tmm.grid import flatten


class NoCDR(BaseExperiment):
    """Baseline experiment with no CDR — q is always zero."""

    def make_q(self, time_current: float, chem: dict, dt: float) -> np.ndarray:
        return np.zeros(1 + 2 * self.m)


class AnthTestMaxAT(BaseExperiment):
    """MaxAT CDR for the anth_in_q comparison: restores preindustrial surface pH."""

    def make_q(self, time_current: float, chem: dict, dt: float) -> np.ndarray:
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
    """Return three experiments (B, C, D) for the anth_in_q equivalence test.

    All use scenario='ssp245', monthly timesteps, 10-year run (2030-2040).
    The old baseline (run A) is omitted — it is all zeros by construction.

    CDR signal comparison:
        old: c_B[1:(m+1)]
        new: c_D[1:(m+1)] - c_C[1:(m+1)]
    """
    grid      = loaders.load_ocim(data_path)
    ocnmask   = grid['ocnmask']
    q_AT_mask = flatten(grid['mldmask'], ocnmask)

    scenario  = 'ssp245'
    start_CDR = 2030.0
    time      = np.arange(2030, 2040 + 1/12, 1/12)

    tag_date = datetime.now().strftime('%Y-%m-%d')

    experiments = []
    for anth_in_q, label, cls in [
        (False, 'B_old_maxAT',    AnthTestMaxAT),
        (True,  'C_new_baseline', NoCDR),
        (True,  'D_new_maxAT',    AnthTestMaxAT),
    ]:
        tag = f'{tag_date}_{label}_{scenario}'
        cfg = ExperimentConfig(
            data_path          = data_path,
            output_path        = output_path + f'anth_test_{tag}.nc',
            scenario           = scenario,
            time               = time,
            max_steps_per_file = 2000,
            start_CDR          = start_CDR,
            q_AT_mask          = q_AT_mask,
            anth_in_q          = anth_in_q,
            attrs              = {
                'experiment': 'anth_test',
                'run':        label,
                'scenario':   scenario,
                'anth_in_q':  str(anth_in_q),
                'tag':        tag,
            },
        )
        experiments.append(cls(cfg))

    return experiments


def main():
    run_cli(build_experiments, 'anth_in_q equivalence test (runs B, C, D)')


if __name__ == '__main__':
    main()
