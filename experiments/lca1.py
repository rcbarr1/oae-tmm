"""
LCA1: AT pulse addition at specific coastal and offshore locations.

Models alkalinity addition of 1 metric ton of OH- at each of four locations:
  - nearshore Colombia  (3.96°N, 281.0°E)  — latitude[47], longitude[140]
  - offshore Colombia   (5.93°N, 269.0°E)  — latitude[48], longitude[134]
  - nearshore Norway   (61.32°N,   3.0°E)  — latitude[76], longitude[1]
  - offshore Norway    (63.30°N, 349.0°E)  — latitude[77], longitude[174]

AT is added as a single pulse during the first month of the simulation, then
the system runs forward for 16 years to track the fate of the alkalinity signal.
Uses the REMIND atmospheric CO2 scenario starting in 2050.

CLI usage:
    python -m experiments.lca1 --exp-id 0
    python -m experiments.lca1 --list
    python -m experiments.lca1 --test
"""

from datetime import datetime

import numpy as np

from experiments.base import BaseExperiment, ExperimentConfig, run_cli
from oae_tmm import loaders, trace
from oae_tmm.grid import flatten


# (lat_idx, lon_idx) on the OCIM2-48L grid, surface layer only
_LOCATIONS = [
    {'name': 'nearshore_colombia', 'lat_idx': 47, 'lon_idx': 140},
    {'name': 'offshore_colombia',  'lat_idx': 48, 'lon_idx': 134},
    {'name': 'nearshore_norway',   'lat_idx': 76, 'lon_idx':   1},
    {'name': 'offshore_norway',    'lat_idx': 77, 'lon_idx': 174},
]


class LCA1(BaseExperiment):
    """AT pulse addition at a single coastal or offshore location.

    Adds a single pulse of AT (mass-based, from NaOH) at the grid cell
    specified by q_AT_mask. The full prescribed mass is added as a flux
    during the first month of the simulation only. After that, make_q returns
    zeros and the system evolves freely under transport and air-sea gas exchange.
    Uses the REMIND atmospheric CO2 scenario via pyTRACE direct calls.

    Attributes passed through cfg.attrs:
        AT_amount_tons : float  mass of NaOH added [metric tons]
    """

    def setup(self):
        """Load temperature_3d and salinity_3d for pyTRACE, then delegate to BaseExperiment.setup()."""
        base = self.cfg.data_path + 'GLODAPv2.2016b.MappedProduct/'
        self.temperature_3d = np.load(base + 'temperature.npy')
        self.salinity_3d = np.load(base + 'salinity.npy')
        super().setup()

    def _calc_canth(self, year: float, scenario: str) -> np.ndarray:
        """Compute Canth by calling pyTRACE directly (required for REMIND scenario)."""
        return trace.calculate_canth(
            scenario, year, self.temperature_3d, self.salinity_3d,
            self.grid['ocnmask'],
            self.grid['latitude'], self.grid['longitude'],
            self.grid['depth'],
        )

    def make_q(self, t_current: float, chem: dict, dt: float) -> np.ndarray:
        """Add AT as a pulse during the first month of CDR deployment only.

        Converts AT_amount_tons metric tons of NaOH (MW = 17.007 g/mol) to a
        flux in µmol AT kg^-1 yr^-1 at the single target grid cell. The flux
        is applied only when t_elapsed < 1 month (0.0834 yr), matching the
        one-month pulse of the original experiment design.
        """
        q = np.zeros(1 + 2 * self.m)
        t_elapsed = t_current - self.cfg.start_CDR
        if t_elapsed < 0.0834:
            AT_amount_tons = self.cfg.attrs['AT_amount_tons']
            V       = self.grid['cell_volume']   # (m,) flattened volumes [m^3]
            rho     = self.grid['rho']
            sw_mass = np.sum(V * self.cfg.q_AT_mask) * rho  # [kg] seawater at target cell
            # metric tons → µmol → normalize by sw_mass → annualize (×12 for monthly flux)
            q[(self.m+1):] = self.cfg.q_AT_mask * AT_amount_tons * 1e6 / 17.007 * 1e6 * 12 / sw_mass
        return q


_AT_AMOUNTS = [
    {'tons': 0.001, 'label': '1kg'},
    {'tons': 1.0,   'label': '1ton'},
    {'tons': 5.0,   'label': '5ton'},
]

_START_YEARS = [2030.0, 2050.0]


def build_experiments(data_path: str, output_path: str, test: bool = False) -> list:
    """Return LCA1 instances for all combinations of location, AT amount, and start year.

    In full mode: 4 locations × 3 AT amounts × 2 start years = 24 experiments,
    16 years at monthly resolution, REMIND scenario.

    In test mode: 1 experiment (nearshore Colombia, 1 ton, 2050), 6 years at
    mixed resolution (monthly for year 0–1, annual thereafter). Monthly steps
    in year 1 ensure the first-month AT pulse in make_q is triggered.
    """
    grid    = loaders.load_ocim(data_path)
    ocnmask = grid['ocnmask']

    if test:
        times       = np.concatenate([np.arange(0, 1, 1/12), np.arange(1, 6, 1)])
        locs        = _LOCATIONS[:1]
        start_years = [2050.0]
        at_amounts  = [_AT_AMOUNTS[1]]  # 1 ton only
    else:
        times       = np.arange(0, 16, 1/12)
        locs        = _LOCATIONS
        start_years = _START_YEARS
        at_amounts  = _AT_AMOUNTS

    tag_date    = datetime.now().strftime('%Y-%m-%d')
    experiments = []
    for loc in locs:
        mask_3d   = np.zeros(ocnmask.shape)
        mask_3d[loc['lat_idx'], loc['lon_idx'], 0] = 1
        q_AT_mask = flatten(mask_3d, ocnmask)

        for start_year in start_years:
            for at in at_amounts:
                tag = f'{tag_date}_{loc["name"]}_{at["label"]}_{int(start_year)}'
                cfg = ExperimentConfig(
                    data_path          = data_path,
                    output_path        = output_path + f'LCA1_{tag}.nc',
                    scenario           = 'REMIND',
                    start_year         = start_year,
                    times              = times,
                    max_steps_per_file = 2000,
                    start_CDR          = start_year,
                    q_AT_mask          = q_AT_mask,
                    attrs              = {
                        'experiment':     'LCA1',
                        'location':       loc['name'],
                        'AT_amount_tons': at['tons'],
                        'scenario':       'REMIND',
                        'tag':            tag,
                    },
                )
                experiments.append(LCA1(cfg))
    return experiments


def main():
    run_cli(build_experiments, 'LCA1: AT pulse addition at coastal/offshore locations')


if __name__ == '__main__':
    main()
