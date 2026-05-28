"""
Physical parameterization functions for oae-tmm.

These functions implement Wanninkhof (2014) gas transfer physics (Schmidt
number, piston velocity) and retrieve atmospheric CO2 concentrations for SSP
emissions scenarios. They do not load observational data from disk (except
get_co2_scenario, which reads a small text file bundled with pyTRACE) and do
not call the PETSc solver.
"""

import numpy as np
import warnings


def schmidt_number(gas: str, temperature: np.ndarray) -> np.ndarray:
    """Calculate the Schmidt number for a gas in seawater (Wanninkhof 2014).

    Coefficients are valid for seawater temperatures from -2°C to 40°C.

    Reference: Wanninkhof, R. (2014). Relationship between wind speed and gas
    exchange over the ocean revisited. Limnology and Oceanography: Methods,
    12(6), 351-362.

    Parameters
    ----------
    gas : str
        Gas of interest. One of: 'O2', 'CO2', 'N2', 'Ar'.
    temperature : float or np.ndarray
        Seawater temperature [degrees C]. Accepts scalars or arrays of any shape.

    Returns
    -------
    float or np.ndarray
        Schmidt number [unitless], same shape as temperature.
    """
    # polynomial coefficients from Table 1 of Wanninkhof (2014)
    sc_coeffs = {
        'O2':  [1920.4, -135.6,   5.2122, -0.10939,  0.00093777],
        'CO2': [2116.8, -136.25,  4.7353, -0.092307, 0.0007555],
        'N2':  [2304.8, -162.75,  6.2557, -0.13129,  0.0011255],
        'Ar':  [2078.1, -146.74,  5.6403, -0.11838,  0.0010148],
    }

    if gas not in sc_coeffs:
        raise ValueError(f"Gas '{gas}' not supported. Choose from {list(sc_coeffs.keys())}")

    a, b, c, d, e = sc_coeffs[gas]
    Sc = a + (b * temperature) + (c * temperature**2) + (d * temperature**3) + (e * temperature**4)

    return Sc


def get_co2_scenario(scenario: str, times: np.ndarray,
                     co2_data_path: str = './pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt') -> np.ndarray:
    """Return atmospheric CO2 concentrations for a given SSP scenario and times.

    Interpolates from the pyTRACE CO2 trajectory file (originally from the
    University of Melbourne greenhouse gas dataset) to the requested times.
    For the 'none' scenario, CO2 is held constant at the value corresponding
    to the first time point — this represents a fixed preindustrial-ish
    baseline with no future emissions change.

    Note: historical data and SSP scenarios differ by less than 1 ppm even
    before the SSPs formally diverge in 2016.

    Available scenarios: 'none', 'ssp119', 'ssp126', 'ssp245', 'ssp370',
    'ssp370_lowNTCF', 'ssp434', 'ssp460', 'ssp534_OS'.

    Data source: https://greenhousegases.science.unimelb.edu.au/#!/ghg?mode=downloads
    (via pyTRACE CO2TrajectoriesAdjusted.txt)

    Note on co2_data_path: this points to pyTRACE/ at the repo root using a
    path relative to the working directory. Update if run from a different cwd.

    Parameters
    ----------
    scenario : str
        Name of the emissions scenario. One of: 'none', 'ssp119', 'ssp126',
        'ssp245', 'ssp370', 'ssp370_lowNTCF', 'ssp434', 'ssp460', 'ssp534_OS'.
    times : np.ndarray
        1D array of times [decimal years CE] at which to return CO2.
    co2_data_path : str, optional
        Path to CO2TrajectoriesAdjusted.txt. Defaults to the location inside
        the pyTRACE submodule.

    Returns
    -------
    np.ndarray
        1D array of atmospheric CO2 concentrations [µmol CO2 (mol air)^-1, i.e. ppm],
        same length as times.
    """
    scenarios = {
        'none': 1, 'ssp119': 2, 'ssp126': 3, 'ssp245': 4, 'ssp370': 5,
        'ssp370_lowNTCF': 6, 'ssp434': 7, 'ssp460': 8, 'ssp534_OS': 9,
    }

    if scenario not in scenarios:
        raise ValueError(f"Invalid scenario {scenario!r}. Must be one of: {', '.join(scenarios.keys())}")

    data = np.loadtxt(co2_data_path)
    CO2_data_years = data[:, 0]
    CO2_data = data[:, scenarios[scenario]]

    if scenario != 'none':
        atmospheric_CO2 = np.interp(times, CO2_data_years, CO2_data)
    else:
        if times[0] > 2022:
            warnings.warn("'none' scenario selected but times start after 2022. "
                          "Canth is held constant based on a linear extrapolation from 2012-2022.")
        # hold CO2 constant at the value for the first time point
        atmospheric_CO2 = np.interp(times[0], CO2_data_years, CO2_data) * np.ones_like(times)

    return atmospheric_CO2


def calc_piston_velocity(sst_2D: np.ndarray, wspd_2D: np.ndarray) -> np.ndarray:
    """Compute the CO2 piston velocity from sea surface temperature and wind speed.

    Implements the Wanninkhof (2014) parameterization:
        k = a * U^2 * (Sc / 660)^(-0.5)
    where a = 0.251 is the Wanninkhof (2014) scaling coefficient for annual-
    mean wind speeds, U is 10-m wind speed, and Sc is the Schmidt number for
    CO2 evaluated at SST. 

    Output is converted from cm h^-1 to m yr^-1 for use with the annual-
    timestep OCIM2-48L transport matrix.

    Reference: Wanninkhof, R. (2014). Relationship between wind speed and gas
    exchange over the ocean revisited. Limnology and Oceanography: Methods,
    12(6), 351-362.

    Parameters
    ----------
    sst_2D : np.ndarray
        Annual mean sea surface temperature [degrees C], shape (n_lat, n_lon).
    wspd_2D : np.ndarray
        Annual mean wind speed at 10 m [m s^-1], shape (n_lat, n_lon).

    Returns
    -------
    np.ndarray
        Piston velocity [m yr^-1], shape (n_lat, n_lon).
    """
    Sc_2D = schmidt_number('CO2', sst_2D)
    k_2D = 0.251 * wspd_2D**2 * (Sc_2D / 660)**-0.5  # [cm h^-1]
    k_2D *= (24 * 365.25 / 100)  # convert cm h^-1 to m yr^-1
    return k_2D

