"""
Physical parameterization functions for oae-tmm.

These functions implement Wanninkhof (2014) gas transfer physics (Schmidt
number, piston velocity). They do not load observational data from disk and do
not call the PETSc solver.
"""

from typing import overload

import numpy as np


@overload
def schmidt_number(gas: str, temperature: float) -> float: ...
@overload
def schmidt_number(gas: str, temperature: np.ndarray) -> np.ndarray: ...
def schmidt_number(gas: str, temperature: float | np.ndarray) -> float | np.ndarray:
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


def calc_piston_velocity(sst_2d: np.ndarray, wspd_2d: np.ndarray) -> np.ndarray:
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
    sst_2d : np.ndarray
        Annual mean sea surface temperature [degrees C], shape (n_lat, n_lon).
    wspd_2d : np.ndarray
        Annual mean wind speed at 10 m [m s^-1], shape (n_lat, n_lon).

    Returns
    -------
    np.ndarray
        Piston velocity [m yr^-1], shape (n_lat, n_lon).
    """
    Sc_2d = schmidt_number('CO2', sst_2d)
    k_2d = 0.251 * wspd_2d**2 * (Sc_2d / 660)**-0.5  # [cm h^-1]
    k_2d *= (24 * 365.25 / 100)  # convert cm h^-1 to m yr^-1
    return k_2d

