"""
Interpolate REMIND CO2 scenario data to a smooth yearly curve.

Fits a PCHIP interpolant to the sparse REMIND CO2 values, applies the
Carter et al. (2025) air-sea lag adjustment (eqn. 5), and writes the
result into the pyTRACE CO2 trajectory files:

    pyTRACE/pyTRACE/data/CO2Trajectories.txt         (unadjusted)
    pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt (adjusted)

Run once when updating the REMIND scenario; these files are read by
pyTRACE at experiment time.

Reference: Carter et al. (2025), ESSD, https://doi.org/10.5194/essd-17-3073-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

co2_trajs = np.genfromtxt(
    './pyTRACE/pyTRACE/data/CO2Trajectories_OLD.txt',
    delimiter='\t', dtype=None, skip_header=0,
    missing_values=None, filling_values=np.nan,
)
co2_trajs = np.vstack(co2_trajs.tolist())

co2_trajs_no_nans = co2_trajs[~np.isnan(co2_trajs).any(axis=1)]
years = co2_trajs_no_nans[:, 0]
remind_co2 = co2_trajs_no_nans[:, 10]

interp = PchipInterpolator(years, remind_co2, extrapolate=True)
years_new = np.arange(0, 2500)
remind_co2_interp = interp(years_new)

# hold CO2 constant after 2100
remind_co2_interp[2100:] = remind_co2_interp[2100]

# save unadjusted values
co2_trajs_interp = co2_trajs.copy()
co2_trajs_interp[:, 10] = remind_co2_interp
np.savetxt('./pyTRACE/pyTRACE/data/CO2Trajectories.txt', co2_trajs_interp, delimiter='\t')

# apply Carter et al. (2025) eqn. 5 air-sea lag adjustment:
#   pCO2_ocean(year) = xCO2_atm(year) - 0.144 * (xCO2_atm(year) - xCO2_atm(year-65))
# adjustment starts at index 65 to match existing trajectory files
remind_co2_interp_adj = np.zeros_like(remind_co2_interp)
remind_co2_interp_adj[:65] = remind_co2_interp[:65]
remind_co2_interp_adj[65:] = remind_co2_interp[65:] - 0.144 * (remind_co2_interp[65:] - remind_co2_interp[:-65])

co2_trajs_adj = np.genfromtxt(
    './pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted_OLD.txt',
    delimiter='\t', dtype=None, skip_header=0,
    missing_values=None, filling_values=np.nan,
)
co2_trajs_adj = np.vstack(co2_trajs_adj.tolist())
co2_trajs_adj = np.hstack((co2_trajs_adj, np.expand_dims(remind_co2_interp_adj, axis=1)))

np.savetxt('./pyTRACE/pyTRACE/data/CO2TrajectoriesAdjusted.txt', co2_trajs_adj, delimiter='\t')