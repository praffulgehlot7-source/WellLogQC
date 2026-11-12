"""
porosity.py
-----------
Porosity calculation routines from petrophysical well logs.
"""

import numpy as np
import pandas as pd

from welllogqc.config.defaults import RHOMAT, RHOFLUID


def calc_density_porosity(
    df: pd.DataFrame,
    rho_matrix: float = RHOMAT,
    rho_fluid: float = RHOFLUID
) -> pd.Series:
    """Calculate density porosity (phi_D) from bulk density (RHOB).

    The standard density porosity formula is:
        phi_D = (rho_matrix - RHOB) / (rho_matrix - rho_fluid)

    Values are intentionally NOT clipped to allow unconstrained auditability
    and identification of borehole washouts or heavy minerals.

    Args:
        df: pandas DataFrame containing well log data, indexed by depth ('DEPT').
            Must contain the 'RHOB' (Bulk Density) column.
        rho_matrix: Matrix grain density in g/cm3. Defaults to 2.65 (quartz sandstone).
        rho_fluid: Pore fluid density in g/cm3. Defaults to 1.00 (fresh water).

    Returns:
        pd.Series: Calculated density porosity named 'PHID', dtype float64, sharing
            the index of df. Contains np.nan where RHOB is np.nan.

    Raises:
        ValueError: If 'RHOB' is not present in df.columns.
        ValueError: If rho_matrix <= rho_fluid.
    """
    if "RHOB" not in df.columns:
        raise ValueError("Required curve 'RHOB' not found in DataFrame columns.")

    if rho_matrix <= rho_fluid:
        raise ValueError(
            f"rho_matrix ({rho_matrix}) must be strictly greater than rho_fluid ({rho_fluid})."
        )

    rhob = df["RHOB"].astype(np.float64)
    phid = (float(rho_matrix) - rhob) / float(rho_matrix - rho_fluid)
    phid = phid.astype(np.float64)
    phid.name = "PHID"
    return phid
