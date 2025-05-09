import numpy as np
#from numba import njit

from src.parameters import p_Grid, p_BGC

def compute_sinking(dt, tra, w, Grid):
    """
    Computes the sinking of a tracer in the water column.

    Parameters:
        tra (np.ndarray): Tracer concentration profile (1D array).
        w (float): Sinking velocity (m/s).
        
    Returns:
        tuple:
            ddt_tra (np.ndarray): Rate of change of tracer concentration due to sinking.
            ddt_tra2sed (float): Flux of tracer to the sediment.
    """
    # Ensure tracer concentrations are non-negative
    tra_loc = np.fmax(tra[:,1], 0.0)

    # Compute sinking flux at the bottom of each box
    adv_tra = np.zeros(len(Grid.zgrid))
    for kk in range(1, len(Grid.zgrid)):
        adv_tra[kk] = w * tra_loc[kk - 1] # mmol/m2/s

    # Compute the change in tracer concentration within each box
    ddt_tra = np.zeros(len(Grid.zgrid))
    for kk in range(len(Grid.zgrid) - 1):
        ddt_tra[kk] = (adv_tra[kk] - adv_tra[kk + 1]) / Grid.dz  # mmol/m3/s

    # Compute the flux of tracer to the sediment
    adv_tra_bot = w * tra_loc[-1]  # Flux at the bottom boundary (m/s * µM)
    ddt_tra[-1] = (adv_tra[-1] - adv_tra_bot) / Grid.dz
    ddt_tra2sed = adv_tra_bot / Grid.dz  # Sediment flux in µM
    
    # apply to the tracer itself
    tra[:,1] = tra[:,1] + ddt_tra * dt
    
    return tra, ddt_tra2sed


def compute_sink_rate(phy, BGC):
    """
    Computes the sinking rate of detritus based on phytoplankton concentration.

    Parameters:
        phy (float): Phytoplankton concentration (µM).
        
    Returns:
        float: Sinking velocity (m/s).
    """
    return BGC.w0 * np.fmax(1e-3, phy - BGC.phy_biothresh) ** 0.21
