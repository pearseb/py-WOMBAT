import numpy as np
#from numba import njit

from src.parameters import p_Grid, p_BGC

def compute_light_profile(pchl_loc, p_Chl_k, par, mld, Grid, BGC):
    """
    Compute the light available in the water column based on chlorophyll concentration.

    Parameters:
        pchl_loc (np.ndarray): Chlorophyll concentration at each depth (mg Chl/m³).
        p_Chl_k (np.ndarray): Attenuation coefficients for blue, green, and red light based on Chl.
        par (float): Surface PAR (Photosynthetically Active Radiation) in µmol photons/m²/s.
        mld (float): Mixed layer depth (meters).
        
    Returns:
        dict: A dictionary containing the following:
            - `par_tot`: Total PAR profile (µmol photons/m²/s).
            - `par_tot_mld`: Mean total PAR in the mixed layer (µmol photons/m²/s).
    """
    
    # Ensure chlorophyll is within bounds
    zchl = np.fmax(0.01, np.fmin(10.0, pchl_loc[:,1]))
    ichl = (40 + 20 * np.log10(zchl)).astype(np.int64)

    # Extract attenuation coefficients
    ek_blu = p_Chl_k[ichl, 1]
    ek_gre = p_Chl_k[ichl, 2]
    ek_red = p_Chl_k[ichl, 3]

    # Initialize PAR arrays
    par_blu = np.zeros(len(Grid.zgrid))
    par_gre = np.zeros(len(Grid.zgrid))
    par_red = np.zeros(len(Grid.zgrid))

    # Compute PAR at the surface
    par_blu[0] = par / 3.0 * np.exp(-0.5 * ek_blu[0] * Grid.dz) * BGC.PAR_bio
    par_gre[0] = par / 3.0 * np.exp(-0.5 * ek_gre[0] * Grid.dz) * BGC.PAR_bio
    par_red[0] = par / 3.0 * np.exp(-0.5 * ek_red[0] * Grid.dz) * BGC.PAR_bio

    # Compute PAR attenuation with depth
    for kk in range(1, len(Grid.zgrid)):
        par_blu[kk] = par_blu[kk - 1] * np.exp(-ek_blu[kk - 1] * Grid.dz)
        par_gre[kk] = par_gre[kk - 1] * np.exp(-ek_gre[kk - 1] * Grid.dz)
        par_red[kk] = par_red[kk - 1] * np.exp(-ek_red[kk - 1] * Grid.dz)

    # Total PAR
    par_tot = par_blu + par_gre + par_red
    
    # Compute mixed layer mean PAR
    k_mld = np.argmin(np.abs(-Grid.zgrid - mld)) + 1
    par_tot_mldsum = np.sum(par_tot[:k_mld] * Grid.dz)
    par_z_mldsum = Grid.dz * k_mld
    par_tot_mld = np.zeros_like(par_tot)
    par_tot_mld[:k_mld] = par_tot_mldsum / par_z_mldsum
    par_tot_mld[k_mld::] = par_tot[k_mld::]
    
    return {
        "par_tot": par_tot,
        "par_tot_mld": par_tot_mld,
    }
