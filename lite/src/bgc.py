import numpy as np
import PyCO2SYS as pyco2
#from numba import njit

#@njit
def compute_light_limit(par_tot, phy_loc, pchl_loc, tc, BGC):
    """
    Simulate phytoplankton light limitation.
    
    Inputs:
        - par_tot   ::  Total light available for phytoplankton
        - phy_loc   ::  phytoplankton concentration (µmolC/L)
        - pchl_loc  ::  phytoplankton chlorophyll concentration (µgChl/L)
    
    
    Returns:
        dict: A dictionary containing the following keys:
        - `phy_limpar` (np.ndarray): The light limitation factor for phytoplankton growth, 
          between 0 (no growth due to light limitation) and 1 (light is not limiting).

        - `phy_pisl` (np.ndarray): The photosynthesis-irradiance (PI) slope, representing 
          the initial slope of the photosynthesis-light response curve. It is influenced 
          by the chlorophyll-to-carbon ratio and defines the efficiency of light utilization 
          by phytoplankton.
          
        - `chlc_ratio` (np.ndarray): The chlorophyll to carbon ratio (µgChl / µgC)
    """
    
    # Non-zero tracers
    phy_loc = np.fmax(0.0, phy_loc[:,1])
    pchl_loc = np.fmax(0.0, pchl_loc[:,1])
    
    # Constants
    d2s = 1.0 / 86400.0
    chlc_ratio = np.zeros(len(phy_loc))
    chlc_ratio[phy_loc > 0.0] = pchl_loc[phy_loc > 0.0] / (phy_loc[phy_loc > 0.0] * 12.0)
    Tfunc_het = BGC.hete_aT * BGC.hete_bT**(tc)
    
    # Calculate PI slope
    phy_pisl = np.fmax(BGC.alpha * chlc_ratio, BGC.alpha * BGC.phy_minchlc)
    # Adjust slope based on the rate of respiration of the cell
    phy_pisl2 = phy_pisl / (1.0 + BGC.phy_lmort / d2s * Tfunc_het)
    # Solve for the light limitation
    phy_limpar = 1.0 - np.exp( -phy_pisl2 * par_tot )
    
    return {
        "phy_limpar": phy_limpar,
        "phy_pisl": phy_pisl,
        "chlc_ratio": chlc_ratio,
    }


#@njit
def compute_nutrient_limit(phy_loc, phyfe_loc, no3_loc, nh4_loc, dfe_loc, chlc_ratio, tc, BGC):
    """
    Simulate phytoplankton nutrient limitation.
    
    Inputs:
        - phy_loc       ::  phytoplankton concentration (µmolC/L)
        - phyfe_loc     ::  phytoplankton intracellular iron concentration (µmolFe/L)
        - no3_loc       ::  nitrate concentration (µmol/L)
        - nh4_loc       ::  ammonium concentration (µmol/L)
        - dfe_loc       ::  dissolved iron concentration (µmol/L)
        - chlc_ratio    ::  chlorophyll to carbon ratio of phytoplankton (µg/µg)
    
    Returns:
        dict: A dictionary containing the following keys:
        - `phy_limnit` (np.ndarray): Limitation of phytoplankton growth by nitrogen (NO3 + NH4).
          This is a fraction between 0 and 1, where 0 indicates severe nitrate limitation
          and 1 indicates no limitation by nitrate.

        - `phy_limnh4` (np.ndarray): Limitation of phytoplankton growth by ammonium (NH4).
          This is a fraction between 0 and 1, where 0 indicates severe nitrate limitation
          and 1 indicates no limitation by nitrate.

        - `phy_limno3` (np.ndarray): Limitation of phytoplankton growth by nitrate (NO3).
          This is a fraction between 0 and 1, where 0 indicates severe nitrate limitation
          and 1 indicates no limitation by nitrate.

        - `phy_limdfe` (np.ndarray): Limitation of phytoplankton growth by dissolved iron (dFe).
          This is a fraction between 0 and 1, where 0 indicates severe iron limitation and
          1 indicates no limitation by iron. The limitation depends on the chlorophyll-to-carbon
          ratio and the availability of dissolved iron.
          
        - `phy_k_dfe` (np.ndarray): Half-saturation coefficient for dissolved iron uptake, 
           considering variations in the phytoplankton community mean cell size proxied by 
           using a power-law function.

        - `phy_limnut` (np.ndarray): Combined nutrient limitation for phytoplankton growth, 
          considering both nitrate and iron availability. This is the minimum of `phy_limnit` 
          and `phy_limdfe`, representing the most limiting nutrient.
    """

    epsi = 1e-20
    
    # Non-zero tracers
    phy_loc = np.fmax(0.0, phy_loc[:,1])
    phyfe_loc = np.fmax(0.0, phyfe_loc[:,1])
    no3_loc = np.fmax(0.0, no3_loc[:,1])
    nh4_loc = np.fmax(0.0, nh4_loc[:,1])
    fe_loc = np.fmax(0.0, dfe_loc[:,1])
    
    # phytoplankton Fe:C ratio
    phyFeC = np.zeros(len(phy_loc))
    phyFeC[phy_loc>0.0] = phyfe_loc[phy_loc>0.0] / phy_loc[phy_loc>0.0]
    
    # Calculate variable half-saturation coefficients for NO3 and dFe
    phy_k_nit = BGC.phy_kn * np.fmax(0.1, phy_loc - BGC.phy_biothresh)**(0.37)
    phy_k_dfe = BGC.phy_kf * np.fmax(0.1, phy_loc - BGC.phy_biothresh)**(0.37)
    
    # Calculate limitation of growth by Nitrogen (NO3 and NH4)
    phy_lnh4 = nh4_loc / (nh4_loc + phy_k_nit)
    phy_lno3 = no3_loc / (no3_loc + phy_k_nit)
    phy_limNtot = (no3_loc + nh4_loc) / ( phy_k_nit + no3_loc + nh4_loc )
    phy_limnh4 = phy_limNtot * 5.0 * phy_lnh4 / ( phy_lno3 + 5.0 * phy_lnh4 + epsi)
    phy_limno3 = phy_limNtot * phy_lno3 / ( phy_lno3 + 5.0 * phy_lnh4 + epsi )
    phy_limnit = phy_limnh4 + phy_limno3
        
    # Calculate limitation of growth by dFe (chl:C ratio is here converted into µg Chl per µmol C)
    phy_minFeC = 0.0016 / 55.85 * chlc_ratio*12.0 + \
                 1.21e-5 * 14.0 / 55.85 / 7.625 * 0.5 * 1.5 * phy_limnit + \
                 1.15e-4 * 14.0 / 55.85 / 7.625 * 0.5 * phy_limno3
    phy_limdfe = np.fmin(1.0, np.fmax(0.0, ( phyFeC - phy_minFeC ) / BGC.phy_optFeC ))
    
    # total nutrient limitation
    phy_limnut = np.fmin(phy_limnit, phy_limdfe)

    return {
        "phy_limnit": phy_limnit,
        "phy_limnh4": phy_limnh4,
        "phy_limno3": phy_limno3,
        "phy_limdfe": phy_limdfe,
        "phy_k_dfe": phy_k_dfe,
        "phy_limnut": phy_limnut,
    }


#@njit
def compute_primary_production(tc, phy_limpar, phy_limnut, BGC):
    """
    Simulate primary production by phytoplankton, considering light and nutrient limitations.

    Inputs:
        - tc (float or np.ndarray): Temperature (°C), which influences the maximum growth rate.
        - phy_limpar (np.ndarray): Light limitation factor for phytoplankton growth, ranging from 
          0 (severe light limitation) to 1 (no light limitation).
        - phy_limnut (np.ndarray): Nutrient limitation factor for phytoplankton growth, ranging from 
          0 (severe nutrient limitation) to 1 (no nutrient limitation).

    Returns:
        dict: A dictionary containing the following keys:
        - `phy_mumax` (float or np.ndarray): Temperature-dependent maximum growth rate (s⁻¹).
           This is the theoretical upper limit of phytoplankton growth rate at the given temperature.
          
        - `phy_mupar` (np.ndarray): Growth rate limited by light availability (s⁻¹). 
           This incorporates the maximum growth rate adjusted for light limitation.

        - `phy_mu` (np.ndarray): Realized phytoplankton growth rate (s⁻¹), incorporating both
           light and nutrient limitations. This is the product of `phy_mupar` and `phy_limnut`.

    """
    
    # Constants
    d2s = 1.0 / 86400.0
    
    # Maximum growth rate
    phy_mumax = BGC.auto_aT * BGC.auto_bT**(tc) * d2s

    # Apply light limitation
    phy_mupar = phy_mumax * phy_limpar

    # Apply nutrient limitation
    phy_mu = phy_mupar * phy_limnut
    
    return {
        "phy_mumax": phy_mumax,
        "phy_mupar": phy_mupar,
        "phy_mu": phy_mu,
    }


#@njit
def compute_chlorophyll_growth_rate(phy_pisl, phy_mumax, phy_limnut, par_tot_mld, phy_mu, phy_loc, BGC):
    """
    Calculate the growth rate of chlorophyll, considering light and nutrient availability.

    Inputs:
        - phy_pisl (np.ndarray): Photosynthesis-irradiance (PI) slope (µmolC/µmol photons).
        - phy_mumax (float or np.ndarray): Maximum phytoplankton growth rate (s⁻¹).
        - phy_limnut (np.ndarray): Nutrient limitation factor for phytoplankton growth (0-1).
        - par_phy_mld (np.ndarray): PAR within the mixed layer (µmol photons/m²/s).
        - phy_mu (np.ndarray): Realized phytoplankton growth rate (s⁻¹).
        - phy_loc (np.ndarray): Phytoplankton concentration (µmolC/L).
        
    Returns:
        dict: A dictionary containing the following keys:
            - `chl_mu` (np.ndarray): Growth rate of chlorophyll (µmolChl/L/day).
            - `chl_limpar` (np.ndarray): Light limitation factor specific to chlorophyll.
    """
    
    # Constants
    d2s = 1.0 / 86400.0

    # Non-zero tracers
    phy_loc = np.fmax(0.0, phy_loc[:,1])
    
    # Calculate PI slope adjusted for maximum growth rate and nutrient limitation
    #   - increase slope if nutrient rich
    #   - decrease slope if growing fast (temperature is high)
    chl_pisl = phy_pisl / (phy_mumax / d2s * (1.0 - phy_limnut))

    # Light limitation factor specific to chlorophyll and average light availabe in mixed layer
    chl_limpar = (1.0 - np.exp(-chl_pisl * par_tot_mld))

    # Minimum and optimal growth rates in chlorophyll
    chl_mumin = BGC.phy_minchlc * phy_mu * phy_loc * 12.0
    chl_muopt = BGC.phy_optchlc * phy_mu * phy_loc * 12.0

    # Realized chlorophyll growth rate
    chl_mu = (chl_muopt - chl_mumin) * chl_limpar * phy_limnut

    # Adjust (increase) chlorophyll growth rate based on (low) light availability to phytoplankton
    for kk in range(len(phy_loc) - 1):
        if (phy_pisl[kk] * par_tot_mld[kk]) > 0.0:
            chl_mu[kk] = chl_mumin[kk] + chl_mu[kk] / (phy_pisl[kk] * par_tot_mld[kk])

    return {
        "chl_mu": chl_mu,
        "chl_limpar": chl_limpar,
    }


#@njit
def compute_iron_uptake(phy_loc, phyfe_loc, dfe_loc, phy_mumax, phy_limdfe, phy_k_dfe, BGC):
    """
    Compute dissolved iron uptake by phytoplankton.

    Parameters:
        phy_loc (np.ndarray): Phytoplankton concentration (µmolC/L).
        phyfe_loc (np.ndarray): Phytoplankton intracellular iron concentration (µmolFe/L).
        dfe_loc (np.ndarray): Dissolved iron concentration (µmolFe/L).
        phy_mumax (float or np.ndarray): Maximum phytoplankton growth rate (s⁻¹).
        phy_limdfe (np.ndarray): Iron limitation factor for phytoplankton growth (0–1).
        phy_k_dfe (float): Half-saturation constant for dissolved iron uptake (µmolFe/L).
        
    Returns:
        dict: A dictionary containing:
            - `phy_dfeupt` (np.ndarray): Rate of dissolved iron uptake by phytoplankton (µmolFe/L/s).
    """
    
    # Constants
    eps = 1e-16
    
    # Non-zero tracers
    phy_loc = np.fmax(0.0, phy_loc[:,1])
    phyfe_loc = np.fmax(0.0, phyfe_loc[:,1])
    dfe_loc = np.fmax(0.0, dfe_loc[:,1])
    
    # Calculate the maximum intracellular iron concentration (Fe:C quota)
    phy_maxQFe = phy_loc * BGC.phy_maxFeC

    # Iron uptake upregulation factor
    phy_Feupt_upreg = 4.0 - 4.5 * phy_limdfe / (phy_limdfe + 0.5)

    # Iron uptake downregulation factor
    phy_Feupt_downreg = np.fmax(0.0, 
                                (1.0 - phyfe_loc / (phy_maxQFe + eps)) / np.abs(1.05 - phyfe_loc / (phy_maxQFe + eps))
    )

    # Dissolved iron uptake rate
    phy_dfeupt = (
        phy_loc * phy_mumax * BGC.phy_maxFeC *
        dfe_loc / (dfe_loc + phy_k_dfe) *
        phy_Feupt_downreg * phy_Feupt_upreg
    )

    return {
        "phy_dfeupt": phy_dfeupt,
    }


#@njit
def compute_grazing(tc, aoa_loc, nob_loc, phy_loc, det_loc, zoo_loc, mphy_loc, mdet_loc, maoa_loc, mnob_loc, BGC):
    """
    Compute zooplankton grazing rates on phytoplankton and detritus.

    Parameters:
        tc (float): Temperature (°C), affecting grazing rates.
        aoa_loc (np.ndarray): AOA concentration (µmolC/L).
        nob_loc (np.ndarray): NOB concentration (µmolC/L).
        phy_loc (np.ndarray): Phytoplankton concentration (µmolC/L).
        det_loc (np.ndarray): Detritus concentration (µmolC/L).
        zoo_loc (np.ndarray): Zooplankton concentration (µmolC/L).
        mphy_loc (np.ndarray): Long-term mean phytoplankton concentration (µmolC/L).
        mdet_loc (np.ndarray): Long-term mean detritus concentration (µmolC/L).
        maoa_loc (np.ndarray): Long-term mean AOA concentration (µmolC/L).
        mnob_loc (np.ndarray): Long-term mean NOB concentration (µmolC/L).
        
    Returns:
        dict: A dictionary containing:
            - `zoo_mumax` (np.ndarray): Maximum zooplankton grazing rate (s⁻¹).
            - `zoo_epsilon` (np.ndarray): Prey capture rate coefficient (m6/(mmol2) s⁻¹).
            - `zoo_mu` (np.ndarray): Realized zooplankton grazing rate (s⁻¹).
            - `zoo_grzphy` (np.ndarray): Grazing rate on phytoplankton (µmolC/L/s).
            - `zoo_grzdet` (np.ndarray): Grazing rate on detritus (µmolC/L/s).
            - `zoo_grzaoa` (np.ndarray): Grazing rate on AOA (µmolC/L/s).
            - `zoo_grznob` (np.ndarray): Grazing rate on NOB (µmolC/L/s).
    """
    
    # Non-zero tracers
    aoa_loc = np.fmax(0.0, aoa_loc[:,1])
    nob_loc = np.fmax(0.0, nob_loc[:,1])
    phy_loc = np.fmax(0.0, phy_loc[:,1])
    det_loc = np.fmax(0.0, det_loc[:,1])
    zoo_loc = np.fmax(0.0, zoo_loc[:,1])
    
    # Maximum grazing rate
    Tfunc_hete = BGC.hete_aT * BGC.hete_bT**(tc)
    zoo_mumax = BGC.zoo_grz * Tfunc_hete

    # Prey availability
    prey = np.fmax(0.0, phy_loc) * BGC.zoo_prefphy + np.fmax(0.0, det_loc) * BGC.zoo_prefdet \
           + np.fmax(0.0, aoa_loc) * BGC.zoo_prefaoa + np.fmax(0.0, nob_loc) * BGC.zoo_prefnob
    mprey = np.fmax(0.0, mphy_loc) * BGC.zoo_prefphy + np.fmax(0.0, mdet_loc) * BGC.zoo_prefdet \
           + np.fmax(0.0, maoa_loc) * BGC.zoo_prefaoa + np.fmax(0.0, mnob_loc) * BGC.zoo_prefnob

    # Prey capture rate based on grazing formulation
    zoo_peffect = np.exp(-mprey * BGC.zoo_epsrat)
    zoo_epsilon = BGC.zoo_epsmin + (BGC.zoo_epsmax - BGC.zoo_epsmin) * zoo_peffect
    
    # Realized grazing rate
    zoo_capt = zoo_epsilon * prey**2
    zoo_mu = zoo_mumax * zoo_capt / (zoo_mumax + zoo_capt)

    # Grazing rates on phytoplankton and detritus
    zoo_grzphy = np.zeros(len(phy_loc))
    zoo_grzdet = np.zeros(len(phy_loc))
    zoo_grzaoa = np.zeros(len(phy_loc))
    zoo_grznob = np.zeros(len(phy_loc))
    valid_indices = prey > BGC.biomin
    zoo_grzphy[valid_indices] = (
        zoo_mu[valid_indices] * zoo_loc[valid_indices] * np.fmax(phy_loc[valid_indices],0.0) * BGC.zoo_prefphy / prey[valid_indices]
    )
    zoo_grzdet[valid_indices] = (
        zoo_mu[valid_indices] * zoo_loc[valid_indices] * np.fmax(det_loc[valid_indices],0.0) * BGC.zoo_prefdet / prey[valid_indices]
    )
    zoo_grzaoa[valid_indices] = (
        zoo_mu[valid_indices] * zoo_loc[valid_indices] * np.fmax(aoa_loc[valid_indices],0.0) * BGC.zoo_prefaoa / prey[valid_indices]
    )
    zoo_grznob[valid_indices] = (
        zoo_mu[valid_indices] * zoo_loc[valid_indices] * np.fmax(nob_loc[valid_indices],0.0) * BGC.zoo_prefnob / prey[valid_indices]
    )
    
    return {
        "zoo_mumax": zoo_mumax,
        "zoo_epsilon": zoo_epsilon,
        "zoo_mu": zoo_mu,
        "zoo_grzphy": zoo_grzphy,
        "zoo_grzdet": zoo_grzdet,
        "zoo_grzaoa": zoo_grzaoa,
        "zoo_grznob": zoo_grznob,
    }


#@njit
def compute_losses(tc, aoa_loc, nob_loc, phy_loc, zoo_loc, det_loc, BGC):
    """
    Compute loss terms for phytoplankton and zooplankton and detritus.

    Parameters:
        tc (float): Temperature (°C), affecting metabolic rates.
        aoa_loc (np.ndarray): AOA concentration (µmolC/L).
        nob_loc (np.ndarray): NOB concentration (µmolC/L).
        phy_loc (np.ndarray): Phytoplankton concentration (µmolC/L).
        zoo_loc (np.ndarray): Zooplankton concentration (µmolC/L).
        det_loc (np.ndarray): Detritus concentration (µmolC/L).

    Returns:
        dict: A dictionary containing:
            - `phy_lmort` (np.ndarray): Linear mortality rate for phytoplankton (µmolC/L/s).
            - `phy_qmort` (np.ndarray): Quadratic mortality rate for phytoplankton (µmolC/L/s).
            - `zoo_zoores` (np.ndarray): Zooplankton respiration loss (µmolC/L/s).
            - `zoo_qmort` (np.ndarray): Quadratic mortality rate for zooplankton (µmolC/L/s).
            - `det_remin` (np.ndarray): Remineralisation rate of detritus (µmolC/L/s).
            - `aoa_lmort` (np.ndarray): Linear mortality rate for AOA (µmolC/L/s).
            - `nob_lmort` (np.ndarray): Linear mortality rate for NOB (µmolC/L/s).
            - `aoa_qmort` (np.ndarray): Quadratic mortality rate for AOA (µmolC/L/s).
            - `nob_qmort` (np.ndarray): Quadratic mortality rate for NOB (µmolC/L/s).
            
    """
    
    # Non-zero tracers
    aoa_loc = np.fmax(0.0, aoa_loc[:,1])
    nob_loc = np.fmax(0.0, nob_loc[:,1])
    phy_loc = np.fmax(0.0, phy_loc[:,1])
    zoo_loc = np.fmax(0.0, zoo_loc[:,1])
    det_loc = np.fmax(0.0, det_loc[:,1])
    
    # Heterotrophic metabolic rate
    Tfunc_hete = BGC.hete_aT * BGC.hete_bT**(tc)
    
    # Initialize mortality terms
    aoa_lmort = np.zeros(len(phy_loc))
    aoa_qmort = np.zeros(len(phy_loc))
    nob_lmort = np.zeros(len(phy_loc))
    nob_qmort = np.zeros(len(phy_loc))
    phy_lmort = np.zeros(len(phy_loc))
    phy_qmort = np.zeros(len(phy_loc))
    zoo_zoores = np.zeros(len(phy_loc))
    zoo_qmort = np.zeros(len(phy_loc))
    det_remin = np.zeros(len(phy_loc))

    # Phytoplankton linear mortality
    valid_phy = phy_loc > BGC.biomin
    phy_lmort[valid_phy] = (
        BGC.phy_lmort * phy_loc[valid_phy] * Tfunc_hete[valid_phy]
    )

    # Phytoplankton quadratic mortality
    phy_qmort[valid_phy] = (
        BGC.phy_qmort * phy_loc[valid_phy]**2.0
    )

    # AOA linear mortality
    valid_aoa = aoa_loc > BGC.biomin
    aoa_lmort[valid_aoa] = (
        BGC.aoa_lmort * aoa_loc[valid_aoa] * Tfunc_hete[valid_aoa]
    )

    # NOB linear mortality
    valid_nob = nob_loc > BGC.biomin
    nob_lmort[valid_nob] = (
        BGC.nob_lmort * nob_loc[valid_nob] * Tfunc_hete[valid_nob]
    )

    # AOA quadratic mortality
    aoa_qmort[valid_aoa] = (
        BGC.aoa_qmort * aoa_loc[valid_aoa]**2.0
    )

    # NOB quadratic mortality
    nob_qmort[valid_nob] = (
        BGC.nob_qmort * nob_loc[valid_nob]**2.0
    )

    # Zooplankton respiration
    valid_zoo = zoo_loc > BGC.biomin
    zoo_zoores[valid_zoo] = (
        zoo_loc[valid_zoo] * BGC.zoo_respi * Tfunc_hete[valid_zoo] *
        zoo_loc[valid_zoo] / (zoo_loc[valid_zoo] + BGC.zoo_kzoo)
    )

    # Zooplankton quadratic mortality
    zoo_qmort[valid_zoo] = (
        BGC.zoo_qmort * zoo_loc[valid_zoo]**2.0
    )
    
    # Detritus remineralisation
    det_remin = (BGC.detrem * Tfunc_hete * det_loc**2.0
    )

    return {
        "phy_lmort": phy_lmort,
        "phy_qmort": phy_qmort,
        "aoa_lmort": aoa_lmort,
        "aoa_qmort": aoa_qmort,
        "nob_lmort": nob_lmort,
        "nob_qmort": nob_qmort,
        "zoo_zoores": zoo_zoores,
        "zoo_qmort": zoo_qmort,
        "det_remin": det_remin,
    }


#@njit
def compute_iron_chemistry(tck, dfe_loc, det_loc, mld, Grid):
    """
    Simulate iron chemistry processes: equilibrium, precipitation, scavenging, and coagulation.

    Inputs:
        - tck       ::  Temperature in degrees kelvin
        - dfe_loc   ::  dissolved iron concentration (µmol/L)
        - det_loc   ::  detritus concentration (µmol/L)
        - mld       ::  mixed layer depth (m)
        
    Returns:
        dict: Iron-related terms (precipitation, scavenging, coagulation, etc.).
            
    """
    
    # Constants
    d2s = 1.0 / 86400.0
    ligand = 0.7 * 1e-9
    
    # Non-zero tracers
    dfe_loc = np.fmax(0.0, dfe_loc[:,1])
    det_loc = np.fmax(0.0, det_loc[:,1])
    
    # Equilibrium fractionation
    fe_keq = 10 ** (17.27 - 1565.7 / tck)
    dfe_III = (-(1.0 + fe_keq * ligand - fe_keq * dfe_loc * 1e-6) + np.sqrt(
        (1.0 + fe_keq * ligand - fe_keq * dfe_loc * 1e-6) ** 2 + 4.0 * dfe_loc * 1e-6 * fe_keq)) / (2.0 * fe_keq) * 1e6
    dfe_lig = np.fmax(0.0, dfe_loc - dfe_III)

    # Precipitation
    sal = 35.0
    zval = 19.924 * sal / (1000.0 - 1.005 * sal)
    fesol1 = 10 ** (-13.486 - 0.1856 * zval ** 0.5 + 0.3073 * zval + 5254.0 / np.fmax(tck, 278.15))
    fesol2 = 10 ** (2.517 - 0.8885 * zval ** 0.5 + 0.2139 * zval - 1320.0 / np.fmax(tck, 278.15))
    fesol3 = 10 ** (0.4511 - 0.3305 * zval ** 0.5 - 1996.0 / np.fmax(tck, 278.15))
    fesol4 = 10 ** (-0.2965 - 0.7881 * zval ** 0.5 - 4086.0 / np.fmax(tck, 278.15))
    fesol5 = 10 ** (4.4466 - 0.8505 * zval ** 0.5 - 7980.0 / np.fmax(tck, 278.15))
    hp = 10 ** (-7.9)
    fe3sol = fesol1 * (hp ** 3 + fesol2 * hp ** 2 + fesol3 * hp + fesol4 + fesol5 / hp) * 1e6
    dfe_prec = np.fmax(0.0, (dfe_III - fe3sol)) * 0.01 * d2s

    # Scavenging
    dfe_scav = dfe_III * (3e-5 + 0.5 * det_loc) * d2s
    
    # Coagulation
    dfe_col = dfe_lig * 0.5
    doc_loc = 40e-6
    zval = (0.369 * 0.3 * doc_loc + 102.4 * det_loc * 1e-6) + (114.0 * 0.3 * doc_loc)
    zval[-Grid.zgrid > mld] *= 0.01
    dfe_coag = dfe_col * zval * d2s

    return {
        "dfe_III": dfe_III,
        "dfe_lig": dfe_lig,
        "dfe_prec": dfe_prec,
        "dfe_scav": dfe_scav,
        "dfe_coag": dfe_coag,
    }


#@njit
def compute_co2_flux(dic_surface, alk_surface, sst, wind_speed, atm_co2):
    """
    Computes the air-sea CO2 gas exchange flux.
    
    Parameters:
        dic_surface (float): Surface DIC concentration (mmol/m3)
        alk_surface (float): Surface alkalinity (mmol/m3)
        sst (float): Sea surface temperature (°C)
        wind_speed (float): Surface wind speed (m/s)
        co2_atm (float): Atmospheric CO2 partial pressure (ppm)
        
    Returns:
        float: Net CO2 flux (mmol C / m2 / s)
    """
    
    # Convert atmospheric CO2 from ppm to atm (assuming 1 atm total pressure)
    pCO2_air = atm_co2 * 1e-6 # atm
    
    # Conditions and constants
    sal = 35.0
    rho = 1025.0
    
    # Estimate pCO2
    params = {
        'par1': alk_surface,
        'par2': dic_surface,
        'par1_type': 1,
        'par2_type': 2,
        'temperature': sst,
        'salinity': sal,
        'pressure': 0.0
        }
    res = pyco2.sys(**params)
    pCO2_water = res['pCO2'] * 1e-6 # atm 
    sol = res['k_CO2'] * rho * 1e3 # mmol/m3/atm
    
    # Compute Schmidt number
    a1 = 2073.1
    a2 = -125.62
    a3 = 3.6276
    a4 = -0.043219
    Sc = a1 + sst*(a2 + sst*(a3 + sst*a4))
    
    # Compute gas transfer velocity k (cm/hr), convert to m/s (Wanninkhof 2014)
    k = 0.251 * wind_speed**2 * (Sc / 660) ** -0.5  # cm/hr
    k *= 1.0 / 3600.0 / 100.0  # Convert cm/hr to m/s
    
    # Compute air-sea CO2 flux (positive = into ocean, negative = out)
    co2_flux = k * sol * (pCO2_air - pCO2_water)  # mmol/m2/s
    
    return {
        "co2_flux": co2_flux,
        "pCO2_water": pCO2_water,
        }


#@njit
def compute_nitrification(tc, nh4_loc, no2_loc, o2_loc, BGC):
    """
    Simulate nitrification by AOA and NOB considering nutrient limitations.

    Inputs:
        - tc (float or np.ndarray): Temperature (°C), which influences the maximum growth rate.
        - nh4_loc (np.ndarray): Ammonium concentration (µmolN/L).
        - no2_loc (np.ndarray): Nitrite concentration (µmolN/L).
        
    Returns:
        dict: A dictionary containing the following keys:
        - `aoa_mumax` (float or np.ndarray): Temperature-dependent maximum growth rate (s⁻¹).
           This is the theoretical upper limit of AOA growth rate at the given temperature.
        - `nob_mumax` (float or np.ndarray): Temperature-dependent maximum growth rate (s⁻¹).
           This is the theoretical upper limit of NOB growth rate at the given temperature.
        - `aoa_limnh4` (np.ndarray): Limitation by ammonium availability (). 
        - `nob_limno2` (np.ndarray): Limitation by nitrite availability (). 
        - `aoa_limo2` (np.ndarray): Limitation by O2 availability (). 
        - `nob_limo2` (np.ndarray): Limitation by O2 availability (). 
        - `aoa_mu` (np.ndarray): Realized AOA growth rate (s⁻¹), incorporating 
           nutrient limitation. This is the product of `phy_mumax` and `phy_limnh4`.
        - `nob_mu` (np.ndarray): Realized NOB growth rate (s⁻¹), incorporating 
           nutrient limitation. This is the product of `phy_mumax` and `phy_limnh4`.
    """
    
    # Constants
    d2s = 1.0 / 86400.0

    nh4_loc = np.fmax(0.0, nh4_loc[:,1])
    no2_loc = np.fmax(0.0, no2_loc[:,1])
    o2_loc = np.fmax(0.0, o2_loc[:,1])

    # Maximum growth rate
    aoa_mumax = BGC.aoa_aT * BGC.chem_bT**(tc) * d2s
    nob_mumax = BGC.nob_aT * BGC.chem_bT**(tc) * d2s

    # Nutrient limitation
    aoa_limnh4 = nh4_loc / (BGC.aoa_kn + nh4_loc)
    nob_limno2 = no2_loc / (BGC.nob_kn + no2_loc)

    # Oxygen limitation
    aoa_limo2 = o2_loc * BGC.aoa_pO2 * BGC.aoa_yo2
    nob_limo2 = o2_loc * BGC.nob_pO2 * BGC.nob_yo2
    
    # Apply nutrient limitation
    aoa_mu = aoa_mumax * np.fmin(aoa_limnh4, aoa_limo2)
    nob_mu = nob_mumax * np.fmin(nob_limno2, nob_limo2)

    return {
        "aoa_mumax": aoa_mumax,
        "aoa_limnh4": aoa_limnh4,
        "aoa_limo2": aoa_limo2,
        "aoa_mu": aoa_mu,
        "nob_mumax": nob_mumax,
        "nob_limno2": nob_limno2,
        "nob_limo2": nob_limo2,
        "nob_mu": nob_mu,
    }


#@njit
def compute_sourcessinks(phy_mu, phy_lmort, phy_qmort, phy_dfeupt, phy_limnh4, phy_limno3,
                         zoo_zoores, zoo_qmort, zoo_grzphy, zoo_grzdet, zoo_grzaoa, zoo_grznob, det_remin,
                         aoa_mu, nob_mu, aoa_lmort, nob_lmort, aoa_qmort, nob_qmort, aoa_loc, nob_loc,
                         phy_loc, phyfe_loc, zoo_loc, zoofe_loc, det_loc, detfe_loc, 
                         dfe_prec, dfe_scav, dfe_coag,
                         chl_mu, chlc_ratio, 
                         co2_flux, 
                         BGC):
    """
    Compute the source and sink terms for various biogeochemical tracers.

    Parameters:
        - phy_mu (np.ndarray): Realized phytoplankton growth rate (s⁻¹).
        - phy_lmort (np.ndarray): Linear mortality rate for phytoplankton (µmolC/L/s).
        - phy_qmort (np.ndarray): Quadratic mortality rate for phytoplankton (µmolC/L/s).
        - phy_dfeupt (np.ndarray): Phytoplankton dissolved iron uptake (µmolFe/L/s).
        - phy_limnh4 (np.ndarray): Phytoplankton limitation by Ammonium ().
        - phy_limno3 (np.ndarray): Phytoplankton limitation by Nitrate ().
        - zoo_zoores (np.ndarray): Zooplankton respiration loss (µmolC/L/s).
        - zoo_qmort (np.ndarray): Quadratic mortality rate for zooplankton (µmolC/L/s).
        - zoo_grzphy (np.ndarray): Grazing rate on phytoplankton (µmolC/L/s).
        - zoo_grzdet (np.ndarray): Grazing rate on detritus (µmolC/L/s).
        - zoo_grzaoa (np.ndarray): Grazing rate on AOA (µmolC/L/s).
        - zoo_grznob (np.ndarray): Grazing rate on NOB (µmolC/L/s).
        - det_remin (np.ndarray): Detritus remineralization rate (µmolC/L/s).
        - aoa_mu (np.ndarray): Realized AOA growth rate (s⁻¹).
        - nob_mu (np.ndarray): Realized NOB growth rate (s⁻¹).
        - aoa_lmort (np.ndarray): Linear mortality rate for AOA (µmolC/L/s).
        - nob_lmort (np.ndarray): Linear mortality rate for NOB (µmolC/L/s).
        - aoa_qmort (np.ndarray): Quadratic mortality rate for AOA (µmolC/L/s).
        - nob_qmort (np.ndarray): Quadratic mortality rate for NOB (µmolC/L/s).
        
        - phyFeC (np.ndarray): Phytoplankton iron-to-carbon ratio.
        - zooFeC (np.ndarray): Zooplankton iron-to-carbon ratio.
        - detFeC (np.ndarray): Detritus iron-to-carbon ratio.
        
        - dfe_prec (np.ndarray): Iron precipitation rate (µmolFe/L/s).
        - dfe_scav (np.ndarray): Scavenging rate for iron (µmolFe/L/s).
        - dfe_coag (np.ndarray): Coagulation of iron into detritus (µmolFe/L/s).
        - chl_mu (np.ndarray): Chlorophyll growth rate (µmolChl/L/s).
        - chlc_ratio (np.ndarray): Chlorophyll-to-carbon ratio (µgChl/µgC).
        - co2_flux (float): Air-sea CO2 flux (mmol/m3/s)

    Returns:
        dict: A dictionary containing:
            - `ddt_no3` (np.ndarray): Tendency for nitrate concentration (µmolN/L/s).
            - `ddt_nh4` (np.ndarray): Tendency for ammonium concentration (µmolN/L/s).
            - `ddt_no2` (np.ndarray): Tendency for nitrite concentration (µmolN/L/s).
            - `ddt_dfe` (np.ndarray): Tendency for dissolved iron concentration (µmolFe/L/s).
            - `ddt_aoa` (np.ndarray): Tendency for AOA concentration (µmolC/L/s).
            - `ddt_nob` (np.ndarray): Tendency for NOB concentration (µmolC/L/s).
            - `ddt_phy` (np.ndarray): Tendency for phytoplankton concentration (µmolC/L/s).
            - `ddt_phyfe` (np.ndarray): Tendency for phytoplankton iron (µmolFe/L/s).
            - `ddt_zoo` (np.ndarray): Tendency for zooplankton concentration (µmolC/L/s).
            - `ddt_zoofe` (np.ndarray): Tendency for zooplankton iron (µmolFe/L/s).
            - `ddt_det` (np.ndarray): Tendency for detritus concentration (µmolC/L/s).
            - `ddt_detfe` (np.ndarray): Tendency for detritus iron (µmolFe/L/s).
            - `ddt_pchl` (np.ndarray): Tendency for chlorophyll concentration (µgChl/L/s).
            - `ddt_dic` (np.ndarray): Tendency for DIC concentration (µmolC/L/s).
            - `ddt_alk` (np.ndarray): Tendency for Alkalinity concentration (µmolEq/L/s).
            - `ddt_o2` (np.ndarray): Tendency for dissolved oxygen concentration (µmolO2/L/s).
    """
    
    # Non-zero tracers
    aoa_loc = np.fmax(0.0, aoa_loc[:,1])
    nob_loc = np.fmax(0.0, nob_loc[:,1])
    phy_loc = np.fmax(0.0, phy_loc[:,1])
    zoo_loc = np.fmax(0.0, zoo_loc[:,1])
    det_loc = np.fmax(0.0, det_loc[:,1])
    phyfe_loc = np.fmax(0.0, phyfe_loc[:,1])
    zoofe_loc = np.fmax(0.0, zoofe_loc[:,1])
    detfe_loc = np.fmax(0.0, detfe_loc[:,1])
    
    # Compute ratios
    phyFeC = np.zeros(len(phy_loc))
    zooFeC = np.zeros(len(phy_loc))
    detFeC = np.zeros(len(phy_loc))
    phyFeC[phy_loc>0.0] = phyfe_loc[phy_loc>0.0] / phy_loc[phy_loc>0.0]
    zooFeC[zoo_loc>0.0] = zoofe_loc[zoo_loc>0.0] / zoo_loc[zoo_loc>0.0]
    detFeC[det_loc>0.0] = detfe_loc[det_loc>0.0] / det_loc[det_loc>0.0]
    
    # Compute tracer tendencies
    ddt_no3 = (
        nob_mu * nob_loc * 1.0/BGC.nob_yno2
        - phy_mu * phy_loc * (phy_limno3 / (phy_limnh4 + phy_limno3)) / BGC.phy_CN
    )

    ddt_no2 = (
        aoa_mu * aoa_loc * (1.0/BGC.aoa_ynh4 - 1.0/BGC.phy_CN) 
        - nob_mu * nob_loc * 1.0/BGC.nob_yno2
    )

    ddt_nh4 = (
        (phy_lmort + aoa_lmort + nob_lmort + zoo_zoores + det_remin 
         + (1.0 - BGC.zoo_assim) * BGC.zoo_excre * (zoo_grzphy + zoo_grzdet + zoo_grzaoa + zoo_grznob)) / BGC.phy_CN
        - phy_mu * phy_loc * (phy_limnh4 / (phy_limnh4 + phy_limno3) ) / BGC.phy_CN
        - aoa_mu * aoa_loc * 1.0/BGC.aoa_ynh4
        - nob_mu * nob_loc * 1.0/BGC.phy_CN
    )
        

    ddt_o2 = (
        phy_mu * phy_loc / BGC.phy_CO 
        - (phy_lmort + aoa_lmort + nob_lmort + zoo_zoores + det_remin +
         (1.0 - BGC.zoo_assim) * BGC.zoo_excre * (zoo_grzphy + zoo_grzdet + zoo_grzaoa + zoo_grznob)) / BGC.phy_CO
        - aoa_mu * aoa_loc * 1.0/BGC.aoa_yo2
        - nob_mu * nob_loc * 1.0/BGC.nob_yo2
    )

    ddt_dfe = (
        (phy_lmort * phyFeC + zoo_zoores * zooFeC + aoa_lmort * BGC.aoa_Fe2C + nob_lmort * BGC.nob_lmort + det_remin * detFeC 
         + (1.0 - BGC.zoo_assim) * BGC.zoo_excre * (zoo_grzphy * phyFeC + zoo_grzdet * detFeC + zoo_grzaoa * BGC.aoa_Fe2C + zoo_grznob * BGC.nob_Fe2C))
        - phy_dfeupt - dfe_prec - dfe_scav - dfe_coag
        - aoa_mu * aoa_loc * BGC.aoa_Fe2C
        - nob_mu * nob_loc * BGC.nob_Fe2C
    )

    ddt_aoa = aoa_mu * aoa_loc - (aoa_lmort + aoa_qmort + zoo_grzaoa)
    ddt_nob = nob_mu * nob_loc - (nob_lmort + nob_qmort + zoo_grznob)

    ddt_phy = phy_mu * phy_loc - (phy_lmort + phy_qmort + zoo_grzphy)

    ddt_phyfe = phy_dfeupt - (phy_lmort + phy_qmort + zoo_grzphy) * phyFeC

    ddt_zoo = (zoo_grzphy + zoo_grzdet + zoo_grzaoa + zoo_grznob) * BGC.zoo_assim - (zoo_zoores + zoo_qmort)

    ddt_zoofe = (
        (zoo_grzphy * phyFeC + zoo_grzdet * detFeC + zoo_grzaoa * BGC.aoa_Fe2C + zoo_grznob * BGC.nob_Fe2C) * BGC.zoo_assim
        - (zoo_zoores + zoo_qmort) * zooFeC
    )

    ddt_det = (
        (zoo_grzphy + zoo_grzdet + zoo_grzaoa + zoo_grznob) * (1.0 - BGC.zoo_assim) * (1.0 - BGC.zoo_excre)
        + phy_qmort + zoo_qmort + aoa_qmort + nob_qmort 
        - det_remin - zoo_grzdet
    )

    ddt_detfe = (
        (zoo_grzphy * phyFeC + zoo_grzdet * detFeC + zoo_grzaoa * BGC.aoa_Fe2C + zoo_grznob * BGC.nob_Fe2C) * (1.0 - BGC.zoo_assim) * (1.0 - BGC.zoo_excre)
        + phy_qmort * phyFeC + zoo_qmort * zooFeC + aoa_qmort * BGC.aoa_Fe2C + nob_qmort * BGC.aoa_Fe2C
        - (det_remin + zoo_grzdet) * detFeC
        + dfe_scav + dfe_coag
    )

    ddt_pchl = chl_mu - chlc_ratio * (phy_lmort + phy_qmort + zoo_grzphy) * 12
    
    ddt_dic = ddt_nh4 * BGC.phy_CN
    ddt_dic[0] = ddt_nh4[0] + co2_flux
    ddt_alk = ddt_no3 * (-1.0)

    return {
        "ddt_no3": ddt_no3,
        "ddt_nh4": ddt_nh4,
        "ddt_no2": ddt_no2,
        "ddt_dfe": ddt_dfe,
        "ddt_aoa": ddt_aoa,
        "ddt_nob": ddt_nob,
        "ddt_phy": ddt_phy,
        "ddt_phyfe": ddt_phyfe,
        "ddt_zoo": ddt_zoo,
        "ddt_zoofe": ddt_zoofe,
        "ddt_det": ddt_det,
        "ddt_detfe": ddt_detfe,
        "ddt_pchl": ddt_pchl,
        "ddt_dic": ddt_dic,
        "ddt_alk": ddt_alk,
        "ddt_o2": ddt_o2,
    }


#@njit
def compute_totalN(aoa_loc, nob_loc, phy_loc, zoo_loc, det_loc, nh4_loc, no2_loc, no3_loc, BGC):
    """
    Returns total N.

    Parameters:
        - aoa_loc (np.ndarray): AOA carbon biomass.
        - nob_loc (np.ndarray): NOB carbon biomass.
        - phy_loc (np.ndarray): Phytoplankton carbon biomass.
        - zoo_loc (np.ndarray): Zooplankton carbon biomass.
        - det_loc (np.ndarray): Detrital carbon biomass.
        - nh4_loc (np.ndarray): Ammonium concentration.
        - no2_loc (np.ndarray): Nitrite concentration.
        - no3_loc (np.ndarray): Nitrate concentration.        
        
    Returns:
        dict: A dictionary containing:
            - `totalN` (np.ndarray): Total Nitrogen content (µmolN/L).
            
    """

    totalN = (aoa_loc[:,1] + nob_loc[:,1] + phy_loc[:,1] + zoo_loc[:,1] + det_loc[:,1]) / BGC.phy_CN \
        + nh4_loc[:,1] + no2_loc[:,1] + no3_loc[:,1]
    return totalN

