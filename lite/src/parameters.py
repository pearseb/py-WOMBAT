import numpy as np


class p_Grid:
    """Defines parameters related to the vertical grid."""

    def __init__(self, bot, dz):
        self.bot = bot  # Bottom of the model grid (m)
        self.dz = dz    # Grid spacing (meters)
        self.zgrid = -np.arange(-self.dz / 2.0, self.bot + self.dz / 2.0, self.dz)
        self.npt = len(self.zgrid)  # Number of vertical points


class p_Diff:
    """Defines parameters related to diffusion."""

    def __init__(self, npt):
        # Precompute Kv within the constructor
        self.Kv = 1.701e-5 * np.ones(npt)  # Vertical diffusion (m²/s)


#class p_Advec:
#    """Defines parameters related to advection."""
#
#    def __init__(self, npt, w=1.585e-6):
#        # Precompute w within the constructor
#        self.w = w * np.ones(npt)  # Vertical advection (m/s)


class p_BGC:
    """Defines parameters related to the biogeochemical model."""

    def __init__(self, alpha=2.25, auto_aT=1.15, auto_bT=1.060, hete_aT=1.0, hete_bT=1.066, phy_kn=2.0, phy_kf=1.2e-3, phy_biothresh=0.5, 
                 phy_lmort=0.005, phy_qmort=0.05, phy_minchlc=0.004, phy_optchlc=0.036, phy_optFeC=10e-6, phy_maxFeC=50e-6, 
                 zoo_respi=0.003, zoo_qmort=0.80, zoo_assim=0.6, zoo_excre=0.75, 
                 zoo_grz=3.0, zoo_epsmin=0.025, zoo_epsmax=0.25, zoo_epsrat=0.2, zoo_inerti=30.0, 
                 zoo_prefdet=0.50, detrem=0.50, w0=25.0,
                 aoa_aT=0.5, nob_aT=1.0, chem_bT=1.020, 
                 aoa_kn=0.1, nob_kn=0.1, aoa_pO2=1171.5, nob_pO2=282.7, aoa_ynh4=0.0245, nob_yno2=0.0126, aoa_yo2=0.0183, nob_yo2=0.022, 
                 aoa_Fe2C=10e-6, nob_Fe2C=10e-6, aoa_lmort=0.005, nob_lmort=0.005, aoa_qmort=0.05, nob_qmort=0.05):
        self.d2s = 1.0 / 86400.0  # Conversion factor: days to seconds
        self.alpha = alpha  # PI curve slope for photosynthesis
        self.PAR_bio = 0.43  # Fraction of PAR available biologically
        self.auto_aT = auto_aT  # Maximum growth rate scaling for autotrophs
        self.auto_bT = auto_bT  # Temperature dependence for autotrophs
        self.hete_aT = hete_aT  # Maximum growth rate scaling for heterotrophs
        self.hete_bT = hete_bT  # Temperature dependence for heterotrophs
        self.phy_kn = phy_kn  # Half-saturation constant for NO3 uptake
        self.phy_kf = phy_kf  # Half-saturation constant for dFe uptake
        self.phy_biothresh = phy_biothresh  # Biomass threshold for phytoplankton processes
        self.phy_lmort = phy_lmort * self.d2s  # Linear mortality rate for phytoplankton (/s)
        self.phy_qmort = phy_qmort * self.d2s  # Quadratic mortality rate for phytoplankton (/s)
        self.phy_CN = 122.0 / 16.0  # Carbon-to-nitrogen ratio for phytoplankton
        self.phy_CO = 122.0 / 172.0  # Carbon-to-Doxygen ratio for phytoplankton
        self.phy_minchlc = phy_minchlc  # Minimum chlorophyll-to-carbon ratio
        self.phy_optchlc = phy_optchlc  # Optimal chlorophyll-to-carbon ratio
        self.phy_optFeC = phy_optFeC  # Optimal iron-to-carbon ratio
        self.phy_maxFeC = phy_maxFeC  # Maximum iron-to-carbon ratio
        self.zoo_grz = zoo_grz * self.d2s  # Maximum grazing rate for zooplankton (/s)
        self.zoo_epsmin = zoo_epsmin * self.d2s  # Minimum grazing efficiency for zooplankton
        self.zoo_epsmax = zoo_epsmax * self.d2s  # Maximum grazing efficiency for zooplankton
        self.zoo_epsrat = zoo_epsrat  # Rate at which epsilon transitions from min to max values
        self.zoo_inerti = zoo_inerti  # Number of previous days for moving average of the prey field
        self.zoo_prefphy = 1.00  # Zooplankton preference for phytoplankton
        self.zoo_prefdet = zoo_prefdet  # Zooplankton preference for detritus
        self.zoo_prefaoa = 1.0  # Zooplankton preference for AOA
        self.zoo_prefnob = 1.0  # Zooplankton preference for NOB
        self.zoo_qmort = zoo_qmort * self.d2s  # Quadratic mortality rate for zooplankton (/s)
        self.zoo_respi = zoo_respi * self.d2s  # Respiration rate for zooplankton (/s)
        self.zoo_assim = zoo_assim  # Zooplankton assimilation efficiency
        self.zoo_excre = zoo_excre  # Zooplankton excretion efficiency
        self.zoo_kzoo = 0.25  # Half-saturation constant for zooplankton linear mortality
        self.detrem = detrem * self.d2s  # Remineralization rate for detritus (/s)
        self.w0 = w0 * self.d2s  # Sinking velocity of detritus (m/day)
        self.biomin = 1e-5  # Minimum biomass concentration (µM)
        self.dfemin = 0.0001  # Minimum dFe concentration (µM)
        self.aoa_aT = aoa_aT  # Maximum growth rate scaling for Ammonium Oxidizing Archaea
        self.nob_aT = nob_aT  # Maximum growth rate scaling for Nitrite Oxidizing Bacteria
        self.chem_bT = auto_bT  # Temperature dependence for chemautotrophs
        self.aoa_kn = aoa_kn  # Half-saturation constant for Ammonium for Ammonium Oxidizing Archaea
        self.nob_kn = nob_kn  # Half-saturation constant for Nitrite for Nitrite Oxidizing Bacteria
        self.aoa_pO2 = aoa_pO2  # diffusive oxygen coefficient for Ammonium for Ammonium Oxidizing Archaea
        self.nob_pO2 = nob_pO2  # diffusive oxygen coefficient for Nitrite for Nitrite Oxidizing Bacteria
        self.aoa_ynh4 = aoa_ynh4  # Growth yield on NH4 for Ammonium Oxidizing Archaea
        self.nob_yno2 = nob_yno2  # Growth yeild on NO2 for Nitrite Oxidizing Bacteria
        self.aoa_yo2 = aoa_yo2  # Growth yield on O2 for Ammonium Oxidizing Archaea
        self.nob_yo2 = nob_yo2  # Growth yeild on O2 for Nitrite Oxidizing Bacteria
        self.aoa_Fe2C = aoa_Fe2C  # Fe:C ratio for Ammonium Oxidizing Archaea
        self.nob_Fe2C = nob_Fe2C  # Fe:C ratio for Nitrite Oxidizing Bacteria
        self.aoa_lmort = aoa_lmort * self.d2s  # Linear mortality rate for AOA (/s)
        self.nob_lmort = nob_lmort * self.d2s  # Linear mortality rate for NOB (/s)
        self.aoa_qmort = aoa_qmort * self.d2s  # Quadratic mortality rate for AOA (/s)
        self.nob_qmort = nob_qmort * self.d2s  # Quadratic mortality rate for NOB (/s)
        
        
        
