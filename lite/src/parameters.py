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


class p_Advec:
    """Defines parameters related to advection."""

    def __init__(self, npt):
        # Precompute w within the constructor
        self.w = 50.0 / 365.0 / 86400.0 * np.ones(npt)  # Vertical advection (m/s)


class p_BGC:
    """Defines parameters related to the biogeochemical model."""

    def __init__(self, zoo_qmort=0.35, zoo_assim=0.6, zoo_excr=0.8, zoo_grz=3.0, grazform=1, detrem=0.40):
        self.d2s = 1.0 / 86400.0  # Conversion factor: days to seconds
        self.alpha = 2.25  # PI curve slope for photosynthesis
        self.PAR_bio = 0.43  # Fraction of PAR available biologically
        self.auto_aT = 1.0  # Maximum growth rate scaling for autotrophs
        self.auto_bT = 1.050  # Temperature dependence for autotrophs
        self.hete_aT = 1.0  # Maximum growth rate scaling for heterotrophs
        self.hete_bT = 1.066  # Temperature dependence for heterotrophs
        self.phy_kn = 2.0  # Half-saturation constant for NO3 uptake
        self.phy_kf = 1.2e-3  # Half-saturation constant for dFe uptake
        self.phy_biothresh = 0.6  # Biomass threshold for phytoplankton processes
        self.phy_lmort = 0.005 * self.d2s  # Linear mortality rate for phytoplankton (/s)
        self.phy_qmort = 0.05 * self.d2s  # Quadratic mortality rate for phytoplankton (/s)
        self.phy_CN = 122.0 / 16.0  # Carbon-to-nitrogen ratio for phytoplankton
        self.phy_minchlc = 0.004  # Minimum chlorophyll-to-carbon ratio
        self.phy_optchlc = 0.036  # Optimal chlorophyll-to-carbon ratio
        self.phy_optFeC = 10e-6  # Optimal iron-to-carbon ratio
        self.phy_maxFeC = 50e-6  # Maximum iron-to-carbon ratio
        self.zoo_grz = zoo_grz * self.d2s  # Maximum grazing rate for zooplankton (/s)
        self.zoo_epsmin = 0.025 * self.d2s  # Minimum grazing efficiency for zooplankton
        self.zoo_epsmax = 0.25 * self.d2s  # Maximum grazing efficiency for zooplankton
        self.zoo_prefphy = 1.00  # Zooplankton preference for phytoplankton
        self.zoo_prefdet = 0.50  # Zooplankton preference for detritus
        self.zoo_qmort = zoo_qmort * self.d2s  # Quadratic mortality rate for zooplankton (/s)
        self.zoo_respi = 0.003 * self.d2s  # Respiration rate for zooplankton (/s)
        self.zoo_assim = zoo_assim  # Zooplankton assimilation efficiency
        self.zoo_excre = zoo_excr  # Zooplankton excretion efficiency
        self.zoo_kzoo = 0.25  # Half-saturation constant for zooplankton linear mortality
        self.grazform = grazform  # Grazing formulation type (1, 2, or 3)
        self.detrem = detrem * self.d2s  # Remineralization rate for detritus (/s)
        self.w0 = 25.0 * self.d2s  # Sinking velocity of detritus (m/day)
        self.biomin = 1e-5  # Minimum biomass concentration (µM)
        self.dfemin = 0.0001  # Minimum dFe concentration (µM)
        
