import numpy as np
import xarray as xr

def initialize_tracers(lat, lon, grid):
    """
    Initialize tracer concentrations in the water column.

    Returns:
        dict: A dictionary containing initial tracer concentrations:
            - `no3` (np.ndarray): Nitrate concentration (µmol N/L).
            - `nh4` (np.ndarray): Ammonium concentration (µmol N/L).
            - `no2` (np.ndarray): Nitrite concentration (µmol N/L).
            - `dfe` (np.ndarray): Dissolved iron concentration (µmol Fe/L).
            - `o2`  (np.ndarray): Dissolved oxygen concentration (µmol O2/L).
            - `n2`  (np.ndarray): Dissolved oxygen concentration (µmol N2/L).
            - `aoa` (np.ndarray): AOA concentration (µmol C/L).
            - `nob` (np.ndarray): NOB concentration (µmol C/L).
            - `aox` (np.ndarray): Anammox Bacteria concentration (µmol C/L).
            - `phy` (np.ndarray): Phytoplankton concentration (µmol C/L).
            - `zoo` (np.ndarray): Zooplankton concentration (µmol C/L).
            - `det` (np.ndarray): Detritus concentration (µmol C/L).
            - `pchl` (np.ndarray): Chlorophyll concentration (µgChl/L).
            - `phyfe` (np.ndarray): Phytoplankton Fe concentration (µmol Fe/L).
            - `zoofe` (np.ndarray): Zooplankton Fe concentration (µmol Fe/L).
            - `detfe` (np.ndarray): Detritus Fe concentration (µmol Fe/L).
    """
    
    # Load boundary conditions from GLODAP and WOA2023
    data = xr.open_dataset('inputs/woa23_all_n00_01.nc', decode_times=False)
    no3 = data['n_an'].squeeze()
    data = xr.open_dataset('inputs/GLODAPv2.2016b.TAlk.nc', decode_times=False)
    dep = data['Depth']
    alk = data['TAlk'].squeeze().assign_coords({"depth_surface":dep})
    data = xr.open_dataset('inputs/GLODAPv2.2016b.TCO2.nc', decode_times=False)
    dic = data['TCO2'].squeeze().assign_coords({"depth_surface":dep})
    data = xr.open_dataset('inputs/woa23_all_o00_01.nc', decode_times=False)
    o2 = data['o_an'].squeeze()
    data.close()
    
    sno3 = no3.sel(lon=lon, lat=lat, depth=0, method='nearest')
    dno3 = no3.sel(lon=lon, lat=lat, depth=-grid.zgrid[-1], method='nearest')
    salk = alk.sel(lon=lon, lat=lat, depth_surface=0, method='nearest')
    dalk = alk.sel(lon=lon, lat=lat, depth_surface=-grid.zgrid[-1], method='nearest')
    sdic = dic.sel(lon=lon, lat=lat, depth_surface=0, method='nearest')
    ddic = dic.sel(lon=lon, lat=lat, depth_surface=-grid.zgrid[-1], method='nearest')
    so2 = o2.sel(lon=lon, lat=lat, depth=0, method='nearest')
    do2 = o2.sel(lon=lon, lat=lat, depth=-grid.zgrid[-1], method='nearest')
    allo2 = o2.sel(lon=lon, lat=lat, method='nearest')
    allo2 = allo2.interp(depth=("depth", -grid.zgrid))
    allo2 = allo2.fillna(so2)
    allno3 = no3.sel(lon=lon, lat=lat, method='nearest')
    allno3 = allno3.interp(depth=("depth", -grid.zgrid))
    allno3 = allno3.fillna(sno3)
    
    # Boundary conditions for tracers
    #no3 = np.linspace(sno3, dno3, grid.npt)               # Nitrate (µM)
    dfe = np.linspace(0.3 / 1000, 0.6 / 1000, grid.npt) # Dissolved iron (nM to µM)
    aoa = np.linspace(0.01, 0.0, grid.npt)              # AOA (µM C)
    nob = np.linspace(0.01, 0.0, grid.npt)              # NOB (µM C)
    aox = np.linspace(0.01, 0.0, grid.npt)              # Anammox Bacteria (µM C)
    phy = np.linspace(0.01, 0.0, grid.npt)              # Phytoplankton (µM C)
    zoo = np.linspace(0.01, 0.0, grid.npt)             # Zooplankton (µM C)
    det = np.linspace(0.01, 0.0, grid.npt)             # Detritus (µM C)
    pchl = np.linspace(0.01 * 0.025, 0.0, grid.npt)     # Chlorophyll (mg Chl/m³)
    phyfe = phy * (7e-6 * np.ones(grid.npt))
    zoofe = zoo * (7e-6 * np.ones(grid.npt))
    detfe = det * (7e-6 * np.ones(grid.npt))
    dic = np.linspace(sdic, ddic, grid.npt)               # DIC (µM)
    alk = np.linspace(salk, dalk, grid.npt)               # Alk (µM)
    o2 = allo2.values               # Oxygen (µM)
    no3 = allno3.values               # Oxygen (µM)
    nh4 = np.linspace(0.01, 0.001, grid.npt)              # NH4 (µM N)
    no2 = np.linspace(0.01, 0.001, grid.npt)              # NO2 (µM N)
    n2 = np.linspace(0.01, 0.001, grid.npt)               # N2 (µM N)

    tracers = {
        "no3": np.repeat(no3[:, np.newaxis], 2, axis=1),
        "nh4": np.repeat(nh4[:, np.newaxis], 2, axis=1),
        "no2": np.repeat(no2[:, np.newaxis], 2, axis=1),
        "dfe": np.repeat(dfe[:, np.newaxis], 2, axis=1),
        "aoa": np.repeat(aoa[:, np.newaxis], 2, axis=1),
        "nob": np.repeat(nob[:, np.newaxis], 2, axis=1),
        "aox": np.repeat(aox[:, np.newaxis], 2, axis=1),
        "phy": np.repeat(phy[:, np.newaxis], 2, axis=1),
        "zoo": np.repeat(zoo[:, np.newaxis], 2, axis=1),
        "det": np.repeat(det[:, np.newaxis], 2, axis=1),
        "pchl": np.repeat(pchl[:, np.newaxis], 2, axis=1),
        "phyfe": np.repeat(phyfe[:, np.newaxis], 2, axis=1),
        "zoofe": np.repeat(zoofe[:, np.newaxis], 2, axis=1),
        "detfe": np.repeat(detfe[:, np.newaxis], 2, axis=1),
        "dic": np.repeat(dic[:, np.newaxis], 2, axis=1),
        "alk": np.repeat(alk[:, np.newaxis], 2, axis=1),
        "o2": np.repeat(o2[:, np.newaxis], 2, axis=1),
        "n2": np.repeat(n2[:, np.newaxis], 2, axis=1),
        }
    
    return tracers

