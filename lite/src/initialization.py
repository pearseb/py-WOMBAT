import numpy as np
import xarray as xr

def initialize_tracers(lat, lon, grid):
    """
    Initialize tracer concentrations in the water column.

    Returns:
        dict: A dictionary containing initial tracer concentrations:
            - `no3` (np.ndarray): Nitrate concentration (µmol N/L).
            - `dfe` (np.ndarray): Dissolved iron concentration (µmol Fe/L).
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
    data.close()
    
    sno3 = no3.sel(lon=lon, lat=lat, depth=0, method='nearest')
    dno3 = no3.sel(lon=lon, lat=lat, depth=-grid.zgrid[-1], method='nearest')
    salk = alk.sel(lon=lon, lat=lat, depth_surface=0, method='nearest')
    dalk = alk.sel(lon=lon, lat=lat, depth_surface=-grid.zgrid[-1], method='nearest')
    sdic = dic.sel(lon=lon, lat=lat, depth_surface=0, method='nearest')
    ddic = dic.sel(lon=lon, lat=lat, depth_surface=-grid.zgrid[-1], method='nearest')
    
    # Boundary conditions for tracers
    no3 = np.linspace(sno3, dno3, grid.npt)               # Nitrate (µM)
    dfe = np.linspace(0.3 / 1000, 0.6 / 1000, grid.npt) # Dissolved iron (nM to µM)
    phy = np.linspace(0.01, 0.0, grid.npt)              # Phytoplankton (µM C)
    zoo = np.linspace(0.01, 0.0, grid.npt)             # Zooplankton (µM C)
    det = np.linspace(0.01, 0.0, grid.npt)             # Detritus (µM C)
    pchl = np.linspace(0.01 * 0.025, 0.0, grid.npt)     # Chlorophyll (mg Chl/m³)
    phyfe = phy * (7e-6 * np.ones(grid.npt))
    zoofe = zoo * (7e-6 * np.ones(grid.npt))
    detfe = det * (7e-6 * np.ones(grid.npt))
    dic = np.linspace(sdic, ddic, grid.npt)               # DIC (µM)
    alk = np.linspace(salk, dalk, grid.npt)               # Alk (µM)
    
    tracers = {
        "no3": np.repeat(no3[:, np.newaxis], 2, axis=1),
        "dfe": np.repeat(dfe[:, np.newaxis], 2, axis=1),
        "phy": np.repeat(phy[:, np.newaxis], 2, axis=1),
        "zoo": np.repeat(zoo[:, np.newaxis], 2, axis=1),
        "det": np.repeat(det[:, np.newaxis], 2, axis=1),
        "pchl": np.repeat(pchl[:, np.newaxis], 2, axis=1),
        "phyfe": np.repeat(phyfe[:, np.newaxis], 2, axis=1),
        "zoofe": np.repeat(zoofe[:, np.newaxis], 2, axis=1),
        "detfe": np.repeat(detfe[:, np.newaxis], 2, axis=1),
        "dic": np.repeat(dic[:, np.newaxis], 2, axis=1),
        "alk": np.repeat(alk[:, np.newaxis], 2, axis=1),
        }
    
    return tracers

