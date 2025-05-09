
import os 
os.chdir("/Users/buc146/OneDrive - CSIRO/pyWOMBAT/lite")

import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ffmpeg

from src.parameters import p_Grid, p_Diff, p_Advec, p_BGC
from src.initialization import initialize_tracers
from src.mixing import advection_diffusion, mix_mld, restore
from src.mld_timeseries import get_mld_timeseries
from src.rsds_timeseries import get_rsds_timeseries
from src.tas_timeseries import get_tas_timeseries
from src.sfcWnd_timeseries import get_sfcWnd_timeseries
from src.sinking import compute_sinking, compute_sink_rate
from src.light import compute_light_profile
from src.bgc import compute_light_limit, compute_nutrient_limit, compute_primary_production, \
                    compute_chlorophyll_growth_rate, compute_iron_uptake, compute_grazing, \
                    compute_losses, compute_iron_chemistry, compute_co2_flux, compute_sourcessinks
from src.plot import plot1D

def main(lon, lat, atm_co2,
         zoo_qmort, zoo_assim, zoo_excre, zoo_grz, grazform, detrem):
    """
    Main function to run the biogeochemical model.
    """
    
    print(f"Running simulation with longitude={lon}, latitude={lat}, and atmospheric CO2={atm_co2}")
    print(f"BGC parameters zoo_qmort={zoo_qmort}, zoo_assim={zoo_assim}, zoo_excre={zoo_excre}, grazform={grazform}, and detrem={detrem}")

    # Initialize the grid
    bot = 1300.0
    dz = 5.0
    Grid = p_Grid(bot, dz)
    
    # Initialise the mixing and BGC parameters
    Diff = p_Diff(Grid.npt)
    Advec = p_Advec(Grid.npt)
    BGC = p_BGC(zoo_qmort=zoo_qmort, zoo_assim=zoo_assim, zoo_excre=zoo_excre, zoo_grz=zoo_grz, grazform=grazform, detrem=detrem)
    
    # Intialize timestepping
    dt = 86400.0 / 12.0  # Time step (seconds)
    days = 10
    total_steps = int(days * 86400 / dt)  # Total number of time steps
    plot_freq = int(86400.0 / dt)
    
    # Initalize the tracers
    tracers = initialize_tracers(lat, lon, Grid)
    no3_loc = tracers['no3']
    dfe_loc = tracers['dfe']
    phy_loc = tracers['phy']
    zoo_loc = tracers['zoo']
    det_loc = tracers['det']
    pchl_loc = tracers['pchl']
    phyfe_loc = tracers['phyfe']
    zoofe_loc = tracers['zoofe']
    detfe_loc = tracers['detfe']
    dic_loc = tracers['dic']
    alk_loc = tracers['alk']
    # Diagnostic tracer 
    days_to_integrate = 30.0
    Nrepeats = int(86400.0 / dt * days_to_integrate / 2.0)
    mphy_loc = np.repeat(phy_loc, Nrepeats, axis=1)
    mdet_loc = np.repeat(det_loc, Nrepeats, axis=1)
    # deep tracer values
    ino3 = no3_loc[:,0]
    idfe = dfe_loc[:,0]
    idic = dic_loc[:,0]
    ialk = alk_loc[:,0]
    
    # Mixed layer depth
    print("Getting MLDs")
    mld_timeseries = get_mld_timeseries(lat, lon, dt)
    #mld_timeseries = mld_timeseries*0 + 50.0  ### constant MLD
    tmld = 1.0 / (86400.0 / 6.0)   # Mixing timescale (1/s)  (every 4 hours)
    trest = 1.0 / (86400.0 * 10.0) # Restoring timescale (1/s)  (every 15 days)
    # Photosynthetic active radiation at surface
    print("Getting PAR")
    par_timeseries = get_rsds_timeseries(lat, lon, dt)
    # Temperature at surface
    print("Getting Temperature")
    tas_timeseries = get_tas_timeseries(lat, lon, dt)
    print("Getting Surface Windspeed")
    sfcWnd_timeseries = get_sfcWnd_timeseries(lat, lon, dt)
    # get the chlorophyll-dependent attenuation coefficients for RGB PAR
    p_Chl_k = np.genfromtxt("inputs/rgb_attenuation_coefs.txt", delimiter="\t", skip_header=1)
    
    # initialise output of key tracers and diagnostics
    no3_output = np.zeros((Grid.npt, total_steps // plot_freq))
    dfe_output = np.zeros((Grid.npt, total_steps // plot_freq))
    phy_output = np.zeros((Grid.npt, total_steps // plot_freq))
    pchl_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zoo_output = np.zeros((Grid.npt, total_steps // plot_freq))
    det_output = np.zeros((Grid.npt, total_steps // plot_freq))
    dic_output = np.zeros((Grid.npt, total_steps // plot_freq))
    alk_output = np.zeros((Grid.npt, total_steps // plot_freq))
    phymu_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zoomu_output = np.zeros((Grid.npt, total_steps // plot_freq))
    fco2_output = np.zeros((total_steps // plot_freq))
    pco2_output = np.zeros((total_steps // plot_freq))
    cexp_output = np.zeros((Grid.npt, total_steps // plot_freq))
    
    # Loop through time steps
    for step in range(total_steps):
        print(f"Running time step {step + 1} of {total_steps}")
        
        ts_within_year = int(step % (365 * 86400 / dt))
        par = np.fmax(par_timeseries[ts_within_year], 1e-10)
        mld = mld_timeseries[ts_within_year]
        tc = np.linspace(tas_timeseries[ts_within_year], 1.0, len(Grid.zgrid))
        tck = tc + 273.15
        wnd = sfcWnd_timeseries[ts_within_year]
        
        # Step 0: Update the tracer array (old becomes new)
        no3_loc[:,0] = np.fmax(no3_loc[:,1], BGC.biomin)
        dfe_loc[:,0] = np.fmax(dfe_loc[:,1], BGC.dfemin)
        phy_loc[:,0] = np.fmax(phy_loc[:,1], BGC.biomin)
        phyfe_loc[:,0] = np.fmax(phyfe_loc[:,1], BGC.biomin*1e-6)
        zoo_loc[:,0] = np.fmax(zoo_loc[:,1], BGC.biomin)
        zoofe_loc[:,0] = np.fmax(zoofe_loc[:,1], BGC.biomin*1e-6)
        det_loc[:,0] = np.fmax(det_loc[:,1], BGC.biomin)
        detfe_loc[:,0] = np.fmax(detfe_loc[:,1], BGC.biomin*1e-6)
        pchl_loc[:,0] = np.fmax(pchl_loc[:,1], BGC.biomin * BGC.phy_minchlc)
        dic_loc[:,0] = np.fmax(dic_loc[:,1], BGC.biomin)
        alk_loc[:,0] = np.fmax(alk_loc[:,1], BGC.biomin)
        # update diagnostics by shifting down and then updating with latest values
        mphy_loc[:,0:-1] = mphy_loc[:,1::]
        mdet_loc[:,0:-1] = mdet_loc[:,1::]
        mphy_loc[:,-1] = phy_loc[:,1]
        mdet_loc[:,-1] = det_loc[:,1]
        
        # Step 1: Compute advection and diffusion
        no3_loc = advection_diffusion(dt, no3_loc, ino3[-1], Grid, Advec, Diff)
        dfe_loc = advection_diffusion(dt, dfe_loc, idfe[-1], Grid, Advec, Diff)
        phy_loc = advection_diffusion(dt, phy_loc, 0.0, Grid, Advec, Diff)
        zoo_loc = advection_diffusion(dt, zoo_loc, 0.0, Grid, Advec, Diff)
        det_loc = advection_diffusion(dt, det_loc, 0.0, Grid, Advec, Diff)
        pchl_loc = advection_diffusion(dt, pchl_loc, 0.0, Grid, Advec, Diff)
        phyfe_loc = advection_diffusion(dt, phyfe_loc, 0.0, Grid, Advec, Diff)
        zoofe_loc = advection_diffusion(dt, zoofe_loc, 0.0, Grid, Advec, Diff)
        detfe_loc = advection_diffusion(dt, detfe_loc, 0.0, Grid, Advec, Diff)
        dic_loc = advection_diffusion(dt, dic_loc, idic[-1], Grid, Advec, Diff)
        alk_loc = advection_diffusion(dt, alk_loc, ialk[-1], Grid, Advec, Diff)
        
        # Step 2: Compute mixed layer depth mixing (entrainment and detrainment when MLD is variable)
        no3_loc = mix_mld(dt, no3_loc, mld, tmld, Grid)
        dfe_loc = mix_mld(dt, dfe_loc, mld, tmld, Grid)
        phy_loc = mix_mld(dt, phy_loc, mld, tmld, Grid)
        zoo_loc = mix_mld(dt, zoo_loc, mld, tmld, Grid)
        det_loc = mix_mld(dt, det_loc, mld, tmld, Grid)
        pchl_loc = mix_mld(dt, pchl_loc, mld, tmld, Grid)
        phyfe_loc = mix_mld(dt, phyfe_loc, mld, tmld, Grid)
        zoofe_loc = mix_mld(dt, zoofe_loc, mld, tmld, Grid)
        detfe_loc = mix_mld(dt, detfe_loc, mld, tmld, Grid)
        dic_loc = mix_mld(dt, dic_loc, mld, tmld, Grid)
        alk_loc = mix_mld(dt, alk_loc, mld, tmld, Grid)
        
        # Step 2.5: Assume some horizontal mixing by restoring towards initial values
        no3_loc = restore(dt, no3_loc, ino3, trest, Grid)
        dfe_loc = restore(dt, dfe_loc, idfe, trest, Grid)
        alk_loc = restore(dt, alk_loc, ialk, trest, Grid)
        dic_loc = restore(dt, dic_loc, idic, trest, Grid)
        
        # Step 3: Sink tracers
        wsink = compute_sink_rate(phy_loc[0,1], BGC)
        det_loc, det2sed = compute_sinking(dt, det_loc, wsink, Grid)
        detfe_loc, detfe2sed = compute_sinking(dt, detfe_loc, wsink, Grid)
        
        # Step 4: Compute ecosystem cycling
        light_profile = compute_light_profile(
            tracers["pchl"], 
            p_Chl_k, 
            par,
            mld, 
            Grid, 
            BGC)
        
        light_limit = compute_light_limit(
            light_profile['par_tot'], 
            phy_loc, 
            pchl_loc, 
            tc, 
            BGC)
        
        nutrient_limit = compute_nutrient_limit(
            phy_loc, 
            phyfe_loc, 
            no3_loc, 
            dfe_loc, 
            light_limit['chlc_ratio'], 
            tc, 
            BGC)
        
        primary_production = compute_primary_production(
            tc, 
            light_limit['phy_limpar'], 
            nutrient_limit['phy_limnut'], 
            BGC)
        
        chlorophyll_growth_rate = compute_chlorophyll_growth_rate(
            light_limit['phy_pisl'], 
            primary_production['phy_mumax'], 
            nutrient_limit['phy_limnut'], 
            light_profile['par_tot_mld'], 
            primary_production['phy_mu'], 
            phy_loc, 
            BGC)
        
        iron_uptake = compute_iron_uptake(
            phy_loc, 
            phyfe_loc, 
            dfe_loc, 
            primary_production['phy_mumax'], 
            nutrient_limit['phy_limdfe'], 
            nutrient_limit['phy_k_dfe'], 
            BGC)
        
        grazing = compute_grazing(
            tc, 
            phy_loc, 
            det_loc, 
            zoo_loc, 
            np.mean(mphy_loc,axis=1), 
            np.mean(mdet_loc,axis=1), 
            1, 
            BGC)
        
        losses = compute_losses(
            tc, 
            phy_loc, 
            zoo_loc, 
            det_loc, 
            BGC)
        
        iron_chemistry = compute_iron_chemistry(
            tck, 
            dfe_loc, 
            det_loc, 
            mld, 
            Grid)
        
        co2_flux = compute_co2_flux(
            dic_loc[0,1],
            alk_loc[0,1],
            tc[0],
            wnd,
            atm_co2)
        
        sourcessinks = compute_sourcessinks(
            primary_production['phy_mu'], losses['phy_lmort'], losses['phy_qmort'], iron_uptake['phy_dfeupt'],
            losses['zoo_zoores'], losses['zoo_qmort'], grazing['zoo_phygrz'], grazing['zoo_detgrz'], losses['det_remin'],
            phy_loc, phyfe_loc, zoo_loc, zoofe_loc, det_loc, detfe_loc, 
            iron_chemistry['dfe_prec'], iron_chemistry['dfe_scav'], iron_chemistry['dfe_coag'],
            chlorophyll_growth_rate['chl_mu'], light_limit['chlc_ratio'], 
            co2_flux['co2_flux'] / Grid.dz,
            BGC)
        
        # Step 5: Update tracer concentrations based on sources and sinks
        no3_loc[:,1] += sourcessinks["ddt_no3"] * dt
        dfe_loc[:,1] += sourcessinks["ddt_dfe"] * dt
        phy_loc[:,1] += sourcessinks["ddt_phy"] * dt
        phyfe_loc[:,1] += sourcessinks["ddt_phyfe"] * dt
        zoo_loc[:,1] += sourcessinks["ddt_zoo"] * dt
        zoofe_loc[:,1] += sourcessinks["ddt_zoofe"] * dt
        det_loc[:,1] += sourcessinks["ddt_det"] * dt
        detfe_loc[:,1] += sourcessinks["ddt_detfe"] * dt
        pchl_loc[:,1] += sourcessinks["ddt_pchl"] * dt
        dic_loc[:,1] += sourcessinks["ddt_dic"] * dt
        alk_loc[:,1] += sourcessinks["ddt_alk"] * dt
        
        if (step % plot_freq) == 0:
            ## Step 6: Plot the output and save as a figure
            #fig = plot1D(no3_loc[:,1], dfe_loc[:,1], \
            #             phy_loc[:,1], zoo_loc[:,1], \
            #             det_loc[:,1], \
            #             primary_production['phy_mu'],  grazing['zoo_mu'], \
            #             light_limit['chlc_ratio'], \
            #             phyfe_loc[:,1]/phy_loc[:,1]*1e6, zoofe_loc[:,1]/zoo_loc[:,1]*1e6, detfe_loc[:,1]/det_loc[:,1]*1e6, \
            #             phy_loc[:,1]*0+16.0/122.0, \
            #             nutrient_limit['phy_limnit'], nutrient_limit['phy_limdfe'], dic_loc[:,1], Grid)
            #fig.savefig("figures/plot_lite_day_{0:08d}".format(int(step*dt/86400)))
            #plt.clf()
            #del fig
    
            # Step 7: Write output
            no3_output[:,step//plot_freq] = no3_loc[:,1]
            dfe_output[:,step//plot_freq] = dfe_loc[:,1]
            phy_output[:,step//plot_freq] = phy_loc[:,1]
            pchl_output[:,step//plot_freq] = pchl_loc[:,1]
            zoo_output[:,step//plot_freq] = zoo_loc[:,1]
            det_output[:,step//plot_freq] = det_loc[:,1]
            dic_output[:,step//plot_freq] = dic_loc[:,1]
            alk_output[:,step//plot_freq] = alk_loc[:,1]
            phymu_output[:,step//plot_freq] = primary_production['phy_mu']
            zoomu_output[:,step//plot_freq] = grazing['zoo_mu']
            fco2_output[step//plot_freq] = co2_flux['co2_flux']
            pco2_output[step//plot_freq] = co2_flux['pCO2_water']
            cexp_output[:,step//plot_freq] = det_loc[:,1] * wsink  # mmol/m2/s
            
    print("Simulation complete!")

    if lat < 0:
        latt = "%iS"%(np.abs(lat))
    else:
        latt = "%iN"%(lat)
    
    ## Step 8: Make video and delete figures
    #if os.path.isfile("movies/lite_%idays_%s_grazform%i.mp4"%(days, latt, BGC.grazform)):
    #    os.remove("movies/lite_%idays_%s_grazform%i.mp4"%(days, latt, BGC.grazform))
    #    
    #(
    #     ffmpeg
    #     .input('figures/plot_lite_day_%08d.png', framerate=24)
    #     .output("movies/lite_%idays_%s_grazform%i.mp4"%(days, latt, BGC.grazform), vcodec='libx264', pix_fmt='yuv420p')
    #     .run(overwrite_output=True)
    #)
    ## remove image files
    #for ii in np.arange(days):
    #    os.remove('figures/plot_lite_day_{0:08d}.png'.format(ii))
    #
    #print("Animation made!")
    
    start_date = np.datetime64("1993-01-01")  # Define start date
    time = start_date + np.arange(0, total_steps // plot_freq).astype("timedelta64[D]")  # Time in datetime64
    ds = xr.Dataset(
        {
            "no3": (["depth", "time"], no3_output),  # Store data
            "dfe": (["depth", "time"], dfe_output),  # Store data
            "phy": (["depth", "time"], phy_output),  # Store data
            "zoo": (["depth", "time"], zoo_output),  # Store data
            "det": (["depth", "time"], det_output),  # Store data
            "pchl": (["depth", "time"], pchl_output),  # Store data
            "dic": (["depth", "time"], dic_output),  # Store data
            "alk": (["depth", "time"], alk_output),  # Store data
            "phymu": (["depth", "time"], phymu_output),  # Store data
            "zoomu": (["depth", "time"], zoomu_output),  # Store data
            "fco2": (["time"], fco2_output),  # Store data
            "pco2": (["time"], pco2_output),  # Store data
            "cexp": (["depth", "time"], cexp_output),  # Store data
        },
        coords={
            "time": time,
            "depth": Grid.zgrid
        }
    )
    
    filename = (
        f"output/lite_{days}days_{latt}_co2{atm_co2}_"
        f"zooqmort{BGC.zoo_qmort * 86400.0:.2f}_"
        f"zooassim{BGC.zoo_assim:.2f}_"
        f"zooexcre{BGC.zoo_excre:.2f}_"
        f"zoogrz{BGC.zoo_grz * 86400.0:.2f}_"
        f"grazform{BGC.grazform}_"
        f"detrem{BGC.detrem * 86400.0:.2f}.nc"
    )
    
    if os.path.isfile(filename):
        os.remove(filename)
    
    ds.to_netcdf(filename)
    print("Output saved to disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pyWOMBAT-lite with different parameters")
    parser.add_argument("--lon", type=float, default=230, help="Longitude of simulation")
    parser.add_argument("--lat", type=float, default=-50, help="Latitude of simulation")
    parser.add_argument("--atm_co2", type=float, default=400.0, help="Atmospheric CO2 level")
    parser.add_argument("--zoo_qmort", type=float, default=0.35, help="Zooplankton quadratic mortality closure term")
    parser.add_argument("--zoo_assim", type=float, default=0.60, help="Zooplankton assimilation efficiency")
    parser.add_argument("--zoo_excre", type=float, default=0.80, help="Zooplankton excretion fraction")
    parser.add_argument("--zoo_grz", type=float, default=3.0, help="Maximum grazing rate for zooplankton")
    parser.add_argument("--grazform", type=int, default=1, help="Grazing formulation type (1, 2, or 3)")
    parser.add_argument("--detrem", type=float, default=0.40, help="Detrital quadratic remineralisation term")

    args = parser.parse_args()
    main(args.lat, args.lon, args.atm_co2, 
         args.zoo_qmort, args.zoo_assim, args.zoo_excre, args.zoo_grz, args.grazform, args.detrem)
    
