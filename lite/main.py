
import os 
import sys
from pathlib import Path
import logging

# Get the path of the current script (assuming it's inside the "lite" subdirectory)
BASE_DIR = Path(__file__).resolve().parent

# Define input and output directories relative to BASE_DIR
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = "/g/data/es60/pjb581/py-WOMBAT/output"

import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#import ffmpeg

from src.parameters import p_Grid, p_Diff, p_BGC
from src.initialization import initialize_tracers
from src.mixing import advection_diffusion, mix_mld, restore
from src.mld_timeseries import get_mld_timeseries
from src.w_timeseries import get_w_timeseries
from src.rsds_timeseries import get_rsds_timeseries
from src.tas_timeseries import get_tas_timeseries
from src.sfcWnd_timeseries import get_sfcWnd_timeseries
from src.sinking import compute_sinking, compute_sink_rate
from src.light import compute_light_profile
from src.bgc import compute_light_limit, compute_nutrient_limit, compute_primary_production, \
                    compute_chlorophyll_growth_rate, compute_iron_uptake, compute_grazing, \
                    compute_losses, compute_iron_chemistry, compute_co2_flux, compute_nitrification, \
                    compute_totalN, compute_sourcessinks
from src.plot import plot1D

def main(expnum, year, days, lon, lat, atm_co2):
    """
    Main function to run the biogeochemical model.
    """
    
    # Initialize the grid
    bot = 1300.0
    dz = 5.0
    Grid = p_Grid(bot, dz)
    
    # Intialize timestepping
    dt = 86400.0 / 12.0  # Time step (seconds)
    total_steps = int(days * 86400 / dt)  # Total number of time steps
    plot_freq = int(86400.0 / dt)

    # Initialise the diffusive mixing and BGC parameters
    Diff = p_Diff(Grid.npt)
    BGC = p_BGC()
    
    # Initalize the tracers
    tracers = initialize_tracers(lat, lon, Grid)
    no3_loc = tracers['no3']
    nh4_loc = tracers['nh4']
    no2_loc = tracers['no2']
    dfe_loc = tracers['dfe']
    aoa_loc = tracers['aoa']
    nob_loc = tracers['nob']
    phy_loc = tracers['phy']
    zoo_loc = tracers['zoo']
    det_loc = tracers['det']
    pchl_loc = tracers['pchl']
    phyfe_loc = tracers['phyfe']
    zoofe_loc = tracers['zoofe']
    detfe_loc = tracers['detfe']
    dic_loc = tracers['dic']
    alk_loc = tracers['alk']
    o2_loc = tracers['o2']
    # Diagnostic tracer 
    Nrepeats = int(86400.0 / dt * BGC.zoo_inerti / 2.0)
    mphy_loc = np.repeat(phy_loc, Nrepeats, axis=1)
    mdet_loc = np.repeat(det_loc, Nrepeats, axis=1)
    maoa_loc = np.repeat(aoa_loc, Nrepeats, axis=1)
    mnob_loc = np.repeat(nob_loc, Nrepeats, axis=1)
    # initial tracer values through water column
    ino3 = no3_loc[:,0]
    idfe = dfe_loc[:,0]
    idic = dic_loc[:,0]
    ialk = alk_loc[:,0]
    io2 = o2_loc[:,0]
    
    # Mixed layer depth
    logging.info("Loading condition data (T, W, MLDs, PAR, WindSpeed) at year %i"%(year))
    mld_timeseries = get_mld_timeseries(year, lat, lon, dt)
    logging.info("Loaded MLD")
    par_timeseries = get_rsds_timeseries(year, lat, lon, dt)
    logging.info("Loaded PAR")
    tas_timeseries = get_tas_timeseries(year, lat, lon, dt)
    logging.info("Loaded T")
    sfcWnd_timeseries = get_sfcWnd_timeseries(year, lat, lon, dt)
    logging.info("Loaded Wind")
    w_timeseries = get_w_timeseries(year, lat, lon, dt, bot)
    logging.info("Loaded W")
    # get the chlorophyll-dependent attenuation coefficients for RGB PAR
    p_Chl_k = np.genfromtxt("inputs/rgb_attenuation_coefs.txt", delimiter="\t", skip_header=1)

    # Mixing timescales
    tmld = 1.0 / (86400.0 / 6.0)   # Mixing timescale (1/s)  (every 4 hours)
    trest = 1.0 / (86400.0 * 15.0) # Restoring timescale (1/s)  (every 15 days)
    
    # initialise output of key tracers and diagnostics
    no3_output = np.zeros((Grid.npt, total_steps // plot_freq))
    nh4_output = np.zeros((Grid.npt, total_steps // plot_freq))
    no2_output = np.zeros((Grid.npt, total_steps // plot_freq))
    dfe_output = np.zeros((Grid.npt, total_steps // plot_freq))
    aoa_output = np.zeros((Grid.npt, total_steps // plot_freq))
    nob_output = np.zeros((Grid.npt, total_steps // plot_freq))
    phy_output = np.zeros((Grid.npt, total_steps // plot_freq))
    pchl_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zoo_output = np.zeros((Grid.npt, total_steps // plot_freq))
    det_output = np.zeros((Grid.npt, total_steps // plot_freq))
    dic_output = np.zeros((Grid.npt, total_steps // plot_freq))
    alk_output = np.zeros((Grid.npt, total_steps // plot_freq))
    o2_output = np.zeros((Grid.npt, total_steps // plot_freq))
    phymu_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zoomu_output = np.zeros((Grid.npt, total_steps // plot_freq))
    aoamu_output = np.zeros((Grid.npt, total_steps // plot_freq))
    nobmu_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zoogrzphy_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zoogrzdet_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zoogrzaoa_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zoogrznob_output = np.zeros((Grid.npt, total_steps // plot_freq))
    zooepsilon_output = np.zeros((Grid.npt, total_steps // plot_freq))
    aoaox_output = np.zeros((Grid.npt, total_steps // plot_freq))
    nobox_output = np.zeros((Grid.npt, total_steps // plot_freq))
    fco2_output = np.zeros((total_steps // plot_freq))
    pco2_output = np.zeros((total_steps // plot_freq))
    cexp_output = np.zeros((Grid.npt, total_steps // plot_freq))
    
    # Loop through time steps
    for step in range(total_steps):

        if (step%100==0):
            logging.info(f"Running time step {step + 1} of {total_steps}")
        
        # Get the conditions for the current timestep
        ts_within_year = int(step % (365 * 86400 / dt))
        par = np.fmax(par_timeseries[ts_within_year], 1e-10)
        mld = mld_timeseries[ts_within_year]
        w = w_timeseries[ts_within_year]
        tc = np.linspace(tas_timeseries[ts_within_year], 1.0, len(Grid.zgrid))
        tck = tc + 273.15
        wnd = sfcWnd_timeseries[ts_within_year]
    
        # Step 0: Update the tracer array (old becomes new)
        no3_loc[:,0] = np.fmax(no3_loc[:,1], BGC.biomin)
        nh4_loc[:,0] = np.fmax(nh4_loc[:,1], BGC.biomin)
        no2_loc[:,0] = np.fmax(no2_loc[:,1], BGC.biomin)
        dfe_loc[:,0] = np.fmax(dfe_loc[:,1], BGC.dfemin)
        aoa_loc[:,0] = np.fmax(aoa_loc[:,1], BGC.biomin)
        nob_loc[:,0] = np.fmax(nob_loc[:,1], BGC.biomin)
        phy_loc[:,0] = np.fmax(phy_loc[:,1], BGC.biomin)
        phyfe_loc[:,0] = np.fmax(phyfe_loc[:,1], BGC.biomin*1e-6)
        zoo_loc[:,0] = np.fmax(zoo_loc[:,1], BGC.biomin)
        zoofe_loc[:,0] = np.fmax(zoofe_loc[:,1], BGC.biomin*1e-6)
        det_loc[:,0] = np.fmax(det_loc[:,1], BGC.biomin)
        detfe_loc[:,0] = np.fmax(detfe_loc[:,1], BGC.biomin*1e-6)
        pchl_loc[:,0] = np.fmax(pchl_loc[:,1], BGC.biomin * BGC.phy_minchlc)
        dic_loc[:,0] = np.fmax(dic_loc[:,1], BGC.biomin)
        alk_loc[:,0] = np.fmax(alk_loc[:,1], BGC.biomin)
        o2_loc[:,0] = np.fmax(o2_loc[:,1], BGC.biomin)
        # update diagnostics by shifting down and then updating with latest values
        mphy_loc[:,0:-1] = mphy_loc[:,1::]
        mdet_loc[:,0:-1] = mdet_loc[:,1::]
        maoa_loc[:,0:-1] = maoa_loc[:,1::]
        mnob_loc[:,0:-1] = mnob_loc[:,1::]
        mphy_loc[:,-1] = phy_loc[:,1]
        mdet_loc[:,-1] = det_loc[:,1]
        maoa_loc[:,-1] = aoa_loc[:,1]
        mnob_loc[:,-1] = nob_loc[:,1]
        
        # Step 1: Compute advection and diffusion
        #  set w as a minimum upwelling rate
        #logging.info("Doing advection diffusion")
        w = np.fmax(w, 1.0/86400.0/365.0) * np.ones(int(Grid.npt))
        no3_loc = advection_diffusion(dt, no3_loc, ino3[-1], Grid, w, Diff)
        nh4_loc = advection_diffusion(dt, nh4_loc, 0.0, Grid, w, Diff)
        no2_loc = advection_diffusion(dt, no2_loc, 0.0, Grid, w, Diff)
        dfe_loc = advection_diffusion(dt, dfe_loc, idfe[-1], Grid, w, Diff)
        aoa_loc = advection_diffusion(dt, aoa_loc, 0.0, Grid, w, Diff)
        nob_loc = advection_diffusion(dt, nob_loc, 0.0, Grid, w, Diff)
        phy_loc = advection_diffusion(dt, phy_loc, 0.0, Grid, w, Diff)
        zoo_loc = advection_diffusion(dt, zoo_loc, 0.0, Grid, w, Diff)
        det_loc = advection_diffusion(dt, det_loc, 0.0, Grid, w, Diff)
        pchl_loc = advection_diffusion(dt, pchl_loc, 0.0, Grid, w, Diff)
        phyfe_loc = advection_diffusion(dt, phyfe_loc, 0.0, Grid, w, Diff)
        zoofe_loc = advection_diffusion(dt, zoofe_loc, 0.0, Grid, w, Diff)
        detfe_loc = advection_diffusion(dt, detfe_loc, 0.0, Grid, w, Diff)
        dic_loc = advection_diffusion(dt, dic_loc, idic[-1], Grid, w, Diff)
        alk_loc = advection_diffusion(dt, alk_loc, ialk[-1], Grid, w, Diff)
        o2_loc = advection_diffusion(dt, o2_loc, io2[-1], Grid, w, Diff)
        
        # Step 2: Compute mixed layer depth mixing (entrainment and detrainment when MLD is variable)
        #logging.info("Doing MLD mixing")
        no3_loc = mix_mld(dt, no3_loc, mld, tmld, Grid)
        nh4_loc = mix_mld(dt, nh4_loc, mld, tmld, Grid)
        no2_loc = mix_mld(dt, no2_loc, mld, tmld, Grid)
        dfe_loc = mix_mld(dt, dfe_loc, mld, tmld, Grid)
        aoa_loc = mix_mld(dt, aoa_loc, mld, tmld, Grid)
        nob_loc = mix_mld(dt, nob_loc, mld, tmld, Grid)
        phy_loc = mix_mld(dt, phy_loc, mld, tmld, Grid)
        zoo_loc = mix_mld(dt, zoo_loc, mld, tmld, Grid)
        det_loc = mix_mld(dt, det_loc, mld, tmld, Grid)
        pchl_loc = mix_mld(dt, pchl_loc, mld, tmld, Grid)
        phyfe_loc = mix_mld(dt, phyfe_loc, mld, tmld, Grid)
        zoofe_loc = mix_mld(dt, zoofe_loc, mld, tmld, Grid)
        detfe_loc = mix_mld(dt, detfe_loc, mld, tmld, Grid)
        dic_loc = mix_mld(dt, dic_loc, mld, tmld, Grid)
        alk_loc = mix_mld(dt, alk_loc, mld, tmld, Grid)
        o2_loc = mix_mld(dt, o2_loc, mld, tmld, Grid)
        
        # Step 2.5: Assume some horizontal mixing by restoring towards initial values
        #logging.info("Doing restoring")
        no3_loc = restore(dt, no3_loc, ino3, trest, Grid)
        dfe_loc = restore(dt, dfe_loc, idfe, trest, Grid)
        alk_loc = restore(dt, alk_loc, ialk, trest, Grid)
        dic_loc = restore(dt, dic_loc, idic, trest, Grid)
        o2_loc = restore(dt, o2_loc, io2, trest, Grid)
        o2_loc[0] = io2[0]*1  # reset surface O2 always
        
        # Step 3: Sink tracers
        #logging.info("Doing sinking")
        wsink = compute_sink_rate(phy_loc[0,1], BGC)
        det_loc, det2sed = compute_sinking(dt, det_loc, wsink, Grid)
        detfe_loc, detfe2sed = compute_sinking(dt, detfe_loc, wsink, Grid)
        
        # Step 4: Compute ecosystem cycling
        #logging.info("Doing light profile")
        light_profile = compute_light_profile(
            tracers["pchl"], 
            p_Chl_k, 
            par,
            mld, 
            Grid, 
            BGC)

        #logging.info("Doing light limitation")
        light_limit = compute_light_limit(
            light_profile['par_tot'], 
            phy_loc, 
            pchl_loc, 
            tc, 
            BGC)
        
        #logging.info("Doing nutrient limitation")
        nutrient_limit = compute_nutrient_limit(
            phy_loc, 
            phyfe_loc, 
            no3_loc, 
            nh4_loc, 
            dfe_loc, 
            light_limit['chlc_ratio'], 
            tc, 
            BGC)
        
        #logging.info("Doing primary production")
        primary_production = compute_primary_production(
            tc, 
            light_limit['phy_limpar'], 
            nutrient_limit['phy_limnut'], 
            BGC)
        
        #logging.info("Doing chlorophyll growth")
        chlorophyll_growth_rate = compute_chlorophyll_growth_rate(
            light_limit['phy_pisl'], 
            primary_production['phy_mumax'], 
            nutrient_limit['phy_limnut'], 
            light_profile['par_tot_mld'], 
            primary_production['phy_mu'], 
            phy_loc, 
            BGC)
        
        #logging.info("Doing iron uptake")
        iron_uptake = compute_iron_uptake(
            phy_loc, 
            phyfe_loc, 
            dfe_loc, 
            primary_production['phy_mumax'], 
            nutrient_limit['phy_limdfe'], 
            nutrient_limit['phy_k_dfe'], 
            BGC)
        
        #logging.info("Doing grazing")
        grazing = compute_grazing(
            tc,
            aoa_loc, 
            nob_loc,
            phy_loc, 
            det_loc, 
            zoo_loc, 
            np.mean(mphy_loc,axis=1), 
            np.mean(mdet_loc,axis=1), 
            np.mean(maoa_loc,axis=1), 
            np.mean(mnob_loc,axis=1), 
            BGC)
        
        #logging.info("Doing mortality")
        losses = compute_losses(
            tc, 
            aoa_loc, 
            nob_loc,
            phy_loc, 
            zoo_loc, 
            det_loc, 
            BGC)
        
        #logging.info("Doing iron chemistry")
        iron_chemistry = compute_iron_chemistry(
            tck, 
            dfe_loc, 
            det_loc, 
            mld, 
            Grid)
        
        #logging.info("Doing co2 fluxes")
        co2_flux = compute_co2_flux(
            dic_loc[0,1],
            alk_loc[0,1],
            tc[0],
            wnd,
            atm_co2)

        #logging.info("Doing nitrification")
        nitrif = compute_nitrification(
            tc,
            nh4_loc,
            no2_loc,
            o2_loc, 
            BGC)
        
        #logging.info("Doing sources and sinks")
        sourcessinks = compute_sourcessinks(
            primary_production['phy_mu'], losses['phy_lmort'], losses['phy_qmort'], iron_uptake['phy_dfeupt'], nutrient_limit['phy_limnh4'], nutrient_limit['phy_limno3'],
            losses['zoo_zoores'], losses['zoo_qmort'], grazing['zoo_grzphy'], grazing['zoo_grzdet'], grazing['zoo_grznob'], grazing['zoo_grznob'], losses['det_remin'],
            nitrif['aoa_mu'], nitrif['nob_mu'], losses['aoa_lmort'], losses['nob_lmort'], losses['aoa_qmort'], losses['nob_qmort'], aoa_loc, nob_loc,
            phy_loc, phyfe_loc, zoo_loc, zoofe_loc, det_loc, detfe_loc, 
            iron_chemistry['dfe_prec'], iron_chemistry['dfe_scav'], iron_chemistry['dfe_coag'],
            chlorophyll_growth_rate['chl_mu'], light_limit['chlc_ratio'], 
            co2_flux['co2_flux'] / Grid.dz,
            BGC)
        
        totalN1 = compute_totalN(aoa_loc, nob_loc, phy_loc, zoo_loc, det_loc, nh4_loc, no2_loc, no3_loc, BGC)

        #logging.info("Updating arrays")
        # Step 5: Update tracer concentrations based on sources and sinks
        no3_loc[:,1] += sourcessinks["ddt_no3"] * dt
        nh4_loc[:,1] += sourcessinks["ddt_nh4"] * dt
        no2_loc[:,1] += sourcessinks["ddt_no2"] * dt
        dfe_loc[:,1] += sourcessinks["ddt_dfe"] * dt
        aoa_loc[:,1] += sourcessinks["ddt_aoa"] * dt
        nob_loc[:,1] += sourcessinks["ddt_nob"] * dt
        phy_loc[:,1] += sourcessinks["ddt_phy"] * dt
        phyfe_loc[:,1] += sourcessinks["ddt_phyfe"] * dt
        zoo_loc[:,1] += sourcessinks["ddt_zoo"] * dt
        zoofe_loc[:,1] += sourcessinks["ddt_zoofe"] * dt
        det_loc[:,1] += sourcessinks["ddt_det"] * dt
        detfe_loc[:,1] += sourcessinks["ddt_detfe"] * dt
        pchl_loc[:,1] += sourcessinks["ddt_pchl"] * dt
        dic_loc[:,1] += sourcessinks["ddt_dic"] * dt
        alk_loc[:,1] += sourcessinks["ddt_alk"] * dt
        o2_loc[:,1] += sourcessinks["ddt_o2"] * dt

        totalN2 = compute_totalN(aoa_loc, nob_loc, phy_loc, zoo_loc, det_loc, nh4_loc, no2_loc, no3_loc, BGC)
        
        # Check for conservation of mass by ecosystem component
        if not (np.allclose(totalN1, totalN2, atol=1e-12)):
            logging.info("Not conserving nitrogen")
            logging.info("totalN1 = %s"%(np.array2string(totalN1, precision=16, separator=", ", suppress_small=True)))
            logging.info("totalN2 = %s"%(np.array2string(totalN2, precision=16, separator=", ", suppress_small=True)))
            logging.info(" ")
            logging.info("ddt_no3 * dt = %s, "%(np.array2string(sourcessinks["ddt_no3"] * dt, precision=16, separator=", ", suppress_small=True)))
            logging.info("ddt_no2 * dt = %s, "%(np.array2string(sourcessinks["ddt_no2"] * dt, precision=16, separator=", ", suppress_small=True)))
            logging.info("ddt_nh4 * dt = %s, "%(np.array2string(sourcessinks["ddt_nh4"] * dt, precision=16, separator=", ", suppress_small=True)))
            logging.info("ddt_phy * dt = %s, "%(np.array2string(sourcessinks["ddt_phy"] * dt, precision=16, separator=", ", suppress_small=True)))
            logging.info("ddt_zoo * dt = %s, "%(np.array2string(sourcessinks["ddt_zoo"] * dt, precision=16, separator=", ", suppress_small=True)))
            logging.info("ddt_det * dt = %s, "%(np.array2string(sourcessinks["ddt_det"] * dt, precision=16, separator=", ", suppress_small=True)))
            logging.info("ddt_aoa * dt = %s, "%(np.array2string(sourcessinks["ddt_aoa"] * dt, precision=16, separator=", ", suppress_small=True)))
            logging.info("ddt_nob * dt = %s, "%(np.array2string(sourcessinks["ddt_nob"] * dt, precision=16, separator=", ", suppress_small=True)))
            print("Not conserving nitrogen... exiting simulation")
            sys.exit()

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
            #logging.info("Writing output")
            no3_output[:,step//plot_freq] = no3_loc[:,1]
            nh4_output[:,step//plot_freq] = nh4_loc[:,1]
            no2_output[:,step//plot_freq] = no2_loc[:,1]
            dfe_output[:,step//plot_freq] = dfe_loc[:,1]
            aoa_output[:,step//plot_freq] = aoa_loc[:,1]
            nob_output[:,step//plot_freq] = nob_loc[:,1]
            phy_output[:,step//plot_freq] = phy_loc[:,1]
            pchl_output[:,step//plot_freq] = pchl_loc[:,1]
            zoo_output[:,step//plot_freq] = zoo_loc[:,1]
            det_output[:,step//plot_freq] = det_loc[:,1]
            dic_output[:,step//plot_freq] = dic_loc[:,1]
            alk_output[:,step//plot_freq] = alk_loc[:,1]
            o2_output[:,step//plot_freq] = o2_loc[:,1]
            phymu_output[:,step//plot_freq] = primary_production['phy_mu']
            zoomu_output[:,step//plot_freq] = grazing['zoo_mu']
            aoamu_output[:,step//plot_freq] = nitrif['aoa_mu']
            nobmu_output[:,step//plot_freq] = nitrif['nob_mu']
            zoogrzphy_output[:,step//plot_freq] = grazing['zoo_grzphy']
            zoogrzdet_output[:,step//plot_freq] = grazing['zoo_grzdet']
            zoogrzaoa_output[:,step//plot_freq] = grazing['zoo_grzaoa']
            zoogrznob_output[:,step//plot_freq] = grazing['zoo_grznob']
            zooepsilon_output[:,step//plot_freq] = grazing['zoo_epsilon']
            aoaox_output[:,step//plot_freq] = aoa_loc[:,1] * nitrif['aoa_mu'] * 1.0/BGC.aoa_ynh4
            nobox_output[:,step//plot_freq] = nob_loc[:,1] * nitrif['nob_mu'] * 1.0/BGC.nob_yno2
            fco2_output[step//plot_freq] = co2_flux['co2_flux']
            pco2_output[step//plot_freq] = co2_flux['pCO2_water']
            cexp_output[:,step//plot_freq] = det_loc[:,1] * wsink  # mmol/m2/s
            
    logging.info("Simulation complete!")

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
    #logging.info("Animation made!")
    
    start_date = np.datetime64("1900-01-01")  # Define start date
    time = start_date + np.arange(0, total_steps // plot_freq).astype("timedelta64[D]")  # Time in datetime64
    ds = xr.Dataset(
        {
            "no3": (["depth", "time"], no3_output),  # Store data
            "nh4": (["depth", "time"], nh4_output),  # Store data
            "no2": (["depth", "time"], no2_output),  # Store data
            "dfe": (["depth", "time"], dfe_output),  # Store data
            "aoa": (["depth", "time"], aoa_output),  # Store data
            "nob": (["depth", "time"], nob_output),  # Store data
            "phy": (["depth", "time"], phy_output),  # Store data
            "zoo": (["depth", "time"], zoo_output),  # Store data
            "det": (["depth", "time"], det_output),  # Store data
            "pchl": (["depth", "time"], pchl_output),  # Store data
            "dic": (["depth", "time"], dic_output),  # Store data
            "alk": (["depth", "time"], alk_output),  # Store data
            "o2": (["depth", "time"], o2_output),  # Store data
            "phymu": (["depth", "time"], phymu_output),  # Store data
            "zoomu": (["depth", "time"], zoomu_output),  # Store data
            "aoamu": (["depth", "time"], aoamu_output),  # Store data
            "nobmu": (["depth", "time"], nobmu_output),  # Store data
            "zoogrzphy": (["depth", "time"], zoogrzphy_output),  # Store data
            "zoogrzdet": (["depth", "time"], zoogrzdet_output),  # Store data
            "zoogrzaoa": (["depth", "time"], zoogrzaoa_output),  # Store data
            "zoogrznob": (["depth", "time"], zoogrznob_output),  # Store data
            "zooepsilon": (["depth", "time"], zooepsilon_output),  # Store data
            "aoaox": (["depth", "time"], aoaox_output),  # Store data
            "nobox": (["depth", "time"], nobox_output),  # Store data
            "fco2": (["time"], fco2_output),  # Store data
            "pco2": (["time"], pco2_output),  # Store data
            "cexp": (["depth", "time"], cexp_output),  # Store data
        },
        coords={
            "time": time.astype("datetime64[ns]"),
            "depth": Grid.zgrid
        }
    )
    filename = (
        f"{OUTPUT_DIR}/lite_{year}_{days}days_{latt}_{atm_co2}ppm_exp{expnum}.nc"
    )
    logging.info("Saving to "+filename)
    if os.path.isfile(filename):
        os.remove(filename)
    ds.to_netcdf(filename)
    logging.info("Output saved to disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pyWOMBAT-lite with different parameters")
    parser.add_argument("--expnum", type=int, default=0, help="Experiment number")
    parser.add_argument("--year", type=int, default=365, help="Repeat year forcing")
    parser.add_argument("--days", type=float, default=365, help="Number of days to run simulation")
    parser.add_argument("--lon", type=float, default=230, help="Longitude of simulation")
    parser.add_argument("--lat", type=float, default=-50, help="Latitude of simulation")
    parser.add_argument("--atm_co2", type=float, default=400.0, help="Atmospheric CO2 level")

    args = parser.parse_args()
    main(args.expnum, args.year, args.days, args.lon, args.lat, args.atm_co2)
    

