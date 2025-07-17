#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:45:46 2025

@author: buc146
"""

import numpy as np
import xarray as xr
import logging

def get_w_timeseries(yyyy, latitude, longitude, dt, zbot):
    """
    Computes and returns the interpolated vertical velocity (w) timeseries.
    
    Parameters:
    yyyy (integer): Year of simulation to extract conditions
    latitude (float): Latitude for spatial selection.
    longitude (float): Longitude for spatial selection.
    dt (float): Time step in minutes.
    bot (float): Time step in minutes.
    
    Returns:
    np.ndarray: Interpolated MLD timeseries.
    """

    yyyb = yyyy - 1
    yyya = yyyy + 1
    
    start = '%i-01-01T00:00:00'%(yyyy)
    end = '%i-12-31T23:59:59'%(yyyy)
    min_per_ts = dt / 60.0 
    times = np.arange(np.datetime64(start, 'ns'), np.datetime64(end, 'ns'), np.timedelta64(int(min_per_ts),'m')).astype('datetime64[ns]')
    
    # Load dataset
    fnames = '/g/data/gb6/BRAN/BRAN2020/month/ocean_w_mth_%i_*.nc'%(yyyy)
    data = xr.open_mfdataset(fnames, chunks={"Time":-1})
    w = data['w']
    logging.info("Loaded W, now selecting the right point")
    data.close()
    
    # Select nearest grid point
    w = w.sel(yt_ocean=latitude, xt_ocean=longitude, method='nearest').compute()
    logging.info("Loaded W, got the right point")
    
    # take the average in depth with minimum upwelling rate of 1 m/year
    w = w.sel(sw_ocean=slice(0,zbot)).mean(dim='sw_ocean')
    logging.info("Loaded W, found mean velocity through water column")
    
    # Wrap timeseries data for interpolation
    w1 = w.isel(Time=-1).assign_coords(Time=np.datetime64("%i-12-31T22:30:00"%(yyyb), 'ns'))
    w2 = w.isel(Time=0).assign_coords(Time=np.datetime64("%i-01-01T01:30:00"%(yyya), 'ns'))
    w_ = xr.concat([w1, w, w2], dim='Time')
    
    # Interpolate values based on number of timesteps
    w__ = w_.interp(Time=times)
    w_timeseries = w__.values
    
    return w_timeseries