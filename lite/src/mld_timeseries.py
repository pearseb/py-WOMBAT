#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:45:46 2025

@author: buc146
"""

import numpy as np
import xarray as xr

def get_mld_timeseries(latitude, longitude, dt):
    """
    Computes and returns the interpolated mixed layer depth (MLD) timeseries.
    
    Parameters:
    latitude (float): Latitude for spatial selection.
    longitude (float): Longitude for spatial selection.
    dt (float): Time step in minutes.
    
    Returns:
    np.ndarray: Interpolated MLD timeseries.
    """
    
    start = '1993-01-01T00:00:00'
    end = '1993-12-31T23:59:59'
    min_per_ts = dt / 60.0 
    times = np.arange(np.datetime64(start, 'ns'), np.datetime64(end, 'ns'), np.timedelta64(int(min_per_ts),'m')).astype('datetime64[ns]')
    
    # Load dataset
    data = xr.open_mfdataset('/Users/buc146/OneDrive - CSIRO/pyWOMBAT/lite/inputs/ocean_mld_1993_*.nc')
    mld = data['mld']
    data.close()
    
    # Select nearest grid point
    mld = mld.sel(yt_ocean=latitude, xt_ocean=longitude, method='nearest')
    
    # Wrap timeseries data for interpolation
    mld1 = mld.isel(Time=-1).assign_coords(Time=np.datetime64("1992-12-31T22:30:00", 'ns'))
    mld2 = mld.isel(Time=0).assign_coords(Time=np.datetime64("1994-01-01T01:30:00", 'ns'))
    mld_ = xr.concat([mld1, mld, mld2], dim='Time')
    
    # Interpolate values based on number of timesteps
    mld__ = mld_.interp(Time=times)
    mld_timeseries = mld__.values
    
    return mld_timeseries