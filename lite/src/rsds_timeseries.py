#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:54:54 2025

@author: buc146
"""

import numpy as np
import xarray as xr

def get_rsds_timeseries(latitude, longitude, dt):
    """
    Computes and returns the interpolated incident radiation at the surface (RSDS) timeseries.
    
    Parameters:
    latitude (float): Latitude for spatial selection.
    longitude (float): Longitude for spatial selection.
    dt (float): Time step in minutes.
    
    Returns:
    np.ndarray: Interpolated RSDS timeseries.
    """
    
    start = '1993-01-01T00:00:00'
    end = '1993-12-31T23:59:59'
    min_per_ts = dt / 60.0 
    times = np.arange(np.datetime64(start, 'ns'), np.datetime64(end, 'ns'), np.timedelta64(int(min_per_ts), 'm')).astype('datetime64[ns]')
    
    # Load dataset
    data = xr.open_dataset('inputs/rsds_1993.nc')
    rsds = data['rsds']
    data.close()
    
    # Select nearest grid point
    rsds = rsds.sel(lat=latitude, lon=longitude, method='nearest')
    
    # Wrap timeseries data for interpolation
    rsds1 = rsds.isel(time=-1).assign_coords(time=np.datetime64("1992-12-31T22:30:00", 'ns'))
    rsds2 = rsds.isel(time=0).assign_coords(time=np.datetime64("1994-01-01T01:30:00", 'ns'))
    rsds_ = xr.concat([rsds1, rsds, rsds2], dim='time')
    
    # Interpolate values based on number of timesteps
    rsds__ = rsds_.interp(time=times)
    rsds_timeseries = rsds__.values
    
    return rsds_timeseries
