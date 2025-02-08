#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:54:54 2025

@author: buc146
"""

import numpy as np
import xarray as xr

def get_tas_timeseries(yyyy, latitude, longitude, dt):
    """
    Computes and returns the interpolated surface air temperature (TAS) timeseries.
    
    Parameters:
    yyyy (integer): Year of simulation to extract conditions
    latitude (float): Latitude for spatial selection.
    longitude (float): Longitude for spatial selection.
    dt (float): Time step in minutes.
    
    Returns:
    np.ndarray: Interpolated TAS timeseries.
    """

    yyyb = yyyy - 1
    yyya = yyyy + 1
    
    start = '%i-01-01T00:00:00'%(yyyy)
    end = '%i-12-31T23:59:59'%(yyyy)
    min_per_ts = dt / 60.0 
    times = np.arange(np.datetime64(start, 'ns'), np.datetime64(end, 'ns'), np.timedelta64(int(min_per_ts), 'm')).astype('datetime64[ns]')
    
    # Load dataset
    data = xr.open_dataset('inputs/tas_%i.nc'%(yyyy))
    tas = data['tas']
    data.close()
    
    # Select nearest grid point
    tas = tas.sel(lat=latitude, lon=longitude, method='nearest')
    
    # Wrap timeseries data for interpolation
    tas1 = tas.isel(time=-1).assign_coords(time=np.datetime64("%i-12-31T22:30:00"%(yyyb), 'ns'))
    tas2 = tas.isel(time=0).assign_coords(time=np.datetime64("%i-01-01T01:30:00"%(yyya), 'ns'))
    tas_ = xr.concat([tas1, tas, tas2], dim='time')
    
    # Interpolate values based on number of timesteps
    tas__ = tas_.interp(time=times)
    tas_timeseries = np.fmax(-1.8, tas__.values-273.15)
    tas_timeseries = np.fmin(30.0, tas_timeseries)
    
    return tas_timeseries
