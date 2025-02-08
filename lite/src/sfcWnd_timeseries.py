#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:54:54 2025

@author: buc146
"""

import numpy as np
import xarray as xr

def get_sfcWnd_timeseries(yyyy, latitude, longitude, dt):
    """
    Computes and returns the interpolated surface windspeed timeseries.
    
    Parameters:
    yyyy (integer): Year of simulation to extract conditions
    latitude (float): Latitude for spatial selection.
    longitude (float): Longitude for spatial selection.
    dt (float): Time step in minutes.
    
    Returns:
    np.ndarray: Interpolated surface windspeed timeseries.
    """

    yyyb = yyyy - 1
    yyya = yyyy + 1
    
    start = '%i-01-01T00:00:00'%(yyyy)
    end = '%i-12-31T23:59:59'%(yyyy)
    min_per_ts = dt / 60.0 
    times = np.arange(np.datetime64(start, 'ns'), np.datetime64(end, 'ns'), np.timedelta64(int(min_per_ts), 'm')).astype('datetime64[ns]')
    
    # Load dataset
    data = xr.open_dataset('inputs/uas_%i.nc'%(yyyy))
    uas = data['uas']
    data = xr.open_dataset('inputs/vas_%i.nc'%(yyyy))
    vas = data['vas']
    data.close()
    
    # Select nearest grid point
    uas = uas.sel(lat=latitude, lon=longitude, method='nearest')
    vas = vas.sel(lat=latitude, lon=longitude, method='nearest')
    
    # Wrap timeseries data for interpolation
    uas1 = uas.isel(time=-1).assign_coords(time=np.datetime64("%i-12-31T22:30:00"%(yyyb), 'ns'))
    uas2 = uas.isel(time=0).assign_coords(time=np.datetime64("%i-01-01T01:30:00"%(yyya), 'ns'))
    uas_ = xr.concat([uas1, uas, uas2], dim='time')
    vas1 = vas.isel(time=-1).assign_coords(time=np.datetime64("%i-12-31T22:30:00"%(yyyb), 'ns'))
    vas2 = vas.isel(time=0).assign_coords(time=np.datetime64("%i-01-01T01:30:00"%(yyya), 'ns'))
    vas_ = xr.concat([vas1, vas, vas2], dim='time')
    
    # Interpolate values based on number of timesteps
    uas__ = uas_.interp(time=times)
    vas__ = vas_.interp(time=times)
    sfcWnd_timeseries = (uas__**2 + vas__**2)**0.5
    
    return sfcWnd_timeseries
