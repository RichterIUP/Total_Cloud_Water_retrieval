#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:50:50 2020

@author: philipp
"""

import os
import sys
import inp
sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
import aux2 as aux
import numpy as np
import netCDF4 as nc
import datetime as dt
import create_nc as cnc
sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "/src/ftsreader"))
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator, interp1d
from ftsreader import ftsreader

def read_era5_argmin(fname, lat, lon, tim, key, time_unit):
    '''
    Read an ERA5 dataset and return the desired quantity. No interpolation will be performed, instead the value nearest to the given
    temporal and spatial position will be returned. The array will be orderd in a way, that the pressure is decreasing with increasing
    array index.  
    '''
    tim = (tim - dt.datetime(time_unit[0], 1, 1)).total_seconds()/time_unit[1]
    with nc.Dataset(fname, "r") as f:
        try:
            plev = f.variables['level'][:]
            plev_units = f.variables['level'].units
            latitude = f.variables['latitude'][:]
            longitude = f.variables['longitude'][:]
        except KeyError:
            plev = f.variables['plev'][:]
            plev_units = f.variables['plev'].units
            latitude = f.variables['lat'][:]
            longitude = f.variables['lon'][:]
        time = f.variables['time'][:]
        return_val = f.variables[key][:]
    idx_lat = np.abs(latitude - lat).argmin()
    idx_lon = np.abs(longitude - lon).argmin()
    idx_time = np.abs(time - tim).argmin()
    try:
        profile = return_val[idx_time, :, idx_lat, idx_lon]
    except IndexError:
        profile = return_val[0, idx_time, :, idx_lat, idx_lon]
        
    if plev[0] < plev[-1]:
        profile = profile[::-1]
        plev = plev[::-1]
    
    ## Pressure unit must be hPa
    if plev_units == "Pa":
        units = 1e-2
    elif plev_units == "hPa" or plev_units == "millibars" or plev_units == "mb":
        units = 1
    return profile, idx_lat, idx_lon, idx_time, plev*units

def read_input(opus_file, atmospheric_file, cloudnet_file, sza):
    
    aux.DATETIME = dt.datetime.strptime(opus_file.split("_")[0].split("nyem")[1], "%Y%m%d%H%M%S")

    ## Read radiances and wavenumber from OPUS file
    s = ftsreader.ftsreader(opus_file, verbose=True, getspc=True, getifg=True)
    aux.WAVENUMBER_FTIR = np.array(s.spcwvn)
    aux.RADIANCE_FTIR = np.array(s.spc)*1e3

    if inp.STDDEV > 0.0:
        noise = inp.STDDEV * np.ones(aux.RADIANCE_FTIR.size)
    else:
        noise = np.zeros(aux.RADIANCE_FTIR.size)
    num_windows = len(aux.MICROWINDOWS)
    wavenumber_av = np.zeros(num_windows)
    radiance_av   = np.zeros(num_windows)
    noise_av      = np.zeros(num_windows)
    for window in range(num_windows):
        idx_window = np.where((aux.WAVENUMBER_FTIR >= aux.MICROWINDOWS[window][0]) & (aux.WAVENUMBER_FTIR <= aux.MICROWINDOWS[window][1]))
        radiance_av[window]   = np.mean(aux.RADIANCE_FTIR[idx_window])
        noise_av[window]      = np.std(aux.RADIANCE_FTIR[idx_window])
        wavenumber_av[window] = np.mean(aux.WAVENUMBER_FTIR[idx_window])
        
    idx_noise = np.where((aux.WAVENUMBER_FTIR > 1925) & (aux.WAVENUMBER_FTIR < 2000))[0]
    func = lambda x, a, b: a * x + b
    popt, pcov = curve_fit(func, aux.WAVENUMBER_FTIR[idx_noise], aux.RADIANCE_FTIR[idx_noise])
    noise = aux.RADIANCE_FTIR[idx_noise] - func(aux.WAVENUMBER_FTIR[idx_noise], popt[0], popt[1]) 

    aux.NOISE_FTIR      = np.std(noise)*np.ones(wavenumber_av.size)
    aux.WAVENUMBER_FTIR = wavenumber_av
    aux.RADIANCE_FTIR   = radiance_av+inp.OFFSET
    cnc.LAT = inp.LAT
    cnc.LON = inp.LON
    aux.SOLAR_ZENITH_ANGLE = sza
    
    ## If cloud height file exists, read cloud heights. Otherwise use data from inp.py
    if os.path.exists(cloudnet_file):
        aux.CLOUD_LAYERS = np.loadtxt(cloudnet_file)
    else:
        aux.CLOUD_LAYERS = inp.CLOUD_LAYERS

    if len(aux.CLOUD_LAYERS) == 0:
        raise Exception("No Cloud!")
    
    ## Read atmospheric profile
    
    ## Set the time according to ERA5 averaging
    time_era5 = dt.datetime(aux.DATETIME.year, aux.DATETIME.month, aux.DATETIME.day, aux.DATETIME.hour)
    if aux.DATETIME.minute != 0:
        time_era5 = time_era5 + dt.timedelta(hours = 1)
    g0 = 9.80665#ms-2 
    relative_humidity = read_era5_argmin(atmospheric_file, inp.LAT, inp.LON, time_era5, inp.KEY_RH, inp.TIME_UNIT)[0]
    height = read_era5_argmin(atmospheric_file, inp.LAT, inp.LON, time_era5, inp.KEY_HEIGHT, inp.TIME_UNIT)[0]/g0 #Convert geopotential to geometric height
    temperature, idx_lat, idx_lon, idx_time, plev = read_era5_argmin(atmospheric_file, inp.LAT, inp.LON, time_era5, inp.KEY_TEMP, inp.TIME_UNIT)

        
    ## Temperature difference between two layers must be less then 10 K. 
    ## If temperature difference is too large, insert an additional layer. 
    idx = np.array([-1])
    while (idx.size != 0 or height.size == 69):
        height_prof_prev = height[:]
        delta_temp = np.abs(np.diff(temperature))
        idx = np.where(delta_temp > 10)[0]
        if idx.size > 1:
            idx = idx[0]
        height = np.insert(height, idx+1, (height[idx]+height[idx+1])/2)
        temperature = np.interp(height, height_prof_prev, temperature)
        relative_humidity = np.interp(height, height_prof_prev, relative_humidity)
        plev = np.interp(height, height_prof_prev, plev)

    temp_f = interp1d(height, temperature, kind='linear', fill_value='extrapolate')
    rh_f = interp1d(height, relative_humidity, kind='linear', fill_value='extrapolate')
    plev_f = interp1d(height, plev, kind='linear', fill_value='extrapolate')
    
    ## Omit layers with negative height. Those might occure after interpolation
    idx_positive = np.where(height > 0.0)[0]
    height = height[idx_positive]
    relative_humidity = relative_humidity[idx_positive]
    temperature = temperature[idx_positive]
    plev = plev[idx_positive]

    aux.ATMOSPHERIC_GRID = {'pressure(hPa)' : plev, \
                        'altitude(m)': height, \
                        'altitude(km)': height*1e-3, \
                        'temperature(K)': temperature, \
                        'humidity(%)': relative_humidity}

    return True
