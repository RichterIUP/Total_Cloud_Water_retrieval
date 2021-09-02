#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:59:16 2021

@author: philipp
"""

import os
import datetime as dt

import pandas as pd
import numpy as np
import netCDF4 as nc
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt

import cdsapi

G0 = 9.80665#ms-2

def read_era5_argmin(fname, lat, lon, tim, key, ftype):
    '''
    Read an ERA5 dataset and return the desired quantity. No interpolation will be performed,
    instead the value nearest to the given temporal and spatial position will be returned.
    The array will be orderd in a way, that the pressure is decreasing with increasing array index.

    Parameter
    ---------
    fname : str
        Name of ERA5 file containing atmospheric profile

    lat : float
        Latitudinal position

    lon : float
        Longitudinal position

    tim : dt.datetime
        Datetime object of measurement time

    key : str
        Key of the requested quantity

    time_unit : int
        Number of seconds in the time unit of ERA5 (seconds=1, hours=3600)

    Returns
    -------
    np.array
        Profile of requested quantity

    np.array
        Profile of air pressure
    '''
    tim = (tim - dt.datetime(1900, 1, 1)).total_seconds()/3600
    with nc.Dataset(fname, "r") as file_:
        if ftype =="profile":
            plev = file_.variables['level'][:]
            plev_units = file_.variables['level'].units
        latitude = file_.variables['latitude'][:]
        longitude = file_.variables['longitude'][:]
        time = file_.variables['time'][:]
        return_val = file_.variables[key][:]
    idx_lat = np.abs(latitude - lat).argmin()
    idx_lon = np.abs(longitude - lon).argmin()
    idx_time = np.abs(time - tim).argmin()
    if ftype == "profile":
        profile = return_val[idx_time, :, idx_lat, idx_lon]
    else:
        profile = return_val[idx_time, idx_lat, idx_lon]
    # Invert order of arrays, if pressure is increasing
    if ftype == "profile":
        if plev[0] < plev[-1]:
            profile = profile[::-1]
            plev = plev[::-1]
    
        ## Pressure unit must be hPa
        if plev_units == "Pa":
            units = 1e-2
        elif plev_units in ("hPa", "millibars", "mb"):
            units = 1
    else:
        plev = 0
        units = 0
    return profile, plev*units

class Atmosphere:
    def __init__(self, time, lat, lon, co2, co, ch4, n2o, o3, o2, z):
        self.__datetime = time
        self.__latitude = lat
        self.__longitude = lon
        self.__co2 = np.loadtxt(co2, delimiter=",")
        self.__n2o = np.loadtxt(n2o, delimiter=",")
        self.__o3  = np.loadtxt(o3, delimiter=",")
        self.__ch4 = np.loadtxt(ch4, delimiter=",")
        self.__co  = np.loadtxt(co, delimiter=",")
        self.__o2  = np.loadtxt(o2, delimiter=",")
        self.__z = np.loadtxt(z, delimiter=",")
    
    def download_atmosphere(self, fname):
        date = self.__datetime+ dt.timedelta(hours=1) if self.__datetime.minute > 30 else self.__datetime
        if not os.path.exists('{}'.format(fname)):
            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'geopotential', 'relative_humidity', 'temperature', 'specific_cloud_liquid_water_content', 'specific_cloud_ice_water_content'
            ],
            'year': "{:04d}".format(date.year),
            'month': "{:02d}".format(date.month),
            'day': "{:02d}".format(date.day), 
            'time': "{:02d}:{:02d}".format(date.hour, 0),
            'pressure_level': [
                '1', '2', '3',
                '5', '7', '10',
                '20', '30', '50',
                '70', '100', '125',
                '150', '175', '200',
                '225', '250', '300',
                '350', '400', '450',
                '500', '550', '600',
                '650', '700', '750',
                '775', '800', '825',
                '850', '875', '900',
                '925', '950', '975',
                '1000',],
            'area': [
                int(self.__latitude+1), int(self.__longitude-1), int(self.__latitude-1),
                int(self.__longitude+1),
            ],
            'format': 'netcdf',
        },
        '{}'.format(fname))
        else:
            print("File exists!")
        
    def download_cloud_position(self, fname):
        date = self.__datetime+ dt.timedelta(hours=1) if self.__datetime.minute > 30 else self.__datetime
        if not os.path.exists('{}'.format(fname)):
            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'cloud_base_height'
            ],
            'year': "{:04d}".format(date.year),
            'month': "{:02d}".format(date.month),
            'day': "{:02d}".format(date.day), 
            'time': "{:02d}:{:02d}".format(date.hour, 0),
            'area': [
                int(self.__latitude+1), int(self.__longitude-1), int(self.__latitude-1),
                int(self.__longitude+1),
            ],
            'format': 'netcdf',
        },
        '{}'.format(fname))
        else:
            print("File exists!")
            
    def create_atmosphere(self, file_):
        relative_humidity = read_era5_argmin(file_, \
                                             self.__latitude, \
                                             self.__longitude, \
                                             self.__datetime, 'r', 'profile')[0]
        height = read_era5_argmin(file_, \
                                  self.__latitude, \
                                  self.__longitude, \
                                  self.__datetime, 'z', 'profile')[0]/G0
        temperature, plev = read_era5_argmin(file_, \
                                             self.__latitude, \
                                             self.__longitude, \
                                             self.__datetime,  't', 'profile')
            
        ## Add a layer at height 18m
        humd_f = scipy.interpolate.interp1d(height, relative_humidity, fill_value="extrapolate", kind="cubic")
        temp_f = scipy.interpolate.interp1d(height, temperature, fill_value="extrapolate", kind="cubic")
        plev_f = scipy.interpolate.interp1d(height, plev, fill_value="extrapolate", kind="cubic")
        height = np.concatenate(([18], np.array(height)))

        temperature = temp_f(height)
        relative_humidity = humd_f(height)
        plev = plev_f(height)

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

        ## Omit layers with negative height. Those might occure after interpolation
        idx_positive = np.where(height > 0.0)[0]
        height = height[idx_positive]
        relative_humidity = relative_humidity[idx_positive]
        temperature = temperature[idx_positive]
        plev = plev[idx_positive]
        
        self.__co2 = np.interp(height*1e-3, self.__z, self.__co2)
        self.__n2o = np.interp(height*1e-3, self.__z, self.__n2o)
        self.__o3 = np.interp(height*1e-3, self.__z, self.__o3)
        self.__ch4 = np.interp(height*1e-3, self.__z, self.__ch4)
        self.__co = np.interp(height*1e-3, self.__z, self.__co)
        self.__o2 = np.interp(height*1e-3, self.__z, self.__o2)
    
        self.__atmosphere = {'pressure(hPa)' : plev, \
                            'altitude(m)': height, \
                            'altitude(km)': height*1e-3, \
                            'temperature(K)': temperature, \
                            'humidity(%)': relative_humidity, \
                            'co2(ppmv)': self.__co2, \
                            'n2o(ppmv)': self.__n2o, \
                            'o3(ppmv)': self.__o3, \
                            'ch4(ppmv)': self.__ch4, \
                            'co(ppmv)': self.__co, \
                            'o2(ppmv)': self.__o2}
            
    def read_cloud_height(self, fname, cbh=-1):
        if cbh < -1:
            self.__cloud_base_height = cbh
        else:
            self.__cloud_base_height = read_era5_argmin(fname, \
                                             self.__latitude, \
                                             self.__longitude, \
                                             self.__datetime, 'cbh', 'single')[0]
        cloud_layer = np.abs(self.__atmosphere['altitude(m)']-self.__cloud_base_height)
        idx = list(cloud_layer).index(np.min(cloud_layer))
        self.__cloud_grid = [self.__atmosphere['altitude(m)'][idx]]
        temp_f = scipy.interpolate.interp1d(self.__atmosphere['altitude(m)'], self.__atmosphere['temperature(K)'])
        #Adjust cloud temperature according to the cloud base height
        self.__atmosphere['temperature(K)'][idx] = temp_f(self.__cloud_base_height)
        self.__cloud_temp = self.__atmosphere['temperature(K)'][idx]
                
    def plot_atmosphere(self, fname=""):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Height coordinates")
        ax[1].set_title("Pressure coordinates")
        ax[0].plot(self.__atmosphere['humidity(%)'], self.__atmosphere['altitude(km)'], label="Humidity")
        ax[0].plot(self.__atmosphere['temperature(K)'], self.__atmosphere['altitude(km)'], label="Temperature")
        ax[1].plot(self.__atmosphere['humidity(%)'], self.__atmosphere['pressure(hPa)'], label="Humidity")
        ax[1].plot(self.__atmosphere['temperature(K)'], self.__atmosphere['pressure(hPa)'], label="Temperature")

        ax[1].set_ylim([1013, 0])
        for i in range(2):
            ax[i].grid(True)
            ax[i].legend()
        if fname != "": plt.savefig(fname)
        plt.show()
        
    def return_atmosphere(self):
        return {'atmosphere': self.__atmosphere, \
                'cbh': self.__cloud_base_height, \
                'cloud_grid': self.__cloud_grid, \
                'cloud_temperature': self.__cloud_temp, \
                'datetime': self.__datetime, \
                'latitude': self.__latitude, \
                'longitude': self.__longitude}