#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:59:45 2021

@author: philipp
"""

import datetime as dt

import numpy as np
import netCDF4 as nc
import scipy.optimize
import matplotlib.pyplot as plt

class Spectrum:
    def __init__(self, spectral_radiance_file, index_of_spec):
        with nc.Dataset(spectral_radiance_file) as rad_f:
            # Find spectrum corresponding to the specified time
            reference_date_of_spectral_radiances = \
                    dt.datetime.strptime(rad_f.variables['time_dec'].units, \
                                         "hours since %Y-%m-%d %H:%M:%S")

            # Read spectrum
            self.__datetime = reference_date_of_spectral_radiances + \
                dt.timedelta(seconds=rad_f.variables['time_dec'][index_of_spec] * 3600.0)
            self.__latitude = float(rad_f.variables['lat'][index_of_spec])
            self.__longitude = float(rad_f.variables['lon'][index_of_spec])
            self.__sza = float(rad_f.variables['sza'][index_of_spec])
            self.__wavenumber_ftir = np.array(rad_f.variables['wavenumber'][index_of_spec][:].flatten())
            self.__radiance_ftir = np.array(rad_f.variables['radiance'][index_of_spec][:].flatten())
            self.__radiance_lbl_clear = np.zeros(self.__wavenumber_ftir.size)
            #self.__radiance_lbl = [np.zeros(self.__wavenumber_ftir.size)]
            #self.__cemissivity_ftir = np.zeros(self.__wavenumber_ftir.size)
            #self.__cemissivity_lbldis = [np.zeros(self.__wavenumber_ftir.size)]

            if self.__sza > 90.0: self.__sza = -1
            
    def average_radiance_per_windows(self, windows):
        # # Average radiance on windows
        num_windows = len(windows)
        wavenumber_av = np.zeros(num_windows)
        radiance_av = np.zeros(num_windows)

        for window in range(num_windows):
            idx_window = np.where((self.__wavenumber_ftir >= windows[window][0]) & \
                                  (self.__wavenumber_ftir <= windows[window][1]))
            radiance_av[window] = np.mean(self.__radiance_ftir[idx_window])
            wavenumber_av[window] = np.mean(self.__wavenumber_ftir[idx_window])


        idx_noise = np.where((self.__wavenumber_ftir > 1925) & (self.__wavenumber_ftir < 2000))[0]
        func = lambda x, a, b: a * x + b
        popt = scipy.optimize.curve_fit(func, self.__wavenumber_ftir[idx_noise], self.__radiance_ftir[idx_noise])[0]
        noise = self.__radiance_ftir[idx_noise] - func(self.__wavenumber_ftir[idx_noise], \
                                                        popt[0], popt[1])
    
        self.__wavenumber_ftir = wavenumber_av
        self.__radiance_ftir = radiance_av
        self.__noise_ftir = np.std(noise)
        #self.__radiance_lbl = [np.zeros(radiance_av.size) for ii in range(num_mcp+1)]
        #self.__radiance_lbl_prev = [np.zeros(radiance_av.size) for ii in range(num_mcp+1)]
        #self.__cemissivity_ftir = np.zeros(radiance_av.size)
        #self.__cemissivity_lbldis = [np.zeros(radiance_av.size) for ii in range(num_mcp+1)]
        #self.__cemissivity_lbldis_prev = [np.zeros(radiance_av.size) for ii in range(num_mcp+1)]
        #self.__radiance_lbl_clear = np.zeros(radiance_av.size)

        #vec_error = np.array([np.mean(self.__noise_ftir)**2 for ii in range(len(self.__wavenumber_ftir))])
        #self.__s_y_inv = np.reciprocal(vec_error) * np.identity(len(vec_error))
        #self.__t_matrix = [np.zeros((num_mcp, len(self.__wavenumber_ftir)))]
        
    def plot_spectrum(self, ylim=[-1, -1], xlim=[-1, -1], fname=""):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(self.__wavenumber_ftir, self.__radiance_ftir, label="FTIR")
        ax.legend()
        ax.grid(True)
        if ylim != [-1, -1]: ax.set_ylim(ylim)
        if xlim != [-1, -1]: ax.set_xlim(xlim)
        if fname != "": plt.savefig(fname)
        plt.show()     
        
    def return_spectrum(self):
        return {'wavenumber': self.__wavenumber_ftir, \
                'radiance_ftir': self.__radiance_ftir, \
                'noise_ftir': self.__noise_ftir, \
                'latitude': self.__latitude, \
                'longitude': self.__longitude, \
                'sza': self.__sza, \
                'datetime': self.__datetime}