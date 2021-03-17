#!/usr/bin/python
'''@package Docstring
Some numerical functions
'''

import os
import sys
import numpy as np

import scipy.interpolate

import inp

sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
#import aux2 as aux
import run_lbldis as rL
import misc

ATMOSPHERIC_GRID = dict()
CLOUD_GRID = None
CLOUD_TEMP = 0.0
CLOUD_LAYERS = []
SOLAR_ZENITH_ANGLE = 0.0
RADIANCE_LBLDIS = None
RADIANCE_FTIR = []
RADIANCE_CLEARSKY = []
RESIDUUM = []
WAVENUMBER_FTIR = []
NOISE_FTIR = []
MCP = []
MCP_APRIORI = []
MICROWINDOWS = []

H = 6.62607015e-34
C = 299792458
K = 1.380649e-23

def planck(wavenumber, temperature):

    return 1e3 * 2 * 10**8 * H * C**2 * wavenumber**3 / (np.exp(100*wavenumber*H*C/(K*temperature))-1)

def read_clear_sky_optical_depths(path, max_layer):
    binary = inp.PATH_TO_LBLDIS + "/read_lbl_ods"
    files = sorted(os.listdir(path))

    delta = 2.5
    od_av = np.array([])
    wn_av = np.arange(MICROWINDOWS[0][0], MICROWINDOWS[-1][-1], delta*2)
    od_av = np.zeros(wn_av.size)
    counter = 0
    for file_ in files:
        if "ODd" in file_ and counter <= max_layer:
            counter += 1
            od = np.array([])
            wn = np.array([])
            os.system("{} {}/{} >> {}/{}.txt".format(binary, path, file_, path, file_))
            with open(path + "/" + file_ + ".txt", "r") as f:
                cont = f.readlines()
                for ii in cont[4:-1]:
                    try:
                        od = np.concatenate((od, [np.float(ii.split(" ")[-1])]))
                        wn = np.concatenate((wn, [np.float(ii.split(" ")[-3])]))
                    except ValueError:
                        continue
            for ii in range(wn_av.size):
                idx = np.where((wn > (wn_av[ii] - delta)) & (wn < (wn_av[ii] + delta)))[0]
                od_av[ii] += np.mean(od[idx])
    
    return scipy.interpolate.interp1d(wn_av, np.exp(-od_av/counter), fill_value="extrapolate")

def calculate_cloud_emissivity():
    wn_emis = np.array(inp.EMISSIVITY).T[0]
    sf_emis = np.array(inp.EMISSIVITY).T[1]
    emis_f = scipy.interpolate.interp1d(wn_emis, sf_emis, fill_value="extrapolate")
        
    below_cloud = np.where(ATMOSPHERIC_GRID['altitude(km)']*1e3 < CLOUD_GRID[0])[0][0]
    func = read_clear_sky_optical_depths(rL.LBLDIR, below_cloud)
    transmissivity = func(WAVENUMBER_FTIR)
    t_surf = ATMOSPHERIC_GRID['temperature(K)'][0]
    rad_semiss_075 = rL.run_lbldis(np.array([MCP[-1]]), False, t_surf-5)[-1]
    rad_semiss_025 = rL.run_lbldis(np.array([MCP[-1]]), False, t_surf+5)[-1]
    reflectivity = (rad_semiss_075 - rad_semiss_025)/(transmissivity * (planck(WAVENUMBER_FTIR, t_surf-5) - planck(WAVENUMBER_FTIR, t_surf+5)))
    cemissivity_lbldis = (RADIANCE_LBLDIS[0][-1][:] - RADIANCE_CLEARSKY - reflectivity*transmissivity**2*planck(WAVENUMBER_FTIR, t_surf)*emis_f(WAVENUMBER_FTIR))/(transmissivity*planck(WAVENUMBER_FTIR, CLOUD_TEMP))
    cemissivity_ftir = (RADIANCE_FTIR - RADIANCE_CLEARSKY - reflectivity*transmissivity**2*planck(WAVENUMBER_FTIR, t_surf)*emis_f(WAVENUMBER_FTIR))/(transmissivity*planck(WAVENUMBER_FTIR, CLOUD_TEMP))
    
    return reflectivity, transmissivity, cemissivity_lbldis, cemissivity_ftir
