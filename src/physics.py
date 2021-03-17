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
import run_lbldis as rL

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

H = 6.62607015e-34#Js
C = 299792458#ms-1
K = 1.380649e-23#JK-1
G0 = 9.80665#ms-2


def planck(wavenumber, temperature):
    '''
    Calculate plack radiation

    Parameter
    ---------
    wavenumber : np.array
        Wavenumber
    temperature : float
        Temperature

    Returns
    -------
    np.array
        Planck radiation
    '''
    return 1e3 * 2 * 10**8 * H * C**2 * wavenumber**3 / \
        (np.exp(100*wavenumber*H*C/(K*temperature))-1)

def read_clear_sky_optical_depths(path, max_layer):
    '''
    Read gaseous optical depths calculated from LBLRTM using read_lbl_ods from LBLDIS

    Parameters
    ----------
    path : str
        Path to optical depths of LBLRTM
    max_layer : int
        Index of uppermost layer below cloud

    Returns
    -------
    func
        Function of transmissivities

    '''
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
            optical_depth = np.array([])
            wavenumber = np.array([])
            os.system("{0} {1}/{2} >> {1}/{2}.txt".format(binary, path, file_))
            with open(path + "/" + file_ + ".txt", "r") as file_:
                cont = file_.readlines()
                for i in cont[4:-1]:
                    try:
                        optical_depth = np.concatenate((optical_depth, \
                                                        [np.float(i.split(" ")[-1])]))
                        wavenumber = np.concatenate((wavenumber, [np.float(i.split(" ")[-3])]))
                    except ValueError:
                        continue
            for i in range(wn_av.size):
                idx = np.where((wavenumber > (wn_av[i] - delta)) & \
                               (wavenumber < (wn_av[i] + delta)))[0]
                od_av[i] += np.mean(optical_depth[idx])

    return scipy.interpolate.interp1d(wn_av, np.exp(-od_av/counter), fill_value="extrapolate")

def calculate_cloud_emissivity():
    '''
    Calculate emssivity of cloud by changing surface temperature

    Returns
    -------
    np.array
        Reflectivity of the cloud
    np.array
        Transmissivity of air below cloud
    np.array
        Emissivity of the cloud calculated by LBLDIS
    np.array
        Emissivity of the cloud from FTIR measurement
    '''
    wn_emis = np.array(inp.EMISSIVITY).T[0]
    sf_emis = np.array(inp.EMISSIVITY).T[1]
    emis_f = scipy.interpolate.interp1d(wn_emis, sf_emis, fill_value="extrapolate")

    below_cloud = np.where(ATMOSPHERIC_GRID['altitude(km)']*1e3 < CLOUD_GRID[0])[0][0]
    func = read_clear_sky_optical_depths(rL.LBLDIR, below_cloud)
    transmissivity = func(WAVENUMBER_FTIR)
    t_surf = ATMOSPHERIC_GRID['temperature(K)'][0]
    rad_semiss_075 = rL.run_lbldis(np.array([MCP[-1]]), False, t_surf-5)[-1]
    rad_semiss_025 = rL.run_lbldis(np.array([MCP[-1]]), False, t_surf+5)[-1]
    reflectivity = (rad_semiss_075 - rad_semiss_025)/\
        (transmissivity * (planck(WAVENUMBER_FTIR, t_surf-5) - planck(WAVENUMBER_FTIR, t_surf+5)))

    # Calculate emissivity of cloud from LBLDIS calulation
    cemissivity_lbldis = (RADIANCE_LBLDIS[0][-1] - RADIANCE_CLEARSKY -\
                          reflectivity*transmissivity**2*planck(WAVENUMBER_FTIR, t_surf)\
                              *emis_f(WAVENUMBER_FTIR))/(transmissivity*\
                                                         planck(WAVENUMBER_FTIR, CLOUD_TEMP))

    # Calculate emissivity of cloud from FTIR measurement
    cemissivity_ftir = (RADIANCE_FTIR - RADIANCE_CLEARSKY - \
                        reflectivity*transmissivity**2*planck(WAVENUMBER_FTIR, t_surf)\
                            *emis_f(WAVENUMBER_FTIR))/(transmissivity*\
                                                       planck(WAVENUMBER_FTIR, CLOUD_TEMP))

    return reflectivity, transmissivity, cemissivity_lbldis, cemissivity_ftir
