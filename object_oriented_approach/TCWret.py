#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:28:34 2021

@author: philipp
"""

import os
import datetime as dt

import numpy as np
import netCDF4 as nc
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt


import run_lbldis as rL
import numerical

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

def read_clear_sky_optical_depths(path, max_layer, microwindows):
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
    binary = "/home/philipp/Doktorandenzeit/SOFTWARE_PHD/Total_Cloud_Water_retrieval/rtm/lbldis" + "/read_lbl_ods"
    files = sorted(os.listdir(path))

    delta = 2.5
    od_av = np.array([])
    wn_av = np.arange(microwindows[0][0], microwindows[-1][-1], delta*2)
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

class TCWret:
    
    def __init__(self, path, results, lblrtm, run_lblrtm, lbldis, run_lbldis, resolution, kurucz_db, ssp_db, ssp_temp, tau, reff, var_tau, var_reff, log_reff=True, max_reff = 100, min_reff=1, max_iter=30, conv=1e-3, lm_init=1e2, lm_decr=2.0, lm_incr=4.0, lm_min=0.0, semiss=[[100., 0.98], [700., 0.98], [800., 0.98], [3000, 0.98]], cemiss=False):
        self.__PATH = path
        self.__RESULTS = results
        self.__run_LBLRTM = run_lblrtm
        self.__run_LBLDIS = run_lbldis
        self.__LBLRTM = lblrtm
        self.__LBLDIS = lbldis
        self.__resolution = resolution
        self.__lm_init = lm_init
        self.__lm_decr = lm_decr
        self.__lm_incr = lm_incr
        self.__lm_min = lm_min
        self.__kurucz_db = kurucz_db
        self.__ssp_db = ssp_db
        self.__ssp_temp = ssp_temp
        self.__tau = tau
        self.__reff = reff
        self.__mcp = np.array([tau, reff]).flatten()
        self.__var_tau = var_tau
        self.__var_reff = var_reff
        self.__variance_apriori = np.array([var_tau, var_reff]).flatten()
        self.__log_reff = log_reff
        self.__max_iter = max_iter
        self.__conv = conv
        self.__max_reff = max_reff
        self.__min_reff = min_reff
        self.__semiss = semiss
        self.__cemiss = cemiss
        self.__stepsize = 1e-3
        self.__chi2 = 1e100
        self.__chi2_prev = 1e100
        self.__mcp_prev = np.array([tau, reff]).flatten()
        self.__radiance_lbl = [None for i in range(len(self.__mcp)+1)]
        self.__radiance_lbl_prev = [None for i in range(len(self.__mcp)+1)]
        self.__radiance_clear = None
        self.__cemissivity_ftir = None
        self.__cemissivity_lbl = [None for i in range(len(self.__mcp)+1)]
        self.__cemissivity_lbl_prev = [None for i in range(len(self.__mcp)+1)]
        
    def microwindows(self, mw):
        '''
        

        Parameters
        ----------
        mw : List
            List containing the intervals of the microwindows.

        Returns
        -------
        None.

        '''
        self.__windows = mw
        with open("mwfile", "w") as f:
            f.write("{}\n".format(len(self.__windows)))
            for i in self.__windows:
                f.write("{} {}\n".format(i[0], i[1]))
                
    def return_microwindows(self):
        return self.__windows
    
    def add_spectrum(self, spec):
        self.__radiance_ftir = spec['radiance_ftir']
        self.__wavenumber_ftir = spec['wavenumber']
        self.__noise_ftir = spec['noise_ftir']
        self.__latitude = spec['latitude']    
        self.__longitude = spec['longitude']
        self.__sza = spec['sza']
        self.__datetime = spec['datetime']
        
    def add_atmosphere(self, atmo):
        self.__atmosphere = atmo['atmosphere']
        self.__cloud_base_height = atmo['cbh']
        self.__cloud_grid = atmo['cloud_grid']
        self.__cloud_temp = atmo['cloud_temperature']

    def plot_spectrum(self, ylim=[-1, -1], xlim=[-1, -1], fname=""):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(self.__wavenumber_ftir, self.__radiance_ftir, label="FTIR")
        ax.plot(self.__wavenumber_ftir, self.__radiance_lbl[0], label="LBLDIS")
        ax.plot(self.__wavenumber_ftir, self.__radiance_lbl_clear, label="Clear Sky")
        ax.legend()
        ax.grid(True)
        if ylim != [-1, -1]: ax.set_ylim(ylim)
        if xlim != [-1, -1]: ax.set_xlim(xlim)
        if fname != "": plt.savefig(fname)
        plt.show()
            
    def plot_emissivity(self, ylim=[-1, -1], xlim=[-1, -1], fname=""):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(self.__wavenumber_ftir, self.__cemissivity_ftir, label="FTIR")
        ax.plot(self.__wavenumber_ftir, self.__cemissivity_lbl[0], label="LBLDIS")
        ax.legend()
        ax.grid(True)
        if ylim != [-1, -1]: ax.set_ylim(ylim)
        if xlim != [-1, -1]: ax.set_xlim(xlim)
        if fname != "": plt.savefig(fname)
        plt.show()
        
    def get_mcp(self):
        return self.__mcp[:]
    
    def set_mcp(self, mcp):
        self.__mcp = mcp
        
    def get_spectrum(self):
        return {'radiance': self.__radiance_lbl[0], \
                'emissivity_lbl' :self.__cemissivity_lbl[0], \
                'emissivity_ftir': self.__cemissivity_ftir}
        
    def prepare_retrieval(self):
        vec_error = np.array([np.mean(self.__noise_ftir)**2 for ii in range(len(self.__wavenumber_ftir))])
        self.__s_y_inv = np.reciprocal(vec_error) * np.identity(len(vec_error))
        self.__t_matrix = [np.zeros((len(self.__mcp), len(self.__wavenumber_ftir)))]
        for i in range(len(self.__ssp_db)):
            if self.__cloud_temp < self.__ssp_temp[i][0] or \
                self.__cloud_temp > self.__ssp_temp[i][1]:
                self.__mcp[i] = 0.0
                self.__variance_apriori[i] = 1e-10**(-2)
    
        # Set up the S_a matrix
        self.__mcp_apriori = self.__mcp[:]
        self.__s_a_inv_matrix = np.array(self.__variance_apriori) * np.identity(self.__mcp_apriori.size)
        self.__path = dt.datetime.strftime(dt.datetime.now(), "%m_%d_%H_%M_%S_%f")
        os.mkdir(self.__path)

    def run_LBLDIS(self, clear_sky=False): 
            
        # Calculate spectrum
        radiance, self.__lbldir = rL.run_lbldis(np.array([self.__mcp]), \
                                                          lblrtm=True, \
                                                          ssp=self.__ssp_db, \
                                                          wn=[self.__windows[0][0], self.__windows[-1][-1]], \
                                                          atm_grid=self.__atmosphere, \
                                                          path_to_run_lblrtm="./run_LBLRTM", \
                                                          path_to_lblrtm=self.__LBLRTM, \
                                                          path_wdir=self.__path, \
                                                          path_to_lbldis=self.__LBLDIS, \
                                                          sza=self.__sza, \
                                                          path_windows="mwfile", \
                                                          cloud_grid=self.__cloud_grid, \
                                                          scatter=True, \
                                                          kurucz=self.__kurucz_db, \
                                                          sfc_em=self.__semiss, \
                                                          log_re=self.__log_reff, \
                                                          lbldir='', \
                                                          resolution=self.__resolution)  
            
        if clear_sky: 
            self.__radiance_lbl_clear = radiance
        else:
            self.__radiance_lbl[0] = radiance
            
    def run_LBLDIS_deriv(self):
        mcp = np.ones((int(self.__mcp.size//2+1), int(self.__mcp.size)))*self.__mcp

        for i in range(1, self.__mcp.size//2+1):
            mcp[i, i-1] += self.__stepsize

        radiance = rL.run_lbldis(np.array(mcp), \
                                 lblrtm=False, \
                                 ssp=self.__ssp_db, \
                                wn=[self.__windows[0][0], self.__windows[-1][-1]], \
                                atm_grid=self.__atmosphere, \
                                path_to_run_lblrtm="./run_LBLRTM", \
                                path_to_lblrtm=self.__LBLRTM, \
                                path_wdir=self.__path, \
                                path_to_lbldis=self.__LBLDIS, \
                                sza=self.__sza, \
                                path_windows="mwfile", \
                                cloud_grid=self.__cloud_grid, \
                                scatter=True, \
                                kurucz=self.__kurucz_db, \
                                sfc_em=self.__semiss, \
                                log_re=self.__log_reff, \
                                lbldir=self.__lbldir, \
                                resolution=self.__resolution)[0]

        for i in range(self.__mcp.size//2+1):
            self.__radiance_lbl[i] = radiance[:, i].flatten()

        delta = np.zeros(self.__mcp.size)
        for i in range(self.__mcp.size//2+1, self.__mcp.size+1):
            delta[i-1] = self.__stepsize
            mcp = [self.__mcp + delta]
            radiance = rL.run_lbldis(np.array(mcp), \
                                     lblrtm=False, \
                                     ssp=self.__ssp_db, \
                                    wn=[self.__windows[0][0], self.__windows[-1][-1]], \
                                    atm_grid=self.__atmosphere, \
                                    path_to_run_lblrtm="./run_LBLRTM", \
                                    path_to_lblrtm=self.__LBLRTM, \
                                    path_wdir=self.__path, \
                                    path_to_lbldis=self.__LBLDIS, \
                                    sza=self.__sza, \
                                    path_windows="mwfile", \
                                    cloud_grid=self.__cloud_grid, \
                                    scatter=True, \
                                    kurucz=self.__kurucz_db, \
                                    sfc_em=self.__semiss, \
                                    log_re=self.__log_reff, \
                                    lbldir=self.__lbldir, \
                                    resolution=self.__resolution)[0]
            try:
                self.__radiance_lbl[i] = radiance[:, 0].flatten()
            except IndexError:
                self.__radiance_lbl_prev[i] = self.__radiance_lbl[i]
                self.__radiance_lbl[i] = radiance.flatten()
            
    def calculate_cloud_emissivity(self):
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
        
        wn_emis = np.array(self.__semiss).T[0]
        sf_emis = np.array(self.__semiss).T[1]
        emis_f = scipy.interpolate.interp1d(wn_emis, sf_emis, fill_value="extrapolate")

        below_cloud = np.where(self.__atmosphere['altitude(m)'] < self.__cloud_base_height)[0][-1]

        func = read_clear_sky_optical_depths(self.__lbldir, below_cloud, self.__windows)
        self.__transmissivity = func(self.__wavenumber_ftir)
        t_surf = self.__atmosphere['temperature(K)'][0]
        rad_semiss_075 = rL.run_lbldis(np.array([self.__mcp]), \
                                       lblrtm=False, \
                                       t_surf=t_surf-5, \
                                    ssp=self.__ssp_db, \
                                    wn=[self.__windows[0][0], self.__windows[-1][-1]], \
                                    atm_grid=self.__atmosphere, \
                                    path_to_run_lblrtm="./run_LBLRTM", \
                                    path_to_lblrtm=self.__LBLRTM, \
                                    path_wdir=self.__path, \
                                    path_to_lbldis=self.__LBLDIS, \
                                    sza=self.__sza, \
                                    path_windows="mwfile", \
                                    cloud_grid=self.__cloud_grid, \
                                    scatter=True, \
                                    kurucz=self.__kurucz_db, \
                                    sfc_em=self.__semiss, \
                                    log_re=self.__log_reff, \
                                    lbldir='', \
                                    resolution=self.__resolution)[0]  
        
        rad_semiss_025 = rL.run_lbldis(np.array([self.__mcp]), \
                                       lblrtm=False, \
                                       t_surf=t_surf+5, \
                                    ssp=self.__ssp_db, \
                                    wn=[self.__windows[0][0], self.__windows[-1][-1]], \
                                    atm_grid=self.__atmosphere, \
                                    path_to_run_lblrtm="./run_LBLRTM", \
                                    path_to_lblrtm=self.__LBLRTM, \
                                    path_wdir=self.__path, \
                                    path_to_lbldis=self.__LBLDIS, \
                                    sza=self.__sza, \
                                    path_windows="mwfile", \
                                    cloud_grid=self.__cloud_grid, \
                                    scatter=True, \
                                    kurucz=self.__kurucz_db, \
                                    sfc_em=self.__semiss, \
                                    log_re=self.__log_reff, \
                                    lbldir='', \
                                    resolution=self.__resolution)[0]  

        self.__reflectivity = (rad_semiss_075 - rad_semiss_025)/\
            (self.__transmissivity * (planck(self.__wavenumber_ftir, t_surf-5) - planck(self.__wavenumber_ftir, t_surf+5)))

        # Calculate emissivity of cloud from LBLDIS calulation
        for i in range(len(self.__radiance_lbl)):
            self.__cemissivity_lbl_prev[i] = self.__cemissivity_lbl[i]
            self.__cemissivity_lbl[i] = (self.__radiance_lbl[i] - self.__radiance_lbl_clear -\
                                  self.__reflectivity*self.__transmissivity**2*planck(self.__wavenumber_ftir, t_surf)\
                                      *emis_f(self.__wavenumber_ftir))/(self.__transmissivity*\
                                                                 planck(self.__wavenumber_ftir, self.__cloud_temp))

        # Calculate emissivity of cloud from FTIR measurement
        self.__cemissivity_ftir = (self.__radiance_ftir - self.__radiance_lbl_clear - \
                            self.__reflectivity*self.__transmissivity**2*planck(self.__wavenumber_ftir, t_surf)\
                                *emis_f(self.__wavenumber_ftir))/(self.__transmissivity*\
                                                           planck(self.__wavenumber_ftir, self.__cloud_temp))

        #return reflectivity, transmissivity, cemissivity_lbldis, cemissivity_ftir
        
    def calc_chi_2_and_residuum(self):
        '''
        Calculate the new cost function and residuum

        Returns
        -------
        float
            New cost function

        float
            New residuum
        '''
        if self.__cemiss:
            ftir = self.__cemissivity_ftir
            lbl = self.__cemissivity_lbl[0]
        else:
            ftir = self.__radiance_ftir
            lbl = self.__radiance_lbl[0]
            
        self.__chi2_prev = self.__chi2
            
        # Get residuum
        res = np.array(ftir) - np.array(lbl)
        self.__res = np.transpose(np.matrix(res))

        # (y - F(x))^T S_y_1 (y - F(x))
        _res = np.float_(np.dot(np.matmul(np.transpose(self.__res), self.__s_y_inv), \
                                res))

        apr_vec = self.__mcp_apriori - self.__mcp

        # (x_a - x_i)^T S_a_1 (x_a - x_i)
        _apr = np.float_(np.dot(np.matmul(np.transpose(apr_vec), self.__s_a_inv_matrix), apr_vec))

        # chi^2 = (y - F(x))^T S_y_1 (y - F(x)) + Calculate (x_a - x_i)^T S_a_1 (x_a - x_i)
        self.__chi2 = np.float_(_res + _apr)
        return self.__chi2, self.__res

    def jacobian(self):
        '''
        Calculation of the jacobian_matrix

        Returns
        -------
        np.array
            Transposed jacobian matrix
        '''

        self.__deriv = [self.__radiance_lbl[0] for i in range(len(self.__mcp))]
        for i in range(1, self.__mcp.size+1):
            if self.__cemiss:
                self.__deriv[i-1] = (self.__cemissivity_lbl[i] \
                         - self.__cemissivity_lbl[0])/self.__stepsize
            else:
                self.__deriv[i-1] = (self.__radiance_lbl[i] \
                        - self.__radiance_lbl[0])/self.__stepsize
            
        return np.matrix(self.__deriv)
    
    def adjust_lm(self):
        if self.__chi2_prev > self.__chi2 and np.where(self.__mcp < 0.0)[0].size == 0:
            self.__lm_init /= self.__lm_decr
        else:
            self.__lm_init *= self.__lm_incr
            self.__mcp = self.__mcp_prev
            self.__radiance_lbl = self.__radiance_lbl_prev
            self.__cemissivity_lbl = self.__cemissivity_lbl_prev
            
    def convergence(self):
        ## Test if convergence reached
        return numerical.convergence(self.__chi2, \
                                     self.__chi2_prev, \
                                     self.__conv)
    
    def optimal_estimation(self):#lm_param, t_matrix):
        '''
        Calculate the adjustment vector according to optimal estimation

        This function solves the equation
        [J^T S_y_1 J + S_a_1 + mu**2 D]s_n = J^T S_y_1 [y - F(x_n)] + S_a_1 (x_a - x_i)
        w.r.t. s_n, the so-called adjustment vector

        Parameter
        ---------

        lm_param : float
            Levenberg-Marquardt parameter

        t_matrix : np.matrix
            Transfer matrix which propagates covariances from measurement to result

        Returns
        -------
        np.array
            Adjustment vector
        np.matrix
            New transfer matrix
        '''

        jacobian_matrix_transposed = np.matrix(self.__deriv)
        jacobian_matrix = np.transpose(jacobian_matrix_transposed)

        # K^T S_y_1
        JT_W = np.matmul(jacobian_matrix_transposed, self.__s_y_inv)

        # K^T S_y_1 K
        JT_W_J = np.matmul(JT_W, jacobian_matrix)

        # K^T S_y_1 (y-F(x))
        JT_W_r = np.matmul(JT_W, self.__res)

        # Calculate S_a_1 (x_a - x_i)
        second_sum = np.matmul(self.__s_a_inv_matrix, self.__mcp_apriori - self.__mcp)

        # Calculate K^T S_y_1 J + S_a_1 + mu**2 D
        left_side = np.add(np.add(JT_W_J, self.__s_a_inv_matrix), self.__lm_init**2*self.__s_a_inv_matrix)

        # Calculate K^T S_y_1 (y-F(x)) + S_a_1 (x_a - x_i)
        right_side = \
        np.transpose(np.add(np.transpose(JT_W_r), second_sum))

        # Solve
        self.__s_n = np.array(np.linalg.solve(left_side, right_side)).flatten()

        # Calculate transfer matrix
        #M = np.linalg.inv(left_side)
        #G = np.matmul(M, JT_W)
        #I_GJ_MR = np.identity(len(self.__mcp)) - np.matmul(G, jacobian_matrix) \
        #        - np.matmul(M, self.__s_a_inv_matrix)
        #T_new = np.add(G, np.matmul(I_GJ_MR, t_matrix))
        return self.__s_n#, T_new
    
    def adjust_mcp(self):
        self.__mcp_prev = self.__mcp
        self.__mcp += self.__s_n

