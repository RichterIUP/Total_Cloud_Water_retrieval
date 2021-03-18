#!/usr/bin/python
'''@package docstring
Call LBLRTM and DISORT
'''

# -*- coding: utf8 -*-

import os
import subprocess
import sys
import numpy             as np
import netCDF4           as nc
#import inp

#sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
#import physics
#import numerical

#sys.path.append(inp.PATH_TO_RUN_LBLRTM)
#import run_LBLRTM

#LBLDIR = ''

def write_lbldis_input(atmospheric_param, t_surf, ssp, path_wdir, path_windows, sza, cloud_grid, scatter, kurucz, sfc_em, log_re, lbldir):
    '''
    Write input file for LBLDIS
    
    Parameter
    ---------
    atmospheric_param : list
        Microphysical cloud parameters
        
    t_surf : float
        Surface temperature. If negative, surface temperature equals temperature of 
        lowermost atmospheric level
        
    ssp : list
        Names of single-scattering databases
        
    path_wdir : str
        Path to output of TCWret
        
    path_windows : str
        Filename of microwindows
        
    sza : float
        Solar Zenith Angle
        
    cloud_grid : list
        Layers of cloud
        
    scatter : bool
        Use scatter in LBLDIS
        
    kurucz : str
        Name of Kurucz database
        
    sfc_em : list
        Surface emissivity
        
    log_re : bool
        Use logarithmic r_eff
        
    lbldir : str
        Path to lblrtm output
        
    '''
    
    if scatter:
        sign = 1
    else:
        sign = -1
    with open("{}/lbldis.parm".format(path_wdir), "w") as file_:
        file_.write("LBLDIS parameter file\n")
        file_.write("16		Number of streams\n")
        file_.write("{:04.1f} 30. 1.0	Solar ".format(sza))
        file_.write("zenith angle (deg), relative azimuth (deg), solar distance (a.u.)\n")
        file_.write(" 180           Zenith angle (degrees): 0 -> ")
        file_.write("upwelling, 180 -> downwelling\n")
        file_.write("-1 0 0 {}\n".format(path_windows))
        file_.write("{}               ".format(np.int(len(atmospheric_param)*sign)))
        file_.write("Cloud parameter option flag: ")
        file_.write("0: reff and numdens, >=1:  reff and tau\n")
        file_.write("{}".format(len(ssp) * len(cloud_grid)))
        file_.write("               Number of cloud layers\n")
        for loop_liq_layer, dummy in enumerate(cloud_grid, start=0):
            #ii = 0
            alt = cloud_grid[loop_liq_layer]*1e-3
            
            for i, dummy in enumerate(ssp, start=0):
                tau = atmospheric_param[:, i]
                if log_re:
                    file_.write("{} {:5.3f} {:10.8f} -1".format(i, alt, np.exp(atmospheric_param[0, len(atmospheric_param[0])//2+i])))  
                else:
                    file_.write("{} {:5.3f} {:10.8f} -1".format(i, alt, atmospheric_param[0, len(atmospheric_param[0])//2+i]))
                for tau_lay in tau:
                    file_.write(" {:10.8f}".format(tau_lay/len(cloud_grid)))
                file_.write("\n")
        file_.write("{}\n".format(lbldir))
        file_.write("{}\n".format(kurucz))
        num_db = len(ssp)
        file_.write("{}       Number of scattering property databases\n".format(num_db))
        for database in ssp:
            file_.write(database + "\n")
        file_.write("{}	Surface temperature (specifying a negative".format(t_surf))
        file_.write("value takes the value from profile)\n")
        file_.write("{}	Number of surface spectral emissivity lines (wnum, emis)\n".format(len(sfc_em)))
        for eps in sfc_em:
            file_.write("{} {}\n".format(eps[0], eps[1]))

def run_lbldis(atmospheric_param, lblrtm, ssp, wn, atm_grid, path_to_run_lblrtm, path_to_lblrtm, path_to_lbldis, path_wdir, path_windows, sza, cloud_grid, scatter, kurucz, sfc_em, log_re, lbldir, t_surf=-1):
    '''
    Set up LBLDIS and run LBLRTM/DISORT
    
    Parameter
    ---------
    atmospheric_param : list
        Microphysical cloud parameters
        
    lblrtm : bool
        If true, run LBLRTM
        
    ssp : list
        Names of single-scattering databases
        
    wn : list
        Spectral limits of calculation
        
    atm_grid : dict
        Atmospheric profile of pressure, altitude, temperature, humidity and trace gases
        
    path_to_lblrtm : str
        Path to binary of lblrtm
        
    path_to_run_lblrtm : 
        Path to source of run_LBLRTM
        
    path_to_lbldis : str
        Path to binary of lbldis
        
    path_wdir : str
        Path to output of TCWret
        
    path_windows : str
        Filename of microwindows
        
    sza : float
        Solar Zenith Angle
        
    cloud_grid : list
        Layers of cloud
        
    scatter : bool
        Use scatter in LBLDIS
        
    kurucz : str
        Name of Kurucz solar database
        
    sfc_em : list
        Surface emissivity
        
    t_surf : float
        Surface temperature. If negative, surface temperature equals temperature of 
        lowermost atmospheric level
        
    log_re : bool
        Use logarithmic r_eff
        
    lbldir : str
        Path to lblrtm output
        
    Returns
    -------
    np.array
        Radiance
    '''
        
    #global LBLDIR


    # Run LBLRTM
    if lblrtm:
        sys.path.append(path_to_run_lblrtm)
        import run_LBLRTM
        lbldir = run_LBLRTM.run_LBLRTM(z = atm_grid['altitude(km)'], \
                                       p = atm_grid['pressure(hPa)'], \
                                       t = atm_grid['temperature(K)'], \
                                       q = atm_grid['humidity(%)'], \
                                       hmd_unit='H', \
                                       wnum1 = wn[0]-50.0, \
                                       wnum2 = wn[1]+50.0, \
                                       lbltp5 = '{}/tp5'.format(path_wdir), \
                                       lbl_home = path_to_lblrtm, \
                                       path = path_wdir, \
                                       co2 = atm_grid['co2(ppmv)'], \
                                       o3 = atm_grid['o3(ppmv)'], \
                                       co = atm_grid['co(ppmv)'], \
                                       ch4 = atm_grid['ch4(ppmv)'], \
                                       n2o = atm_grid['n2o(ppmv)'], \
                                       o2 = atm_grid['o2(ppmv)'])

    # Write LBLDIS input file
    write_lbldis_input(atmospheric_param, t_surf, ssp, path_wdir, path_windows, sza, cloud_grid, scatter, kurucz, sfc_em, log_re, lbldir)

    # Run LBLDIS
    lbldisout_file = '{}/lbldisout'.format(path_wdir)
    lbldislog = '{}/lbldislog.txt'.format(path_wdir)
    with open("{}/run_disort.sh".format(path_wdir), "w") as file_:
        file_.write("#!/bin/bash\n")
        exec_lbldis = '({}/lbldis {}/lbldis.parm 0 {}) >& {}\n'
        file_.write(exec_lbldis.format(path_to_lbldis, path_wdir, \
                                       lbldisout_file, lbldislog))
    subprocess.call(["bash", "{}/run_disort.sh".format(path_wdir)])
    
    # Read LBLDIS results and perform convolution
    with nc.Dataset("{}/lbldisout.cdf".format(path_wdir)) as disort_out:
        wavenumber = np.array(disort_out.variables['wnum'][:])
        radiance = numerical.conv(wavenumber, np.array(disort_out.variables['radiance'][:]), atmospheric_param)

    return radiance, lbldir
