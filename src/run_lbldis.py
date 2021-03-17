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
import inp

sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
import physics
import numerical

sys.path.append(inp.PATH_TO_RUN_LBLRTM)
import run_LBLRTM

LBLDIR = ''

def write_lbldis_input(atmospheric_param, t_surf):
    '''
    Write input file for LBLDIS
    
    Parameter
    ---------
    atmospheric_param : list
        Microphysical cloud parameters
        
    t_surf : float
        Surface temperature. If negative, surface temperature equals temperature of 
        lowermost atmospheric level
    '''
    if inp.SCATTER:
        sign = 1
    else:
        sign = -1
    with open("{}/lbldis.parm".format(inp.PATH), "w") as file_:
        file_.write("LBLDIS parameter file\n")
        file_.write("16		Number of streams\n")
        file_.write("{:04.1f} 30. 1.0	Solar ".format(physics.SOLAR_ZENITH_ANGLE))
        file_.write("zenith angle (deg), relative azimuth (deg), solar distance (a.u.)\n")
        file_.write(" 180           Zenith angle (degrees): 0 -> ")
        file_.write("upwelling, 180 -> downwelling\n")
        file_.write("-1 0 0 {}\n".format(inp.WINDOWS))
        file_.write("{}               ".format(np.int(len(atmospheric_param)*sign)))
        file_.write("Cloud parameter option flag: ")
        file_.write("0: reff and numdens, >=1:  reff and tau\n")
        file_.write("{}".format(len(inp.DATABASES) * len(physics.CLOUD_GRID)))
        file_.write("               Number of cloud layers\n")
        for loop_liq_layer, dummy in enumerate(physics.CLOUD_GRID, start=0):#range(n_layer):
            #ii = 0
            alt = physics.CLOUD_GRID[loop_liq_layer]*1e-3
            
            for i, dummy in enumerate(inp.DATABASES, start=0):#range(len(inp.DATABASES)):
                tau = atmospheric_param[:, i]
                if inp.LOG:
                    file_.write("{} {:5.3f} {:10.8f} -1".format(i, alt, np.exp(atmospheric_param[0, inp.MCP.size//2+i])))  
                else:
                    file_.write("{} {:5.3f} {:10.8f} -1".format(i, alt, atmospheric_param[0, inp.MCP.size//2+i]))
                for tau_lay in tau:
                    file_.write(" {:10.8f}".format(tau_lay/len(physics.CLOUD_GRID)))
                file_.write("\n")
        file_.write("{}\n".format(LBLDIR))
        file_.write("{}\n".format(inp.KURUCZ))
        num_db = len(inp.DATABASES)
        file_.write("{}       Number of scattering property databases\n".format(num_db))
        for database in inp.DATABASES:
            file_.write(database + "\n")
        file_.write("{}	Surface temperature (specifying a negative".format(t_surf))
        file_.write("value takes the value from profile)\n")
        file_.write("{}	Number of surface spectral emissivity lines (wnum, emis)\n".format(len(inp.EMISSIVITY)))
        for eps in inp.EMISSIVITY:
            file_.write("{} {}\n".format(eps[0], eps[1]))

def run_lbldis(atmospheric_param, lblrtm, t_surf=-1):
    '''
    Set up LBLDIS and run LBLRTM/DISORT
    
    Parameter
    ---------
    atmospheric_param : list
        Microphysical cloud parameters
        
    lblrtm : bool
        If true, run LBLRTM
        
    t_surf : float
        Surface temperature. If negative, surface temperature equals temperature of 
        lowermost atmospheric level
        
    Returns
    -------
    np.array
        Wavenumber
        
    np.array
        Radiance
    '''
        
    global LBLDIR
    
    # Increase interval by 50.0 cm-1 at both interval limits
    wnum1 = physics.MICROWINDOWS[0][0]-50.0
    wnum2 = physics.MICROWINDOWS[-1][-1]+50.0
    
    ##Read trace gases and interpolate to atmospheric grid
    co2 = np.loadtxt(inp.CO2_FILE, delimiter=",")
    n2o = np.loadtxt(inp.N2O_FILE, delimiter=",")
    o3  = np.loadtxt(inp.O3_FILE, delimiter=",")
    ch4 = np.loadtxt(inp.CH4_FILE, delimiter=",")
    co  = np.loadtxt(inp.CO_FILE, delimiter=",")
    o2  = np.loadtxt(inp.O2_FILE, delimiter=",")
    height = np.loadtxt(inp.HEIGHT_FILE, delimiter=",")
    co2 = np.interp(physics.ATMOSPHERIC_GRID['altitude(km)'], height, co2)
    n2o = np.interp(physics.ATMOSPHERIC_GRID['altitude(km)'], height, n2o)
    o3 = np.interp(physics.ATMOSPHERIC_GRID['altitude(km)'], height, o3)
    ch4 = np.interp(physics.ATMOSPHERIC_GRID['altitude(km)'], height, ch4)
    co = np.interp(physics.ATMOSPHERIC_GRID['altitude(km)'], height, co)
    o2 = np.interp(physics.ATMOSPHERIC_GRID['altitude(km)'], height, o2)

    # Run LBLRTM
    if lblrtm:
        LBLDIR = run_LBLRTM.run_LBLRTM(z = physics.ATMOSPHERIC_GRID['altitude(km)'], \
                                       p = physics.ATMOSPHERIC_GRID['pressure(hPa)'], \
                                       t = physics.ATMOSPHERIC_GRID['temperature(K)'], \
                                       q = physics.ATMOSPHERIC_GRID['humidity(%)'], \
                                       hmd_unit='H', \
                                       wnum1 = wnum1, \
                                       wnum2 = wnum2, \
                                       lbltp5 = '{}/tp5'.format(inp.PATH), \
                                       lbl_home = inp.PATH_TO_LBLRTM, \
                                       path = inp.PATH, co2=co2, o3=o3, co=co, ch4=ch4, n2o=n2o, o2=o2)

    # Write LBLDIS input file
    write_lbldis_input(atmospheric_param, t_surf)

    # Run LBLDIS
    lbldisout_file = '{}/lbldisout'.format(inp.PATH)
    lbldislog = '{}/lbldislog.txt'.format(inp.PATH)
    with open("{}/run_disort.sh".format(inp.PATH), "w") as file_:
        file_.write("#!/bin/bash\n")
        exec_lbldis = '({}/lbldis {}/lbldis.parm 0 {}) >& {}\n'
        file_.write(exec_lbldis.format(inp.PATH_TO_LBLDIS, inp.PATH, \
                                       lbldisout_file, lbldislog))
    subprocess.call(["bash", "{}/run_disort.sh".format(inp.PATH)])
    
    # Read LBLDIS results and perform convolution
    with nc.Dataset("{}/lbldisout.cdf".format(inp.PATH)) as disort_out:
        radiance = np.array(disort_out.variables['radiance'][:])
        wavenumber = np.array(disort_out.variables['wnum'][:])
        if inp.CONVOLUTION: radiance = numerical.conv(wavenumber, radiance, atmospheric_param)

    return wavenumber, radiance
