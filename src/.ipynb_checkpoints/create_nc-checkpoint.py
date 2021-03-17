#!/usr/bin/python3
'''@package docstring
Write the results to a netCDF file
'''

import sys
import os
import datetime as dt
import scipy.io as sio
import numpy as np
from scipy.interpolate import interp1d
import inp
sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
import aux2 as aux
#import inversion
import log
import numerical
import run_lbldis as rL
import physics

#def read_clear_sky_optical_depths(path, max_layer):
#    binary = os.getenv('HOME') + "/radiative_transfer/lbldis/read_lbl_ods"

#    files = sorted(os.listdir(path))

#    delta = 2.5
#    od_av = np.array([])
#    wn_av = np.arange(aux.MICROWINDOWS[0][0], aux.MICROWINDOWS[-1][-1], delta*2)
#    od_av = np.zeros(wn_av.size)
#    counter = 0
#    for file_ in files:
#        if "ODd" in file_ and counter <= max_layer:
#            counter += 1
#            od = np.array([])
#            wn = np.array([])
#            os.system("{} {}/{} >> {}/{}.txt".format(binary, path, file_, path, file_))
#            with open(path + "/" + file_ + ".txt", "r") as f:
#                cont = f.readlines()
#                for ii in cont[4:-1]:
#                    try:
#                        od = np.concatenate((od, [np.float(ii.split(" ")[-1])]))
#                        wn = np.concatenate((wn, [np.float(ii.split(" ")[-3])]))
#                    except ValueError:
#                        continue
#            for ii in range(wn_av.size):
#                idx = np.where((wn > (wn_av[ii] - delta)) & (wn < (wn_av[ii] + delta)))[0]
#                od_av[ii] += np.mean(od[idx])
    
#    return interp1d(wn_av, np.exp(-od_av/counter), fill_value="extrapolate")

#def calculate_emissivity():
#    below_cloud = np.where(aux.ATMOSPHERIC_GRID['altitude(km)']*1e3 < aux.CLOUD_GRID[0])[0][0]
#    func = read_clear_sky_optical_depths(aux.LBL_WORK, below_cloud)
#    transmissivity = func(aux.WAVENUMBER_FTIR)
#    t_surf = aux.ATMOSPHERIC_GRID['temperature(K)'][0]
#    rad_semiss_075 = rL.run_lbldis(np.array([aux.MCP[-1]]), False, tsurf-5)[-1]
#    rad_semiss_025 = rL.run_lbldis(np.array([aux.MCP[-1]]), False, tsurf+5)[-1]
#    reflectivity = (rad_semiss_075 - rad_semiss_025)/(transmissivity * (numerical.planck(aux.WAVENUMBER_FTIR, t_surf-5) - numerical.planck(aux.WAVENUMBER_FTIR, t_surf+5)))
#    cemissivity_lbldis = (aux.RADIANCE_LBLDIS[0][-1][:] - aux.RADIANCE_CLEARSKY - reflectivity*transmissivity**2*numerical.planck(aux.WAVENUMBER_FTIR, t_surf)*inp.EMISSIVITY)/(transmissivity*numerical.planck(aux.WAVENUMBER_FTIR, aux.CLOUD_TEMP))
#    cemissivity_ftir = (aux.RADIANCE_FTIR - aux.RADIANCE_CLEARSKY - reflectivity*transmissivity**2*numerical.planck(aux.WAVENUMBER_FTIR, t_surf)*inp.EMISSIVITY)/(transmissivity*numerical.planck(aux.WAVENUMBER_FTIR, aux.CLOUD_TEMP))
#    return cemissivity_lbldis, cemissivity_ftir
    
    
LAT = 0.0
LON = 0.0

def create_nc(num_iter, index=-1, avk_matrix=None, errors=None, covariance_matrix=None, transfer_matrix=None, res_name=None):
    '''
    Create the netCDF file
    '''
    if not os.path.exists(inp.RESULTS):
        os.mkdir(inp.RESULTS)
        
    reflectivity, transmissivity, cemissivity_lbldis, cemissivity_ftir = physics.calculate_cloud_emissivity()
    
    #wn_emis = np.array(inp.EMISSIVITY).T[0]
    #sf_emis = np.array(inp.EMISSIVITY).T[1]
    #emis_f = interp1d(wn_emis, sf_emis, fill_value="extrapolate")
        
    #below_cloud = np.where(aux.ATMOSPHERIC_GRID['altitude(km)']*1e3 < aux.CLOUD_GRID[0])[0][0]
    #func = read_clear_sky_optical_depths(aux.LBL_WORK, below_cloud)
    #transmissivity = func(aux.WAVENUMBER_FTIR)
    #t_surf = aux.ATMOSPHERIC_GRID['temperature(K)'][0]
    #rad_semiss_075 = rL.run_lbldis(np.array([aux.MCP[-1]]), False, t_surf-5)[-1]
    #rad_semiss_025 = rL.run_lbldis(np.array([aux.MCP[-1]]), False, t_surf+5)[-1]
    #reflectivity = (rad_semiss_075 - rad_semiss_025)/(transmissivity * (numerical.planck(aux.WAVENUMBER_FTIR, t_surf-5) - numerical.planck(aux.WAVENUMBER_FTIR, t_surf+5)))
    #cemissivity_lbldis = (aux.RADIANCE_LBLDIS[0][-1][:] - aux.RADIANCE_CLEARSKY - reflectivity*transmissivity**2*numerical.planck(aux.WAVENUMBER_FTIR, t_surf)*emis_f(aux.WAVENUMBER_FTIR))/(transmissivity*numerical.planck(aux.WAVENUMBER_FTIR, aux.CLOUD_TEMP))
    #cemissivity_ftir = (aux.RADIANCE_FTIR - aux.RADIANCE_CLEARSKY - reflectivity*transmissivity**2*numerical.planck(aux.WAVENUMBER_FTIR, t_surf)*emis_f(aux.WAVENUMBER_FTIR))/(transmissivity*numerical.planck(aux.WAVENUMBER_FTIR, aux.CLOUD_TEMP))
        
    if res_name == None:
        nc_fname = "{}/results_{}.nc".format(inp.RESULTS, int(dt.datetime.strftime(aux.DATETIME, "%Y%m%d%H%M%S")))
    else:
        nc_fname = res_name
        
    if os.path.exists(nc_fname):
        os.system("rm {}".format(nc_fname))
        
    with sio.netcdf_file(nc_fname, "w") as outfile:
        outfile.createDimension("const", 1)
        outfile.createDimension("mcp", inp.MCP.size)
        outfile.createDimension("mcp_err", 2)
        outfile.createDimension("wp", 3)
        outfile.createDimension("level", len(aux.ATMOSPHERIC_GRID['altitude(km)']))
        outfile.createDimension("wavenumber", len(aux.WAVENUMBER_FTIR))
        outfile.createDimension('cgrid', len(aux.CLOUD_GRID))
        
        lat = outfile.createVariable("lat", "f8", ("const", ))
        lat.units = "deg"
        lon = outfile.createVariable("lon", "f8", ("const", ))
        lon.units = "deg"

        clevel = outfile.createVariable("clevel", "f8", ("cgrid", ))
        clevel.units = "1"
        sza = outfile.createVariable("sza", "f8", ("const", ))
        sza.units = "deg"
        
        niter = outfile.createVariable('niter', 'f8', ('const', ))
        niter.units = "1"
        
        pres = outfile.createVariable("P", "f8", ("level", ))
        pres.units = "hPa"
        temp = outfile.createVariable("T", "f8", ("level", ))
        temp.units = "K"
        humd = outfile.createVariable("humidity", "f8", ("level", ))
        humd.units = "%"
        alt = outfile.createVariable("z", "f8", ("level", ))
        alt.units = "m"
    
        wavenumber = outfile.createVariable("wavenumber", "f8", ("wavenumber",))
        wavenumber.units = "cm-1"
        cts_transmissivity = outfile.createVariable("cloud to surface transmissivity", "f8", ("wavenumber", ))
        cts_transmissivity.units = "1"
        reflectivity_out = outfile.createVariable("cloud reflectivity", "f8", ("wavenumber", ))
        reflectivity_out.units = "1"
        ftir_radiance = outfile.createVariable("ftir radiance", "f8", ("wavenumber", ))
        ftir_radiance.units = "mW * (sr*cm-1*m2)**(-1)"
        clearsky_radiance = outfile.createVariable('clearsky radiance', 'f8', ('wavenumber', ))
        clearsky_radiance.units = "mW * (sr*cm-1*m2)**(-1)"
        lbldis_radiance = outfile.createVariable("lbldis radiance", "f8", ("wavenumber", ))
        lbldis_radiance.units = "mW * (sr*cm-1*m2)**(-1)"
        cemissivity_lbldis_out = outfile.createVariable("cemissivity lbldis", "f8", ("wavenumber", ))
        cemissivity_lbldis_out.units = "1"
        cemissivity_ftir_out = outfile.createVariable("cemissivity ftir", "f8", ("wavenumber", ))
        cemissivity_ftir_out.units = "1"
        residuum = outfile.createVariable("residuum", "f8", ("wavenumber", ))
        residuum.units = "mW * (sr*cm-1*m2)**(-1)"
        rms = outfile.createVariable("Root-Mean-Square", "f8", ("const", ))
        rms.units = "mW * (sr*cm-1*m2)**(-1)"
        std = outfile.createVariable('std radiance', 'f8', ('const', ))
        std.units = 'mW * (sr*cm-1*m2)**(-1)'
        s_y_inv_out = outfile.createVariable('S_y_inv', 'f8', ('wavenumber', 'wavenumber', ))
        
        avk = outfile.createVariable("averaging kernel matrix", "f8", ("mcp", "mcp"))
        avk.units = "1"

        cov_mat = outfile.createVariable("covariance matrix", "f8", ("mcp", "mcp"))
        cov_mat.units = "1"
                
        t_mat = outfile.createVariable("transfer matrix", "f8", ("mcp", "wavenumber"))
        t_mat.units = "1"    

        red_chi2_out = outfile.createVariable("red_chi_2", "f8", ("const", ))
        red_chi2_out[:] = numerical.reduced_chi_square_test()
        log.write("{}\n".format(numerical.reduced_chi_square_test()))
        x_ret = outfile.createVariable('x_ret', 'f8', ('mcp', ))
        x_ret.units = '1'
        x_ret_err = outfile.createVariable('x_ret_err', 'f8', ('mcp', ))
        x_ret_err.units = '1'
                  
        x_a = outfile.createVariable("x_a", "f8", ("mcp", ))
        x_a.units = "1"
        x_a_err = outfile.createVariable('x_a_err', 'f8', ('mcp', ))
        x_a_err.units = '1'
    
        # Write data to NC
        reflectivity_out[:] = reflectivity[:]
        cts_transmissivity[:] = transmissivity[:]
        cemissivity_lbldis_out[:] = cemissivity_lbldis[:] 
        cemissivity_ftir_out[:] = cemissivity_ftir[:] 
        wavenumber[:] = aux.WAVENUMBER_FTIR[:]
        clearsky_radiance[:] = aux.RADIANCE_CLEARSKY[:]
        ftir_radiance[:] = aux.RADIANCE_FTIR[:]
        lbldis_radiance[:] = aux.RADIANCE_LBLDIS[0][-1][:]
        residuum[:] = list(aux.RESIDUUM[index]) 
        rms[:] = np.sqrt(np.mean(np.array(aux.RESIDUUM[index])**2))
        clevel[:] = aux.CLOUD_GRID[:]
        lat[:] = LAT
        lon[:] = LON
        sza[:] = aux.SOLAR_ZENITH_ANGLE
        pres[:] = aux.ATMOSPHERIC_GRID['pressure(hPa)'][:]
        alt[:] = aux.ATMOSPHERIC_GRID['altitude(km)'][:]*1e3
        temp[:] = aux.ATMOSPHERIC_GRID['temperature(K)'][:]
        humd[:] = aux.ATMOSPHERIC_GRID['humidity(%)'][:]
        s_y_inv_out[:] = aux.S_Y_INV_MATRIX[:]
        std[:] = np.sqrt(aux.S_Y_INV_MATRIX.item(0,0)**(-1))
        niter[:] = num_iter
        
        if type(avk_matrix) != type(None):
            avk[:] = avk_matrix[:]
        if type(covariance_matrix) != type(None):
            cov_mat[:] = covariance_matrix[:]
        if type(transfer_matrix) != type(None):
            t_mat[:] = transfer_matrix.reshape((len(inp.MCP), len(aux.WAVENUMBER_FTIR)))[:]
        
        x_a[:] = inp.MCP_APRIORI[:]
        x_ret[:] = aux.MCP[-1]
        
        x_a_err[:] = np.sqrt(np.reciprocal(np.array(inp.VARIANCE_APRIORI)))
    
        if type(errors) != type(None):
            d = np.zeros(inp.MCP.size)
            for ii in range(inp.MCP.size):
                d[ii] = np.float_(errors[ii])
                x_ret_err[:] = d
        
    return
