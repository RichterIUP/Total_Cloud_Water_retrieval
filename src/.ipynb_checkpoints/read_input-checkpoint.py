#!/usr/bin/python3
'''@package docstring
Read the case-file
'''

import os
import sys
import inp
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
import aux2 as aux
import numpy as np
import netCDF4 as nc
import datetime as dt
#import create_nc as cnc
import misc

sys.path.append(inp.PATH_TO_FTSREADER)
import ftsreader

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

def read_input_opus(opus_file, atmospheric_file, cloudnet_file, sza):
    print(opus_file)
    aux.DATETIME = dt.datetime.strptime(opus_file.split("nyem")[-1].split("_")[0], "%Y%m%d%H%M%S")

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
    misc.LAT = inp.LAT
    misc.LON = inp.LON
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


def read_input_nc(spectral_radiance_file, path_to_atmospheric_profiles, cloud_height_file, date_of_spec):

    '''
    Read informations about the spectral radiances
    '''
    with nc.Dataset(spectral_radiance_file) as rad_f:
        '''
        Find the spectrum according to the specified time
        '''

        reference_date_of_spectral_radiances = dt.datetime.strptime(rad_f.variables['time_dec'].units, "hours since %Y-%m-%d %H:%M:%S")
        hours_of_spec = (date_of_spec - reference_date_of_spectral_radiances).total_seconds()/3600.0
        time = np.round(rad_f.variables['time_dec'][:], 3)
        index_of_spec = np.where(time == np.round(hours_of_spec, 3))[0]
        if index_of_spec.size == 0:
            return False
        '''
        Read the spectrum
        '''
        aux.DATETIME           = reference_date_of_spectral_radiances + dt.timedelta(seconds=hours_of_spec*3600.0)
        misc.LAT               = np.float64(rad_f.variables['lat'][index_of_spec])
        misc.LON               = np.float64(rad_f.variables['lon'][index_of_spec])
        aux.SOLAR_ZENITH_ANGLE = np.float64(rad_f.variables['sza'][index_of_spec])
        aux.WAVENUMBER_FTIR    = rad_f.variables['wavenumber'][index_of_spec][:].flatten()
        aux.RADIANCE_FTIR      = rad_f.variables['radiance'][index_of_spec][:].flatten()
        aux.NOISE_FTIR         = rad_f.variables['stdDev'][index_of_spec][:].flatten()
        
        if aux.SOLAR_ZENITH_ANGLE > 90.0:
            aux.SOLAR_ZENITH_ANGLE = -1
            
        if inp.RECALIBRATE: 
            ## Load emissivity of blackbody
            alpha = pd.read_csv(inp.FILE_EMISS)
            
            ## Create Interpolation function
            emiss_f = interp1d(np.array(alpha['wavenumber']), np.array(alpha['emissivity']), fill_value="extrapolate")
            
            ## Load temperature of laboratory            
            temp_lab = pd.read_csv(inp.FILE_TEMP, parse_dates=[1])
            idx = np.where((temp_lab['time'] > aux.DATETIME) & (temp_lab['time'] < aux.DATETIME + dt.timedelta(minutes=10)))[0][0]
            t_lab = temp_lab['temperature(K)'].iloc[idx]
            h = 6.62607015e-34
            c = 299792458
            k = 1.380649e-23
            planck = lambda T, v: 1e3 * 2 * 10**8 * h * c**2 * v**3 / (np.exp(100*v*h*c/(k*T))-1)
            aux.RADIANCE_FTIR = emiss_f(aux.WAVENUMBER_FTIR) * aux.RADIANCE_FTIR + (1 - emiss_f(aux.WAVENUMBER_FTIR)) * planck(t_lab, aux.WAVENUMBER_FTIR)

        
    '''
    Average radiance on windows
    '''
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
    #plt.plot(aux.WAVENUMBER_FTIR[idx_noise], noise, label=np.std(noise))
    #plt.legend()
    #plt.grid(True)
    #plt.savefig("noise.png")
    #exit(-1)
    aux.WAVENUMBER_FTIR = wavenumber_av
    aux.RADIANCE_FTIR   = radiance_av+inp.OFFSET
    aux.NOISE_FTIR      = np.std(noise)*np.ones(wavenumber_av.size)#noise_av
    #f = open("noise", "a")
    #f.write("{} {}\n".format(np.mean(aux.NOISE_FTIR), np.mean(noise_av)))
    #f.close()
    #return False

    '''
    Read cloud height informations
    '''
    date_of_spec_no_sec = dt.datetime(date_of_spec.year, date_of_spec.month, date_of_spec.day, date_of_spec.hour, date_of_spec.minute)
    if type(inp.CLOUD_LAYERS) == type(None):#
        with nc.Dataset(cloud_height_file) as cld_f:
            '''
            Find cloud information corresponding to the time
            '''
            reference_date_of_cloud_height = dt.datetime.strptime(cld_f.variables['time'].units, "Hours since %Y-%m-%d %H:%M")
            hours_of_cld = (date_of_spec_no_sec - reference_date_of_cloud_height).total_seconds()/3600.0
            time = np.round(cld_f.variables['time'][:], 3)
            index_of_cld = np.where(time == np.round(hours_of_cld, 3))[0]
            if index_of_cld.size == 0:
                return False
            levels = cld_f.variables['levels'][:]
            layers = np.where(cld_f.variables['layers_merged'][index_of_cld][0] != 0)[0]
            if inp.ONLY_CLOUD_BASE:
                aux.CLOUD_LAYERS = [levels[layers[0]]]
            elif inp.ONLY_CLOUD_TOP:
                aux.CLOUD_LAYERS = [levels[layers[1]]]
            else:
                aux.CLOUD_LAYERS = levels[layers]
                
    else:
        aux.CLOUD_LAYERS = inp.CLOUD_LAYERS

    '''
    Read atmospheric profile
    '''
    atmospheric_profiles = np.array(sorted(os.listdir(path_to_atmospheric_profiles)))
    if len(atmospheric_profiles) > 1:
        time_diff = []
        time_atm  = []
        for file_ in atmospheric_profiles:
            time_diff.append(np.abs((dt.datetime.strptime(file_, "prof%Y%m%d_%H%M.nc")-date_of_spec).total_seconds()))
            time_atm.append((dt.datetime.strptime(file_, "prof%Y%m%d_%H%M.nc")-date_of_spec).total_seconds())
        time_diff = np.array(time_diff)
        time_atm  = np.array(time_atm)
        '''
        Find the two atmospheric profile nearest the desired time
        '''
        idx = [list(time_diff).index(np.min(time_diff)), -999]

        if idx[0] + 1 == len(time_diff):
            idx[1] = idx[0] - 1
        elif idx[0] == 0:
            idx[1] = 1
        elif time_diff[idx[0] + 1] > time_diff[idx[0] - 1]:
            idx[1] = idx[0] - 1
        else:
            idx[1] = idx[0] + 1
        idx = np.sort(idx)
        atmospheric_profiles = atmospheric_profiles[idx]
    else:
        atmospheric_profiles = [atmospheric_profiles[0], atmospheric_profiles[0]]
        time_atm = np.array([(dt.datetime.strptime(atmospheric_profiles[0], "prof%Y%m%d_%H%M.nc")-date_of_spec).total_seconds(), \
                            (dt.datetime.strptime(atmospheric_profiles[0], "prof%Y%m%d_%H%M.nc")-date_of_spec).total_seconds()])
        idx = np.array([0, 1])
        
    '''
    Read profiles
    '''

    reduced_altitude = np.loadtxt(inp.ALTITUDE_GRID, delimiter=",")
    
    altitude    = np.array([None, None])
    pressure    = np.array([None, None])
    temperature = np.array([None, None])
    humidity    = np.array([None, None])

    for ii in range(len(atmospheric_profiles)):
        with nc.Dataset("{}/{}".format(path_to_atmospheric_profiles, atmospheric_profiles[ii])) as atm_f:
            altitude_pre_int = atm_f.variables['z'][:]
            pressure_pre_int = atm_f.variables['P'][:]
            temperature_pre_int = atm_f.variables['T'][:]
            humidity_pre_int = atm_f.variables['rh'][:]

            pressure[ii]    = np.interp(reduced_altitude, altitude_pre_int, pressure_pre_int)
            temperature[ii] = np.interp(reduced_altitude, altitude_pre_int, temperature_pre_int)
            humidity[ii]    = np.interp(reduced_altitude, altitude_pre_int, humidity_pre_int)
            altitude[ii]    = reduced_altitude

    size_atm = len(altitude[0])
    altitude_atm = np.zeros(size_atm)
    pressure_atm = np.zeros(size_atm)
    temperature_atm = np.zeros(size_atm)
    humidity_atm = np.zeros(size_atm)    
    for ii in range(size_atm):
        altitude_atm[ii]    = np.interp(0, time_atm[idx], np.array([altitude[0][ii], altitude[1][ii]]))
        pressure_atm[ii]    = np.interp(0, time_atm[idx], np.array([pressure[0][ii], pressure[1][ii]]))
        temperature_atm[ii] = np.interp(0, time_atm[idx], np.array([temperature[0][ii], temperature[1][ii]]))
        humidity_atm[ii]    = np.interp(0, time_atm[idx], np.array([humidity[0][ii], humidity[1][ii]]))

    aux.ATMOSPHERIC_GRID = {'pressure(hPa)': pressure_atm, \
                            'altitude(km)': altitude_atm, \
                            'temperature(K)' : temperature_atm, \
                            'humidity(%)' : humidity_atm*inp.SCALE_HUMIDITY}
    return True
