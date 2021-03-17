#!/usr/bin/python3
'''@package docstring
Provide a number of functions for handling input and output
'''

import sys
import os
import datetime as dt
import numpy as np
import netCDF4 as nc
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import inp

sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
import numerical
import physics

try:
    sys.path.append(inp.PATH_TO_FTSREADER)
    import ftsreader
except ImportError:
    print("Unable to load ftsreader! Cannot use OPUS spectra")

DATETIME = None
LBL_WORK = ''
FNAME = ''
LAT = 0.0
LON = 0.0


def read_era5_argmin(fname, lat, lon, tim, key, time_unit):
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
    tim = (tim - dt.datetime(time_unit[0], 1, 1)).total_seconds()/time_unit[1]
    with nc.Dataset(fname, "r") as file_:
        try:
            plev = file_.variables['level'][:]
            plev_units = file_.variables['level'].units
            latitude = file_.variables['latitude'][:]
            longitude = file_.variables['longitude'][:]
        except KeyError:
            plev = file_.variables['plev'][:]
            plev_units = file_.variables['plev'].units
            latitude = file_.variables['lat'][:]
            longitude = file_.variables['lon'][:]
        time = file_.variables['time'][:]
        return_val = file_.variables[key][:]
    idx_lat = np.abs(latitude - lat).argmin()
    idx_lon = np.abs(longitude - lon).argmin()
    idx_time = np.abs(time - tim).argmin()
    try:
        profile = return_val[idx_time, :, idx_lat, idx_lon]
    except IndexError:
        profile = return_val[0, idx_time, :, idx_lat, idx_lon]

    # Invert order of arrays, if pressure is increasing
    if plev[0] < plev[-1]:
        profile = profile[::-1]
        plev = plev[::-1]

    ## Pressure unit must be hPa
    if plev_units == "Pa":
        units = 1e-2
    elif plev_units in ("hPa", "millibars", "mb"):
        units = 1
    return profile, plev*units

def read_radiances(spectral_radiance_file, date_of_spec, sza):
    '''

    Read radiances either from OPUS file or netCDF4 file
    Parameters
    ----------
    spectral_radiance_file : str
        Name of file containing spectral radiances

    date_of_spec : dt.datetime
        Date of spectrum. Only read if inp.RAD_INP == "NC"

    sza : float
        Solar Zenith Angle. Only read if inp.RAD_INP = "OPUS"
    '''
    global DATETIME, LAT, LON

    if inp.RAD_INP == 'OPUS':
        DATETIME = dt.datetime.strptime(spectral_radiance_file.split("nyem")[-1].split("_")[0], \
                                        "%Y%m%d%H%M%S")

        ## Read radiances and wavenumber from OPUS file
        opus_spec = ftsreader.ftsreader(spectral_radiance_file, \
                                        verbose=True, getspc=True, getifg=True)
        physics.WAVENUMBER_FTIR = np.array(opus_spec.spcwvn)
        physics.RADIANCE_FTIR = np.array(opus_spec.spc)*1e3
        LAT = inp.LAT
        LON = inp.LON
        physics.SOLAR_ZENITH_ANGLE = sza

    else:
        with nc.Dataset(spectral_radiance_file) as rad_f:
            # Find spectrum corresponding to the specified time
            reference_date_of_spectral_radiances = \
                dt.datetime.strptime(rad_f.variables['time_dec'].units, \
                                     "hours since %Y-%m-%d %H:%M:%S")
            hours_of_spec = (date_of_spec - reference_date_of_spectral_radiances).total_seconds() \
                / 3600.0
            time = np.round(rad_f.variables['time_dec'][:], 3)
            index_of_spec = np.where(time == np.round(hours_of_spec, 3))[0]
            if index_of_spec.size == 0:
                raise IndexError("No spectrum!")

            # Read spectrum
            DATETIME = reference_date_of_spectral_radiances + dt.timedelta(seconds=hours_of_spec * \
                                                                           3600.0)
            LAT = np.float64(rad_f.variables['lat'][index_of_spec])
            LON = np.float64(rad_f.variables['lon'][index_of_spec])
            physics.SOLAR_ZENITH_ANGLE = np.float64(rad_f.variables['sza'][index_of_spec])
            physics.WAVENUMBER_FTIR = rad_f.variables['wavenumber'][index_of_spec][:].flatten()
            physics.RADIANCE_FTIR = rad_f.variables['radiance'][index_of_spec][:].flatten()
            physics.NOISE_FTIR = rad_f.variables['stdDev'][index_of_spec][:].flatten()

            if physics.SOLAR_ZENITH_ANGLE > 90.0:
                physics.SOLAR_ZENITH_ANGLE = -1

    if inp.RECALIBRATE:
        # Load emissivity of blackbody
        alpha = pd.read_csv(inp.FILE_EMISS)
        # Create Interpolation function
        emiss_f = interp1d(np.array(alpha['wavenumber']), np.array(alpha['emissivity']), \
                           fill_value="extrapolate")

        # Load temperature of laboratory
        temp_lab = pd.read_csv(inp.FILE_TEMP, parse_dates=[1])
        idx = np.where((temp_lab['time'] > DATETIME) & (temp_lab['time'] < DATETIME + \
                                                        dt.timedelta(minutes=10)))[0][0]
        t_lab = temp_lab['temperature(K)'].iloc[idx]
        planck = lambda T, v: 1e3 * 2 * 10**8 * physics.H * physics.C**2 * v**3 / \
            (np.exp(100*v*physics.H*physics.C/(physics.K*T))-1)
        physics.RADIANCE_FTIR = emiss_f(physics.WAVENUMBER_FTIR) * physics.RADIANCE_FTIR + \
            (1 - emiss_f(physics.WAVENUMBER_FTIR)) * planck(t_lab, physics.WAVENUMBER_FTIR)


    # Average radiance on windows
    num_windows = len(physics.MICROWINDOWS)
    wavenumber_av = np.zeros(num_windows)
    radiance_av = np.zeros(num_windows)
    noise_av = np.zeros(num_windows)

    for window in range(num_windows):
        idx_window = np.where((physics.WAVENUMBER_FTIR >= physics.MICROWINDOWS[window][0]) & \
                              (physics.WAVENUMBER_FTIR <= physics.MICROWINDOWS[window][1]))
        radiance_av[window] = np.mean(physics.RADIANCE_FTIR[idx_window])
        noise_av[window] = np.std(physics.RADIANCE_FTIR[idx_window])
        wavenumber_av[window] = np.mean(physics.WAVENUMBER_FTIR[idx_window])


    idx_noise = np.where((physics.WAVENUMBER_FTIR > 1925) & (physics.WAVENUMBER_FTIR < 2000))[0]
    func = lambda x, a, b: a * x + b
    popt = curve_fit(func, physics.WAVENUMBER_FTIR[idx_noise], physics.RADIANCE_FTIR[idx_noise])[0]
    noise = physics.RADIANCE_FTIR[idx_noise] - func(physics.WAVENUMBER_FTIR[idx_noise], \
                                                    popt[0], popt[1])

    physics.WAVENUMBER_FTIR = wavenumber_av
    physics.RADIANCE_FTIR = radiance_av+inp.OFFSET
    physics.NOISE_FTIR = np.std(noise)*np.ones(wavenumber_av.size)

def create_atmosphere(atmospheric_file):
    '''
    Read atmospheric profiles of temperature, pressure and relative humidity

    Parameters
    ----------
    atmospheric_file : str
        Name of file containing atmospheric profile
    '''
    if inp.ATM_INP == "ERA5":
        ## Set the time according to ERA5 time grid
        time_era5 = dt.datetime(DATETIME.year, DATETIME.month, DATETIME.day, DATETIME.hour)
        if DATETIME.minute != 0:
            time_era5 = time_era5 + dt.timedelta(hours=1)
        relative_humidity = read_era5_argmin(atmospheric_file, \
                                             LAT, LON, time_era5, inp.KEY_RH, inp.TIME_UNIT)[0]
        height = read_era5_argmin(atmospheric_file, \
                                  LAT, LON, time_era5, inp.KEY_HEIGHT, inp.TIME_UNIT)[0]/physics.G0
        temperature, plev = read_era5_argmin(atmospheric_file, \
                                             LAT, LON, time_era5, inp.KEY_TEMP, inp.TIME_UNIT)

        ## Add a surface layer at height 18
        humd_f = interp1d(height, relative_humidity, fill_value="extrapolate", kind="cubic")
        temp_f = interp1d(height, temperature, fill_value="extrapolate", kind="cubic")
        plev_f = interp1d(height, plev, fill_value="extrapolate", kind="cubic")
        height = np.concatenate(([18], height))
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

    elif inp.ATM_INP == "RASO":
        atmospheric_profiles = np.array(sorted(os.listdir(atmospheric_file)))
        if len(atmospheric_profiles) > 1:
            time_diff = []
            time_atm = []
            for file_ in atmospheric_profiles:
                time_diff.append(np.abs((dt.datetime.strptime(file_, "prof%Y%m%d_%H%M.nc") - \
                                         DATETIME).total_seconds()))
                time_atm.append((dt.datetime.strptime(file_, "prof%Y%m%d_%H%M.nc") - \
                                 DATETIME).total_seconds())
            time_diff = np.array(time_diff)
            time_atm = np.array(time_atm)
            # Find atmospheric profile nearest the desired time.
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
            time_atm = np.array([(dt.datetime.strptime(atmospheric_profiles[0], \
                                                       "prof%Y%m%d_%H%M.nc") - \
                                  DATETIME).total_seconds(), \
                                 (dt.datetime.strptime(atmospheric_profiles[0], \
                                                       "prof%Y%m%d_%H%M.nc") - \
                                  DATETIME).total_seconds()])
            idx = np.array([0, 1])


        # Read atmospheric profiles
        reduced_altitude = np.loadtxt(inp.ALTITUDE_GRID, delimiter=",")

        altitude_atm = np.array([None, None])
        pressure_atm = np.array([None, None])
        temperature_atm = np.array([None, None])
        humidity_atm = np.array([None, None])

        for i, dummy in enumerate(atmospheric_profiles, start=0):# range(len(atmospheric_profiles)):
            with nc.Dataset("{}/{}".format(atmospheric_file, atmospheric_profiles[i])) as atm_f:
                altitude_pre_int = atm_f.variables['z'][:]
                pressure_pre_int = atm_f.variables['P'][:]
                temperature_pre_int = atm_f.variables['T'][:]
                humidity_pre_int = atm_f.variables['rh'][:]

                pressure_atm[i] = np.interp(reduced_altitude, \
                                                altitude_pre_int, pressure_pre_int)
                temperature_atm[i] = np.interp(reduced_altitude, \
                                                altitude_pre_int, temperature_pre_int)
                humidity_atm[i] = np.interp(reduced_altitude, \
                                                altitude_pre_int, humidity_pre_int)
                altitude_atm[i] = reduced_altitude

        size_atm = len(altitude_atm[0])
        height = np.zeros(size_atm)
        plev = np.zeros(size_atm)
        temperature = np.zeros(size_atm)
        relative_humidity = np.zeros(size_atm)
        for i in range(size_atm):
            height[i] = np.interp(0, time_atm[idx], \
                                   np.array([altitude_atm[0][i], altitude_atm[1][i]]))
            plev[i] = np.interp(0, time_atm[idx], \
                                 np.array([pressure_atm[0][i], pressure_atm[1][i]]))
            temperature[i] = np.interp(0, time_atm[idx], \
                                        np.array([temperature_atm[0][i], temperature_atm[1][i]]))
            relative_humidity[i] = np.interp(0, time_atm[idx], \
                                              np.array([humidity_atm[0][i], humidity_atm[1][i]]))
        height *= 1e3

    physics.ATMOSPHERIC_GRID = {'pressure(hPa)' : plev, \
                        'altitude(m)': height, \
                        'altitude(km)': height*1e-3, \
                        'temperature(K)': temperature, \
                        'humidity(%)': relative_humidity}

def read_cloud_position(cloud_height_file):
    '''

    Read position of cloud. If no cloud information is available, read cloud information
    from inp.py

    Parameters
    ----------
    cloud_height_file:
        Name of file containing cloud height informations
    '''
    if inp.CLD_INP == "ASCII":
        if os.path.exists(cloud_height_file):
            physics.CLOUD_LAYERS = np.array([np.loadtxt(cloud_height_file)]).flatten()
        else:
            physics.CLOUD_LAYERS = inp.CLOUD_LAYERS

        if len(physics.CLOUD_LAYERS) == 0:
            raise IndexError("No Cloud!")
    elif inp.CLD_INP == "NC":
        # Read cloud height informations
        date_of_spec_no_sec = DATETIME
        if not isinstance(inp.CLOUD_LAYERS, (list)):#type(inp.CLOUD_LAYERS) == type(None):
            with nc.Dataset(cloud_height_file) as cld_f:
                # Find cloud information corresponding to the time
                reference_date_of_cloud_height = \
                    dt.datetime.strptime(cld_f.variables['time'].units, \
                                         "Hours since %Y-%m-%d %H:%M")
                hours_of_cld = (date_of_spec_no_sec - \
                                reference_date_of_cloud_height).total_seconds()/3600.0
                time = np.round(cld_f.variables['time'][:], 3)
                index_of_cld = np.where(time == np.round(hours_of_cld, 3))[0]
                if index_of_cld.size == 0:
                    raise IndexError("No Cloud!")
                levels = cld_f.variables['levels'][:]
                layers = np.where(cld_f.variables['layers_merged'][index_of_cld][0] != 0)[0]
                if inp.ONLY_CLOUD_BASE:
                    physics.CLOUD_LAYERS = [levels[layers[0]]]
                elif inp.ONLY_CLOUD_TOP:
                    physics.CLOUD_LAYERS = [levels[layers[1]]]
                else:
                    physics.CLOUD_LAYERS = levels[layers]

        else:
            physics.CLOUD_LAYERS = inp.CLOUD_LAYERS

def log_prog_start():
    '''
    Initialise logfile
    '''
    with open("{}/retrieval_log.dat".format(inp.PATH), "a") as file_:
        file_.write("\n\n#########################################\n")
        file_.write("# TCWret\n")
        file_.write("#\n")
        file_.write("# Spec: {}\n".format(dt.datetime.strftime(DATETIME, "%Y-%m-%dT%H:%M:%S")))
        for i in range(len(physics.MICROWINDOWS)):
            file_.write("# Microwindow: {}\n".format(physics.MICROWINDOWS[i]))
        for element in physics.CLOUD_GRID:
            file_.write("# Cloud layer: {}\n".format(element))
        file_.write("# Cloud Temperature: {}\n".format(physics.CLOUD_TEMP))
        file_.write("#########################################\n\n")

def write(text):
    '''
    Write text to retrieval_log.dat

    Parameters
    ----------
    text : str
        Text to be written to retrieval_log.dat
    '''
    with open("{}/retrieval_log.dat".format(inp.PATH), "a") as file_:
        file_.write("{}\n".format(text))

def create_nc(num_iter, avk_matrix, errors, covariance_matrix, transfer_matrix):
    '''
    Create the netCDF file containing the inputs and results of the retrieval

    Parameters
    ----------
    num_iter : int
        Number of iterations

    avk_matrix : np.matrix
        Averaging Kernel Matrix

    errors : np.array
        Standard deviations read from the covariance matrix

    covariance_matrix : np.matrix
        Covariance matrix of the final state vector

    transfer_matrix : np.matrix
        Matrix which propagates covariances from the measurement to the final state vector
    '''
    if not os.path.exists(inp.RESULTS):
        os.mkdir(inp.RESULTS)

    reflectivity, transmissivity, cemissivity_lbldis, cemissivity_ftir = \
        physics.calculate_cloud_emissivity()

    nc_fname = "{}/results_{}.nc".format(inp.RESULTS, \
                                         int(dt.datetime.strftime(DATETIME, "%Y%m%d%H%M%S")))

    if os.path.exists(nc_fname):
        os.system("rm {}".format(nc_fname))

    with nc.Dataset(nc_fname, "w") as outfile:
        outfile.createDimension("const", 1)
        outfile.createDimension("mcp", inp.MCP.size)
        outfile.createDimension("mcp_err", 2)
        outfile.createDimension("wp", 3)
        outfile.createDimension("level", len(physics.ATMOSPHERIC_GRID['altitude(km)']))
        outfile.createDimension("wavenumber", len(physics.WAVENUMBER_FTIR))
        outfile.createDimension('cgrid', len(physics.CLOUD_GRID))

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
        cts_transmissivity = outfile.createVariable("cloud to surface transmissivity", \
                                                    "f8", ("wavenumber", ))
        cts_transmissivity.units = "1"
        reflectivity_out = outfile.createVariable("cloud reflectivity", "f8", ("wavenumber", ))
        reflectivity_out.units = "1"
        ftir_radiance = outfile.createVariable("ftir radiance", "f8", ("wavenumber", ))
        ftir_radiance.units = "mW * (sr*cm-1*m2)**(-1)"
        clearsky_radiance = outfile.createVariable('clearsky radiance', 'f8', ('wavenumber', ))
        clearsky_radiance.units = "mW * (sr*cm-1*m2)**(-1)"
        lbldis_radiance = outfile.createVariable("lbldis radiance", "f8", ("wavenumber", ))
        lbldis_radiance.units = "mW * (sr*cm-1*m2)**(-1)"
        cemissivity_lbldis_out = outfile.createVariable("cemissivity lbldis", "f8", \
                                                        ("wavenumber", ))
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
        x_ret = outfile.createVariable('x_ret', 'f8', ('mcp', ))
        x_ret.units = '1'
        x_ret_err = outfile.createVariable('x_ret_err', 'f8', ('mcp', ))
        x_ret_err.units = '1'

        x_a = outfile.createVariable("x_a", "f8", ("mcp", ))
        x_a.units = "1"
        x_a_err = outfile.createVariable('x_a_err', 'f8', ('mcp', ))
        x_a_err.units = '1'

        reflectivity_out[:] = reflectivity[:]
        cts_transmissivity[:] = transmissivity[:]
        cemissivity_lbldis_out[:] = cemissivity_lbldis[:]
        cemissivity_ftir_out[:] = cemissivity_ftir[:]
        wavenumber[:] = physics.WAVENUMBER_FTIR[:]
        clearsky_radiance[:] = physics.RADIANCE_CLEARSKY[:]
        ftir_radiance[:] = physics.RADIANCE_FTIR[:]
        lbldis_radiance[:] = physics.RADIANCE_LBLDIS[0][-1][:]
        residuum[:] = list(physics.RESIDUUM[-1])
        rms[:] = np.sqrt(np.mean(np.array(physics.RESIDUUM[-1])**2))
        clevel[:] = physics.CLOUD_GRID[:]
        lat[:] = LAT
        lon[:] = LON
        sza[:] = physics.SOLAR_ZENITH_ANGLE
        pres[:] = physics.ATMOSPHERIC_GRID['pressure(hPa)'][:]
        alt[:] = physics.ATMOSPHERIC_GRID['altitude(km)'][:]*1e3
        temp[:] = physics.ATMOSPHERIC_GRID['temperature(K)'][:]
        humd[:] = physics.ATMOSPHERIC_GRID['humidity(%)'][:]
        s_y_inv_out[:] = numerical.S_Y_INV_MATRIX[:]
        std[:] = np.sqrt(numerical.S_Y_INV_MATRIX.item(0, 0)**(-1))
        niter[:] = num_iter

        avk[:] = avk_matrix[:]
        cov_mat[:] = covariance_matrix[:]
        t_mat[:] = transfer_matrix.reshape((len(inp.MCP), len(physics.WAVENUMBER_FTIR)))[:]

        x_a[:] = inp.MCP_APRIORI[:]
        x_ret[:] = physics.MCP[-1]

        x_a_err[:] = np.sqrt(np.reciprocal(np.array(inp.VARIANCE_APRIORI)))
        err = np.zeros(inp.MCP.size)
        for i in range(inp.MCP.size):
            err[i] = np.float_(errors[i])
            x_ret_err[:] = err
