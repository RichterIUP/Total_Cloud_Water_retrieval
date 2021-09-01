#!/usr/bin/python
'''@package docstring
Entrance point for TCWret. Prepare the data and call the iteration
'''
import sys
import os
import datetime as dt
import netCDF4 as nc

sys.path.append(os.getcwd())
import inp
sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
import inversion
import tcwret_io
import physics

def get_times(radiance_file):
    '''
    Read time information from a netCDF4 and return array of datetime objects

    Parameters
    ----------
    radiance_file : str
        Name of a netCDF4-file containing spectral radiances.

    Returns
    -------
    np.array(dt.datetime)
        Array containing datetime objects corresponding to the times of the spectra.

    '''
    with nc.Dataset(radiance_file) as rad_f:
        reference_date_of_spectral_radiances = \
            dt.datetime.strptime(rad_f.variables['time_dec'].units, "hours since %Y-%m-%d %H:%M:%S")
        time = [None for i in range(len(rad_f.variables['time_dec'][:]))]
        for i in range(len(rad_f.variables['time_dec'][:])):
            time[i] = reference_date_of_spectral_radiances + \
                dt.timedelta(seconds=int(3600.0*rad_f.variables['time_dec'][i]))

    return time

def main(spectral_radiance_file, \
         atmospheric_profiles, \
         cloud_height_file, \
         date_of_spec=None, sza=-1):
    '''
    Entrance function for the retrieval. Read microwindows, creates directories, calls the
    retrieval loop and calls save function

    Parameters
    ----------
    spectral_radiance_file: str
        Name of a netCDF4-file containing spectral radiances.

    atmospheric_profiles: str
        Path to directory containing atmospheric profiles (inp.FORMAT == "NC") or name of
        a netCDF4-file containing ERA5 profiles (inp.FORMAT == "OPUS")

    cloud_height_file: str
        Name of a netCDF4-file (inp.FORMAT == "NC") or ASCII-file (inp.FORMAT=="OPUS") containing
        informations of the cloud position

    date_of_spec: dt.datetime
        Date of spectrum. Only used if inp.FORMAT == "NC"

    sza: float
        Solar Zenith Angle of measurement in degrees. Only used if inp.FORMAT == "OPUS"

    '''

    # Store initial MCP in a local variable. This prevents overwriting in case of multiple runs
    # in one call of TCWret (if more than one spectrum is in spectral_radiance_file)
    mcp = inp.MCP

    with open(inp.WINDOWS, "r") as file_:
        m_window = file_.readlines()
        physics.MICROWINDOWS = [[] for ii in range(int(m_window[0]))]
        for line in range(1, len(physics.MICROWINDOWS)+1):
            physics.MICROWINDOWS[line-1] = [float(m_window[line].split(" ")[0]), \
                                        float(m_window[line].split(" ")[1])]

    # Load spectrum, atmospheric profile and cloud information
    #try:
    if True:
        tcwret_io.read_radiances(spectral_radiance_file, date_of_spec, sza)
        tcwret_io.create_atmosphere(atmospheric_profiles)
        tcwret_io.read_cloud_position(cloud_height_file)
    #except IndexError:
    #    print("Fatal. Loading input failed")
    #    return
    directory = dt.datetime.strftime(dt.datetime.now(), "%m_%d_%H_%M_%S_%f")
    path = inp.PATH[:]

    # Create all the necessary folders
    ftir = spectral_radiance_file.split("/")[-1].split(".nc")[0]
    if not os.path.exists("{}".format(inp.PATH)):
        os.mkdir("{}".format(inp.PATH))
    if not os.path.exists("{}/{}".format(inp.PATH, ftir)):
        os.mkdir("{}/{}".format(inp.PATH, ftir))
    inp.PATH = "{}/{}/{}".format(inp.PATH, ftir, directory)
    if not os.path.exists("{}".format(inp.PATH)):
        os.mkdir("{}".format(inp.PATH))


    inversion.set_up_retrieval()

    try:
        # Run retrieval loop and save results
        success = inversion.retrieve()
        tcwret_io.create_nc(num_iter=success[0], \
                            avk_matrix=success[1], \
                            errors=success[2], \
                            covariance_matrix=success[3], \
                            transfer_matrix=success[4])
    except FileNotFoundError:
        print("FileNotFoundError: {}".format(tcwret_io.DATETIME))
    except MemoryError:#IndexError:
        print("IndexError: {}".format(tcwret_io.DATETIME))
    finally:
        # Delete lblrtm output and restore path and mcp
        os.system("rm -rf {}/.lblrtm*".format(inp.PATH))
        inp.PATH = path
        inp.MCP = mcp
    return

if __name__ == '__main__':
    if len(sys.argv) > 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sza=float(sys.argv[4]))
    else:
        for date in get_times(sys.argv[1]):
            #try:
            main(sys.argv[1], sys.argv[2], sys.argv[3], date_of_spec=date)
            #except Exception:
            #    pass
