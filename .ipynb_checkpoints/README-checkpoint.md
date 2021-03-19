# How to install and run TCWret

## 1. Download source codes:

- [LBLRTM](http://rtweb.aer.com/lblrtm.html)
- [LBLDIS](https://web.archive.org/web/20170508194542/http://www.nssl.noaa.gov/users/dturner/public_html/lbldis/index.html)

### Additional files

- [Additional Single scattering databases](https://web.archive.org/web/20170516023452/http://www.nssl.noaa.gov/users/dturner/public_html/lbldis/ADDITIONAL_INFO.html) 


## 2. Set up virtual environment for Python 3 

If you don't want to 'contaminate' your local Python, you can create a virtual environment

```sh
> sudo apt-get install python3-venv
> python3 -m venv path/to/python/virtenv
> source path/to/python/virtenv/bin/activate
> path/to/python/virtenv/bin/pip3 install numpy
> path/to/python/virtenv/bin/pip3 install scipy
> path/to/python/virtenv/bin/pip3 install matplotlib
> path/to/python/virtenv/bin/pip3 install netcdf4
```

## 3. Inputfile of TCWret:

Modify variables in inp.py.bck according to your measurements and inputs. Copy to the run directory and rename to inp.py

## 4. Run TCWret:

Go to TCWret and run TCWret_retrieval.py. TCWret_retrieval.py works on real data described below.

```sh
python3 TCWret.py <spectral_radiance_file> <path_to_atmosphere> <cloud_height_file> <sza>
```
If atmospheric input is RASO, then <path_to_atmosphere> must be the path to the netCDF4 files. If it is ERA5, then <path_to_atmosphere> must be the path to the ERA5 file. If radiance input is from NC, sza will be ignored. Otherwise TCWret will read sza from command line. Setting sza to negative values deactivates solar input.
References: 
- [Atmospheric profiles are from radiosondes of PS106.1](https://doi.pangaea.de/10.1594/PANGAEA.882736)
- [Cloud height informations are from Vaisala CL51 Ceilometer mounted on the RV Polarstern](https://doi.org/10.1594/PANGAEA.883320)

## 5. Format of TCWret_retrieval input (OPTION 'NC'):

In sample_run are example files which can be used as template to build own input files.

## 6. Output

TCWret provides the output in the netCDF4 format.
