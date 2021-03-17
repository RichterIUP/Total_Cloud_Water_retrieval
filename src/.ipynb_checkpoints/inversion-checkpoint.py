#!/usr/bin/python
'''@package docstring
Perform the inversion
'''
import sys
import os
import datetime as dt
import numpy as np

import inp
sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))
import numerical
import physics
import tcwret_io
import run_lbldis as rL


def parameter_out_of_bounds(mcp):
    '''
    Check if the current effective radius is outside the interval specified in inp.py

    Parameters
    ----------
    mcp : np.array(float)
        Microphysical cloud parameters

    Returns
    -------
    bool
        True of outside the interval. False otherwise

    '''
    for i in range(len(mcp)//2):
        if mcp[i+mcp.size//2] < inp.MIN_REFF or mcp[i+mcp.size//2] > inp.MAX_REFF:
            return True
    return False

def retrieve_step(lm_param, loop_count):
    '''
    Calculate the next parameters using Least Squares with Gauss-Newton/Levenberg-Marquardt.
    If the prior cost function is better than the current one, the current elements of the
    lists are deleted. Otherwise the new values are appended

    Parameters
    ----------

    lm_param : float
        Levenberg-Marquardt parameter nu
    loop_count : int
        Iteration counter

    Returns
    -------
    float
        New Levenberg-Marquardt parameter
    '''

    # Calculate cost function and residuum
    chi2, residuum = numerical.calc_chi_2_and_residuum()
    tcwret_io.write("# Current X^2: {}".format(chi2))

    # Test if the new cost function is less than the previous one. If so, decrease the Levenberg-
    # Marquardt parameter and save new radiances. Otherwise, reject new radiances and increase
    # the Levenberg-Marquadt parameter
    if loop_count > 0:
        tcwret_io.write("# Prev X^2: {}".format(numerical.CHI2[-1]))
    if loop_count == 0 or chi2 <= numerical.CHI2[-1]:
        numerical.CHI2.append(chi2)
        physics.RESIDUUM.append(residuum)
        if lm_param / inp.DECREASE_LM > inp.LM_MIN:
            lm_param = lm_param / inp.DECREASE_LM

    elif chi2 > numerical.CHI2[-1]:
        lm_param = lm_param * inp.INCREASE_LM
        if lm_param == 0.0 and inp.LM_INIT > 0.0:
            lm_param = inp.LM_INIT
        elif lm_param == 0.0 and inp.LM_INIT == 0.0:
            lm_param = 1.0
        numerical.CHI2.append(numerical.CHI2[-1])
        physics.RESIDUUM.append(physics.RESIDUUM[-1])
        for num_iter in range(inp.MCP.size+1):
            physics.RADIANCE_LBLDIS[num_iter][-1] = physics.RADIANCE_LBLDIS[num_iter][-2]
        physics.MCP[-1] = physics.MCP[-2]
        numerical.T_MATRIX[-1] = numerical.T_MATRIX[-2]

    # Calculate the adjustment vector
    delta = numerical.iteration(lm_param, numerical.T_MATRIX[-1])
    s_n = delta[0]
    t_matrix_new = delta[1]
    this_mcp = physics.MCP[-1] + s_n

    # If the optical depth is below zero, set it to zero
    for i in range(len(inp.MCP)//2):
        if this_mcp[i] < 0.0:
            this_mcp[i] = 0.0

    # If the parameter for the reff is above MAX_REFF or below MIN_REFF, repeat the
    # calculation s_n with different LM_PARAM
    while parameter_out_of_bounds(this_mcp):
        lm_param = lm_param * inp.INCREASE_LM
        if lm_param == 0.0 and inp.LM_INIT > 0.0:
            lm_param = inp.LM_INIT
        elif lm_param == 0.0 and inp.LM_INIT == 0.0:
            lm_param = 1.0
        delta = numerical.iteration(lm_param, numerical.T_MATRIX[-1])
        s_n = delta[0]
        t_matrix_new = delta[1]
        this_mcp = physics.MCP[-1] + s_n


    # Save new parameters, transfer matrix and print adjustment vector
    tcwret_io.write("# s_n = {}".format(s_n))
    physics.MCP.append(this_mcp)
    numerical.T_MATRIX.append(t_matrix_new)

    return lm_param

################################################################################

def convergence(loop_count):
    '''
    Test if convergence reached

    Parameters
    ----------

    loop_count : int
        Iteration counter

    Returns
    -------
    bool
        True if converged
    '''

    if loop_count in (0, 1):
        conv_test = 1e10
    else:
        conv_test = np.abs((numerical.CHI2[-1]-numerical.CHI2[-2])/numerical.CHI2[-1])

    if loop_count != 0:
        condition = conv_test < inp.CONVERGENCE
        condition = condition and conv_test > 0.0
        condition = condition and numerical.CHI2[-2] > numerical.CHI2[-1]

        if loop_count != 0 and condition or loop_count == inp.MAX_ITER-1:
            return True
    return False


################################################################################

def set_up_retrieval():
    '''
    Initialise the retrieval and the matrizes
    '''
    physics.RADIANCE_LBLDIS = [[] for ii in range(len(inp.MCP)+1)]

    # Prepare the atmospheric data. Calculate the temperature of the cloud
    cloud_grid = []
    cloud_temp = []
    for layer in physics.CLOUD_LAYERS:
        cloud_layer = np.abs(physics.ATMOSPHERIC_GRID['altitude(km)']*1e3-layer)
        idx = list(cloud_layer).index(np.min(cloud_layer))
        cloud_grid.append(physics.ATMOSPHERIC_GRID['altitude(km)'][idx]*1e3)
        physics.ATMOSPHERIC_GRID['temperature(K)'][idx] += inp.DISTURB_CTEMPERATURE
        cloud_temp.append(physics.ATMOSPHERIC_GRID['temperature(K)'][idx])
    physics.CLOUD_GRID = np.intersect1d(cloud_grid, cloud_grid)
    physics.CLOUD_TEMP = np.mean(np.intersect1d(cloud_temp, cloud_temp))

    # Check the temperature of the cloud. If the temperature is outside the interval
    # of the given database set the a priori to 0 and the covariance to 10^-10
    for i in range(len(inp.DATABASES)):
        if physics.CLOUD_TEMP < inp.TEMP_DATABASES[i][0] or \
            physics.CLOUD_TEMP > inp.TEMP_DATABASES[i][1]:
            inp.MCP[i] = 0.0
            inp.VARIANCE_APRIORI[i] = 1e-10**(-2)
    physics.MCP = [np.zeros(inp.MCP.size)]
    for i in range(len(inp.MCP)):
        physics.MCP[0][i] = inp.MCP[i]

    # Set up the S_a matrix
    inp.MCP_APRIORI = np.array(inp.MCP[:])
    numerical.S_A_INV_MATRIX = np.array(inp.VARIANCE_APRIORI) * np.identity(inp.MCP.size)
    tcwret_io.log_prog_start()

    # Calculate the clear sky spectrum
    clear_sky_param = inp.MCP[:]
    clear_sky_param[0:int(inp.MCP.size//2)] = 0.
    physics.RADIANCE_CLEARSKY = rL.run_lbldis(np.array([clear_sky_param]), lblrtm=True)[-1]
    physics.RADIANCE_LBLDIS = [[] for ii in range(len(inp.MCP)+1)]

    # Calculate the S_y matrix
    numerical.s_y_inv()

################################################################################

def run_lbldis_and_derivatives():
    '''
    Run lbldis for the main cloud parameters and the derivatives.
    '''

    mcp = np.ones((int(inp.MCP.size//2+1), int(inp.MCP.size)))*physics.MCP[-1]

    for i in range(1, inp.MCP.size//2+1):
        mcp[i, i-1] += numerical.STEPSIZE

    radiance = rL.run_lbldis(np.array(mcp), lblrtm=False)[1]
    for i in range(inp.MCP.size//2+1):
        physics.RADIANCE_LBLDIS[i].append(radiance[:, i].flatten())
    delta = np.zeros(inp.MCP.size)
    for i in range(inp.MCP.size//2+1, inp.MCP.size+1):
        delta[i-1] = numerical.STEPSIZE
        mcp = [physics.MCP[-1] + delta]
        radiance = rL.run_lbldis(np.array(mcp), lblrtm=False)[1]
        try:
            physics.RADIANCE_LBLDIS[i].append(radiance[:, 0].flatten())
        except IndexError:
            physics.RADIANCE_LBLDIS[i].append(radiance.flatten())
        delta = np.zeros(inp.MCP.size)


################################################################################

def retrieve():
    '''
    Retrieval loop

    Returns
    -------
    int
        Iteration counter
    np.matrix
        Averaging kernel matrix
    np.array
        Errors calculated from covariance matrix
    np.matrix
        Covariance matrix from which the errors are calculated
    np.matrix
        Transfer matrix used for calculation of averaging kernel matrix and covariance matrix
    '''
    numerical.T_MATRIX = [np.zeros((len(inp.MCP), len(physics.WAVENUMBER_FTIR)))]
    lm_param = inp.LM_INIT

    for retr_loop in range(inp.MAX_ITER):

        tcwret_io.write("# Iteration: {}".format(retr_loop))
        tcwret_io.write("# [{}]".format(dt.datetime.now()))
        tcwret_io.write("# MCP of the current iteration: ")
        tcwret_io.write("# MCP = {}".format(physics.MCP[-1]))
        tcwret_io.write("# Levenberg-Marquardt parameter: {}".format(lm_param))

        run_lbldis_and_derivatives()
        converged = convergence(retr_loop)
        if converged or retr_loop == inp.MAX_ITER-1:
            break

        lm_param = retrieve_step(lm_param, retr_loop)

    averaging_kernel = numerical.calc_avk(numerical.T_MATRIX[-1])
    errors = numerical.calc_error()
    tcwret_io.write("Final Parameters: x_{} = {}\n".format(retr_loop, physics.MCP[-1]))
    return retr_loop, averaging_kernel, errors, errors[-1], numerical.T_MATRIX[-1]
