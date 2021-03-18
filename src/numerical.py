#!/usr/bin/python
'''@package Docstring
Some numerical functions
'''

import os
import sys

#Third-party modules
import numpy          as np
import scipy.signal   as sig


#Self-defined modules
import inp
sys.path.append(os.path.join(inp.PATH_TO_TCWRET, "src"))

import physics

STEPSIZE = 1e-3
S_A_INV_MATRIX = None
S_Y_INV_MATRIX = None
VARIANCE_RA = 0.0
CHI2 = []

T_MATRIX = []

def reduced_chi_square_test():
    '''
    Perform a reduced chi 2 test of the goodness of fit

    Returns
    -------
    float
        Reduced Chi 2 results
    '''
    number_of_parameters = inp.MCP.size
    number_of_observations = len(physics.WAVENUMBER_FTIR)
    red_chi_2 = 0.0
    for i in range(number_of_observations):
        red_chi_2 += (physics.RADIANCE_FTIR[i] - physics.RADIANCE_LBLDIS[0][-1][i])**2/VARIANCE_RA
    red_chi_2 /= (number_of_observations - number_of_parameters)

    return red_chi_2

####################################################################################

def s_y_inv():
    '''
    Calculate the noise of the spectrum
    '''

    global S_Y_INV_MATRIX
    global VARIANCE_RA
    physics.RADIANCE_LBLDIS = [[] for ii in range(len(inp.MCP)+1)]

    if inp.STDDEV < 0.0:
        variance_ra = np.mean(np.array(physics.NOISE_FTIR))**2
    elif inp.STDDEV == 0.0:
        variance_ra = 1.0
    else:
        variance_ra = inp.STDDEV**2
    vec_error = np.array([variance_ra for ii in range(len(physics.WAVENUMBER_FTIR))])
    S_Y_INV_MATRIX = np.reciprocal(vec_error) * np.identity(len(vec_error))
    VARIANCE_RA = variance_ra


####################################################################################

def calc_chi_2_and_residuum():
    '''
    Calculate the new cost function and residuum

    Returns
    -------
    float
        New cost function

    float
        New residuum
    '''

    # Get residuum
    res = np.array(physics.RADIANCE_FTIR) - np.array(physics.RADIANCE_LBLDIS[0][-1])
    res = np.transpose(np.matrix(res))

    # (y - F(x))^T S_y_1 (y - F(x))
    _res = np.float_(np.dot(np.matmul(np.transpose(res), S_Y_INV_MATRIX), \
                            res))

    apr_vec = np.array(inp.MCP_APRIORI) - physics.MCP[-1]

    # (x_a - x_i)^T S_a_1 (x_a - x_i)
    _apr = np.float_(np.dot(np.matmul(np.transpose(apr_vec), S_A_INV_MATRIX), apr_vec))

    # chi^2 = (y - F(x))^T S_y_1 (y - F(x)) + Calculate (x_a - x_i)^T S_a_1 (x_a - x_i)
    chi2 = np.float_(_res + _apr)

    return chi2, res

def jacobian():
    '''
    Calculation of the jacobian_matrix

    Returns
    -------
    np.array
        Transposed jacobian matrix
    '''

    deriv = [np.zeros(len(physics.RADIANCE_FTIR)) for i in range(inp.MCP.size)]
    for i in range(1, inp.MCP.size+1):
        deriv[i-1] = (np.array(physics.RADIANCE_LBLDIS[i][-1]) \
                     - np.array(physics.RADIANCE_LBLDIS[0][-1]))/STEPSIZE

    return deriv

def calc_avk(t_matrix):
    '''
    Calculation of the averaging kernel matrix

    Parameter
    ---------
    t_matrix : np.matrix
        Transfer matrix which propagates covariances from measurement to result

    Results
    -------
    np.matrix
        Averaging kernel matrix
    '''
    jacobian_matrix_transposed = np.matrix(jacobian())
    jacobian_matrix = np.transpose(jacobian_matrix_transposed)
    averaging_kernel = np.matmul(t_matrix, jacobian_matrix)

    return averaging_kernel

def calc_vcm(t_matrix):
    '''
    Calculation of the variance-covariance matrix

    Parameter
    ---------
    t_matrix : np.matrix
        Transfer matrix which propagates covariances from measurement to result

    Results
    -------
    np.matrix
        Variance-covariance matrix of the retrieval
    '''
    s_y_matrix = np.linalg.inv(S_Y_INV_MATRIX)
    cov = np.matmul(t_matrix, s_y_matrix)
    cov = np.matmul(cov, np.transpose(t_matrix))
    return cov

def iteration(lm_param, t_matrix):
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

    jacobian_matrix_transposed = np.matrix(jacobian())
    jacobian_matrix = np.transpose(jacobian_matrix_transposed)

    # K^T S_y_1
    JT_W = np.matmul(jacobian_matrix_transposed, S_Y_INV_MATRIX)

    # K^T S_y_1 K
    JT_W_J = np.matmul(JT_W, jacobian_matrix)

    # K^T S_y_1 (y-F(x))
    JT_W_r = np.matmul(JT_W, physics.RESIDUUM[-1])

    # Calculate S_a_1 (x_a - x_i)
    second_sum = np.matmul(S_A_INV_MATRIX, inp.MCP_APRIORI - np.array(physics.MCP[-1]))

    # Calculate K^T S_y_1 J + S_a_1 + mu**2 D
    left_side = np.add(np.add(JT_W_J, S_A_INV_MATRIX), lm_param**2*S_A_INV_MATRIX)

    # Calculate K^T S_y_1 (y-F(x)) + S_a_1 (x_a - x_i)
    right_side = \
    np.transpose(np.add(np.transpose(JT_W_r), second_sum))

    # Solve
    s_n = np.array(np.linalg.solve(left_side, right_side))

    # Calculate transfer matrix
    M = np.linalg.inv(left_side)
    G = np.matmul(M, JT_W)
    I_GJ_MR = np.identity(len(physics.MCP[-1])) - np.matmul(G, jacobian_matrix) \
            - np.matmul(M, S_A_INV_MATRIX)
    T_new = np.add(G, np.matmul(I_GJ_MR, t_matrix))
    return s_n.flatten(), T_new

def calc_error():
    '''

    Calculate covariance matrix and standard deviations

    Returns
    -------
    list
        Standard deviation and covariance matrix
    '''

    cov = calc_vcm(T_MATRIX[-1])
    var = [0.0 for i in range(inp.MCP.size)]
    for i in range(inp.MCP.size):
        var[i] = np.sqrt(cov.item(i, i))

    var.append(cov)

    return var
