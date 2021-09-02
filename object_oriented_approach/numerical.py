#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 21:20:00 2021

@author: philipp
"""

import numpy as np

def convergence(chi2, chi2_prev, conv):
    '''
    Test if convergence reached

    Parameters
    ----------

    chi2 : float
        Iteration counter
        
    chi2_prev: float
        XX
        
    conv: Bool
        XX

    Returns
    -------
    bool
        True if converged
    '''

    conv_test = np.abs((chi2-chi2_prev)/chi2)

    condition = conv_test < conv
    condition = condition and conv_test > 0.0
    condition = condition and chi2_prev > chi2
    
    if condition:
        return True
    return False