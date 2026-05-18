# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:34:05 2026

@author: Saurabh
"""

import numpy as np 
import pandas as pd 


def f2(psi, psi_a, psi_d, psi_w):
    """
    # plant stress function
    Computes f2(psi) based on a piecewise function.

    Parameters:
    psi   : input pressure head (can be scalar or numpy array)
    psi_a : air entry pressure head
    psi_d : critical pressure head (dry threshold)
    psi_w : wilting point pressure head

    Returns:
    f2_val : result of f2(psi)
    """
    psi = np.array(psi, dtype=float)
    f2_val = np.zeros_like(psi)
    
    # psi >= psi_a 
    mask1 = (psi >= psi_a)
    f2_val[mask1] = 0.01
    
    # Case: psi_a > psi > psi_d --> f2 = 1
    mask2 = (psi < psi_a) & (psi > psi_d)
    f2_val[mask2] = 1

    # Case: psi_d >= psi >= psi_w --> f2 = 1 - (psi - psi_d)/(psi_w - psi_d)
    mask3 = (psi <= psi_d) & (psi >= psi_w)
    f2_val[mask3] = 1 - (psi[mask3] - psi_d) / (psi_w - psi_d)

    # psi < psi_w 
    mask4 = (psi < psi_w)
    f2_val[mask4] = 0
    
    return f2_val

def f1(depth,a,Lr,zmax):
    """
    exponential root distribution function
    
    """
    depth = np.array(depth, dtype=float)
    f1 = np.zeros_like(depth)
    mask = depth > Lr
    f1[~mask] = a/Lr *( np.exp(-a)-np.exp(-a*depth[~mask]/Lr) ) / ( (1+a)*np.exp(-a) -1)                                                                
    f1[mask] = 0
 
    return f1


def RootUptakeModel(psi, RWUData,profileData,Ep_current ):
    
    """
    Feddes Plant Uptake Function 
    psi:  pressure head 
    RWUData:  Parameters of Feddes Function 
    profileData: grid information  
    Ep_current: potential evapotanspiration 
    RWU: Root water Uptake at given time step 
    
    """
   
    zmax, depth = profileData.zmax, profileData.depth
    psi_a,psi_d,psi_w, Lr = RWUData.psi_a,RWUData.psi_d,RWUData.psi_w, RWUData.Lr
    # a = zmax - Lr # m # measuremnet of how fast root zone declines with depth
    a = 2
    
    f1_vals = f1(depth,a,Lr,zmax) # 1st term of the root water uptake model by Feddes
    f2_vals = f2(psi, psi_a, psi_d, psi_w)
    RWU = f1_vals*f2_vals*Ep_current
    
    return RWU

