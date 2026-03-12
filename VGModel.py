# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:29:27 2026

@author: Saurabh
"""
import pandas as pd
import numpy as np 

def VGModel(vgData, psi ):
    rhoWgCrcW = 9.81*10**(-7)  # rho*g*(Cr + Cw) = 9.81*10^(-7) m-1 
    
    
    thetas = np.array(vgData["thetas"])
    thetar = np.array(vgData["thetar"])
    alpha = np.array(vgData["alpha (m-1)"])
    N = np.array(vgData["N"])
    Ksat = np.array(vgData["Ksat (m/day)"])  # m/day
    n_eta = np.array(vgData["n_eta"])  # m/day
    m = 1- 1/N
    tempI = psi >= 0 
    
    Se = (1 + ( abs(alpha*psi))**N )**(-m) 
    theta = (thetas - thetar)*Se + thetar 
    K = Ksat*( Se**n_eta ) *( ( 1 - (  (1 - Se**(1/m) )**m ) )**2) 
    
    psi_safe = np.where(np.abs(psi) < 1e-12, -1e-12, psi)
    dSe_dpsi = (1-N)*Se * (np.abs(alpha*psi_safe)**N) / ((1 + (np.abs(alpha*psi_safe)**N)) * psi_safe)
    # dSe_dpsi = (1-N)*Se * ( (abs(alpha*psi))**N ) / (  (1 + ( abs(alpha*psi))**N )*psi  ) 
    
    K[tempI] = Ksat[tempI]
    Se[tempI] = 1 
    theta[tempI] = thetas[tempI]    
    dSe_dpsi[tempI] = 0 
    
    C = (thetas-thetar)*dSe_dpsi + theta*rhoWgCrcW
    C = np.maximum(C, 1e-10)  # Prevent division by zero in dpsi_dt = dtheta_dt/C

    return Se, K, theta , C


def VGfromSe(vgData, Se):
    
    # this function is used to estimate the values of theta, h and K at ponding condition where Se = 0.999    
    thetas = np.array(vgData["thetas"][0] )
    thetar = np.array(vgData["thetar"][0])
    alpha = np.array(vgData["alpha (m-1)"][0])
    N = np.array(vgData["N"][0])
    Ksat = np.array(vgData["Ksat (m/day)"][0])  # m/day
    n_eta = np.array(vgData["n_eta"][0])  # m/day
    m = 1- 1/N

    Se = np.clip(Se, 0.0, 1.0)

    theta = (thetas - thetar)*Se + thetar 
    K = Ksat*( Se**n_eta ) *( ( 1 - (  (1 - Se**(1/m) )**m ) )**2) 
    # psi = ( Se**(m) - 1 )**(-N) / alpha
    
    Se_inv = np.clip(Se, 1e-12, 1.0 - 1e-12)
    psi_mag = ((Se_inv ** (-1.0 / m) - 1.0) ** (1.0 / N)) / alpha
    psi = -psi_mag

    # If Se>=1 => saturated
    psi = np.where(Se >= 1.0, 0.0, psi)
    
   
    return theta, K, psi 