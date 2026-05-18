# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:27:04 2026

@author: Saurabh
"""

import pandas as pd
import numpy as np 
from scipy.integrate import solve_ivp
from grid_classes import ProfileGridSpec, RWUSpec, TimeSpec, InitialCondition, SolverOptions, PostProcerssingOutputs
from PlantUptakeFunction import RootUptakeModel 
from VGModel import VGModel, VGfromSe
from UtilitiesFunctions import assign_vg


def RichardsEq( tCurrent, y, vgData,profileData, RWUData, timeData, MetData, bottom_BC):
    
    qP_mday = np.array( MetData['rain_mm'] )/1000 # rainfall (m/day)
    qPE_mday = np.array( MetData['pet_mm_per_day'] )/1000

    tDays = timeData.time_given
    z = profileData.z
    # zB = profileData.zB
    dz_all = profileData.dz_all
    dz = profileData.dz
    # Get RO from first node of solver vector
    
    RO = y[0]
    psi = y[1:]
    znodes = profileData.znodes
        
    # forcing at time t
    qP  = np.interp(tCurrent, tDays, qP_mday)     # rainfall (m/day)
    qPE = np.interp(tCurrent, tDays, qPE_mday)    # PET (m/day)
    
    Se, K, theta , C = VGModel(vgData, psi )    
    # total head at centers (depth positive down)
    H = psi - z
    # Calculate head gradient
    dHdz= np.diff(H)/np.diff(z) 
    # Calculate Darcy flux in the central zone
    # KB=(K[1:znodes]  +K[0:znodes-1])/2
    KB = (K[1:] + K[:-1])/2  # More pythonic and correct
    q_central =-KB*dHdz 
    # Calculate boundary flux assuming ponded conditions
    theta_pond, K_pond, psi_pond = VGfromSe(vgData, Se = 0.999)
    qPond=-K_pond*(H[0]-psi_pond) /  (dz)  
    # Choose minimum flux
    q0=min(qPond,qP);
    
    if bottom_BC == "free_drainage":
        q_bottom =  K[znodes-1]  #  bottom boundary condition : free drainage 
    elif bottom_BC == "no_flow":
        q_bottom = 0 # bottom boundary condition : zero flux condition 
        

        
    
    
    # q_bottom =  K[znodes-1]  #  bottom boundary condition : free drainage 
    # fluxes on edges (N+1): top, internal, bottom
    q = np.concatenate(([q0], np.asarray(q_central).ravel(), [q_bottom]))

    PlantUptake = RootUptakeModel(psi, RWUData,profileData,qPE )
    
    # dthetadt = -(q[1:] - q[:-1]) / dz - PlantUptake
    # Apply continuity equation continuity: dtheta/dt = -(q[i+1]-q[i])/dz - uptake
    dtheta_dt = - np.diff(q)/dz_all  - PlantUptake
    
    # Apply chain rule to get change in psi
    dpsi_dt=dtheta_dt/C

    # Calculate daily surface runoff in mm
    ROin=(qP-q0)*1000;
    # Put the runoff into a tank to attenuate the data enabling capture from
    # daily model abstractions during post processing.
    Tr=1;
    dROdt=(ROin-RO)/Tr
    # Add RO to solver vector
    dpsidt = np.concatenate(([dROdt], dpsi_dt))
    
    return dpsidt #   np.concatenate(([dROdt], dpsi_dt))




def RESolver(SoilData,profileData, RWUData, timeData, MetData, IniData,solver_opts, bottom_BC):
    # setting the timer 


    # SHP data
    vgData = assign_vg(profileData.depth, SoilData,profileData.zmax)   # getting vg data at each soil layer  
    # tCurrent = 1
    y0 = IniData.y0
    tDays = timeData.time_given
    rtol = solver_opts.rtol
    atol =  solver_opts.atol
    max_step = solver_opts.max_step
    method = solver_opts.method


    sol = solve_ivp(
        fun=lambda t, y: RichardsEq(t, y, vgData,profileData,RWUData,timeData, MetData, bottom_BC),
        t_span=(tDays[0], tDays[-1]),
        y0=y0,
        t_eval=tDays,
        method = method,       # closest to ode15s for stiff problems
        max_step = max_step,       # like MATLAB MaxStep=1
        rtol = rtol,
        atol = atol,
        )
    
    print(sol.success, sol.message)
    
    ProcessedOutputs = RichardsModelOutputs( sol, SoilData,profileData,RWUData, MetData, timeData, bottom_BC)





    return ProcessedOutputs, sol 




def RichardsModelOutputs( sol, SoilData,profileData,RWUData, MetData, timeData, bottom_BC):
    
    qP_mday = np.array( MetData['rain_mm'] )/1000 # rainfall (m/day)
    qPE_mday = np.array( MetData['pet_mm_per_day'] )/1000
    tDays = timeData.time_given
    vgData = assign_vg(profileData.depth, SoilData,profileData.zmax)   # getting vg data at each soil layer  

    h = sol.y[1:, :]
    t = sol.t 
    RO =sol.y[0,:]
    
    z = profileData.depth
    dz_all = profileData.dz_all
    dz = profileData.dz
    znodes = profileData.znodes 
    ii = (z <= RWUData.Lr)  # index over which 0 <= z < = Lr 
        
    Se = np.zeros(np.shape(h))
    K = np.zeros(np.shape(h))
    theta = np.zeros(np.shape(h))
    PlantUptake = np.zeros(np.shape(h))
    Q_flux = np.zeros(np.shape(h))
    Actual_ET = np.zeros(len(t))  # Calculate daily actual evapotranspiration  in mm
    NetPrecipitation = np.zeros(len(t)) #  daily effective precipitation in mm
    ROin = np.zeros(len(t)) #  daily surface runoff in mm
    
    
    for i in range(len(t)):
        psi = h[:,i]
        tCurrent = tDays[i]
        # forcing at time t
        qP  = np.interp(tCurrent, tDays, qP_mday)     # rainfall (m/day)
        qPE = np.interp(tCurrent, tDays, qPE_mday)    # PET (m/day)
        # total head at centers (depth positive down)
        H = psi - z
        # Calculate head gradient
        dHdz= np.diff(H)/np.diff(z) 
        # calculating theta, Se, K , Actual ET 
        Se[:,i], K[:,i], theta[:,i] , _ = VGModel(vgData, psi )
        PlantUptake[:,i] = RootUptakeModel(h[:,i] , RWUData,profileData,qPE )
        # Calculate Darcy flux in the central zone
        # KB=(K[1:znodes]  +K[0:znodes-1])/2
        KB = (K[1:,i] + K[:-1,i])/2  # More pythonic and correct
        q_central =-KB*dHdz 
        # Calculate boundary flux assuming ponded conditions
        theta_pond, K_pond, psi_pond = VGfromSe(vgData, Se = 0.999)
        qPond=-K_pond*(H[0]-psi_pond) /  (dz)  
        # Choose minimum flux
        q0=min(qPond,qP)
        if bottom_BC == "free_drainage":
            q_bottom =  K[znodes-1,i]  #  bottom boundary condition : free drainage 
        elif bottom_BC == "no_flow":
            q_bottom = 0 # bottom boundary condition : zero flux condition 
 
        # fluxes on edges (N+1): top, internal, bottom
        q = np.concatenate(([q0], np.asarray(q_central).ravel(), [q_bottom]))
        # computing total flux 
        Q_flux [:, i] = q[1:]

        # Calculate daily effective precipitation in mm
        NetPrecipitation[i] =q[0]*1000;
        # Calculate daily surface runoff in mm
        ROin [i] =(qP-q0)*1000
    
    # daily model abstractions during post processing.
    Tr=1;
    dROdt = (ROin-RO)/Tr
    # Calculate daily actual evapotranspiration  in mm
    Actual_ET = np.sum(PlantUptake[ii, :] * dz_all[ii, None], axis=0) 
    # calculate total storage in mm 
    STORAGE = np.sum(theta[:, :]*dz_all[:, None], axis=0)*1000

    return PostProcerssingOutputs (theta = theta,
                                   K= K,
                                   h = h,
                                   Se = Se,
                                   Actual_ET = Actual_ET,
                                   Q_flux =  Q_flux,
                                   ROin = ROin,
                                   STORAGE = STORAGE,
                                   PlantUptake = PlantUptake,
                                   dROdt = dROdt,
                                   NetPrecipitation = NetPrecipitation 
                                   ) 











