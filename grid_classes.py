# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:29:44 2026

@author: Saurabh
"""
import numpy as np
from dataclasses import dataclass, field

# ------------------- setting dataclass for each input type --------------------# 
# soil grid data and code for soil profile generation
@dataclass
class ProfileGridSpec:
    zmin: float
    zmax: float
    dz: float                 # requested min spacing
    z: np.ndarray = field(init=False)
    dz_all: np.ndarray = field(init=False)
    depth: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.z = np.arange(self.zmin + self.dz/2 ,self.zmax + self.dz/2 ,self.dz )
        self.zB = np.arange(self.zmin, self.zmax + self.dz, self.dz)
        self.dz_all = self.dz * np.ones(len(self.z))        
        self.depth = self.z  # downwards postive so z and depth are same 
        self.znodes = len(self.z)
        
# Root water uptake parameters 
@dataclass

class RWUSpec:
    psi_a: float        # critical pressure heads associated with anaerobiosis,
    psi_d: float        #critical pressure heads associated with soilwater-limited evapotranspiration
    psi_w: float      # critical pressure head associated with plant wilting
    Lr: float      # max root length (uniform grid)

# run time details and interval  
@dataclass
class TimeSpec:
    tmin: float # start time 
    tmax: float # finsh time length 
    dt : float  #in day 
    time_given: np.ndarray = field(init=False) # daily time 
    def __post_init__(self):
        self.time_given = np.arange(self.tmin,self.tmax,self.dt)   

# initial condition 
@dataclass
class InitialCondition:
    # hydrostatic conidtion is assumed in which pressure head = soil depth from surface - water table depth 
    z_wt : float # depth of water table 
    depth : np.ndarray
    RO0 : float # runoff assumption 
    psi0: np.ndarray = field(init=False) # initial pressure head
    y0:  np.ndarray = field(init=False) # initial input in model by combining both runoff and initial pressure head value 
    def __post_init__(self):
        self.psi0 = self.depth -self.z_wt
        self.y0 = np.concatenate(([self.RO0], self.psi0))


# Solver Options 
@dataclass
class SolverOptions:
    rtol: float = 1e-6 # default values 
    atol: float = 1e-7 # change them as needed 
    method: str = "BDF"  # method supported are BDF, RK45, Radau etc. check the scipy solve_ivp module 
    max_step: float = 1


@dataclass
class PostProcerssingOutputs:
    theta: np.ndarray
    K: np.ndarray
    h: np.ndarray
    Se: np.ndarray
    Actual_ET: np.ndarray
    Q_flux: np.ndarray
    ROin: np.ndarray
    STORAGE: np.ndarray
    PlantUptake: np.ndarray
    dROdt: np.ndarray
    NetPrecipitation : np.ndarray

