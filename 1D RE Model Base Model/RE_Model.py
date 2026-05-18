# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:26:49 2026

@author: Saurabh
"""

import numpy as np 
import pandas as pd 
import time
import datetime as dt
import pickle    
from grid_classes import ProfileGridSpec, RWUSpec, TimeSpec, InitialCondition, SolverOptions
from UtilitiesFunctions import plot_variable_at_depths
from RE_Model_function_files import RESolver  # , RichardsModelOutputs
t0= time.perf_counter() # setting the timer 



# reading met data
MetData = pd.read_excel('data_Clonroche.xlsx',sheet_name='met_data') 
# reading soil data  
SoilData = pd.read_excel('data_Clonroche.xlsx',sheet_name='vg_parameters_obs')  
# soil grid data and code
profileData = ProfileGridSpec( zmin=0, zmax=2, dz=0.02 )
# initial state 
IniData = InitialCondition(z_wt = 0.3,  # depth of water table 
                           depth = profileData.depth, # column length 
                           RO0 = 0.0 ) # runoff assumption 

# run time details and interval  
timeData = TimeSpec(
    tmin = 0,
    tmax = len(MetData),
    dt = 1 ,  #in day 
    )

# Root water uptake parameters 
RWUData = RWUSpec(
    psi_a=-0.05,  # critical pressure heads associated with anaerobiosis,
    psi_d=-4,   # critical pressure heads associated with soilwater-limited evapotranspiration
    psi_w=-150,  # # critical pressure head associated with plant wilting
    Lr= 1   # m # depth of root zone
    ) 


# solver options 
solver_opts = SolverOptions(rtol=1e-3, atol=1e-5, max_step = 1, method = "BDF") # default method is BDF 
# # running the solver 
# # the solver solves the pde in terms of pressure head or psi, in order to know other information it is processed
ProcessedOutputs, sol   = RESolver(SoilData,profileData, RWUData, timeData, MetData, IniData, solver_opts , 
                                   bottom_BC="no_flow" )   # "fixed_head" or "free_drainage"
# saving outputs 
with open("RE_model_results.pkl", "wb") as f:
    pickle.dump((ProcessedOutputs, sol), f)


t1 = time.perf_counter() - t0
print("Time elapsed: ", t1) # CPU seconds elapsed
                                    
    
#---------------------------Post Processing and Plots  ----------------------------- # 

target_depths = np.array([ 0.15, 0.45, 0.90, 1.20])
start_date = dt.datetime(1998, 1, 1)
t_dates = start_date + pd.to_timedelta(sol.t, unit="D")

theta_targets = plot_variable_at_depths(
    target_depths, profileData.z, ProcessedOutputs.theta, t_dates,
    ylabel=r"$\theta$", title="Soil moisture",
    save_path="theta_depth_plot.png"
)

h_targets = plot_variable_at_depths(
    target_depths, profileData.z, ProcessedOutputs.h, t_dates,
    ylabel="Tension (hPa)",
    title="Water tension",
    transform=lambda x: -x * 9.81 * 10,
    save_path="h_depth_plot.png"
)













































