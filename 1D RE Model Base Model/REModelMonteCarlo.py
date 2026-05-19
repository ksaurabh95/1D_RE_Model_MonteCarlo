# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:22:22 2026

@author: Saurabh
"""


# import numpy as np 

import pandas as pd 
import os
# import datetime as dt
from grid_classes import ProfileGridSpec, RWUSpec, TimeSpec, InitialCondition, SolverOptions
from UtilitiesFunctions import plot_variable_at_depthsMonteCarlo, compute_mc_stats
from REModelMonteCarlo_functions import RESolverMonteCarloParallel


# reading met data
MetData = pd.read_excel('data_johnstown.xlsx',sheet_name='met_data') 
# reading soil data  
soil_params = pd.read_excel('data_johnstown.xlsx',sheet_name='vg_dist_rosetta')   

 
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

if __name__ == "__main__":
    summary_df, failed_df = RESolverMonteCarloParallel(
        soil_params=soil_params,
        profileData=profileData,
        RWUData=RWUData,
        timeData=timeData,
        MetData=MetData,
        IniData=IniData,
        solver_opts=solver_opts,
        Nmc=100,
        bottom_BC="no_flow",
        n_workers= 8,
        save_detailed=True,
        output_dir="mc_outputs"
    )

    print(summary_df.head())
    print(failed_df)
    
# ------------- Post Processing and Results --------------------------------    

# read summary
summary = pd.read_csv("mc_summary.csv")
# keep only successful runs
successful = summary[summary["success"] == True]
print(f"Successful runs: {len(successful)}")




