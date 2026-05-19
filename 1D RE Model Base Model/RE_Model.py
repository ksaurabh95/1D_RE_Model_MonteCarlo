# -*- coding: utf-8 -*-
"""
@author: Saurabh
"""
# Modules Used in this study  
import numpy as np 
import pandas as pd 
import pickle  
import matplotlib.dates as mdates
import matplotlib.pyplot as plt  
from grid_classes import ProfileGridSpec, RWUSpec, TimeSpec, InitialCondition, SolverOptions
from UtilitiesFunctions import plot_variable_at_depths
from RE_Model_function_files import RESolver 
# ----------------- Input Data ------------------------------------------
# reading met data
MetData = pd.read_excel('data_johnstown.xlsx',sheet_name='met_data') 
# reading soil data  
SoilData = pd.read_excel('data_johnstown.xlsx',sheet_name='VGParams_Rosetta')  

# ------------- Model Specifications -----------------------------
# soil grid data and code
profileData = ProfileGridSpec( zmin=0, zmax=2, dz=0.02 )

# initial state 
IniData = InitialCondition(z_wt = 0.3,  # depth of water table 
                           depth = profileData.depth, # column length 
                           RO0 = 0.0 ) # runoff assumption 

# Root water uptake parameters 
RWUData = RWUSpec(
    psi_a=-0.05,  # critical pressure heads associated with anaerobiosis,
    psi_d=-4,   # critical pressure heads associated with soilwater-limited evapotranspiration
    psi_w=-150,  # # critical pressure head associated with plant wilting
    Lr= 1   # m # depth of root zone
    ) 

# Run time details and interval  -------------------
timeData = TimeSpec(
    tmin = 0,
    tmax = len(MetData),
    dt = 1 ,  #in day 
    )

# solver options 
solver_opts = SolverOptions(rtol=1e-3, atol=1e-5, max_step = 1, method = "BDF") # default method is BDF 
# # running the solver 
# # the solver solves the pde in terms of pressure head or h, in order to know other information it is processed
ProcessedOutputs, sol   = RESolver(SoilData,profileData, RWUData, timeData, MetData, IniData, solver_opts , 
                                   bottom_BC="no_flow" )   # "no_flow" or "free_drainage"

#--------------- saving outputs -------------------------------------------------------------- 
with open("RE_model_results.pkl", "wb") as f:
    pickle.dump((ProcessedOutputs, sol), f)

    
#---------------------------Post Processing and Plots  ----------------------------- # 

target_depths = np.array([ 0.15,0.30, 0.45, 0.60,0.90, 1.20])
start_date = pd.to_datetime( MetData['date']).iloc[0]
end_date   = pd.to_datetime( MetData['date']).iloc[-1]
t_dates = start_date + pd.to_timedelta(timeData.time_given, unit="D")


theta_targets = plot_variable_at_depths(
    target_depths, profileData.z, ProcessedOutputs.theta, t_dates,
    ylabel=r"$\theta$", 
    title="Soil moisture",
    transform= None, # example lambda x: -x * 9.81 * 10,
    # save_path="theta_depth_plot.png"
)

h_targets = plot_variable_at_depths(
    target_depths, profileData.z, ProcessedOutputs.h, t_dates,
    ylabel="Tension (hPa)",
    title="Water tension",
    transform=lambda x: -x * 9.81 * 10, # or transform= None 
    # save_path="h_depth_plot.png"
)

#--------------------- Comparison with Observations ---------------------------
lets = ['a)', 'b)','c)','d)','e)', 'f)', 'g)','h)']
obsData = pd.read_excel('data_johnstown.xlsx',sheet_name='obsDateInterPolated')

fig, axes = plt.subplots(6, 1, figsize=(8, 10), sharex='col')
# ---- Observed Pressure head ----
axes[0].plot(obsData['date'], obsData['Tension_value_hPa_15cm'], label='Diamond and Sills (2001)')
axes[1].plot(obsData['date'], obsData['Tension_value_hPa_30cm'], label='Diamond and Sills (2001)')
axes[2].plot(obsData['date'], obsData['Tension_value_hPa_45cm'], label='Diamond and Sills (2001)')
axes[3].plot(obsData['date'], obsData['Tension_value_hPa_60cm'], label='Diamond and Sills (2001)')
axes[4].plot(obsData['date'], obsData['Tension_value_hPa_90cm'], label='Diamond and Sills (2001)')
axes[5].plot(obsData['date'], obsData['Tension_value_hPa_120cm'], label='Diamond and Sills (2001)')


# ----  Water Tension  ----
for i in range(len(target_depths)):
    axes[i].plot(t_dates, h_targets[i, :],    label="RE Model")  
    axes[i].set_ylabel(r"Tension, $\psi$ (hPa)")
    axes[i].set_ylim([-200, 600])
    axes[i].set_title(f"{lets[i]} Comparison of pressure head  at depth {target_depths[i]*100:.0f} cm")
    axes[i].grid(True)
    axes[i].grid(which='major', linestyle='--', linewidth=0.6, color='black')    
    axes[i].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))


axes[2].legend(ncol=2)
fig.tight_layout()
plt.savefig('Johnstown_dataset.png', dpi = 300)
plt.show()
