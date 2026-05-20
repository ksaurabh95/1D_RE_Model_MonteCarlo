# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:46:06 2026

@author: Saurabh
"""
import numpy as np 
import pandas as pd 
import pickle
import os
import matplotlib.pyplot as plt  
import matplotlib.dates as mdates
from grid_classes import ProfileGridSpec,  TimeSpec
from UtilitiesFunctions import plot_variable_at_depthsMonteCarlo

# reading met data
MetData = pd.read_excel('data_johnstown.xlsx',sheet_name='met_data') 

# soil grid data and code
profileData = ProfileGridSpec( zmin=0, zmax=2, dz=0.02 )
# run time details and interval  
timeData = TimeSpec(
    tmin = 0,
    tmax = len(MetData),
    dt = 1 ,  #in day 
    )


# read summary
summary = pd.read_csv("mc_summary.csv")
# keep only successful runs
successful = summary[summary["success"] == True]
print(f"Successful runs: {len(successful)}")

processed_outputs = []

for mc_id in successful["MC_id"]:

    filepath = os.path.join("mc_outputs", f"run_{mc_id:04d}.pkl")

    with open(filepath, "rb") as f:
        data = pickle.load(f)
        
    processed_outputs.append(data["ProcessedOutputs"])

# variables present are 'h', 'theta', 'Actual_ET', STORAGE , PlantUptake , NetPrecipitation
target_depths = np.array([ 0.15,0.30, 0.45, 0.60, 0.90, 1.20])
start_date = pd.to_datetime( MetData['date']).iloc[0]
end_date   = pd.to_datetime( MetData['date']).iloc[-1]


t_dates = start_date + pd.to_timedelta(timeData.time_given, unit="D")


theta_all , _, _ , theta_mean_targets, theta_std_targets  = plot_variable_at_depthsMonteCarlo( target_depths, profileData.z, processed_outputs, t_dates, 
                                        varname = 'theta', 
                                        ylabel=r"$\theta$", 
                                        title="Soil moisture", 
                                        transform=None,   #  example of transform=lambda x: -x * 9.81 * 10,
                                        save_path= "theta_depth_plotMC.png" ) #  save_path= "theta_depth_plotMC.png" 




h_all , _, _ , h_mean_targets, h_std_targets = plot_variable_at_depthsMonteCarlo( target_depths, profileData.z, processed_outputs, t_dates, 
                                        varname = 'h', 
                                        ylabel=r"$h$", 
                                        title="Pressure head", 
                                        transform= lambda x: -x * 9.81 * 10,   #  example of transform=lambda x: -x * 9.81 * 10,
                                        save_path= "h_depth_plotMC.png"  ) #  save_path= "h_depth_plotMC.png" 



obsData = pd.read_excel('data_johnstown.xlsx',sheet_name='obsDateInterPolated')


start_date = pd.to_datetime( MetData['date']).iloc[0]
end_date   = pd.to_datetime( MetData['date']).iloc[-1]
t_dates = start_date + pd.to_timedelta(timeData.time_given, unit="D") 


# computing the water storgae 

dz_all = profileData.dz_all
# idx_10_cm =  np.arange(0,5,1 )
# idx_100_cm = np.arange(0,50,1 )
idx_10_cm = np.searchsorted(profileData.z, 0.10, side="right")
idx_100_cm = np.searchsorted(profileData.z, 1.00, side="right")

# definig the storing parameters 
nmc, nz,nt = np.shape(theta_all)
STORAGE_10_cm = np.empty([nmc,nt])
STORAGE_100_cm = np.empty([nmc,nt])

for i in range(nmc):
    for j in range(nt):
        STORAGE_10_cm[i,j] = np.sum(theta_all[i,:idx_10_cm,j]*dz_all[:idx_10_cm])*1000
        STORAGE_100_cm[i,j]=np.sum(theta_all[i, :idx_100_cm,j]*dz_all[:idx_100_cm])*1000


# 10 cm
STORAGE_10_cm_mean = np.mean(STORAGE_10_cm, axis = 0 )
STORAGE_10_cm_std = np.std(STORAGE_10_cm, axis = 0 )

# 100 cm 
STORAGE_100_cm_mean = np.mean(STORAGE_100_cm, axis = 0 )
STORAGE_100_cm_std = np.std(STORAGE_100_cm, axis = 0 )


leftts = ['a)', 'b)','c)','d)','e)', 'f)']

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

fig, axes = plt.subplots(6, 1, figsize=(10,10), sharex='col')
# ---- Observed Pressure head ----
axes[0].plot(obsData['date'], obsData['Tension_value_hPa_15cm'], label='Diamond and Sills (2001)')
axes[1].plot(obsData['date'], obsData['Tension_value_hPa_30cm'], label='Diamond and Sills (2001)')

axes[2].plot(obsData['date'], obsData['Tension_value_hPa_45cm'], label='Diamond and Sills (2001)')
axes[3].plot(obsData['date'], obsData['Tension_value_hPa_60cm'], label='Diamond and Sills (2001)')

axes[4].plot(obsData['date'], obsData['Tension_value_hPa_90cm'], label='Diamond and Sills (2001)')

axes[5].plot(obsData['date'], obsData['Tension_value_hPa_120cm'], label='Diamond and Sills (2001)')


# ----  Pressure head ----
for i in range(len(target_depths)):
    line, =  axes[i].plot(t_dates, h_mean_targets[i, :],    label="RE Model")
    axes[i].fill_between(t_dates, h_mean_targets[i, :] - h_std_targets[i, :],  
                         h_mean_targets[i, :] + h_std_targets[i, :] , color=line.get_color(), alpha=0.6 )
    
    axes[i].set_ylabel(r"$\psi$ (hPa)")
    axes[i].set_ylim([-200, 600])
    axes[i].set_title("Model")
    axes[i].set_title(f"{leftts[i]} Comparison of water tension at depth {target_depths[i]*100:.0f} cm")
    axes[i].grid(True)
    axes[i].grid(which='major', linestyle='--', linewidth=0.6, color='black')

    
    
    axes[i].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
    axes[i].tick_params(axis='x', rotation=0)

    axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))



axes[2].legend(ncol=1)


fig.tight_layout()

plt.savefig('Johnstown_MonteCarlo.svg', dpi = 300)
plt.savefig('Johnstown_MonteCarlo.png', dpi = 300)

plt.show()


fig, axes = plt.subplots(2, 1, figsize=(8,4), sharex=True)


axes[0].plot(t_dates, STORAGE_10_cm_mean ,label="RE Model" )
axes[0].fill_between(t_dates, STORAGE_10_cm_mean - STORAGE_10_cm_std, STORAGE_10_cm_mean + STORAGE_10_cm_std ,alpha=0.6)   
axes[0].set_ylabel(r"$\Theta $ (mm) ")
axes[0].grid()
axes[0].set_title('a) Water storage till 10 cm depth')
axes[0].set_ylim([0,50])

axes[1].plot(t_dates, STORAGE_100_cm_mean ,label="RE Model" )
axes[1].fill_between(t_dates, STORAGE_100_cm_mean - STORAGE_100_cm_std, STORAGE_100_cm_mean + STORAGE_100_cm_std ,alpha=0.6)   
axes[1].set_ylabel(r"$\Theta$ (mm)")
axes[1].grid()
axes[1].set_title('b) Water storage till 100 cm depth')
axes[1].set_ylim([0,500])
axes[1].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])

axes[1].legend(ncol=4)
axes[1].tick_params(axis='x', rotation=0)

axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
for ax in axes:
    # ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth=0.6, color='black')

fig.tight_layout()
plt.savefig('Johnstown_dataset_water_storage.svg', dpi = 300)
plt.savefig('Johnstown_dataset_water_storage.png', dpi = 300)

plt.show()


# export the files 

with open("RE_model_MoteCarlo_Results.pkl", "wb") as f:
    pickle.dump({
        "processed_outputs": processed_outputs,
        "theta_all": theta_all, "h_all":h_all,
        "STORAGE_100_cm_mean": STORAGE_100_cm_mean,
        "STORAGE_100_cm_std": STORAGE_100_cm_std,
        "STORAGE_10_cm_mean": STORAGE_10_cm_mean,
        "STORAGE_10_cm_std": STORAGE_10_cm_std,
        "t_dates": t_dates,
        "theta_mean_targets": theta_mean_targets,
        "theta_std_targets": theta_std_targets,
        "target_depths": target_depths
    }, f)
    
    





