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
MetData = pd.read_excel('data_Clonroche.xlsx',sheet_name='met_data') 

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
target_depths = np.array([ 0.15, 0.45, 0.90, 1.20])
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



obs15 = pd.read_excel('data_Clonroche.xlsx',sheet_name='obs_data_15_cm')
obs45 = pd.read_excel('data_Clonroche.xlsx',sheet_name='obs_data_45_cm')
obs90 = pd.read_excel('data_Clonroche.xlsx',sheet_name='obs_data_90_cm')
obs120 = pd.read_excel('data_Clonroche.xlsx',sheet_name='obs_data_120_cm')





start_date = pd.to_datetime( MetData['date']).iloc[0]
end_date   = pd.to_datetime( MetData['date']).iloc[-1]
t_dates = start_date + pd.to_timedelta(timeData.time_given, unit="D") 
lets = ['a)','c)','e)', 'g)']
subplot2 = ['b)','d)','f)', 'h)']

fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex='col')
# ---- Observed Pressure head ----
axes[0, 0].plot(obs15['Date'], obs15['Tension_value_hPa'], label='obs')
# axes[0, 0].fill_between(obs15['Date'], obs15['lower_Tension_value_hPa'], obs15['upper_Tension_value_hPa'], label='obs')

axes[1, 0].plot(obs45['Date'], obs45['Tension_value_hPa'], label='obs')
# axes[1, 0].fill_between(obs45['Date'], obs45['lower_Tension_value_hPa'], obs45['upper_Tension_value_hPa'], label='obs')

axes[2, 0].plot(obs90['Date'], obs90['Tension_value_hPa'], label='obs')
# axes[2, 0].fill_between(obs90['Date'], obs90['lower_Tension_value_hPa'], obs90['upper_Tension_value_hPa'], label='obs')

axes[3, 0].plot(obs120['Date'], obs120['Tension_value_hPa'], label='obs')
# axes[3, 0].fill_between(obs90['Date'], obs90['lower_Tension_value_hPa'], obs90['upper_Tension_value_hPa'], label='obs')


# ---- Observed Water content ----
axes[0, 1].plot(obs15['Date'], obs15['theta_obs'], label='Diamond and Sills (2001)')
axes[1, 1].plot(obs45['Date'], obs45['theta_obs'], label='Diamond and Sills (2001)')
axes[2, 1].plot(obs120['Date'], obs120['theta_obs'], label='Diamond and Sills (2001)')
axes[3, 1].plot(obs120['Date'], obs120['theta_obs'], label='Diamond and Sills (2001)')

# ----  Pressure head ----
for i in range(len(target_depths)):
    axes[i, 0].plot(t_dates, h_mean_targets[i, :],    label="RE Model")
    axes[i,0].fill_between(t_dates, h_mean_targets[i, :] - h_std_targets[i, :],  
                         h_mean_targets[i, :] + h_std_targets[i, :] , color="orange", alpha=0.3 )
    
    axes[ i, 0].set_ylabel("Tension (hPa)")
    axes[ i, 0].set_ylim([-250, 1000])
    axes[ i, 0].set_title("Model")
    axes[i, 0].set_title(f"{lets[i]} Comparison of pressure head  at depth {target_depths[i]*100:.0f} cm")
    axes[ i, 0].grid(True)
    axes[i, 0].grid(which='major', linestyle='--', linewidth=0.6, color='black')


# ---- Model Water content ----
    axes[i, 1].plot(t_dates, theta_mean_targets[i, :], color="orange", label='RE Model')
    axes[i,1].fill_between(t_dates, theta_mean_targets[i,:] - theta_std_targets[i,:], theta_mean_targets[i,:] + theta_std_targets[i,:], color="orange",alpha=0.3 )   
    
    axes[i, 1].set_ylabel(r"$\theta$")
    axes[ i, 1].set_ylim([0.0, 0.5])
    # axes[1, 0].legend()
    axes[i, 1].grid(True)
    axes[i, 1].grid(which='major', linestyle='--', linewidth=0.6, color='black')
    axes[i, 1].set_title(f"{subplot2[i]} Comparison of soil moisture at depth {target_depths[i]*100:.0f} cm")
    
    
    axes[i,0].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
    axes[i,0].tick_params(axis='x', rotation=30)

    axes[i,0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[i,0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))


    axes[i,1].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
    axes[i,1].tick_params(axis='x', rotation=30)

    axes[i,1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[i,1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))


axes[2,1].legend(ncol=1)


# --- Common labels ---
# axes[2].set_xlabel("Date")

# for ax in axes:
#     # ax.minorticks_on()
#     ax.grid(which='major', linestyle='--', linewidth=0.6, color='black')
    


fig.tight_layout()

# plt.savefig('ISMON_dataset.svg', dpi = 300)
# plt.savefig('ISMON_dataset.png', dpi = 300)

plt.show()






































# ObsSoilMoistureHourly = pd.read_excel('data_johnstown.xlsx',sheet_name='ObsSoilMoistureHourly')

# plt.rcParams.update({
#     "font.family": "Times New Roman",
#     "font.size": 10,
#     "axes.titlesize": 10,
#     "axes.labelsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 10,
# })



# fig, axes = plt.subplots(6, 1, figsize=(10,8), sharex=True)
# # --- 5 cm ---
# axes[0].plot(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_5cm_mean'], label='obs',linewidth = 2)
# axes[0].fill_between(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_5cm_mean'] - ObsSoilMoistureHourly['theta_5cm_std'],  
#                      ObsSoilMoistureHourly['theta_5cm_mean'] + ObsSoilMoistureHourly['theta_5cm_std'], alpha=0.6, label="±1 std")
# axes[0].set_ylabel(r"$\theta $")
# axes[0].grid()
# axes[0].set_ylim([0,0.6])
# axes[0].set_title('a) Soil moisture at 5 cm depth')

# # --- 10 cm ---

# axes[1].plot(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_10cm_mean'], label='obs',linewidth = 2)
# axes[1].fill_between(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_10cm_mean'] - ObsSoilMoistureHourly['theta_10cm_std'],  
#                      ObsSoilMoistureHourly['theta_10cm_mean'] + ObsSoilMoistureHourly['theta_10cm_std'], alpha=0.6, label="±1 std")
# axes[1].set_ylabel(r"$\theta $")
# axes[1].grid()
# axes[1].set_ylim([0,0.6])
# # axes[1].legend(ncol=2)
# axes[1].set_title('b) Soil moisture at 10 cm depth')

# # --- 20 cm ---

# axes[2].plot(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_20cm_mean'], label='depth = 45 cm',linewidth = 2)
# axes[2].fill_between(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_20cm_mean'] - ObsSoilMoistureHourly['theta_20cm_std'],  
#                      ObsSoilMoistureHourly['theta_20cm_mean'] + ObsSoilMoistureHourly['theta_20cm_std'], alpha=0.6, label="±1 std")
# axes[2].set_ylabel(r"$\theta $")
# axes[2].grid()
# # axes[2].legend()
# axes[2].set_ylim([0,0.6])

# axes[2].set_title('c) Soil moisture at 20 cm depth')
# # --- 30 cm ---

# axes[3].plot(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_30cm_mean'], label='depth = 90 cm',linewidth = 2)
# axes[3].fill_between(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_30cm_mean'] - ObsSoilMoistureHourly['theta_30cm_std'],  
#                      ObsSoilMoistureHourly['theta_30cm_mean'] + ObsSoilMoistureHourly['theta_30cm_std'], alpha=0.6, label="±1 std")
# axes[3].set_ylabel(r"$\theta $")
# axes[3].grid()
# # axes[3].legend()
# axes[3].set_title('d) Soil moisture at 30 cm depth')
# axes[3].set_ylim([0,0.6])

# # --- 50 cm ---

# axes[4].plot(ObsSoilMoistureHourly["date"],ObsSoilMoistureHourly['theta_50cm_mean'], label='depth = 120 cm',linewidth = 2)
# axes[4].fill_between(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_50cm_mean'] - ObsSoilMoistureHourly['theta_50cm_std'],  
#                      ObsSoilMoistureHourly['theta_50cm_mean'] + ObsSoilMoistureHourly['theta_50cm_std'], alpha=0.6, label="±1 std")
# axes[4].set_ylabel(r"$\theta $")
# axes[4].grid()
# # axes[2].legend()
# axes[4].set_title('e) Soil moisture 50 cm depth')
# axes[4].set_ylim([0,0.6])


# # --- 100 cm ---

# axes[5].plot(ObsSoilMoistureHourly["date"],ObsSoilMoistureHourly['theta_100cm_mean'], label='ISMON Dataset',linewidth = 2)
# axes[5].fill_between(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['theta_100cm_mean'] - ObsSoilMoistureHourly['theta_100cm_std'],  
#                      ObsSoilMoistureHourly['theta_100cm_mean'] + ObsSoilMoistureHourly['theta_100cm_std'], alpha=0.6)
# axes[5].set_ylabel(r"$\theta $")
# axes[5].grid()
# axes[5].set_title('f) Soil moisture at 100 cm depth')
# axes[5].set_ylim([0,0.6])


# for i in range(len(target_depths)):   
#     axes[i].plot(t_dates, theta_mean_targets[i,:] ,label="RE Model" )
#     axes[i].fill_between(t_dates, theta_mean_targets[i,:] - theta_std_targets[i,:], theta_mean_targets[i,:] + theta_std_targets[i,:] ,alpha=0.6)   




# axes[5].legend(ncol=4)


# # --- Common labels ---
# # axes[2].set_xlabel("Date")
# axes[5].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
# axes[5].tick_params(axis='x', rotation=0)

# axes[5].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
# axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

# for ax in axes:
#     # ax.minorticks_on()
#     ax.grid(which='major', linestyle='--', linewidth=0.6, color='black')
    

# fig.tight_layout()

# # plt.savefig('ISMON_dataset.svg', dpi = 300)
# # plt.savefig('ISMON_dataset.png', dpi = 300)

# plt.show()



# # computing the water storgae 

# dz_all = profileData.dz_all
# # idx_10_cm =  np.arange(0,5,1 )
# # idx_100_cm = np.arange(0,50,1 )
# idx_10_cm = np.searchsorted(profileData.z, 0.10, side="right")
# idx_100_cm = np.searchsorted(profileData.z, 1.00, side="right")

# # definig the storing parameters 
# nmc, nz,nt = np.shape(theta_all)
# STORAGE_10_cm = np.empty([nmc,nt])
# STORAGE_100_cm = np.empty([nmc,nt])

# for i in range(nmc):
#     for j in range(nt):
#         STORAGE_10_cm[i,j] = np.sum(theta_all[i,:idx_10_cm,j]*dz_all[:idx_10_cm])*1000
#         STORAGE_100_cm[i,j]=np.sum(theta_all[i, :idx_100_cm,j]*dz_all[:idx_100_cm])*1000


# # 10 cm
# STORAGE_10_cm_mean = np.mean(STORAGE_10_cm, axis = 0 )
# STORAGE_10_cm_std = np.std(STORAGE_10_cm, axis = 0 )

# # 100 cm 
# STORAGE_100_cm_mean = np.mean(STORAGE_100_cm, axis = 0 )
# STORAGE_100_cm_std = np.std(STORAGE_100_cm, axis = 0 )


# plt.rcParams.update({
#     "font.family": "Times New Roman",
#     "font.size": 10,
#     "axes.titlesize": 10,
#     "axes.labelsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 10,
# })

# fig, axes = plt.subplots(2, 1, figsize=(8,4), sharex=True)

# axes[0].plot(ObsSoilMoistureHourly["date"],ObsSoilMoistureHourly['storage10_mean'], label='ISMON Dataset',linewidth = 2)
# axes[0].fill_between(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['storage10_mean'] - ObsSoilMoistureHourly['storage10_std'],  
#                      ObsSoilMoistureHourly['storage10_mean'] + ObsSoilMoistureHourly['storage10_std'], alpha=0.6)


# axes[0].plot(t_dates, STORAGE_10_cm_mean ,label="RE Model" )
# axes[0].fill_between(t_dates, STORAGE_10_cm_mean - STORAGE_10_cm_std, STORAGE_10_cm_mean + STORAGE_10_cm_std ,alpha=0.6)   
# axes[0].set_ylabel(r"$\Theta $ (mm) ")
# axes[0].grid()
# axes[0].set_title('a) Water storage till 10 cm depth')
# axes[0].set_ylim([0,50])

# axes[1].plot(ObsSoilMoistureHourly["date"],ObsSoilMoistureHourly['storage100_mean'], label='ISMON Dataset',linewidth = 2)
# axes[1].fill_between(ObsSoilMoistureHourly["date"], ObsSoilMoistureHourly['storage100_mean'] - ObsSoilMoistureHourly['storage100_std'],  
#                      ObsSoilMoistureHourly['storage100_mean'] + ObsSoilMoistureHourly['storage10_std'], alpha=0.6)


# axes[1].plot(t_dates, STORAGE_100_cm_mean ,label="RE Model" )
# axes[1].fill_between(t_dates, STORAGE_100_cm_mean - STORAGE_100_cm_std, STORAGE_100_cm_mean + STORAGE_100_cm_std ,alpha=0.6)   
# axes[1].set_ylabel(r"$\Theta$ (mm)")
# axes[1].grid()
# axes[1].set_title('b) Water storage till 100 cm depth')
# axes[1].set_ylim([0,500])
# axes[1].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])

# axes[1].legend(ncol=4)
# axes[1].tick_params(axis='x', rotation=0)

# axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
# axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
# for ax in axes:
#     # ax.minorticks_on()
#     ax.grid(which='major', linestyle='--', linewidth=0.6, color='black')

# fig.tight_layout()
# # plt.savefig('ISMON_dataset_water_storage.svg', dpi = 300)
# # plt.savefig('ISMON_dataset_water_storage.png', dpi = 300)

# plt.show()


# # export the files 

# with open("RE_model_MoteCarlo_Results.pkl", "wb") as f:
#     pickle.dump({
#         "processed_outputs": processed_outputs,
#         "theta_all": theta_all, "h_all":h_all,
#         "STORAGE_100_cm_mean": STORAGE_100_cm_mean,
#         "STORAGE_100_cm_std": STORAGE_100_cm_std,
#         "STORAGE_10_cm_mean": STORAGE_10_cm_mean,
#         "STORAGE_10_cm_std": STORAGE_10_cm_std,
#         "t_dates": t_dates,
#         "theta_mean_targets": theta_mean_targets,
#         "theta_std_targets": theta_std_targets,
#         "target_depths": target_depths
#     }, f)
    
    











