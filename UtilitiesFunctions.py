# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:32:43 2026

@author: Saurabh
"""
import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def assign_vg(depth, df, extend_to):
    # extend last horizon if needed
    if extend_to and extend_to > df["Depth_max"].iloc[-1]:
        last = df.iloc[-1].copy()
        last["Depth_min"] = last["Depth_max"]
        last["Depth_max"]   = extend_to
        df = pd.concat([df, pd.DataFrame([last])], ignore_index=True)

    # assign parameters to each depth
    out = []
    for i in depth:
        row = df[(i >= df["Depth_min"]) & (i < df["Depth_max"])].iloc[0]
        out.append({"depth": i, **row.to_dict()})
        
    out = pd.DataFrame(out)
    out = out.drop(
            ['Horizon', 'Depth_min', 'Depth_max'],
            axis=1)   
    out['z'] = depth
    return out

def GetOutputAtRequiredDepths(target_depths,z, variable ):
    

    """
    # Function to get output at required depths
    Parameters:
    target depths: Depths at which output variable is required' 
    z : profile depth 
    variable : interested output variable whose output is needed

    Returns:
    variable_targets : interpolated values at required depths
    """
    
    f_variable = interp1d(z, variable, axis=0, bounds_error=False, fill_value="extrapolate")
    variable_targets = f_variable(target_depths)

    return  variable_targets





def plot_variable_at_depths(
    target_depths,
    z,
    variable,
    time,
    ylabel="Value",
    title="Variable at required depths",
    transform=None,
    save_path=None   # new argument
):
    
    start_date = time[0]
    end_date   = time[-1]

    target_depths = np.array(target_depths, dtype=float)
    variable_targets = GetOutputAtRequiredDepths(target_depths, z, variable)

    if transform is not None:
        variable_targets = transform(variable_targets)

    plt.figure(figsize=(8, 5))
    for i, d in enumerate(target_depths):
        plt.plot(time, variable_targets[i, :], label=f"{d:.2f} m")

    plt.xlabel("Time")
    plt.xlim(start_date, end_date)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


    plt.show()

    return variable_targets

def compute_mc_stats(processed_outputs, varname):

    if not hasattr(processed_outputs[0], varname):
        raise ValueError(f"{varname} not found in ProcessedOutputs")

    var_all = np.array([getattr(p, varname) for p in processed_outputs])

    var_mean = var_all.mean(axis=0)
    var_std = var_all.std(axis=0)

    return var_all, var_mean, var_std




def plot_variable_at_depthsMonteCarlo( target_depths, z, processed_outputs, time,
                                      varname = 'theta', # variables available are 'h', 'theta', 'Actual_ET', STORAGE , PlantUptake , NetPrecipitation
                                        ylabel="Value",
                                        title="Variable at required depths",
                                        transform=None,
                                        save_path=None   ):  # to save plot
                                                       
    
    if not hasattr(processed_outputs[0], varname):
        raise ValueError(f"{varname} not found in ProcessedOutputs")
        
    start_date = time[0]
    end_date   = time[-1]

    target_depths = np.array(target_depths, dtype=float)
    var_all , var_mean, var_std = compute_mc_stats(processed_outputs, varname) 

    var_mean_targets = GetOutputAtRequiredDepths(target_depths, z, var_mean)
    var_std_targets = GetOutputAtRequiredDepths(target_depths, z, var_std)


    if transform is not None:
        var_mean_targets = transform(var_mean_targets)
        var_std_targets = transform(var_std_targets)

    plt.figure(figsize=(8, 5))
    for i, d in enumerate(target_depths):
        plt.plot(time, var_mean_targets[i, :], label=f"{d:.2f} m")
        plt.fill_between(time, var_mean_targets[i, :] - var_std_targets[i,:], var_mean_targets[i, :] + var_std_targets[i,:] ,alpha=0.6, label="±1 std" ) 
        

    plt.xlabel("Time")
    plt.xlim(start_date, end_date)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol = 3)
    plt.tight_layout()
    # save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


    plt.show()

    return var_all , var_mean, var_std





















