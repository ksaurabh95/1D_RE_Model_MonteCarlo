# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:06:20 2026

@author: Saurabh
"""

import pandas as pd
import numpy as np 
# from scipy.integrate import solve_ivp
import time
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from grid_classes import ProfileGridSpec, RWUSpec, TimeSpec, InitialCondition, SolverOptions, PostProcerssingOutputs
from PlantUptakeFunction import RootUptakeModel 
from VGModel import VGModel, VGfromSe
from UtilitiesFunctions import assign_vg
from RE_Model_function_files import RESolver 


def generate_mc_vg_params(df_stats, seed=None):
    """
    Generate ONE Monte Carlo realization of VG parameters
    Returns dataframe ready for Richards solver
    """
    rng = np.random.default_rng(seed)
    df = df_stats.copy()

    # ---- Linear-space parameters ----
    df["thetas"] = rng.normal(
        df["mean_thetas"], df["std_thetas"]
    )
    df["thetar"] = rng.normal(
        df["mean_thetar"], df["std_thetar"]
    )

    # enforce physical bounds
    df["thetas"] = df["thetas"].clip(0.3, 0.7)
    df["thetar"] = df["thetar"].clip(0.0, df["thetas"] - 0.02)

    # ---- Log-space parameters ----
    log_alpha = rng.normal(
        df["mean_log10(alpha) (1/cm)"],
        df["std_log10(alpha) (1/cm)"]
    )

    log_n = rng.normal(
        df["mean_log10(n)"],
        df["std_log10(n)"]
    )

    log_ksat = rng.normal(
        df["mean_log10(Ksat) (cm/day)"],
        df["std_log10(Ksat) (cm/day)"]
    )

    # ---- Back-transform + unit conversion ----
    # alpha: 1/cm → 1/m
    df["alpha (m-1)"] = 10**log_alpha * 100.0

    # n: dimensionless
    df["N"] = 10**log_n

    # Ksat: cm/day → m/day
    df["Ksat (m/day)"] = 10**log_ksat / 100.0

    # ---- Final formatting ----
    out = df[[
        "Horizon", "Depth_min", "Depth_max",
        "thetas", "thetar",
        "alpha (m-1)", "N", "Ksat (m/day)",
        "n_eta", "layer"
    ]].reset_index(drop=True)

    return out



def generate_mc_ensemble(df_stats, Nmc , seed=223):
    rng = np.random.default_rng(seed)
    ensemble = []

    for i in range(Nmc):
        df_i = generate_mc_vg_params(df_stats, seed=rng.integers(1e9))
        df_i["MC_id"] = i
        ensemble.append(df_i)

    return pd.concat(ensemble, ignore_index=True)







def run_one_mc(mc_id, soil_params, profileData, RWUData, timeData, MetData, IniData, solver_opts,
               bottom_BC="no_flow", base_seed=223, save_detailed=True, output_dir="mc_outputs"):
    """
    Run one Monte Carlo realization.
    """
    seed = base_seed + mc_id
    SoilData_i = generate_mc_vg_params(soil_params, seed=seed)

    ProcessedOutputs, sol = RESolver(
        SoilData_i, profileData, RWUData, timeData, MetData, IniData, solver_opts,
        bottom_BC=bottom_BC
    )

    if save_detailed:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"run_{mc_id:04d}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "MC_id": mc_id,
                    "SoilData": SoilData_i,
                    "ProcessedOutputs": ProcessedOutputs,
                    "sol": sol,
                },
                f
            )

    row = {
        "MC_id": mc_id,
        "success": bool(sol.success),
        "message": str(sol.message),
        "final_storage": float(ProcessedOutputs.STORAGE[-1]),
        "total_runoff": float(np.sum(ProcessedOutputs.ROin)),
        "total_AET": float(np.sum(ProcessedOutputs.Actual_ET)),
        "theta_top_final": float(ProcessedOutputs.theta[0, -1]),
        "theta_bottom_final": float(ProcessedOutputs.theta[-1, -1]),
    }

    # optionally add sampled parameters as flattened columns
    for j, r in SoilData_i.iterrows():
        row[f"layer{j+1}_thetas"] = r["thetas"]
        row[f"layer{j+1}_thetar"] = r["thetar"]
        row[f"layer{j+1}_alpha"] = r["alpha (m-1)"]
        row[f"layer{j+1}_N"] = r["N"]
        row[f"layer{j+1}_Ksat"] = r["Ksat (m/day)"]

    return row



def RESolverMonteCarloParallel(
    soil_params,
    profileData,
    RWUData,
    timeData,
    MetData,
    IniData,
    solver_opts,
    Nmc=100,
    bottom_BC="no_flow",
    n_workers=None,
    save_detailed=True,
    output_dir="mc_outputs"
):
    """
    Parallel Monte Carlo driver.
    Returns summary dataframe.
    """
    results = []
    failed = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = { executor.submit(
                run_one_mc,
                i,
                soil_params,
                profileData,
                RWUData,
                timeData,
                MetData,
                IniData,
                solver_opts,
                bottom_BC,
                223,
                save_detailed,
                output_dir ): i
            for i in range(Nmc)
        }
        
        completed = 0   # progress counter
        start_time = time.time()

        
        for future in as_completed(futures):
            
            print(f"\rCompleted {completed}/{Nmc} simulations", end="") # print progress 
            
            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = avg_time * (Nmc - completed)
        
            print(
                f"\rCompleted {completed}/{Nmc} | "
                f"Elapsed: {elapsed/60:.2f} min | "
                f"Remaining: {remaining/60:.2f} min",
                end=""
            )
    
            
            
            i = futures[future]
            try:
                row = future.result()
                results.append(row)
                if not row["success"]:
                    failed.append((row["MC_id"], row["message"]))
            except Exception as e:
                failed.append((i, str(e)))
                results.append({
                    "MC_id": i,
                    "success": False,
                    "message": str(e)
                })

    summary_df = pd.DataFrame(results).sort_values("MC_id").reset_index(drop=True)
    summary_df.to_csv("mc_summary.csv", index=False)

    failed_df = pd.DataFrame(failed, columns=["MC_id", "message"])
    failed_df.to_csv("mc_failed.csv", index=False)

    return summary_df, failed_df


