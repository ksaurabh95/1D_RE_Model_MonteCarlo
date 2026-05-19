
# Richards Equation (RE) Model for Irish Grasslands

## Overview

This repository contains a 1D Richards Equation (RE) based numerical framework for simulating transient unsaturated flow in Irish grassland soils. The framework supports the Monte Carlo simulations of soil water dynamics using physically-based soil hydraulic functions.

The model integrates:

- Richards Equation (RE) for variably saturated flow
- van Genuchten (VG) soil hydraulic formulation
- Feddes root water uptake model
- Monte Carlo uncertainty propagation
- Finite difference spatial discretization
- Implicit time integration using `scipy.solve_ivp`
- Post-processing and visualization utilities

The framework was developed primarily for Irish grassland applications but can be adapted for other climates, crops, and soil profiles.

---

# Governing Equation

The Richards Equation solved in the model is:

###  $\frac{\partial \theta (h)}{\partial t} = \left[K(h) \left( \frac{\partial h}{\partial z} - 1 \right) \right]   -\lambda $. 
where:

- $\theta$ = volumetric water content
- $q$ = Darcy flux
- $S$ = root water uptake term

Darcy flux:
$q = -K(h)\left(\frac{\partial H}{\partial z}\right)$

where:

$H = h-z$
where:

- $K(h)$ = unsaturated hydraulic conductivity
- $h$ = pressure head
- $z$ = vertical depth coordinate

---

# Repository Structure

```text

├── RE_Model_function_files.py
├── REModelMonteCarlo.py
├── REModelMonteCarlo_functions.py
├── VGModel.py
├── PlantUptakeFunction.py
├── UtilitiesFunctions.py
├── grid_classes.py
├── MonteCarloResultAnalysis.py
├── data_johnstown.xlsx
├── README.md
└── outputs/
```
---

## Monte Carlo Simulation Framework

The Monte Carlo framework propagates uncertainty in soil hydraulic properties by:

- Randomly sampling VG parameters
- Running ensemble simulations in parallel
- Computing uncertainty bounds
- Saving individual realizations

Main script:

```python
REModelMonteCarlo.py
```

Parallel execution is implemented using:

```python
ProcessPoolExecutor
```

---

# File Descriptions

## `grid_classes.py`

Contains dataclasses used throughout the framework.

### Main Dataclasses

| Dataclass | Description |
|---|---|
| `ProfileGridSpec` | Soil profile discretization |
| `RWUSpec` | Root water uptake parameters |
| `TimeSpec` | Temporal discretization |
| `InitialCondition` | Initial pressure head configuration |
| `SolverOptions` | Numerical solver settings |
| `PostProcerssingOutputs` | Output storage structure |

---

## `VGModel.py`

Implements van Genuchten hydraulic relationships.

### Functions

#### `VGModel()`

Computes:

- Effective saturation
- Water content
- Hydraulic conductivity
- Moisture capacity

### van Genuchten Equation

$S_e = \left(1 + |\alpha h|^n\right)^{-m}$

where:

$m = 1 - \frac{1}{n}$

Water content:

$\theta = (\theta_s - \theta_r)S_e + \theta_r$

Hydraulic conductivity:

$K = K_s S_e^{\eta}\left[1-(1-S_e^{1/m})^m\right]^2$

---

## `PlantUptakeFunction.py`

Implements the Feddes root water uptake model.

### Main Components

### Functions

#### $f_1(z)$ : Root distribution function.

#### $f_2(h)$ : Plant stress response function.

#### RootUptakeModel(): Computes actual root water uptake using Feddes Uptake Equation
####  $\lambda =f_1(z)f_2(h)ET_0$

---

## `RE_Model_function_files.py`

Core implementation of the Richards Equation solver.

### Main Functions

#### `RichardsEq()`

Computes:

- Darcy fluxes
- Hydraulic gradients
- Root uptake
- Runoff
- Pressure head evolution

#### `RESolver()`

Wrapper around:

```python
scipy.integrate.solve_ivp
```

Recommended method:

```python
method="BDF"
```

#### `RichardsModelOutputs()`

Processes model outputs including:

- Soil moisture
- Pressure head
- Storage
- Actual evapotranspiration
- Water fluxes
- Runoff

---

## `UtilitiesFunctions.py`

Contains helper and visualization utilities.

### Functions

| Function | Description |
|---|---|
| `assign_vg()` | Assigns VG parameters to soil grid |
| `GetOutputAtRequiredDepths()` | Interpolates outputs |
| `plot_variable_at_depths()` | Deterministic plotting |
| `compute_mc_stats()` | Monte Carlo statistics |
| `plot_variable_at_depthsMonteCarlo()` | Ensemble visualization |

---

## `REModelMonteCarlo_functions.py`

Implements the Monte Carlo workflow.

### Features

- VG parameter sampling
- Parallel execution
- Ensemble statistics
- Automatic saving of realizations
- Failure handling

### Main Functions

| Function | Description |
|---|---|
| `generate_mc_vg_params()` | Generates VG realizations |
| `run_one_mc()` | Runs one simulation |
| `RESolverMonteCarloParallel()` | Parallel ensemble driver |

---

## `MonteCarloResultAnalysis.py`

Post-processing and visualization of ensemble results.

### Outputs

- Depth-wise soil moisture
- Pressure head uncertainty
- Water storage statistics
- Confidence intervals
- Exported figures

---

# Numerical Method

## Spatial Discretization

The soil profile is discretized using:

- One-dimensional vertical grid
- Cell-centered finite difference formulation
- Uniform spatial spacing

Fluxes are evaluated at cell boundaries using arithmetic averaging of hydraulic conductivity.

---

## Temporal Integration

Time integration uses:

```python
scipy.integrate.solve_ivp
```

Recommended solver:

```python
method='BDF'
```

because Richards Equation systems are stiff.

---

# Boundary Conditions

## Upper Boundary

The upper boundary combines:

- Rainfall infiltration
- Ponding limitation
- Surface runoff generation

Infiltration flux:

```python
q0 = min(qPond, qP)
```

where:

- `qPond` = ponded infiltration capacity
- `qP` = rainfall flux

---

## Lower Boundary Conditions

Supported options:

### No Flow

```python
bottom_BC="no_flow"
```

### Free Drainage

```python
bottom_BC="free_drainage"
```

---

# Required Input Data

## Meteorological Data

Required columns:

| Column | Units | Description |
|---|---|---|
| `rain_mm` | mm/day | Daily rainfall |
| `pet_mm_per_day` | mm/day | Potential evapotranspiration |

---

## Soil Hydraulic Parameters

| Parameter | Description |
|---|---|
| `thetas` | Saturated water content |
| `thetar` | Residual water content |
| `alpha (m-1)` | VG alpha parameter |
| `N` | VG pore-size parameter |
| `Ksat (m/day)` | Saturated hydraulic conductivity |
| `n_eta` | Tortuosity parameter |

---

# Example Usage

## Deterministic Simulation

```python
ProcessedOutputs, sol = RESolver(
    SoilData,
    profileData,
    RWUData,
    timeData,
    MetData,
    IniData,
    solver_opts,
    bottom_BC='no_flow'
)
```

---

## Monte Carlo Simulation

```python
summary_df, failed_df = RESolverMonteCarloParallel(
    soil_params=soil_params,
    profileData=profileData,
    RWUData=RWUData,
    timeData=timeData,
    MetData=MetData,
    IniData=IniData,
    solver_opts=solver_opts,
    Nmc=200,
    bottom_BC="no_flow",
    n_workers=5
)
```

---

# Output Variables

| Variable | Description |
|---|---|
| `theta` | Soil moisture |
| `h` | Pressure head |
| `K` | Hydraulic conductivity |
| `Se` | Effective saturation |
| `STORAGE` | Water storage |
| `Actual_ET` | Actual evapotranspiration |
| `PlantUptake` | Root uptake |
| `ROin` | Surface runoff |
| `Q_flux` | Water flux |

---


# Citation

If using this repository in academic work, please cite the associated manuscript and acknowledge the repository authors.

