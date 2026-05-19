# 1D Richards Equation (RE) Model 

### Project Description
This repository contains a 1D Richards Equation (RE) based numerical model for simulating transient unsaturated water flow in Irish grassland soils. 
The model integrates:

- Soil hydraulic properties using the van Genuchten (VG) formulation
- Meteorological forcing (rainfall and potential evapotranspiration)
- Root water uptake using the Feddes approach
- Numerical integration using `scipy.solve_ivp`
- Post-processing utilities for visualization 

The framework is designed primarily for Irish grassland applications but can be adapted for other locations.

---

# Repository Structure

```text
.
├── RE_Model.py
├── RE_Model_function_files.py
├── VGModel.py
├── PlantUptakeFunction.py
├── UtilitiesFunctions.py
├── grid_classes.py
├── data_johnstown.xlsx
├── README.md
└── outputs/
```

---

# File Descriptions

## 1. RE_Model.py
Main execution script for the Richards Equation model.

### Responsibilities
- Reads meteorological and soil hydraulic data
- Defines simulation settings
- Initializes model parameters
- Runs the RE solver
- Saves outputs
- Generates plots

### Main Components

#### Input Data
```python
MetData = pd.read_excel('data_johnstown.xlsx',sheet_name='met_data')
SoilData = pd.read_excel('data_johnstown.xlsx',sheet_name='VGParams_Rosetta')
```

#### Grid Definition
```python
profileData = ProfileGridSpec(zmin=0, zmax=2, dz=0.02)
```
#### Root water uptake parameters
```python
RWUData = RWUSpec(
    psi_a=-0.05,  # critical pressure heads associated with anaerobiosis,
    psi_d=-4,   # critical pressure heads associated with soilwater-limited evapotranspiration
    psi_w=-150,  # # critical pressure head associated with plant wilting
    Lr= 1   # m # depth of root zone
    ) 
```

#### Run time details and time steps
```python
timeData = TimeSpec(
    tmin = 0,
    tmax = len(MetData),
    dt = 1 ,  #in day 
    )

```
#### Initial Conditions
```python
IniData = InitialCondition(
    z_wt=0.3,
    depth=profileData.depth,
    RO0=0.0
)
```

#### Solver Execution
```python
ProcessedOutputs, sol = RESolver(...)
```

---

## 2. RE_Model_function_files.py
Core numerical implementation of the Richards Equation solver.

### Main Functions

#### RichardsEq()
Computes the time derivative of pressure head.

### Responsibilities
- Computes hydraulic properties
- Calculates Darcy fluxes
- Applies boundary conditions
- Computes root water uptake
- Computes runoff
- Forms ODE system for solver

### Governing Equation
The Richards Equation is given by: 

###  $\frac{\partial \theta (h)}{\partial t} = \left[K(h) \left( \frac{\partial h}{\partial z} - 1 \right) \right]   -\lambda $. 
where:

- $\theta$ = volumetric water content
- $q$ = Darcy flux
- $S$ = root water uptake term

Darcy flux:
$q = -K(h)\left(\frac{\partial H}{\partial z}\right)$

where:

$H = h-z$

---

#### RESolver()
Wrapper around `scipy.integrate.solve_ivp`.

### Responsibilities
- Assigns soil hydraulic properties
- Configures numerical solver
- Runs time integration
- Calls post-processing routines

---

#### RichardsModelOutputs()
Post-processing function.

### Outputs
- Pressure head
- Soil moisture
- Effective saturation
- Hydraulic conductivity
- Root water uptake
- Actual evapotranspiration
- Runoff
- Storage

---

## 3. VGModel.py
Implements van Genuchten hydraulic functions.

### Functions

#### VGModel()
Computes:

- Effective saturation \($S_e$\)
- Hydraulic conductivity \(K\)
- soil moisture content \($\theta\$)
- Specific moisture capacity \(C\)

### van Genuchten Equation

$S_e = \left(1 + |\alpha h|^n\right)^{-m}$

where:

$m = 1 - \frac{1}{n}$

Water content:

$\theta = (\theta_s - \theta_r)S_e + \theta_r$

Hydraulic conductivity:

$K = K_s S_e^{\eta}\left[1-(1-S_e^{1/m})^m\right]^2$

---

#### VGfromSe()
Computes pressure head and conductivity from effective saturation.

Used primarily for ponded boundary conditions.

---

## 4. PlantUptakeFunction.py
Implements Feddes root water uptake model.

### Functions

#### $f_1(z)$ : Root distribution function.

#### $f_2(h)$ : Plant stress response function.

#### RootUptakeModel(): Computes actual root water uptake using Feddes Uptake Equation
####  $\lambda =f_1(z)f_2(h)ET_0$

where:

- $f_1$ = root distribution function
- $f_2$ = water stress function
- $ET_0$ = potential evapotranspiration

---

## 5. UtilitiesFunctions.py
Contains utility and plotting functions.

### Functions

#### assign_vg()
Assigns VG parameters to each soil grid cell.

#### GetOutputAtRequiredDepths()
Interpolates outputs at requested depths.

#### plot_variable_at_depths()
Plots variables versus time.

---

## 6. grid_classes.py
Contains all simulation configuration dataclasses.

### Dataclasses

#### ProfileGridSpec
Defines soil discretization.

#### RWUSpec
Defines root water uptake parameters.

#### TimeSpec
Defines temporal discretization.

#### InitialCondition
Defines initial pressure head.

#### SolverOptions
Defines ODE solver parameters.

#### PostProcerssingOutputs
Stores model outputs.

---

# Model Workflow

```text
Meteorological Data
        ↓
Soil Hydraulic Parameters
        ↓
Grid Generation
        ↓
Initial Conditions
        ↓
Richards Equation Solver
        ↓
Root Water Uptake
        ↓
Boundary Conditions
        ↓
ODE Integration
        ↓
Post Processing
        ↓
Plots and Outputs
```

---

# Numerical Method

## Spatial Discretization
The soil profile is discretized using a finite difference grid.

### Grid Representation
- Cell-centered nodes
- Uniform spacing
- Vertical one-dimensional domain

### Flux Computation
Fluxes are evaluated at cell boundaries using arithmetic averaging of hydraulic conductivity.

---

## Temporal Integration
The model uses:

```python
scipy.integrate.solve_ivp
```

Recommended solver:

```python
method='BDF'
```

because Richards Equation is stiff.

---

# Boundary Conditions

## Upper Boundary
The upper boundary condition combines:

- Rainfall infiltration
- Ponding limitation
- Surface runoff generation

### Infiltration Condition

```python
q0 = min(qPond, qP)
```

where:

- `qPond` = maximum infiltration under ponded conditions
- `qP` = rainfall flux

---

## Lower Boundary
Supported lower boundary conditions:

### Free Drainage
```python
bottom_BC='free_drainage'
```

### No Flow
```python
bottom_BC='no_flow'
```

---

# Required Input Data

## Meteorological Data
The meteorological input file must contain:

| Column | Units | Description |
|---|---|---|
| rain_mm | mm/day | Daily rainfall |
| pet_mm_per_day | mm/day | Potential evapotranspiration |

---

## Soil Hydraulic Parameters

| Parameter | Description |
|---|---|
| $\theta_s$ | Saturated water content |
| $\theta_r$ | Residual water content |
| $\alpha$ | VG alpha parameter |
| $n$ | VG pore-size parameter |
| $K_{sat}$ | Saturated hydraulic conductivity |
| $\eta$ | Tortuosity/connectivity parameter |

---

# Example Usage

## Basic Simulation

```python
profileData = ProfileGridSpec(zmin=0, zmax=2, dz=0.02)

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

# Output Variables

| Variable | Description |
|---|---|
| theta | Soil moisture |
| h | Pressure head |
| K | Hydraulic conductivity |
| Se | Effective saturation |
| STORAGE | Total water storage |
| Actual_ET | Actual evapotranspiration |
| PlantUptake | Root water uptake |
| ROin | Surface runoff |
| Q_flux | Water flux |

---
