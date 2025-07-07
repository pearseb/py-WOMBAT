# py-WOMBAT (1D) – WOMBAT-lite Biogeochemical Model for Gadi

This repository contains a **1D water column implementation** of the **WOMBAT-lite biogeochemical model**, adapted to run on the **Gadi supercomputer** at the National Computational Infrastructure (NCI), Australia.

The model is intended for use within a **Jupyter Notebook interface** on **Gadi’s Australian Research Environment (ARE)**, making it suitable for testing, development, and biogeochemical experimentation in a lightweight setting.

---

## 🧪 Model Overvieww

- **WOMBAT-lite** is a reduced-complexity ocean biogeochemical model, focusing on key oceanic biogeochemical processes.
- This 1D version simulates vertical dynamics (mixing, sinking, remineralisation) and biological activity in a single water column.
- Ideal for prototyping new parameterisations, testing trait-based formulations, or running sensitivity experiments.
- Tracers include:
  - Nitrate (no3) [mmolN/m3]
  - Dissolved Iron (dfe) [mmolFe/m3]
  - Phytoplankton carbon biomass (phy) [mmolC/m3]
  - Zooplankton carbon biomass (zoo) [mmolC/m3]
  - Detritus carbon biomass (det) [mmolC/m3]
  - Phytoplankton chlorophyll biomass (pchl) [mg/m3]
  - Phytoplankton iron biomass (phyfe) [mmolFe/m3]
  - Zooplankton iron biomass (zoofe) [mmolFe/m3]
  - Detritus iron biomass (detfe) [mmolFe/m3]

---

## 🚀 Getting Started

### Prerequisites

- Access to the [Gadi supercomputer](https://opus.nci.org.au/display/Help/Gadi+User+Guide)
- An active NCI project allocation to gb6, qv56 --> these are for access to the forcing files (e.g., surface temperature, etc.)
- Access to the [Australian Research Environment (ARE)](https://are.nci.org.au/)
- A Gadi-compatible environment with required Python dependencies (see below)

### 1. Clone the Repository on Gadi

Once on Gadi:

```bash
git clone https://github.com/pearseb/py-WOMBAT.git
cd py-WOMBAT
git checkout pyWOMBAT-on-Gadi
```

I would recommend making a new experimental branch for yourself where developments and experiments can be run
```bash
git checkout -b my_new_branch
```

### 2. Create a custom conda environment called "pyWOMBAT_env"

```bash
module use /g/data/hh5/public/modules
```
Follow [these instructions](http://climate-cms.wikis.unsw.edu.au/Conda#Creating_personal_environments) to create your conda environment, and then run
```bash
conda env create -f py-WOMBAT.yml
```

### 3. Spin-up an ARE Jupyter notebook

Go to the [Australian Research Environment (ARE)](https://are.nci.org.au/) and click on **JupyterLab**


### 4. Run the model

Open the **run_standard.ipynb** notebook and execute the code chunks to run the basic model. You can play around with changing the year, latitude, longitude, and run length.

The year, latitude and longitude changes the conditions of the 1D water column. The code works by looking for the relevant surface temperature, wind speeds, downward shortwave radiation (i.e., incident light), mixed layer depths and vertical velocities at that location for the given year. So, if 2001 is chosen and a latitude-longitude point of 30S and 200E, then the code will extract the conditions at this location in the South Pacific for the year 2001. NOTE that these data come from the JRA55do and BRAN2020 (in the case of mixed layer depth and vertical velocities).








