# py-WOMBAT (1D)
## WOMBAT-lite biogeochemical model

This repository contains a **1D water column implementation** of the **WOMBAT-lite biogeochemical model**, adapted to run on the **Gadi supercomputer** at the National Computational Infrastructure (NCI), Australia.

The model is intended for use within a **Jupyter Notebook interface** on **Gadi’s Australian Research Environment (ARE)**, making it suitable for testing, development, and biogeochemical experimentation in a lightweight setting.

---

## 🧪 Model Overview

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

---

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

---

### 2. Create a custom conda environment called "pyWOMBAT_env"

```bash
module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
```
Follow [these instructions](http://climate-cms.wikis.unsw.edu.au/Conda#Creating_personal_environments) to set up your conda workspace, particularly the .condarc file in your home directory, and then run
```bash
conda env create -f py-WOMBAT.yml
```
This will create an environment called **pyWOMBAT_env** that you will need to point to when you spin-up your jupyter notebook on the ARE (next step).

---

### 3. Spin-up an ARE Jupyter notebook

Go to the [Australian Research Environment (ARE)](https://are.nci.org.au/) and click on **JupyterLab**

Make sure you charge to the correct project code, that you have enough compute and that you have access to the right project datasets...\
\
![image](https://github.com/user-attachments/assets/78a0d923-2c93-4d86-b404-3babf48babca)

Make sure that you click on **advanced settings** and point the notebook towards your custom conda environment...\
\
![image](https://github.com/user-attachments/assets/23277732-8a24-448c-ad11-08745839a9e1)

And launch!\
\
![image](https://github.com/user-attachments/assets/0ce32bbd-bf8c-4e35-b046-4adeb0f3e878)

---

### 4. Running the Standard Model

To get started:

1. Open the notebook: **`run_standard.ipynb`**
2. Execute the code cells sequentially to run the basic 1D model.
3. You can experiment with changing:

   * **Year**
   * **Latitude**
   * **Longitude**
   * **Run length** (in days)

---

The **year**, **latitude**, and **longitude** determine the environmental conditions for the 1D water column simulation. The model automatically retrieves:

* **Surface temperature**
* **Wind speeds**
* **Downward shortwave radiation** (i.e., sunlight reaching the surface)
* **Mixed layer depth**
* **Vertical velocities**

For example:

> If you set the year to **2001** and the coordinates to **30°S, 200°E**, the model will extract data for that location in the **central south Pacific** for the year 2001.

---

The forcing datasets used include

* **Atmospheric forcing**: [JRA55-do](https://climate.mri-jma.go.jp/~yukimoto/jra55do/)
* **Oceanographic fields** (e.g. mixed layer depth and vertical velocities): [BRAN2020](https://research.csiro.au/bluelink/outputs/bran/)








