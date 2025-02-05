#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:02:24 2025

@author: buc146
"""

import os
import subprocess
import itertools
import multiprocessing

os.chdir("/Users/buc146/OneDrive - CSIRO/pyWOMBAT/lite")

# Define experiment parameters (same as before)
latitudes = [-50]
longitudes = [240]
atm_co2_levels = [400.0]
zoo_qmort_values = [0.35]
zoo_assim_values = [0.60]
zoo_excr_values = [0.80]
zoo_grz_values = [3.0]
grazform_values = [1, 2, 3]
detrem_values = [0.20, 0.40]

experiments = list(itertools.product(
    latitudes, longitudes, atm_co2_levels, 
    zoo_qmort_values, zoo_assim_values, zoo_excr_values, 
    zoo_grz_values, grazform_values, detrem_values
))


def run_experiment(exp):
    lat, lon, atm_co2, zoo_qmort, zoo_assim, zoo_excr, zoo_grz, grazform, detrem = exp
    command = (
        f"python main.py --lat {lat} --lon {lon} --atm_co2 {atm_co2} "
        f"--zoo_qmort {zoo_qmort} --zoo_assim {zoo_assim} --zoo_excr {zoo_excr} "
        f"--zoo_grz {zoo_grz} --grazform {grazform} --detrem {detrem}"
    )
    print(f"Running: {command}")
    subprocess.run(command, shell=True)


# Run experiments in parallel
if __name__ == "__main__":
    with multiprocessing.Pool(processes=4) as pool:  # Adjust 'processes' as needed
        pool.map(run_experiment, experiments)


print("All experiments completed.")