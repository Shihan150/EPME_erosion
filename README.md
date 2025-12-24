# EPME_erosion
This repository contains scripts, proxy compilations, and MCMC inversion workflows used to quantify erosion forcing, carbon emissions, and associated environmental changes across the End-Permian Mass Extinction (EPME). (Li et al., 2025, in revision)

## Citation

If you use this repository, please cite:
Li et al. (2025), Erosion-driven delayed warming and environmental change across the End-Permian mass extinction, in revision.

## Contact

For questions or issues, please contact:
Shihan Li — shihan@tamu.edu
Texas A&M University



## Repository Structure
```
├── S1_Early_PTB.xlsx              # Proxy data compilation
├── hprc_job_code/                # Scripts for running MCMC jobs on HPC
├── mcmc_results_analysis/        # Post-processing and forward modeling
├── README.md
```


## Proxy data compilation
The file ***S1_Early_PTB.xlsx*** contains a compilation of proxy datasets across the EPME, including:
* carbonate δ13C
* conodont δ18O
* pCO2  estimates
* boron-based pH
* 87Sr/86Sr ratios

LOESS-smoothed δ13C and δ18O curves and references are also provided.

These datasets are used as targets in the Bayesian MCMC inversion.

---

## Workflow Overview

The workflow consists of four main stages:

1. **Run MCMC inversions on HPRC**
2. **Post-process MCMC chains**
3. **Generate forward model ensembles**
4. **Plot figures and summarize results**

---

## 1. MCMC Experiments on HPRC

Scripts in `hprc_job_code/` define **17 MCMC experiments**  
(1 baseline + 16 sensitivity tests).

- Baseline experiment uses:
  - **48 walkers**
  - **8000 production steps** (after a short warm-up)
  - Requires **48 CPU cores**
  - Runtime: ~5–6 hours per experiment

Each job outputs:
- `ln_{exp_name}.dat` → log-posterior probabilities  
- `{exp_name}.npy` → MCMC samples of all parameters  

These files should be downloaded and saved to:

mcmc_results_analysis/mcmc_results/

---

## 2. Post-processing MCMC Results

Inside `mcmc_results_analysis/`, scripts analyze chains and sample posterior scenarios.

For example, scripts like:

d13c_mc_run_sclim.py

are used to:
- Read MCMC outputs
- Randomly select **1000 posterior samples**
- Run the forward model for each scenario
- Save outputs to:

mcmc_forward_results/

Each folder contains 1000 forward runs of environmental variables.

---

## 3. Forward Ensemble Simulations

The 1000 posterior scenarios are used to derive time series and uncertainties for:

- Surface temperature
- Seawater 87Sr/86Sr
- Surface δ13C
- Surface pH
- Atmospheric pCO2
- Ocean oxygen

and more. 

These results form the basis for uncertainty envelopes and quantile statistics.

---

## 4. Plotting and Visualization

The notebook:

Fig_plot_v2.ipynb

is used to:

- Load forward ensemble results
- Compute quantiles
- Save processed data tables
- Generate publication figures

Figures are saved to:

figs/

---

## Example: Baseline Experiment

To reproduce the baseline case (`exp_name = d13c_sens_test_sclim_4`):

1. **Submit job on HPRC**

hprc_job_code/1_2_sclim/

2. **Download outputs** and place into:

mcmc_results_analysis/mcmc_results/

Files:
- `ln_d13c_sens_test_sclim_4.dat`
- `d13c_sens_test_sclim_4.npy`

3. **Run forward ensemble**
```bash
cd mcmc_results_analysis
python d13c_mc_run_sclim.py 1000 4
```

4.	Move forward results
```bash
mv d13c_sens_test_sclim_4 ./mcmc_forward_results/
```

5.	Plot results  
	•	Open Fig_plot_v2.ipynb   
	•	Set: exp_name = 'd13c_sens_test_sclim_4'   
	•	Run all cells to generate figures and data tables.  
 



