# EPME_erosion
Codes for investigating the environmental dynamics across the end-Permian mass extinction (Li et al., 2025, in revision)

```
├── S1_Early_PTB.xlsx
├── Code
│   ├── Fig_plot_v2.ipynb
│   ├── MCMC_experiments_for_HPRC_jobs
│   ├── MCMC_results_analysis
├── MCMC_results
│   ├── forcing_{exp_name}_results.xlsx
│   ├── mcmc_{exp_name}_results.xlsx
│   ├── mcmc_{exp_name}_quantile_results.xlsx
```

### Proxy data compilation
The file ***petm_steady.dat*** contains a compilation of proxy datasets across the EPME, including:
* carbonate δ¹³C
* conodont δ¹⁸O
* pCO₂  estimates
* boron-based pH
* ⁸⁷Sr/⁸⁶Sr ratios

Additionally, the file includes LOESS-smoothed δ¹³C and δ¹⁸O curves alongside the corresponding references.

### Markov chain Monte Carlo (MCMC) experiments
We conducted an ensemble of seven MCMC experiments, each using 25 walkers and 4,000 model steps.


This totals 100,000 forward model calls per experiment. These numerical experiments were performed on the advanced computing infrastructure provided by Texas A&M High Performance Research Computing (HPRC).

#### File and Code Structure
* MCMC configuration
The configuration files for all MCMC experiments can be found in the Code/MCMC_experiments_for_HPRC_jobs folder.
* MCMC results and analysis
The MCMC experiments produce forcing terms for all forward model calls. These are used to derive posterior distributions for forcing parameters and key model outputs.

    * Analysis codes are located in Code/MCMC_results_analysis.
    * Processed results are saved in the MCMC_results folder.
* Figure Generation:
Figures used in the main text and summarized modeling results are generated using the Jupyter notebook Code/Fig_plot_v2.ipynb.

