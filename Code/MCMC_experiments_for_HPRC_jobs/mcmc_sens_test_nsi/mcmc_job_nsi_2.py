#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:10:08 2024

@author: shihan
"""

import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
import pandas as pd
from scipy import interpolate, optimize
from scipy.stats import norm

import matplotlib.pyplot as plt

from ptb_loscar_p_sr import parameters, init_start, model_run

import emcee

import shutil
import time

from getdist import MCSamples
import os

os.environ["OMP_NUM_THREADS"] = "1"

# read the target data
pco2_proxy = pd.read_excel('for_multi_inversion.xlsx', sheet_name = 'pCO2').to_numpy()
sr_proxy = pd.read_excel('for_multi_inversion.xlsx', sheet_name = 'Sr').to_numpy()
temp_loess = pd.read_excel('for_multi_inversion.xlsx', sheet_name = 'T_loess').to_numpy()
temp_loess_relative = pd.read_excel('for_multi_inversion.xlsx', sheet_name = 'T_loess_relative').to_numpy()
temp_meishan = pd.read_excel('for_multi_inversion.xlsx', sheet_name = 'T_Meishan').to_numpy()
d13c_loess = pd.read_excel('for_multi_inversion.xlsx', sheet_name = 'd13c_loess').to_numpy()

nsi = 0.2

exp_name = f'sens_test_nsi_{nsi}'

def log_probability(theta, exp_name = 'test', switch = 1):
    # switch: 1: loess_d18o, sr; 2: pco2, Sr; 3: loess_relative, sr
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    else:

        try:

            n = int(len(theta)/2)
            params = parameters(d13cin = 2, d13cvc = -1.8, svresults = 1, printflag = 0, pcycle = 2, LOADFLAG = 1, svstart = 0, cinpflag = 1, cinp = 6e3, tcspan = 6e4,  dccinp = -25, nsi = 0.4)
            exp_name = exp_name + str((os.getpid() * int(time.time())) % 123456789)
            params['exp_name'] = exp_name
            params['RUN_TYPE'] = 0
            params['cinp_over_time'] = theta[0:n]
            params['erosion_over_time'] = theta[n:2*n]
            params['sclim'] = 4.5
            params['nsi'] = nsi
#             params['dccinp_inv1'] = theta[2*n]
#             params['dccinp_inv2'] = theta[-1]
            init_start(params)
            model_run()

        except:

            return -np.inf



        pco2_model = pd.read_csv(f'{exp_name}/pCO2_d13c.csv').to_numpy()
        sr_model = pd.read_csv(f'{exp_name}/Sr.csv').to_numpy()
        temp_model = pd.read_csv(f'{exp_name}/tcb.csv', usecols=[0,1,2]).to_numpy()
#         d13c_model = pd.read_csv(f'{exp_name}/Surface_dic_alk_d13c_ph_dox.csv', usecols=[0,3]).to_numpy()


        fpco2 = interpolate.interp1d(pco2_model[:,0], pco2_model[:,1],  bounds_error = False,  fill_value = 0)
        fsr = interpolate.interp1d(sr_model[:,0], sr_model[:,1],  bounds_error = False,  fill_value = 0)
        ftemp_loess = interpolate.interp1d(temp_model[:,0], temp_model[:,1] - temp_model[0,1],  bounds_error = False,  fill_value = 'extrapolate')
#         fd13c = interpolate.interp1d(d13c_model[:,0], d13c_model[:,1],  bounds_error = False,  fill_value = 'extrapolate')


        pr_sr = np.average(norm.logpdf(fsr(sr_proxy[:,0])*1e5, sr_proxy[:,1]*1e5, sr_proxy[:,2]*1e5))

#         pr_d13c = np.average(norm.logpdf(fd13c(d13c_loess[:,0]), d13c_loess[:,1], d13c_loess[:,2]))

        if switch == 1:
            pr_temp = np.average(norm.logpdf(ftemp_loess(temp_loess[:,0]), temp_loess[:,1], temp_loess[:,2]))

            sum_diff = pr_sr + pr_temp

        elif switch == 2:
            pr_pco2 = np.average(norm.logpdf(fpco2(pco2_proxy[:,0]), pco2_proxy[:,1], pco2_proxy[:,2]))
            sum_diff = pr_sr + pr_pco2

        shutil.rmtree(params['exp_name'])




        return lp + sum_diff


def log_prior(theta):
    n = int(len(theta)/2)
    cinp_over_time = np.zeros(n)
    erosion_over_time = np.zeros(n)

    cinp_over_time = theta[0:n]
    erosion_over_time = theta[n:2*n]
#     d13c_over_time = theta[2*n:]

    if (cinp_over_time<=0.8).all() and (cinp_over_time>=0.01).all() and (erosion_over_time<=5).all() and (erosion_over_time>=0.6).all() :
        return 0.0
    return -np.inf

if __name__ == '__main__':
    # theta_init = np.array([  0.15714361, 0.15,  0.12167933,  0.12,  0.3869759 , 0.38,  0.57921896, 0.57,  0.52326151, 0.52 ,  2.34762547, 2.4,   3.55339855, 3.5,  1.86813788,1.8,  0.82641922, 0.8,  0.78367564, 0.8])

    theta_init = np.array([  0.15714361,   0.12167933,    0.3869759 ,  0.57921896,   0.52326151,   2.34762547,    3.55339855,   1.86813788,  0.82641922,   0.78367564])

    n_walker = 25
    n_step = 4000

    ndim = len(theta_init)

    pos = np.array(theta_init) +  1e-2 * np.random.randn(n_walker, ndim)
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args = ('mcmc_temp_sr', 1), pool = pool)
        pos, lnprob, rstate = sampler.run_mcmc(pos, n_step, progress=True)




    samples = sampler.get_chain()


    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    np.save(f'{exp_name}',flat_samples)
    # np.savetxt("mcmc_temp_chain.txt", flat_samples ,fmt='%f',delimiter=',')
    np.savetxt(f'ln_{exp_name}.dat', sampler.get_log_prob(flat = True))




    # print(sampler.get_autocorr_time())

    #  xxx = MCSamples(samples=np.load('xxx_flat_samples chain.npy'),names =name1, labels = label1)
