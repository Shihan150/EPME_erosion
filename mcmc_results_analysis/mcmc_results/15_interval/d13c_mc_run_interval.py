#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codes to:
    1. inversely calculated the d13c according to the mcmc emission and erosion results
    2. run the forward model with 1000 scenarios of highest posterier probability
"""

import numpy as np
from multiprocessing import Pool
import timeit
import emcee

import time
from ptb_loscar_p_sr import parameters, init_start, model_run

from functools import wraps
import signal
import threading

import sys

import os
from os.path import join, exists
from os import makedirs

n = int(sys.argv[1])

exp_name = f'd13c_sens_test_interval'

folder = exists(exp_name)
if not folder:
    makedirs(exp_name)

samples = np.load(f'./{exp_name}.npy')
pr = np.loadtxt(f'./ln_{exp_name}.dat')

n_walkers = 70
n_steps = 12000

n_total, ndim = samples.shape
assert n_total == n_walkers * n_steps, "Shape mismatch, check n_walkers/n_steps."

# Reshape back to emcee-style chain: (n_steps, n_walkers, ndim)
chain = samples.reshape((n_steps, n_walkers, ndim))
lnprob = pr.reshape((n_steps, n_walkers))

tau = emcee.autocorr.integrated_time(chain, tol=0)

tau_max = int(np.max(tau))

print(tau_max)

burnin = 10*tau_max     # adjust if needed
thin   =  tau_max

post_chain = chain[burnin::thin, :, :]


post_chain = post_chain.reshape(-1, ndim)
np.save(f'./{exp_name}_forcing_mcmc', post_chain)
index = np.random.choice(post_chain.shape[0], size=n, replace=False)



t_eval = np.arange(-252e6, -251.9e6, 2e3)
d13c_emi = np.zeros((n, len(t_eval)))

def stop_function():
    os.kill(os.getpid(), signal.SIGINT)


def stopit_after_timeout(s, raise_exception=True):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = threading.Timer(s, stop_function)
            try:
                timer.start()
                result = func(*args, **kwargs)
            except KeyboardInterrupt:
                msg = f'function {func.__name__} took longer than {s} s.'
                if raise_exception:
                    raise TimeoutError(msg)
                result = msg
            finally:
                timer.cancel()
            return result

        return wrapper

    return actual_decorator


@stopit_after_timeout(500, raise_exception=True)

def main(i):

    theta = samples[index[i],:]

    n = int((len(theta)-2)/2)

    params = parameters(d13cin = 2, d13cvc = -1.8, svresults = 1, printflag = 0, pcycle = 2, LOADFLAG = 1, svstart = 0, cinpflag = 1, cinp = 6e3, tcspan = 6e4,  dccinp = -25, nsi = 0.4, sclim = 3)

    params['sclim'] = 4.5



    params['exp_name'] = f'{exp_name}'
    params['id'] = i
    params['RUN_TYPE'] = 0
    params['cinp_over_time'] = theta[0:n]
    params['erosion_over_time'] = theta[n:2*n]
    params['d13c_source'] = theta[2*n:]
    #             params['dccinp_inv1'] = theta[2*n]
    #             params['dccinp_inv2'] = theta[-1]
    init_start(params)
    model_run()


if __name__ == '__main__':
    start_time = timeit.default_timer()
    arg = np.arange(n)
    with Pool() as pool:
        results = list()
        iterator = pool.imap(main, arg)
        while True:
            try:
                results.append(next(iterator))
            except StopIteration:
                break
            except Exception as e:
                results.append(0)
    results = np.array(results)
    # main(0)


    elapse = timeit.default_timer() - start_time
    print(f'{elapse: .2f}s used.')
