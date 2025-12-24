#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:35:50 2023

@author: Shihan Li

Simplified LOSCAR for PTB
1. The sediment box is closed. CaCO3 is precipitated exclusively over the shallow sediment boxes (driven by saturation state; equation (6)),
rendering the deep sediments free of calcite, meaning that calcite is accumulated only in shallow waters.
2. The Tetheys bathemetry is adjusted.
3. P cycle is added following Komar and Zeebe 2017.

# update log:
08/26 Add the d13c soucre of inversion


Please also refer to: PTB paper link

Original reference:
   iLOSCAR paper link

   Zeebe, R. E., Zachos, J. C., and Dickens, G. R. Carbon dioxide
   forcing alone insufficient to explain Paleocene-Eocene Thermal
   Maximum warming. Nature Geoscience, doi:10.1038/NGEO578 (2009)

Based on the .py version
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import root_scalar, toms748
from scipy.integrate import solve_ivp
from numba import jit
import matplotlib.pyplot as plt

import platform
from os.path import join, exists
from os import makedirs

import timeit
import time


ODE_SOLVER = 'BDF'



    # return ysol

def parameters(exp_name = 'test', LOADFLAG = 0, RUNTYPE = 1, pcycle = 1, svstart = 0, svresults = 1, printflag = 1,
               t0 = -252.00e6, tfinal = -251.899e6, pref = 425, pcsi = 425, fepl = 0.8, eph = 1.8, rrain = 1e50,
               fsi0 = 4.2, finc0 = 9.8, d13cin = 3, d13cvc = -2, thc = 25, cac = 0.013, mgc = 0.042, sclim = 3, nsi = 0.2, ncc = 0.4,
               cinpflag = 0, scale_factor = 'scale_factor.csv', initfile = 'ptb_steady.dat', cinp = 3000, dccinp = -55, tcin0 = 0, tcspan = 6e3, cinpfile = 'emi_file.dat',
               svfile = 'ptb_steady.dat', target_file = 'ptb_d13c.csv', toms_low = -0.05, toms_high = 3, dccinp_inv1 = -5, dccinp_inv2 = -5, d13c_source = [-20, -20]
               ):
    params = {
        'exp_name': exp_name,
        'LOADFLAG': LOADFLAG,
        'RUN_TYPE': 1,              # 1: forward, 2: inverse, 0: multiple proxy optimize, 3: d13c inverse and export results for plotting
        'pcycle': pcycle,

        'svstart': svstart,               # save the final y
        'svresults': svresults,     # save the modeling results
        'printflag': printflag,

        't0': t0,
        'tfinal': tfinal,
        'pref': pref,
        'pcsi': pcsi,
        'fepl': fepl,
        'eph': eph,
        'rrain': rrain,

        'fsi0': fsi0,
        'finc0': finc0,

        'd13cin': d13cin,
        'd13cvc': d13cvc,
        'thc': thc,
        'cac': cac,
        'mgc': mgc,
        'sclim': sclim,
        'nsi': nsi,
        'ncc': ncc,

        'cinpflag': cinpflag,            # 0/1/2/3

        # scale factors for the flux
        # 'scale_factor': scale_factor,


        ################  Required when initflag = 1
        'initfile': initfile,

        ################  Required when cinpflag = 1

        'cinp': cinp,             # carbon release in Gt
        'dccinp': dccinp,            # d13c of emitted carbon in per mil
        'tcin0': tcin0,               # start of emission
        'tcspan': tcspan,            # duration of emission

        ###############  Required when cinpflag = 2 or 3
        'cinpfile': cinpfile,

        ###############  Required when svstart == 1
        'svfile': svfile,

        ### Inv ###

        'target_file': target_file,

        'toms_low':  toms_low,
        'toms_high': toms_high,
        'dccinp_inv1': dccinp_inv1,        # d13c for the first
        'dccinp_inv2': dccinp_inv2,

        # cinp and erosion over time
        'cinp_over_time': [0,0, 0],
        'erosion_over_time': [1,1,1],
        'd13c_source': d13c_source,

        'id': 0

        }

    return params


def init_start(params):
    global RUN_TYPE
    RUN_TYPE = params['RUN_TYPE']


    if RUN_TYPE == 2:
        global target_file, toms_low, toms_high, dccinp_inv1, dccinp_inv2
        target_file = params['target_file']
        toms_low = params['toms_low']
        toms_high = params['toms_high']
        dccinp_inv1 = params['dccinp_inv1']
        dccinp_inv2 = params['dccinp_inv2']

    # global fscale_bio_pump, fscale_silw, fscale_carbw, fscale_shelf_calcification

    global exp_name
    exp_name = params['exp_name']
    # scale_factor = pd.read_csv(params['scale_factor'])
    # fscale_silw = interpolate.interp1d(scale_factor.iloc[:,0].values,scale_factor.iloc[:,1].values, bounds_error = False, fill_value = 1)


    # fscale_carbw = interpolate.interp1d(scale_factor.iloc[:,0].values,scale_factor.iloc[:,2].values, bounds_error = False, fill_value = 1)

    # fscale_shelf_calcification = interpolate.interp1d(scale_factor.iloc[:,0].values,scale_factor.iloc[:,3].values, bounds_error = False, fill_value = 1)

    # fscale_bio_pump = interpolate.interp1d(scale_factor.iloc[:,0].values,scale_factor.iloc[:,4].values, bounds_error = False, fill_value = 1)
    erosion_factor = params['erosion_over_time']

    global fscale_erosion
    if len(erosion_factor) == 3:
        fscale_erosion =interpolate.interp1d([-252e6, -251.952e6, -251.94e6, -251.899e6], [erosion_factor[0], erosion_factor[1], erosion_factor[2], 1], bounds_error = False, kind = 'zero', fill_value = 1)
    elif len(erosion_factor) == 5:
        fscale_erosion =interpolate.interp1d([-252e6, -251.976e6, -251.954e6, -251.94e6, -251.92e6, -251.899e6], [erosion_factor[0], erosion_factor[1], erosion_factor[2], erosion_factor[3], erosion_factor[4],1], bounds_error = False, kind = 'zero', fill_value = 1)
    elif len(erosion_factor) == 6:
        fscale_erosion =interpolate.interp1d([-252e6, -251.982e6, -251.972e6, -251.962e6, -251.944e6, -251.92e6,  -251.90e6], [erosion_factor[0], erosion_factor[1], erosion_factor[2], erosion_factor[3], erosion_factor[4], erosion_factor[5], 1], bounds_error = False, kind = 'zero', fill_value = 1)
    elif len(erosion_factor) == 10:
        fscale_erosion =interpolate.interp1d([-252e6, -251.99e6,  -251.98e6, -251.97e6, -251.96e6, -251.95e6, -251.94e6, -251.93e6, -251.92e6, -251.91e6,  -251.899e6], [erosion_factor[0], erosion_factor[1], erosion_factor[2], erosion_factor[3], erosion_factor[4], erosion_factor[5],erosion_factor[6], erosion_factor[7], erosion_factor[8], erosion_factor[9],1], bounds_error = False, kind = 'zero', fill_value = 0)

    else:
        fscale_erosion =interpolate.interp1d([-252e6,  -251.9e6], [1,1], bounds_error = False, kind = 'zero', fill_value = 1)

    global FTYS
    FTYS = 1

    global LOADFLAG, svstart, pcycle, svresults, printflag

    LOADFLAG = params['LOADFLAG']
    pcycle = params['pcycle']
    svstart = params['svstart']
    svresults = params['svresults']
    printflag = params['printflag']

    global t0, tfinal, pref, pcsi, fepl, eph0, rrain,  fvc0, finc0, d13cin, d13cvc, thc0, cac, mgc, sclim, nsi, ncc
    t0 = params['t0']
    tfinal = params['tfinal']
    pref = params['pref']
    pcsi = params['pcsi']
    fepl = params['fepl']
    eph0 = params['eph']
    rrain = params['rrain']

    fsi0 = params['fsi0']*1e12
    finc0 = params['finc0']*1e12

    d13cin = params['d13cin']
    d13cvc = params['d13cvc']
    thc0 = params['thc']
    cac = params['cac']
    mgc = params['mgc']
    sclim = params['sclim']
    nsi = params['nsi']
    ncc = params['ncc']


    global cinpflag
    cinpflag = params['cinpflag']

    global RST
    RST = 0.011

    global rccinp, frccinp
    if cinpflag != 0:
        global fcinp, frccinp

    if cinpflag == 1:
        cinp = params['cinp']
        dccinp = params['dccinp']
        rccinp = ((dccinp/1e3)+1)*RST
        tcin0 = params['tcin0']
        tcspan = params['tcspan']
        fcinp = interpolate.interp1d([tcin0, tcin0+tcspan],[cinp/tcspan, cinp/tcspan], fill_value=(0,0), bounds_error=False)
        frccinp = interpolate.interp1d([tcin0, tcin0+tcspan],[rccinp, rccinp], fill_value=(0,0), bounds_error=False)


    if LOADFLAG == 1:
        global initfile
        initfile = params['initfile']

    if cinpflag == 2 or cinpflag == 3:
        cinpfile = params['cinpfile']

        co2_emi = np.loadtxt(cinpfile)
        tems = co2_emi[:,0]
        ems = co2_emi[:,1]

        if cinpflag == 2:
            dccinp = params['dccinp']
            rccinp = (dccinp/1e3+1)*RST
            frccinp = interpolate.interp1d([t0, tfinal],[rccinp, rccinp], fill_value=(0,0), bounds_error=False)
        else:
            dccinp = co2_emi[:,2]
            rccinp = (dccinp/1e3+1)*RST
            frccinp = interpolate.interp1d(tems,rccinp, fill_value=(0,0), bounds_error=False)


        fcinp = interpolate.interp1d(tems, ems, fill_value = (0,0), bounds_error = False)

    if RUN_TYPE == 0 or RUN_TYPE == 3:
        cinp = params['cinp_over_time']
        if len(cinp) == 3:
            fcinp = interpolate.interp1d([-252e6, -251.952e6, -251.94e6, -251.899e6], [cinp[0], cinp[1], cinp[2], 0], bounds_error = False, kind = 'zero', fill_value = 0)
        elif len(cinp) == 5:
            fcinp = interpolate.interp1d([-252e6, -251.976e6, -251.954e6, -251.94e6, -251.92e6, -251.899e6], [cinp[0], cinp[1], cinp[2], cinp[3], cinp[4],0], bounds_error = False, kind = 'zero', fill_value = 0)
        elif len(cinp) == 6:
            fcinp = interpolate.interp1d([-252e6, -251.982e6, -251.972e6, -251.962e6, -251.944e6, -251.92e6, -251.90e6], [cinp[0], cinp[1], cinp[2], cinp[3], cinp[4], cinp[5], 0], bounds_error = False, kind = 'zero', fill_value = 0)
        elif len(cinp) == 10:
            fcinp = interpolate.interp1d([-252e6, -251.99e6,  -251.976e6, -251.97e6, -251.96e6, -251.954e6, -251.94e6, -251.93e6, -251.92e6, -251.91e6,  -251.899e6], [cinp[0], cinp[1], cinp[2], cinp[3], cinp[4], cinp[5],cinp[6], cinp[7], cinp[8], cinp[9],0], bounds_error = False, kind = 'zero', fill_value = 0)

        rccinp = (dccinp/1e3+1)*RST
        # dccinp_inv1 = params['dccinp_inv1']
        # dccinp_inv2 = params['dccinp_inv2']
        # frccinp = interpolate.interp1d([-252e6, -251.952e6, -251.90e6],[((dccinp_inv1/1e3)+1)*RST, ((dccinp_inv2/1e3)+1)*RST, 0], fill_value=(0,0), kind = 'zero',  bounds_error=False)
        frccinp = interpolate.interp1d([t0, tfinal],[rccinp, rccinp], fill_value=(0,0), bounds_error=False)
        d13c_source = params['d13c_source']
        frccinp = interpolate.interp1d([t0, -251.954e6, tfinal],[((d13c_source[0]/1e3)+1)*RST, ((d13c_source[1]/1e3)+1)*RST, 0], fill_value=(0,0), kind = 'zero',  bounds_error=False)

    global id
    id = params['id']

    # if RUN_TYPE == 3:
    #
    #
    #     global id
    #     id = params['id']
    #     target_file = params['target_file']
    #
    #     LOADFLAG = 1






    if svstart == 1:
        global svfile
        svfile = params['svfile']

    # global constants, here only simplified for FTYS = 1

    global CNTI, VOC, AOC, HAV, RHO, SALOCN, YTOSC, MMTOM, PPMTOMMSQ, REDPC, REDNC, REDO2C, TKLV, HGUESS, HIMAX, HCONV, NCSWRT, BORT, CAM, MGM, ALPKC

    RST = 0.011
    CNTI = 0.01   # scaling
    VOC = 1.29e18 # m3 volume ocean, ptb; (Winguth and Winguth, 2012; Jurikova et al., 2020)
    AOC = 3.63e14        # m2 area ocean, ptb
    HAV = VOC / AOC        # m average depth
    RHO = 1.025e3        # kg/m3 seawater
    # RHOS = 2.50e3        # kg/m3 sediment
    SALOCN = 35.2        # psu ocean salinity, ptb; (Osen, 2014)

    YTOSEC = 3600 * 24 * 365 # years to secs
    MMTOM = 1e-3         # mmol to mol
    PPMTOMMSQ = 2.2e15 / 12 / AOC # ppmv to mol/m2

    # biological pump
    REDPC = 1 / 130        # Redfield P:C
    REDNC = 15 / 130       # Redfield N:C
    REDO2C = 165 / 130     # Redfield O2:C


    # co2 chemistry
    TKLV = 273.15        # TC -> TKelvin
    HGUESS = 1e-8        # H+ first guess
    HIMAX = 50           # H+ max iterations
    HCONV = 1e-4         # H+ rel. accuracy
    NCSWRT = 5           # calc csys output vars
    # HALK2DIC = 1.6       # high alk/dic ratio
    BORT = 432.5         # 416 DOE94, 432.5 Lee10

    # Ca,Mg
    CAM = 10.3e-3        # mol/kg modern Ca
    MGM = 53.0e-3        # mol/kg modern Mg
    ALPKC = 0.0833       # Slope Mg/Ca on Kspc

    global TAUS, TAUI, TAUD, TSCAL, PMMK, PMM0, PMMV, KMMOX, ALPRSTHA13_CHOICE, MOOK
    TAUS = 20
    TAUI = 200
    TAUD = 1000
    TSCAL = 10

    PMMK = 0.1e-3
    PMM0 = 0.1e-3
    PMMV = 1 + PMM0 / PMMK


    KMMOX = 1e-3

    ALPRSTHA13_CHOICE = 'MOOK'
    MOOK = 1



    global NOC, NB, NLS, KTY, KOC, NOCT, NCTM, NCCATN, NHS, NTS, NOATM, NEQ

    NOC = 4
    NB = 13
    NLS = 4
    KTY = 1

    KOC = NOC + KTY
    NOCT = 6          # Li and Ca Coupled, if possible. Then NOCT = 8
    NCATM = 1
    NCCATM = 1
    NHS = 1
    NTS = NLS + NHS
    NOATM = NOCT * NB + NCATM + NCCATM

    NEQ = NOCT * NB + NCATM + NCCATM + 2  # 2 for sr cycle


    ####  Initialize the model
    # """ =========     Silicate Weathering: volc degass   =======""" #

    global fconv, thc, tto, thbra, thbri, gtha, gthi, tso

    # Note: 5e12 is steady-state degassing @280 uatm,
    # balanced by steady-state Si weathering @280 uatm.
    # change fvc(t) in later
    fvc0 = (fsi0 / AOC) * (pcsi / pref)**nsi

    # CaCO3 influx
    finc0 = ((finc0) * (pcsi / pref)**ncc) / AOC

    # area fractions

    #area fractions      A    I     P   T
    fanoc = np.array([0.15,0.14,0.52,0.09])

    fdvol = np.array([ 0.16,0.16,0.68])  #deep volume fractions
    fhl = 0.1  # high box area fraction

    # height          L   I     D
    hlid = np.array([100,900,HAV-1000])


    """ =======     Temperature     =========         """
    tc3 = np.array([22,12,5])  #AIP
    tct = np.array([18,14,12])  #TETHYS

    """ ======   Mixing paratmeters =======       """

    mv0 = np.array([3.5,3.5,7.0,3.2,2])*1e6  # low <-> intermediate box
    mhd0 = np.array([4.0,4.0,6,0.7])*1e6     # deep boxes <--> H-box

    """ ======  fraction EHP, remineralized in deep A I P boxes ====="""
    gp0 = np.array([0.3,0.3,0.4])

    fconv = 3
    thc = thc0 * 1e6 * YTOSEC # m3/y Conveyor transport
    tto = 2e6 * YTOSEC   # m3/y Tethys Transport

    # PETM: add SO to NPDW
    tso = 0

    # TH branches
    thbra = 0.2 # 0.2 upwelled into intermdt Atl
    thbri = 0.2 # 0.2 upwelled into intermdt Ind
    gtha = 1 - thbra         # export into deep Ind
    gthi = 1 - thbra - thbri # export into deep Pac

    global ystart, vb, ab, hb, gp, tcb0, tcb0c, salv, prsv, fdapi, kkv, kiv, tcb0, salv, prsv, mxv, mhd

    # allocate globals
    ystart = np.zeros(NEQ, dtype = 'float64')     # ystart
    vb = np.zeros(NB,dtype = 'float64')           # volume boxes
    ab = np.zeros(NB,dtype = 'float64')           # areas
    hb = np.zeros(NB,dtype = 'float64')           # height boxes
    gp = np.zeros(NB,dtype = 'float64')           # H-frac remineralized in AIP
    tcb0 = np.zeros(NB, dtype = 'float64')        # temperature boxes init
    tcb0c = np.zeros(NB)                          # temperature boxes init cntrl
    salv = np.ones(NB)                            # salinity boxes
    prsv = np.zeros(NB)                           # pressure boxes
    fdapi = np.zeros(3)                                     # # init fdapi. change initial dic, alk, po4, see later initstart()



    """

    AL, IL, PL: 0,1,2
    AI, II, PI: 3,4,5
    AD, ID, PD: 6,7,8
    H: 9
    TL, TI, TD: 10,11,12

    """

    #areas
    ab[0:3] = fanoc[0:3] * AOC
    ab[3:6] = fanoc[0:3] * AOC
    ab[6:9] = fanoc[0:3]* AOC
    ab[9] = fhl * AOC
    ab[10:13] = fanoc[3]*AOC

    # height
    hb[0:3] = hlid[0]
    hb[3:6] = hlid[1]
    hb[6:9] = hlid[2]
    hb[9] = 250
    hb[10], hb[11] = hlid[0:2]
    hb[12] = 200

    # residual volume
    vres = VOC - (hlid[0]+hlid[1])*(1-fhl)*AOC-hb[9]*ab[9]-hb[12]*ab[12]
    # distribute into deep AIP
    hb[6:9] = vres * fdvol/ab[6:9]
    vb = ab * hb

    # set box indices
    # kkv: surface: low-lat（NLS） + High(1)
    # kiv: interm
    kkv = np.array([0,1,2,9,10], dtype = 'int8')
    kiv = np.array([3,4,5,11], dtype = 'int8')

    """ =====      temp, sal, pressure, Mg, Ca   =========="""
    # set interal default
    # temp (deg C)  L I D
    tcb0[0:3] = tc3[0]

    tcb0[3:6] = tc3[1]
    tcb0[6:9] = tc3[2]
    tcb0[9] = 12          # H-box
    tcb0[10],tcb0[11],tcb0[12] = tct[0],tct[1],tct[2]

    tcb0c = tcb0

    # salinity
    salv = salv * SALOCN

    # pressure
    prsv[0:3] =  0.5 * hb[0:3]                                          * 0.1  # surf
    prsv[3:6] = (0.5 * hb[3:6] + hb[0:3])                     * 0.1  # interm
    prsv[6:9] = (0.5 * hb[6:9] + hb[0:3] + hb[3:6]) * 0.1  # deep
    prsv[9] = 0.5 * hb[9] * 0.1
    k = 10
    prsv[k]   =  0.5 * hb[k]                                          * 0.1   # surf
    prsv[k+1] = (0.5 * hb[k+1] + hb[k])                     * 0.1   # interm
    prsv[k+2] = (0.5 * hb[k+2] + hb[k] + hb[k+1]) * 0.1   # deep


    # store SP to matrix
    spm = np.vstack((salv,prsv))  # Salinity Pressure matrix


    """============        mixing parameters          ==========="""
    mxv = 3.8 * YTOSEC * mv0         #Sv -> m3/year  3.8: tuning
    mhd = 1.3 * YTOSEC * mhd0        #Sv -> m3/year  1.3: tuning


    """===========        Biological pump           =========="""
    global nuwd, frei, oi
    eph0 = eph0 * ab[9]    # mol/y 1.8 H export, mol C/m2/y * A = mol/year


    nuwd = 0.0      # water column dissolution

    frei = 0.78          # fraction EPL, remineralized in I boxes
    oi = 1 - frei

    # fraction EPH, remineralized in deep A,I,P boxes
    gp[6:9] = gp0


    global fkrg0, d13ckrg, epscorg, alpcorg, rvccc, rincc, rkrgcc
    # Some of the values ofor the C-Cycle parameters below
    # are only loosely constrained, some have been be tuned
    # within reasonable ranges. A few references that provide
    # ranges are given in brackets:
    # WK92 = WalkerKasting92, Berner = GEOCARN 123
    # KA99 = KumpArthur99, Veizer = VeizerEtAl99
    # Hayes = HayesEtAl99

    # kerogen oxidation, tuded to match d13c-sw results with observations
    fkrg0 = 5e12/AOC    # mol C/m2/y 7 [WK92, Berner]


    d13ckrg = -23.2    # -23.2 kerogen [WK92/tuned]
    epscorg = -33      # -33   eps(Corg-DIC) [tuned_Berner, Hayes]

    alpcorg = epscorg/1e3 + 1  # eps(Corg-DIC) [tuned_Berner, Hayes]

    rvccc = (d13cvc/1e3 + 1) * RST
    rincc = (d13cin/1e3 + 1) * RST
    rkrgcc = (d13ckrg/1e3 + 1) * RST


    global m1, m2, m3
    m1 = np.array([0,1,2,10])
    m2 = np.array([0,1,2,8])
    m3 = np.array([0,1,2,6])


    """ =========     Initialize y-start values (default or load) ===="""
    """
    sequence of tracers:
        ocn: 1 = dic; 2 = alk; 3 = po4; 4 = tcb; 5 = dox; 6 = dicc
        Catm, C13atm


    """

    # init DIC
    dic0 = np.ones(NB) * 2.3 * 1e-3 * RHO    # 2.3 mmol/kg -> mol/m3

    # init alk
    alk0 = np.ones(NB) * 2.4 * 1e-3 * RHO    # 2.4 mmol/kg -> mol/m3

    po4pe = 0.87 * 1.0133122
    po40 = np.ones(NB) * 2.50 * 1e-3 * po4pe # mol/kg (po4 at t = 0)
    po40 = po40 * RHO                        # -> mol/m3

    d13c0 = np.hstack((2.35* np.ones(3), 0.5* np.ones(3), 0.67* np.ones(3), 1.63))
    d13c0 = np.hstack((d13c0, 2.35,0.5,0.67))

    rccb0 = (d13c0/1e3 +1)* RST
    dicc0 = rccb0 * dic0




    # copy all to ystart
    ystart[0:3*NB] = np.hstack((dic0,alk0,po40))

    # copy tcb0 to ystart
    ystart[3*NB:4*NB] = tcb0/TSCAL

    # dox
    ystart[4*NB : 5*NB] = 0.2   # mol/m3

    # dicc
    ystart[5*NB : 6*NB] = dicc0/CNTI

    # nocatm
    catm0 = 280*PPMTOMMSQ # ppmv -> mol/m2 atm co2 inventory
                           # 1 ppmv = 2.2 Pg C
    ystart[NOCT*NB] = catm0*CNTI

    d13catm0 = -6.45
    ccatm0 = catm0 * (d13catm0/1e3 + 1) * RST
    ystart[NOCT*NB +1] = ccatm0





    if LOADFLAG:
        ystart = np.loadtxt(initfile)

        if len(ystart) != NEQ:
            print('Wrong restart values')


        ystart[3*NB:4*NB] /= TSCAL
        tcb0 = ystart[3*NB:4*NB] * TSCAL

    # sr cycle
    global sr_silw0, sr_carbw0, sr_ht0, sr_sedb0, srr_carbw, srr_ht, srr_silw
    sr0 = 1.079e17
    sr_silw0 = 9e9
    sr_carbw0 = 6e9
    sr_ht0 = 12e9
    sr_sedb0 = sr_silw0 + sr_carbw0 + sr_ht0

    # 87/86
    srr_sw0 = 0.70704
    srr_carbw = 0.7096
    srr_ht = 0.7028
    # srr_silw = (sr_sedb0 * srr_sw0 - sr_carbw0 * srr_carbw - sr_ht0 * srr_ht)/sr_silw0
    srr_silw = 0.7122
    srr_ht = (sr_sedb0 * srr_sw0 - sr_carbw0 * srr_carbw - sr_silw0 * srr_silw)/sr_ht0

    ystart[-2] = sr0
    ystart[-1] = sr0 * srr_sw0



    if pcycle:
        global fcexp0, fpexp0, focb0, fopb0, fcap0,  ffep0, fpw0, deep_o20, ocbf0, po4bf0, capk

        # burial efficiency
        ocbf0 = 0.01
        po4bf0 = 0.005

        po4 = ystart[2*NB:3*NB] * MMTOM

        fcexp0 = np.sum(fepl * mxv[0:NLS] * po4[kiv]/REDPC) + eph0
        fpexp0 = np.sum(fepl * mxv[0:NLS] * po4[kiv]) + eph0 * REDPC

        focb0 = fcexp0 * ocbf0
        fopb0 = fpexp0 * po4bf0
        fcap0 = 2 * fopb0
        ffep0 = 1 * fopb0

        fkrg0 = focb0/AOC
        fpw0 = fopb0 * 4

        deep_o20 = np.sum(ystart[4*NB:5*NB][m3+6] * vb[m3+6]) / np.sum(vb[m3+6])
        capk = fcap0 / (np.sum(fepl * mxv[0:NLS] * po4[kiv]) * (1-po4bf0)/AOC ) #  rate const. for Ca sorbed P burial

    if printflag:
        if RUN_TYPE == 1:
            if LOADFLAG:
                print( "\n@ Loaded initial values are used")
            else:
                print( "\n@ Default initial values are used")

        if RUN_TYPE == 1:
            if cinpflag:
                print("\n@ The carbon injection is ON")
            else:
                print( "\n@ The carbon injection is OFF")

        elif RUN_TYPE == 2:
            print('\n@ This experiment calculates the carbon emission scenario inversely.')

        if pcycle==0:
            print('\n Organic carbon cycle is static in this experiment')
        elif pcycle == 1:
            print('\n This experiment spins up the oragnic carbon and P cycle to the steady state')
        elif pcycle == 2:
            print('\n This experiment for the perturbation with the dynamic ocean P cycle')


        print(f'\n@ The experiment name      : {exp_name}')
        print("\n@ No. ocean basins         : %d" %NOC)
        print("\n@ No. ocean boxes          : %d" %NB)
        print("\n@ No. ocean tracers        : %d" % NOCT)
        print("\n@ Atmospheric Carbon       : %d" % NCATM)
        print("\n@ Atmospheric Carbon-13    : %d" % NCCATM)
        print("\n@ No. equations            : %d" % NEQ)

        if RUN_TYPE == 1:
            print("\n@ Starting integration\n")
            print( "[tstart  tfinal]=[%.2e  %.2e]\n" % (t0, tfinal))
        elif RUN_TYPE == 2:
            print( "\n@ Starting inversion")

def model_run():

    global RUN_TYPE


    if RUN_TYPE == 1 or RUN_TYPE == 0:
        start_time = timeit.default_timer()


        hpls = np.ones(NB) * HGUESS

        t_eval = np.arange(-252e6, -251.9e6, 2e3)
        if platform.system() == 'Windows':
            ysol = solve_ivp(derivs, (t0, tfinal), ystart, args = [hpls], t_eval=t_eval,  method = ODE_SOLVER)

            if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                ysol = solve_ivp(derivs, (t0, tfinal), ystart, args = [hpls], t_eval=t_eval, method = ODE_SOLVER, first_step = 5e2)


                if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                    ysol =  solve_ivp(derivs, (t0, tfinal), ystart, args = [ hpls], t_eval=t_eval, method = ODE_SOLVER, first_step = 1e2)



        else:
            first_step = None
            ysol = solve_ivp(derivs, (t0, tfinal), ystart, args = [ hpls], t_eval=t_eval, method = ODE_SOLVER)

        elapse = timeit.default_timer() - start_time


        if svstart:
            yfinal = np.copy(ysol['y'][:,-1])
            yfinal[3*NB:4*NB] *= TSCAL
            np.savetxt(svfile, yfinal)
            print((f'System variables at t={tfinal:.2e} has been saved to ', svfile))
        if printflag:
            print('Integration finished.')
            print(f'{elapse: .2f}s used.')

        if svresults:
            if printflag:
                print('Starting to save the modeling results.')
                print('Results are saved')
            wo_results(ysol['y'], ysol['t'])




    if RUN_TYPE == 2:

        start_time = timeit.default_timer()

        target = pd.read_csv(target_file).dropna().to_numpy()

        global t_target
        t_target = target[:,0]
        tracer_target = target[:,1]

        tracer_eval = np.zeros(len(t_target))

        global f_target, y0
        f_target = interpolate.interp1d(t_target, tracer_target, bounds_error=False, fill_value=(tracer_target[0], tracer_target[-1]))

        global frccinp

        frccinp = interpolate.interp1d([t_target[0], -251.952e6, t_target[-1]],[((dccinp_inv1/1e3)+1)*RST, ((dccinp_inv2/1e3)+1)*RST, 0], fill_value=(0,0), kind = 'zero',  bounds_error=False)






        for i in range(len(t_target)-1):
            if i:
                y0 = np.loadtxt(f'inv_ystart_{-dccinp_inv1}_{-dccinp_inv2}.dat')
                y0[3*NB:4*NB] /= TSCAL
            else:
                y0 = ystart

            tems = t_target[i:i+2]
            ems = tracer_eval[i]

            n = int((tems[-1]-t_target[0])/(t_target[-1]-t_target[0])*100)
            print(n, "% finished.")



            # if (np.abs(fcinp(tems))<0.005).all():
            #     try:
            #         tracer_eval[i] = tracer_eval[i-1]
            #     except:
            #         tracer_eval[i] = -20
            #
            # else:

            hpls = np.ones(NB) * HGUESS
            # ems_d13c_new = fmin(cost_function, init_guess, args = (tracer_type, hpls, tems, ems_d13c), disp = True)

            # ems_d13c_new = root_scalar(cost_function, x0 = init_guess, x1 = init_guess+10, method = 'secant',
            #              args = (tracer_type, hpls, tems, ems_d13c))
            # ems_d13c_new = ems_d13c_new.root

            ems_new, results = toms748(cost_function, toms_low, toms_high,
                                    args = ('d13c_for_emi', hpls, tems, ems),
                                    xtol = 1e-4, rtol = 1e-8,
                                    full_output = True)


            tracer_eval[i]= ems_new


            # write out the ysol as the initial values for next loop

            if platform.system() == 'Windows':
                ysol = solve_ivp(derivs, (tems[0], tems[1]), y0, args = [hpls], t_eval = [tems[1]], method = ODE_SOLVER)

                if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                    ysol =solve_ivp(derivs, (tems[0], tems[1]), y0, args = [hpls], t_eval = [tems[1]],  method = ODE_SOLVER, first_step = 5e2)


                    if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                        ysol =  solve_ivp(derivs, (tems[0], tems[1]), y0, args = [hpls], t_eval = [tems[1]], method = ODE_SOLVER, first_step = 1e2)



            else:
                first_step = None
                ysol = solve_ivp(derivs, (tems[0], tems[1]), y0, args = [ hpls], t_eval = [tems[1]], method = ODE_SOLVER, first_step = first_step)




            yfinal = ysol.y[:,-1]
            yfinal[3*NB:4*NB] *= TSCAL
            np.savetxt(f'inv_ystart_{-dccinp_inv1}_{-dccinp_inv2}.dat', yfinal)

        elapse = timeit.default_timer() - start_time
        if printflag:
            print(f'{elapse: .2f}s used.')

        # tt = np.linspace(t_target[0], t_target[-1]-10, int((t_target[-1]-t_target[0])/100))
        #
        # RUN_TYPE = 3
        #
        # frccinp = interpolate.interp1d(t_target, (tracer_eval/1e3+1)*RST, bounds_error=False, fill_value=((tracer_eval[0]/1e3+1)*RST,0), kind = 'zero')
        #
        # emi_d13c = (frccinp(tt)/RST-1)*1e3
        #
        # emi_d13c_scenario = pd.DataFrame(np.vstack([tt, emi_d13c]).T)
        # emi_d13c_scenario.columns = ['Age', 'd13C_of_carbon_emission']
        # emi_d13c_scenario.to_csv('inverse_emission_d13c_results.csv')

        global fcinp
        fcinp = interpolate.interp1d(t_target, tracer_eval, bounds_error=False, fill_value=0, kind = 'zero')

        tt = np.concatenate((t_target, t_target[1:]-1))
        tt = np.sort(tt)

        emi_c = fcinp(tt)

        ts = tt[1:]-tt[0:-1]

        emi_interval = np.concatenate([[0],ts * emi_c[0:-1]])

        emi_c_cum = np.cumsum(emi_interval)

        emi_scenario = pd.DataFrame(np.vstack([tt, emi_c, emi_c_cum]).T)
        emi_scenario.columns = ['Age', 'Carbon_emission_Gt', 'total_carbon_emission_Gt']

        dir_name = f'./{exp_name}'
        folder = exists(dir_name)
        if not folder:
            makedirs(dir_name)

        emi_scenario.to_csv(f'./{exp_name}/inverse_emission_from_d13c.csv')

        RUN_TYPE = 3


        if platform.system() == 'Windows':
            ysol = solve_ivp(derivs, (t_target[0], t_target[-1]), ystart, args = [ hpls], method = ODE_SOLVER)

            if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                ysol =solve_ivp(derivs, (t_target[0], t_target[-1]), ystart, args = [hpls],   method = ODE_SOLVER, first_step = 5e2)


                if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                    ysol =  solve_ivp(derivs, (t_target[0], t_target[-1]), ystart, args = [hpls],  method = ODE_SOLVER, first_step = 1e2)



        else:
            first_step = None
            ysol = solve_ivp(derivs, (t_target[0], t_target[-1]), ystart, args = [hpls], t_eval = np.arange(t_target[0], t_target[-1], 2e3), method = ODE_SOLVER)

        wo_results(ysol['y'], ysol['t'])
        if printflag:
            print(f'{elapse: .2f}s used.')
            print('The calculated emission scenario has been saved to inverse_emission_from_d13c.csv')

        return ysol['y'], ysol['t']

    if RUN_TYPE == 3:
        # inversely calculate the d13c of carbon emit firstly


        target_d13c = pd.read_csv(target_file).dropna().to_numpy()
        t_target_d13c = target_d13c[:,0]
        d13c_target = target_d13c[:,1]


        f_target = interpolate.interp1d(t_target_d13c, d13c_target, bounds_error=False, fill_value=(d13c_target[0],d13c_target[-1]))


        d13c_eval = np.zeros(len(t_target_d13c))

        for i in range(len(t_target_d13c)-1):
            if i:
                y0 = np.loadtxt(f'inv_ystart_{id}.dat')
                y0[3*NB:4*NB] /= TSCAL
            else:
                y0 = ystart

            tems_d13c = t_target_d13c[i:i+2]
            ems_d13c = d13c_eval[i]





            if i:
                init_guess = d13c_eval[i-1]
            else:
                init_guess = -20

            if (np.abs(fcinp(tems_d13c))<0.0001).all():
                try:
                    d13c_eval[i] = d13c_eval[i-1]
                except:
                    d13c_eval[i] = -20

            else:


                hpls = np.ones(NB) * HGUESS
                # ems_d13c_new = fmin(cost_function, init_guess, args = ('d13c', hpls, tems_d13c, ems_d13c), disp = True)
                ems_d13c_new = root_scalar(cost_function, x0 = init_guess, x1 = init_guess+5, method = 'secant',
                                      xtol = 1e-4, rtol = 1e-8,
                                      args = ('d13c', hpls, tems_d13c, ems_d13c))


                ems_d13c_new = ems_d13c_new.root

                # if ems_d13c_new > 10:
                #     ems_d13c_new = 10
                # elif ems_d13c_new < -60:
                #     ems_d13c_new = -60

                d13c_eval[i]= ems_d13c_new






            # write out hte ysol as the initial values for next loop


            if platform.system() == 'Windows':
                ysol = solve_ivp(derivs, (tems_d13c[0], tems_d13c[1]), y0, args = [hpls], t_eval = [tems_d13c[1]], method = ODE_SOLVER)

                if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                    ysol =solve_ivp(derivs, (tems_d13c[0], tems_d13c[1]), y0, args = [ hpls], t_eval = [tems_d13c[1]], method = ODE_SOLVER, first_step = 5e2)


                    if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                        ysol =  solve_ivp(derivs, (tems_d13c[0], tems_d13c[1]), y0, args = [hpls], t_eval = [tems_d13c[1]], method = ODE_SOLVER, first_step = 1e1)



            else:
                first_step = 1e3
                ysol = solve_ivp(derivs, (tems_d13c[0], tems_d13c[1]), y0, args = [hpls], t_eval = [tems_d13c[1]], method = ODE_SOLVER)

            yfinal = ysol.y[:,-1]
            yfinal[3*NB:4*NB] *= TSCAL
            np.savetxt(f'inv_ystart_{id}.dat', yfinal)

        # elapse = timeit.default_timer() - start_time
        # print(f'{elapse: .2f}s used.')


        RUN_TYPE = 3
        frccinp = interpolate.interp1d(t_target_d13c, (d13c_eval/1e3+1)*RST, bounds_error=False, fill_value=((d13c_eval[0]/1e3+1)*RST,0), kind = 'zero')
        # emi_d13c_scenario = np.vstack([t_target_d13c, d13c_eval]).T
        # np.savetxt('inverse_emission_d13c_results.dat', emi_d13c_scenario)
        # t_end = np.minimum(t_target[-1], t_target_d13c[-1])
        # t_start = np.minimum(t_target[0], t_target_d13c[0])
        # tt = np.concatenate((t_target_d13c, t_target_d13c[1:]-1))
        # tt = np.sort(tt)
        # # np.linspace(t_target_d13c[0], t_target_d13c[-1]-10, int((t_target[-1]-t_target[0])/100))
        # emi_d13c = (frccinp(tt)/RST-1)*1e3
        #
        # emi_d13c_scenario = pd.DataFrame(np.vstack([tt, emi_d13c]).T)
        # emi_d13c_scenario.columns = ['Age', 'd13C_of_carbon_emission']
        # emi_d13c_scenario.to_csv('double_inversion_emission_d13c_pH.csv')


        t_eval = np.arange(-252e6, -251.90e6, 2e3)

        if platform.system() == 'Windows':
            ysol = solve_ivp(derivs, (t_target_d13c[0], t_target_d13c[-1]), ystart, args = [ hpls], t_eval = t_eval, method = ODE_SOLVER)

            if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                ysol =solve_ivp(derivs, (t_target_d13c[0], t_target_d13c[-1]), ystart, args = [ hpls], method = ODE_SOLVER, t_eval = t_eval, first_step = 5e2)


                if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                    ysol =  solve_ivp(derivs, (t_target_d13c[0], t_target_d13c[-1]), ystart, args = [ hpls], method = ODE_SOLVER,t_eval = t_eval,  first_step = 1e2)



        else:
            first_step = 1e3
            ysol = solve_ivp(derivs, (t_target_d13c[0], t_target_d13c[-1]), ystart, args = [ hpls], method = ODE_SOLVER,t_eval = t_eval,  first_step = first_step)



        wo_results(ysol['y'], ysol['t'])

        return (frccinp(ysol.t)/RST-1)*1e3



def cost_function(ems_new, tracer_type, hpls, tems, ems):
    if tracer_type == 'd13c_for_emi':
        ems = np.array([ems_new, ems_new])
        global fcinp
        fcinp = interpolate.interp1d([tems[0], tems[1]], [ems[0], ems[1]], bounds_error=False, fill_value=0)

        if platform.system() == 'Windows':
            ysol = solve_ivp(derivs, (tems[0], tems[1]), y0, t_eval = [tems[1]], args = [ hpls], method = ODE_SOLVER)

            if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                ysol =solve_ivp(derivs, (tems[0], tems[1]), y0, t_eval = [tems[1]], args = [ hpls], method = ODE_SOLVER, first_step = 5e2)


                if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                    ysol =  solve_ivp(derivs, (tems[0], tems[1]), y0, t_eval = [tems[1]], args = [ hpls], method = ODE_SOLVER, first_step = 1e2)



        else:
            first_step = None
            ysol = solve_ivp(derivs, (tems[0], tems[1]), y0, t_eval = [tems[1]], args = [ hpls], method = ODE_SOLVER, first_step = first_step)

        dic = ysol.y[0:NB,:]
        dicc = ysol.y[5*NB:6*NB,:] * CNTI*1e3/RHO
        temp = dicc/(dic*1e3/RHO)/RST
        d13c_mol = np.transpose((temp-1)*1e3)
        d13c_surf = np.sum(d13c_mol[:, kkv] * vb[kkv],axis = 1)/np.sum(vb[kkv])

        errors = (d13c_surf-f_target(tems[-1]))/np.abs(f_target(tems[-1]))

        return errors[0]

    if tracer_type == 'd13c':
        global frccinp
        # ems = np.array([(arg[0]/1e3 + 1) *RST, (ems_new[0]/1e3 + 1) * RST])


        rccinp = (ems_new/1e3+1)*RST

        try:
            frccinp = interpolate.interp1d([tems[0], tems[1]], [rccinp, rccinp], bounds_error=False, fill_value=0, kind = 'zero')
        except:
            frccinp = interpolate.interp1d([tems[0], tems[1]], [rccinp[0], rccinp[0]], bounds_error=False, fill_value=0, kind = 'zero')

        if platform.system() == 'Windows':
            ysol = solve_ivp(derivs, (tems[0], tems[1]), y0, t_eval = [tems[1]], args = [hpls], method = ODE_SOLVER)

            if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                ysol =solve_ivp(derivs, (tems[0], tems[1]), y0, t_eval = [tems[1]], args = [hpls], method = ODE_SOLVER, first_step = 5e2)


                if np.isnan(ysol['y']).any() or len(ysol['y']) == 0:

                    ysol =  solve_ivp(derivs, (tems[0], tems[1]), y0, t_eval = [tems[1]], args = [hpls], method = ODE_SOLVER, first_step = 1e1)



        else:
            first_step = 1e3
            ysol = solve_ivp(derivs, (tems[0], tems[1]), y0, t_eval = [tems[1]], args = [hpls], method = ODE_SOLVER)


        dic = ysol.y[0:NB,:]
        dicc = ysol.y[5*NB:6*NB,:] * CNTI*1e3/RHO
        temp = dicc/(dic*1e3/RHO)/RST
        d13c_mol = np.transpose((temp-1)*1e3)
        d13c_surf = np.sum(d13c_mol[:, kkv] * vb[kkv],axis = 1)/np.sum(vb[kkv])

        errors = (d13c_surf-f_target(tems[-1]))/np.abs(f_target(tems[-1]))


        return errors

def derivs(t, y, hpls, *flux_flag):



    yp = np.zeros(NEQ)

    # read tracer values from y
    # dydt default: all to zero
    dic = y[0:NB]
    dicp = np.zeros(NB)
    alk = y[NB:2*NB]
    alkp = np.zeros(NB)
    po4 = y[2*NB:3*NB]*MMTOM  # po4 convert to mol/m3
    po4p = np.zeros(NB)


    # warning if high-lat po4 < 0
    if po4[9]<0:
        print('\n HighLat PO4 = {po4[9]:.e} mol/m3')
        print("derivs(): HighLat PO4 is negative. Increase initial PO4?")


    tcb = y[3*NB:4*NB]*TSCAL
    tcbp = np.zeros(NB)

    dox = y[4*NB:5*NB]
    doxp = np.zeros(NB)

    if dox.any() < 0:
        print('\n Dissolved Oxygen: dox[{k:.d} = {dox[k]:.e} mol/m3')
        print("derivs(): Diss. O2 is negative (<anoxic). Reduce CBIOH?")



    dicc = y[5*NB:6*NB] * CNTI   #->mol/m3
    diccp = np.zeros(NB)
    rccb = dicc/dic

    pco2a = y[NOCT*NB] / PPMTOMMSQ/CNTI  #atmospheric co2, scaling /CNTI
    catmp = 0

    pcco2a = y[NOCT*NB+1]/PPMTOMMSQ           # not scaled
    ccatmp = 0.0



    eplv = np.zeros(NLS)
    ealv = np.ones(NLS)
    enlv = np.zeros(NLS)
    pplv = np.zeros(NLS)
    eclv = np.zeros(NLS)
    exlv = np.zeros(NLS)
    co2 =  np.zeros(NB)
    pco2 = np.zeros(NB)
    co3 =  np.zeros(NB)

    ph =  np.zeros(NB)
    kh =  np.zeros(NB)
    o2sat =  np.zeros(NB)
    kasv0 =  np.zeros(NB)
    kasv =  np.zeros(NB)
    vask0 =  np.zeros(NB)
    vask =  np.zeros(NB)
    fmmo =  np.zeros(NB)
    alpdb =  np.zeros(NB)
    alpdg =  np.zeros(NB)
    alpcb =  np.zeros(NB)
    eplvcc = np.zeros(NLS)
    ealvcc = np.zeros(NLS)
    eclvcc = np.zeros(NLS)
    exlvcc = np.zeros(NLS)
    fkrgccb = np.zeros(NB)

    # co2 system and o2 of boxes (surface 0:LA 1:LI 3:LP 9:H )
    # requires mol/kg

    co2, pco2, co3, hpls, ph, kh, o2sat = fcsys(dic/RHO, alk/RHO, hpls, tcb, salv, prsv, cac, mgc)

    # air-sea co2 exchange coeff

    kasv0[kkv] = 0.06
    vask0[kkv] = 3*365
    kasv = kasv0 * ab
    vask = vask0 * ab

    # air-sea 13co2 alphas
    alpdb[kkv], alpdg[kkv], alpcb[kkv] = falpha(tcb[kkv])

    if MOOK:
        alpu = 0.9995
    else:
        alpu = 0.9991

    """---------------------- Biological pump -------------------------"""

    # CaSiO3 weathering


    fsi = fvc0* (pco2a/pcsi) ** nsi *(fscale_erosion(t) ** 0.8)

    # CaCO3 weathering
    finc = finc0 * (pco2a/pcsi) ** ncc * (fscale_erosion(t) ** 0.86)

    # calculate the saturation state for the surface box

    kspc = fkspc(tcb[m1], salv[m1], prsv[m1], cac, mgc)
    omegacalc = co3[m1]*cac/kspc




    kcarb = 0.009  # mol C/m2/y
    mean_omegacalc = np.sum(omegacalc*vb[m1])/np.sum(vb[m1])




    eplv = fepl * mxv[0:NLS] * po4[kiv]/REDPC
    ealv = kcarb * (omegacalc-1)**2 * ab[m1] # ALK mol/year
    # (2*finc*AOC/NOC + 2*fsi*AOC/NOC)/(1-nuwd)
    # kcarb * (omegacalc-1)**2 * ab[m1]  # ALK mol/year

    pplv = eplv * REDPC       # PO4 mol/year
    enlv = eplv * REDNC       # NO3 mol/year

    # total carbon: Corg + CaCO3
    eclv = eplv + 0.5 * ealv

    # high lat export
    eph = eph0

    eah = 0                      ##  2*eph/init.rrain   ###???? 0000?
    pph = eph * REDPC
    enh = eph * REDNC
    # total carbon
    ech = eph+0.5*eah

    eplvcc[0:3] = alpcorg * rccb[0:3] * eplv[0:3]
    ealvcc[0:3] =           rccb[0:3] *ealv[0:3]
    eclvcc[0:3] = eplvcc[0:3] + 0.5 * ealvcc[0:3]

    ephcc = alpcorg*rccb[9]*eph
    eahcc =         rccb[9]*eah
    echcc = ephcc+0.5*eahcc
    eplvcc[3] = alpcorg * rccb[10] * eplv[3]
    ealvcc[3] =           rccb[10] *ealv[3]
    eclvcc[3] = eplvcc[3] + 0.5 * ealvcc[3]

    if po4[9] < PMMK:

        if po4[9]<0:
            pph = 0

        else:
            pph = pph * PMMV * po4[9] /(PMM0 + po4[9])

    # MM kinetic dissolved oxygen
    fmmo = dox/(dox+KMMOX)
    fmmo [np.where(fmmo<0)] = 0

    # ----- Long C-Cycle fluxes ---------


    fkrg = fkrg0 * (fscale_erosion(t) ** 0.25)
    if pcycle == 1:

        # deep_o2 = np.sum(dox[m3+6]*vb[m3+6])/np.sum(vb[m3+6])
        # ratio_o = deep_o2/deep_o20
        # c_p_ratio = (260 * 4000)/(ratio_o * 4000 + (1-ratio_o) * 260)

        # po4bf = po4bf0 * (0.25+0.75*ratio_o)


        # ocbf = po4bf*ocbf0/po4bf0

        focb = 0

        # fpw = fpw0 * (finc+fsi)/(finc0+fvc0)
        # fopb = 0
        # ffep = ffep0 * ratio_o
        # fcap = capk * (np.sum(pplv)/AOC - np.sum(pplv)/AOC * po4bf * (0.1+0.9*ratio_o))
        # oi = 1 - frei - ocbf * c_p_ratio / 260
        # oip = 1- frei - po4bf

        oi = 1- frei - ocbf0
        oip = 1-frei - po4bf0
        fpw = fpw0
        ffep = fpw/4
        fcap = fpw/2

        fopb = 0

        fkrg = ocbf0 * (np.sum(eplv)+eph)/AOC

    elif pcycle == 2:
        deep_o2 = np.sum(dox[m3+6]*vb[m3+6])/np.sum(vb[m3+6])
        ratio_o = deep_o2/deep_o20
        c_p_ratio = (260 * 4000)/(ratio_o * 4000 + (1-ratio_o) * 260)

        po4bf = po4bf0 * (0.25+0.75*ratio_o)
        ocbf = po4bf*ocbf0/po4bf0

        # fpw = fpw0 * (fsi)/(fvc0)
        fpw = fpw0 * (0.8 * (fsi/fvc0) + 0.2 * (finc/finc0))
        fopb = 0
        ffep = ffep0 * ratio_o *fscale_erosion(t)


        fcap = capk * (np.sum(pplv)/AOC - np.sum(pplv)/AOC * po4bf * (0.1+0.9*ratio_o))
        oi = 1 - frei - ocbf * c_p_ratio / 260
        oip = 1- frei - po4bf

        focb = 0
        fkrg = fkrg0 * (fscale_erosion(t) ** 0.25)

    else:
        oi = 1 - frei
        oip = 1 - frei
        focb = fkrg

    # long c-cycle fluxes 13c


    fsicc = rincc*fsi
    fincc = rincc*finc

    if pcycle:

        focbccb = alpcorg * focb/AOC * rccb
    else:

        focbccb = alpcorg * fkrg * rccb




    exlv = 1 * eplv
    exh = eph
    exlvcc = eplvcc * 1.0
    exhcc = ephcc

    sdn = 0







    """

    Right-hand side of DEQs
    Units ocean tracer: mol/m3/y

    """
    # =========================== DIC ===================================

    # TH and mixing

    dicp = thmfun(dic,fconv,thc,tso,tto,mxv,mhd,vb,gtha,thbra,gthi,thbri)



    # air-sea

    dicp[kkv] = dicp[kkv] + kasv[kkv] * (pco2a-pco2[kkv])/vb[kkv]


    # bio pump Corg

    dicp[m1] = dicp[m1] - eclv/vb[m1]
    dicp[m2+3] = dicp[m2+3] + frei*exlv/vb[m2+3]
    dicp[m3+6] = dicp[m3+6] + oi*exlv/vb[m3+6]
    dicp[m3+6] = dicp[m3+6] + 0.5*nuwd*ealv/vb[m3+6]


    dicp[9] = dicp[9] - ech/vb[9]

    dicp[6:9] = dicp[6:9] + (frei + oi) * exh/vb[6:9]*gp[6:9] + 0.5*nuwd*eah/vb[6:9]*gp[6:9]

    # Long-term c-cycle

    dicp[m1] += 2 * finc*AOC/vb[m1]/NOC
    dicp[m1] += 2 * fsi *AOC/vb[m1]/NOC
    dicp[m1] -= 1 * focb*AOC/vb[m1]/NOC





    # =========================== ALK ===================================
    # TH and mixing
    alkp = thmfun(alk,fconv,thc,tso,tto,mxv,mhd,vb,gtha,thbra,gthi,thbri)
    # bio pump

    alkp[m1] += -ealv/vb[m1] + enlv/vb[m1]
    alkp[m2+3] += frei*(sdn*ealv/vb[m2+3] - enlv/vb[m2+3])
    alkp[m3+6] += oip * (sdn*ealv/vb[m3+6]-enlv/vb[m3+6]) + nuwd*ealv/vb[m3+6]
    alkp[9] += -eah/vb[9] + enh/vb[9]

    alkp[6:9] += (sdn*eah/vb[6:9] - (frei + oip) * enh/vb[6:9])*gp[6:9] + nuwd*eah/vb[6:9] * gp[6:9]

    if pcycle:
        alkp[m3+6] += ffep * REDNC/REDPC / vb[m3+6]/NOC  # Alkalinity source from iron-sorbed burial
        alkp[m3+6] += fcap * REDNC/REDPC  / vb[m3+6]/NOC  # Alkalinity source from fluorapatite burial

        alkp[m1] -= fpw * REDNC/REDPC / vb[m1] / NOC     # Alkalinity sink from phosphate weathering


    alkp[m1] += 2*finc*AOC/vb[m1]/NOC
    alkp[m1] += 2*fsi*AOC/vb[m1]/NOC



    # =========================== PO4 ===================================
    # TH and mixing
    po4p = thmfun(po4,fconv,thc,tso,tto,mxv,mhd,vb,gtha,thbra,gthi,thbri)

    # bio pump Porg
    po4p[m1] -= pplv/vb[m1]
    po4p[m2+3] += frei*pplv/vb[m2 + 3]
    po4p[m3+6] += oip*pplv/vb[m3 + 6]

    po4p[9] -= pph/vb[9]

    po4p[6:9] += (frei+oip) * pph/vb[6:9]*gp[6:9]

    if pcycle:
        po4p[m3+6] = po4p[m3+6] - ffep /vb[m3+6]/NOC - fcap /vb[m3+6]/NOC
        po4p[m1] += fpw/vb[m1]/NOC

    total_p = np.sum(po4 * vb)
    total_p0 = np.sum(ystart[2*NB:3*NB]*MMTOM*vb)



    # =========================== Temp ===================================
    # Ocean temp change, using simple climate sensitivity (Archer,2005)
    # TAU: relax times(y): surf, interm, deep
    if ((pco2a >= 150)):
        tmp = sclim*np.log(pco2a/pcsi)/np.log(2)
        tcbp[m1] = (tcb0[m1] + tmp-tcb[m1])/TAUS
        tcbp[m2+3] = (tcb0[m2+3]+tmp-tcb[m2+3])/TAUI
        tcbp[m3+6] = (tcb0[m3+6]+tmp-tcb[m3+6])/TAUD
        tcbp[9] = (tcb0[9]+tmp-tcb[9])/TAUS

    # =========================== Oxygen ===================================
    doxp = thmfun(dox,fconv,thc,tso,tto,mxv,mhd,vb,gtha,thbra,gthi,thbri)
    # air-sea, o2sat = o2 at saturation(atm)
    doxp[kkv] += vask[kkv] * (o2sat[kkv]-dox[kkv])/vb[kkv]

    # bio pump o2
    doxp[m1] += eplv*REDO2C/vb[m1]
    doxp[m2+3] -= frei*fmmo[m2+3]*eplv*REDO2C/vb[m2+3]
    doxp[m3+6] -= oi*fmmo[m3+6]*eplv*REDO2C/vb[m3+6]
    doxp[9] += (frei+oi) * eph*REDO2C/vb[9]

    doxp[6:9] -= gp[6:9]*fmmo[6:9]* (frei+oi) * eph*REDO2C/vb[6:9]

    # # =========================== DICC:13C ===================================
    # TH and mixing
    diccp = thmfun(dicc,fconv,thc,tso,tto,mxv,mhd,vb,gtha,thbra,gthi,thbri)
    # air-sea

    tmp = alpdg[kkv]*pcco2a-alpdb[kkv]*rccb[kkv]*pco2[kkv]
    diccp[kkv] += kasv[kkv]*alpu*tmp/vb[kkv]

    # bio pump Corg
    diccp[m1] += -eclvcc/vb[m1]
    diccp[m2+3] += frei*exlvcc/vb[m2+3]
    diccp[m3+6] += oi*exlvcc/vb[m3+6] + 0.5*nuwd*ealvcc/vb[m3+6]
    diccp[9] -= echcc/vb[9]

    diccp[6:9] += (frei+oi) * exhcc/vb[6:9] * gp[6:9] + 0.5*nuwd*eahcc/vb[6:9] * gp[6:9]


    # riverine and sediment fluxes
    diccp[m1] += 2*fincc * AOC/vb[m1]/NOC
    diccp[m1] += 2*fsicc * AOC/vb[m1]/NOC
    diccp[m1] -= 1*focbccb[m1]*AOC/vb[m1]/NOC

        # ===========================   C atm   ===================================

    catmp -= np.sum(kasv[kkv]*(pco2a-pco2[kkv])/AOC)


    fvc = fvc0
    catmp += fvc - finc - 2*fsi + fkrg

    # total_c_p = np.sum(dicp * vb) + catmp*AOC
    # print(total_c_p)

    # total_c = np.sum(dic * vb) + pco2a * 2.2e15 / 12

    # total_c_p = (fvc - finc - 2*fsi + fkrg) * AOC

    # ===========================  13C atm  ===================================

    tmp = alpdg[kkv]*pcco2a-alpdb[kkv]*rccb[kkv]*pco2[kkv]
    ccatmp-= np.sum(kasv[kkv]*alpu*tmp/AOC)

    fvccc = rvccc*fvc
    fkrgcc = rkrgcc*fkrg
    ccatmp +=fvccc - fincc - 2*fsicc +fkrgcc



    # cabon input scenarios
    if cinpflag or RUN_TYPE == 0 or RUN_TYPE == 3:

        cinp = fcinp(t)


        catmp += cinp*1e15/12/AOC
        # dicp[m1] += cinp*1e15/12/vb[m1]/NOC
        #
        # diccp[m1] += cinp*1e15/12*frccinp(t)/vb[m1]/NOC


        ccatmp+=frccinp(t)*cinp*1e15/12/AOC


        # except:
        #     ccatmp+=rccinp*fcinp(t)*1e15/12/AOC


    # ===========================  Sr and Srr  ===================================
    sr = y[-2]
    srr = y[-1]/y[-2]
    sr_silw = sr_silw0 * fsi/fvc0
    sr_carbw = sr_carbw0 * finc/finc0
    sr_sedb = 0.5 * np.sum((ealv))/((fvc0+finc0)*AOC) * sr_sedb0

    srp = (sr_silw + sr_carbw + sr_ht0 - sr_sedb)

    srrp = (srr_silw * sr_silw + srr_carbw * sr_carbw + sr_ht0 * srr_ht- sr_sedb * srr)



    # ========================= all to yp   ===================================
    yp[0    : NB]   = dicp
    yp[NB   : 2*NB] = alkp
    yp[2*NB : 3*NB] = po4p/MMTOM      #convert back to mmol/m3

    yp[3*NB : 4*NB] = tcbp/TSCAL  #Temp: scale to order 1


    yp[4*NB : 5*NB] = doxp


    yp[5*NB : 6*NB] = diccp/CNTI


    yp[NOCT*NB] = catmp*CNTI


    yp[NOCT*NB+1] = ccatmp

    yp[-2] = srp
    yp[-1] = srrp

    # yp = np.zeros(NEQ)
    # yp[2*NB: 3*NB] = yp[2*NB : 3*NB] = po4p/MMTOM      #convert back to mmol/m3

    if flux_flag:
        return np.sum(kasv[kkv]*(pco2a-pco2[kkv])),  fsi/fvc0, finc/finc0,fpw/fpw0, ffep/ffep0, fcap/fcap0,  np.sum((1-oip-frei)*pplv)+(1-oip-frei)*pph, np.sum((1-oi-frei)*exlv)+(1-oi-frei)*eph
    else:
        return yp

def wo_results(y, t):
    """
        FILE.dat  UNIT     VARIABLE
        -----------------------------------------------------------------
        time       (y)           time
        tcb       (deg C)   OCN temperature
        dic       (mmol/kg) OCN total dissolved inorganic carbon
        alk       (mmol/kg) OCN total alkalinity
        po4       (umol/kg) OCN phosphate
        dox       (mol/m3)  OCN dissolved oxygen
        dicc      (mmol/kg) OCN DIC-13
        d13c      (per mil) OCN delta13C(DIC)
        d13ca     (per mil) ATM delta13C(atmosphere)
        pco2a     (ppmv)    ATM atmospheric pCO2
        co3       (umol/kg) OCN carbonate ion concentration
        ph        (-)       OCN pH (total scale)
        pco2ocn   (uatm)    OCN ocean pCO2
        omegaclc  (-)       OCN calcite saturation state
        omegaarg  (-)       OCN aragonite saturation state

        surface d13c        OCN surface delta13c(dic)
        Total C   (mol)     Total carbon in the ocean
        Total ALK (mol)     Total alkalinity in the ocean

        ------------------------------------------------------------------
        """

    ysol = np.array(y)

    dir_name = f'./{exp_name}/{id}'

    folder = exists(dir_name)
    if not folder:
        makedirs(dir_name)

    # time.dat
    np.savetxt(join(dir_name,"time.dat"), t)

    # dic, alk, po4.dat

    dic = ysol[0:NB,:]

    alk = ysol[NB:2*NB,:]
    po4 = ysol[2*NB:3*NB,:]  # po4 convert to mol/m3
    temp = np.transpose(po4*1e3/RHO)
    average_po4 = np.sum(temp * vb, axis = 1)/np.sum(vb)


    total_c = np.sum(np.transpose(dic) * vb, axis = 1)
    total_alk = np.sum(np.transpose(alk) * vb, axis = 1)

    surface_dic = np.sum(np.transpose(dic*1e3/RHO)[:, kkv] * vb[kkv],axis = 1)/np.sum(vb[kkv])
    surface_alk = np.sum(np.transpose(alk*1e3/RHO)[:, kkv] * vb[kkv],axis = 1)/np.sum(vb[kkv])

    if FTYS:
        col_name = ['LA', 'LI', 'LP', 'IA', 'II', 'IP', 'DA', 'DI', 'DP', 'H',  'LT', 'IT', 'DT' ]
    else:
        col_name = ['LA', 'LI', 'LP', 'IA', 'II', 'IP', 'DA', 'DI', 'DP', 'H']

    pd_dic = pd.DataFrame(np.transpose(dic*1e3/RHO), index = t)
    pd_dic.columns = col_name
    pd_dic.index.name = 'Age'
    pd_dic.to_csv(join(dir_name, "dic.csv"))

    pd_alk = pd.DataFrame(np.transpose(alk*1e3/RHO), index = t)
    pd_alk.columns = col_name
    pd_alk.index.name = 'Age'
    pd_alk.to_csv(join(dir_name, "alk.csv"))

    pd_po4 = pd.DataFrame(np.transpose(po4*1e3/RHO), index = t)
    pd_po4.columns = col_name
    pd_po4.index.name = 'Age'
    pd_po4.to_csv(join(dir_name, "po4.csv"))


    pd_ocean_inventory = pd.DataFrame({
        'total carbon': total_c,
        'total alkalinity': total_alk
        }, index = t)
    pd_ocean_inventory.index.name = 'Age'
    pd_ocean_inventory.to_csv(join(dir_name, "Carbon_inventory.csv"), float_format = '%.4e')





    # np.savetxt(join(dir_name, "dic.dat"), np.transpose(dic*1e3/RHO), fmt='%18.15f')
    # np.savetxt(join(dir_name, "alk.dat"), np.transpose(alk*1e3/RHO), fmt='%18.15f')
    # np.savetxt(join(dir_name, "po4.dat"), np.transpose(po4*1e3/RHO), fmt='%18.15f')
    # np.savetxt(join(dir_name, "tc_ocn.dat"), total_c, fmt='%18.15f')
    # np.savetxt(join(dir_name, "talk_ocn.dat"), total_alk, fmt='%18.15f')
    # np.savetxt(join(dir_name, "surface_dic.dat"), surface_dic, fmt = '%18.15f')
    # np.savetxt(join(dir_name, "surface_alk.dat"), surface_alk, fmt = '%18.15f')


    # tcb.dat
    if NOCT >= 4:
        tcb = ysol[3*NB:4*NB,:]*TSCAL

        pd_tcb = pd.DataFrame(np.transpose(tcb), index = t)
        pd_tcb.columns = col_name
        pd_tcb.index.name = 'Age'
        pd_tcb.to_csv(join(dir_name, "tcb.csv"))

        # np.savetxt(join(dir_name, "tcb.dat"), np.transpose(tcb), fmt = "%18.15f")

    # dox.dat
    if NOCT >= 5:
        dox = ysol[4*NB:5*NB,:]

        pd_dox = pd.DataFrame(np.transpose(dox), index = t)
        pd_dox.columns = col_name
        pd_dox.index.name = 'Age'
        pd_dox.to_csv(join(dir_name, "dox.csv"))
        # np.savetxt(join(dir_name , "dox.dat"), np.transpose(dox), fmt = "%18.15f")

        dox_surf = np.sum(np.transpose(dox)[:, kkv] * vb[kkv],axis = 1)/np.sum(vb[kkv])
        dox_interm = np.sum(np.transpose(dox)[:, kiv] * vb[kiv],axis = 1)/np.sum(vb[kiv])
        dox_deep = np.sum(np.transpose(dox)[:, [6,7,8,12]] * vb[[6,7,8,12]],axis = 1)/np.sum(vb[[6,7,8,12]])





    if NOCT >= 6:
        dicc = ysol[5*NB:6*NB,:] * CNTI*1e3/RHO   #->mmol/kg
        pd_dicc = pd.DataFrame(np.transpose(dicc), index = t)
        pd_dicc.columns = col_name
        pd_dicc.index.name = 'Age'
        pd_dicc.to_csv(join(dir_name, "dicc.csv"))
        # np.savetxt(join(dir_name , "dicc.dat"), np.transpose(dicc), fmt = "%18.15f")

        temp = dicc/(dic*1e3/RHO)/RST
        pd_d13c = pd.DataFrame(np.transpose((temp-1)*1e3), index = t)
        pd_d13c.columns = col_name
        pd_d13c.index.time = 'Age'
        pd_d13c.to_csv(join(dir_name, "d13c.csv"))

        # np.savetxt(join(dir_name , "d13c.dat"), np.transpose((temp-1)*1e3), fmt='%18.15f')

        d13c_mol = np.transpose((temp-1)*1e3)
        d13c_surf = np.sum(d13c_mol[:, kkv] * vb[kkv],axis = 1)/np.sum(vb[kkv])



        # np.savetxt(join(dir_name , "d13c_surf.dat"), d13c_surf, fmt='%18.15f')




    pco2a = ysol[NOCT*NB,:] / PPMTOMMSQ/CNTI  #atmospheric co2, scaling /CNTI
        # np.savetxt(join(dir_name , "pco2a.dat"), pco2a, fmt='%18.15f')


    temp = ysol[NOCT*NB+1]/(pco2a*PPMTOMMSQ)/RST           # not scaled

    pd_atm = pd.DataFrame(
        {'pCO2': pco2a,
         'pCO2_d13c': (temp-1)*1e3
            },
        index = t
        )
    pd_atm.index.name = 'Age'
    pd_atm.to_csv(join(dir_name, "pCO2_d13c.csv"))

        # np.savetxt(join(dir_name , "d13ca.dat"), (temp-1)*1e3, fmt='%18.15f')



    hgssv = np.ones(NB) * HGUESS
    salv0 = salv.reshape(NB,1) * np.ones((1, len(t)))
    prsv0 = prsv.reshape(NB,1) * np.ones((1, len(t)))
    hgssv = np.ones(NB*len(t)) * HGUESS

    co2, pco2, co3, hpls, ph, kh, o2sat = fcsys(dic.flatten()/RHO, alk.flatten()/RHO, hgssv, tcb.flatten(), salv0.flatten(), prsv0.flatten(), cac, mgc)

    kspc = fkspc(tcb.flatten(), salv0.flatten(), prsv0.flatten(), cac, mgc)
    kspa = fkspa(tcb.flatten(), salv0.flatten(), prsv0.flatten(), cac, mgc)

    co3 = np.array(co3*1e6).reshape(NB, len(t))
    ph = np.array(ph).reshape(NB, len(t))
    pco2 = np.array(pco2).reshape(NB, len(t))
    kspc = np.array(kspc).reshape(NB, len(t))
    kspa = np.array(kspa).reshape(NB, len(t))

    surface_ph = np.sum(np.transpose(ph)[:, kkv] * vb[kkv],axis = 1)/np.sum(vb[kkv])

    pd_surface = pd.DataFrame(
        {'surface_dic': surface_dic,
         'surface_alk': surface_alk,
         'surface_d13c': d13c_surf,
         'surface_pH': np.transpose(surface_ph),
         'surface_dox': dox_surf,
         'intermediate_dox': dox_interm,
         'deep_dox': dox_deep,
         'average_po4': average_po4
            },
        index = t
        )
    pd_surface.index.name = 'Age'
    pd_surface.to_csv(join(dir_name, "Surface_dic_alk_d13c_ph_dox.csv"), float_format = '%.4f')

    pd_co3 = pd.DataFrame(np.transpose(co3), index = t)
    pd_co3.columns = col_name
    pd_co3.index.name = 'Age'
    pd_co3.to_csv(join(dir_name , "co3.csv"))

    pd_ph = pd.DataFrame(np.transpose(ph), index = t)
    pd_ph.columns = col_name
    pd_ph.index.name = 'Age'
    pd_ph.to_csv(join(dir_name , "ph.csv"))

    pd_pco2ocn = pd.DataFrame(np.transpose(pco2), index = t)
    pd_pco2ocn.columns = col_name
    pd_pco2ocn.index.name = 'Age'
    pd_pco2ocn.to_csv(join(dir_name , "pco2ocn.csv"))

    pd_omegaclc = pd.DataFrame(np.transpose(co3/1e6*cac/kspc), index = t)
    pd_omegaclc.columns = col_name
    pd_omegaclc.index.name = 'Age'
    pd_omegaclc.to_csv(join(dir_name , "omegaclc.csv"))

    pd_omegaarg = pd.DataFrame(np.transpose(co3/1e6*cac/kspa), index = t)
    pd_omegaarg.columns = col_name
    pd_omegaarg.index.name = 'Age'
    pd_omegaarg.to_csv(join(dir_name , "omegaarg.csv"))


    # forcing for weathering proxies
    fsilw = (pco2a/pcsi) ** nsi
    fcarbw = (pco2a/pcsi) ** ncc
    fcarbb = 0.5 * np.sum((0.009 * (co3[m1,:]/1e6*cac/kspc[m1,:] - 1)**2  * ab[m1, np.newaxis]), axis = 0)/((fvc0+finc0)*AOC)

    pd_sr_forcing = pd.DataFrame(
        {'silw_scale': fsilw,
         'carbw_scale': fcarbw,
         'burial_scale': fcarbb,

            },
        index = t
        )
    pd_sr_forcing.index.name = 'Age'
    pd_sr_forcing.to_csv(join(dir_name, "Forcing_Sr_cycle.csv"), float_format = '%.4f')


    sr = ysol[-2,:]
    srr = ysol[-1,:]


    pd_sr = pd.DataFrame(
        {'87Sr/86Sr': srr/sr,
            },
        index = t
        )
    pd_sr.index.name = 'Age'
    pd_sr.to_csv(join(dir_name, "Sr.csv"), float_format = '%.7f')
    if printflag:
        print('The experiment succeeds. Congratulations!')
        print(['The final average surface pH is: ', round(surface_ph[-1],2)])
        print(['The final average surface d13c is: ', round(d13c_surf[-1],2)])
        print(['The final pCO2 is: ', round(pco2a[-1],2)], 'uatm')

    # flux cal

    air_sea_exchange = np.zeros(len(t))
    carb_diss_l = np.zeros(len(t))
    carb_diss_i = np.zeros(len(t))
    carb_diss_d = np.zeros(len(t))
    r_fsi =  np.zeros(len(t))
    r_finc = np.zeros(len(t))
    r_fpw = np.zeros(len(t))
    r_ffep = np.zeros(len(t))
    r_fcap = np.zeros(len(t))
    r_ocp = np.zeros(len(t))
    r_ocb = np.zeros(len(t))


    for i in range(len(t)):
        y_temp = ysol[:,i]
        air_sea_exchange[i],  r_fsi[i], r_finc[i], r_fpw[i], r_ffep[i], r_fcap[i], r_ocp[i], r_ocb[i]  = derivs(t[i], y_temp, np.ones(NB) * HGUESS, 1)

    pd_flux = pd.DataFrame(
        {'Air_sea_exchange': air_sea_exchange,
         'r_fsilw': r_fsi,
         'r_fcarbw': r_finc,
         'r_fpw': r_fpw,
         'r_ffep': r_ffep,
         'r_fcap': r_fcap,
         'r_ocp': r_ocp,
         'r_ocb': r_ocb
            },
        index = t
        )

    pd_flux.to_csv(join(dir_name , "flux.csv"))


    return





@jit(nopython=True)
def fcsys(dic,alk,hgss,tc,sal,prs,calc,mgcl):


    tk = tc + TKLV
    bor =(BORT * (sal/35))*1e-6  # 416 DOE94, 432.5 Lee10

    hx = hgss

    khx = fkh(tk,sal)

    k1,k2  = fk1k2(tk,sal)
    kb  = fkb(tk,sal)
    kw  = fkw(tk,sal)
    o2x = fo2(tk,sal)




  #---------------------------------------------------------------------
    # pressure correction
    pcorrk = fpcorrk(tk, prs)
    k1 *= pcorrk[0,:]
    k2 *= pcorrk[1,:]
    kb *= pcorrk[2,:]
    kw *= pcorrk[3,:]


    # for i in range(NB):
    #     if prs[i] > 0:
    #         pcorrk = fpcorrk(tk[i],prs[i])
    #         k1[i] *= pcorrk[0]
    #         k2[i] *= pcorrk[1]
    #         kb[i] *= pcorrk[2]
    #         kw[i] *= pcorrk[3]
    #kspc = kspc*pcorrk[4]
    #kspa = kspa**pcorrk[5]


    # Ca, Mg corrections
    if calc!=CAM or mgcl != MGM:


        k1 = k1 + fdelk1(k1, calc, mgcl)
        k2 = k2 + fdelk2(k2, calc, mgcl)



    PRINTCSU = 0
    # iterative solution for H+, Follows et al. 2006

    for i in range(HIMAX):
        hgss = hx
        boh4g = bor*kb/(hgss+kb)
        fg = -boh4g-(kw/hgss) + hgss
        calkg = alk + fg
        gam = dic/calkg
        tmp = (1-gam)*(1-gam)*k1*k1 - 4 *k1 *k2 *(1-2*gam)

        hx = 0.5 *((gam-1)*k1+np.sqrt(tmp))
        temp = np.fabs(hx-hgss) - hgss*1e-4
        if np.all(temp<0) :
            break



              # if PRINTCSU:
              #     print(f'\n {i:d} i')


       # if i == HIMAX:
       #     print(f'\n [H+] = {hx:.e}]')
       #     print(f'\n [H+] iteration did not converge after {HIMAX:.d} steps')
       #     if hx<0:
       #         print('\n [H+] < 0. Your ALK/DIC ratio may be too high')
       #         print('csys(): check co2 system input')


    co2 = dic/(1+k1/hx + k1*k2/hx/hx)
    pco2 = co2 * 1e6/khx
    co3 = dic/(1+hx/k2 + hx*hx/k1/k2)
    h = hx
    ph = - np.log10(hx)
    kh = khx
    o2sat = o2x



    if PRINTCSU:
         print("%20.15e  kh\n",khx)
         print("%20.15e  k1\n",k1)
         print("%20.15e  k2\n",k2)
         print("%20.15e  kb\n",kb)
         print("%20.15e  kw\n",kw)
         print("%20.15e  O2  \n",o2x)


    return co2, pco2, co3, h, ph, kh, o2sat


# =========================== fcsys() end ===================================


@jit(nopython=True)
def fkspc(tc,sal, prs, calc, mgcl):
    #------Kspc (calcite)  --------
    # Apparent solubility product of calcite
    # Kspc = [Ca2+][coe2-]T   T:total, free ions + ion pairs
    # Mucci 1983 mol/kg(solution)
    tk = tc + TKLV
    tmp1 = -171.9065-0.077993*tk+2839.319/tk+71.595*np.log10(tk)
    tmp2 = (-0.77712 + 0.0028426 * tk + 178.34/tk) * np.sqrt(sal)
    tmp3 = -0.07711*sal+0.0041249* (sal **1.5)
    log10kspc = tmp1 + tmp2 +tmp3

    kspc = 10**log10kspc


    # pressure correction
    kspc = kspc*fpcorrk(tk,prs)[4,:]
    # if prs>0:
    #     kspc = kspc*fpcorrk(tk,prs)[4]

    # Ca, Mg corrections
    if ((np.fabs((calc-CAM)/CAM))>0) or ((np.fabs((mgcl-MGM)/MGM))>0):
        kspc *= 1-ALPKC * (MGM/CAM - mgcl/calc)

    return kspc
# =========================== fkspc() end ===================================

@jit(nopython=True)
def fkspa(tc,sal, prs, calc, mgcl):
    #------Kspc (calcite), Kspa(arogonite)  --------
    # Apparent solubility product of calcite
    # Kspc = [Ca2+][coe2-]T   T:total, free ions + ion pairs
    # Mucci 1983 mol/kg(solution)
    tk = tc + TKLV

    tmp1 = -171.945 - 0.077993*tk + 2903.293/tk + 71.595*np.log10(tk)
    tmp2 = +(-0.068393+0.0017276*tk +88.135/tk)*np.sqrt(sal)
    tmp3 = -0.10018*sal+0.0059415*(sal**1.5)
    log10kspa = tmp1 + tmp2 + tmp3

    kspa = 10 ** (log10kspa)

    # pressure correction

    kspa = kspa*fpcorrk(tk,prs)[5,:]
    # Ca, Mg corrections


    return kspa

# =========================== fkspa() end ===================================

    #-------kh (K Henry)-----
    #
    # CO2(g) <-> CO2 (aq.)
    # kh  =  [co2]/pco2
    # Weiss 1974, mol/kg/atm

@jit(nopython=True)
def fkh(tk,sal):
    tmp = 9345.17/tk - 60.2409 + 23.3585*np.log(tk/100)
    nkhwe74 = tmp + sal * (0.023517 - 2.3656e-4 * tk + 0.0047036e-4 * tk * tk)

    khx = np.exp(nkhwe74)
    return khx

# ===========================  khx() end  ===================================


    #--------k1,k2-------
    # first, second acidity constant
    # pH-scale: total
    # Mehrbach et al. 1973, efit by Lueker et al.(2000)
@jit(nopython=True)
def fk1k2(tk,sal):
    pk1mehr = 3633.86/tk - 61.2172 + 9.6777 * np.log(tk)-0.011555* sal + 1.152e-4*sal*sal
    k1 = 10**(-pk1mehr)

    pk2mehr = 471.78/tk +25.9290 - 3.16967 * np.log(tk) - 0.01781 * sal + 1.122e-4*sal*sal
    k2 = 10**(-pk2mehr)

    return k1,k2

# =========================== fk1k2() end ===================================

    #-------kb-----------
    #Kbor = [H+][B(OH)4-]/[B(OH)3] = kp7/km7
    #(Dickson, 1990 in Dickson and Goyet, 1994, Chapter 5)
    #pH-scale: total mol/kg -soln
@jit(nopython=True)
def fkb(tk,sal):
    tmp1 = (-8966.90-2890.53*np.sqrt(sal)-77.942*sal+1.728*sal**1.5 - 0.0996 *sal*sal)
    tmp2 = 148.0248 + 137.1942 * np.sqrt(sal) + 1.62142*sal
    tmp3 = (-24.4344-25.085*np.sqrt(sal)-0.2474*sal)*np.log(tk)
    lnkb = tmp1/tk +tmp2 +tmp3+0.053105*np.sqrt(sal)*tk

    kb = np.exp(lnkb)
    return kb

# =========================== fkb() end ===================================


    #--------kwater----------
    #
    # Millero(1995) (in Dickson and Goyet (1994, Chapter 5))
    # $K_w$ in mol/kg-soln
    # pH -scale: total scal
@jit(nopython=True)
def fkw(tk,sal):
    tmp1 = -13847.26/tk + 148.96502 - 23.6521 * np.log(tk)
    tmp2 = (118.67/tk -5.977 +1.0495 * np.log(tk)) * np.sqrt(sal) - 0.01615 * sal
    lnkw = tmp1 + tmp2
    kw = np.exp(lnkw)

    return kw
# =========================== fkw() end ===================================


    #------solubility of O2 （ml/l = l/m3)--
    #
    # Weiss 1970 DSR,17, p.721
@jit(nopython=True)
def fo2(tk,sal):

    # ml/kg
    #A = np.array([-177.7888,255.5907,146.4813,-22.2040])
    #B = np.array([-0.037362,0.016504,-0.0020564])


    A = np.array([-173.4292,249.6339,143.3483,-21.8492])
    B = np.array([-0.033096, 0.014259, -0.0017000])
    lno2 = A[0] + A[1] * 100/tk + A[2]*np.log(tk/100) + A[3]* (tk/100)+ sal * (B[0] + B[1]*tk/100 +B[2] * ((tk/100)**2))

    o2x = np.exp(lno2)/22.4 # -> mol/m3
    return o2x

# =========================== fo2() end ===================================

@jit(nopython=True)
def fpcorrk(tk,prs):

    n = len(tk)
        ### pressure effect on K's (Millero, 1995)###

    R = 83.131            # J mol-1 deg-1 (Perfect Gas)
                          # conversion cm3 -> m3 1e-6
                          #            bar -> Pa = N/m-2 1e5
                          #                => 1/10
    # index: k1 1, k2 2, kb 3, kw 4,  kspc 5, kspa 6


    #----- note: there is an error in Table 9 of Millero, 1995
    #----- The coefficents -b0 and b1 have to be multiplied by 1e-3
    #                k1,   k2,     kb,     kw,      kspc,    kspa

    a0 = -np.array([25.5 , 15.82,  29.48,  25.60,   48.76,   46])
    a1 = np.array([0.1271, -0.0219,0.1622, 0.2324, 0.5304,  0.5304])

    a2 = np.array([0.0,    0.0,    -2.608, -3.6246, 0     ,  0])*1e-3

    b0 = -np.array([3.08, -1.13,    2.84,  5.13,    11.76,   11.76])*1e-3

    b1 = np.array([0.0877, -0.1475, 0.0,   0.0794,  0.3692,  0.3692])*1e-3
    #b2 = np.zeros(len(a0))
    tc = tk - TKLV

    deltav = a0.reshape(6,1) * np.ones((1,n)) + a1.reshape(6,1) * tc.reshape((1,n)) + \
             a2.reshape(6,1) * tc.reshape((1,n)) * tc.reshape((1,n))
    deltak = b0.reshape(6,1) * np.ones((1,n)) + b1.reshape(6,1) * tc.reshape((1,n)) #+ b2 * tc * tc
    lnkpok0 = -(deltav / (R*tk)) * prs + (0.5 * deltak/(R*tk)) * prs * prs

    pcorrk = np.exp(lnkpok0)

    return pcorrk


# =========================== fpcorrkksp() end ===================================
@jit(nopython=True)
def fdelk1(k1, calc, mgcl):
    sk1ca =  33.73e-3
    sk1mg = 155.00e-3

    #/* add Ca,Mg correction K* (Ben-Yaakov & Goldhaber, 1973) */
    delk1ca = sk1ca*k1*(calc/CAM-1.)
    delk1mg = sk1mg*k1*(mgcl/MGM-1.)

    delk1   = delk1ca+delk1mg

    return delk1

# =========================== fdelk1() end ===================================
@jit(nopython=True)
def fdelk2(k2, calc, mgcl):
    # /* sensitivity parameters for Ca,Mg effect on K* */
    sk2ca =  38.85e-3
    sk2mg = 442.00e-3

     #/* add Ca,Mg correction K* (Ben-Yaakov & Goldhaber, 1973) */
    delk2ca = sk2ca*k2*(calc/CAM-1.)
    delk2mg = sk2mg*k2*(mgcl/MGM-1.)

    delk2   = delk2ca+delk2mg;

    return delk2

# =========================== fdelk2() end ===================================

@jit(nopython=True)
def falpha(tcb):
    # 13C alphas for co2 gas exchange
    # Mook 1986 or Zhang et al. 1995
    tkb = tcb+TKLV
    if MOOK:
        epsdb = -9866/tkb +24.12
        epsdg = -373/tkb + 0.19
        epscb = -867/tkb + 2.52
    else:
        epsbg = -0.141 * tcb + 10.78
        epsdg = 0.0049 * tcb - 1.31
        epscg = -0.052 * tcb + 7.22
        epsdb = (epsdg-epsbg)/(1+epsbg/1e3)
        epscb = (epscg-epsbg)/(1+epsbg/1e3)
    alpdb = epsdb/1e3 + 1
    alpdg = epsdg/1e3 + 1
    alpcb = epscb/1e3 + 1

    return alpdb,alpdg,alpcb

# =========================== falpha() end ===================================

@jit(nopython=True)
def thmfun(y, fconvl, thcl, tsol, ttol, mvl, mhdl, vb, ga, ta, gi, ti):




    fa = 0.4
    fi = 0.3
    fp = 0.3


    if FTYS==0:
        ttol = tsol
        tsol = ttol

    yp = np.zeros(NB)

    if fconvl == 1:
        yp[3] = (ga*thcl*y[4] + ta*thcl*y[6] - thcl*y[3])/vb[3]    # IA
        yp[4] = (gi*thcl*y[5] + ti*thcl*y[7] - ga*thcl*y[4])/vb[4] # II
        yp[5] = gi*thcl*(y[8]-y[5])/vb[5]                          # IP

        yp[6] = thcl*(y[9]-y[6])/vb[6]                             # DA
        yp[7] = ga*thcl*(y[6]-y[7])/vb[7]                          # DI
        yp[8] = gi*thcl*(y[7]-y[8])/vb[8]                          # DP

        yp[9] = thcl*(y[3]-y[9])/vb[9]                             # H-box

    elif fconvl == 2:
        yp[5] = (ga*thcl*y[4] + ta*thcl*y[8] - thcl*y[5])/vb[5]    # IP
        yp[4] = (gi*thcl*y[3] + ti*thcl*y[7] - ga*thcl*y[4])/vb[4] # II
        yp[3] = gi*thcl*(y[6]-y[3])/vb[3]                          # IA
        yp[8] = thcl*(y[9]-y[8])/vb[8]                             # DP
        yp[7] = ga*thcl*(y[8]-y[7])/vb[7]                          # DI
        yp[6] = gi*thcl*(y[7]-y[6])/vb[6]                          # DA
        yp[9] = thcl*(y[5]-y[9])/vb[9]                            # H

        yp[3] = yp[3] + fa*tsol*(y[6]-y[3])/vb[3]                  # IA
        yp[4] = yp[4] + fi*tsol*(y[7]-y[4])/vb[4]                  # II
        yp[5] = yp[5] + fp*tsol*(y[8]-y[5])/vb[5]                  # IP
        yp[6] = yp[6] + fa*tsol*(y[9]-y[6])/vb[6]                  # DA
        yp[7] = yp[7] + fi*tsol*(y[9]-y[7])/vb[7]                  # DI
        yp[8] = yp[8] + fp*tsol*(y[9]-y[8])/vb[8]                  # DP
        yp[9] = yp[9] + (fa*tsol*(y[3]-y[9]) + fi*tsol*(y[4]-y[9]) + fp*tsol*(y[5]-y[9]))/vb[9]

    elif fconvl == 3:
        yp[3] = fa*thcl*(y[6]-y[3])/vb[3]
        yp[4] = fi*thcl*(y[7]-y[4])/vb[4]
        yp[5] = fp*thcl*(y[8]-y[5])/vb[5]
        yp[6] = fa*thcl*(y[9]-y[6])/vb[6]
        yp[7] = fi*thcl*(y[9]-y[7])/vb[7]
        yp[8] = fp*thcl*(y[9]-y[8])/vb[8]
        yp[9] = (fa*thcl*(y[3]-y[9]) + fi*thcl*(y[4]-y[9]) + fp*thcl*(y[5]-y[9]))/vb[9]

    if FTYS:
        yp[10] = ttol*(y[1]-y[10])/vb[10]
        yp[12] = ttol*(y[10]-y[12])/vb[12]
        yp[7] = yp[7] + ttol*(y[12]-y[7])/vb[7]
        yp[4] = yp[4] + ttol*(y[7]-y[4])/vb[4]
        yp[1] = yp[1] + ttol*(y[4]-y[1])/vb[1]

    # mixing AIP H
    # for k in range(3):
    #     yp[k] = yp[k] + mvl[k]*(y[k+3]-y[k])/vb[k]
    #     yp[k+3] = yp[k+3] + mvl[k]*(y[k]-y[k+3])/vb[k+3]
    #     yp[k+6] = yp[k+6] + mhdl[k]*(y[9]-y[k+6])/vb[k+6]
    #     yp[9] = yp[9] + mhdl[k]*(y[k+6]-y[9])/vb[9]
    yp[0:3] = yp[0:3] + mvl[0:3]*(y[3:6]-y[0:3])/vb[0:3]
    yp[3:6] = yp[3:6] + mvl[0:3]*(y[0:3]-y[3:6])/vb[3:6]
    yp[6:9] = yp[6:9] + mhdl[0:3]*(y[9]-y[6:9])/vb[6:9]
    yp[9] = yp[9] + np.sum(mhdl[0:3]*(y[6:9]-y[9])/vb[9])

    if FTYS:
        yp[11] = mvl[3]*(y[10]-y[11])/vb[11]
        yp[10] = yp[10] + mvl[3]*(y[11] -y[10])/vb[10]
        yp[11] = yp[11] + mvl[4]*(y[4]-y[11])/vb[11]
        yp[4] = yp[4] + mvl[4]*(y[11]-y[4])/vb[4]
        yp[10] = yp[10] + mhdl[3]*(y[12]-y[10])/vb[10]
        yp[12] = yp[12] + mhdl[3]*(y[10]-y[12])/vb[12]

    return yp


# def derivs_po4(t, po4):

#     po4p = thmfun(po4,fconv,thc,tso,tto,mxv,mhd,vb,gtha,thbra,gthi,thbri)

#     eplv = fepl * mxv[0:NLS] * po4[kiv]/REDPC



#     pplv = eplv * REDPC       # PO4 mol/year
#     enlv = eplv * REDNC       # NO3 mol/year

#     # total carbon: Corg + CaCO3

#     oip = 0.215

#     # high lat export
#     eph = eph0

#     eah = 0                      ##  2*eph/init.rrain   ###???? 0000?
#     pph = eph * REDPC
#     enh = eph * REDNC
#     # total carbon
#     ech = eph+0.5*eah

#     fopb = 1.6e10
#     ffep = fopb
#     fcap = 2 * fopb

#     fpw = 4 * fopb



#     # bio pump Porg
#     po4p[m1] -= pplv/vb[m1]
#     po4p[m2+3] += frei*pplv/vb[m2 + 3]
#     po4p[m3+6] += oip*pplv/vb[m3 + 6]

#     po4p[9] -= pph/vb[9]

#     po4p[6:9] += (frei+oip) * pph/vb[6:9]*gp[6:9]

#     if pcycle:
#         po4p[m3+6] -= ffep /vb[m3+6]/NOC - fcap /vb[m3+6]/NOC
#         po4p[m1] += fpw/vb[m1]/NOC

#     print(po4p)

#     return po4p
