#!/global/common/software/lsst/common/miniconda/current/envs/stack/bin/python
# set mode: which class from which to match the hosts
import sys
from sklearn.neighbors import NearestNeighbors
import numpy as np
from matplotlib import pyplot as plt
import os
import GCRCatalogs
from astropy.io import fits
import pandas as pd
from astropy.cosmology import Planck15 as P15
from astropy import units as u
import matplotlib
import time
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
import numpy.ma as ma
import multiprocessing as mp
import sys

plotting = False
full = True

if full:
    tot = 3000000
else:
    tot = 5000

start = time.time()
#modes = np.array(['SN Ia', 'SN II', 'SLSN-I', 'SN IIP', 'SN IIb', 'SN IIn', 'SN Ib', 'SN Ic', 'SN Ibc'])
#modes = np.array(['SN Ia', 'SN II', 'SN Ibc'])
modes = np.array(['rand'])

#neigh_dict = {'SN Ia':407, 'SN II':1040, 'SN Ibc':8902}
#neigh_dict = {'SN Ia':815, 'SN II':2081, 'SN Ibc':17804}
#neigh_dict = {'SN Ia':678, 'SN II':3147, 'SN Ibc':9508}
#neigh_dict = {'SN Ia':678, 'SN II':3147, 'SN Ibc':9508}
neigh_dict = {'rand':0}
#neigh_dict = {'SN Ia':1356, 'SN II':6295, 'SN Ibc':19017}
#neigh_dict = {'SN Ia':381, 'SN II':1282, 'SLSN-I':103448, 'SN IIP':10791, 'SN IIb':34482, 'SN IIn':12448, 'SN Ib':21582, 'SN Ic':14354, 'SN Ibc':7957}

if full:
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4")
    print("Loaded cosmoDC2 catalog")
else:
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")
    print("Loaded cosmoDC2 catalog")

for mode in modes:

    n_neigh = neigh_dict[mode]

    if ' ' in mode:
        modestr = mode.replace(' ','')
    else:
        modestr = mode

#add the ghost matched catalog
#    cdc2_matched_nn = pd.read_csv("/global/cscratch1/sd/agaglian/matchedSamples_1218/cdc2_matched_ghost_%s_z3_unq_zwgt_5pct_k%i.tar.gz" % (modestr, n_neigh), memory_map=True, low_memory=True)
    cdc2_matched_nn = pd.read_csv("/global/cscratch1/sd/agaglian/matchedSamples_1218/rand_hostlib_SFRMsol.tar.gz", memory_map=True, low_memory=True)

    print("Loaded the GHOST-DC2 matched catalog")

    galaxy_ids = cdc2_matched_nn['galaxy_id']
    features = np.array(['galaxy_id', 'bulge_to_total_ratio_i'])

    filters=[(lambda x: np.isin(x, galaxy_ids), 'galaxy_id')]
    cdc2_true = {}
    for feature in features:
        cdc2_true[feature] = []
    c = 0
    print(cdc2_true.keys())
    start_time = time.time()
    for gal in cosmo.get_quantities(features, filters=filters, return_iterator=True):
        for feature in features:
           cdc2_true[feature].append(gal[feature]) 
        elapsed = time.time() - start_time
        print(c, "Elapsed time = ", elapsed)
        c+=1
    for feature in features:
        cdc2_true[feature] = np.concatenate(cdc2_true[feature])

    cdc2_true = pd.DataFrame(cdc2_true)
    cdc2_nbrs = pd.merge(cdc2_matched_nn, cdc2_true, on=['galaxy_id'], how='left')

    #cdc2_nbrs.to_csv('/global/cscratch1/sd/agaglian/matchedSamples_1218/cdc2_matched_ghost_%s_z3_unq_zwgt_5pct_k%i_SFRMsol.tar.gz' % (modestr, n_neigh), index=False) 
    cdc2_nbrs.to_csv('/global/cscratch1/sd/agaglian/matchedSamples_1218/rand_hostlib_SFRMsol_bulge.tar.gz', index=False)


    print("Saved file for %s."%mode)

    end = time.time()
    elapsed = end - start
    print("time: %.2f" % (elapsed))

end = time.time()
elapsed = end - start
print("Total time for adding DC2 true features: %.2f" % (elapsed))
