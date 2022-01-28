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

if full:
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4")
    print("Loaded cosmoDC2 catalog")
else:
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")
    print("Loaded cosmoDC2 catalog")

modes = ['rand']
for mode in modes:

    if ' ' in mode:
        modestr = mode.replace(' ','')
    else:
        modestr = mode

#add the ghost matched catalog
    cdc2_matched_nn = pd.read_csv("/global/cscratch1/sd/agaglian/randLib/rand_hostlib_Ixx.csv")

    print("Loaded the GHOST-DC2 matched catalog")

    galaxy_ids = cdc2_matched_nn['galaxy_id']

    features = np.array(['bulge_to_total_ratio_i', 'galaxy_id'])

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

    cdc2_nbrs.to_csv("/global/cscratch1/sd/agaglian/randLib/rand_hostlib_wIxx.tar.gz")

    print("Saved file for %s."%mode)

    end = time.time()
    elapsed = end - start
    print("time: %.2f" % (elapsed))

end = time.time()
elapsed = end - start
print("Total time for adding DC2 true features: %.2f" % (elapsed))
