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
print("Made it into the file!")
from sklearn.preprocessing import StandardScaler
import numpy.ma as ma
import sys

#mode = sys.argv[1]
plotting = False
full = True
if full:
    tot = 3000000
else:
    tot = 5000

#!/global/common/software/lsst/common/miniconda/current/envs/stack/bin/python
# set mode: which class from which to match the hosts

# read in file of CosmoDC2 galaxies, with PZFlow SFR and redshifts, limited to abs r-band magnitude < -15
# and -0.18 < i-z < 0.5
if full:
    cdc2 = pd.read_csv("/global/homes/a/agaglian/data_files/ghost_matched/DC2full_pzRedshifts_SFR_39iter.csv", memory_map=True, low_memory=True)
else:
    cdc2 = pd.read_csv("/global/cscratch1/sd/mlokken/sn_hostenv/DC2_pzRedshifts_SFR_RMag_lt_neg15.csv", memory_map=True, low_memory=True)

cI = cdc2['Mag_true_i_sdss_z0']
cZ = cdc2['Mag_true_z_sdss_z0']
c_iz = cI-cZ
keep = (c_iz < 0.5)&(c_iz>-0.18)
cdc2 = cdc2.loc[keep]
cdc2.reset_index(inplace=True, drop=True)
cG = cdc2['Mag_true_g_sdss_z0']
cR = cdc2['Mag_true_r_sdss_z0']
cI = cI.loc[keep]
cZ = cZ.loc[keep]
c_iz = c_iz.loc[keep]
c_gr = cG-cR
c_ellip = cdc2['morphology/totalEllipticity']
c_rshift = cdc2['PZflowredshift']

sim_keyparams= np.vstack((cR, cI, c_gr, c_iz, c_ellip, c_rshift)).T

# The purpose of is this is so that the nearest-neighbors algorithm is searching in a multidimensional space
# where typical distances are similar in all dimensions
scaler = StandardScaler()
scaler.fit(sim_keyparams)
sim_keyparams_norm = scaler.transform(sim_keyparams)

div = 20.
sim_keyparams_norm[:,5]/=div

#get rid of a few just for testing
#sim_keyparams_norm = sim_keyparams_norm[::10, :]

print(np.shape(sim_keyparams_norm))

sim_full_indices = np.arange(len(sim_keyparams)) # all the indices for the simulation data that's going in
start = time.time()

# Dimension of our vector space
dimension = np.shape(sim_keyparams_norm)[1]
print("Number of nearest-neighbors matching features:", dimension)

from annoy import AnnoyIndex
import random

t = AnnoyIndex(dimension, 'angular')  # Length of item vector that will be indexed

for index in np.arange(len(sim_keyparams_norm)):
    row = sim_keyparams_norm[index,:]
    t.add_item(index, row)

t.build(10) # 10 trees

t.save('/global/cscratch1/sd/agaglian/build_cdc2.ann')
