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
from itertools import product
from collections import Counter
print("Made it into the file!")
from sklearn.preprocessing import StandardScaler
import numpy.ma as ma
import multiprocessing as mp
import sys

#mode = sys.argv[1]
plotting = False
full = True
if full:
    tot = 3000000
else:
    tot = 5000

fn = '/global/homes/a/agaglian/data_files/GHOST_restFrame_condenseLabels_0323.tar.gz'
ghost = pd.read_csv(fn)

transient_class = ghost['TransientClass']
gMag_G = ghost['gKronMag_SDSS_abs']
gMag_R = ghost['rKronMag_SDSS_abs']
gMag_I = ghost['iKronMag_SDSS_abs']
gMag_Z = ghost['zKronMag_SDSS_abs']
g_rshift = ghost['NED_redshift']
g_rshift2 = ghost['TransientRedshift']
g_ellip  = ghost['r_ellip']
g_gr   = ghost['g-r_SDSS_rest']
g_ri   = ghost['r-i_SDSS_rest']
g_iz   = ghost['i-z_SDSS_rest']

# keep track of indices from original file
og_ghost_idx = np.arange(len(ghost))
keydata = np.vstack((gMag_G, gMag_R, gMag_I, gMag_Z, g_gr, g_ri, g_iz, g_ellip, g_rshift, g_rshift2)).T
# first remove all -999s:
keydata[np.logical_or(keydata<-50,keydata>100)] = np.nan
# get rid of redshifts with nan
delete_znans = []
z_nans = 0
for i in range(len(keydata)):
    if np.isnan(keydata[i,8]):
        z_nans += 1
for i in range(len(keydata)):
    if np.isnan(keydata[i,8]):
        # if transient redshift is not nan, replace with transient redshift
        if not np.isnan(keydata[i,9]):
            keydata[i,8] = keydata[i,9]
        else:
            delete_znans.append(i)
    if keydata[i,8] <= 0:
        delete_znans.append(i)
keydata = np.delete(keydata, delete_znans, axis=0)
og_ghost_idx = np.delete(og_ghost_idx, delete_znans)
delete_rows = []
# delete rows with more than one nan
for i in range(len(keydata)):
    if np.isnan(np.sum(keydata[i])):
        nan_counter = 0
        for j in range(1, len(keydata[i])):
            if np.isnan(keydata[i,j]):
                nan_counter+=1
        if nan_counter > 1:
            delete_rows.append(i)
keydata = np.delete(keydata, delete_rows, axis=0)
og_ghost_idx = np.delete(og_ghost_idx, delete_rows)
# finally for rows with just one nan, replace with the average value
for i in range(len(keydata)):
    if np.isnan(np.sum(keydata[i])):
        for j in range(1, len(keydata[i])):
            if np.isnan(keydata[i,j]):
                keydata[i,j] = np.nanmean(keydata[:,j])

gG = keydata[:,0]
gR = keydata[:,1]
gI = keydata[:,2]
gZ = keydata[:,3]
g_gr = keydata[:,4]
g_ri   = keydata[:,5]
g_iz   = keydata[:,6]
g_ellip = keydata[:,7]
g_rshift = keydata[:,8]
ghost_objIDs = ghost['objID'].values[og_ghost_idx]

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
data_keyparams= np.vstack((gR, gI, g_gr, g_iz, g_ellip, g_rshift)).T

# The purpose of is this is so that the nearest-neighbors algorithm is searching in a multidimensional space
# where typical distances are similar in all dimensions
ghost_cdc2 = np.vstack((data_keyparams, sim_keyparams))
scaler = StandardScaler()
scaler.fit(ghost_cdc2)
keyparams_norm = scaler.transform(ghost_cdc2)
data_keyparams_norm = keyparams_norm[0:len(data_keyparams[:,0]),:]
sim_keyparams_norm  = keyparams_norm[len(data_keyparams[:,0]):,:]

div = 20.
data_keyparams_norm[:,5]/=div
sim_keyparams_norm[:,5]/=div

ghost_scaled = pd.DataFrame(data=data_keyparams_norm, columns=['R', 'I', 'g-r', 'i-z', 'ellipticity', 'redshift'])
dc2_scaled = pd.DataFrame(data=sim_keyparams_norm, columns=['R', 'I', 'g-r', 'i-z', 'ellipticity', 'redshift'])
ghost_scaled['TransientClass'] = transient_class
ghost_scaled['objID'] = ghost_objIDs
ghost_scaled['og_idx'] = og_ghost_idx
ghost_scaled.to_csv("ghost_scaled.csv", index=False)

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

#t = AnnoyIndex(dimension, 'angular')  # Length of item vector that will be indexed
t = AnnoyIndex(dimension, 'euclidean')

#DC2_vecs = [sim_keyparams_norm[index,:] for index in np.arange(len(sim_keyparams_norm))]
#idxs = [index for index in np.arange(len(sim_keyparams_norm))]
#DC2_vecs = np.array(DC2_vecs)
#idxs = np.array(idxs)
#pool = mp.Pool(mp.cpu_count())
#results = pool.starmap(t.add_item, product(idxs, DC2_vecs))
#pool.close()

for index in np.arange(len(sim_keyparams_norm)):
    row = sim_keyparams_norm[index,:]
    t.add_item(index, row)
    if index %10000 == 0:
        print(index)

print("Successfully added all items")

t.build(10) # 10 trees

t.save('/global/cscratch1/sd/agaglian/build_cdc2_euclidean.ann')

end = time.time()

print(end - start)
