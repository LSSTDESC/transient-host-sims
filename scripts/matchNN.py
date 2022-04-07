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
# from nearpy import Engine
# from nearpy.hashes import RandomBinaryProjections
# from nearpy.filters import NearestFilter, UniqueFilter
# from nearpy.distances import EuclideanDistance
from sklearn.preprocessing import StandardScaler
import numpy.ma as ma
import multiprocessing as mp
import sys

full = True

print("Starting our matching sequence!") 

if full:
    tot = 12000000
else:
    tot = 5000

def nn(gal):
    u = AnnoyIndex(dimension, 'euclidean')
    #u.load('/global/cscratch1/sd/agaglian/build_cdc2.ann') # super fast, will just mmap the file
    u.load("/global/cscratch1/sd/mlokken/sn_hostenv/build_cdc2_euclidean_z3_31healpix_updMag.ann")

    # Get nearest neighbours
    N = u.get_nns_by_vector(gal, n_neigh, search_k=-1, include_distances=True)
    return N

def min_max_normalize(feature, feature_name):
    norm_feature = (feature - np.amin(feature))/(np.amax(feature)-np.amin(feature))
    plt.hist(norm_feature)
    plt.title(feature_name)
    plt.show()
    plt.clf()
    return(norm_feature)


fn = "./ghost_scaled.csv"
ghost_orig = pd.read_csv(fn, memory_map=True, low_memory=True)
# fn = '/global/cscratch1/sd/agaglian/GHOSTwithImageSizes.csv'
fn = '../data_files/GHOST_flagCuts_restFrame.csv'
ghost_full = pd.read_csv(fn, memory_map=True, low_memory=True)
#og_ghost_idx = np.arange(len(ghost_full))
#og_ghost_idx = ghost_orig['og_idx']

#!/global/common/software/lsst/common/miniconda/current/envs/stack/bin/python
# set mode: which class from which to match the hosts

modes = np.array(['SN Ia', 'SN II', 'SN Ibc'])

# read in file of CosmoDC2 galaxies, with PZFlow SFR and redshifts, limited to abs r-band magnitude < -15
# and -0.18 < i-z < 0.5

if full:
    # cdc2 = pd.read_csv("/global/cscratch1/sd/agaglian/DC2full_pzRedshifts_twentyHealpix_sdss_updMag_Rkpc_Final.tar.gz", memory_map=True, low_memory=True)
    cdc2 = pd.read_csv("/global/cscratch1/sd/mlokken/sn_hostenv/DC2full_pzRedshifts_31Healpix_sdss_updMag_Rkpc_Final.tar.gz")
    print("Loaded pzflow-oversampled catalog")
else:
    cdc2 = pd.read_csv("/global/cscratch1/sd/mlokken/sn_hostenv/DC2_pzRedshifts_SFR_RMag_lt_neg15.csv", memory_map=True, low_memory=True)
    print("Loaded pzflow-oversampled catalog")

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
#c_ellip = cdc2['morphology/totalEllipticity'
#c_rkpc= cdc2['R_kpc']
c_rshift = cdc2['PZflowredshift']

sim_keyparams= np.vstack((cR, cI, c_gr, c_iz, c_rshift)).T # c_rkpc,

div = 20.
percentage = 1/div*100
print("Redshift weighting: {:.1f} percent".format(percentage))

start = time.time()

# Dimension of our vector space
dimension = np.shape(sim_keyparams)[1]
print("Number of nearest-neighbors matching features:", dimension)

from annoy import AnnoyIndex
import random

for mode in modes:

    if ' ' in mode:
        modestr = mode.replace(' ','')

    if mode == 'SN II':
        ghost = ghost_orig[ghost_orig['TransientClass'].isin(['SN II', 'SN IIP', 'SN IIn'])]
    elif mode == 'SN Ibc':
        # lump in SLSN hosts because there are too few of them for it to be its own hostlib
        ghost = ghost_orig[ghost_orig['TransientClass'].isin(['SN Ib', 'SN Ic', 'SN Ib/c', 'SN IIb', 'SN Ic', 'SNIbc', 'SNIb/c', 'SLSN-I', 'SLSN'])]
    elif mode == 'SN Ia':
        ghost = ghost_orig[ghost_orig['TransientClass'].isin(['SN Ia', 'SN Ia Pec'])]
    og_ghost_idx = ghost['og_idx'].values
    # ghost is already normalized, as read from ghost_scaled.csv
    data_keyparams_norm = np.array(ghost[['R', 'I', 'g-r', 'i-z', 'redshift']])
    
    # Weight redshift
    data_keyparams_norm[:,4]/=div

    global n_neigh
    n_neigh = int(tot/len(data_keyparams_norm))
    print("Number of neighbors: ", n_neigh)

    start = time.time()
    # Query each of the ghost galaxies in the database
    dist = []
    indices = []
    ghost_vecs = [data_keyparams_norm[i, :] for i in np.arange(len(data_keyparams_norm))]

    m = 0
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(nn, ghost_vecs, n_neigh)
    pool.close()

    indices = np.array([x[0] for x in results])
    dist = np.array([x[1] for x in results])

    end = time.time()
    print("time:", end-start)

    save_array  = []
    check_array = []

    #indices = int(indices) #convert to integers
    sim_idx_shape  = indices.shape
    simR_reshaped  = (sim_keyparams[:,0][indices.flatten()]).reshape(sim_idx_shape)
    simI_reshaped  = (sim_keyparams[:,1][indices.flatten()]).reshape(sim_idx_shape)
    simgr_reshaped  = (sim_keyparams[:,2][indices.flatten()]).reshape(sim_idx_shape)
    simiz_reshaped  = (sim_keyparams[:,3][indices.flatten()]).reshape(sim_idx_shape)
    simrshift_reshaped  = (sim_keyparams[:,4][indices.flatten()]).reshape(sim_idx_shape)
                        
    for j in range(len(simR_reshaped)):
        for m in range(simR_reshaped.shape[1]):
            line_to_add = np.array((simR_reshaped[j][m],simI_reshaped[j][m],simgr_reshaped[j][m],simiz_reshaped[j][m], simrshift_reshaped[j][m], og_ghost_idx[j],dist[j][m]))
            check_array.append(line_to_add)
            save_array.append(np.array((indices[j][m],og_ghost_idx[j],dist[j][m]))) # index of CosmoDC2 galaxy from resampled,index of GHOST from original file,distance between matches
            
    save_array = np.asarray(save_array)
    check_array = np.asarray(check_array)
    #find duplicates
    galaxy_ids = cdc2['galaxy_id'].to_numpy()[save_array[:,0].astype(int)]
    # quick version. Takes only unique cosmoDC2 galaxies from array. Indices returns the first instance for any duplicate galaxy.
    # Not good if you want to track closest nn distance.
    unq, idx = np.unique(galaxy_ids, return_index=True)
    check_array_uniques = check_array[idx]

    save_array_uniques  = save_array[idx]
    #print(np.shape(save_array_uniques))
    #print(save_array_uniques)
    matched_indices = save_array_uniques[:,0].astype(int)
    cdc2_matched = cdc2.iloc[matched_indices]
    galaxy_ids = cdc2_matched['galaxy_id']
    nn_dict = {'GHOST_objID':ghost_full['objID'].to_numpy()[save_array_uniques[:,1].astype(int)], 'nn_distance':save_array_uniques[:,2]}
    nn_df   = pd.DataFrame(nn_dict)
    nn_df.reset_index(inplace=True, drop=True)
    cdc2_matched.reset_index(inplace=True, drop=True)
    cdc2_matched_nn = pd.concat([cdc2_matched, nn_df], axis=1)
    cdc2_matched_nn.reset_index(inplace=True, drop=True)
    cdc2_matched_nn.to_csv("/global/cscratch1/sd/agaglian/matchedDC2_euclid_z3_%s_%i.tar.gz" % (modestr, n_neigh), index=False)


end = time.time()
elapsed = end - start
print("Total time elapsed for matching: %.2f s" % elapsed)


