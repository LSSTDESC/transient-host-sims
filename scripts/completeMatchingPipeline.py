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

print("Starting our matching sequence!") 

if full:
    tot = 3000000
else:
    tot = 5000

def nn(gal):
    u = AnnoyIndex(dimension, 'angular')
    u.load('/global/cscratch1/sd/agaglian/build_cdc2.ann') # super fast, will just mmap the file

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

if full:
    cdc2 = pd.read_csv("/global/homes/a/agaglian/data_files/ghost_matched/DC2full_pzRedshifts_SFR_39iter.csv", memory_map=True, low_memory=True)
    cosmo_pz = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_v1")
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4")
    print("Loaded pzflow-oversampled catalog")
else:
    cdc2 = pd.read_csv("/global/cscratch1/sd/mlokken/sn_hostenv/DC2_pzRedshifts_SFR_RMag_lt_neg15.csv", memory_map=True, low_memory=True)
    cosmo_pz = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small_with_photozs_v1")
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")
    print("Loaded pzflow-oversampled catalog")

fn = '/global/homes/a/agaglian/data_files/GHOST_restFrame_condenseLabels_0323.tar.gz'
ghost_orig = pd.read_csv(fn, memory_map=True, low_memory=True)

#!/global/common/software/lsst/common/miniconda/current/envs/stack/bin/python
# set mode: which class from which to match the hosts

modes = np.array(['SN Ia', 'SN II', 'SLSN-I', 'SN IIP', 'SN IIb', 'SN IIn', 'SN Ib', 'SN Ic', 'SN Ibc'])

# read in file of CosmoDC2 galaxies, with PZFlow SFR and redshifts, limited to abs r-band magnitude < -15
# and -0.18 < i-z < 0.5
#add photoz_errs from the other catalog
features = np.array(['galaxy_id', 'photoz_mask', 'mag_err_u_photoz', 'mag_err_g_photoz', 'mag_err_r_photoz', 'mag_err_i_photoz','mag_err_z_photoz', 'mag_err_y_photoz'])

start_time = time.time()
galid = []
photoz_mask  = []
magerr_u   = []
magerr_g      = []
magerr_r   = []
magerr_i      = []
magerr_z      = []
magerr_y      = []

c = 0
for gal in cosmo_pz.get_quantities(features, return_iterator=True):
    galid.append(gal['galaxy_id'])
    photoz_mask.append(gal['photoz_mask'])
    magerr_u.append(gal['mag_err_u_photoz'][gal['photoz_mask']])
    magerr_g.append(gal['mag_err_g_photoz'][gal['photoz_mask']])
    magerr_r.append(gal['mag_err_r_photoz'][gal['photoz_mask']])
    magerr_i.append(gal['mag_err_i_photoz'][gal['photoz_mask']])
    magerr_z.append(gal['mag_err_z_photoz'][gal['photoz_mask']])
    magerr_y.append(gal['mag_err_y_photoz'][gal['photoz_mask']])

    elapsed = time.time() - start_time
    print(c, "Elapsed time = ", elapsed)
    c+=1
    
gals = cosmo_pz.get_quantities(features, return_iterator=True)
pool = mp.Pool(mp.cpu_count())
results = pool.map(process_gal, gals)
pool.close()
sys.exit()

cdc2_pz = {}
cdc2_pz['galaxy_id'] = np.concatenate(galid)
cdc2_pz['mag_err_u_photoz'] = np.concatenate(magerr_u)
cdc2_pz['mag_err_g_photoz'] = np.concatenate(magerr_g)
cdc2_pz['mag_err_r_photoz'] = np.concatenate(magerr_r)
cdc2_pz['mag_err_i_photoz']  = np.concatenate(magerr_i)
cdc2_pz['mag_err_z_photoz']  = np.concatenate(magerr_z)
cdc2_pz['mag_err_y_photoz']  = np.concatenate(magerr_y)

cdc2_pz = pd.DataFrame(cdc2_pz)

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
        a = ghost_orig['TransientClass']==mode
        b = ghost_orig['TransientClass']=='SN IIP'
        c = ghost_orig['TransientClass']=='SN IIn'
        ghost = ghost_orig[a|b|c]
    elif mode == 'SN Ibc': # group all stripped-envelope supernovae
        a = ghost_orig['TransientClass'] == 'SN Ib'
        b = ghost_orig['TransientClass']=='SN IIb'
        c = ghost_orig['TransientClass'] == 'SN Ic'
        d = ghost_orig['TransientClass'] == 'SLSN-I' # lump in SLSN hosts because there are too few of them for it to be its own hostlib
        ghost = ghost_orig[a | b | c | d]
    else:
        ghost = ghost_orig[ghost_orig['TransientClass']==mode]
    ghost.reset_index(inplace=True, drop=True)
    print("Number of {:s}: {:d}".format(mode,len(ghost)))
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

    data_keyparams= np.vstack((gR, gI, g_gr, g_iz, g_ellip, g_rshift)).T

    # normalize for knn
    # The purpose of is this is so that the nearest-neighbors algorithm is searching in a multidimensional space
    # where typical distances are similar in all dimensions
    #ghost_cdc2 = np.vstack((data_keyparams, sim_keyparams))
    scaler = StandardScaler()
    #scaler.fit(ghost_cdc2)
    scaler.fit(data_keyparams)
    data_keyparams_norm = scaler.transform(data_keyparams)
    #data_keyparams_norm = keyparams_norm[0:len(data_keyparams[:,0]),:]
    #sim_keyparams_norm  = keyparams_norm[len(data_keyparams[:,0]):,:]

    # Weight redshift
    data_keyparams_norm[:,5]/=div

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
    sime_reshaped  = (sim_keyparams[:,4][indices.flatten()]).reshape(sim_idx_shape)
    simrshift_reshaped  = (sim_keyparams[:,5][indices.flatten()]).reshape(sim_idx_shape)

    for j in range(len(simR_reshaped)):
        for m in range(simR_reshaped.shape[1]):
            line_to_add = np.array((simR_reshaped[j][m],simI_reshaped[j][m],simgr_reshaped[j][m],simiz_reshaped[j][m], sime_reshaped[j][m],simrshift_reshaped[j][m], og_ghost_idx[j],dist[j][m]))
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
    matched_indices = save_array_uniques[:,0].astype(int)
    cdc2_matched = cdc2.iloc[matched_indices]
    galaxy_ids = cdc2_matched['galaxy_id']
    nn_dict = {'GHOST_objID':ghost['objID'].to_numpy()[save_array_uniques[:,1].astype(int)], 'nn_distance':save_array_uniques[:,2]}
    nn_df   = pd.DataFrame(nn_dict)
    nn_df.reset_index(inplace=True, drop=True)
    cdc2_matched.reset_index(inplace=True, drop=True)
    cdc2_matched_nn = pd.concat([cdc2_matched, nn_df], axis=1)
    cdc2_matched_nn.reset_index(inplace=True, drop=True)
#    cdc2_matched_nn.to_csv("/global/cscratch1/sd/agaglian/matchedDC2_%s_%i.tar.gz" % (modestr, n_neigh), index=False)

    features = ['mag_true_u_lsst', 'mag_true_g_lsst','mag_true_r_lsst',
                    'mag_true_i_lsst', 'mag_true_z_lsst',
                     'mag_true_Y_lsst','size_true', 'size_minor_true',
                     'size_disk_true', 'size_minor_disk_true', 'size_bulge_true',
                     'size_minor_bulge_true','galaxy_id', 'sersic_disk', 'sersic_bulge',
                     'position_angle_true', 'ra', 'dec']

    filters=[(lambda x: np.isin(x, galaxy_ids), 'galaxy_id')]
    cdc2_true = {}
    for feature in features:
        cdc2_true[feature] = []
    c = 0
    start_time = time.time()
    for gal in cosmo.get_quantities(features, filters=filters, return_iterator=True):
        for features in features:
           cdc2_true[feature].append(gal[feature])
        elapsed = time.time() - start_time
        print(c, "Elapsed time = ", elapsed)
        c+=1
    for feature in features:
        cdc2_true[feature] = np.concatenate(cdc2_true[feature])

    cdc2_true = pd.DataFrame(cdc2_true)
    cdc2_nbrs = pd.merge(cdc2_matched_nn, cdc2_true, on=['galaxy_id'], how='left')

    cdc2_nbrs.to_csv('/global/cscratch1/sd/agaglian/cdc2_matched_wTrue_{:s}_unq_zwgt_5pct_k{:d}.tar.gz'.format(modestr, n_neigh), index=False)

    print("Saved file for %s."%mode)

    end = time.time()
    elapsed = end - start
    print("time: %.2f" % (elapsed))

    if plotting:
        if not os.path.exists('../plots/{:s}'.format(modestr)):
            os.mkdir('../plots/{:s}'.format(modestr))
        sns.set_context("poster")
        fig, ax = plt.subplots(nrows=2,ncols=1, figsize=[10,10], sharex=True)
        ax[0].hist(ghost['TransientRedshift'],bins=40, alpha=0.5, color='blue', label='GHOST, {:d} total'.format(len(data_keyparams)))
        ax[1].hist(cdc2['PZflowredshift'][save_array_uniques[:,0]],bins=40,alpha=0.5, color='orange', label='CosmoDC2, {:d} total'.format(len(save_array_uniques)))
        plt.xlabel("redshift")
        plt.legend()
        ax[0].set_ylabel("Number of galaxies")
        ax[1].set_ylabel("Number of galaxies")
        ax[1].ticklabel_format(axis='y', style='sci')
        ax[0].legend()
        plt.savefig("../plots/{0}/cdc2_ghost_{0}_k10_lowz_weighting_matches_unq_histogram.png".format(modestr))

        # check all properties against each other for CosmoDC2
        labels=['R', 'I', 'g-r', 'i-z', 'ellipticity', 'redshift']
        lims =[[-25,-14.5],[-25,-14.5],[-0.3,1.2],[-0.2,0.5],[0,0.4],[0,1]]
        for i in range(6):
            for j in range(6):
                if i!=j:
                    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=[7,10],sharex=True)
                    im2=ax[1].scatter(data_keyparams[:,j],data_keyparams[:,i],c=dist[:,0], cmap='plasma_r', alpha=0.5, s=1, vmin=0, vmax=3)
                    im1=ax[0].scatter(check_array_uniques[:,j],check_array_uniques[:,i],c=check_array_uniques[:,7], cmap='plasma_r', alpha=0.2, s=1, vmin=0, vmax=3)
                    ax[1].set_xlabel(labels[j])
                    ax[0].set_ylabel(labels[i])
                    ax[1].set_ylabel(labels[i])
                    ax[0].set_title("CDC2 matched gals")
                    ax[1].set_title("GHOST")
                    fig.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    cbar = fig.colorbar(im1, cax=cbar_ax)
                    cbar.set_label("distance")
                    ax[0].set_xlim(lims[j])
                    ax[1].set_xlim(lims[j])
                    ax[1].set_ylim(lims[i])
                    ax[0].set_ylim(lims[i])
                    plt.savefig("../plots/{:s}/ghost_cdc2_k10_lowz_weighting_unq_{:s}_vs_{:s}_{:s}.png".format(modestr,labels[i],labels[j],modestr), bbox_inches='tight')
                    plt.clf()

end = time.time()
elapsed = end - start
print("Total time elapsed for matching + true DC2 features: %.2f s" % elapsed)


