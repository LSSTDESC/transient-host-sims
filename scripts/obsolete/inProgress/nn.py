#!/global/common/software/lsst/common/miniconda/current/envs/stack/bin/python
# set mode: which class from which to match the hosts
# mode = 'SN Ia'
# mode = 'SN II'
# mode = 'SLSN-I'
# mode = 'SN IIP'
# mode = 'SN IIb'
# mode = 'SN IIn'
# mode = 'SN Ib'
# mode = 'SN Ic'
mode = 'SN Ibc'
plotting = False
full = True
if full:
    tot = 3000000
else:
    tot = 5000
if ' ' in mode:
    modestr = mode.replace(' ','')

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
import faiss


class FaissKNeighbors:
    def __init__(self, n_neighbors=5):
        self.index = None
        self.y = None
        self.k = n_neighbors

#shouldn't need y
    def fit(self, X):
        X = X.copy(order='C')
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        return self

#    def fit(self, X, y):
#        self.index = faiss.IndexFlatL2(X.shape[1])
#        self.index.add(X.astype(np.float32))
#        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

def min_max_normalize(feature, feature_name):
    norm_feature = (feature - np.amin(feature))/(np.amax(feature)-np.amin(feature))
    plt.hist(norm_feature)
    plt.title(feature_name)
    plt.show()
    plt.clf()
    return(norm_feature)

start = time.time()
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

sim_keyparams_norm = np.load("./sim_keyparams_norm.npz")
sim_keyparams_norm = sim_keyparams_norm['arr_0']
sim_full_indices = np.arange(len(sim_keyparams_norm)) # all the indices for the simulation data that's going in

# Dimension of our vector space
dimension = np.shape(sim_keyparams_norm)[1]
print(dimension)

# Create a random binary hash with 10 bits
rbp = RandomBinaryProjections('rbp', 10)

# Create engine with pipeline configuration
engine = Engine(dimension, lshashes=[rbp])

# Index 1000000 random vectors (set their data to a unique string)
for index in np.arange(len(sim_keyparams_norm)):
    row = sim_keyparams_norm[index,:]
    engine.store_vector(row, index)

# Create random query vector
for i in np.arange(8000):
    query = np.random.randn(dimension)

    # Get nearest neighbours
    N = engine.neighbours(query)
    if len(N) > 0:
        N = N[0]
        arr = N[0]
        idx = N[1]
        dist = N[2]
        print(idx)

#distances, indices = self.index.search(X.astype(np.float32), k=self.k)
end = time.time()
print("time:", end-start)

save_array  = []
check_array = []
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
print(cdc2_matched_nn)

if full:
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4")
else:
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")

filters=[(lambda x: np.isin(x, galaxy_ids), 'galaxy_id')]
cdc2_true = cosmo.get_quantities(['mag_true_u_lsst', 'mag_true_g_lsst','mag_true_r_lsst', 
                'mag_true_i_lsst', 'mag_true_z_lsst',
                 'mag_true_Y_lsst','size_true', 'size_minor_true', 
                 'size_disk_true', 'size_minor_disk_true', 'size_bulge_true', 
                 'size_minor_bulge_true','galaxy_id', 'sersic_disk', 'sersic_bulge', 
                 'position_angle_true', 'ra', 'dec'], filters=filters)

#clear cosmo from memory
del cosmo

#add photoz_errs from the other catalog
if full:
    cosmo_pz = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_v1")
else:
    cosmo_pz = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small_with_photozs_v1")

cdc2_pz = cosmo_pz.get_quantities(['galaxy_id', 'mag_err_u_photoz', 'mag_err_g_photoz', 'mag_err_r_photoz', 'mag_err_i_photoz', 'mag_err_z_photoz', 'mag_err_y_photoz', 'photoz_mask'])

#clear cosmo_pz from memory
del cosmo_pz

photoz_mask  = cdc2_pz['photoz_mask']
pz_galaxy_id = cdc2_pz['galaxy_id']
pz_magerr_u  = cdc2_pz['mag_err_u_photoz'][photoz_mask]
pz_magerr_g  = cdc2_pz['mag_err_g_photoz'][photoz_mask]
pz_magerr_r  = cdc2_pz['mag_err_r_photoz'][photoz_mask]
pz_magerr_i  = cdc2_pz['mag_err_i_photoz'][photoz_mask]
pz_magerr_z  = cdc2_pz['mag_err_z_photoz'][photoz_mask]
pz_magerr_y  = cdc2_pz['mag_err_y_photoz'][photoz_mask]

cdc2_pz = pd.DataFrame({'galaxy_id':pz_galaxy_id, 'mag_err_u_photoz':pz_magerr_u, 'mag_err_g_photoz':pz_magerr_g, 'mag_err_r_photoz':pz_magerr_r, 'mag_err_i_photoz':pz_magerr_i, 'mag_err_z_photoz':pz_magerr_z, 'mag_err_y_photoz':pz_magerr_y})
cdc2_true = pd.DataFrame(cdc2_true)
cdc2_pz  = pd.DataFrame(cdc2_pz)
cdc2_nbrs = pd.merge(cdc2_matched_nn, cdc2_true, on=['galaxy_id'], how='left')
cdc2_nbrs_pz = pd.merge(cdc2_nbrs, cdc2_pz, on=["galaxy_id"], how='left')

print(cdc2_nbrs_pz)
# save it all
save_dict = {

'galaxy_id':cdc2_nbrs_pz['galaxy_id'], 'ra':cdc2_nbrs_pz['ra'],'dec':cdc2_nbrs_pz['dec'], 'redshift':cdc2_nbrs_pz['PZflowredshift'], 'mag_true_u_lsst':cdc2_other['mag_true_u_lsst'][sorted_dc2_idx], 'mag_err_u_photoz':cdc2_other['mag_err_u_photoz'][sorted_dc2_idx],
    'mag_true_g_lsst':cdc2_other['mag_true_g_lsst'][sorted_dc2_idx], 'mag_err_g_photoz':cdc2_other['mag_err_g_photoz'][sorted_dc2_idx],'mag_true_r_lsst':cdc2_other['mag_true_r_lsst'][sorted_dc2_idx],'mag_err_r_photoz':cdc2_other['mag_err_r_photoz'][sorted_dc2_idx], 'mag_true_i_lsst':cdc2_other['mag_true_i_lsst'][sorted_dc2_idx],'mag_err_i_photoz':cdc2_other['mag_err_i_photoz'][sorted_dc2_idx], 'mag_true_z_lsst':cdc2_other['mag_true_z_lsst'][sorted_dc2_idx],'mag_err_z_photoz':cdc2_other['mag_err_z_photoz'][sorted_dc2_idx], 'mag_true_Y_lsst':cdc2_other['mag_true_Y_lsst'][sorted_dc2_idx],'mag_err_y_photoz':cdc2_other['mag_err_y_photoz'][sorted_dc2_idx],'size_true':cdc2_other['size_true'][sorted_dc2_idx],'size_minor_true':cdc2_other['size_minor_true'][sorted_dc2_idx],'totalSersicIndex':cdc2['morphology/totalSersicIndex'].to_numpy()[matched_indices],'position_angle_true':cdc2_other['position_angle_true'][sorted_dc2_idx], 'totalEllipticity':cdc2['morphology/totalEllipticity'].to_numpy()[matched_indices],'stellar_mass':cdc2['stellar_mass'].to_numpy()[matched_indices], 'SFRtot':cdc2['PZflowSFRtot'].to_numpy()[matched_indices],'Mag_true_g_sdss_z0':cdc2['Mag_true_g_sdss_z0'].to_numpy()[matched_indices],'GHOST_objID':ghost['objID'][save_array_uniques_neworder[:,1]], 'nn_distance':save_array_uniques_neworder[:,2]}
    
save_df = pd.DataFrame(save_dict)
save_df.to_csv('/global/cscratch1/sd/mlokken/sn_hostenv/cdc2_matched_ghost_{:s}_unq_zwgt_5pct_k{:d}.csv'.format(modestr, n_neigh), index=False)

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

                
