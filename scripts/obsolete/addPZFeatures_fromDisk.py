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

start = time.time()

plotting = False
full = True

if full:
    tot = 3000000
else:
    tot = 5000

modes = np.array(['SN Ia', 'SN II', 'SLSN-I', 'SN IIP', 'SN IIb', 'SN IIn', 'SN Ib', 'SN Ic', 'SN Ibc'])

# read in file of CosmoDC2 galaxies, with PZFlow SFR and redshifts, limited to abs r-band magnitude < -15
# and -0.18 < i-z < 0.5
cdc2_pz = pd.read_csv("/global/cscratch1/sd/agaglian/photoz_catalog.tar.gz")
print("Loaded the cosmoDC2 image catalog with photo-zs!")

#add photoz_errs from the other catalog

neigh_dict = {'SN Ia':381, 'SN II':1282, 'SLSN-I':103448, 'SN IIP':10791, 'SN IIb':34482, 'SN IIn':12448, 'SN Ib':21582, 'SN Ic':14354, 'SN Ibc':7957}

#clear cosmo_pz from memory
#del cosmo_pz

for mode in modes:

    n_neigh = neigh_dict[mode]

    if ' ' in mode:
        modestr = mode.replace(' ','')
    else:
        modestr = mode

    cdc2_nbrs = pd.read_csv('/global/cscratch1/sd/agaglian/cdc2_matched_wTrue_{:s}_unq_zwgt_5pct_k{:d}.csv'.format(modestr, n_neigh))
    print("Loaded up cdc2_nbrs catalog")

    cdc2_nbrs_pz = pd.merge(cdc2_nbrs, cdc2_pz, on=["galaxy_id"], how='left')

    print("Time to save it all.")

    # save it all
    cdc2_nbrs_pz['redshift'] = cdc2_nbrs_pz['PZFlowredshift']
    cdc2_nbrs_pz['totalEllipticity'] = cdc2_nbrs_pz['morphology/totalEllipticity']
    cdc2_nbrs_pz['SFRtot'] = cdc2_nbrs_pz['PZflowSFRtot']

    del cdc2_nbrs_pz['PZFlowredshift']
    del cdc2_nbrs_pz['morphology/totalEllipticity']
    del cdc2_nbrs_pz['PZflowSFRtot']

    cdc2_nbrs_pz.rest_index(inplace=True, drop=True)
    #sort by objID
    cdc2_nbrs_pz.sort_values(by=['galaxy_id'], inplace=True)

    cdc2_nbrs_pz.to_csv('/global/cscratch1/sd/agaglian/cdc2_matched_ghost_{:s}_unq_zwgt_5pct_k{:d}.tar.gz'.format(modestr, n_neigh), index=False)

    print("Saved full dataset for %s."%mode)
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
print("Total time for adding PZ Features: %.2f" % (elapsed))
