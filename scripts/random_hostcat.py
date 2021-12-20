tot = 3000000

import numpy as np
from matplotlib import pyplot as plt
import os
import GCRCatalogs
from astropy.io import fits
import pandas as pd
from astropy import units as u
import matplotlib
import time
import seaborn as sns
import numpy.ma as ma

# read in file of CosmoDC2 galaxies, with PZFlow SFR and redshifts, limited to abs r-band magnitude < -15
# and -0.18 < i-z < 0.5
cdc2 = pd.read_csv("/global/cscratch1/sd/mlokken/sn_hostenv/DC2full_pzRedshifts_twentyHealpix_sdss_updMag_Rkpc_Final.tar.gz", memory_map=True)
healpix_toUse = [9556, 9557, 9558, 9559, 9560, 9683, 9684, 9685, 9686, 9687, 9688, 9812, 9813, 9814, 9815, 9816, 9940, 9941, 9942, 9943, 9944, 10068, 10069, 10070, 10071, 10072, 10195,10196,10197, 10198, 10199]
cI = cdc2['Mag_true_i_sdss_z0']
print(len(cI))
cZ = cdc2['Mag_true_z_sdss_z0']
c_iz = cI-cZ
keep = (c_iz < 0.5)&(c_iz>-0.18)
cdc2 = cdc2.loc[keep]
cdc2.reset_index(inplace=True, drop=True)

print("Selecting random galaxies")
idx = np.arange(len(cdc2['PZflowredshift']))
rand = np.random.choice(idx, replace=False, size=tot)
cdc2_sel = cdc2.loc[rand]
galaxy_ids = cdc2_sel['galaxy_id']
# print("plotting")
# plt.figure(figsize=[6,5])
# subsel = np.random.choice(np.arange(tot), replace=False, size=10000)
# plt.scatter(cdc2_sel['PZflowredshift'], cdc2_sel['Mag_true_z_sdss_z0'], s=1, alpha=.1)
# plt.xlabel("PZFlow Redshift")
# plt.ylabel("Mag_true_z_sdss_z0")
# plt.savefig("../plots/random_hostlib.png")
# plt.clf()

print("loading full CosmoDC2 catalog")
features = ['mag_true_u_lsst', 'mag_true_g_lsst','mag_true_r_lsst', 
                'mag_true_i_lsst', 'mag_true_z_lsst',
                 'mag_true_Y_lsst','size_true', 'size_minor_true', 
                 'size_disk_true', 'size_minor_disk_true', 'size_bulge_true', 
                 'size_minor_bulge_true','galaxy_id', 'sersic_disk', 'sersic_bulge', 
                 'position_angle_true', 'bulge_to_total_ratio_i', 'totalStarFormationRate', 'stellar_mass', 'ra', 'dec']
cdc2_full = {}
for feature in features:
    cdc2_full[feature] = []

cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image")
filters=[(lambda x: np.isin(x, galaxy_ids), 'galaxy_id')]
print("Number of galaxies: ",len(galaxy_ids))
for gal in cosmo.get_quantities(features, filters=filters, return_iterator=True):
    for feature in features:
        cdc2_full[feature].extend(gal[feature])
cdc2_full = pd.DataFrame(cdc2_full)
print(cdc2_full.head)    
print("Number of found galaxies from full catalog: ", len(cdc2_full))

# #clear cosmo from memory
del cosmo
print("Loading photoz-catalog")

#add photoz_errs from the other catalog
pz_features = ['galaxy_id', 'mag_err_u_photoz', 'mag_err_g_photoz', 'mag_err_r_photoz', 'mag_err_i_photoz', 'mag_err_z_photoz', 'mag_err_y_photoz']
pz = {}
for feature in pz_features:
    pz[feature] = []



# for gal in cosmo_pz.get_quantities(pz_features+['photoz_mask'], filters=filters, return_iterator=True):
#     photoz_mask = gal['photoz_mask']
#     for feature in pz_features:
#         pz[feature].extend(gal[feature][photoz_mask])
cosmo_pz = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_v1")        
for c in range(len(healpix_toUse)):
    healpix = healpix_toUse[c]
    print("Getting Healpix %d from the photo-z add-on catalog"%healpix)
    gal = cosmo_pz.get_quantities(pz_features+['photoz_mask'], native_filters=f"healpix_pixel == {healpix}")
    photoz_mask  = gal['photoz_mask']
    for feature in pz_features[1:]:
        pz[feature].extend(gal[feature][photoz_mask])
    pz['galaxy_id'].extend(gal['galaxy_id'])
pz = pd.DataFrame(pz)
print("number of galaxies from photo-z add on: ",len(pz))
del cosmo_pz

cdc2_mrg = pd.merge(cdc2_sel, cdc2_full, on=['galaxy_id'], how='left')
cdc2_mrg_pz = pd.merge(cdc2_mrg, pz, on=["galaxy_id"], how='left')


cdc2_mrg_pz.to_csv('/global/cscratch1/sd/mlokken/sn_hostenv/rand_hostlib.csv', index=False)
print(len(cdc2_mrg['galaxy_id']))
print(len(cdc2_mrg_pz['galaxy_id']))