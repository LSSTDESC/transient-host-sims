full = True
if full:
    tot = 3000000
else:
    tot = 5000

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

# read in file of CosmoDC2 galaxies, with PZFlow SFR and redshifts, limited to abs r-band magnitude < -15
# and -0.18 < i-z < 0.5
if full:
    cdc2 = pd.read_csv("/global/cscratch1/sd/mlokken/sn_hostenv/DC2full_pzRedshifts_SFR_39iter.csv")
else:
    cdc2 = pd.read_csv("/global/cscratch1/sd/mlokken/sn_hostenv/DC2_pzRedshifts_SFR_RMag_lt_neg15.csv")
cI = cdc2['Mag_true_i_sdss_z0']
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
plt.figure(figsize=[6,5])
subsel = np.random.choice(np.arange(tot), replace=False, size=10000)
plt.scatter(cdc2_sel['PZflowredshift'], cdc2_sel['Mag_true_z_sdss_z0'], s=1, alpha=.1)
plt.xlabel("PZFlow Redshift")
plt.ylabel("Mag_true_z_sdss_z0")
plt.savefig("../plots/random_hostlib.png")
plt.clf()

print("loading full CosmoDC2 catalog")
quantities = ['mag_true_u_lsst', 'mag_true_g_lsst','mag_true_r_lsst', 
                'mag_true_i_lsst', 'mag_true_z_lsst',
                 'mag_true_Y_lsst','size_true', 'size_minor_true', 
                 'size_disk_true', 'size_minor_disk_true', 'size_bulge_true', 
                 'size_minor_bulge_true','galaxy_id', 'sersic_disk', 'sersic_bulge', 
                 'position_angle_true', 'ra', 'dec']
dct = {'mag_true_u_lsst':[], 'mag_true_g_lsst':[],'mag_true_r_lsst':[], 
                'mag_true_i_lsst':[], 'mag_true_z_lsst':[],
                 'mag_true_Y_lsst':[],'size_true':[], 'size_minor_true':[], 
                 'size_disk_true':[], 'size_minor_disk_true':[], 'size_bulge_true':[], 
                 'size_minor_bulge_true':[],'galaxy_id':[], 'sersic_disk':[], 'sersic_bulge':[], 
                 'position_angle_true':[], 'ra':[], 'dec':[]}
if full:
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image")
    c=0
    for gal in cosmo.get_quantities(quantities, return_iterator=True):
        if c<39:
            print("Reading healpixel {:d}".format(c+1))
            for string in quantities:
                dct[string].extend(gal[string])
        c+=1
    cdc2_true = pd.DataFrame(dct)
    print(cdc2_true)
else:
    cosmo = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")
    filters=[(lambda x: np.isin(x, galaxy_ids), 'galaxy_id')]
    cdc2_true = cosmo.get_quantities(quantities, filters=filters)

# #clear cosmo from memory
del cosmo
print("Loading photoz-catalog")

#add photoz_errs from the other catalog
pz_quantities = ['galaxy_id', 'mag_err_u_photoz', 'mag_err_g_photoz', 'mag_err_r_photoz', 'mag_err_i_photoz', 'mag_err_z_photoz', 'mag_err_y_photoz', 'photoz_mask']
if full:
    dct = {'galaxy_id':[], 'mag_err_u_photoz':[], 'mag_err_g_photoz':[], 'mag_err_r_photoz':[], 'mag_err_i_photoz':[], 'mag_err_z_photoz':[], 'mag_err_y_photoz':[]}
    cosmo_pz = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_v1")
    c=0
    for gal in cosmo_pz.get_quantities(pz_quantities, return_iterator=True):
        if c<39:
            print("Reading healpixel {:d}".format(c+1))
            photoz_mask  = gal['photoz_mask']
            dct['galaxy_id'].extend(gal['galaxy_id'])
            dct['mag_err_u_photoz'].extend(gal['mag_err_u_photoz'][photoz_mask])
            dct['mag_err_g_photoz'].extend(gal['mag_err_g_photoz'][photoz_mask])
            dct['mag_err_r_photoz'].extend(gal['mag_err_r_photoz'][photoz_mask])
            dct['mag_err_i_photoz'].extend(gal['mag_err_i_photoz'][photoz_mask])
            dct['mag_err_z_photoz'].extend(gal['mag_err_z_photoz'][photoz_mask])
            dct['mag_err_y_photoz'].extend(gal['mag_err_y_photoz'][photoz_mask])
        c+=1
    cdc2_pz = pd.DataFrame(dct)
else:
    cosmo_pz = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small_with_photozs_v1")
    cdc2_pz = cosmo_pz.get_quantities(pz_quantities)
    photoz_mask  = cdc2_pz['photoz_mask']
    pz_galaxy_id = cdc2_pz['galaxy_id']
    pz_magerr_u  = cdc2_pz['mag_err_u_photoz'][photoz_mask]
    pz_magerr_g  = cdc2_pz['mag_err_g_photoz'][photoz_mask]
    pz_magerr_r  = cdc2_pz['mag_err_r_photoz'][photoz_mask]
    pz_magerr_i  = cdc2_pz['mag_err_i_photoz'][photoz_mask]
    pz_magerr_z  = cdc2_pz['mag_err_z_photoz'][photoz_mask]
    pz_magerr_y  = cdc2_pz['mag_err_y_photoz'][photoz_mask]

    cdc2_pz = pd.DataFrame({'galaxy_id':pz_galaxy_id, 'mag_err_u_photoz':pz_magerr_u, 'mag_err_g_photoz':pz_magerr_g, 'mag_err_r_photoz':pz_magerr_r, 'mag_err_i_photoz':pz_magerr_i, 'mag_err_z_photoz':pz_magerr_z, 'mag_err_y_photoz':pz_magerr_y})
#clear cosmo_pz from memory
del cosmo_pz

cdc2_true = pd.DataFrame(cdc2_true)
cdc2_pz  = pd.DataFrame(cdc2_pz)
cdc2_mrg = pd.merge(cdc2_sel, cdc2_true, on=['galaxy_id'], how='left')
cdc2_mrg_pz = pd.merge(cdc2_mrg, cdc2_pz, on=["galaxy_id"], how='left')
cdc2_mrg_pz.to_csv('/global/cscratch1/sd/mlokken/sn_hostenv/rand_hostlib.csv', index=False)
print(len(cdc2_mrg['galaxy_id']))
print(len(cdc2_mrg_pz['galaxy_id']))