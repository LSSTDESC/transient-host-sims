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
from astropy.cosmology import FlatLambdaCDM
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

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

#df_sdss_comb = pd.read_csv("/global/cscratch1/sd/agaglian/DC2full_pzRedshifts_tenHealpix_sdss_updMag.tar.gz")
df_sdss_comb = pd.read_csv("/global/cscratch1/sd/agaglian/DC2full_pzRedshifts_tenHealpix_secondSet_sdss_updMag.tar.gz")

print("Starting the catalog loading!")
#combine with the image catalog and get physical radius values
df_Image = pd.read_csv("/global/cscratch1/sd/agaglian/FullImageMomentsCatalog.tar.gz", usecols=['cosmoDC2_ID', 'RSQ_pixel_gal'])
df_Image['galaxy_id'] = df_Image['cosmoDC2_ID']
print("Length of sdss df is", len(df_sdss_comb))
print(np.sort(df_Image['galaxy_id']))
print(np.sort(df_sdss_comb['galaxy_id']))

print("Length of the DF_Image catalog is", len(df_Image))
del df_Image['cosmoDC2_ID']
print("Merging with the image moments catalog")

df_sdss_wImg = df_Image.merge(df_sdss_comb, on='galaxy_id')
#df_sdss_wImg = df_sdss_comb.merge(df_Image, on='galaxy_id', how='outer')
df_sdss_wImg.dropna(subset=['DC2redshift'], inplace=True)

print(len(df_sdss_wImg))

print("Calculating the size of each galaxy...")

#first take sqrt(RSQ) to get R in px, then convert to arcsec, then convert to physical distance using the following formula:
#(angle in arcsec)/206265 = d/D ---> d = (angle in arcsec * luminosity distance)/206265 (divide by 1.e3 to get kpc)
df_sdss_wImg['R_kpc'] = np.sqrt(df_sdss_wImg['RSQ_pixel_gal'])*0.2 * cosmo.luminosity_distance(df_sdss_wImg['PZflowredshift']).to(u.pc).value/206265./1.e3

# get rid of this column and save the full catalog
del df_sdss_wImg['RSQ_pixel_gal']

print("Done. Saving! Here's a sneak peek at the catalog:")
print(df_sdss_wImg.head())

df_sdss_wImg.to_csv("/global/cscratch1/sd/agaglian/DC2full_pzRedshifts_tenHealpix_sdss_updMag_Rkpc_secondSet.tar.gz",index=False)

end = time.time()

print("Saved successfully.")
print(end - start)
