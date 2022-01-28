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

df_comb = pd.read_csv("/global/cscratch1/sd/agaglian/DC2full_pzRedshifts_tenHealpix_Final.tar.gz",usecols=['galaxy_id', 'PZflowredshift', 'DC2redshift'])
#galid = df_comb['galaxy_id'].values
#del df_comb
cosmo=GCRCatalogs.load_catalog("cosmoDC2_v1.1.4")
#gal = cosmo.get_quantities(quantities, native_filters=f"healpix_pixel == {healpix}")
features = ['galaxy_id', 'Mag_true_g_sdss_z0', 'Mag_true_r_sdss_z0', 'Mag_true_i_sdss_z0', 'Mag_true_z_sdss_z0']
c = 0

gal = cosmo.get_quantities(features, filters=[(lambda x: np.in1d(x, df_comb['galaxy_id']), 'galaxy_id')])#, return_iterator=True)
df_sdss = pd.DataFrame(gal)
df_sdss_comb = df_sdss.merge(df_comb, on='galaxy_id')

#shift to apparent with the old mags, then back to absolute with the new mags 
#df_sdss_comb

df_sdss_comb.to_csv("/global/cscratch1/sd/agaglian/DC2full_pzRedshifts_tenHealpix_Final_sdss.tar.gz",index=False)

end = time.time()

print(end - start)
