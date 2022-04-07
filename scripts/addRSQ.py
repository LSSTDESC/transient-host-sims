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
modes = np.array(['SN Ia', 'SN II', 'SN Ibc'])

#modes = np.array(['rand'])
#neigh_dict = {'rand':0}

neigh_dict = {'SN Ia':2211, 'SN II':6984, 'SN Ibc':39473}

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

print("Starting the catalog loading!")
#combine with the image catalog and get RSQ
df_Image = pd.read_csv("/global/cscratch1/sd/agaglian/FullImageMomentsCatalog.tar.gz", usecols=['cosmoDC2_ID', 'RSQ_pixel_gal'])
df_Image['galaxy_id'] = df_Image['cosmoDC2_ID']
print(np.sort(df_Image['galaxy_id']))

print("Length of the DF_Image catalog is", len(df_Image))
del df_Image['cosmoDC2_ID']
print("Merging with the image moments catalog")

for mode in modes:
    print("Running for %s" % mode)
    n_neigh = neigh_dict[mode]

    if ' ' in mode:
        modestr = mode.replace(' ','')
    else:
        modestr = mode

    cdc2_nbrs = pd.read_csv('/global/cscratch1/sd/agaglian/matchedSamples_0407/cdc2_matched_ghost_%s_z3_unq_zwgt_5pct_k%i_SFRMsol.tar.gz'.format(modestr, n_neigh))
    cdc2_wRSQ = df_Image.merge(cdc2_nbrs, on='galaxy_id')
    
    cdc2_wRSQ.rename(columns={"RSQ_pixel_gal":"RSQ"}, inplace=True)
    
    print("Done. Here's a sneak peek at the catalog:")
    print(cdc2_wRSQ.head())

    cdc2_wRSQ.to_csv('/global/cscratch1/sd/agaglian/matchedSamples_0407/cdc2_matched_ghost_%s_z3_unq_zwgt_5pct_k%i_SFRMsolRSQ.tar.gz'.format(modestr, n_neigh), index=False)

end = time.time()

print("Saved successfully.")
print(end - start)
