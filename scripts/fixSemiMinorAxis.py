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

# Bug fixes, shamelessly stolen (with permission)
# from Eve Kovacs via https://github.com/LSSTDESC/gcr-catalogs/blob/ellipticity_bug_fix/GCRCatalogs/cosmodc2.py
def fix_axis_ratio(q_bad):
    # back out incorrect computation of q using Johnsonb function
    e_jb = np.sqrt((1 - q_bad**2)/(1 + q_bad**2))
    q_new = np.sqrt((1 - e_jb)/(1 + e_jb)) # use correct relationship to compute q from e_jb 
    return q_new

def fix_size_minor(a, b):
    q_bad = b/a
    q_new = fix_axis_ratio(q_bad)
    return q_new*a

start = time.time()
modes = np.array(['SN Ia', 'SN II', 'SN Ibc'])

#modes = np.array(['rand'])
#neigh_dict = {'rand':0}

neigh_dict = {'SN Ia':2211, 'SN II':6984, 'SN Ibc':39473}

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

for mode in modes:
    print("Running for %s" % mode)
    n_neigh = neigh_dict[mode]

    if ' ' in mode:
        modestr = mode.replace(' ','')
    else:
        modestr = mode

    cdc2_wRSQ = pd.read_csv('/global/cscratch1/sd/agaglian/matchedSamples_0407/cdc2_matched_ghost_%s_z3_unq_zwgt_5pct_k%i_SFRMsolRSQ.tar.gz' % (modestr, n_neigh))

    cdc2_wRSQ['size_minor_bulge_true_fixed'] = fix_size_minor(cdc2_wRSQ['size_bulge_true'], cdc2_wRSQ['size_minor_bulge_true'])
    cdc2_wRSQ['size_minor_disk_true_fixed'] = fix_size_minor(cdc2_wRSQ['size_disk_true'], cdc2_wRSQ['size_minor_disk_true'])
    cdc2_wRSQ['size_minor_true_fixed'] = fix_size_minor(cdc2_wRSQ['size_true'], cdc2_wRSQ['size_minor_true'])
                           
    del cdc2_wRSQ['size_minor_bulge_true']
    del cdc2_wRSQ['size_minor_disk_true']
    del cdc2_wRSQ['size_minor_true']
                            
    cdc2_wRSQ.rename(columns={"size_minor_bulge_true_fixed":"size_minor_bulge_true", 
                              "size_minor_disk_true_fixed":"size_minor_disk_true", 
                              "size_minor_true_fixed":"size_minor_true"},inplace=True)
                            
    print("Done fixing minor columns.")
                            
    cdc2_wRSQ.to_csv('/global/cscratch1/sd/agaglian/matchedSamples_0407/cdc2_matched_ghost_%s_z3_unq_zwgt_5pct_k%i_SFRMsolRSQ_fixedMinor.tar.gz' % (modestr, n_neigh), index=False)

end = time.time()

print("Saved successfully.")
print(end - start)
