# Code for applying a pre-trained flow to a subset of the full CosmoDC2 catalog
# Smooths out unphysical discreteness in SFR and redshift
# If you don't have jax and pzflow installed, you need to run the following in a terminal: 
# pip install pzflow
# pip install jax

import jax.numpy as np
import matplotlib.pyplot as plt
import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling
from pzflow.examples import galaxy_data
from pzflow.distributions import Uniform, Joint, Normal
import pandas as pd
import GCRCatalogs
import time
import numpy as onp
import seaborn as sns
sns.set_context("talk")
start = time.process_time()
def f(logmass):
    return 2*logmass - 24
print(GCRCatalogs.__version__)
def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

plotting = True
# load in CosmoDC2 catalog
cosmo=GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image")
quantities = ['galaxy_id', 'redshift', 'Mag_true_g_sdss_z0', 'Mag_true_r_sdss_z0', 'Mag_true_i_sdss_z0', 'Mag_true_z_sdss_z0', 'stellar_mass', 'totalStarFormationRate']
filters = ['mag_true_r_lsst<28']
# 
logsfrtot = []
redshift  = []
logmass   = []
g = []
r = []
i = []
z = []
galid = []
#ellip = []

subselect = False
for healpix in cosmo.available_healpix_pixels:
    print(healpix)
    gal = cosmo.get_quantities(quantities, filters=filters, native_filters=f"healpix_pixel == {healpix}")

    if c<1:
#    if c<39:
        logsfrtot.append(np.log10(gal['totalStarFormationRate']))
        redshift.append(gal['redshift'])
        print(max(gal['redshift']))
        # sample = onp.random.choice(len(gal['redshift']), size=100)
        # plt.plot(gal['ra'][sample], gal['dec'][sample], 'ro')
        # plt.show()
        logmass.append(np.log10(gal['stellar_mass']))
        g.append(gal['Mag_true_g_sdss_z0'])
        r.append(gal['Mag_true_r_sdss_z0'])
        i.append(gal['Mag_true_i_sdss_z0'])
        z.append(gal['Mag_true_z_sdss_z0'])
        galid.append(gal['galaxy_id'])
        #ellip.append(gal['morphology/totalEllipticity'])
        print("Reading healpixel {:d}".format(healpix+1))
        c+=1
    else:
        break

data_unscaled = {}
data_unscaled['redshift']  = np.concatenate(redshift)
print("Catalog length", len(data_unscaled['redshift']))
data_unscaled['logSFRtot'] = np.concatenate(logsfrtot)
data_unscaled['logmass'] = np.concatenate(logmass)
data_unscaled['Mag_true_g_sdss_z0']  = np.concatenate(g)
data_unscaled['Mag_true_r_sdss_z0']  = np.concatenate(r)
data_unscaled['Mag_true_i_sdss_z0']  = np.concatenate(i)
data_unscaled['Mag_true_z_sdss_z0']  = np.concatenate(z)
data_unscaled['galaxy_id'] = np.concatenate(galid)
# data_unscaled['morphology/totalEllipticity'] = np.concatenate(ellip)
data_unscaled = pd.DataFrame(data_unscaled)
print("Cutting SFR/logmass clump")
# get rid of a weird clump at M* > 10^10 Msol and logSFR = -5
data_unscaled = data_unscaled.iloc[np.asarray((data_unscaled['logSFRtot'] - 9) > f(data_unscaled['logmass']))]
print("Getting colors.")
# get colors
data_unscaled['g-r'] = data_unscaled['Mag_true_g_sdss_z0'] - data_unscaled['Mag_true_r_sdss_z0']
data_unscaled['r-i'] = data_unscaled['Mag_true_r_sdss_z0'] - data_unscaled['Mag_true_i_sdss_z0']
data_unscaled['i-z'] = data_unscaled['Mag_true_i_sdss_z0'] - data_unscaled['Mag_true_z_sdss_z0']
# standard scale the reference magnitude and colors
print("Standard scaling.")
data = data_unscaled.copy()
for quality in ['logmass', 'Mag_true_g_sdss_z0', 'Mag_true_r_sdss_z0', 'Mag_true_i_sdss_z0', 'Mag_true_z_sdss_z0', 'g-r', 'r-i', 'i-z']:
    data[quality] = (data_unscaled[quality]-data_unscaled[quality].mean())/data_unscaled[quality].std()

# reduce to only the relevant quantities
data = data[['redshift', 'logSFRtot', 'logmass', 'Mag_true_g_sdss_z0', 'Mag_true_r_sdss_z0',
             'Mag_true_i_sdss_z0', 'Mag_true_z_sdss_z0', 'g-r', 'r-i', 'i-z']]

# split into 100 chunks
data_list = split_dataframe(data, chunk_size=int(len(data)/1.e2))
print("Loading flow.")
# load pre-trained flow
flow = Flow(file ='../data_files/pzflow_dc2small_nofilter_div500.pkl')
allSamples = []
print("Predicting SFR and redshift.")
for chunk in data_list:
    samples = flow.sample(1, conditions=chunk, seed=0)
    allSamples.append(samples)
samples=pd.concat(allSamples)

if plotting:
    print("Plotting.")
    # plots to check that it worked
    # subsamp = onp.random.choice(np.arange(len(samples['logmass'])), size=1000, replace=False)
    subsamp = onp.arange(len(samples['logmass']))
    samples_small = samples.iloc[subsamp]
    data_full_small = data_unscaled.iloc[subsamp]

    plt.figure(figsize=(6,5))
    plt.plot(data_full_small['redshift'], data_full_small['logSFRtot'], 'o', ms=0.05, label='DC2', c='k')
    plt.plot(samples_small['redshift'], samples_small['logSFRtot'], 'o', ms=0.05, label='pzflow')
    plt.xlim((0, 1.0))
    plt.ylim((4, 10.0))
    plt.xlabel("Redshift")
    plt.ylabel("logSFRtot")
    plt.legend(fontsize=16, loc='upper right')
    plt.savefig("../plots/pzflow/SFR_pzflow_DC2_vs_Redshift.png",dpi=300, bbox_inches='tight')
    plt.clf()
    plt.figure(figsize=(6,5))
    plt.hist(samples_small['logSFRtot'].values - data_full_small['logSFRtot'].values, bins=500, density=True)
    plt.axvline(x=0, c='r')
    plt.xlim((-3, 3))
    plt.xlabel(r"$logSFRtot_{pred} - logSFRtot_{DC2}$")
    plt.savefig("../plots/pzflow/SFRdiff_pzflow_DC2.png",dpi=300, bbox_inches='tight')
    plt.clf()
    plt.figure(figsize=(6,5))
    plt.hist(samples_small['redshift'].values - data_full_small['redshift'].values, bins=500, density=True)
    plt.xlim((-0.7, 0.7))
    plt.xlabel(r"$z_{pred} - z_{DC2}$")
    plt.savefig("../plots/pzflow/zdiff_pzflow_DC2.png",dpi=300, bbox_inches='tight')
    plt.clf()
    plt.figure(figsize=(6,5))
    plt.plot(data_full_small['logSFRtot'], data_full_small['g-r'], 'o', ms=0.05, label='DC2', c='k')
    plt.plot(samples_small['logSFRtot'], data_full_small['g-r'], 'o', ms=0.05, label='pzflow')
    plt.xlabel("log(SFR)")
    plt.ylabel("g-r")
    plt.legend()
    plt.savefig("../plots/pzflow/gminusr_v_sfr_pzflow_DC2.png",dpi=300, bbox_inches='tight')
    plt.clf()
    plt.figure(figsize=(10,7))
    plt.plot(data_full_small['redshift'], data_full_small['g-r'], 'o', ms=0.05, label='DC2', c='k')
    plt.plot(samples_small['redshift'], data_full_small['g-r'], 'o', ms=0.05, label='pzflow')
    plt.legend()
    plt.xlabel("Redshift")
    plt.ylabel("g-r")
    plt.savefig("../plots/pzflow/gminusr_v_redshift_pzflow_DC2.png",dpi=300, bbox_inches='tight')
    plt.clf()

# rename the original DC2 redshift and SFR, and add the PZFlow redshift and SFR
data_unscaled['DC2redshift'] = data_unscaled['redshift'].values
data_unscaled['PZflowredshift'] = samples['redshift'].values
data_unscaled['DC2SFRtot'] = 10**(data_unscaled['logSFRtot'].values)
data_unscaled['PZflowSFRtot'] = 10**(samples['logSFRtot'].values)
# convert stellar mass back to non-log
data_unscaled['stellar_mass'] = 10**(data_unscaled['logmass'].values)
# delete the extraneous columns
del data_unscaled['redshift']
del data_unscaled['logSFRtot']
del data_unscaled['logmass']

# quality cuts
data_unscaled = data_unscaled[(data_unscaled['PZflowredshift']>=0.0) & (data_unscaled['PZflowredshift']<=1.0)]
data_unscaled = data_unscaled[(data_unscaled['PZflowSFRtot'] > 1.e1) & (data_unscaled['PZflowSFRtot'] < 1.e10)]
for band in ['Mag_true_g_sdss_z0', 'Mag_true_r_sdss_z0', 'Mag_true_i_sdss_z0']:
    m, b = onp.polyfit(data_unscaled[band], data_unscaled['Mag_true_z_sdss_z0'], 1)
    samples_outliers = data_unscaled[np.abs((m*data_unscaled[band] + b) - data_unscaled['Mag_true_z_sdss_z0']) > 1.3]
    samples_inliers = data_unscaled[np.abs((m*data_unscaled[band] + b) - data_unscaled['Mag_true_z_sdss_z0']) < 1.3]
    data_unscaled =  data_unscaled[np.abs((m*data_unscaled[band] + b) - data_unscaled['Mag_true_z_sdss_z0']) < 1.3]
# save
print("Saving.")
data_unscaled.to_csv("/global/cscratch1/sd/mlokken/sn_hostenv/DC2full_pzRedshifts_SFR_1itertest_div500.csv",index=False)

print("CPU time taken: ", time.process_time() - start)