import jax.numpy as np
import matplotlib.pyplot as plt
import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling
from pzflow.distributions import Uniform, Joint, Normal
import pandas as pd
import GCRCatalogs
import numpy as onp
import time
import seaborn as sns
start = time.process_time()

def f(logmass):
    return 2*logmass - 24

cosmo=GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")
quantities = ['redshift', 'mag_true_g_lsst', 'mag_true_r_lsst', 'mag_true_i_lsst', 'mag_true_z_lsst', 'Mag_true_g_sdss_z0', 'Mag_true_r_sdss_z0', 'Mag_true_i_sdss_z0', 'Mag_true_z_sdss_z0', 
 'stellar_mass', 'totalStarFormationRate']

print("Reading CosmoDC2 small catalog")
gal = cosmo.get_quantities(quantities)
print("Catalog read.")
data_unscaled = {}
data_unscaled['logSFRtot'] = np.log10(gal['totalStarFormationRate'])
data_unscaled['redshift']  = gal['redshift']
data_unscaled['logmass'] = np.log10(gal['stellar_mass'])
data_unscaled['Mag_true_g_sdss_z0']  = gal['Mag_true_g_sdss_z0']
data_unscaled['Mag_true_r_sdss_z0']  = gal['Mag_true_r_sdss_z0']
data_unscaled['Mag_true_i_sdss_z0']  = gal['Mag_true_i_sdss_z0']
data_unscaled['Mag_true_z_sdss_z0']  = gal['Mag_true_z_sdss_z0']
# data_unscaled['morphology/totalEllipticity'] = gal['morphology/totalEllipticity']

data_unscaled = pd.DataFrame(data_unscaled)

#get rid of that weird clump M* > 10^10 Msol and logSFR = -5
data_unscaled = data_unscaled[onp.asarray((data_unscaled['logSFRtot'] - 9) > f(data_unscaled['logmass']))]
data_unscaled['g-r'] = data_unscaled['Mag_true_g_sdss_z0'] - data_unscaled['Mag_true_r_sdss_z0']
data_unscaled['r-i'] = data_unscaled['Mag_true_r_sdss_z0'] - data_unscaled['Mag_true_i_sdss_z0']
data_unscaled['i-z'] = data_unscaled['Mag_true_i_sdss_z0'] - data_unscaled['Mag_true_z_sdss_z0']

# standard scale the reference magnitude and colors
data = data_unscaled.copy()
for quality in ['logmass', 'Mag_true_g_sdss_z0', 'Mag_true_r_sdss_z0', 'Mag_true_i_sdss_z0', 'Mag_true_z_sdss_z0', 'g-r', 'r-i', 'i-z']:
    data[quality] = (data_unscaled[quality]-data_unscaled[quality].mean())/data_unscaled[quality].std()
    
# input - df: a Dataframe, chunkSize: the chunk size
# output - a list of DataFrame
# purpose - splits the DataFrame into smaller chunks
def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

# take a subset of 1/500th
# have tried 1/1000, 1/100,
data_subset = data.sample(n=int(len(data)/5.e2))
print('Training on {} CosmoDC2 galaxies.'.format(len(data_subset)))
conditional_columns = data_subset.columns.drop(['redshift', 'logSFRtot'])
latent = Joint(Uniform((-3,3)), Normal(1))
means = np.array([data_subset['redshift'].mean(), data_subset['logSFRtot'].mean()])
stds = np.array([data_subset['redshift'].std(), data_subset['logSFRtot'].std()]) #switched 1/6 to 1/5.9
bijector = Chain(
    StandardScaler(means, stds),
    RollingSplineCoupling(nlayers=2, n_conditions=len(conditional_columns)),
)
# To create the conditional flow, we have to provide
# 1. The names of the data columns
# 2. The bijector
# 3. The names of the conditional columns

flow = Flow(
    data_columns = ("redshift", "logSFRtot"),
    conditional_columns = conditional_columns,
    bijector = bijector,
    latent = latent,          
)

losses = flow.train(data_subset, epochs=30, verbose=True)

sns.set_context("talk")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.savefig("../plots/pzflow_traning_loss.png")
# save the results, then apply them with the script apply_pzflow_dc2full.py
flow.save('../data_files/pzflow_dc2small_nofilter_div500')
print("CPU time taken: ", time.process_time() - start)