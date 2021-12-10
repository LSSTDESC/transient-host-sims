import jax.numpy as np
import matplotlib.pyplot as plt
import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling, ShiftBounds
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
quantities = ['redshift', 'mag_true_u_lsst', 'mag_true_g_lsst', 'mag_true_r_lsst', 'mag_true_i_lsst', 'mag_true_z_lsst', 'mag_true_y_lsst', 'stellar_mass', 'totalStarFormationRate']

print("Reading CosmoDC2 small catalog")
data = cosmo.get_quantities(quantities)
print("Catalog read.")
data = pd.DataFrame(data)
data['logSFRtot'] = onp.log10(data['totalStarFormationRate'])
data['logmass']   = onp.log10(data['stellar_mass'])
data.drop(columns=['totalStarFormationRate', 'stellar_mass'], inplace=True)

plt.figure(figsize=(10,7))
plt.plot(data['redshift'].sample(n=100000, random_state=1), (data['mag_true_g_lsst'] - data['mag_true_r_lsst']).sample(n=100000, random_state=1), 'o', ms=0.1)
plt.xlabel("DC2 Redshift")
plt.ylabel(r"$g-r$")
plt.savefig("../plots/ogdc2_zvcolor_gr.png")
plt.clf()

#get rid of that weird clump M* > 10^10 Msol and logSFR = -5
data = data[onp.asarray((data['logSFRtot'] - 9) > f(data['logmass']))]


# standard scale the reference magnitude and colors, but keep an copy of the original
data_scaled = data.copy()
for quality in ['logmass', 'logSFRtot', 'mag_true_u_lsst', 'mag_true_g_lsst', 'mag_true_r_lsst', 'mag_true_i_lsst', 'mag_true_z_lsst', 'mag_true_y_lsst']:
    data_scaled[quality] = (data[quality]-data[quality].mean())/data[quality].std()
    
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
subset_size = int(len(data_scaled)/1.e3)
data_subset = data_scaled.sample(n=subset_size, random_state=1)
del data_scaled
data = data.sample(n=subset_size, random_state=1) # same sample from unscaled data
print('Training on {} CosmoDC2 galaxies.'.format(len(data_subset)))
conditional_columns = data_subset.columns.drop(['redshift'])
print("Conditional columns:", conditional_columns)
mins = np.array([0])
maxs = np.array([data_subset['redshift'].max()+0.1])
latent = Uniform((-5,5))

bijector = Chain(
    ShiftBounds(mins, maxs, 5),
    RollingSplineCoupling(nlayers=1, n_conditions=len(conditional_columns), K=2),
)

# To create the conditional flow, we have to provide
# 1. The names of the data columns
# 2. The bijector
# 3. The names of the conditional columns

flow = Flow(
    data_columns = ['redshift'],
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
plt.clf()
# save the results, then apply them with the script apply_pzflow_dc2full.py
flow.save('../data_files/pzflow_dc2small_nofilter_div1000.pkl')
print("CPU time taken: ", time.process_time() - start)

allSamples = []
#split into 100 chunks
data_list = split_dataframe(data_subset, chunk_size=int(len(data_subset)/100.))

for chunk in data_list:
    samples = flow.sample(1, conditions=chunk.drop(columns=['redshift']), seed=0)
    allSamples.append(samples)
samples=pd.concat(allSamples)

data_subset['redshift_old'] = data_subset['redshift']
data_subset['redshift_new'] = samples['redshift']
plt.hist(data_subset['redshift_old'] - data_subset['redshift_new'], bins=30)
plt.savefig("../plots/pzflow_zdiff.png")
plt.clf()

plt.figure(figsize=(10,7))
plt.plot(data_subset['redshift_new'], (data['mag_true_g_lsst'] - data['mag_true_r_lsst']), 'o', ms=0.1)
plt.xlabel("DC2 Redshift")
plt.ylabel(r"$g-r$")
plt.savefig("../plots/pzflow_zvcolor_gr.png")
plt.clf()

plt.figure(figsize=(10,7))
plt.plot(data_subset['redshift_new'], (data['mag_true_r_lsst'] - data['mag_true_i_lsst']), 'o', ms=0.1)
plt.xlabel("DC2 Redshift")
plt.ylabel(r"$r-i$")
plt.savefig("../plots/pzflow_zvcolor_ri.png")
plt.clf()