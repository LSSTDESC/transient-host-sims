import jax.numpy as np
import matplotlib.pyplot as plt
import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling, ShiftBounds
from pzflow.distributions import Tdist, Uniform, Joint, Normal
import pandas as pd
import GCRCatalogs
import numpy as onp
import os
import time
import seaborn as sns

sharpnesses =[1]
spl_binses = [64]#[16, 32, 64, 128]
n_eps = [100]
params = [(sh, sp, n) for sh in sharpnesses for sp in spl_binses for n in n_eps]
print(len(params))
# sys.exit(0)
idx = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
sharpness, spl_bins, n_ep = params[idx]

print(params[idx])

# def f(logmass):
#     return 2*logmass - 24

# input - df: a Dataframe, chunkSize: the chunk size
# output - a list of DataFrame
# purpose - splits the DataFrame into smaller chunks
def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

cosmo=GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")
get_cols = ['redshift', 'mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_Y_lsst']#, 
            # 'magerr_u_lsst', 'magerr_g_lsst', 'magerr_r_lsst', 'magerr_i_lsst', 'magerr_z_lsst', 'magerr_Y_lsst']
# , 'stellar_mass', 'totalStarFormationRate']
### TODO: extract error columns as well

print(cosmo.list_all_quantities())

print("Reading CosmoDC2 small catalog")
data = cosmo.get_quantities(get_cols)
print("Catalog read.")
df = pd.DataFrame(data)

allcols = ['redshift', 'u', 'g', 'r', 'i', 'z', 'y']#, 'u_err', 'g_err', 'r_err', 'i_err', 'z_err', 'y_err']
data = df.rename(columns={'mag_Y_lsst': 'y', 
                   'mag_r_lsst': 'r', 
                   'mag_u_lsst': 'u', 
                   'mag_g_lsst': 'g', 
                   'mag_z_lsst': 'z', 
                   'mag_i_lsst': 'i',})[allcols]
                   # 'magerr_Y_lsst': 'y_err', 
                   # 'magerr_r_lsst': 'r_err', 
                   # 'magerr_u_lsst': 'u_err', 
                   # 'magerr_g_lsst': 'g_err', 
                   # 'magerr_z_lsst': 'z_err', 
                   # 'magerr_i_lsst': 'i_err',})[allcols]
z_col = 0
# data['logSFRtot'] = onp.log10(data['totalStarFormationRate'])
# data['logmass']   = onp.log10(data['stellar_mass'])
# data.drop(columns=['totalStarFormationRate', 'stellar_mass'], inplace=True)
### TODO: make error columns as [colname]_err

conditional_columns = data.columns.drop(['redshift'])
print("Original conditional columns:", conditional_columns)

# plt.figure(figsize=(10,7))
# plt.plot(data['redshift'].sample(n=100000, random_state=1), (data['mag_true_g_lsst'] - data['mag_true_r_lsst']).sample(n=100000, random_state=1), 'o', ms=0.1)
# plt.xlabel("DC2 Redshift")
# plt.ylabel(r"$g-r$")
# plt.savefig("../plots/ogdc2_zvcolor_gr.png")
# plt.clf()

# #get rid of that weird clump M* > 10^10 Msol and logSFR = -5
# data = data[onp.asarray((data['logSFRtot'] - 9) > f(data['logmass']))]

### TODO: create colors and one magnitude and save scaling factors
data_scaled = data.copy()
quantities = allcols#[:-6]
for i in range(len(quantities)-2):
    data_scaled[quantities[i+1]+'-'+quantities[i+2]] = data[quantities[i+1]] - data[quantities[i+2]]
    # data_scaled[quantities[i+1]+'-'+quantities[i+2]+'_err'] = np.sqrt(data[quantities[6+i+1]]**2 + data[quantities[6+i+2]]**2)
data_scaled = data_scaled.drop(columns=conditional_columns)
print(data_scaled.columns)

# mean = data['r'].mean()
# std = data['r'].std()
# print('rmags normed by N(%.3f, %.3f)'%(mean, std))
# print('TODO: normalize rmag')
data_scaled['r'] = data['r']#(data['r'] - mean) / std
### TODO: may need to rescale the one magnitude
# conditional_columns = data_scaled.columns.drop(['redshift'])
# print("Scaled conditional columns:", conditional_columns)
print(data_scaled.columns)

# standard scale the reference magnitude and colors, but keep an copy of the original
# data_scaled = data.copy()
# for quality in [
#     # 'logmass', 'logSFRtot', 
#                 'mag_true_u_lsst', 'mag_true_g_lsst', 'mag_true_r_lsst', 'mag_true_i_lsst', 'mag_true_z_lsst', 'mag_true_y_lsst']:
#     data_scaled[quality] = (data[quality]-data[quality].mean())/data[quality].std()
# means = data['redshift'].mean()
# stds = data['redshift'].std()
means = data_scaled.mean(axis=0).values
stds = data_scaled.std(axis=0).values

mins = [0, -1, -1, -1, -1, -1, 15]
maxs = [3, 5, 5, 5, 5, 5, 35]
# B = 5

# take a subset of 1/500th
# have tried 1/1000, 1/100,
subset_size = int(len(data_scaled) / 1.e3)
data_subset = data_scaled.sample(n=subset_size, random_state=42)
# del data_scaled
# data = data.sample(n=subset_size, random_state=42) # same sample from unscaled data
print('Training on {} CosmoDC2 galaxies.'.format(len(data_subset)))

# mins = np.array([0])
# maxs = np.array([data_subset['redshift'].max()+0.1])
# latent = Uniform((-5, 5))
# latent = Tdist(input_dim=1)
# latent = Uniform((0., 3.), (-1., 5.), (-1., 5.), (-1., 5.), (-1., 5.), (-1., 5.), (15, 35))
latent = Uniform((-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5))

### TODO: vary these between runs
### iterate over sharpness = 1, 3, 10, 30
# sharpness = 5#10
### iterate over spl_bins = 2, 4, 8, 16, 32
# spl_bins = 8#2
# B = 5 <-- vary this, something to do with spacing of spline bins?
# spl_bins increase to broaden posteriors

bijector = Chain(
    # InvSoftplus(z_col, sharpness),
    # StandardScaler(means, stds),
    ShiftBounds(mins, maxs, B=5),
    RollingSplineCoupling(nlayers=7, K=spl_bins),#1, n_conditions=6),
)

# To create the conditional flow, we have to provide
# 1. The names of the data columns
# 2. The bijector
# 3. The names of the conditional columns

flow = Flow(
    data_columns = data_scaled.columns,#['redshift'],
    # conditional_columns = conditional_columns,
    bijector = bijector,
    latent = latent,          
)


### TODO: try more epochs
### iterate over n_ep = 30, 100, 300, 1000
# n_ep = 100#30

start = time.perf_counter()
print(start)

### TODO: retrain with errors
# losses = flow.train(data_subset, epochs=n_ep, verbose=True)
losses = flow.train(data_subset, convolve_errs=True, epochs=n_ep, verbose=True)

sns.set_context("talk")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.savefig("../plots/model_photo-zs_errs_splbin%d_epoch%d_traning_loss.png" % (spl_bins, n_ep))
plt.clf()

# save the results, then apply them with the script apply_pzflow_dc2full.py
flow.save('../data_files/model_photo-zs_errs_splbin%d_epoch%d_flow.pkl' % (spl_bins, n_ep))

end = time.perf_counter()
print(end)
print("time taken for %d spline bins %d epochs with errors training: "%(spl_bins, n_ep)+str(end - start))

# allSamples = []
# #split into 100 chunks
# data_list = split_dataframe(data_subset, chunk_size=int(len(data_subset)/100.))

# for chunk in data_list:
#     samples = flow.sample(1, conditions=chunk.drop(columns=['redshift']), seed=0)
#     allSamples.append(samples)
# samples=pd.concat(allSamples)

# data_subset['redshift_old'] = data_subset['redshift']
# data_subset['redshift_new'] = samples['redshift']
# plt.hist(data_subset['redshift_old'] - data_subset['redshift_new'], bins=30)
# plt.savefig("../plots/pzflow_zdiff.png")
# plt.clf()

# plt.figure(figsize=(10,7))
# plt.plot(data_subset['redshift_new'], (data['mag_true_g_lsst'] - data['mag_true_r_lsst']), 'o', ms=0.1)
# plt.xlabel("DC2 Redshift")
# plt.ylabel(r"$g-r$")
# plt.savefig("../plots/pzflow_zvcolor_gr.png")
# plt.clf()

# plt.figure(figsize=(10,7))
# plt.plot(data_subset['redshift_new'], (data['mag_true_r_lsst'] - data['mag_true_i_lsst']), 'o', ms=0.1)
# plt.xlabel("DC2 Redshift")
# plt.ylabel(r"$r-i$")
# plt.savefig("../plots/pzflow_zvcolor_ri.png")
# plt.clf()