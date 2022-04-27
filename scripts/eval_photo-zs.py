import corner
import numpy as np
import pandas as pd

import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling, ShiftBounds
from pzflow.distributions import Uniform, Joint, Normal

import qp

import rail
# from rail.creation import Creator, engines
from rail.creation.degradation import LSSTErrorModel
from rail.evaluation import Evaluator

# import jax.numpy as np
# import matplotlib.pyplot as plt
# import pzflow
# from pzflow import Flow
# from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling, ShiftBounds
# from pzflow.distributions import Tdist, Uniform, Joint, Normal
# import pandas as pd
# import GCRCatalogs
# import numpy as onp
import os
import time
# import seaborn as sns


start = time.process_time()

# sharpnesses =[1]
# spl_binses = [16]#, 32, 64]
# n_eps = [30]#[100]#30
prepends = ['SNIa_GHOST', 'SNII_GHOST', 'SNIbc_GHOST', 'UNMATCHED_KN_SHIFT_GHOST_ABS', 'UNMATCHED_COSMODC2_GHOST']
# params = [(sh, sp, n) for sh in sharpnesses for sp in spl_binses for n in n_eps]
# print(len(params))
# sys.exit(0)
idx = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
which_hl = prepends[idx]
hl_path = '/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/'+which_hl+'_PHOTOZ.HOSTLIB'
# skip 26lines
df = pd.read_csv(hl_path, skiprows=18, delimiter=' ', header=0)
nhost = len(df)
print('starting '+which_hl+' with '+str(nhost)+' hosts')
hl_df = df.rename(columns={'ZTRUE':'redshift',
                           'Y_obs':'y', 
                           'r_obs':'r', 
                           'u_obs':'u', 
                           'g_obs':'g', 
                           'z_obs':'z', 
                           'i_obs':'i',})[['redshift', 
                                                  'u', 'g', 'r', 'i', 'z', 'y']]
                            # 'Y_obs_err':'y_err', 
                           # 'r_obs_err':'r_err', 
                           # 'u_obs_err':'u_err', 
                           # 'g_obs_err':'g_err', 
                           # 'z_obs_err':'z_err', 
                           # 'i_obs_err':'i_err'})[['redshift', 
                           #                        'u', 'g', 'r', 'i', 'z', 'y', 
                           #                        'u_err', 'g_err', 'r_err', 'i_err', 'z_err', 'y_err']]
# print(params[idx])

hl_df_colors = hl_df.copy()
quantities = hl_df.columns
for i in range(len(quantities)-2):
    hl_df_colors[quantities[i+1]+'-'+quantities[i+2]] = hl_df[quantities[i+1]] -hl_df[quantities[i+2]]
hl_df_colors = hl_df_colors.drop(columns = quantities)
hl_df_colors['r'] = hl_df['r']# * std + mean
hl_df_colors['redshift'] = hl_df['redshift']
hl_df = hl_df_colors[['redshift', 'u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']]

# nvis = len(hl_df)
# hl_subset = hl_df[:nvis]

flow = Flow(file='../data_files/model_photo-zs_uniform_splbin16_epoch30_flow.pkl')
flow.latent = Uniform((-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5))

zgrid = np.logspace(-3., np.log10(3.), 300)
quants = np.linspace(0., 1., 11)[1:-1]
q50_ind = np.argwhere(quants == 0.5)

# input - df: a Dataframe, chunkSize: the chunk size
# output - a list of DataFrame
# purpose - splits the DataFrame into smaller chunks
def split_dataframe(df, chunk_size = 1000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

hl_subsets = split_dataframe(hl_df)
df_subsets = split_dataframe(df)

for j in range(len(hl_subsets)):
    print(j)
    hl_subset = hl_subsets[j]
    df_subset = df_subsets[j]
    flow_z = flow.posterior(hl_subset[['u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']], column='redshift', grid=zgrid)#, err_samples=err_samples)

    in_pdfs = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=flow_z, check_input=True))
    zquants = in_pdfs.ppf(quants)
    zmeds = np.reshape(zquants[:, q50_ind], zquants.shape[0])
    logp50 = np.empty_like(zmeds)
    for i in range(in_pdfs.npdf):
        logp50[i] = in_pdfs[i].logpdf(zmeds[i])
    out_pdfs = in_pdfs.convert_to(qp.quant_piecewise_gen, quants=quants, check_input=False)

    out_pdfs.set_ancil(dict(GALID=df_subset['GALID'].values, logp50=logp50))
    out_pdfs.write_to('/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/zquants/quants'+str(j)+which_hl+'.fits')

    
print('finished '+which_hl+' in '+str(time.process_time() - start))

# def f(logmass):
#     return 2*logmass - 24



# cosmo=GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")
# get_cols = ['redshift', 'mag_true_u_lsst', 'mag_true_g_lsst', 'mag_true_r_lsst', 'mag_true_i_lsst', 'mag_true_z_lsst', 'mag_true_y_lsst']
# # , 'stellar_mass', 'totalStarFormationRate']
# ### TODO: extract error columns as well

# print("Reading CosmoDC2 small catalog")
# data = cosmo.get_quantities(get_cols)
# print("Catalog read.")
# df = pd.DataFrame(data)

# quantities = ['redshift', 'u', 'g', 'r', 'i', 'z', 'y']#, 'u_err', 'g_err', 'r_err', 'i_err', 'z_err', 'y_err']
# data = df.rename(columns={'mag_true_y_lsst': 'y', 
#                    'mag_true_r_lsst': 'r', 
#                    'mag_true_u_lsst': 'u', 
#                    'mag_true_g_lsst': 'g', 
#                    'mag_true_z_lsst': 'z', 
#                    'mag_true_i_lsst': 'i',})[quantities]
#                            # 'Y_obs_err':'y_err', 
#                            # 'r_obs_err':'r_err', 
#                            # 'u_obs_err':'u_err', 
#                            # 'g_obs_err':'g_err', 
#                            # 'z_obs_err':'z_err', 
#                            # 'i_obs_err':'i_err'})[quantities]
# z_col = 0
# # data['logSFRtot'] = onp.log10(data['totalStarFormationRate'])
# # data['logmass']   = onp.log10(data['stellar_mass'])
# # data.drop(columns=['totalStarFormationRate', 'stellar_mass'], inplace=True)
# ### TODO: make error columns as [colname]_err

# # conditional_columns = data.columns.drop(['redshift'])
# # print("Original conditional columns:", conditional_columns)

# # plt.figure(figsize=(10,7))
# # plt.plot(data['redshift'].sample(n=100000, random_state=1), (data['mag_true_g_lsst'] - data['mag_true_r_lsst']).sample(n=100000, random_state=1), 'o', ms=0.1)
# # plt.xlabel("DC2 Redshift")
# # plt.ylabel(r"$g-r$")
# # plt.savefig("../plots/ogdc2_zvcolor_gr.png")
# # plt.clf()

# # #get rid of that weird clump M* > 10^10 Msol and logSFR = -5
# # data = data[onp.asarray((data['logSFRtot'] - 9) > f(data['logmass']))]

# ### TODO: create colors and one magnitude and save scaling factors
# data_scaled = data.copy()
# for i in range(len(quantities)-2):
#     data_scaled[quantities[i+1]+'-'+quantities[i+2]] = data[quantities[i+1]] - data[quantities[i+2]]
# data_scaled = data_scaled.drop(columns=conditional_columns)
# print(data_scaled.columns)

# # mean = data['r'].mean()
# # std = data['r'].std()
# # print('rmags normed by N(%.3f, %.3f)'%(mean, std))
# # print('TODO: normalize rmag')
# data_scaled['r'] = data['r']#(data['r'] - mean) / std
# ### TODO: may need to rescale the one magnitude
# # conditional_columns = data_scaled.columns.drop(['redshift'])
# # print("Scaled conditional columns:", conditional_columns)
# # print(data_scaled.columns)

# # standard scale the reference magnitude and colors, but keep an copy of the original
# # data_scaled = data.copy()
# # for quality in [
# #     # 'logmass', 'logSFRtot', 
# #                 'mag_true_u_lsst', 'mag_true_g_lsst', 'mag_true_r_lsst', 'mag_true_i_lsst', 'mag_true_z_lsst', 'mag_true_y_lsst']:
# #     data_scaled[quality] = (data[quality]-data[quality].mean())/data[quality].std()
# # means = data['redshift'].mean()
# # stds = data['redshift'].std()
# # means = data_scaled.mean(axis=0).values
# # stds = data_scaled.std(axis=0).values

# # mins = [0, -1, -1, -1, -1, -1, 15]
# # maxs = [3, 5, 5, 5, 5, 5, 35]
# # B = 5

# # # take a subset of 1/500th
# # # have tried 1/1000, 1/100,
# # subset_size = int(len(data_scaled) / 1.e3)
# # data_subset = data_scaled.sample(n=subset_size, random_state=42)
# # # del data_scaled
# # # data = data.sample(n=subset_size, random_state=42) # same sample from unscaled data
# # print('Training on {} CosmoDC2 galaxies.'.format(len(data_subset)))

# # mins = np.array([0])
# # maxs = np.array([data_subset['redshift'].max()+0.1])
# # latent = Uniform((-5, 5))
# # latent = Tdist(input_dim=1)
# # latent = Uniform((0., 3.), (-1., 5.), (-1., 5.), (-1., 5.), (-1., 5.), (-1., 5.), (15, 35))
# # latent = Uniform((-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5))

# ### TODO: vary these between runs
# ### iterate over sharpness = 1, 3, 10, 30
# # sharpness = 5#10
# ### iterate over spl_bins = 2, 4, 8, 16, 32
# # spl_bins = 8#2
# # B = 5 <-- vary this, something to do with spacing of spline bins?
# # spl_bins increase to broaden posteriors

# # bijector = Chain(
# #     # InvSoftplus(z_col, sharpness),
# #     # StandardScaler(means, stds),
# #     ShiftBounds(mins, maxs, B=5),
# #     RollingSplineCoupling(nlayers=7, K=spl_bins),#1, n_conditions=6),
# # )

# # To create the conditional flow, we have to provide
# # 1. The names of the data columns
# # 2. The bijector
# # 3. The names of the conditional columns

# # flow = Flow(
# #     data_columns = data_scaled.columns,#['redshift'],
# #     # conditional_columns = conditional_columns,
# #     bijector = bijector,
# #     latent = latent,          
# # )


# ### TODO: try more epochs
# ### iterate over n_ep = 30, 100, 300, 1000
# # n_ep = 100#30

# ### TODO: retrain with errors
# # losses = flow.train(data_subset, epochs=n_ep, verbose=True)
# # losses = flow.train(data_subset, convolve_errs=True, epochs=n_ep, verbose=True)

# # sns.set_context("talk")
# # plt.plot(losses)
# # plt.xlabel("Epoch")
# # plt.ylabel("Training loss")
# # plt.savefig("../plots/model_photo-zs_uniform_splbin%d_epoch%d_traning_loss.png" % (spl_bins, n_ep))

# # plt.clf()
# # save the results, then apply them with the script apply_pzflow_dc2full.py
# # flow.save('../data_files/model_photo-zs_uniform_splbin%d_epoch%d_flow.pkl' % (spl_bins, n_ep))
# # print("time taken for uniform latent %d spline bins %d epochs training: "%(spl_bins, n_ep)+str(time.process_time() - start))

# # allSamples = []
# # #split into 100 chunks
# # data_list = split_dataframe(data_subset, chunk_size=int(len(data_subset)/100.))

# # for chunk in data_list:
# #     samples = flow.sample(1, conditions=chunk.drop(columns=['redshift']), seed=0)
# #     allSamples.append(samples)
# # samples=pd.concat(allSamples)

# # data_subset['redshift_old'] = data_subset['redshift']
# # data_subset['redshift_new'] = samples['redshift']
# # plt.hist(data_subset['redshift_old'] - data_subset['redshift_new'], bins=30)
# # plt.savefig("../plots/pzflow_zdiff.png")
# # plt.clf()

# # plt.figure(figsize=(10,7))
# # plt.plot(data_subset['redshift_new'], (data['mag_true_g_lsst'] - data['mag_true_r_lsst']), 'o', ms=0.1)
# # plt.xlabel("DC2 Redshift")
# # plt.ylabel(r"$g-r$")
# # plt.savefig("../plots/pzflow_zvcolor_gr.png")
# # plt.clf()

# # plt.figure(figsize=(10,7))
# # plt.plot(data_subset['redshift_new'], (data['mag_true_r_lsst'] - data['mag_true_i_lsst']), 'o', ms=0.1)
# # plt.xlabel("DC2 Redshift")
# # plt.ylabel(r"$r-i$")
# # plt.savefig("../plots/pzflow_zvcolor_ri.png")
# # plt.clf()