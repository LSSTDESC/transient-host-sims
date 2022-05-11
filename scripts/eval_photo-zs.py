import corner
import numpy as np
import os
import pandas as pd
import time

import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling, ShiftBounds
from pzflow.distributions import Uniform, Joint, Normal

import qp

import rail
# from rail.creation import Creator, engines
from rail.creation.degradation import LSSTErrorModel
from rail.evaluation import Evaluator

start = time.process_time()

hl_heads = {'SNIa': 18,
            'SNII': 18, 
            'SNIbc': 18, 
            'UNMATCHED_KN_SHIFT': 18,
            'UNMATCHED_COSMODC2': 17}

idx = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))

which_hl = list(hl_heads.keys())[idx]
print(which_hl)
hl_path = '/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/'+which_hl+'_GHOST.HOSTLIB.gz'
df = pd.read_csv(hl_path, skiprows=hl_heads[which_hl], delimiter=' ', header=0)

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

hl_df_colors = hl_df.copy()
quantities = hl_df.columns
for i in range(len(quantities)-2):
    hl_df_colors[quantities[i+1]+'-'+quantities[i+2]] = hl_df[quantities[i+1]] - hl_df[quantities[i+2]]
hl_df_colors = hl_df_colors.drop(columns = quantities)
hl_df_colors['r'] = hl_df['r']# * std + mean
hl_df_colors['redshift'] = hl_df['redshift']
hl_df = hl_df_colors[['redshift', 'u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']]

flow = Flow(file='../data_files/model_photo-zs_uniform_splbin64_epoch100_flow.pkl')
flow.latent = Uniform((-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5))

zgrid = np.logspace(-3., np.log10(3.), 300)
quants = np.linspace(0., 1., 11)
quants[0] += 0.001
quants[-1] -= 0.001
print(quants)
# q50_ind = np.argwhere(quants == 0.5)

# input - df: a Dataframe, chunkSize: the chunk size
# output - a list of DataFrame
# purpose - splits the DataFrame into smaller chunks

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    print('anticipate '+str(len(chunks))+' files')
    return chunks

hl_subsets = split_dataframe(hl_df)
df_subsets = split_dataframe(df)

for j in range(len(hl_subsets)):
    print(j)
    hl_subset = hl_subsets[j]
    df_subset = df_subsets[j]
    flow_z = flow.posterior(hl_subset[['u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']], column='redshift', grid=zgrid)#, err_samples=err_samples)

    in_pdfs = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=flow_z, check_input=True))
    # zquants = in_pdfs.ppf(quants)
    # zmeds = np.reshape(zquants[:, q50_ind], zquants.shape[0])
    # p50 = np.empty_like(zmeds)
    # for i in range(in_pdfs.npdf):
    #     p50[i] = in_pdfs[i].pdf(zmeds[i])
    out_pdfs = in_pdfs.convert_to(qp.quant_piecewise_gen, quants=quants, check_input=False)

    out_pdfs.set_ancil(dict(GALID=df_subset['GALID'].values))#, p50=p50))
    out_pdfs.write_to('/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/zquants/quants'+str(j)+which_hl+'.fits')

    
print('finished '+which_hl+' in '+str(time.process_time() - start))