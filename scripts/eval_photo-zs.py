# import contextlib
# from contextlib import redirect_stdout
import datetime
import math
import numpy as np
import os
import pandas as pd
import sys
import time

import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling, ShiftBounds
from pzflow.distributions import Uniform, Joint, Normal

import qp

glob_path = '/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/zquants/'

# with open(glob_path+'log.txt', 'a') as f:
#     with redirect_stdout(f):

print(datetime.datetime.now())

start = time.process_time()
# SNII_GHOST.HOSTLIB.gz: 2449022
# SNIa_GHOST.HOSTLIB.gz: 2141291
# SNIbc_GHOST.HOSTLIB.gz: 3354171
# UNMATCHED_COSMODC2_GHOST.HOSTLIB.gz: 1907393
# UNMATCHED_KN_SHIFT_GHOST.HOSTLIB.gz: 1907394

hl_heads = {'SNIa': (19, 2141291), # 2.75 hours
            'SNII': (19, 2449022), # 3.25 hours
            'SNIbc': (19, 3354171), # 4.25 hours
            'UNMATCHED_KN_SHIFT': (19, 1907394), #2.5 hours
            'UNMATCHED_COSMODC2': (18, 1907393)} # 2.5 hours

idx = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
# idx from job array, one core with 128GB memory
batch_size = 100#don't change me!
# batch_factor = 100

which_hl = 'UNMATCHED_KN_SHIFT'
# which_hl = list(hl_heads.keys())[idx]
print(which_hl)
hl_path = '/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/'+which_hl+'_GHOST.HOSTLIB.gz'
# hl_head = int(os.system(f"zcat {hl_path} | cat -n | sed -n '/VARNAMES/ {{ p; q }}'  | awk '{{print $1-1}}'"))
df = pd.read_csv(hl_path, skiprows=hl_heads[which_hl][0], delimiter=' ', header=1)

nhost = len(df)
print('starting '+which_hl+' of '+str(nhost)+' hosts')

# nhost / 32 = batch_size * batch_factor
batch_factor = math.ceil(nhost / (32 * batch_size))

df_subset = df[idx * batch_size * batch_factor: min((idx+1) * batch_size * batch_factor, nhost)]#hl_subsets[j]
print('batching number '+str(idx)+' of size '+str(len(df_subset)))

hl_df = df_subset.rename(columns={'ZTRUE':'redshift',
                           'Y_obs':'y', 
                           'r_obs':'r', 
                           'u_obs':'u', 
                           'g_obs':'g', 
                           'z_obs':'z', 
                           'i_obs':'i',#})[['redshift','u', 'g', 'r', 'i', 'z', 'y']]
                            'Y_obs_err':'y_err', 
                           'r_obs_err':'r_err', 
                           'u_obs_err':'u_err', 
                           'g_obs_err':'g_err', 
                           'z_obs_err':'z_err', 
                           'i_obs_err':'i_err'})[['redshift', 
                                                  'u', 'g', 'r', 'i', 'z', 'y', 
                                                  'u_err', 'g_err', 'r_err', 'i_err', 'z_err', 'y_err']]

quantities = hl_df.columns
print(quantities)

hl_df_colors = hl_df.copy()[['redshift', 'r', 'r_err']]
for i in range(len(quantities)-2-6):
    hl_df_colors[quantities[i+1]+'-'+quantities[i+2]] = hl_df[quantities[i+1]] - hl_df[quantities[i+2]]
    hl_df_colors[quantities[i+1]+'-'+quantities[i+2]+'_err'] = np.sqrt(hl_df[quantities[6+i+1]]**2 + hl_df[quantities[6+i+2]]**2)
print(hl_df_colors.columns)
      
hl_df = hl_df_colors[['redshift', 'u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r', 'u-g_err', 'g-r_err', 'r-i_err', 'i-z_err', 'z-y_err', 'r_err']][:nhost]

flow = Flow(file='../data_files/model_photo-zs_uniform_splbin64_epoch100_flow.pkl')
# this path will not change any time soon
flow.latent = Uniform((-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5))

zgrid = np.logspace(-2.5, np.log10(3.), 250)
quants = np.linspace(0., 1., 11)
quants[0] = 0.001
quants[-1] = 0.999

# print(quants)
# q50_ind = np.argwhere(quants == 0.5)

# # input - df: a Dataframe, chunkSize: the chunk size
# # output - a list of DataFrame
# # purpose - splits the DataFrame into smaller chunks
# # manual breakdown, better for continuing after interruption
# def split_dataframe(df, chunk_size = batch_size):#2**idx):
#     chunks = list()
#     num_chunks = len(df) // chunk_size + 1
#     for i in range(num_chunks):
#         chunks.append(df[i*chunk_size:(i+1)*chunk_size])
#     print('anticipate '+str(len(chunks))+' files')
#     return chunks
# # # 8*10000 in 30 minutes on debug queue, out of 191
# hl_subsets = split_dataframe(hl_df)
# df_subsets = split_dataframe(df)

# j = idx
# # for j in [idx]:#range(len(hl_subsets)):
# hl_subset = hl_df[j * batch_size * batch_factor: min((j+1) * batch_size * batch_factor, nhost)]#hl_subsets[j]
# df_subset = df[j * batch_size * batch_factor: min((j+1) * batch_size * batch_factor, nhost)]#df_subsets[j]
hl_subset = hl_df

print('running external chunk '+str(idx)+' with '+str(len(hl_subset))+' hosts in internal chunks of '+str(batch_size))
flow_z = flow.posterior(hl_subset,#[['u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']], 
                            column='redshift', grid=zgrid, err_samples=10, batch_size=min(batch_size, len(hl_subset)))

print('evaluated batch '+str(idx)+' out of '+str(math.ceil(nhost / (batch_size * batch_factor)))+' of external chunk size '+str(len(hl_subset))+' in internal chunks of '+str(batch_size)+' for '+which_hl+' in '+str(time.process_time() - start))

in_pdfs = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=flow_z, check_input=True))
    # p50 = in_pdfs.pdf(in_pdf.median())
iqr = in_pdfs.ppf(0.75) - in_pdfs.ppf(0.25)
p50 = in_pdfs.pdf(in_pdfs.median())
out_pdfs = in_pdfs.convert_to(qp.quant_piecewise_gen, quants=quants, check_input=False)

print('compressed batch '+str(idx)+' out of '+str(math.ceil(nhost / (batch_size * batch_factor)))+' of external chunk size '+str(len(hl_subset))+' in internal chunks of '+str(batch_size)+' for '+which_hl+' in '+str(time.process_time() - start))

# # zmeds = out_pdfs.objdata()['locs'][:,q50_ind].reshape(in_pdfs.npdf)

#     # def f(i):
#     #     return in_pdfs[i].pdf(zmeds[i])

#     # with Pool(multiprocessing.cpu_count()) as p:
#     #     p50 = p.map(f, range(len(hl_subset)))
#     # zquants = in_pdfs.ppf(quants)
#     # zmeds = np.reshape(zquants[:, q50_ind], zquants.shape[0])
#     # p50 = np.empty_like(zmeds)
#     # for i in range(in_pdfs.npdf):
#     #     p50[i] = in_pdfs[i].pdf(zmeds[i])
#     print(which_hl+' part '+str(j)+' completed')
#     out_pdfs.set_ancil(dict(GALID=df_subset['GALID'].values, p50=p50, iqr=iqr))
#     out_pdfs.write_to('/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/zquants/'+which_hl+'test'+str(j)+'errs.fits')
    
# batch_size = 1000#2**idx
# flow_z = flow.posterior(hl_df,#[['u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']], 
#                             column='redshift', grid=zgrid, err_samples=10, batch_size=batch_size)
# etime = time.process_time() - start
# print(which_hl+' evaluated in '+str(etime))
# in_pdfs = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=flow_z, check_input=True))
# iqr = in_pdfs.ppf(0.75) - in_pdfs.ppf(0.25)
# p50 = in_pdfs.pdf(in_pdfs.median())
# out_pdfs = in_pdfs.convert_to(qp.quant_piecewise_gen, quants=quants, check_input=True)
# ctime = time.process_time() - start
# print(which_hl+' compressed in '+str(ctime))

out_pdfs.set_ancil(dict(GALID=df['GALID'].values[idx * batch_size * batch_factor : min((idx+1) * batch_size * batch_factor, nhost)], p50=p50, iqr=iqr))
out_name = 'pz'+which_hl+'batch'+str(idx)+'_'+str(batch_size * batch_factor)+'chunks'+str(batch_size)
out_pdfs.write_to(glob_path+out_name+'.fits')
    
print('saved batch '+str(idx)+' out of '+str(math.ceil(nhost / (batch_size * batch_factor)))+' of external chunk size '+str(len(hl_subset))+' in internal chunks of '+str(batch_size)+' for '+which_hl+' in '+str(time.process_time() - start))

print(datetime.datetime.now())
