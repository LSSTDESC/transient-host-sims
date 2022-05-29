import corner
# import multiprocessing
# from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import time

import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling, ShiftBounds
from pzflow.distributions import Uniform, Joint, Normal

import qp

# import rail
# # from rail.creation import Creator, engines
# from rail.creation.degradation import LSSTErrorModel
# from rail.evaluation import Evaluator

start = time.process_time()

hl_heads = {'SNIa': 19,
            'SNII': 19, 
            'SNIbc': 19, 
            'UNMATCHED_KN_SHIFT': 19,
            'UNMATCHED_COSMODC2': 18}

idx = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))

# which_hl = 'UNMATCHED_KN_SHIFT'
which_hl = list(hl_heads.keys())[idx]
print(which_hl)
hl_path = '/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/'+which_hl+'_GHOST.HOSTLIB.gz'
# hl_head = int(os.system(f"zcat {hl_path} | cat -n | sed -n '/VARNAMES/ {{ p; q }}'  | awk '{{print $1-1}}'"))
df = pd.read_csv(hl_path, skiprows=hl_heads[which_hl], delimiter=' ', header=1)

nhost = len(df)
# nhost = 2 ** (idx + 1)
print('starting '+which_hl+' with '+str(nhost)+' hosts')

hl_df = df.rename(columns={'ZTRUE':'redshift',
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

# hl_df_colors = hl_df.copy()
quantities = hl_df.columns
print(quantities)

hl_df_colors = hl_df.copy()[['redshift', 'r', 'r_err']]
for i in range(len(quantities)-2-6):
    # print(quantities[i+1])
    hl_df_colors[quantities[i+1]+'-'+quantities[i+2]] = hl_df[quantities[i+1]] - hl_df[quantities[i+2]]
    # print(hl_df_colors.columns)
    hl_df_colors[quantities[i+1]+'-'+quantities[i+2]+'_err'] = np.sqrt(hl_df[quantities[6+i+1]]**2 + hl_df[quantities[6+i+2]]**2)
    # print(hl_df_colors.columns)
# hl_df_colors = hl_df_colors.drop(columns = quantities)
# print(hl_df_colors.columns)
# hl_df_colors['r'] = hl_df['r']# * std + mean
# hl_df_colors['r_err'] = hl_df['r_err']
# print(hl_df_colors.columns)
# hl_df_colors['redshift'] = hl_df['redshift']
print(hl_df_colors.columns)
hl_df = hl_df_colors[['redshift', 'u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r', 'u-g_err', 'g-r_err', 'r-i_err', 'i-z_err', 'z-y_err', 'r_err']][:nhost]

flow = Flow(file='../data_files/model_photo-zs_uniform_splbin64_epoch100_flow.pkl')
# this path will not change any time soon
flow.latent = Uniform((-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5))

zgrid = np.logspace(-3., np.log10(3.), 100)
quants = np.linspace(0., 1., 11)
# print(quants)
q50_ind = np.argwhere(quants == 0.5)

# # input - df: a Dataframe, chunkSize: the chunk size
# # output - a list of DataFrame
# # purpose - splits the DataFrame into smaller chunks

# # manual breakdown, better for continuing after interruption
# def split_dataframe(df, chunk_size = 1000):#2**idx):
#     chunks = list()
#     num_chunks = len(df) // chunk_size + 1
#     for i in range(num_chunks):
#         chunks.append(df[i*chunk_size:(i+1)*chunk_size])
#     print('anticipate '+str(len(chunks))+' files')
#     return chunks

# # 8*10000 in 30 minutes on debug queue, out of 191
# hl_subsets = split_dataframe(hl_df)
# df_subsets = split_dataframe(df)

# for j in range(len(hl_subsets)):
#     hl_subset = hl_subsets[j]
#     df_subset = df_subsets[j]
#     flow_z = flow.posterior(hl_subset,#[['u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']], 
#                             column='redshift', grid=zgrid, err_samples=10)# batch_size=10000)

#     in_pdfs = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=flow_z, check_input=True))
#     # p50 = in_pdfs.pdf(in_pdf.median())
#     iqr = in_pdfs.ppf(0.75) - in_pdfs.ppf(0.25)
#     p50 = in_pdfs.pdf(in_pdfs.median())
#     out_pdfs = in_pdfs.convert_to(qp.quant_piecewise_gen, quants=quants, check_input=False)

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
    
batch_size = 2**idx
flow_z = flow.posterior(hl_df,#[['u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']], 
                            column='redshift', grid=zgrid, err_samples=10, batch_size=batch_size)
etime = time.process_time() - start
print(which_hl+' evaluated in '+str(etime))
in_pdfs = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=flow_z, check_input=True))
iqr = in_pdfs.ppf(0.75) - in_pdfs.ppf(0.25)
p50 = in_pdfs.pdf(in_pdfs.median())
out_pdfs = in_pdfs.convert_to(qp.quant_piecewise_gen, quants=quants, check_input=True)
ctime = time.process_time() - start
print(which_hl+' compressed in '+str(ctime))
out_pdfs.set_ancil(dict(GALID=df['GALID'].values[:nhost], p50=p50, iqr=iqr))
out_pdfs.write_to('/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/zquants/test'+which_hl+'batched'+str(batch_size)+'lowres.fits')
    
print('finished '+str(nhost)+' of '+which_hl+' in '+str(time.process_time() - start))