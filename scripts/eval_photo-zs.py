import datetime
import math
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time

import pzflow
from pzflow import Flow
from pzflow.bijectors import Chain, ColorTransform, InvSoftplus, StandardScaler, RollingSplineCoupling, ShiftBounds
from pzflow.distributions import Uniform, Joint, Normal

import qp

glob_path = '/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/zquants/'

print(datetime.datetime.now())
sys.stdout.flush()

start = time.process_time()
# SNII_GHOST.HOSTLIB.gz: 2449022
# SNIa_GHOST.HOSTLIB.gz: 2141291
# SNIbc_GHOST.HOSTLIB.gz: 3354171
# UNMATCHED_COSMODC2_GHOST.HOSTLIB.gz: 1907393
# UNMATCHED_KN_SHIFT_GHOST.HOSTLIB.gz: 1907394

hl_heads = {'SNIa': (10, 2141270), # 2.75 hours
            'SNII': (19, 2449001), # 3.25 hours
            'SNIbc_Pt1': (19, 1224500), # 4.25 hours
            'SNIbc_Pt2': (19, 1224500),##3354171), # 4.25 hours
            'UNMATCHED_KN_SHIFT': (19, 1907372), #2.5 hours
            'UNMATCHED_COSMODC2': (18, 1907373)} # 2.5 hours


idx = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
chunks = 64#int(os.getenv('SLURM_ARRAY_TASK_COUNT', '32'))#sys.argv[2]

# idx from job array, one core with 128GB memory
batch_size = 50#100#don't change me! (determines memory needed)

which_hl = sys.argv[1]
hl_path = '/global/cfs/cdirs/lsst/groups/TD/SN/SNANA/SURVEYS/LSST/ROOT/PLASTICC_DEV/HOSTLIB/magerr/'+which_hl+'_GHOST.HOSTLIB.gz'
hl_head = int(subprocess.check_output(f"zcat {hl_path} | cat -n | sed -n '/VARNAMES/ {{ p; q }}'  | awk '{{print $1-1}}'", shell=True))

nhost = hl_heads[which_hl][1]
print('starting '+which_hl+' of '+str(nhost)+' hosts')
sys.stdout.flush()

batch_factor = math.ceil(nhost / (chunks * batch_size))
chunk_size = batch_size * batch_factor#math.ceil(nhost / chunks)
for i, chunk in enumerate(pd.read_csv(hl_path, skiprows=hl_head, delimiter=' ', header=0, chunksize=chunk_size)):
    if i == idx:
        df_subset = chunk
        break
print('batching number '+str(idx)+' of size '+str(len(df_subset)))
sys.stdout.flush()

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

# convert magnitudes to colors
hl_df_colors = hl_df.copy()[['redshift', 'r', 'r_err']]
for i in range(len(quantities)-2-6):
    hl_df_colors[quantities[i+1]+'-'+quantities[i+2]] = hl_df[quantities[i+1]] - hl_df[quantities[i+2]]
    hl_df_colors[quantities[i+1]+'-'+quantities[i+2]+'_err'] = np.sqrt(hl_df[quantities[6+i+1]]**2 + hl_df[quantities[6+i+2]]**2)
      
hl_df = hl_df_colors[['redshift', 'u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r', 'u-g_err', 'g-r_err', 'r-i_err', 'i-z_err', 'z-y_err', 'r_err']][:nhost]

flow = Flow(file='../data_files/model_photo-zs_uniform_splbin64_epoch100_flow.pkl')# this path will not change any time soon
flow.latent = Uniform((-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5))

zgrid = np.logspace(-2.5, np.log10(3.), 250)
quants = np.linspace(0., 1., 11)
quants[0] = 0.001
quants[-1] = 0.999

# relic of changing reading of data to not load the whole hostlib at once
hl_subset = hl_df

# evaluate the PDFs on fine grid
print('running external chunk '+str(idx)+' with '+str(len(hl_subset))+' hosts in internal chunks of '+str(batch_size))
sys.stdout.flush()
flow_z = flow.posterior(hl_subset,#[['u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'r']], 
                            column='redshift', grid=zgrid, err_samples=10, batch_size=min(batch_size, len(hl_subset)))

print('evaluated batch '+str(idx)+' out of '+str(math.ceil(nhost / (batch_size * batch_factor)))+' of external chunk size '+str(len(hl_subset))+' in internal chunks of '+str(batch_size)+' for '+which_hl+' in '+str(time.process_time() - start))
sys.stdout.flush()

# compress PDFs and isolate quantiles, IQR, p(median)
in_pdfs = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=flow_z, check_input=True))
iqr = in_pdfs.ppf(0.75) - in_pdfs.ppf(0.25)
p50 = in_pdfs.pdf(in_pdfs.median())
out_pdfs = in_pdfs.convert_to(qp.quant_piecewise_gen, quants=quants, check_input=False)

print('compressed batch '+str(idx)+' out of '+str(math.ceil(nhost / (batch_size * batch_factor)))+' of external chunk size '+str(len(hl_subset))+' in internal chunks of '+str(batch_size)+' for '+which_hl+' in '+str(time.process_time() - start))
sys.stdout.flush()

# attach to hostlib IDs and save
out_pdfs.set_ancil(dict(GALID=df_subset['GALID'].values,#[idx * batch_size * batch_factor : min((idx+1) * batch_size * batch_factor, nhost)], 
                        p50=p50, iqr=iqr))
out_name = 'pz'+which_hl+'batch'+str(idx)+'_'+str(batch_size * batch_factor)+'chunks'+str(batch_size)
out_pdfs.write_to(glob_path+out_name+'.fits')
    
print('saved batch '+str(idx)+' out of '+str(math.ceil(nhost / (batch_size * batch_factor)))+' of external chunk size '+str(len(hl_subset))+' in internal chunks of '+str(batch_size)+' for '+which_hl+' in '+str(time.process_time() - start))
sys.stdout.flush()

print(datetime.datetime.now())
sys.stdout.flush()
