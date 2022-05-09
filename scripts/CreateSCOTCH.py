import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy import table
import random
import glob
import astropy.units as u
from astropy.coordinates import SkyCoord
import h5py
#a very useful tool (this is in python3.9 but better not to risk it)
def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


GID = np.arange(1, 5000001)
GID = ["G%07i" %x for x in GID]
#randomly shuffle
random.shuffle(GID)
GID_df = pd.DataFrame({'GID':GID})
GID_df.to_csv("/global/cscratch1/sd/mlokken/sn_hostenv/SCOTCH_GIDs.tar.gz",index=False)

bands = 'ugrizY'
NBAND = len(bands)

master_host_dict = {}  
master_photometry_dict = {}  
    
transDir = "/global/cscratch1/sd/kessler/SNANA_LSST_SIM/SCOTCH_Z3/"
prefix = "MLAG_GP_SCOTCH_FINAL"

# get all models
models = [x.split("/")[-1] for x in glob.glob(transDir + "*")]
models = [remove_prefix(x, prefix + "_") for x in models]
print(models)
cadence_dict = {}

for model in models:
    if model == 'AGN01':
        cadence_dict[model] = '0.1d'
    elif model in ['KN_K17', 'KN_B19']:
        cadence_dict[model] = 'var_kn'
    elif model in ['TDE']:
        cadence_dict[model] = 'var'
    else:
        cadence_dict[model] = '2.0d'


# classes = [x.split("+")[0] for x in models]

# for i in np.arange(len(classes)):
#     if 'Templates' in classes[i]:
#         classes[i] = classes[i][:-10]
#     elif 'SALT2' in classes[i]:
#         classes[i] = classes[i][:-6]
#     elif 'MOSFIT' in classes[i]:
#         classes[i] = classes[i][:-7]
#     elif ('B19' in classes[i]) or ('K17' in classes[i]) or ('NMF' in classes[i]):
#         classes[i] = classes[i][:-4]
#     elif 'AGN' in classes[i]:
#         classes[i] = 'AGN'
# unq_classes = np.unique(classes)

# do this mapping manually to get it specifically how we want
class_model_dict = {'AGN':['AGN01', 'AGN20'],
'KN':['KN_K17', 'KN_B19'],
'SLSN-I':['SLSN-I'],
'SNII':['SNII-Templates', 'SNII+HostXT_V19', 'SNII-NMF', 'SNIIn+HostXT_V19', 'SNIIn-MOSFIT'],
'SNIIb':['SNIIb+HostXT_V19'],
'SNIa':['SNIa-SALT2', 'SNIa-91bg', 'SNIax'],
'SNIb':['SNIb-Templates', 'SNIb+HostXT_V19'],
'SNIc':['SNIc+HostXT_V19', 'SNIc-Templates', 'SNIcBL+HostXT_V19'],
'TDE':['TDE']}


TID_counter = 1
GID_counter = 1

#fiducial skycoord at 0,0
c1 = SkyCoord(0.0*u.deg, 0.0*u.deg, frame='icrs')

#dictionary between our calculated GIDs and dc2 galaxy_id
dc2_map = {}

#initialize the mapping between GID and cosmoDC2 ID
for tempGID in GID:
    dc2_map[tempGID] = []

# make the HDF5 file
f    = h5py.File("/global/cscratch1/sd/mlokken/sn_hostenv/scotch10k_z3_fixID.hdf5", "a")
transients = f.require_group("TransientTable")
hosts      = f.require_group("HostTable")

# for cl in class_model_dict: # loop over classes / groupings in the hdf5 file
for cl in {'SNII':['SNII-Templates', 'SNII+HostXT_V19', 'SNII-NMF', 'SNIIn+HostXT_V19', 'SNIIn-MOSFIT']}: # loop over classes / groupings in the hdf5 file
    print(hosts.keys())
    if cl in hosts.keys():
        continue
    print("Starting on class %s"%cl)

    for i in np.arange(len(bands)):
        master_host_dict['mag_%s'%bands[i]] = []
        master_host_dict['magerr_%s'%bands[i]] = []
    

    master_host_dict['GID'] = []
    #master_host_dict['ZPHOT'] = []
    #master_host_dict['ZPHOT_QP0'] = []
    #master_host_dict['ZPHOT_QP1'] = []
    #master_host_dict['ZPHOT_QP2'] = []
    #master_host_dict['ZPHOT_QP3'] = []
    #master_host_dict['ZPHOT_QP4'] = []
    #master_host_dict['ZPHOT_QP5'] = []
    #master_host_dict['ZPHOT_QP6'] = []
    #master_host_dict['ZPHOT_QP7'] = []
    #master_host_dict['ZPHOT_QP8'] = []
    #master_host_dict['ZPHOT_QP9'] = []
    master_host_dict['logMstar'] = []
    master_host_dict['logSFR'] = []
    master_host_dict['T'] = []
    master_host_dict['e'] = []
    master_host_dict['e0'] = []
    master_host_dict['e1'] = []
    master_host_dict['a0'] = []
    master_host_dict['b0'] = []
    master_host_dict['n0'] = []
    master_host_dict['w0'] = []
    master_host_dict['a1'] = []
    master_host_dict['b1'] = []
    master_host_dict['w1'] = []
    master_host_dict['n1'] = []
    master_host_dict['dc2ID'] = []
    master_host_dict['TID'] = []
    master_host_dict['a_rot'] = []
    master_host_dict['model']  = []
    master_host_dict['z']  = []

    models   = class_model_dict[cl]
    tranclas = transients.require_group("%s"%cl)
    hostclas = hosts.require_group("%s"%cl)
    for model in models:
        # set / reset the datasets
        master_photometry_dict['TID'] = []
        master_photometry_dict['MJD'] = []
        for i in np.arange(len(bands)):
            master_photometry_dict['mag_%s'%bands[i]] = []
        master_photometry_dict['model'] = []
        master_photometry_dict['z'] = []
        master_photometry_dict['ra_off'] = []
        master_photometry_dict['dec_off'] = []
        master_photometry_dict['sep'] = []
        master_photometry_dict['GID'] = []
        master_photometry_dict['cadence']  = []

        fn = transDir + prefix + "_" +  model + "/"
        print(fn)
        #glob automatically sorts by fn, so we're guaranteed that these correspond to each other! 
        #thank goodness.
        headfiles = sorted(glob.glob(fn + "*HEAD.FITS.gz"))
        photfiles = sorted(glob.glob(fn + "*PHOT.FITS.gz"))

        print("Starting on model %s"%model)
    
        # for i in np.arange(len(headfiles)): # loop over files in model directory
        for i in np.arange(2): # test mode: only do 2 files
            tempHead_fn = headfiles[i]
            tempPhot_fn = photfiles[i]

            tempPhot = table.Table.read(tempPhot_fn, format='fits').to_pandas()
            tempHead = table.Table.read(tempHead_fn, format='fits').to_pandas()

            tempPhot.replace(-999, np.nan, inplace=True)
            tempHead.replace(-999, np.nan, inplace=True)

            tempPhot = tempPhot[tempPhot['MJD'] > 0]
            NEPOCH = len(np.unique(tempPhot['MJD']))

            N = (NBAND*(NEPOCH))
            for k in np.arange(int(len(tempPhot)/N)): # loop over transients in one file
                if TID_counter%5000 == 0:
                    print("Stored info for transient %i" %(TID_counter))
                oneLC = tempPhot.iloc[N*k:N*(k+1)]
                dc2ID = tempHead['SIM_HOSTLIB_GALID'].values[k] #host galaxy cosmoDC2       

                z = tempHead['SIM_REDSHIFT_CMB'].values[k]
                repGID = GID[TID_counter-1]
                #repTID = TID  

                #flipping ra and dec to get the offset of the transient from the galaxy, not the offset of the galaxy from the transient
                #...it is this easy, right?
                repRAoff = tempHead['HOSTGAL_RA'].values[k]*3600#] #in arcsec of RA
                repDECoff = tempHead['HOSTGAL_DEC'].values[k]*3600#] #in arcsec of Dec

                #append to master dict
                TID = "T%.7i"%TID_counter

                master_host_dict['dc2ID'].append(dc2ID)
                master_host_dict['GID'].append(repGID) 
                master_host_dict['logMstar'].append(tempHead['SIM_HOSTLIB(LOGMASS_TRUE)'].values[k]) 
                master_host_dict['logSFR'].append(tempHead['SIM_HOSTLIB(LOG_SFR)'].values[k]) 
                master_host_dict['T'].append(tempHead['SIM_HOSTLIB(SQRADIUS)'].values[k]) 
                master_host_dict['a0'].append(tempHead['SIM_HOSTLIB(a0_Sersic)'].values[k]) 
                master_host_dict['b0'].append(tempHead['SIM_HOSTLIB(b0_Sersic)'].values[k]) 
                master_host_dict['n0'].append(tempHead['SIM_HOSTLIB(n0_Sersic)'].values[k]) 
                master_host_dict['w0'].append(tempHead['SIM_HOSTLIB(w0_Sersic)'].values[k]) 
                master_host_dict['a1'].append(tempHead['SIM_HOSTLIB(a1_Sersic)'].values[k]) 
                master_host_dict['b1'].append(tempHead['SIM_HOSTLIB(b1_Sersic)'].values[k]) 
                master_host_dict['n1'].append(tempHead['SIM_HOSTLIB(n1_Sersic)'].values[k]) 
                master_host_dict['w1'].append(tempHead['SIM_HOSTLIB(w1_Sersic)'].values[k])
                master_host_dict['a_rot'].append(tempHead['SIM_HOSTLIB(a_rot)'].values[k]) 
                master_host_dict['TID'].append(TID) 
                master_host_dict['z'].append(z)
                master_photometry_dict['TID'].append(TID)
                master_photometry_dict['cadence'].append(cadence_dict[model])
                master_photometry_dict['model'].append(model)
                master_photometry_dict['GID'].append(repGID)
                master_photometry_dict['z'].append(z)
                master_photometry_dict['ra_off'].append(repRAoff)
                master_photometry_dict['dec_off'].append(repDECoff)
                master_photometry_dict['MJD'].append(np.unique([oneLC['MJD'].values]))

                #calculate separation 
                c2 = SkyCoord(tempHead['HOSTGAL_RA'].values[k]*u.deg, tempHead['HOSTGAL_DEC'].values[k]*u.deg, frame='icrs')
                sep = c1.separation(c2)
                master_photometry_dict['sep'].append(sep.arcsec)
                #calculate corrected ellipticity
                q_disk  = tempHead['SIM_HOSTLIB(b0_Sersic)'].values[k]/tempHead['SIM_HOSTLIB(a0_Sersic)'].values[k]
                q_bulge = tempHead['SIM_HOSTLIB(b1_Sersic)'].values[k]/tempHead['SIM_HOSTLIB(a1_Sersic)'].values[k]
                e_disk  = (1-q_disk)/(1+q_disk)
                e_bulge = (1-q_bulge)/(1+q_bulge)
                w_disk  = tempHead['SIM_HOSTLIB(w0_Sersic)'].values[k]
                w_bulge = tempHead['SIM_HOSTLIB(w1_Sersic)'].values[k]
                e_tot   = w_disk*e_disk + w_bulge*e_bulge # weights add to 1 so no normalization required
                master_host_dict['e'].append(e_tot) 
                master_host_dict['e0'].append(e_disk)
                master_host_dict['e1'].append(e_bulge)
                oneLC['BAND'] = [str(x).strip()[2] for x in oneLC['BAND']]

                for j in np.arange(len(bands)): # loop over bands for one transient
                    oneLC_oneBand = oneLC[oneLC['BAND'] == '%s'%bands[j]]   
                    master_photometry_dict['mag_%s'%bands[j]].append(oneLC_oneBand['SIM_MAGOBS'].values)
                    master_host_dict['mag_%s'%bands[j]].append(tempHead['HOSTGAL_MAG_%s'%bands[j]].values[k])
                    master_host_dict['magerr_%s'%bands[j]].append(tempHead['HOSTGAL_MAGERR_%s'%bands[j]].values[k])
                TID_counter += 1
                for l in master_photometry_dict['mag_u']:
                    if len(l)!= 101:
                        print("no")
                        print(tempPhot_fn)
        # create a new sub-group for this model, within the class group
        TID_sorted = np.argsort(master_photometry_dict['TID'])
        print("Photometry dict TID shape after single model: ",len(master_photometry_dict['TID']))
        print("Uniqueness of phot TIDs for this class:", len(np.unique(master_photometry_dict['TID'])))
        mod = tranclas.require_group("%s"%model)

        for key in ['TID','z', 'MJD', 'mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_Y',
        'cadence','GID', 'ra_off', 'dec_off', 'sep']:
            print("Attempting to create dataset %s"%key)
            if key not in mod.keys():
                if type(master_photometry_dict[key][0]) is str: # only true for GID and TID
                    print(key, "is string")
                    attr = np.asarray(master_photometry_dict[key], dtype='S7')
                elif key in ['MJD', 'mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_Y']:
                    dtype = master_photometry_dict[key][0][0].dtype
                    # these keys contain lists-of-arrays. arrays are equal sized however, so can be dtype 'float'
                    attr = np.asarray(master_photometry_dict[key], dtype=dtype)
                else:
                    attr = np.asarray(master_photometry_dict[key]) # allow auto-dtyping
                mod.create_dataset(key, data=attr[TID_sorted])

    GID_sorted = np.argsort(master_host_dict['GID'])
    print("Host dict TID shape after looping through models: ", len(master_host_dict['TID']))
    print("Uniqueness of host TIDs for this class:", len(np.unique(master_host_dict['TID'])))
    print("Host dict GID shape after looping through models: ", len(master_host_dict['GID']))
    print("Uniqueness of host GIDs for this class:", len(np.unique(master_host_dict['GID'])))

    for key in ['GID', 'dc2ID', 'z', 'mag_u', 'magerr_u', 'mag_g', 'magerr_g', 'mag_r', 'magerr_r',
           'mag_i', 'magerr_i', 'mag_z', 'magerr_z', 'mag_Y', 'magerr_Y',
            'logMstar', 'logSFR', 'T', 'a0', 'b0', 'n0', 'e0',
           'w0', 'a1', 'b1', 'n1', 'w1', 'e1', 'e', 'a_rot','TID']:
        if key not in hostclas.keys():
            if type(master_host_dict[key][0]) is str: # only true for GID and TID
                print(key, "is string")
                attr = np.asarray(master_host_dict[key], dtype='S7')
                hostclas.create_dataset(key, data=attr[GID_sorted], dtype='S7')
            else:
                attr = np.asarray(master_host_dict[key]) # allow auto-dtyping
                hostclas.create_dataset(key, data=attr[GID_sorted])

f.close()