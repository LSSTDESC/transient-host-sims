import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy import table
import random
import glob
import astropy.units as u
from astropy.coordinates import SkyCoord

#a very useful tool (this is in python3.9 but better not to risk it)
def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


#plotting functionality!!
sns.set_context("talk",font_scale=1.5)

sns.set_style('white', {'axes.linewidth': 0.5})
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['ytick.major.size'] = 15

plt.rcParams['xtick.minor.size'] = 10
plt.rcParams['ytick.minor.size'] = 10
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.minor.width'] = 2

plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


GID = np.arange(1, 5000001)
GID = ["G%07i" %x for x in GID]
#randomly shuffle
random.shuffle(GID)


bands = 'ugrizY'
NBAND = len(bands)

master_host_dict = {}  
master_photometry_dict = {}  
    
transDir = "/global/cscratch1/sd/kessler/SNANA_LSST_SIM/SCOTCH_CATALOG/"
prefix = "MLAG_GP_SCOTCH_FINAL"

# get all models
models = [x.split("/")[-1] for x in glob.glob(transDir + "*")]
models = [remove_prefix(x, prefix + "_") for x in models]

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

classes = [x.split("+")[0] for x in models]
for i in np.arange(len(classes)):
    if 'Templates' in classes[i]:
        classes[i] = classes[i][:-10]
    elif 'SALT2' in classes[i]:
        classes[i] = classes[i][:-6]
    elif 'MOSFIT' in classes[i]:
        classes[i] = classes[i][:-7]
    elif ('B19' in classes[i]) or ('K17' in classes[i]) or ('NMF' in classes[i]):
        classes[i] = classes[i][:-4]
    elif 'AGN' in classes[i]:
        classes[i] = 'AGN'
        
TID_counter = 1
GID_counter = 1

#fiducial skycoord at 0,0
c1 = SkyCoord(0.0*u.deg, 0.0*u.deg, frame='icrs')

#dictionary between our calculated GIDs and dc2 galaxy_id
dc2_map = {}

#initialize the mapping between GID and cosmoDC2 ID
for tempGID in GID:
    dc2_map[tempGID] = []

#only do the first three for now
for r in np.arange(len(models)): # loop over models
    fn = transDir + prefix + "_" + models[r] + "/"
    
    #glob automatically sorts by fn, so we're guaranteed that these correspond to each other! 
    #thank goodness.
    headfiles = sorted(glob.glob(fn + "*HEAD.FITS.gz"))
    photfiles = sorted(glob.glob(fn + "*PHOT.FITS.gz"))
    
    print("Starting on model %s"%models[r])
    
    #do it for each class separately, and save it, instead of trying to do all 5M at once 
    master_photometry_dict['TID'] = []
    master_photometry_dict['MJD'] = []

    for i in np.arange(len(bands)):
        master_photometry_dict['mag_%s'%bands[i]] = []
        master_host_dict['mag_%s'%bands[i]] = []
        master_host_dict['magerr_%s'%bands[i]] = []
    
    master_photometry_dict['Class'] = []
    master_photometry_dict['Model'] = []
    master_photometry_dict['z'] = []
    master_photometry_dict['RA_Off'] = []
    master_photometry_dict['Dec_Off'] = []
    master_photometry_dict['Sep'] = []
    master_photometry_dict['GID'] = []
    master_photometry_dict['cadence']  = []

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
    master_host_dict['logMsol'] = []
    master_host_dict['logSFR'] = []
    master_host_dict['T'] = []
    master_host_dict['Ellipticity'] = []
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
    master_host_dict['TID'] = []
    
    for i in np.arange(len(headfiles)): # loop over files in model directory
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
            repModel = models[r] 
            repClass = classes[r] 
            #repTID = TID  

            #flipping ra and dec to get the offset of the transient from the galaxy, not the offset of the galaxy from the transient
            #...it is this easy, right?
            repRAoff = tempHead['HOSTGAL_RA'].values[k]*3600#] #in arcsec of RA
            repDECoff = tempHead['HOSTGAL_DEC'].values[k]*3600#] #in arcsec of Dec
            
            #append to master dict
            TID = "T%.7i"%TID_counter
            
            master_host_dict['dc2ID'].append(dc2ID)
            master_host_dict['GID'].append(GID[TID_counter-1]) 
            master_host_dict['logMsol'].append(tempHead['SIM_HOSTLIB(LOGMASS_TRUE)'].values[k]) 
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
            master_host_dict['Ellipticity'].append(tempHead['SIM_HOSTLIB(ELLIPTICITY)'].values[k]) 
            master_host_dict['a_rot'].append(tempHead['SIM_HOSTLIB(a_rot)'].values[k]) 
            master_host_dict['TID'].append(TID) 
            
            master_photometry_dict['TID'].append(TID)
            master_photometry_dict['Class'].append(repClass)
            master_photometry_dict['cadence'].append(cadence_dict[models[r]])
            master_photometry_dict['Model'].append(repModel)
            master_photometry_dict['GID'].append(repGID)
            master_photometry_dict['z'].append(z)
            master_photometry_dict['RA_Off'].append(repRAoff)
            master_photometry_dict['Dec_Off'].append(repDECoff)
            
            master_photometry_dict['MJD'].append(np.unique([oneLC['MJD'].values]))

            #calculate separation 
            c2 = SkyCoord(tempHead['HOSTGAL_RA'].values[k]*u.deg, tempHead['HOSTGAL_DEC'].values[k]*u.deg, frame='icrs')
            sep = c1.separation(c2)
            master_photometry_dict['Sep'].append(sep.arcsec)
                
            oneLC['BAND'] = [str(x).strip()[2] for x in oneLC['BAND']]
            
            for j in np.arange(len(bands)): # loop over bands for one transient
                oneLC_oneBand = oneLC[oneLC['BAND'] == '%s'%bands[j]]   
                master_photometry_dict['mag_%s'%bands[j]].append(oneLC_oneBand['SIM_MAGOBS'].values)
                master_host_dict['mag_%s'%bands[j]].append(tempHead['HOSTGAL_MAG_%s'%bands[j]].values[k])
                master_host_dict['magerr_%s'%bands[j]].append(tempHead['HOSTGAL_MAGERR_%s'%bands[j]].values[k])
            TID_counter += 1
            
    #order them correctly 
    master_hostDF = pd.DataFrame(master_host_dict)[['GID', 'dc2ID','mag_u', 'magerr_u', 'mag_g', 'magerr_g', 'mag_r', 'magerr_r',
       'mag_i', 'magerr_i', 'mag_z', 'magerr_z', 'mag_Y', 'magerr_Y',
        'logMsol', 'logSFR', 'T', 'Ellipticity', 'a0', 'b0', 'n0',
       'w0', 'a1', 'b1', 'w1', 'n1', 'a_rot','TID']]

    master_photDF = pd.DataFrame(master_photometry_dict)[['TID','z', 'MJD', 'mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_Y',
       'Class', 'Model', 'cadence','GID', 'RA_Off', 'Dec_Off', 'Sep']]
 
    master_hostDF.sort_values(by='GID', inplace=True)
    master_photDF.sort_values(by='TID', inplace=True)

    master_photDF.to_csv("/global/cscratch1/sd/agaglian/SCOTCH_TransientTable_%s.tar.gz"%models[r],index=False)
    master_hostDF.to_csv("/global/cscratch1/sd/agaglian/SCOTCH_HostTable_%s.tar.gz"%models[r],index=False)


print(master_hostDF.head())
print(master_photDF.head())
