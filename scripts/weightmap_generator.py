#create a weightmap associated with the x1 and SNIa rate correlations presented
#in Vincenzi+2020
#create a 3D meshgrid over logmass, SFR, and x1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import logistic
from scipy.stats import norm

%matplotlib inline

#######################################################################################################################################################
############                       First, generate meshgrids for all weightmaps                                                                     ###
#######################################################################################################################################################

SNIa_HOSTLIB = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/HOSTLIBs/Z3_HOSTLIB/SNIa_GHOST.HOSTLIB", delim_whitespace=True)
#SNII_HOSTLIB = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/HOSTLIBs/BIG_HOSTLIBS/SNII_GHOST.HOSTLIB", delim_whitespace=True)
#SNIbc_HOSTLIB = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/HOSTLIBs/BIG_HOSTLIBS/SNIbc_GHOST.HOSTLIB", delim_whitespace=True)
rand = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/HOSTLIBs/Z3_HOSTLIB/UNMATCHED_COSMODC2_GHOST.HOSTLIB", delim_whitespace=True)

plt.hist(10**rand['LOGMASS'], bins=np.logspace(5, 13), density=False)
plt.hist(10**SNIa_HOSTLIB['LOGMASS'], bins=np.logspace(5, 13), density=False)
#plt.yscale("log")
plt.xscale("log")

#rand['stellar_mass']
np.log10(np.nanmin(rand['totalStarFormationRate']))-9
#np.log10(np.nanmax(rand['totalStarFormationRate']))-9
#-12.192453269137832
#1.992177909419377

np.nanmin(rand['mag_true_g_lsst'])
np.nanmax(rand['mag_true_g_lsst'])
#14.506942
#29.82764229.827642

np.nanmin(rand['mag_true_r_lsst'])
np.nanmax(rand['mag_true_r_lsst'])

np.nanmin(rand['mag_true_i_lsst'])
np.nanmax(rand['mag_true_i_lsst'])

np.log10(np.nanmin(rand['stellar_mass']))
np.log10(np.nanmax(rand['stellar_mass']))
#4.6991316186676215
#12.202728692734057
#np.nanmin(SNIbc_HOSTLIB['LOG_SFR'])
#np.nanmax(SNIbc_HOSTLIB['LOG_SFR'])
#SNIa hostlib:
# logmass goes from 4.85 to 12.08
# sfr goes from -7.5 to 1

#SNII hostlib:
# logmass goes from 4.97 to 12.08
# sfr goes from -7.5 to 1

#SNIbc hostlib:
# logmass goes from 5.059 to 12.08
# sfr goes from -7.5 to 1

#change the plot styling
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
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

path_matched = '/Users/alexgagliano/Documents/Research/DESC/tables/matchedSample/matchedSamples_1218/'
fnII = path_matched + "/cdc2_matched_ghost_SNII_z3_unq_zwgt_5pct_k3147_SFRMsol.tar.gz"
fnIa = path_matched + "/cdc2_matched_ghost_SNIa_z3_unq_zwgt_5pct_k678_SFRMsol.tar.gz"
fnIbc = path_matched + "/cdc2_matched_ghost_SNIbc_z3_unq_zwgt_5pct_k9508_SFRMsol.tar.gz"

SNII_hostlib = pd.read_csv(fnII)
SNIa_hostlib = pd.read_csv(fnIa)
SNIbc_hostlib = pd.read_csv(fnIbc)

#np.log10(np.nanmin(SNII_hostlib['totalStarFormationRate']))
#np.log10(np.nanmax(SNII_hostlib['totalStarFormationRate']))

#np.log10(np.nanmin(SNII_hostlib['stellar_mass']))
#np.log10(np.nanmax(SNII_hostlib['stellar_mass']))

#np.log10(np.nanmin(SNIa_hostlib['totalStarFormationRate']))
#np.log10(np.nanmax(SNIa_hostlib['totalStarFormationRate']))

#np.log10(np.nanmin(SNIa_hostlib['stellar_mass']))
#np.log10(np.nanmax(SNIa_hostlib['stellar_mass']))

#np.log10(np.nanmin(SNIbc_hostlib['totalStarFormationRate']))-9
#np.log10(np.nanmax(SNIbc_hostlib['totalStarFormationRate']))-9

#np.log10(np.nanmin(SNIbc_hostlib['stellar_mass']))
#np.log10(np.nanmax(SNIbc_hostlib['stellar_mass']))

#SNIa_hostlib = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/matchedGals_IaGhostlib.tar.gz")
#SNIa_hostlib = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/matchedGals_IbcSLSNIGhostlib.tar.gz")
#unweighted_hostlib = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/HOSTLIBs/Mabs_HOSTLIBS/UNMATCHED_COSMODC2_GHOST_abs.HOSTLIB", delim_whitespace=True)

stepsize = 0.2
stepsizeIc = 0.5
stepsizeAGN = 0.1

#N = 101
#a bit of funnny business to ensure that the stepsize is the same for all the params
#commented-out grids are from Maria Vincenzi's original WGTMAPS!
x1_start = -5
x1_end = 5
x1_ = np.linspace(x1_start, x1_end, int((x1_end - x1_start)/stepsize)+1)
#x1_ = np.array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.]) #forcing to the Vincenzi vals

#SFR = [<-3.2, >11]
#Msol = [<4.6, >12.6]

logmass_start = 4.5
logmass_end = 13.5
logmass_ = np.linspace(logmass_start, logmass_end, int((logmass_end - logmass_start)/stepsize)+1)
#logmass_ = np.array([ 5.   ,  5.225,  5.45 ,  5.675,  5.9  ,  6.125,  6.35 ,  6.575,
#        6.8  ,  7.025,  7.25 ,  7.475,  7.7  ,  7.925,  8.15 ,  8.375,
#        8.6  ,  8.825,  9.05 ,  9.275,  9.5  ,  9.725,  9.95 , 10.175,
#       10.4  , 10.625, 10.85 , 11.075, 11.3  , 11.525, 11.75 , 11.975,
#       12.2  , 12.425, 12.65 , 12.875, 13.1  , 13.325, 13.55 , 13.775])

logSFR_start = -12.5
logSFR_end = 2.5
logSFR_ = np.linspace(logSFR_start, logSFR_end, int((logSFR_end - logSFR_start)/stepsize)+1) #
#SFR_ = 10**np.array([-20.1  , -19.875, -19.65 , -19.425, -19.2  , -18.975, -18.75 ,
#       -18.525, -18.3  , -18.075, -17.85 , -17.625, -17.4  , -17.175,
#       -16.95 , -16.725, -16.5  , -16.275, -16.05 , -15.825, -15.6  ,
#       -15.375, -15.15 , -14.925, -14.7  , -14.475, -14.25 , -14.025,
#       -13.8  , -13.575, -13.35 , -13.125, -12.9  , -12.675, -12.45 ,
#       -12.225, -12.   , -11.775, -11.55 , -11.325, -11.1  , -10.875,
#       -10.65 , -10.425, -10.2  ,  -9.975,  -9.75 ,  -9.525,  -9.3  ,
#        -9.075,  -8.85 ,  -8.625,  -8.4  ,  -8.175,  -7.95 ,  -7.725,
#        -7.5  ,  -7.275,  -7.05 ,  -6.825,  -6.6  ,  -6.375,  -6.15 ,
#        -5.925,  -5.7  ,  -5.475,  -5.25 ,  -5.025,  -4.8  ,  -4.575,
#        -4.35 ,  -4.125,  -3.9  ,  -3.675,  -3.45 ,  -3.225,  -3.   ,
#        -2.775,  -2.55 ,  -2.325,  -2.1  ,  -1.875,  -1.65 ,  -1.425,
#        -1.2  ,  -0.975,  -0.75 ,  -0.525,  -0.3  ,  -0.075,   0.15 ,
#         0.375,   0.6  ,   0.825,   1.05 ,   1.275,   1.5  ,   1.725,
#         1.95 ,   2.175,   2.4  ,   2.625,   2.85 ,   3.075,   3.3  ,
#         3.525,   3.75 ,   3.975])
SFR_ = 10**(logSFR_)

#Mg should go between -24, -14
#g 12.5 to 32.5
#r 12.5 to 32.5
Mg_start = -24.5
Mg_end = -14
g_start = 12.5
g_end = 32.5
r_start = 12.5
r_end = 32.5

#run it for Ic and Ic-BL
Mg_ = np.linspace(Mg_start, Mg_end, int((Mg_end - Mg_start)/stepsizeIc)+1) #
g_  = np.linspace(g_start, g_end, int((g_end - g_start)/stepsizeIc)+1)
r_  = np.linspace(r_start, r_end, int((r_end - g_start)/stepsizeIc)+1)
logmassIc_ = np.linspace(logmass_start, logmass_end, int((logmass_end - logmass_start)/stepsizeIc)+1)
logSFRIc_ = np.linspace(logSFR_start, logSFR_end, int((logSFR_end - logSFR_start)/stepsizeIc)+1) #

Mg_mesh, g_mesh, r_mesh, logmass_meshIc, SFR_meshIc = np.meshgrid(Mg_, g_, r_, logmassIc_, logSFRIc_)

x1_mesh, logmass_mesh, SFR_mesh = np.meshgrid(x1_, logmass_, logSFR_)
x1_allMesh = np.ravel(x1_mesh)
logmass_allMesh = np.ravel(logmass_mesh)
mass_allMesh = 10**logmass_allMesh
logSFR_allMesh = np.ravel(SFR_mesh)
SFR_allMesh = 10**logSFR_allMesh

Mg_allMesh = np.ravel(Mg_mesh)
g_allMesh = np.ravel(g_mesh)
r_allMesh = np.ravel(r_mesh)
logmass_allMeshIc = np.ravel(logmass_meshIc)
mass_allMeshIc = 10**logmass_allMeshIc
logSFR_allMeshIc = np.ravel(SFR_meshIc)
SFR_allMeshIc = 10**logSFR_allMeshIc

logmass_allMeshAGN = np.ravel(np.linspace(logmass_start, logmass_end, int((logmass_end - logmass_start)/stepsize)+1))
mass_allMeshAGN = 10**logmass_allMeshAGN

#######################################################################################################################################################
############                       Next, define the rate functions for all classes                                                                  ###
#######################################################################################################################################################
# A kde based on the DES-3YR sample
def R_AB_kde(Msol, SFR):
    import statsmodels.api as sm
    DES_3YR = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/WGTMAPs/Smith2020/DES3YR.csv", delim_whitespace=True)
    DES_3YR['LOG_SFR'] = np.log10(10**DES_3YR['LOG_sSFR']*10**DES_3YR['LOG_MSTELLAR'])
    DES_3YR_cut = DES_3YR[['LOG_MSTELLAR','LOG_SFR']]
    DES_3YR_cut.dropna(inplace=True)
    dens = sm.nonparametric.KDEMultivariate(data=DES_3YR_cut, var_type='cc')
    Rate = dens.pdf([np.log10(Msol), np.log10(SFR)])
    return Rate

def RateII(Msol, SFR):
    rate = np.zeros(len(Msol))
    sSFR = SFR/Msol
    cond1 = np.log10(sSFR) >= -11.5
    rate[cond1] = Msol[cond1]**0.16
    return rate

def RateIax(Msol, SFR, x1):
    rate = np.zeros(len(Msol))
    sSFR = SFR/Msol
    cond1 = np.log10(sSFR) >= -11.5
    cond2 = np.log10(sSFR) < -11.5 #exponential dropoff

    norm1 = np.nanmax(R_AB_kde(Msol[cond1], SFR[cond1])*Rstar(x1[cond1], Msol[cond1]))
    norm2 = np.nanmax(np.exp(np.log10(sSFR[cond2])+11.5))

    rate[cond1] = R_AB_kde(Msol[cond1], SFR[cond1])*Rstar(x1[cond1], Msol[cond1])
    rate[cond2] = R_AB_kde(Msol[cond2], SFR[cond2])*Rstar(x1[cond2], Msol[cond2])*np.exp(5*np.log10(sSFR[cond2])+11.5)

#    rate[cond2] /= np.nanmax(rate[cond2])
#    rate[cond2] *= 1.e-10
    return rate

def Rate91bg(Msol, SFR, x1):
    rate = np.zeros(len(Msol))
    sSFR = SFR/Msol
    cond1 = np.log10(sSFR) < -11.5
    cond2 = np.log10(sSFR) >= -11.5 #exponential dropoff in active hosts

    rate[cond1] = R_AB_kde(Msol[cond1], SFR[cond1])*Rstar(x1[cond1], Msol[cond1])
    rate[cond2] =  R_AB_kde(Msol[cond2], SFR[cond2])*Rstar(x1[cond2], Msol[cond2])*np.exp(-5*(np.log10(sSFR[cond2])+11.5))

    #rate[cond2] /= np.nanmax(rate[cond2])
    #rate[cond2] *= 1.e-10
    return rate

def metal_PDF_SNIc(logOH_12):
    return logistic.pdf(9*(logOH_12-8.9))

def metal_PDF_SNIc_BL(logOH_12):
    return logistic.pdf(10*(logOH_12-8.5))

def metal_PDF_SNIc_del(logOH_12):
    return logistic.pdf(100*(logOH_12-8.9))*4

def metal_PDF_SNIc_BL_del(logOH_12):
    return logistic.pdf(100*(logOH_12-8.5))*4

def get_metallicity(Mg, gmag, rmag):
    alpha = 10.5
    p3 = -9.02/1.e5
    p2 = -18.82/1.e3
    p1 = -836.54/1.e3
    p0 = -1.87
    mu = Mg - alpha*(gmag - rmag)
    logOH_12 = p0 + p1*mu + p2*mu**2 + p3*mu**3
    return logOH_12

#def R_SNIc_metal(Mg, gmag, rmag):
#    logOH_12 = get_metallicity(Mg, gmag, rmag)
#    return metal_PDF_SNIc(logOH_12)
#    return metal_PDF_SNIc_del(logOH_12)

#def R_SNIc_BL_metal(Mg, gmag, rmag):
#    logOH_12 = get_metallicity(Mg, gmag, rmag)
#    return metal_PDF_SNIc_BL_del(logOH_12)

def get_metallicity_FMR(Msol, SFR):
    m = np.log10(Msol)-10
    s = np.log10(SFR)
    logOH_12 = 8.90 + 0.37*m - 0.14*s - 0.19*m**2  + 0.12*m*s - 0.054*s**2
    return logOH_12

def R_SNIc_metal(Msol, SFR):
    logOH_12 = get_metallicity_FMR(Msol, SFR)
    return metal_PDF_SNIc(logOH_12)

def R_SNIc_BL_metal(Msol, SFR):
    logOH_12 = get_metallicity_FMR(Msol, SFR)
    return metal_PDF_SNIc_BL(logOH_12)

def R_AB(Msol, SFR):
    #from https://ui.adsabs.harvard.edu/abs/2006ApJ...648..868S
    A = 5.3e-14 #SNe/yr/Msol
    B = 3.9e-4 #SNe/yr * (Msol/yr)^-1
    return A*Msol + B*SFR

def R_AB2(Msol, SFR):
    #from https://ui.adsabs.harvard.edu/abs/2006ApJ...648..868S
    A = 5.1e-14
    B = 4.1e-4
    return A*Msol + B*SFR

def Rstar(x1, Msol):
    #from https://arxiv.org/pdf/2012.07180.pdf
    rate = np.ones(len(x1))
    cond1 = ((x1 < 0) & (Msol < 1.e10))
    rate[cond1] = np.exp(-x1[cond1]**2)
    return rate

#previously -- rate was defined over absolute g-mag, apparent g and r-mag.
#now have switched over to a metallicity function defined over SFR and Msol.
#def RateIc_BL(Msol, SFR, Mg, gmag, rmag):
#    rate = np.zeros(len(Msol))
#    sSFR = SFR/Msol
#    cond1 = np.log10(sSFR) >= -11.5
#    rate[cond1] = Msol[cond1]**0.36
#    rate2 = R_SNIc_BL_metal(Mg, gmag, rmag)
#    rate_comb = rate*rate2
#    return rate_comb

#def RateIc(Msol, SFR, Mg, gmag, rmag):
#    rate = np.zeros(len(Msol))
#    sSFR = SFR/Msol
#    cond1 = np.log10(sSFR) >= -11.5
#    rate[cond1] = Msol[cond1]**0.36
#    rate2 = R_SNIc_metal(Mg, gmag, rmag)
#    rate_comb = rate*rate2
#    return rate_comb

def RateIc(Msol, SFR):
    rate = np.zeros(len(Msol))
    sSFR = SFR/Msol
    cond1 = np.log10(sSFR) >= -11.5
    rate[cond1] = Msol[cond1]**0.36
    rate2 = R_SNIc_metal(Msol, SFR)
    #DEBUGGING PURPOSES ONLY
    rate2 /= np.nanmax(rate2)
    rate /= np.nanmax(rate)
    rate_comb = rate*rate2
    #return rate, rate2, rate_comb
    #return rate2/np.nanmax(rate2)
    return rate_comb

def RateIc_BL(Msol, SFR):
    rate = np.zeros(len(Msol))
    sSFR = SFR/Msol
    cond1 = np.log10(sSFR) >= -11.5
    rate[cond1] = Msol[cond1]**0.36
    rate2 = R_SNIc_BL_metal(Msol, SFR)
    rate2 /= np.nanmax(rate2)
    rate /= np.nanmax(rate)
    rate_comb = rate*rate2
    #return rate, rate2, rate_comb
    return rate_comb
    #return rate2/np.nanmax(rate2)

def RateIb(Msol, SFR):
    rate = np.zeros(len(Msol))
    sSFR = SFR/Msol
    cond1 = np.log10(sSFR) >= -11.5
    rate[cond1] = Msol[cond1]**0.36
    return rate

def RateTDE(Msol, SFR):
    rate = np.ones(len(Msol))
    cond1 = (np.log10(Msol) < 11) & (np.log10(SFR) >= 0)
    cond2 = (np.log10(Msol) >= 11) & (np.log10(SFR) < 0)
    cond3 = (np.log10(Msol) > 11) & (np.log10(SFR) >= 0)
    rate[cond1] = np.exp(-np.log10(SFR[cond1]))
    rate[cond2] = np.exp(-np.log10(Msol[cond2]))
    rate[cond3] = np.exp(-(np.log10(Msol[cond3]) + np.log10(SFR[cond3])))
    #normalize
    rate[cond1] /= np.nanmax(rate[cond1])
    rate[cond2] /= np.nanmax(rate[cond2])
    rate[cond3] /= np.nanmax(rate[cond3])
    return rate

def RateAGN(Msol):
    return norm.pdf(np.log10(Msol), loc=10.8, scale=0.5)

def RateSLSNI(Msol, SFR):
    #from https://arxiv.org/pdf/2012.07180.pdf
    sSFR = SFR/Msol
    rate = np.ones(len(Msol))
    cond1 = (np.log10(sSFR) < -10) #& (Msol > 1.e10)
    cond2 = np.logical_and(np.log10(Msol) > 10., np.log10(SFR) > -0.5) #& (Msol > 1.e10)
    rate[cond1] = np.exp(-np.abs(np.log10(sSFR[cond1])+10))
    rate[cond2] = np.exp(-np.abs(np.log10(Msol[cond2])-3))/(2*np.nanmax(np.exp(-np.abs(np.log10(Msol[cond2])-3)))) #normalize this part
    return rate

# A kde based on the original Sullivan (2016) hosts.
def R_AB_kde_mod(Msol, SFR):
    import statsmodels.api as sm
    Sullivan06 = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/WGTMAPs/Sullivan06_FullData.csv", header=None, names=['log10Msol', 'log10SFR'])
    Sullivan06.dropna(inplace=True)
    dens = sm.nonparametric.KDEMultivariate(data=Sullivan06, var_type='cc')
    Rate = dens.pdf([np.log10(Msol), np.log10(SFR)])
    return Rate

def RateIa(Msol, SFR, x1):
    #return R_AB(Msol, SFR)*Rstar(x1, Msol)
    return R_AB_kde(Msol, SFR)*Rstar(x1, Msol)



#######################################################################################################################################################
############                                    Make and save the weightmap files                                                                   ###
#######################################################################################################################################################

sPath = '/Users/alexgagliano/Documents/Research/DESC/tables/WGTMAPs/'

rate_allMesh_Ia = RateIa(mass_allMesh, SFR_allMesh, x1_allMesh)
rate_allMesh_Iax = RateIax(mass_allMesh, SFR_allMesh, x1_allMesh)
rate_allMesh_91bg = Rate91bg(mass_allMesh, SFR_allMesh, x1_allMesh)
rate_allMesh_TDE = RateTDE(mass_allMesh, SFR_allMesh)
rate_allMesh_II = RateII(mass_allMesh, SFR_allMesh)
rate_allMesh_Ib = RateIb(mass_allMesh, SFR_allMesh)
rate_allMesh_Ic = RateIc(mass_allMeshIc, SFR_allMeshIc)
rate_allMesh_Ic_BL = RateIc_BL(mass_allMeshIc, SFR_allMeshIc)
rate_allMesh_SLSNI = RateSLSNI(mass_allMesh, SFR_allMesh)
rate_allMesh_AGN = RateAGN(mass_allMeshAGN)

weight_prefix = np.array(['WGT:']*len(logmass_allMesh))
weight_prefixIc = np.array(['WGT:']*len(logmass_allMeshIc))
weight_prefixAGN = np.array(['WGT:']*len(logmass_allMeshAGN))

snmagshift = np.zeros(len(logmass_allMesh))
snmagshift_AGN = np.zeros(len(logmass_allMeshAGN))
snmagshift_Ic = np.zeros(len(logmass_allMeshIc))
snmagshift_TDE = np.zeros(len(logmass_allMesh))

WGTMAP_Ia = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh), 'x1':x1_allMesh,'WGT':rate_allMesh_Ia, 'SNMAGSHIFT':snmagshift})
#WGTMAP_Ia.to_csv(sPath + "/SNIa_GHOST_Z3.WGTMAP",index=False, float_format='%.3e', sep=' ')

WGTMAP_Iax = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh), 'x1':x1_allMesh,'WGT':rate_allMesh_Iax, 'SNMAGSHIFT':snmagshift})
WGTMAP_Iax.to_csv(sPath + "/SNIax.WGTMAP",index=False, float_format='%.3e', sep=' ')

WGTMAP_91bg = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh), 'x1':x1_allMesh,'WGT':rate_allMesh_91bg, 'SNMAGSHIFT':snmagshift})
WGTMAP_91bg.to_csv(sPath + "/SN91bg.WGTMAP",index=False, float_format='%.3e', sep=' ')

WGTMAP_II = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh),'WGT':rate_allMesh_II, 'SNMAGSHIFT':snmagshift})
#WGTMAP_II.to_csv(sPath + "/SNII_GHOST_Z3.WGTMAP",index=False, float_format='%.3e', sep=' ')

WGTMAP_Ib = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh),'WGT':rate_allMesh_Ib, 'SNMAGSHIFT':snmagshift})
#WGTMAP_Ib.to_csv(sPath + "/SNIb_GHOST_Z3.WGTMAP",index=False, float_format='%.3e', sep=' ')

#WGTMAP_Ic = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefixIc, 'LOGMASS':logmass_allMeshIc, 'LOG_SFR':np.log10(SFR_allMeshIc), 'Mag_true_g_sdss_z0': Mg_allMesh, 'g_obs':g_allMesh, 'r_obs':r_allMesh, 'WGT':rate_allMesh_Ic})
WGTMAP_Ic = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefixIc, 'LOGMASS':logmass_allMeshIc, 'LOG_SFR':np.log10(SFR_allMeshIc), 'WGT':rate_allMesh_Ic, 'SNMAGSHIFT':snmagshift_Ic})
#WGTMAP_Ic.to_csv(sPath + "/SNIc_GHOST_Z3.WGTMAP",index=False, float_format='%.3e', sep=' ')#float_format='%.3f',

#WGTMAP_Ic_BL = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefixIc, 'LOGMASS':logmass_allMeshIc, 'LOG_SFR':np.log10(SFR_allMeshIc),'Mag_true_g_sdss_z0': Mg_allMesh, 'g_obs':g_allMesh, 'r_obs':r_allMesh, 'WGT':rate_allMesh_Ic_BL})
WGTMAP_Ic_BL = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefixIc, 'LOGMASS':logmass_allMeshIc, 'LOG_SFR':np.log10(SFR_allMeshIc), 'WGT':rate_allMesh_Ic_BL, 'SNMAGSHIFT':snmagshift_Ic})
#WGTMAP_Ic_BL.to_csv(sPath + "/SNIcBL_GHOST_Z3.WGTMAP",float_format='%.3e', index=False, sep=' ') #float_format='%.3f',

WGTMAP_AGN = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefixAGN, 'LOGMASS':logmass_allMeshAGN,'WGT':rate_allMesh_AGN, 'SNMAGSHIFT':snmagshift_AGN})
#WGTMAP_AGN.to_csv(sPath + "/AGN_GHOST_Z3.WGTMAP",index=False, float_format='%.3e', sep=' ')

WGTMAP_TDE = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh,'LOG_SFR':np.log10(SFR_allMesh),'WGT':rate_allMesh_TDE, 'SNMAGSHIFT':snmagshift})
#WGTMAP_TDE.to_csv(sPath + "/TDE_GHOST_Z3.WGTMAP",index=False, float_format='%.3e', sep=' ')

WGTMAP_SLSNI = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh),'WGT':rate_allMesh_SLSNI, 'SNMAGSHIFT':snmagshift})
#WGTMAP_SLSNI.to_csv(sPath + "/SLSNI_GHOST_Z3.WGTMAP",index=False, float_format='%.3e', sep=' ')


#Iax should be in star-forming galaxies
#91bg-like should be in passive galaxies
WGTMAP_Iax['LOGSFR'] = WGTMAP_Iax['LOG_SFR']
plt.figure(figsize=(10,7))
sns.scatterplot(WGTMAP_Iax['LOGMASS'], WGTMAP_Iax['LOGSFR'], hue=WGTMAP_Iax['WGT'], legend=True)

WGTMAP_91bg['LOGSFR'] = WGTMAP_91bg['LOG_SFR']
plt.figure(figsize=(10,7))
sns.scatterplot(WGTMAP_91bg['LOGMASS'], WGTMAP_91bg['LOGSFR'], hue=WGTMAP_91bg['WGT'], legend=True)

#######################################################################################################################################################
############                       Make diagnostic plots to validate the weightmaps                                                                 ###
#######################################################################################################################################################
plt.figure(figsize=(10,7))
#plt.hist(rand['stellar_mass'], bins=np.logspace(5, 13), density=True, label='Rand Hostlib')
plt.hist(10**rand['LOGMASS'], bins=np.logspace(5, 13),  label='Rand Hostlib')
#plt.hist(10**SNIa_HOSTLIB['LOGMASS'], bins=np.logspace(5, 12), label='SNIa Hostlib')
plt.xscale("log")
plt.xlabel(r"Mass ($M_{\odot}$)")
plt.ylabel("PDF")
plt.plot(10**WGTMAP_AGN['LOGMASS'], WGTMAP_AGN['WGT']*1.5e5, label='AGN Weightmap')
plt.legend()
#plt.savefig("/Users/alexgagliano/Documents/Research/DESC/transient-host-sims/plots/AGN_tooFewHosts.png",dpi=200, bbox_inches='tight')

########### quick plots to compare the distributions in metallicity space for Ic and Ic-BL
########### the "_del" suffix refers to a delta-like function centered on the median metallicity
########### for that class.
plt.figure(figsize=(10,7))
x = np.linspace(7.5, 9.5, num=500)
plt.ylim((0, 1))
plt.xlim((7.5, 9.5))
plt.plot(x, metal_PDF_SNIc_BL_del(x), c='tab:blue', label='SN Ic-BL')
plt.plot(x, metal_PDF_SNIc_BL(x), c='tab:blue')
plt.plot(x, metal_PDF_SNIc_del(x), c='k', label='SN Ic')
plt.plot(x, metal_PDF_SNIc(x), c='k')
plt.xlabel("log(O/H)+12")
plt.legend(fontsize=14)
plt.ylabel("PDF")

plt.figure(figsize=(7,5))
plt.xlim((7, 11))
plt.ylim((-6, 2))
plt.xlabel("log10(M*)")
plt.ylabel("log10(SFR)")
sns.scatterplot(np.log10(mass_allMesh), np.log10(SFR_allMesh), hue=RateIc(mass_allMesh, SFR_allMesh), legend=True, s=320, edgecolor=None)
plt.title("SN Ic Weightmap (FMR)")

plt.figure(figsize=(7,5))
plt.xlim((7, 11))
plt.ylim((-6, 2))
plt.xlabel("log10(M*)")
plt.ylabel("log10(SFR)")
sns.scatterplot(np.log10(mass_allMesh), np.log10(SFR_allMesh), hue=RateIc_BL(mass_allMesh, SFR_allMesh), legend=False, s=320, edgecolor=None)
plt.title("SN Ic-BL Weightmap (FMR)")

########### New method for SNIa -- fitting KDE to DES sample in loglog space
DES_3YR = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/WGTMAPs/Smith2020/DES3YR.csv", delim_whitespace=True)
Sullivan06 = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/WGTMAPs/Sullivan06_FullData.csv", header=None, names=['log10Msol', 'log10SFR'])

#what does the distribution of Ia hosts from literature look like?
plt.figure(figsize=(10,8))
plt.plot(Sullivan06['log10Msol'], Sullivan06['log10SFR'],'o', c='#55D6BE', mec='k', label='Sullivan+2006', zorder=100)
DES_3YR['LOG_SFR'] = np.log10(10**DES_3YR['LOG_sSFR']*10**DES_3YR['LOG_MSTELLAR'])
plt.plot(DES_3YR['LOG_MSTELLAR'], DES_3YR['LOG_SFR'], 'o', c='#7D5BA6', mec='k', label='Smith+2020 (DES 3YR)')
plt.legend(fontsize=18)
plt.xlabel(r"log$_{10}$($M_*$)")
plt.ylabel(r"log$_{10}$(SFR)")
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/LiteratureSNIaHosts.png",dpi=200, bbox_inches='tight')

#show the weightmap for the Sullivan data
import statsmodels.api as sm
Sullivan06 = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/WGTMAPs/Sullivan06_FullData.csv", header=None, names=['log10Msol', 'log10SFR'])
dens = sm.nonparametric.KDEMultivariate(data=Sullivan06, var_type='cc')

y = dens.pdf([np.log10(mass_allMesh), np.log10(SFR_allMesh)])
plt.figure(figsize=(7, 7))
plt.xlabel(r"log_{10}(Msol)")
plt.ylabel(r"log_{10}(SFR)")
plt.ylim((-6, 2))
plt.xlim((7, 11))
sns.scatterplot(np.log10(mass_allMesh), np.log10(SFR_allMesh), hue=y, legend=False, edgecolor='None', s=400)
plt.title("Sullivan+2006 kde-based Weightmap")
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/Sullivan06_kdeWGTMAP.png",dpi=200, bbox_inches='tight')

####### TDE
#plt.figure(figsize=(10,7))
#sns.scatterplot(WGTMAP_TDE['LOGMASS'], WGTMAP_TDE['LOG_SFR'], hue=WGTMAP_TDE['WGT'], legend=False, s=150, edgecolor=None,palette='cividis')
#plt.xlabel(r"log$_{10}$($M/M_{\odot})$");
#plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)");
#plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/TDE_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')

####### SLSNI
#plt.figure(figsize=(10,7))
#sns.scatterplot(WGTMAP_SLSNI['LOGMASS'], WGTMAP_SLSNI['LOG_SFR'], hue=WGTMAP_SLSNI['WGT'], legend=False, s=150, edgecolor=None,palette='cividis')
#plt.xlabel(r"log$_{10}$($M/M_{\odot})$");
#plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)");
#plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SLSNI_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')


#marginalize over x1 to just get logSFR vs logMsol
logmass_x1_marg = []
SFR_x1_marg = []
WGT_x1_marg = []

for temp_logmass in logmass_:
    tempDF = WGTMAP_Iax[WGTMAP_Iax['LOGMASS']==temp_logmass]
    for temp_SFR in SFR_:
        tempDF_WGT = tempDF.loc[10**(tempDF['LOG_SFR'])==temp_SFR, 'WGT'].values
        newWGT = np.nansum(tempDF_WGT)
        logmass_x1_marg.append(temp_logmass)
        SFR_x1_marg.append(temp_SFR)
        WGT_x1_marg.append(newWGT)

WGTMAPIax_x1_marg = pd.DataFrame({'logmass':np.array(logmass_x1_marg), 'SFR':np.array(SFR_x1_marg), 'WGT':np.array(WGT_x1_marg)})
plt.figure(figsize=(10,7))
#plt.xlim((5, 13))
#plt.ylim((-20, 5))
sns.scatterplot(logmass_x1_marg, np.log10(SFR_x1_marg), hue=-np.log10(WGT_x1_marg),s=180, edgecolor=None,legend=False)
plt.xlabel(r"log$_{10}$($M/M_{\odot})$");
plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)");
#plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SNIax_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')

Vincenzi_Ia = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/WGTMAPs/V21_SNIa_REVISED_data.WGTMAP", delim_whitespace=True)

logmass_x1_margV = []
SFR_x1_margV = []
WGT_x1_margV = []

for temp_logmass in np.unique(Vincenzi_Ia['LOGMASS']):
    tempDF = Vincenzi_Ia[Vincenzi_Ia['LOGMASS']==temp_logmass]
    for temp_SFR in np.unique(Vincenzi_Ia['LOG_SFR']):
        tempDF_WGT = tempDF.loc[tempDF['LOG_SFR']==temp_SFR, 'WGT'].values
        newWGT = np.nansum(tempDF_WGT)
        logmass_x1_margV.append(temp_logmass)
        SFR_x1_margV.append(temp_SFR)
        WGT_x1_margV.append(newWGT)

Vincenzi_marg = pd.DataFrame({'logmass':logmass_x1_margV, 'SFR':SFR_x1_margV, 'WGT':WGT_x1_margV})
Vincenzi_marg_cut = Vincenzi_marg[Vincenzi_marg['logmass']<12.5]

plt.figure(figsize=(10,7))
sns.scatterplot(Vincenzi_marg_cut['logmass'], Vincenzi_marg_cut['SFR'], hue=np.log10(Vincenzi_marg_cut['WGT']),  s=900, palette='viridis', edgecolor=None)
plt.xlabel(r"log$_{10}$($M/M_{\odot})$")
plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)")
plt.xlim((7, 11))
plt.ylim((-6, 2s))
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/Vincenzi_SNIa_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')

####### SN Ibc
plt.figure(figsize=(10,7))
sns.scatterplot(WGTMAP_Ibc['LOGMASS'], WGTMAP_Ibc['LOG_SFR'], hue=WGTMAP_Ibc['WGT'], legend=False, s=50, edgecolor=None,palette='cividis')
plt.xlabel(r"log$_{10}$($M/M_{\odot})$")
plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)")
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SNIbc_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')

Vincenzi_Ibc = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/V21_SNIbc.WGTMAP_clean", delim_whitespace=True)
Vincenzi_Ibc_cut = Vincenzi_Ibc[Vincenzi_Ibc['LOGMASS']<12.5]

plt.figure(figsize=(10,7))
sns.scatterplot(Vincenzi_Ibc_cut['LOGMASS'], Vincenzi_Ibc_cut['LOGSFR'], hue=Vincenzi_Ibc_cut['WGT'], legend=False, s=300, edgecolor=None, palette='cividis')
plt.xlabel(r"log$_{10}$($M/M_{\odot})$")
plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)")
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/Vincenzi_SNIbc_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')

######## SN II
plt.figure(figsize=(10,7))
sns.scatterplot(WGTMAP_II['logmass'], np.log10(WGTMAP_II['SFR']), hue=WGTMAP_II['WGT'], legend=False, s=50, edgecolor=None, palette='viridis')
plt.xlabel(r"log$_{10}$($M/M_{\odot})$")
plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)")
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SNII_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')

Vincenzi_II = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/V21_SNII.WGTMAP_clean", delim_whitespace=True)
Vincenzi_II_cut = Vincenzi_II[Vincenzi_II['LOGMASS']<12.5]

plt.figure(figsize=(10,7))
sns.scatterplot(Vincenzi_II_cut['LOGMASS'], Vincenzi_II_cut['LOGSFR'], hue=Vincenzi_II_cut['WGT'], legend=False, s=300, edgecolor=None, palette='viridis')
plt.xlabel(r"log$_{10}$($M/M_{\odot})$")
plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)")
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/Vincenzi_SNII_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')

#np.unique(Vincenzi_Ia['LOG_SFR'])

############1D distributions in weight
WGT_logmassV = []
for temp_logmass in np.unique(Vincenzi_Ia['LOGMASS']):
    tempDF = Vincenzi_Ia[Vincenzi_Ia['LOGMASS']==temp_logmass]
    newWGT = np.nansum(tempDF['WGT'].values)
    WGT_logmassV.append(newWGT)

WGT_logmass = []
for temp_logmass in logmass_:
    tempDF = WGTMAP_Ia[WGTMAP_Ia['LOGMASS']==temp_logmass]
    newWGT = np.nansum(tempDF['WGT'].values)
    WGT_logmass.append(newWGT)

plt.figure(figsize=(10,7))
plt.plot(np.unique(Vincenzi_Ia['LOGMASS']), WGT_logmassV/np.nansum(WGT_logmassV), ls='--', label='Vincenzi', lw=3,zorder=100)
plt.plot(logmass_, WGT_logmass/np.nansum(WGT_logmass), label='This work', lw=3)
plt.ylim(ymax=0.1)
plt.legend()
plt.xlabel("Logmass")
plt.ylabel('Normalized Weight')
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SNIa_LogmassWeight.png",dpi=300,bbox_inches='tight')

############1D distributions in weight
WGT_logsfrV = []
for temp_logsfr in np.unique(Vincenzi_Ia['LOG_SFR']):
    tempDF = Vincenzi_Ia[Vincenzi_Ia['LOG_SFR']==temp_logsfr]
    newWGT = np.nansum(tempDF['WGT'].values)
    WGT_logsfrV.append(newWGT)

WGT_logsfr = []
for temp_logsfr in logSFR_:
    tempDF = WGTMAP_Ia[WGTMAP_Ia['LOG_SFR']==temp_logsfr]
    newWGT = np.nansum(tempDF['WGT'].values)
    WGT_logsfr.append(newWGT)

plt.figure(figsize=(10,7))
plt.plot(np.unique(Vincenzi_Ia['LOG_SFR']), WGT_logsfrV/np.nansum(WGT_logsfrV), ls='--', label='Vincenzi', lw=3,zorder=100)
plt.plot(logSFR_, WGT_logsfr/np.nansum(WGT_logsfr), label='This work', lw=3)
plt.ylim(ymax=0.1)
plt.legend()
plt.xlabel("LogSFR")
plt.ylabel('Normalized Weight')
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SNIa_LogSFRWeight.png",dpi=300,bbox_inches='tight')

############1D distributions in weight
WGT_x1V = []
for temp_x1 in np.unique(Vincenzi_Ia['x1']):
    tempDF = Vincenzi_Ia[Vincenzi_Ia['x1']==temp_x1]
    newWGT = np.nansum(tempDF['WGT'].values)
    WGT_x1V.append(newWGT)

WGT_x1 = []
for temp_x1 in x1_:
    tempDF = WGTMAP_Ia[WGTMAP_Ia['x1']==temp_x1]
    newWGT = np.nansum(tempDF['WGT'].values)
    WGT_x1.append(newWGT)

plt.figure(figsize=(10,7))
#plt.plot(np.unique(Vincenzi_Ia['x1']), WGT_x1V/np.nansum(WGT_x1V), ls='--', label='Vincenzi', lw=3,zorder=100)
plt.plot(x1_, WGT_x1/np.nansum(WGT_x1), label='This work', lw=3)
plt.legend()
plt.xlabel("Salt2 x1")
plt.ylabel('Normalized Weight')
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SNIa_x1Weight.png",dpi=300,bbox_inches='tight')

Vincenzi_Ia['SFR'] =10**Vincenzi_Ia['LOG_SFR']

sns.scatterplot(Vincenzi_Ia.loc[Vincenzi_Ia['x1']==-1, 'LOGMASS'], np.log10(Vincenzi_Ia.loc[Vincenzi_Ia['x1']==-1, 'SFR']), hue=Vincenzi_Ia.loc[Vincenzi_Ia['x1']==-1, 'WGT']/np.nansum(Vincenzi_Ia.loc[Vincenzi_Ia['x1']==-1, 'WGT']),legend=False)
sns.scatterplot(WGTMAP_Ia.loc[WGTMAP_Ia['x1']==-1, 'logmass'], np.log10(WGTMAP_Ia.loc[WGTMAP_Ia['x1']==-1, 'SFR']), hue=WGTMAP_Ia.loc[WGTMAP_Ia['x1']==-1, 'WGT']/np.nansum(WGTMAP_Ia.loc[WGTMAP_Ia['x1']==-1, 'WGT']),legend=False)

Vx1 = Vincenzi_Ia.loc[Vincenzi_Ia['LOGMASS']==5]
Vx1 = Vx1[Vx1['LOG_SFR']==3.975]

x1 = WGTMAP_Ia.loc[WGTMAP_Ia['logmass']==5]
x1 = x1[x1['SFR']==10**3.975]

Vincenzi_Ia.loc[Vincenzi_Ia['x1']==-2]

#x1 = -1, logmass = 5, log10SFR =
RateIa(np.array([10**5]), np.array([10**3.975]), np.array([-1]))
x1[x1['x1']==-1]
Vx1[Vx1['x1']==-1]
plt.figure(figsize=(15,7))
plt.plot(Vx1['x1'], Vx1['WGT'], label='Vincenzi')
plt.plot(x1['x1'], x1['WGT'], ls='--',label='This work')
plt.plot([-1], [(A*10**5 + B*10**3.975)/np.exp(1)], '*', ms=30, c='tab:red', label=r'$(A\times10^5 + B\times10^{3.975})\times e^{-1}$')
plt.title(r"Mass=$10^{5} M_{\odot}$, SFR=$10^{3.975} M_{\odot}/yr$")
plt.xlabel(r"$x1$")
plt.ylabel("WGT")
plt.legend()
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SNIa_x1_discrepancy.png",dpi=300,bbox_inches='tight')

LOGSFR_ = np.linspace(-6, 4, 50)
LOGMASS_ = np.linspace(6, 12, 50)
SFR_ = 10**LOGSFR_
MASS_ = 10**LOGMASS_

#fig 6 of sullivan
plt.figure(figsize=(10,7))
plt.semilogy(np.log10(SFR_/MASS_), R_AB(MASS_, SFR_)/MASS_, 'o')
plt.xlabel("Log[sSFR (Msol/yr)/Msol]")
plt.ylabel("Rate (SNe/yr)/stellar mass")
plt.savefig("Fig6_Sullivan2006.png",bbox_inches='tight', dpi=200)

#fig 8 of sullivan +2006
plt.figure(figsize=(10,7))
plt.plot(np.log10(MASS_), np.log10(R_AB(MASS_, SFR_)), 'o')
plt.xlabel("log10(Stellar Mass)")
plt.ylabel("log10(Rate, SNe/yr)")
plt.xlim((7.6, 12))
plt.ylim((-5, -1))
plt.savefig("Fig8_Sullivan2006.png",bbox_inches='tight', dpi=200)
