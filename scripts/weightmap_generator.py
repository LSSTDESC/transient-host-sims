#create a weightmap associated with the x1 and SNIa rate correlations presented
#in Vincenzi+2020
#create a 3D meshgrid over logmass, SFR, and x1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


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

SNIa_hostlib = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/matchedGals_IaGhostlib.tar.gz")

SNLS = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/SNLS_IaFits.csv", delimiter='\t')
np.nanmin(SNLS['x1'])
np.nanmax(SNLS['x1'])
np.nanmin(SNIa_hostlib['PZflowSFRtot'])

plt.hist(SNIa_hostlib['PZflowSFRtot']/1.e9)

plt.hist(np.log10(SNIa_hostlib['PZflowSFRtot']/1.e9))

np.nanmin(np.log10(SNIa_hostlib['PZflowSFRtot']/1.e9))
#x1: maybe 100 values between -3 and 3
#logmass: 5.5 to 12.5
#SFR: 10^1 to 10^10

N = 101
#commented-out grids are from Maria Vincenzi's original WGTMAPS!
x1_ = np.linspace(-5, 5, int((N-1)/5)+1)
#x1_ = np.array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.]) #forcing to the Vincenzi vals

logmass_ = np.linspace(4.5, 14, N)
#logmass_ = np.array([ 5.   ,  5.225,  5.45 ,  5.675,  5.9  ,  6.125,  6.35 ,  6.575,
#        6.8  ,  7.025,  7.25 ,  7.475,  7.7  ,  7.925,  8.15 ,  8.375,
#        8.6  ,  8.825,  9.05 ,  9.275,  9.5  ,  9.725,  9.95 , 10.175,
#       10.4  , 10.625, 10.85 , 11.075, 11.3  , 11.525, 11.75 , 11.975,
#       12.2  , 12.425, 12.65 , 12.875, 13.1  , 13.325, 13.55 , 13.775])
logSFR_ = np.linspace(-8.5, 2, N) #
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

x1_mesh, logmass_mesh, SFR_mesh = np.meshgrid(x1_, logmass_, logSFR_)

x1_allMesh = np.ravel(x1_mesh)
logmass_allMesh = np.ravel(logmass_mesh)
mass_allMesh = 10**logmass_allMesh
logSFR_allMesh = np.ravel(SFR_mesh)
SFR_allMesh = 10**logSFR_allMesh

def RateII(Msol, SFR):
    rate = np.zeros(len(Msol))
    sSFR = SFR/Msol
    cond1 = np.log10(sSFR) >= -11.5
    rate[cond1] = Msol[cond1]**0.16
    return rate

def RateIbc(Msol, SFR):
    rate = np.zeros(len(Msol))
    sSFR = SFR/Msol
    cond1 = np.log10(sSFR) >= -11.5
    rate[cond1] = Msol[cond1]**0.36
    return rate

def RateIa(Msol, SFR, x1):
    return R_AB(Msol, SFR)*Rstar(x1, Msol)

def R_AB(Msol, SFR):
    #from https://ui.adsabs.harvard.edu/abs/2006ApJ...648..868S
    A = 5.3e-14 #SNe/yr/Msol
    B = 3.9e-4 #SNe/yr * (Msol/yr)^-1
    return A*Msol + B*SFR

def Rstar(x1, Msol):
    #from https://arxiv.org/pdf/2012.07180.pdf
    rate = np.ones(len(x1))
    cond1 = ((x1 < 0) & (Msol < 1.e10))
    #cond2 = ((x1 > 0) & (Msol < 1.e10)) rate is one
    #cond3 = (Msol > 1.e10) rate is one
    rate[cond1] = np.exp(-x1[cond1]**2)
    return rate

rate_allMesh_Ia = RateIa(mass_allMesh, SFR_allMesh, x1_allMesh)
rate_allMesh_II = RateII(mass_allMesh, SFR_allMesh)
rate_allMesh_Ibc = RateIbc(mass_allMesh, SFR_allMesh)

##
#VARNAMES_WGTMAP: x1 LOGMASS LOG_SFR  WGT  SNMAGSHIFT

#DOCUMENTATION:
#    PURPOSE: Weighting map for SNe Ia host galaxy association (based on the Vincenzi WGTMAPS)
#    USAGE_KEY:  WGTMAP_FILE
#    VALIDATE_SCIENCE: used in PLASTiCC v2.0
#    NOTES:
#    - The map includes also the x1-Mass "relation" (see Sec 4.5.1 eq 4 from Vincenzi)
#    - The map is defined over a logM range of 4.5 to 14, a logSFR range of -8.5 to 2 and an x1 range of -5 to 5
#    VERSIONS:
#    - DATE:  2021
#      AUTHORS: A. GAGLIANO, M. LOKKEN
#DOCUMENTATION_END:
##

snmagshift = np.zeros(len(logmass_allMesh))
#VARNAMES_WGTMAP:

weight_prefix = np.array(['WGT:']*len(logmass_allMesh))
#WGT:

WGTMAP_Ia = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh), 'x1':x1_allMesh,'WGT':rate_allMesh_Ia, 'SNMAGSHIFT':snmagshift})
WGTMAP_Ia.to_csv("/Users/alexgagliano/Documents/Research/DESC/tables/SNIa.WGTMAP",index=False, float_format='%.3f')

WGTMAP_II = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh),'WGT':rate_allMesh_II, 'SNMAGSHIFT':snmagshift})
WGTMAP_II.to_csv("/Users/alexgagliano/Documents/Research/DESC/tables/SNII.WGTMAP",index=False, float_format='%.3f')

WGTMAP_Ibc = pd.DataFrame({'VARNAMES_WGTMAP:':weight_prefix, 'LOGMASS':logmass_allMesh, 'LOG_SFR':np.log10(SFR_allMesh),'WGT':rate_allMesh_Ibc, 'SNMAGSHIFT':snmagshift})
WGTMAP_Ibc.to_csv("/Users/alexgagliano/Documents/Research/DESC/tables/SNIbc.WGTMAP",index=False, float_format='%.3f')

#marginalize over x1 to just get logSFR vs logMsol
logmass_x1_marg = []
SFR_x1_marg = []
WGT_x1_marg = []

for temp_logmass in logmass_:
    tempDF = WGTMAP_Ia[WGTMAP_Ia['logmass']==temp_logmass]
    for temp_SFR in SFR_:
        tempDF_WGT = tempDF.loc[tempDF['SFR']==temp_SFR, 'WGT'].values
        newWGT = np.nansum(tempDF_WGT)
        logmass_x1_marg.append(temp_logmass)
        SFR_x1_marg.append(temp_SFR)
        WGT_x1_marg.append(newWGT)

WGTMAPIa_x1_marg = pd.DataFrame({'logmass':np.array(logmass_x1_marg), 'SFR':np.array(SFR_x1_marg), 'WGT':np.array(WGT_x1_marg)})

plt.figure(figsize=(10,7))
sns.scatterplot(logmass_x1_marg, np.log10(SFR_x1_marg), hue=np.log10(WGT_x1_marg), legend=False, s=50, edgecolor=None)
plt.xlabel(r"log$_{10}$($M/M_{\odot})$")
plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)")
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/SNIa_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')


Vincenzi_Ia = pd.read_csv("/Users/alexgagliano/Documents/Research/DESC/tables/V21_SNIa_REVISED.WGTMAP_clean", delim_whitespace=True)

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
sns.scatterplot(Vincenzi_marg_cut['logmass'], Vincenzi_marg_cut['SFR'], hue=np.log10(Vincenzi_marg_cut['WGT']), legend=False, s=300, edgecolor=None)
plt.xlabel(r"log$_{10}$($M/M_{\odot})$")
plt.ylabel(r"log$_{10}$(SFR/$M_{\odot}$ yr${-1}$)")
plt.savefig("/Users/alexgagliano/Documents/Research/DESC/plots/Vincenzi_SNIa_WGTMAP_logSFR_v_logmass.png",dpi=300,bbox_inches='tight')


####### SN Ibc
plt.figure(figsize=(10,7))
sns.scatterplot(WGTMAP_Ibc['logmass'], np.log10(WGTMAP_Ibc['SFR']), hue=WGTMAP_Ibc['WGT'], legend=False, s=50, edgecolor=None,palette='cividis')
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
    tempDF = WGTMAP_Ia[WGTMAP_Ia['logmass']==temp_logmass]
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
    tempDF = WGTMAP_Ia[WGTMAP_Ia['SFR']==10**temp_logsfr]
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
plt.plot(np.unique(Vincenzi_Ia['x1']), WGT_x1V/np.nansum(WGT_x1V), ls='--', label='Vincenzi', lw=3,zorder=100)
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
