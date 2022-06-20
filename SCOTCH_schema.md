# SCOTCH Schema

There are two available versions of SCOTCH, one 'zlim' where z<0.8 and one 'z3' where z<3. Each share the same file format and variables.

## File Format
The files are in HDF5 format. Tutorials are available at https://github.com/LSSTDESC/transient-host-sims/blob/main/notebooks/SCOTCH_walkthroughs.ipynb.
There is a 'HostTable' and a 'TransientTable'. One level below each table, the subgroups are all classes of supernova that we simulate: 'AGN', 'KN', 'SLSN-I', 'SNII', 'SNIIb', 'SNIa', 'SNIb', 'SNIc', 'TDE'. For each of these classes, the Host Table has a variety of datasets containing properties described under 'Host Table' below. The Transient Table, however, has a further level of subgroups: for each class, there are subgroups which are the models with which the light curves were simulated. For example, under the 'SNII' subgroup, there are models called 'SNII+HostXT_V19', 'SNII-NMF', 'SNII-Templates', 'SNIIn+HostXT_V19', 'SNIIn-MOSFIT'. These models are described in the Plasticc models paper (https://arxiv.org/abs/1903.11756, Kessler+ 2019). Each model contains light curve information contained within datasets; these are described under the 'Transient Table' header below.

### Host Table

$m_{<\mathrm{band}>}$: Apparent AB magnitudes in LSST $ugrizY$ bands

$\sigma_{m,<\mathrm{band}>}$: 10-year estimated apparent AB magnitude errors for LSST

$e$: Shear ellipticity $(1-q)/(1+q)$, where $q$ is the axis ratio

$R_d$: Disk half-light radius in physical kpc

$R_s$: Spheroid half-light radius in physical kpc

$\log{(M_*)}$: Log stellar mass $\left[M_{\odot}\right]$

$\log{(SFR)}$: Log star formation rate
$\left[M_{\odot}/\textrm{yr}\right]$ 

$n_i$: Sersic index for $i=[0,1]$; $n_0=1$ (exponential disk) and $n_1=4$ (deVaucouleurs bulge) 

$w_i$: Weight of $i=[0,1]$ Sersic components (bulge and disk) 

$a_i$: Major-axis half-light size $\left[''\right]$ for $i=[0,1]$ Sersic components 

$b_i$: Minor-axis half-light size $\left[''\right]$ for $i=[0,1]$ Sersic components 

$e_i$: Ellipticity of $i=[0,1]$ Sersic components 

$e_{\rm tot}$: Luminosity-weighted sum of bulge and disk ellipticities 

$a_{\mathrm{rot}}$: Rotation angle of major axis with respect to the +RA coordinate [$^{\circ}$] 

        
### Transient Table
TID: Transient ID

$z$: True Redshift (of host and transient) 

GID: Host-galaxy ID  

MJD: Array of Modified Julian Dates of light curve observations (only spacing is meaningful)

$m_{<\mathrm{band}>}$: Apparent brightness in LSST $ugrizY$ bands (AB magnitudes) 

Class: Transient class

Model: Simulation model

Cadence: Time-spacing of light curve samples [days] 

$\rm RA_{\rm off}$: Transient offset from host nucleus in R.A. [$''$]

$\rm \delta{\rm off}$: Transient offset from host nucleus in Dec. [$''$]

Sep: Total great-circle distance between transient and host nucleus [$''$]
