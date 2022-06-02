# The Simulated Catalogue of Optical Transients and Correlated Hosts (SCOTCH)
Welcome! This repository contains the code used to genenerate the SCOTCH Catalogue. The catalogue and associated data files can be downloaded from Zenodo: https://zenodo.org/record/6601211#.Ypjl2pDMLhO. If you'd just like to learn how to use the SCOTCH catalogue, a great place to start is with the tutorials at `notebooks/SCOTCH_walkthroughs.ipynb`. 

Our pipeline uses the code SNANA (Kessler+2009) for simulating transients, and is described in detail in Lokken and Gagliano, 2022 (in review). The methodology for constructing our hostlibs for SNe Ia, SNe II, and SNe Ib/c (requires access to cosmoDC2 through GCRCatalogs) is outlined as follows:

1. Use a normalizing flow (pzflow) to smooth out cosmoDC2 catalog redshifts by running `scripts/apply_pzflow_dc2full.py`.
2. Create the index of cosmoDC2 galaxies, and normalize GHOST + cosmoDC2 catalogues, by running `scripts/buildANN.py`.
3. Match GHOST galaxies by normalized properties for each primary class (Ia/II/Ibc) by running `scripts/matchNN.py`.
4. Add derived features for the matched cosmoDC2 galaxies by running `scripts/addTrueFeatures.py`, `scripts/addSFR_and_Msol.py`, and `scripts/addRSQ.py` (this may take a while).
5. Add photometric features for the matched cosmoDC2 galaxies by running `scripts/addPZFeatures.py`.
6. Correct a bug in the host galaxy semi-minor axes values from cosmoDC2 (bulge, disk, and total) by running `scripts/fixSemiMinorAxis.py`.
7. Construct the HOSTLIBS for these classes with `notebooks/Hostlib_Constructor.ipynb`.
8. Separately generate the random HOSTLIB for a random subset of cosmoDC2 by running `scripts/random_hostcat.py` and repeat steps 4-7 with random dataset. 
9. If needed, add gaussian smearing to the catalog-level photometry using `notetbooks/AddHostlibScatter.ipynb` (we do this to create a HOSTLIB for kilonovae).
10. Generate the weightmaps for fine-tuned transient-host galaxy correlations with `scripts/weightmap_generator.py`.


Please contact Martine Lokken (lokken@astro.utoronto.ca) and Alex Gagliano (gaglian2@illinois.edu) with any questions or comments.
