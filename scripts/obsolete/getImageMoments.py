import GCRCatalogs
from GCRCatalogs import GCRQuery
import pandas as pd
import numpy as np

# We load the catalog with addons
cat = GCRCatalogs.load_catalog('dc2_object_run2.2i_dr6_with_addons')

columns_to_get0 = ["objectId", "Ixx_pixel", "Iyy_pixel", "Ixy_pixel", "IxxPSF_pixel", "IyyPSF_pixel", 'IxyPSF_pixel']
#columns_to_get0 = ["objectId"]
columns_to_get2 = ["match_objectId", "cosmodc2_id_truth"]

DF0 = cat.catalogs[0].get_quantities(columns_to_get0)
DF0 = pd.DataFrame(DF0)
print(DF0.head())
DF2 = cat.catalogs[2].get_quantities(columns_to_get2)
DF2 = pd.DataFrame(DF2)
print(DF2.head())
# rename match_objectid in DF2
DF2.rename(columns={"match_objectId":"objectId"}, inplace=True)
DF_merged = pd.merge(DF0, DF2, on=["objectId"])
print(DF_merged.head())
DF_merged.rename(columns={"cosmodc2_id_truth":"cosmoDC2_ID"}, inplace=True)

#get a sense for the ranges of IDs
print(np.sort(DF_merged.loc[DF_merged['cosmoDC2_ID'] > 0, 'cosmoDC2_ID']))

print("Number of nans in Ixx_pixel: ", np.sum(np.isnan(DF_merged['Ixx_pixel'])))
DF_merged = DF_merged.loc[np.logical_not(np.isnan(DF_merged['Ixx_pixel']))] # remove the nans
DF_merged['RSQ_pixel_gal'] = (DF_merged['Ixx_pixel']+DF_merged['Iyy_pixel']) - (DF_merged['IxxPSF_pixel']+DF_merged['IyyPSF_pixel'])
print("saving file.")

print(np.sort(DF_merged.loc[DF_merged['cosmoDC2_ID'] > 0, 'cosmoDC2_ID']))

DF_merged.to_csv("/global/cscratch1/sd/mlokken/sn_hostenv/FullImageMomentsCatalog.tar.gz")
print("Done.")