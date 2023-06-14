#!/usr/bin/env python
# coding: utf-8

# # Make monthly means of PIOMAS SIT

# #### This code loads daily PIOMAS SIT and converts them to monthly means

# #### Creation date: 20230501
# #### Contact: valentin.ludwig@awi.de
# #### Current status: Converted from nb
# #### Last used: 20230614

# ### Module import

# In[1]:


import numpy as np
import xarray as xr
import os,sys


# ### Load and prepare data

# In[2]:


basedir = os.path.join(os.getenv("HOME"),"05_SINXS")# SINXS directory
indir = os.path.join(basedir,"02_DATA/01_INPUT/01_PIOMAS") # daily data are located here
outdir = os.path.join(basedir,"02_DATA/02_OUTPUT/01_PIOMAS/02_NPY") # monthly output will be stored here

if not os.path.exists(outdir):
    print(f"{outdir} does not exist yet, I create it!")
    os.makedirs(outdir)

# In[4]:


f_sit = xr.open_dataset(os.path.join(indir,"hiday_2023.nc")) # open netCDF file


# In[5]:


hiday_1d = np.array(f_sit["hiday"][0,:,:]) # get hiday variable


# In[6]:

n_lon = 120
n_lat = 360
n_days = int(hiday_1d.size/(n_lon*n_lat))
hiday = np.reshape(hiday_1d,(n_days,n_lon,n_lat)) # reshape hiday (dims: time, lon, lat)


# In[30]:


sit_monthly = np.nanmean(hiday[0:31,:,:],axis = 0) # get monthly mean in January
sit_monthly[sit_monthly<1e-3] = np.nan # set 0 values to NaN
np.savez_compressed(os.path.join(outdir,"sit_piomas_202301.npz"),sit_monthly = sit_monthly) # save as npz array


# In[31]:


lon_1d = np.array(f_sit["x"]) # get longitudes
lat_1d = np.array(f_sit["y"]) # get latitudes
lon = np.reshape(lon_1d,(n_lon,n_lat)) # get 2d lon
lat = np.reshape(lat_1d,(n_lon,n_lat)) # get 2d lat
np.savez_compressed(os.path.join(outdir,"lon_curvilinear.npz"),lon = lon) # save longitudes
np.savez_compressed(os.path.join(outdir,"lat_curvilinear.npz"),lat = lat) # save latitudes


# In[ ]:




