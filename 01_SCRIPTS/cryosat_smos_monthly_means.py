#!/usr/bin/env python
# coding: utf-8

# # Make monthly means of CryosatSMOS SIT

# #### This code loads weekly CS2-SMOS SIT and converts them to monthly means. Data are available here: https://data.meereisportal.de/relaunch/thickness.php, downloaded manually so far...

# #### Creation date: 20230501
# #### Contact: valentin.ludwig@awi.de
# #### Current status: COnverted from nb
# #### Last used: 20230614

# ### Module import

# In[10]:


import numpy as np
import xarray as xr
import os
import datetime


# ### Function declaration

# In[11]:


def load_sit(indir,fn):
    f_sit = xr.open_dataset(os.path.join(indir,fn))
    sit = np.array(f_sit["analysis_sea_ice_thickness"][0,:,:])
    return sit


# In[12]:


def load_lonlat(indir,fn):
    f_lonlat = xr.open_dataset(os.path.join(indir,fn))
    lon = np.array(f_lonlat["lon"][:,:])
    lat = np.array(f_lonlat["lat"][:,:])

    return lon,lat


# ### Load and prepare data

# ##### CS-SMOS SIT and geolocation

# In[13]:


sit_daily = np.empty((432,432,25)) # 25: comes because CS2-SMOS is a daily dataset which comprises the following week's worth of data. Thus, the 25th day is the last one which contains only January days


# In[14]:


basedir = os.path.join(os.getenv("HOME"),"05_SINXS") # SINXS directory
indir = os.path.join(basedir,"02_DATA/01_INPUT/02_CS2-SMOS") # downloaded CS2-SMOS data are here
outdir = os.path.join(basedir,"02_DATA/02_OUTPUT/02_CS2-SMOS/02_NPY") # monthly output will be saved here

if not os.path.exists(outdir):
    print(f"{outdir} does not exist yet, I create it!")
    os.makedirs(outdir)

# In[15]:


start = datetime.datetime.strptime("20230101", "%Y%m%d") # first day
end = datetime.datetime.strptime("20230126", "%Y%m%d") # day AFTER last days
startdates = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
enddates = [start + datetime.timedelta(days=x+6) for x in range(0, (end-start).days)]


# In[18]:


for i,(startdate,enddate) in enumerate(zip(startdates,enddates)): # loop over dates
    fn = f"W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_{startdate.strftime('%Y%m%d')}_{enddate.strftime('%Y%m%d')}_r_v205_01_l4sit.nc" # get file name
    sit_daily[:,:,i] = load_sit(indir,fn) # load SIT data
    if i==0:
        lon,lat = load_lonlat(indir,fn) # load coordinates at first iteration


# In[19]:


sit_monthly = np.nanmean(sit_daily,axis = 2) # get monthly mean
np.savez_compressed(os.path.join(outdir,"sit_cs2-smos_202301.npz"),sit_monthly = sit_monthly) # save monthly mean
np.savez_compressed(os.path.join(outdir,"lon_ease_25km.npz"),lon = lon) # save longitudes
np.savez_compressed(os.path.join(outdir,"lat_ease_25km.npz"),lat = lat)

