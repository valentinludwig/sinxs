#!/usr/bin/env python
# coding: utf-8

# # Compare SMOS/SMAP and PIOMAS SIT

# ### Module import

# In[1]:


import netCDF4
import numpy as np
import xarray as xr
import pyproj
import cartopy.crs as ccrs
import pylab as plt
import pyhdf
from pyhdf.SD import SD, SDC
from pyresample import geometry,kd_tree,image
from osgeo import gdal


# In[2]:


def regrid_swath_to_swath(data,lon_in,lat_in,lon_out,lat_out,radius_of_influence = 10000,eps = 0.5,fill_value = np.nan):
    """Regrid array using nearest neighbour. Change radius_of_influence depending on dataset. Input:
        - data: data to be gridded to the other grid (input dataset)
        - lon_in/lat_in: coordinates of input dataset
        - lon_out/lat_out: coordinates of target grid
        - radius_of_influence: If no pixel is found within this distance (in meters) on the input dataset, the pixel in the output dataset will be set to nan (or another fill_value to be specified in the function below)
        - eps: measure of uncertainty, larger eps mean shorter execution time
        - fill_value: Pixel will be set to this value if no pixel is found within radius_of_influence (in meters)"""

    swath_def_in = geometry.SwathDefinition(lons=lon_in, lats=lat_in)
    swath_def_out = geometry.SwathDefinition(lons=lon_out, lats=lat_out)
    data_out = kd_tree.resample_nearest(swath_def_in, data, swath_def_out, radius_of_influence=radius_of_influence, epsilon=eps, fill_value = fill_value)
    return data_out


# ### Load and prepare data

# ##### SMOS SIT and geolocation

# In[3]:


f_pmw = xr.open_dataset("/Users/vludwig/05_SINXS/02_DATA/20230430_north_mix_sit_v300.nc") # pmw = passive microwave
sit_smos = f_pmw["smos_thickness"]
sit_smap = f_pmw["smap_thickness"]
sit_combined = f_pmw["combined_thickness"]


# In[4]:


hdf = SD("/Users/vludwig/05_SINXS/02_DATA/LongitudeLatitudeGrid-n12500-Arctic.hdf", SDC.READ)
lat_nsidc = hdf.select('Latitudes')[:,:]
#lat_nsidc = lat_tmp[:,:]
lon_nsidc = hdf.select('Longitudes')[:,:]
#lon_nsidc = lon_tmp[:,:]


# ##### PIOMAS

# In[9]:


f_piomas = netCDF4.Dataset("/Users/vludwig/Desktop/PIOMAS/hiday.H2018",mode = "r")
sit_piomas = np.reshape(f_piomas.variables["hiday"],(1,365,120,360))[0,0,:,:]
sit_piomas[sit_piomas==0] = np.nan
lon_piomas_1d = np.array(f_piomas.variables["x"])
lat_piomas_1d = np.array(f_piomas.variables["y"])
lon_piomas =np.reshape(lon_piomas_1d,(120,360))
lat_piomas =np.reshape(lat_piomas_1d,(120,360))


# ##### Regrid PIOMAS

# In[18]:


sit_piomas_nsidc = regrid_swath_to_swath(sit_piomas,lon_piomas,lat_piomas,lon_nsidc,lat_nsidc,radius_of_influence=25000)


# ### Plotting

# In[13]:


# Combined thickness
fig = plt.figure()
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 60,90], ccrs.PlateCarree())

ax.coastlines();
ax.gridlines();
ax.set_title("Combined SMOS/SMAP SIT, 20230430")
im = ax.pcolormesh(lon_nsidc,lat_nsidc,sit_combined,zorder=2,transform = ccrs.PlateCarree())
cb = plt.colorbar(im)
cb.set_label("SIT [cm]")
fig.show()


# In[14]:


# PIOMAS original grid
fig = plt.figure()
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 60,90], ccrs.PlateCarree())
ax.coastlines();
ax.gridlines();
ax.set_title("PIOMAS SIT, 20180101")
im = ax.pcolormesh(lon_piomas,lat_piomas,sit_piomas,zorder=2,transform = ccrs.PlateCarree())
cb = plt.colorbar(im)
cb.set_label("SIT [cm]")
fig.show()


# In[55]:





# In[19]:


fig = plt.figure()
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 60,90], ccrs.PlateCarree())
# Plot the data as usual
#z.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=0, cmap='terrain') 
# Add details to the plot:
ax.coastlines();
ax.gridlines();
ax.set_title("PIOMAS SIT regridded, 20180101")
ax.pcolormesh(lon_nsidc,lat_nsidc,sit_piomas_nsidc,zorder=2,transform = ccrs.PlateCarree())
cb = plt.colorbar(im)
cb.set_label("SIT [cm]")
fig.show()


# ##### Playground

# In[22]:


f_sic = xr.open_dataset("/Users/vludwig/05_SINXS/02_DATA/sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_20230508.nc")
f_lonlat = xr.open_dataset("/Users/vludwig/05_SINXS/02_DATA/coordinates_npstere_1km_arctic.nc")


# In[26]:


sic = f_sic["sic_merged"]
lon = f_lonlat["lon"]
lat = f_lonlat["lat"]


# In[ ]:




