#!/usr/bin/env python
# coding: utf-8

# # Download PIOMAS data

# #### This code downloads PIOMAS sea-ice thickness data. It is based on the PyPIOMAS Python module provided at https://github.com/Weiming-Hu/PyPIOMAS. All credit to this guy :-)

# #### Creation date: 20230501
# #### Contact: valentin.ludwig@awi.de
# #### Current status: Converted from nb to .py
# #### Last used: 20230614



from PyPIOMAS.PyPIOMAS import PyPIOMAS # install from here: https://github.com/Weiming-Hu/PyPIOMAS
import os,sys

homedir = os.getenv("HOME") # home directory
outdir = os.path.join(homedir,"05_SINXS/02_DATA/01_INPUT/01_PIOMAS") # files shall be stored here
if not os.path.exists(outdir):
    print(f"{outdir} does not exist yet, I create it!")
    os.makedirs(outdir)

print("Available variables")
print(PyPIOMAS.supported_variables)
print("hiday is sea-ice thickness")

variables = ["hiday"] # these variables will be downloaded
years = [2023] # these years will be downloaded
downloader = PyPIOMAS(outdir, variables, years) # instance to download data (not yet in netCDF!)

print(downloader) # print info about downloader

downloader.download() # download data
downloader.unzip() # unzip downloaded data
downloader.to_netcdf(os.path.join(outdir,"hiday_2023.nc")) # convert downloaded data to netCDF. Adapt filename if needed!

