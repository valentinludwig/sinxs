# SIN'XS REPOSITORY 


###  Contact: valentin.ludwig@awi.de
###  Created: 20230614
### Last updated: 20230614

###   How to use:
    - clone the repo to your local machine
    - create the conda environment from 01_SCRIPTS/sinxs.yml
    - adapt filepaths in the scripts. It should suffice to adapt the variable "basedir". All other directories will be created as subdirectories of "basedir" and the structure of this repository should be maintained automatically
### Purpose of the scripts (all in 01_SCRIPTS):
    - download_piomas.py: Use this to download PIOMAS data. 
    - piomas_monthly_means.py: Calculates the monthly mean of the daily PIOMAS data
    - cryosat_smos_monthly_means.py: Calculates monthly mean CS2-SMOS ice thickness based on the weekly data
    - compare_piomas_cs2-smos.py: Some visualisation and evaluation. Plots maps, difference maps, histograms and boxplots of the single datasets

### Purpose of the notebooks (all in 01_SCRIPTS):
    - largely similar to the scripts with same names. Notebooks were used for initial code developlemnt, but then converted to .py scripts on 20230614 for further usage. Highly likely, the *.py scripts will be more up to date.

