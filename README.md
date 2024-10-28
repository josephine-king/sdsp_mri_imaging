All of the MATLAB files and data can be found in the MRI_datasets folder

mri_top.m contains the main scripts to run our code and plot results. The code is split into sections, which should be run sequentially one by one to see the different results. To change which scan is used, change the "image_idx" variable to a string containing a numerical digit from 1 to 8.

common_functions.m contains many of the common functions that are used in our different MATLAB files

mri_wiener.m contains the code for the Wiener filters

mri_kalman.m contains the code for the Kalman filter (not used in our final report)

mri_cs.m contains the code for compressed sensing (explained in the appendix of our report)
