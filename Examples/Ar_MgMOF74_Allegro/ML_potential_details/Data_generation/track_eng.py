# 3/17/2021
# Author: Kaihang Shi
# This script reads the output file of RASPA and extracts the host-adsorbate energy

import os
import re 
import numpy as np
#import matplotlib.pyplot as plt

# number of frames
nframes = 10000

# get all folder name (i.e. MOF id) in the top level
parent_path = os.getcwd()
outdir = parent_path+'/Output/System_0'

# output file
fout = open('host-ads_energies.txt','a')


# initialize variables 
ls_host_ads_eng = np.zeros((nframes))



# check if simulation finished or not
if os.path.isdir(outdir):
    os.chdir(outdir)
    outputfile=os.listdir('.')[0]

    # read in file
    f1 = open(outputfile).readlines()

    # counter
    jstep = 0
    
    # loop over lines
    for line in f1:

        line = line.strip()

        # host-ads
        match = re.search("Current Host-Adsorbate energy",line)
        if match:
            ls_host_ads_eng[jstep]=float(line.split()[3])
            jstep += 1

else:
    print('Warning: some output files do not exist!')

# write ls_host_ads_eng data to file and for each line, add a new line
for i in range(nframes):
    fout.write(str(ls_host_ads_eng[i])+'\n')

fout.close()








